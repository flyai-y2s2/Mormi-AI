from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import httpx
from sqlalchemy import select

from .db import Database, OutboxEventRecord
from .schemas import utc_now

logger = logging.getLogger(__name__)

OutboxAction = Literal["sent", "retry", "failed", "auth_failed"]

DIALOGUE_OBSERVATION_EVENT_TYPE = "mormi.dialogue.observation.recorded"
STAR_NOTE_CREATED_EVENT_TYPE = "mormi.star_note.created"

_EVENT_ENVELOPES: dict[str, tuple[str, str]] = {
    DIALOGUE_OBSERVATION_EVENT_TYPE: ("dialogue_observation", "observation"),
    STAR_NOTE_CREATED_EVENT_TYPE: ("star_note_created", "star_note"),
}

_RETRIABLE_CONFLICT_CODES = {
    "unknown_conversation",
    "unknown_learner",
    "unknown_observation",
    "missing_evidence_observation",
}


@dataclass(frozen=True, slots=True)
class OutboxDelivery:
    event_id: str
    event_type: str
    schema_version: int
    payload: dict[str, object]
    attempt: int


@dataclass(frozen=True, slots=True)
class DeliveryDecision:
    action: OutboxAction
    reason: str


@dataclass(frozen=True, slots=True)
class DeliveryCycleResult:
    claimed: int = 0
    sent: int = 0
    retried: int = 0
    failed: int = 0
    authentication_failed: bool = False


class OutboxStore:
    """Atomically claim and transition durable observation events.

    ``available_at`` doubles as a processing lease deadline. A different
    process can reclaim an event after a crashed worker's lease expires.
    The attempt number is a generation token, so a stale worker cannot
    overwrite the newer worker's result.
    """

    def __init__(self, database: Database, *, lease_seconds: float) -> None:
        self.database = database
        self.lease_seconds = lease_seconds

    async def claim_due(
        self,
        limit: int,
        *,
        now: datetime | None = None,
        blocked_event_types: Collection[str] = (),
    ) -> list[OutboxDelivery]:
        claimed_at = now or utc_now()
        async with self.database.sessions() as db:
            conditions = [
                OutboxEventRecord.status.in_(("pending", "retry", "processing")),
                OutboxEventRecord.available_at <= claimed_at,
            ]
            if blocked_event_types:
                conditions.append(
                    OutboxEventRecord.event_type.not_in(tuple(blocked_event_types))
                )
            statement = (
                select(OutboxEventRecord)
                .where(*conditions)
                .order_by(
                    OutboxEventRecord.available_at.asc(),
                    OutboxEventRecord.created_at.asc(),
                    OutboxEventRecord.event_id.asc(),
                )
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            records = list((await db.execute(statement)).scalars())
            lease_until = claimed_at + timedelta(seconds=self.lease_seconds)
            deliveries: list[OutboxDelivery] = []
            for record in records:
                record.status = "processing"
                record.attempts += 1
                record.available_at = lease_until
                record.last_error = None
                deliveries.append(
                    OutboxDelivery(
                        event_id=record.event_id,
                        event_type=record.event_type,
                        schema_version=record.schema_version,
                        payload=dict(record.payload_json),
                        attempt=record.attempts,
                    )
                )
            await db.commit()
            return deliveries

    async def mark_sent(self, delivery: OutboxDelivery, *, now: datetime | None = None) -> bool:
        transition_at = now or utc_now()
        return await self._transition(
            delivery,
            status="sent",
            available_at=transition_at,
            delivered_at=transition_at,
            error=None,
        )

    async def mark_retry(
        self,
        delivery: OutboxDelivery,
        *,
        delay_seconds: float,
        error: str,
        now: datetime | None = None,
    ) -> bool:
        transition_at = now or utc_now()
        return await self._transition(
            delivery,
            status="retry",
            available_at=transition_at + timedelta(seconds=delay_seconds),
            delivered_at=None,
            error=error,
        )

    async def mark_failed(
        self,
        delivery: OutboxDelivery,
        *,
        error: str,
        now: datetime | None = None,
    ) -> bool:
        transition_at = now or utc_now()
        return await self._transition(
            delivery,
            status="failed",
            available_at=transition_at,
            delivered_at=None,
            error=error,
        )

    async def _transition(
        self,
        delivery: OutboxDelivery,
        *,
        status: str,
        available_at: datetime,
        delivered_at: datetime | None,
        error: str | None,
    ) -> bool:
        async with self.database.sessions() as db:
            record = await db.get(
                OutboxEventRecord,
                delivery.event_id,
                with_for_update=True,
            )
            if (
                record is None
                or record.status != "processing"
                or record.attempts != delivery.attempt
            ):
                return False
            record.status = status
            record.available_at = available_at
            record.delivered_at = delivered_at
            record.last_error = error[:300] if error else None
            await db.commit()
            return True


class OutboxDispatcher:
    """Deliver versioned integration events without blocking child dialogue turns."""

    def __init__(
        self,
        store: OutboxStore,
        *,
        endpoint_url: str,
        service_key: str,
        poll_interval_seconds: float = 2.0,
        batch_size: int = 20,
        request_timeout_seconds: float = 5.0,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 300.0,
        star_note_events_enabled: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.store = store
        self.endpoint_url = endpoint_url
        self.service_key = service_key
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.blocked_event_types = (
            frozenset()
            if star_note_events_enabled
            else frozenset({STAR_NOTE_CREATED_EVENT_TYPE})
        )
        self._stop_event = asyncio.Event()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=request_timeout_seconds)

    async def run_once(self) -> DeliveryCycleResult:
        claimed = 0
        sent = 0
        retried = 0
        failed = 0
        for _ in range(self.batch_size):
            deliveries = await self.store.claim_due(
                1,
                blocked_event_types=self.blocked_event_types,
            )
            if not deliveries:
                break
            delivery = deliveries[0]
            claimed += 1
            decision = await self._deliver(delivery)
            if decision.action == "sent":
                await self.store.mark_sent(delivery)
                sent += 1
                logger.info(
                    "ai_outbox_sent event_id=%s event_type=%s attempt=%s",
                    delivery.event_id,
                    delivery.event_type,
                    delivery.attempt,
                )
                continue
            if decision.action == "retry":
                delay = self._retry_delay(delivery.attempt)
                await self.store.mark_retry(
                    delivery,
                    delay_seconds=delay,
                    error=decision.reason,
                )
                retried += 1
                logger.warning(
                    "ai_outbox_retry event_id=%s event_type=%s attempt=%s "
                    "delay_seconds=%s reason=%s",
                    delivery.event_id,
                    delivery.event_type,
                    delivery.attempt,
                    delay,
                    decision.reason,
                )
                continue

            if decision.action == "auth_failed":
                delay = self._retry_delay(delivery.attempt)
                await self.store.mark_retry(
                    delivery,
                    delay_seconds=delay,
                    error=decision.reason,
                )
                retried += 1
                logger.error(
                    "ai_outbox_authentication_failed event_id=%s event_type=%s attempt=%s",
                    delivery.event_id,
                    delivery.event_type,
                    delivery.attempt,
                )
                return DeliveryCycleResult(
                    claimed=claimed,
                    sent=sent,
                    retried=retried,
                    failed=failed,
                    authentication_failed=True,
                )
            await self.store.mark_failed(delivery, error=decision.reason)
            failed += 1
            logger.error(
                "ai_outbox_failed event_id=%s event_type=%s attempt=%s reason=%s",
                delivery.event_id,
                delivery.event_type,
                delivery.attempt,
                decision.reason,
            )
        return DeliveryCycleResult(
            claimed=claimed,
            sent=sent,
            retried=retried,
            failed=failed,
        )

    async def run_forever(self) -> None:
        logger.info("ai_outbox_dispatcher_started")
        try:
            while not self._stop_event.is_set():
                try:
                    result = await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("ai_outbox_cycle_failed")
                    await self._wait_for_next_poll()
                    continue
                if result.authentication_failed:
                    logger.error("ai_outbox_dispatcher_stopped reason=authentication")
                    return
                if result.claimed >= self.batch_size:
                    continue
                await self._wait_for_next_poll()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ai_outbox_dispatcher_crashed")
        finally:
            logger.info("ai_outbox_dispatcher_stopped")

    async def stop(self) -> None:
        self._stop_event.set()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _wait_for_next_poll(self) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self.poll_interval_seconds,
            )

    async def _deliver(self, delivery: OutboxDelivery) -> DeliveryDecision:
        envelope = _EVENT_ENVELOPES.get(delivery.event_type)
        if envelope is None:
            return DeliveryDecision(
                "failed",
                f"unsupported_outbox_event_type:{delivery.event_type}",
            )
        public_event_type, payload_key = envelope
        request_body = {
            "event_id": delivery.event_id,
            "schema_version": delivery.schema_version,
            "event_type": public_event_type,
            payload_key: delivery.payload,
        }
        try:
            response = await self._client.post(
                self.endpoint_url,
                headers={"X-Mormi-Service-Key": self.service_key},
                json=request_body,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            return DeliveryDecision("retry", f"network_{type(error).__name__}")
        except httpx.HTTPError as error:
            return DeliveryDecision("retry", f"http_{type(error).__name__}")

        error_code = self._error_code(response)
        if 200 <= response.status_code < 300:
            return DeliveryDecision("sent", "accepted")
        if response.status_code in {401, 403}:
            return DeliveryDecision("auth_failed", f"http_{response.status_code}")
        if response.status_code == 409 and error_code in _RETRIABLE_CONFLICT_CODES:
            return DeliveryDecision("retry", error_code)
        if response.status_code == 429 or response.status_code >= 500:
            return DeliveryDecision("retry", f"http_{response.status_code}")
        if response.status_code == 422:
            return DeliveryDecision("failed", error_code or "http_422")
        return DeliveryDecision("failed", error_code or f"http_{response.status_code}")

    def _retry_delay(self, attempt: int) -> float:
        return float(
            min(
                self.retry_base_seconds * (2 ** max(attempt - 1, 0)),
                self.retry_max_seconds,
            )
        )

    @staticmethod
    def _error_code(response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        code = payload.get("code")
        return code if isinstance(code, str) else None
