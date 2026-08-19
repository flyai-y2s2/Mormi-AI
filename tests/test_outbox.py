from __future__ import annotations

import json
from datetime import UTC, timedelta

import httpx
import pytest

from mormi_api.db import Database, OutboxEventRecord
from mormi_api.outbox import (
    DIALOGUE_OBSERVATION_EVENT_TYPE,
    STAR_NOTE_CREATED_EVENT_TYPE,
    OutboxDispatcher,
    OutboxStore,
)
from mormi_api.schemas import utc_now
from mormi_api.settings import Settings


async def _database_with_events(tmp_path: object, count: int = 1) -> Database:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/outbox.db")
    await database.create_schema()
    now = utc_now()
    async with database.sessions() as db:
        for index in range(count):
            db.add(
                OutboxEventRecord(
                    event_id=f"event_{index}",
                    aggregate_type="dialogue_observation",
                    aggregate_id=f"observation_{index}",
                    event_type=DIALOGUE_OBSERVATION_EVENT_TYPE,
                    schema_version=1,
                    payload_json={
                        "observation_id": f"observation_{index}",
                        "conversation_id": "conversation_1",
                        "observed_at": now.isoformat(),
                    },
                    available_at=now,
                    created_at=now,
                )
            )
        await db.commit()
    return database


async def _record(database: Database, event_id: str = "event_0") -> OutboxEventRecord:
    async with database.sessions() as db:
        record = await db.get(OutboxEventRecord, event_id)
        assert record is not None
        db.expunge(record)
        return record


async def _add_star_note_event(database: Database, event_id: str = "event_star") -> None:
    now = utc_now()
    async with database.sessions() as db:
        db.add(
            OutboxEventRecord(
                event_id=event_id,
                aggregate_type="star_note",
                aggregate_id="note_1",
                event_type=STAR_NOTE_CREATED_EVENT_TYPE,
                schema_version=1,
                payload_json={
                    "note_id": "note_1",
                    "note_version": 1,
                    "learner_id": 1,
                    "conversation_id": "conversation_1",
                    "text": "점을 하나씩 세면 모두 3개야.",
                },
                available_at=now,
                created_at=now,
            )
        )
        await db.commit()


def _dispatcher(
    database: Database,
    handler: httpx.MockTransport,
    *,
    batch_size: int = 20,
    star_note_events_enabled: bool = False,
) -> OutboxDispatcher:
    return OutboxDispatcher(
        OutboxStore(database, lease_seconds=30),
        endpoint_url="https://backend.example/internal/v1/observations/events",
        service_key="test-service-key",
        batch_size=batch_size,
        retry_base_seconds=2,
        retry_max_seconds=30,
        star_note_events_enabled=star_note_events_enabled,
        client=httpx.AsyncClient(transport=handler),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("response_body", [{"processed": True}, {"duplicate": True}])
async def test_success_and_duplicate_are_marked_sent(
    tmp_path: object,
    response_body: dict[str, bool],
) -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("X-Mormi-Service-Key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_body)

    database = await _database_with_events(tmp_path)
    dispatcher = _dispatcher(database, httpx.MockTransport(handle))

    result = await dispatcher.run_once()
    record = await _record(database)

    assert result.sent == 1
    assert record.status == "sent"
    assert record.delivered_at is not None
    assert captured["header"] == "test-service-key"
    assert captured["body"] == {
        "event_id": "event_0",
        "schema_version": 1,
        "event_type": "dialogue_observation",
        "observation": record.payload_json,
    }
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_star_note_event_uses_separate_envelope_when_enabled(tmp_path: object) -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"processed": True})

    database = await _database_with_events(tmp_path, count=0)
    await _add_star_note_event(database)
    dispatcher = _dispatcher(
        database,
        httpx.MockTransport(handle),
        star_note_events_enabled=True,
    )

    result = await dispatcher.run_once()
    record = await _record(database, "event_star")

    assert result.sent == 1
    assert captured["body"] == {
        "event_id": "event_star",
        "schema_version": 1,
        "event_type": "star_note_created",
        "star_note": record.payload_json,
    }
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_star_note_event_remains_pending_until_feature_flag_is_enabled(
    tmp_path: object,
) -> None:
    calls = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"processed": True})

    database = await _database_with_events(tmp_path, count=0)
    await _add_star_note_event(database)
    disabled = _dispatcher(database, httpx.MockTransport(handle))

    disabled_result = await disabled.run_once()
    pending = await _record(database, "event_star")

    assert disabled_result.claimed == 0
    assert pending.status == "pending"
    assert pending.attempts == 0
    assert calls == 0
    await disabled.close()

    enabled = _dispatcher(
        database,
        httpx.MockTransport(handle),
        star_note_events_enabled=True,
    )
    enabled_result = await enabled.run_once()
    sent = await _record(database, "event_star")

    assert enabled_result.sent == 1
    assert sent.status == "sent"
    assert calls == 1
    await enabled.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_unknown_outbox_event_type_fails_without_network_call(tmp_path: object) -> None:
    calls = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    database = await _database_with_events(tmp_path)
    async with database.sessions() as db:
        record = await db.get(OutboxEventRecord, "event_0")
        assert record is not None
        record.event_type = "mormi.unknown.event"
        await db.commit()
    dispatcher = _dispatcher(database, httpx.MockTransport(handle))

    result = await dispatcher.run_once()
    record = await _record(database)

    assert result.failed == 1
    assert calls == 0
    assert record.last_error == "unsupported_outbox_event_type:mormi.unknown.event"
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    [
        "unknown_conversation",
        "unknown_learner",
        "unknown_observation",
        "missing_evidence_observation",
    ],
)
async def test_dependency_conflicts_retry_with_exponential_backoff(
    tmp_path: object,
    error_code: str,
) -> None:
    database = await _database_with_events(tmp_path)
    dispatcher = _dispatcher(
        database,
        httpx.MockTransport(
            lambda _: httpx.Response(409, json={"code": error_code})
        ),
    )
    before = utc_now()

    result = await dispatcher.run_once()
    record = await _record(database)

    assert result.retried == 1
    assert record.status == "retry"
    assert record.attempts == 1
    available_at = record.available_at
    if available_at.tzinfo is None:
        available_at = available_at.replace(tzinfo=UTC)
    assert available_at >= before + timedelta(seconds=2)
    assert record.last_error == error_code
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_unprocessable_event_is_failed_without_retry(tmp_path: object) -> None:
    database = await _database_with_events(tmp_path)
    dispatcher = _dispatcher(
        database,
        httpx.MockTransport(
            lambda _: httpx.Response(422, json={"code": "unsupported_schema_version"})
        ),
    )

    result = await dispatcher.run_once()
    record = await _record(database)

    assert result.failed == 1
    assert result.retried == 0
    assert record.status == "failed"
    assert record.last_error == "unsupported_schema_version"
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_authentication_failure_stops_before_claiming_another_event(tmp_path: object) -> None:
    calls = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"code": "unauthorized"})

    database = await _database_with_events(tmp_path, count=2)
    dispatcher = _dispatcher(database, httpx.MockTransport(handle))

    result = await dispatcher.run_once()
    first = await _record(database, "event_0")
    second = await _record(database, "event_1")

    assert calls == 1
    assert result.authentication_failed is True
    assert result.retried == 1
    assert first.status == "retry"
    assert first.last_error == "http_401"
    assert second.status == "pending"
    assert second.attempts == 0
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 500, 503])
async def test_transient_http_failures_are_retried(
    tmp_path: object,
    status_code: int,
) -> None:
    database = await _database_with_events(tmp_path)
    dispatcher = _dispatcher(
        database,
        httpx.MockTransport(lambda _: httpx.Response(status_code)),
    )

    result = await dispatcher.run_once()
    record = await _record(database)

    assert result.retried == 1
    assert record.status == "retry"
    assert record.last_error == f"http_{status_code}"
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_network_failure_is_retried_without_storing_sensitive_data(tmp_path: object) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    database = await _database_with_events(tmp_path)
    dispatcher = _dispatcher(database, httpx.MockTransport(fail))

    result = await dispatcher.run_once()
    record = await _record(database)

    assert result.retried == 1
    assert record.status == "retry"
    assert record.last_error == "network_ConnectError"
    assert "test-service-key" not in (record.last_error or "")
    await dispatcher.close()
    await database.dispose()


@pytest.mark.asyncio
async def test_expired_processing_lease_is_reclaimed_and_stale_worker_is_ignored(
    tmp_path: object,
) -> None:
    database = await _database_with_events(tmp_path)
    store = OutboxStore(database, lease_seconds=10)
    first_at = utc_now()
    first = (await store.claim_due(1, now=first_at))[0]
    second = (await store.claim_due(1, now=first_at + timedelta(seconds=11)))[0]

    assert first.attempt == 1
    assert second.attempt == 2
    assert await store.mark_sent(first, now=first_at + timedelta(seconds=12)) is False
    assert await store.mark_sent(second, now=first_at + timedelta(seconds=12)) is True
    assert (await _record(database)).status == "sent"
    await database.dispose()


def test_observation_ingest_settings_read_the_dedicated_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MORMI_OBSERVATION_INGEST_URL",
        "https://backend.example/internal/v1/observations/events",
    )
    monkeypatch.setenv("MORMI_OBSERVATION_INGEST_KEY", "configured-outside-repository")
    monkeypatch.setenv("MORMI_STAR_NOTE_EVENTS_ENABLED", "true")

    settings = Settings(_env_file=None)

    assert settings.observation_ingest_enabled is True
    assert settings.observation_ingest_key == "configured-outside-repository"
    assert settings.star_note_events_enabled is True


def test_outbox_lease_must_exceed_request_timeout() -> None:
    settings = Settings(
        outbox_request_timeout_seconds=10,
        outbox_lease_seconds=5,
    )

    with pytest.raises(RuntimeError, match="LEASE_SECONDS"):
        settings.validate_runtime_safety()
