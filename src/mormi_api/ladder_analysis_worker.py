from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass

from sqlalchemy import select

from .db import ConversationRecord, Database, TurnRecord
from .ladder_analysis import LadderEvidence, LevelPerformance, decide_ladder_adjustment
from .ladder_analysis_repository import LadderAnalysisJob, LadderAnalysisRepository
from .ladder_model.runtime import LadderModelRuntime
from .security import StoredTextCodec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LadderWorkerCycleResult:
    claimed: int = 0
    completed: int = 0
    failed: int = 0


class DatabaseSpeechLoader:
    def __init__(self, database: Database, codec: StoredTextCodec) -> None:
        self.database = database
        self.codec = codec

    async def __call__(self, job: LadderAnalysisJob) -> list[str]:
        async with self.database.sessions() as db:
            statement = (
                select(TurnRecord.response_raw_encrypted)
                .join(
                    ConversationRecord,
                    ConversationRecord.conversation_id == TurnRecord.conversation_id,
                )
                .where(
                    ConversationRecord.learner_id == job.learner_id,
                    ConversationRecord.learning_session_id.in_(job.session_ids),
                    TurnRecord.expression_level.in_(("L3", "L4")),
                    TurnRecord.response_raw_encrypted.is_not(None),
                )
                .order_by(TurnRecord.created_at.asc(), TurnRecord.id.asc())
            )
            payloads = list((await db.execute(statement)).scalars())
        return [self.codec.load(payload) for payload in payloads if payload]


class LadderAnalysisWorker:
    def __init__(
        self,
        store: LadderAnalysisRepository,
        runtime: LadderModelRuntime,
        *,
        load_speech: Callable[[LadderAnalysisJob], Awaitable[list[str]]],
        poll_interval_seconds: float = 2.0,
        batch_size: int = 10,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.load_speech = load_speech
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self._stop_event = asyncio.Event()

    async def run_once(self) -> LadderWorkerCycleResult:
        jobs = await self.store.claim_pending(self.batch_size)
        completed = 0
        failed = 0
        for job in jobs:
            try:
                texts = await self.load_speech(job)
            except Exception:
                logger.exception(
                    "ladder_analysis_speech_load_failed analysis_id=%s", job.analysis_id
                )
                await self.store.fail(job, error_code="SPEECH_LOAD_FAILED")
                failed += 1
                continue
            result = self.runtime.predict(texts)
            if not result.available:
                await self.store.fail(job, error_code=result.error_code or "MODEL_INFERENCE_FAILED")
                failed += 1
                continue
            evidence = LadderEvidence(
                current_level=job.current_level,
                performance_by_level={
                    level: LevelPerformance(
                        correct=values["correct"], attempts=values["attempts"]
                    )
                    for level, values in job.performance_by_level.items()
                },
                recent_predictions=tuple(item.level for item in result.predictions[-2:]),
                valid_speech_count=len(result.predictions),
                lower_rule_evidence_count=job.lower_rule_evidence_count,
                completed_session_count=len(job.session_ids),
            )
            decision = decide_ladder_adjustment(evidence)
            payload = asdict(decision)
            payload["action"] = decision.action.value
            payload["current_level"] = decision.current_level.value
            payload["recommended_level"] = decision.recommended_level.value
            payload["recent_predictions"] = [
                {"level": item.level.value, "confidence": item.confidence}
                for item in result.predictions[-2:]
            ]
            if await self.store.complete(
                job, decision=payload, model_version=self.runtime.model_version
            ):
                completed += 1
        return LadderWorkerCycleResult(
            claimed=len(jobs), completed=completed, failed=failed
        )

    async def run_forever(self) -> None:
        logger.info("ladder_analysis_worker_started")
        try:
            while not self._stop_event.is_set():
                try:
                    result = await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("ladder_analysis_cycle_failed")
                    result = LadderWorkerCycleResult()
                if result.claimed >= self.batch_size:
                    continue
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.poll_interval_seconds
                    )
        finally:
            logger.info("ladder_analysis_worker_stopped")

    async def stop(self) -> None:
        self._stop_event.set()
