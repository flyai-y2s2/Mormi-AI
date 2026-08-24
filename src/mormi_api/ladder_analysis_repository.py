from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from .db import Database, LadderAnalysisRecord, LearnerProfileRecord
from .ladder_model.dataset import LadderLevel, canonical_level
from .schemas import ExpressionLevel, LearnerProfile, SkillProfile, utc_now


class LadderAnalysisNotFoundError(LookupError):
    pass


class LadderAnalysisStaleError(RuntimeError):
    pass


class LadderAnalysisNotApplicableError(RuntimeError):
    pass


@dataclass(frozen=True)
class LadderAnalysisRequest:
    idempotency_key: str
    learner_id: int
    skill_id: str
    trigger_session_id: str
    session_ids: tuple[str, str]
    current_level: LadderLevel
    performance_by_level: dict[LadderLevel, dict[str, int]]
    lower_rule_evidence_count: int = 0


@dataclass(frozen=True)
class LadderAnalysisJob:
    analysis_id: str
    learner_id: int
    skill_id: str
    trigger_session_id: str
    session_ids: tuple[str, ...]
    current_level: LadderLevel
    performance_by_level: dict[LadderLevel, dict[str, int]]
    lower_rule_evidence_count: int
    status: str
    attempt: int
    available_at: datetime
    decision: dict[str, Any]
    model_version: str | None
    recommendation_version: int
    error_code: str | None
    approved_at: datetime | None
    created_at: datetime


class LadderAnalysisRepository:
    REQUEUEABLE_MODEL_ERRORS = frozenset(
        {"MODEL_NOT_FOUND", "MODEL_DEPENDENCY_MISSING", "MODEL_LOAD_FAILED"}
    )

    def __init__(self, database: Database, *, lease_seconds: float) -> None:
        self.database = database
        self.lease_seconds = lease_seconds

    @staticmethod
    def _analysis_id(idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:24]
        return f"ladder_{digest}"

    @staticmethod
    def _job(record: LadderAnalysisRecord) -> LadderAnalysisJob:
        return LadderAnalysisJob(
            analysis_id=record.analysis_id,
            learner_id=record.learner_id,
            skill_id=record.skill_id,
            trigger_session_id=record.trigger_session_id,
            session_ids=tuple(record.session_ids_json),
            current_level=canonical_level(record.current_level),
            performance_by_level={
                canonical_level(level): {
                    "correct": int(values.get("correct", 0)),
                    "attempts": int(values.get("attempts", 0)),
                }
                for level, values in record.performance_json.items()
                if isinstance(values, dict)
            },
            lower_rule_evidence_count=record.lower_rule_evidence_count,
            status=record.status,
            attempt=record.attempts,
            available_at=record.available_at,
            decision=dict(record.decision_json or {}),
            model_version=record.model_version,
            recommendation_version=record.recommendation_version,
            error_code=record.error_code,
            approved_at=record.approved_at,
            created_at=record.created_at,
        )

    async def enqueue(self, request: LadderAnalysisRequest) -> LadderAnalysisJob:
        analysis_id = self._analysis_id(request.idempotency_key)
        now = utc_now()
        async with self.database.sessions() as db:
            record = await db.get(LadderAnalysisRecord, analysis_id)
            if record is None:
                record = LadderAnalysisRecord(
                    analysis_id=analysis_id,
                    idempotency_key=request.idempotency_key,
                    learner_id=request.learner_id,
                    skill_id=request.skill_id,
                    trigger_session_id=request.trigger_session_id,
                    session_ids_json=list(request.session_ids),
                    current_level=request.current_level.value,
                    performance_json={
                        level.value: dict(values)
                        for level, values in request.performance_by_level.items()
                    },
                    lower_rule_evidence_count=request.lower_rule_evidence_count,
                    status="pending",
                    attempts=0,
                    available_at=now,
                    created_at=now,
                    updated_at=now,
                )
                db.add(record)
                await db.commit()
                await db.refresh(record)
            return self._job(record)

    async def enqueue_from_dict(self, payload: Mapping[str, object]) -> LadderAnalysisJob:
        raw_performance = payload.get("performance_by_level")
        if not isinstance(raw_performance, Mapping):
            raise ValueError("performance_by_level is required")
        performance: dict[LadderLevel, dict[str, int]] = {}
        for raw_level, raw_values in raw_performance.items():
            if not isinstance(raw_values, Mapping):
                raise ValueError("invalid performance values")
            performance[canonical_level(raw_level)] = {
                "correct": int(raw_values.get("correct", 0)),
                "attempts": int(raw_values.get("attempts", 0)),
            }
        raw_sessions = payload.get("session_ids")
        if not isinstance(raw_sessions, (list, tuple)) or len(raw_sessions) != 2:
            raise ValueError("exactly two session_ids are required")
        return await self.enqueue(
            LadderAnalysisRequest(
                idempotency_key=str(payload["idempotency_key"]),
                learner_id=int(str(payload["learner_id"])),
                skill_id=str(payload["skill_id"]),
                trigger_session_id=str(payload["trigger_session_id"]),
                session_ids=(str(raw_sessions[0]), str(raw_sessions[1])),
                current_level=canonical_level(payload["current_level"]),
                performance_by_level=performance,
                lower_rule_evidence_count=int(
                    str(payload.get("lower_rule_evidence_count", 0))
                ),
            )
        )

    async def claim_pending(
        self, limit: int, *, now: datetime | None = None
    ) -> list[LadderAnalysisJob]:
        claimed_at = now or utc_now()
        async with self.database.sessions() as db:
            statement = (
                select(LadderAnalysisRecord)
                .where(
                    LadderAnalysisRecord.status.in_(("pending", "running")),
                    LadderAnalysisRecord.available_at <= claimed_at,
                )
                .order_by(LadderAnalysisRecord.available_at, LadderAnalysisRecord.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            records = list((await db.execute(statement)).scalars())
            lease_until = claimed_at + timedelta(seconds=self.lease_seconds)
            for record in records:
                record.status = "running"
                record.attempts += 1
                record.available_at = lease_until
                record.error_code = None
                record.updated_at = claimed_at
            await db.commit()
            return [self._job(record) for record in records]

    async def complete(
        self,
        job: LadderAnalysisJob,
        *,
        decision: dict[str, Any],
        model_version: str,
        now: datetime | None = None,
    ) -> bool:
        completed_at = now or utc_now()
        async with self.database.sessions() as db:
            record = await db.get(LadderAnalysisRecord, job.analysis_id, with_for_update=True)
            if record is None or record.status != "running" or record.attempts != job.attempt:
                return False
            record.status = "completed"
            record.decision_json = dict(decision)
            record.model_version = model_version
            record.completed_at = completed_at
            record.available_at = completed_at
            record.updated_at = completed_at
            await db.commit()
            return True

    async def fail(
        self, job: LadderAnalysisJob, *, error_code: str, now: datetime | None = None
    ) -> bool:
        failed_at = now or utc_now()
        allowed = {
            "MODEL_NOT_FOUND",
            "MODEL_DEPENDENCY_MISSING",
            "MODEL_LOAD_FAILED",
            "MODEL_INFERENCE_FAILED",
            "SPEECH_LOAD_FAILED",
            "UNEXPECTED_ANALYSIS_ERROR",
        }
        bounded = error_code if error_code in allowed else "UNEXPECTED_ANALYSIS_ERROR"
        async with self.database.sessions() as db:
            record = await db.get(LadderAnalysisRecord, job.analysis_id, with_for_update=True)
            if record is None or record.status != "running" or record.attempts != job.attempt:
                return False
            record.status = "failed"
            record.error_code = bounded
            record.available_at = failed_at
            record.updated_at = failed_at
            await db.commit()
            return True

    async def latest_for_learner(self, learner_id: int) -> list[LadderAnalysisJob]:
        async with self.database.sessions() as db:
            records = list(
                (
                    await db.execute(
                        select(LadderAnalysisRecord)
                        .where(LadderAnalysisRecord.learner_id == learner_id)
                        .order_by(LadderAnalysisRecord.created_at.desc())
                    )
                ).scalars()
            )
            return [self._job(record) for record in records]

    async def requeue_model_failures(self) -> int:
        requeued_at = utc_now()
        async with self.database.sessions() as db:
            records = list(
                (
                    await db.execute(
                        select(LadderAnalysisRecord)
                        .where(
                            LadderAnalysisRecord.status == "failed",
                            LadderAnalysisRecord.error_code.in_(
                                self.REQUEUEABLE_MODEL_ERRORS
                            ),
                        )
                        .with_for_update(skip_locked=True)
                    )
                ).scalars()
            )
            for record in records:
                record.status = "pending"
                record.error_code = None
                record.available_at = requeued_at
                record.updated_at = requeued_at
            await db.commit()
            return len(records)

    async def approve(
        self,
        analysis_id: str,
        *,
        learner_id: int,
        recommendation_version: int,
    ) -> LadderAnalysisJob:
        approved_at = utc_now()
        async with self.database.sessions() as db:
            record = await db.get(LadderAnalysisRecord, analysis_id, with_for_update=True)
            if record is None or record.learner_id != learner_id:
                raise LadderAnalysisNotFoundError(analysis_id)
            if record.status != "completed" or not record.decision_json:
                raise LadderAnalysisNotApplicableError("analysis is not completed")
            if record.recommendation_version != recommendation_version:
                raise LadderAnalysisStaleError("recommendation version is stale")
            if record.approved_at is not None:
                return self._job(record)
            action = str(record.decision_json.get("action", ""))
            if action not in {"UPGRADE", "ADJUST_DOWN"}:
                raise LadderAnalysisNotApplicableError("recommendation does not change level")
            current_level = ExpressionLevel(
                canonical_level(record.decision_json.get("current_level")).value
            )
            recommended_level = ExpressionLevel(
                canonical_level(record.decision_json.get("recommended_level")).value
            )
            ladder_ranks = {"L0": 0, "L2": 1, "L3": 2, "L4": 3}
            if abs(
                ladder_ranks[recommended_level.canonical().value]
                - ladder_ranks[current_level.canonical().value]
            ) != 1:
                raise LadderAnalysisNotApplicableError("recommendation must move one step")

            profile_record = await db.get(LearnerProfileRecord, learner_id, with_for_update=True)
            profile = (
                LearnerProfile.model_validate(profile_record.profile_json)
                if profile_record is not None
                else LearnerProfile(learner_id=learner_id)
            )
            skill = profile.skills.get(record.skill_id) or SkillProfile(skill_id=record.skill_id)
            if skill.highest_stable_expression_level.canonical() is not current_level.canonical():
                raise LadderAnalysisStaleError("learner stable level has changed")
            skill.highest_stable_expression_level = recommended_level.canonical()
            profile.skills[record.skill_id] = skill
            profile.updated_at = approved_at
            if profile_record is None:
                db.add(
                    LearnerProfileRecord(
                        learner_id=learner_id,
                        profile_json=profile.model_dump(mode="json"),
                        updated_at=approved_at,
                    )
                )
            else:
                profile_record.profile_json = profile.model_dump(mode="json")
                profile_record.updated_at = approved_at
            record.approved_at = approved_at
            record.updated_at = approved_at
            await db.commit()
            await db.refresh(record)
            return self._job(record)
