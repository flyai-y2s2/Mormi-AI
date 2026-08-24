from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from mormi_api.db import Database, LadderAnalysisRecord
from mormi_api.ladder_analysis import LadderAction
from mormi_api.ladder_analysis_repository import (
    LadderAnalysisRepository,
    LadderAnalysisRequest,
)
from mormi_api.ladder_analysis_worker import LadderAnalysisWorker
from mormi_api.ladder_model.dataset import LadderLevel
from mormi_api.ladder_model.runtime import RuntimeBatchResult, RuntimePrediction
from mormi_api.schemas import utc_now


async def repository(tmp_path: Path) -> tuple[Database, LadderAnalysisRepository]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/ladder-worker.db")
    await database.create_schema()
    return database, LadderAnalysisRepository(database, lease_seconds=30)


def request(key: str = "learner-7:skill-a:s2") -> LadderAnalysisRequest:
    return LadderAnalysisRequest(
        idempotency_key=key,
        learner_id=7,
        skill_id="skill-a",
        trigger_session_id="session-2",
        session_ids=("session-1", "session-2"),
        current_level=LadderLevel.L2,
        performance_by_level={LadderLevel.L2: {"correct": 9, "attempts": 10}},
        lower_rule_evidence_count=0,
    )


@pytest.mark.asyncio
async def test_enqueue_is_idempotent_and_schema_never_stores_raw_speech(
    tmp_path: Path,
) -> None:
    database, store = await repository(tmp_path)

    first = await store.enqueue(request())
    second = await store.enqueue(request())

    assert first.analysis_id == second.analysis_id
    assert first.status == "pending"
    column_names = set(LadderAnalysisRecord.__table__.columns.keys())
    assert not {"speech", "utterance", "response_text", "response_raw"} & column_names
    assert "민감" not in {field.name for field in fields(first)}
    await database.dispose()


@pytest.mark.asyncio
async def test_pending_job_transitions_to_running_and_completed(tmp_path: Path) -> None:
    database, store = await repository(tmp_path)
    created = await store.enqueue(request())

    claimed = await store.claim_pending(1)
    assert [job.analysis_id for job in claimed] == [created.analysis_id]
    assert claimed[0].status == "running"

    completed = await store.complete(
        claimed[0],
        decision={"action": "MAINTAIN", "recommended_level": "L2"},
        model_version="test-v2",
    )
    assert completed is True
    latest = await store.latest_for_learner(7)
    assert latest[0].status == "completed"
    assert latest[0].decision == {
        "action": "MAINTAIN",
        "recommended_level": "L2",
    }
    await database.dispose()


@pytest.mark.asyncio
async def test_expired_running_job_is_reclaimed_after_restart(tmp_path: Path) -> None:
    database, store = await repository(tmp_path)
    await store.enqueue(request())
    first = (await store.claim_pending(1, now=utc_now()))[0]

    reclaimed = await store.claim_pending(
        1,
        now=first.available_at,
    )

    assert reclaimed[0].analysis_id == first.analysis_id
    assert reclaimed[0].attempt == 2
    await database.dispose()


class FakeRuntime:
    model_version = "fake-ladder-v2"

    def predict(self, texts: list[str]) -> RuntimeBatchResult:
        assert len(texts) == 2
        return RuntimeBatchResult(
            available=True,
            predictions=(
                RuntimePrediction(LadderLevel.L3, 0.8),
                RuntimePrediction(LadderLevel.L4, 0.9),
            ),
        )


@pytest.mark.asyncio
async def test_worker_processes_speech_without_persisting_it(tmp_path: Path) -> None:
    database, store = await repository(tmp_path)
    await store.enqueue(request())

    async def load_speech(_: object) -> list[str]:
        return ["긴 설명 발화", "또 다른 실제 발화"]

    worker = LadderAnalysisWorker(
        store,
        FakeRuntime(),  # type: ignore[arg-type]
        load_speech=load_speech,
        poll_interval_seconds=0.01,
        batch_size=5,
    )

    result = await worker.run_once()
    latest = (await store.latest_for_learner(7))[0]

    assert result.completed == 1
    assert latest.decision["action"] == LadderAction.UPGRADE.value
    assert latest.decision["recommended_level"] == LadderLevel.L3.value
    assert "긴 설명" not in repr(latest)
    await database.dispose()


@pytest.mark.asyncio
async def test_worker_stores_bounded_error_code_and_continues(tmp_path: Path) -> None:
    database, store = await repository(tmp_path)
    await store.enqueue(request("first"))
    await store.enqueue(request("second"))

    class MissingRuntime:
        model_version = "unavailable"

        def predict(self, texts: list[str]) -> RuntimeBatchResult:
            del texts
            return RuntimeBatchResult(available=False, error_code="MODEL_NOT_FOUND")

    async def load_speech(_: object) -> list[str]:
        return ["원문"]

    worker = LadderAnalysisWorker(
        store,
        MissingRuntime(),  # type: ignore[arg-type]
        load_speech=load_speech,
        batch_size=5,
    )
    result = await worker.run_once()

    assert result.failed == 2
    assert {row.error_code for row in await store.latest_for_learner(7)} == {
        "MODEL_NOT_FOUND"
    }
    await database.dispose()


@pytest.mark.asyncio
async def test_requeue_model_failures_preserves_ids_and_excludes_other_failures(
    tmp_path: Path,
) -> None:
    database, store = await repository(tmp_path)
    error_by_key = {
        "missing": "MODEL_NOT_FOUND",
        "dependency": "MODEL_DEPENDENCY_MISSING",
        "load": "MODEL_LOAD_FAILED",
        "speech": "SPEECH_LOAD_FAILED",
    }
    created_by_key = {
        key: await store.enqueue(request(key)) for key in error_by_key
    }
    claimed_by_id = {
        job.analysis_id: job for job in await store.claim_pending(len(error_by_key))
    }
    for key, error_code in error_by_key.items():
        await store.fail(
            claimed_by_id[created_by_key[key].analysis_id],
            error_code=error_code,
        )

    requeued = await store.requeue_model_failures()
    latest = {job.analysis_id: job for job in await store.latest_for_learner(7)}

    assert requeued == 3
    for key in ("missing", "dependency", "load"):
        original = created_by_key[key]
        assert latest[original.analysis_id].analysis_id == original.analysis_id
        assert latest[original.analysis_id].status == "pending"
        assert latest[original.analysis_id].error_code is None
    speech = latest[created_by_key["speech"].analysis_id]
    assert speech.status == "failed"
    assert speech.error_code == "SPEECH_LOAD_FAILED"
    await database.dispose()
