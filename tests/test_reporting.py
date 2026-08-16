from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mormi_api.db import ConversationRecord, Database, TurnRecord
from mormi_api.main import app
from mormi_api.repository import Repository
from mormi_api.schemas import (
    CompletionOutcome,
    ExpressionLevel,
    HintLevel,
    RetentionPolicy,
    SceneType,
    SessionState,
    SessionStatus,
    utc_now,
)
from mormi_api.security import TextCipher
from mormi_api.settings import Settings


async def reporting_repository(tmp_path: object) -> Repository:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/reporting.db")
    await database.create_schema()
    return Repository(database, TextCipher("test-encryption-key"))


async def seed_completed_conversation(
    repository: Repository,
    *,
    learner_id: int,
    response_text: str,
    raw_storage_enabled: bool = True,
    retention_policy: RetentionPolicy = RetentionPolicy.PERMANENT,
    raw_retention_until: datetime | None = None,
) -> str:
    conversation_id = f"conversation_{learner_id}"
    now = utc_now()
    state = SessionState(
        conversation_id=conversation_id,
        learner_id=learner_id,
        learning_session_id=f"session_{learner_id}",
        scene=SceneType.CAFE,
        scenario_id="cafe_queue_demo",
        task_ids=["compare_quantity_in_context"],
        expression_level=ExpressionLevel.L2,
        status=SessionStatus.COMPLETED,
        completion_outcome=CompletionOutcome.TAUGHT,
        teach_reward_eligible=True,
        verified_slots={"fewer": "left"},
        task_max_hint=HintLevel.H1,
        raw_storage_enabled=raw_storage_enabled,
        retention_policy=retention_policy,
        raw_retention_until=raw_retention_until,
        created_at=now,
        updated_at=now,
    )
    async with repository.database.sessions() as db:
        db.add(
            ConversationRecord(
                conversation_id=conversation_id,
                learner_id=learner_id,
                learning_session_id=state.learning_session_id,
                scene=state.scene.value,
                scenario_id=state.scenario_id,
                state_json=state.model_dump(mode="json"),
                state_version=state.state_version,
                status=state.status.value,
                raw_retention_until=state.raw_retention_until,
                created_at=now,
                updated_at=now,
            )
        )
        db.add(
            TurnRecord(
                turn_id=f"turn_{learner_id}",
                conversation_id=conversation_id,
                task_id="compare_quantity_in_context",
                state_version=state.state_version,
                turn_contract={
                    "pedagogy": {
                        "expression_level": "L2",
                        "hint_level": "H1",
                        "subgoal_id": "compare",
                        "verified_slots": {"fewer": "left"},
                    }
                },
                response_id=f"response_{learner_id}",
                response_type="text",
                response_raw_encrypted=repository.cipher.encrypt(response_text),
                response_category="correct_full",
                expression_level="L2",
                hint_level="H1",
                created_at=now,
            )
        )
        await db.commit()
    return conversation_id


@pytest.mark.asyncio
async def test_report_evidence_returns_only_the_requested_learner(tmp_path: object) -> None:
    repository = await reporting_repository(tmp_path)
    first = await seed_completed_conversation(
        repository, learner_id=11, response_text="두 개가 더 적어요"
    )
    await seed_completed_conversation(repository, learner_id=12, response_text="다섯 개예요")

    evidence = await repository.report_evidence(11, include_raw=True)

    assert [row.conversation_id for row in evidence.conversations] == [first]
    assert evidence.conversations[0].turns[0].response == "두 개가 더 적어요"
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_omits_raw_text_when_not_allowed(tmp_path: object) -> None:
    repository = await reporting_repository(tmp_path)
    await seed_completed_conversation(repository, learner_id=11, response_text="두 개가 더 적어요")

    evidence = await repository.report_evidence(11, include_raw=False)

    assert evidence.conversations[0].turns[0].response is None
    assert evidence.conversations[0].turns[0].response_category == "correct_full"
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_omits_expired_raw_text_without_mutating_the_record(
    tmp_path: object,
) -> None:
    repository = await reporting_repository(tmp_path)
    await seed_completed_conversation(
        repository,
        learner_id=11,
        response_text="두 개가 더 적어요",
        retention_policy=RetentionPolicy.DAYS_30,
        raw_retention_until=utc_now() - timedelta(seconds=1),
    )

    evidence = await repository.report_evidence(11, include_raw=True)

    assert evidence.conversations[0].turns[0].response is None
    async with repository.database.sessions() as db:
        stored = await db.get(TurnRecord, 1)
        assert stored is not None
        assert stored.response_raw_encrypted is not None
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_omits_raw_text_when_stored_state_has_no_consent(
    tmp_path: object,
) -> None:
    repository = await reporting_repository(tmp_path)
    await seed_completed_conversation(
        repository,
        learner_id=11,
        response_text="두 개가 더 적어요",
        raw_storage_enabled=False,
        retention_policy=RetentionPolicy.NO_RAW,
    )

    evidence = await repository.report_evidence(11, include_raw=True)

    assert evidence.conversations[0].turns[0].response is None
    assert evidence.conversations[0].turns[0].response_category == "correct_full"
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_route_requires_the_shared_key_and_isolates_the_learner(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = await reporting_repository(tmp_path)
    first = await seed_completed_conversation(
        repository, learner_id=11, response_text="두 개가 더 적어요"
    )
    await seed_completed_conversation(repository, learner_id=12, response_text="다섯 개예요")
    monkeypatch.setattr(
        app.state,
        "settings",
        Settings(service_api_key="shared-secret"),
        raising=False,
    )
    monkeypatch.setattr(app.state, "repository", repository, raising=False)

    client = TestClient(app)
    rejected = client.get(
        "/v1/internal/learners/11/report-evidence?include_raw=true",
        headers={"X-Mormi-Service-Key": "different-secret"},
    )
    accepted = client.get(
        "/v1/internal/learners/11/report-evidence?include_raw=true",
        headers={"X-Mormi-Service-Key": "shared-secret"},
    )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["learner_id"] == 11
    assert [row["conversation_id"] for row in accepted.json()["conversations"]] == [first]
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_route_fails_closed_when_the_service_key_is_not_configured(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = await reporting_repository(tmp_path)
    monkeypatch.setattr(app.state, "settings", Settings(service_api_key=None), raising=False)
    monkeypatch.setattr(app.state, "repository", repository, raising=False)

    response = TestClient(app).get(
        "/v1/internal/learners/11/report-evidence?include_raw=false",
        headers={"X-Mormi-Service-Key": "shared-secret"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "service_key_not_configured", "issues": []}}
    await repository.database.dispose()
