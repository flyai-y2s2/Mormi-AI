from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mormi_api.db import Database
from mormi_api.ladder_analysis_repository import LadderAnalysisRepository
from mormi_api.main import app
from mormi_api.repository import Repository
from mormi_api.schemas import ExpressionLevel, LearnerProfile, SkillProfile
from mormi_api.security import StoredTextCodec
from mormi_api.settings import Settings


async def setup_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Database, LadderAnalysisRepository, Repository]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/ladder-api.db")
    await database.create_schema()
    store = LadderAnalysisRepository(database, lease_seconds=30)
    repository = Repository(database, StoredTextCodec("test-key"))
    monkeypatch.setattr(
        app.state, "settings", Settings(service_api_key="shared-secret"), raising=False
    )
    monkeypatch.setattr(app.state, "ladder_analysis_store", store, raising=False)
    monkeypatch.setattr(app.state, "repository", repository, raising=False)
    return database, store, repository


def create_payload() -> dict[str, object]:
    return {
        "idempotency_key": "learner-7:skill-a:session-2",
        "learner_id": 7,
        "skill_id": "skill-a",
        "trigger_session_id": "session-2",
        "session_ids": ["session-1", "session-2"],
        "current_level": "L2",
        "performance_by_level": {"L2": {"correct": 9, "attempts": 10}},
        "lower_rule_evidence_count": 0,
    }


@pytest.mark.asyncio
async def test_registration_requires_key_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _, _ = await setup_state(tmp_path, monkeypatch)
    client = TestClient(app)

    rejected = client.post("/v1/internal/ladder-analyses", json=create_payload())
    first = client.post(
        "/v1/internal/ladder-analyses",
        headers={"X-Mormi-Service-Key": "shared-secret"},
        json=create_payload(),
    )
    second = client.post(
        "/v1/internal/ladder-analyses",
        headers={"X-Mormi-Service-Key": "shared-secret"},
        json=create_payload(),
    )

    assert rejected.status_code == 401
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["analysis_id"] == second.json()["analysis_id"]
    await database.dispose()


@pytest.mark.asyncio
async def test_approval_applies_only_matching_learner_and_latest_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, store, repository = await setup_state(tmp_path, monkeypatch)
    await repository.save_profile(
        LearnerProfile(
            learner_id=7,
            skills={
                "skill-a": SkillProfile(
                    skill_id="skill-a",
                    highest_stable_expression_level=ExpressionLevel.L2,
                )
            },
        )
    )
    created = await store.enqueue_from_dict(create_payload())
    claimed = (await store.claim_pending(1))[0]
    await store.complete(
        claimed,
        decision={
            "action": "UPGRADE",
            "current_level": "L2",
            "recommended_level": "L3",
        },
        model_version="test-v2",
    )
    evidence = await repository.report_evidence(7, include_raw=False)
    assert len(evidence.ladder_recommendations) == 1
    assert evidence.ladder_recommendations[0].skill_id == "skill-a"
    assert evidence.ladder_recommendations[0].action == "UPGRADE"
    client = TestClient(app)
    headers = {"X-Mormi-Service-Key": "shared-secret"}

    wrong_learner = client.post(
        f"/v1/internal/ladder-analyses/{created.analysis_id}/approve",
        headers=headers,
        json={"learner_id": 8, "recommendation_version": 1},
    )
    stale = client.post(
        f"/v1/internal/ladder-analyses/{created.analysis_id}/approve",
        headers=headers,
        json={"learner_id": 7, "recommendation_version": 99},
    )
    accepted = client.post(
        f"/v1/internal/ladder-analyses/{created.analysis_id}/approve",
        headers=headers,
        json={"learner_id": 7, "recommendation_version": 1},
    )

    assert wrong_learner.status_code == 404
    assert stale.status_code == 409
    assert accepted.status_code == 200
    assert (
        await repository.get_profile(7)
    ).skills["skill-a"].highest_stable_expression_level is ExpressionLevel.L3
    await database.dispose()


@pytest.mark.asyncio
async def test_approval_is_idempotent_when_profile_already_has_recommended_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, store, repository = await setup_state(tmp_path, monkeypatch)
    await repository.save_profile(
        LearnerProfile(
            learner_id=7,
            skills={
                "skill-a": SkillProfile(
                    skill_id="skill-a",
                    highest_stable_expression_level=ExpressionLevel.L2,
                )
            },
        )
    )
    payload = create_payload()
    payload["current_level"] = "L3"
    created = await store.enqueue_from_dict(payload)
    claimed = (await store.claim_pending(1))[0]
    await store.complete(
        claimed,
        decision={
            "action": "ADJUST_DOWN",
            "current_level": "L3",
            "recommended_level": "L2",
        },
        model_version="test-v2",
    )
    client = TestClient(app)

    response = client.post(
        f"/v1/internal/ladder-analyses/{created.analysis_id}/approve",
        headers={"X-Mormi-Service-Key": "shared-secret"},
        json={"learner_id": 7, "recommendation_version": 1},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    evidence = await repository.report_evidence(7, include_raw=False)
    assert evidence.ladder_recommendations[0].approved is True
    assert (
        await repository.get_profile(7)
    ).skills["skill-a"].highest_stable_expression_level is ExpressionLevel.L2
    await database.dispose()
