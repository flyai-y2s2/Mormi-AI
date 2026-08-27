from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from conftest import FakeGateway
from sqlalchemy import select

from mormi_api.db import (
    Database,
    DialogueTurnObservationRecord,
    NoteEvidenceLinkRecord,
)
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.dialogue_v2_speaker import SpeakerOutputV2
from mormi_api.engine import ConversationEngine
from mormi_api.main import health
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    DialogueRuntimeContractVersion,
    InputKind,
    ResponseType,
    SessionCreate,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService
from mormi_api.settings import Settings


class _NoTruthSpeakerGateway:
    """V2 speaker double that never introduces task facts or numeric answers."""

    async def understand_v2(self, _: object) -> object:  # pragma: no cover
        raise AssertionError("structured/no-response routing must not call understanding")

    async def speak_v2(self, plan: Any) -> SpeakerOutputV2:
        return SpeakerOutputV2(
            text="내가 아직 헷갈려... 남은 것을 알려줄 수 있어?",
            mood="curious",
        )

    async def bridge_speak_v2(self, _: object) -> SpeakerOutputV2:  # pragma: no cover
        raise AssertionError("no bridge response is used in this fixture")


def _home_request(
    *,
    learner_id: int = 71,
    learning_session_id: str = "v2-routing-session",
    conversation_round: int = 1,
) -> SessionCreate:
    return SessionCreate(
        learner_id=learner_id,
        scene="home_teach",
        scenario_id="home_teach",
        learning_session_id=learning_session_id,
        conversation_round=conversation_round,
        practice_result_id=f"practice-{learning_session_id}",
        practice_summary={
            "curriculum_session_id": "money-count",
            "skill_id": "money_count",
            "question_count": 5,
            "first_try_correct_count": 3,
            "wrong_attempt_count": 2,
        },
    )


def _service(
    repository: Repository,
    *,
    configured_version: DialogueRuntimeContractVersion,
    canary_percent: int,
    with_v2_engine: bool = True,
) -> ConversationService:
    legacy = ConversationEngine(FakeGateway())  # type: ignore[arg-type]
    v2 = (
        DialogueV2Engine(_NoTruthSpeakerGateway())  # type: ignore[arg-type]
        if with_v2_engine
        else None
    )
    return ConversationService(
        repository,
        legacy,
        v2_engine=v2,
        runtime_contract_version=configured_version,
        dialogue_v2_canary_percent=canary_percent,
        dialogue_v2_canary_salt="test-dialogue-v2-routing",
    )


@pytest.mark.asyncio
async def test_health_advertises_installed_dialogue_runtime_capabilities(
    tmp_path: object,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path}/health-capabilities.db"
    database = Database(database_url)
    repository = Repository(database, TextCipher("test-encryption-key"))
    conversation = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.LEGACY_V1,
        canary_percent=0,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=Settings(_env_file=None, database_url=database_url),
                gateway=SimpleNamespace(configured=True),
                service=conversation,
            )
        )
    )

    response = await health(request)  # type: ignore[arg-type]

    assert response.model_dump(mode="json")["dialogue_runtime_capabilities"] == [
        "legacy-v1",
        "verdict-v1",
    ]
    assert response.model_dump(mode="json")[
        "dialogue_snapshot_reader_capabilities"
    ] == [
        "dialogue-v2-snapshot-reader-v2",
    ]
    assert response.conversation_identity_reader_capabilities == [
        "conversation-scenario-idempotency-reader-v1"
    ]
    assert response.conversation_identity_schema_phase == "unchecked"
    assert response.environment == "development"
    assert (
        response.runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )
    assert response.dialogue_v2_canary_percent == 0
    await database.dispose()


@pytest.mark.asyncio
async def test_only_new_eligible_home_conversations_select_v2_and_pin_it(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/v2-routing.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    selecting_service = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.VERDICT_V1,
        canary_percent=100,
    )
    request = _home_request()

    started = await selecting_service.create_conversation(request)
    state = await repository.get_state(started.conversation_id)

    assert state.runtime_contract_version is DialogueRuntimeContractVersion.VERDICT_V1
    assert state.pinned_dialogue_v2 is not None
    assert state.pinned_dialogue_v2.pack_id == "home.money-count.v2"
    assert state.pinned_dialogue_v2.selector_reason == "native_pack_canary_selected"
    assert started.turn.task_anchor is not None
    assert started.turn.task_anchor.anchor_id == "v2.home.money-count.v2"

    # Changing rollout settings cannot reinterpret an existing conversation.
    legacy_configured_service = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.LEGACY_V1,
        canary_percent=0,
    )
    retried = await legacy_configured_service.create_conversation(request)
    assert retried.conversation_id == started.conversation_id
    assert retried.turn.task_anchor == started.turn.task_anchor

    advanced = await legacy_configured_service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id=uuid4(),
            type=ResponseType.NO_RESPONSE,
        ),
    )
    advanced_state = await repository.get_state(started.conversation_id)
    assert advanced.turn.state_version == 2
    assert (
        advanced_state.runtime_contract_version
        is DialogueRuntimeContractVersion.VERDICT_V1
    )
    assert advanced_state.pinned_dialogue_v2 is not None

    async with database.sessions() as session:
        observation = (
            await session.execute(
                select(DialogueTurnObservationRecord).where(
                    DialogueTurnObservationRecord.conversation_id
                    == started.conversation_id
                )
            )
        ).scalar_one()
    assert observation.versions_json["runtime_contract"] == "verdict-v1"
    assert observation.versions_json["dialogue_content_pack"] == (
        "home.money-count.v2"
    )
    assert observation.versions_json["dialogue_content_version"] == 1
    assert observation.versions_json["dialogue_content_source_hash"] == (
        advanced_state.pinned_dialogue_v2.source_hash
    )
    assert observation.runtime_json["runtime_contract_version"] == "verdict-v1"
    assert observation.runtime_json["understanding_source"] == "explicit_no_response"
    assert observation.runtime_json["evidence_guard_status"] == "not_applicable"
    assert observation.runtime_json["content_pack_id"] == "home.money-count.v2"
    assert observation.runtime_json["content_version"] == 1
    assert observation.runtime_json["content_source_hash"] == (
        advanced_state.pinned_dialogue_v2.source_hash
    )
    assert observation.runtime_json["stable_copy_status"] == "reviewed_fallback"
    assert "child_utterance" not in observation.runtime_json

    cafe = await selecting_service.create_conversation(
        SessionCreate(
            learner_id=72,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )
    cafe_state = await repository.get_state(cafe.conversation_id)
    assert (
        cafe_state.runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )
    assert cafe_state.pinned_dialogue_v2 is None

    assert selecting_service.dialogue_runtime_capabilities == (
        DialogueRuntimeContractVersion.LEGACY_V1,
        DialogueRuntimeContractVersion.VERDICT_V1,
    )
    assert selecting_service.dialogue_snapshot_reader_capabilities == (
        "dialogue-v2-snapshot-reader-v2",
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_v2_selection_and_persisted_v2_state_fail_closed_without_engine(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/v2-fail-closed.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    selecting_service = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.VERDICT_V1,
        canary_percent=100,
    )
    started = await selecting_service.create_conversation(_home_request())

    unavailable_service = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.LEGACY_V1,
        canary_percent=0,
        with_v2_engine=False,
    )
    assert unavailable_service.dialogue_snapshot_reader_capabilities == ()
    with pytest.raises(RuntimeError, match="V2 engine is unavailable"):
        await unavailable_service.snapshot(started.conversation_id)

    unavailable_selector = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.VERDICT_V1,
        canary_percent=100,
        with_v2_engine=False,
    )
    unavailable_request = _home_request(
        learner_id=73,
        learning_session_id="v2-unavailable-new",
    )
    with pytest.raises(RuntimeError, match="V2 engine is unavailable"):
        await unavailable_selector.create_conversation(unavailable_request)
    assert (
        await repository.conversation_id_for_learning_session(
            unavailable_request.learner_id,
            unavailable_request.learning_session_id or "",
            unavailable_request.scene,
            unavailable_request.scenario_id,
            unavailable_request.conversation_round,
        )
        is None
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_v2_joint_response_requires_exact_server_completion_values(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/v2-joint-exact.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = _service(
        repository,
        configured_version=DialogueRuntimeContractVersion.VERDICT_V1,
        canary_percent=100,
    )
    current = await service.create_conversation(_home_request())

    # L4 -> L3 -> L2 -> L0/H3, using only the explicit help route.
    for _ in range(3):
        current = await service.respond(
            current.conversation_id,
            ChildResponse(
                turn_id=current.turn.turn_id,
                response_id=uuid4(),
                type=ResponseType.NO_RESPONSE,
            ),
        )

    assert current.turn.input.kind is InputKind.JOINT
    expected = dict(current.turn.input.config["completion_values"])
    fact_key = next(key for key in expected if key.startswith("fact:"))
    missing = dict(expected)
    missing.pop(fact_key)
    extra = {**expected, "fact:forged": 999}
    wrong_type = {**expected, fact_key: True}

    for tampered in (missing, extra, wrong_type):
        with pytest.raises(ValueError, match="exactly match"):
            await service.respond(
                current.conversation_id,
                ChildResponse(
                    turn_id=current.turn.turn_id,
                    response_id=uuid4(),
                    type=ResponseType.ACTION,
                    values=tampered,
                ),
            )
        unchanged = await service.snapshot(current.conversation_id)
        assert unchanged.turn.turn_id == current.turn.turn_id

    completed = await service.respond(
        current.conversation_id,
        ChildResponse(
            turn_id=current.turn.turn_id,
            response_id=uuid4(),
            type=ResponseType.ACTION,
            values=expected,
        ),
    )
    assert completed.turn.status.value == "completed"
    assert completed.turn.completion is not None
    assert completed.turn.completion.outcome.value == "supported"
    assert completed.turn.completion.teach_reward_eligible is False
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.attribution.value == "coauthored"
    assert completed.turn.note_update.attribution_label == "아이와 함께 공부함"
    notes = await repository.list_notes(learner_id=71)
    assert [note.note_id for note in notes] == [completed.turn.note_update.note_id]
    async with database.sessions() as session:
        links = list((await session.execute(select(NoteEvidenceLinkRecord))).scalars())
    assert links
    assert set().union(*(set(link.source_slot_ids_json) for link in links)) == {
        "add_money_values"
    }
    await database.dispose()
