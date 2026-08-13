from __future__ import annotations

import pytest
from conftest import FakeGateway
from sqlalchemy import select

from mormi_api.db import Database, TurnRecord
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    ExpressionLevel,
    LearnerProfile,
    SessionCreate,
    SkillProfile,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


@pytest.mark.asyncio
async def test_choice_flow_completes_and_replay_returns_original_result(tmp_path: object) -> None:
    database_path = str(tmp_path) + "/mormi-test.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    await repository.save_profile(
        LearnerProfile(
            learner_id=1,
            skills={
                "compare_quantity_in_context": SkillProfile(
                    skill_id="compare_quantity_in_context",
                    highest_stable_expression_level=ExpressionLevel.L2,
                )
            },
        )
    )
    engine = ConversationEngine(FakeGateway())  # type: ignore[arg-type]
    service = ConversationService(repository, engine)

    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue_demo",
            practice_summary={
                "skill_id": "compare_quantity_in_context",
                "question_count": 5,
                "first_try_correct_count": 0,
            },
        )
    )
    conversation_id = started.conversation_id
    state = await repository.get_state(conversation_id)
    left_count = int(state.scenario_data["left_count"])
    right_count = int(state.scenario_data["right_count"])
    shorter = "left" if left_count < right_count else "right"

    first_response = ChildResponse(
        turn_id=started.turn.turn_id,
        response_id="9956cd80-b1e4-45f9-81aa-638218ebdc86",
        type="choice",
        choice_ids=[str(left_count)],
    )
    after_left = await service.respond(conversation_id, first_response)
    first_result_turn_id = after_left.turn.turn_id

    after_right = await service.respond(
        conversation_id,
        ChildResponse(
            turn_id=after_left.turn.turn_id,
            response_id="51e3317b-f04c-48f2-94c5-7ff0b4077728",
            type="choice",
            choice_ids=[str(right_count)],
        ),
    )

    replay = await service.respond(conversation_id, first_response)
    assert replay.turn.turn_id == first_result_turn_id
    assert replay.turn.turn_id != after_right.turn.turn_id

    after_side = await service.respond(
        conversation_id,
        ChildResponse(
            turn_id=after_right.turn.turn_id,
            response_id="7307c9af-2440-4d56-aabc-41ec9600db77",
            type="choice",
            choice_ids=[shorter],
        ),
    )
    completed = await service.respond(
        conversation_id,
        ChildResponse(
            turn_id=after_side.turn.turn_id,
            response_id="cbd2ad2d-4cea-4f02-b574-f1610533c21e",
            type="choice",
            choice_ids=["fewer"],
        ),
    )

    assert completed.turn.status.value == "completed"
    assert completed.turn.completion is not None
    assert completed.turn.completion.outcome.value == "supported"
    assert completed.turn.completion.teach_reward_eligible is True
    assert completed.turn.completion.verified_facts["left_count"] == left_count
    assert completed.turn.completion.verified_facts["right_count"] == right_count
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.attribution.value == "coauthored"

    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert notes[0].note_id == completed.turn.note_update.note_id

    transcript = await repository.raw_turns(conversation_id)
    assert transcript[0]["question"] == started.turn.mormi.text
    assert transcript[0]["response"] == str(left_count)
    assert transcript[0]["structured"] is not None

    async with database.sessions() as db:
        initial_record = (
            await db.execute(select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id))
        ).scalar_one()
        assert initial_record.turn_contract["mormi"]["text"] == ""
        assert initial_record.mormi_question_encrypted != started.turn.mormi.text

    await database.dispose()


@pytest.mark.asyncio
async def test_existing_no_raw_conversation_is_upgraded_once_to_permanent(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/storage-upgrade.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )

    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue_demo",
            conversation_storage_consent=False,
            retention_policy="no_raw",
        )
    )
    before = await repository.get_state(started.conversation_id)
    assert before.raw_storage_enabled is False

    await repository.migrate_existing_storage_to_permanent()
    await repository.migrate_existing_storage_to_permanent()

    after = await repository.get_state(started.conversation_id)
    assert after.raw_storage_enabled is True
    assert after.retention_policy.value == "permanent"
    assert after.raw_retention_until is None
    await database.dispose()


@pytest.mark.asyncio
async def test_inline_practice_snapshot_uses_top_level_ownership(tmp_path: object) -> None:
    database_path = str(tmp_path) + "/mormi-practice-test.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )

    await service.create_conversation(
        SessionCreate(
            learner_id=7,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_frontend_123",
            practice_result_id="practice_frontend_123",
            practice_summary={
                "curriculum_session_id": "add-pictures",
                "skill_id": "basic_addition",
                "question_count": 5,
                "first_try_correct_count": 4,
                "wrong_attempt_count": 1,
            },
        )
    )

    stored = await repository.get_practice_summary("practice_frontend_123")
    assert stored is not None
    assert stored.learner_id == 7
    assert stored.success_rate == 0.8
    await database.dispose()
