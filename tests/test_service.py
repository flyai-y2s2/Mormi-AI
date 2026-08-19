from __future__ import annotations

import pytest
from conftest import FakeGateway
from sqlalchemy import select

from mormi_api.db import Database, OutboxEventRecord, TurnRecord
from mormi_api.engine import ConversationEngine
from mormi_api.main import _turn_sse_events
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
    assert started.turn.task_anchor is not None
    assert started.turn.task_anchor.prompt == "왼쪽 줄에는 몇 명이 있어?"
    assert started.turn.task_anchor.target_slots == ["left_count"]
    assert started.turn.task_anchor.completed_items == []
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
    assert after_left.turn.task_anchor is not None
    assert after_left.turn.task_anchor.prompt == "오른쪽 줄에는 몇 명이 있어?"
    assert after_left.turn.task_anchor.target_slots == ["right_count"]
    assert [item.slot_id for item in after_left.turn.task_anchor.completed_items] == [
        "left_count"
    ]
    assert after_left.turn.task_anchor.completed_items[0].value == left_count
    assert "왼쪽 줄" in after_left.turn.task_anchor.completed_items[0].display_text

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
    assert completed.turn.task_anchor is None

    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert notes[0].note_id == completed.turn.note_update.note_id

    async with database.sessions() as db:
        star_note_outbox = (
            await db.execute(
                select(OutboxEventRecord).where(
                    OutboxEventRecord.event_type == "mormi.star_note.created"
                )
            )
        ).scalar_one()
    assert star_note_outbox.payload_json["attribution"] == "coauthored"
    assert star_note_outbox.payload_json["evidence"] == "supported_completion"
    assert star_note_outbox.payload_json["text"] == completed.turn.note_update.text

    transcript = await repository.raw_turns(conversation_id)
    assert transcript[0]["question"] == started.turn.mormi.text
    assert transcript[0]["response"] == str(left_count)
    assert transcript[0]["structured"] is not None

    async with database.sessions() as db:
        initial_record = (
            await db.execute(select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id))
        ).scalar_one()
        assert initial_record.turn_contract["mormi"]["text"] == ""
        assert initial_record.mormi_question_encrypted == f"plain:{started.turn.mormi.text}"
        assert initial_record.response_raw_encrypted == f"plain:{left_count}"

    await database.dispose()


@pytest.mark.asyncio
async def test_streaming_response_uses_the_same_persisted_turn_path(tmp_path: object) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/streaming.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=2,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )
    response = ChildResponse(
        turn_id=started.turn.turn_id,
        response_id="c65cf607-586b-492d-8d10-8a784e394973",
        type="no_response",
    )

    events = [
        event
        async for event in service.respond_stream(started.conversation_id, response)
    ]

    assert [event.name for event in events] == [
        "accepted",
        "progress",
        "progress",
        "progress",
        "progress",
        "turn",
    ]
    assert [event.stage for event in events if event.name == "progress"] == [
        "understanding",
        "planning",
        "speaking",
        "validating",
    ]
    final = events[-1].envelope
    assert final is not None
    assert final.turn.task_anchor is not None
    sse_chunks = [chunk async for chunk in _turn_sse_events(final, replayed=False)]
    sse_event_names = [chunk.splitlines()[0] for chunk in sse_chunks]
    assert sse_event_names[0] == "event: mormi.start"
    assert sse_event_names[1] == "event: turn.metadata"
    assert sse_event_names.index("event: turn.metadata") < sse_event_names.index(
        "event: mormi.delta"
    )
    assert '"task_anchor"' in sse_chunks[1]
    snapshot = await service.snapshot(started.conversation_id)
    assert snapshot.turn.turn_id == final.turn.turn_id
    assert snapshot.turn.task_anchor == final.turn.task_anchor

    replay = [
        event
        async for event in service.respond_stream(started.conversation_id, response)
    ]
    assert len(replay) == 1
    assert replay[0].name == "turn"
    assert replay[0].replayed is True
    assert replay[0].envelope is not None
    assert replay[0].envelope.turn.turn_id == final.turn.turn_id
    assert replay[0].envelope.turn.task_anchor == final.turn.task_anchor
    await database.dispose()


@pytest.mark.asyncio
async def test_snapshot_backfills_task_anchor_for_legacy_active_turn(tmp_path: object) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/legacy-anchor.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=3,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )

    async with database.sessions() as db:
        record = (
            await db.execute(
                select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id)
            )
        ).scalar_one()
        legacy_contract = dict(record.turn_contract)
        legacy_contract.pop("task_anchor", None)
        record.turn_contract = legacy_contract
        await db.commit()

    snapshot = await service.snapshot(started.conversation_id)
    assert snapshot.turn.task_anchor is not None
    assert snapshot.turn.task_anchor.prompt
    assert snapshot.turn.task_anchor.target_slots == snapshot.turn.input.target_slots
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
