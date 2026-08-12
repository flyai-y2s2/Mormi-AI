from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import pytest
from conftest import FakeGateway

from mormi_api.db import Database
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


async def make_service(
    tmp_path: object,
    *,
    skills: tuple[str, ...] = (),
) -> tuple[ConversationService, Repository, Database]:
    database_path = str(tmp_path) + f"/cafe-{uuid4().hex}.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    if skills:
        await repository.save_profile(
            LearnerProfile(
                learner_id=1,
                skills={
                    skill: SkillProfile(
                        skill_id=skill,
                        highest_stable_expression_level=ExpressionLevel.L2,
                    )
                    for skill in skills
                },
            )
        )
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    return service, repository, database


async def choose(service: ConversationService, conversation_id: str, turn: object, value: str):
    return await service.respond(
        conversation_id,
        ChildResponse(
            turn_id=turn.turn_id,  # type: ignore[attr-defined]
            response_id=uuid4(),
            type="choice",
            choice_ids=[value],
        ),
    )


@pytest.mark.asyncio
async def test_budget_menu_automatically_shows_overage_and_allows_correction(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mormi_api.service.create_scenario_data",
        lambda _scenario_id: {"payment": 10000, "budget": 8000, "mormi_menu_id": "yogurt"},
    )
    service, repository, database = await make_service(tmp_path)
    started = await service.create_conversation(
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_budget_menu")
    )

    over = await choose(service, started.conversation_id, started.turn, "yogurt")
    assert over.turn.status.value == "active"
    assert over.turn.visual.data["budget_status"] == "over"
    assert over.turn.visual.data["total"] == 10400
    state = await repository.get_state(started.conversation_id)
    assert "child_menu_id" not in state.scenario_data

    corrected = await choose(service, started.conversation_id, over.turn, "lemon")
    assert corrected.turn.status.value == "completed"
    assert corrected.turn.note_update is not None
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert "예산" in notes[0].text
    await database.dispose()


@pytest.mark.asyncio
async def test_menu_total_uses_both_selected_menu_prices_and_creates_one_note(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mormi_api.service.create_scenario_data",
        lambda _scenario_id: {"payment": 10000, "mormi_menu_id": "choco"},
    )
    service, repository, database = await make_service(tmp_path, skills=("add_menu_prices",))
    started = await service.create_conversation(
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_menu_total")
    )

    picked = await choose(service, started.conversation_id, started.turn, "lemon")
    assert picked.turn.task_id == "cafe_total_calculation"
    assert picked.turn.visual.data["left"] == 3200
    assert picked.turn.visual.data["right"] == 2800
    assert picked.turn.visual.type == "cafe_calculation"
    operation = await choose(service, started.conversation_id, picked.turn, "add")
    completed = await choose(service, started.conversation_id, operation.turn, "6000")

    assert completed.turn.status.value == "completed"
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert "더해서" in notes[0].text
    await database.dispose()


@pytest.mark.asyncio
async def test_change_is_always_from_10000_without_regrouping_requirement(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mormi_api.service.create_scenario_data",
        lambda _scenario_id: {"payment": 10000, "mormi_menu_id": "yogurt"},
    )
    service, repository, database = await make_service(tmp_path, skills=("calculate_change",))
    started = await service.create_conversation(
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_change")
    )

    assert started.turn.visual.data["left"] == 10000
    assert started.turn.visual.data["right"] == 5200
    assert "method" not in started.turn.input.target_slots
    operation = await choose(service, started.conversation_id, started.turn, "subtract")
    assert "method" not in operation.turn.input.target_slots
    completed = await choose(service, started.conversation_id, operation.turn, "4800")

    assert completed.turn.status.value == "completed"
    assert "받아내림" not in completed.turn.note_update.text  # type: ignore[union-attr]
    assert len(await repository.list_notes(1)) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_integrated_stage_reselects_after_manual_over_budget_total(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context: Mapping[str, object] = {
        "payment": 10000,
        "budget": 8000,
        "mormi_menu_id": "yogurt",
        "left_count": 5,
        "right_count": 2,
    }
    monkeypatch.setattr(
        "mormi_api.service.create_scenario_data", lambda _scenario_id: dict(context)
    )
    service, repository, database = await make_service(
        tmp_path,
        skills=("compare_quantity_in_context", "add_menu_prices", "calculate_change"),
    )
    started = await service.create_conversation(
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_integrated")
    )

    turn = started.turn
    for answer in ("5", "2", "right", "fewer"):
        turn = (await choose(service, started.conversation_id, turn, answer)).turn
    assert turn.task_id == "cafe_integrated_menu_pick"

    turn = (await choose(service, started.conversation_id, turn, "yogurt")).turn
    assert turn.task_id == "cafe_integrated_total"
    turn = (await choose(service, started.conversation_id, turn, "add")).turn
    turn = (await choose(service, started.conversation_id, turn, "10400")).turn
    assert turn.task_id == "cafe_integrated_menu_pick"
    assert turn.visual.data["budget_status"] == "over"

    turn = (await choose(service, started.conversation_id, turn, "lemon")).turn
    assert turn.task_id == "cafe_integrated_total"
    turn = (await choose(service, started.conversation_id, turn, "add")).turn
    turn = (await choose(service, started.conversation_id, turn, "8000")).turn
    assert turn.task_id == "cafe_integrated_change"
    assert turn.visual.data["left"] == 10000
    assert turn.visual.data["right"] == 8000
    turn = (await choose(service, started.conversation_id, turn, "subtract")).turn
    completed = await choose(service, started.conversation_id, turn, "2000")

    assert completed.turn.status.value == "completed"
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert "사람이 적은 줄" in notes[0].text
    await database.dispose()


@pytest.mark.asyncio
async def test_random_session_values_are_valid_and_persisted(tmp_path: object) -> None:
    service, repository, database = await make_service(tmp_path)
    started = await service.create_conversation(
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_integrated")
    )
    before = await repository.get_state(started.conversation_id)
    restored = await service.snapshot(started.conversation_id)
    after = await repository.get_state(restored.conversation_id)

    assert 1 <= int(before.scenario_data["left_count"]) <= 5
    assert 1 <= int(before.scenario_data["right_count"]) <= 5
    assert before.scenario_data["left_count"] != before.scenario_data["right_count"]
    assert before.scenario_data["budget"] in {8000, 9000, 10000}
    assert before.scenario_data == after.scenario_data
    await database.dispose()
