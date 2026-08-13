from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway
from pydantic import ValidationError

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

FRONTEND_MENU = [
    {
        "id": "americano",
        "name": "아메리카노",
        "price": 3000,
        "image_url": "/figma/cafe/americano.png?v=2",
    },
    {
        "id": "milk",
        "name": "우유",
        "price": 2000,
        "image_url": "/figma/cafe/milk.png?v=2",
    },
    {
        "id": "strawberry-juice",
        "name": "딸기주스",
        "price": 4000,
        "image_url": "/figma/cafe/strawberry-juice.png?v=2",
    },
    {
        "id": "cookie",
        "name": "쿠키",
        "price": 2000,
        "image_url": "/figma/cafe/cookie.png?v=2",
    },
    {
        "id": "strawberry-cake",
        "name": "딸기케이크",
        "price": 3000,
        "image_url": "/figma/cafe/strawberry-cake.png?v=2",
    },
    {
        "id": "sandwich",
        "name": "샌드위치",
        "price": 4000,
        "image_url": "/figma/cafe/sandwich.png?v=2",
    },
]


def cafe_context(
    mormi_menu_id: str,
    budget: int | None = None,
    child_menu_id: str | None = None,
    paid_amount: int | None = None,
) -> dict[str, object]:
    return {
        "menu_items": FRONTEND_MENU,
        "mormi_menu_id": mormi_menu_id,
        "budget": budget,
        "child_menu_id": child_menu_id,
        "paid_amount": paid_amount,
    }


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


def test_menu_scenarios_require_frontend_context() -> None:
    with pytest.raises(ValidationError):
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_menu_total")

    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_budget_menu",
            cafe_context=cafe_context("strawberry-juice", budget=3000),
        )


@pytest.mark.asyncio
async def test_budget_menu_uses_frontend_menu_and_allows_correction(
    tmp_path: object,
) -> None:
    service, repository, database = await make_service(tmp_path)
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_budget_menu",
            cafe_context=cafe_context("strawberry-juice", budget=6000),
        )
    )

    assert [item["id"] for item in started.turn.visual.data["menu_items"]] == [
        item["id"] for item in FRONTEND_MENU
    ]
    assert started.turn.visual.data["mormi_pick"]["id"] == "strawberry-juice"
    mormi_choice = next(
        choice for choice in started.turn.input.choices if choice.id == "strawberry-juice"
    )
    assert mormi_choice.disabled is True
    assert started.turn.input.config["allow_same_menu"] is False
    over = await choose(service, started.conversation_id, started.turn, "sandwich")
    assert over.turn.status.value == "active"
    assert over.turn.visual.data["budget_status"] == "over"
    assert over.turn.visual.data["total"] == 8000
    state = await repository.get_state(started.conversation_id)
    assert "child_menu_id" not in state.scenario_data

    corrected = await choose(service, started.conversation_id, over.turn, "milk")
    assert corrected.turn.status.value == "completed"
    assert corrected.turn.note_update is not None
    assert len(await repository.list_notes(1)) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_menu_total_uses_frontend_prices_and_creates_one_note(tmp_path: object) -> None:
    service, repository, database = await make_service(tmp_path, skills=("add_menu_prices",))
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_menu_total",
            cafe_context=cafe_context("americano"),
        )
    )

    picked = await choose(service, started.conversation_id, started.turn, "milk")
    assert picked.turn.task_id == "cafe_total_calculation"
    assert picked.turn.visual.data["left"] == 3000
    assert picked.turn.visual.data["right"] == 2000
    assert picked.turn.visual.type == "cafe_calculation"
    operation = await choose(service, started.conversation_id, picked.turn, "add")
    completed = await choose(service, started.conversation_id, operation.turn, "5000")

    assert completed.turn.status.value == "completed"
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert "더해서" in notes[0].text
    await database.dispose()


@pytest.mark.asyncio
async def test_change_subtracts_both_menus_from_the_cash_actually_paid(
    tmp_path: object,
) -> None:
    """Change must match the general backend's `paidAmount - orderTotal`.

    Mormi picks the cake (3,000) and the child picked the sandwich (4,000), so
    a 10,000 payment leaves 3,000. Subtracting only Mormi's menu would show the
    child 7,000 for a problem whose answer is 3,000.
    """
    service, repository, database = await make_service(tmp_path, skills=("calculate_change",))
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_change",
            cafe_context=cafe_context(
                "strawberry-cake",
                child_menu_id="sandwich",
                paid_amount=10000,
            ),
        )
    )

    assert started.turn.visual.data["left"] == 10000
    assert started.turn.visual.data["right"] == 7000
    assert "method" not in started.turn.input.target_slots
    operation = await choose(service, started.conversation_id, started.turn, "subtract")
    completed = await choose(service, started.conversation_id, operation.turn, "3000")

    assert completed.turn.status.value == "completed"
    assert "받아내림" not in completed.turn.note_update.text  # type: ignore[union-attr]
    assert len(await repository.list_notes(1)) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_change_honours_a_payment_other_than_10000(tmp_path: object) -> None:
    """The paid cash is whatever the backend graded, not a fixed 10,000."""
    service, _repository, database = await make_service(tmp_path, skills=("calculate_change",))
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_change",
            cafe_context=cafe_context(
                "strawberry-cake",
                child_menu_id="milk",
                paid_amount=6000,
            ),
        )
    )

    assert started.turn.visual.data["left"] == 6000
    assert started.turn.visual.data["right"] == 5000
    await database.dispose()


def test_change_requires_the_earlier_stage_results() -> None:
    """Independent conversations cannot recover the pick or the payment."""
    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_change",
            cafe_context=cafe_context("strawberry-cake", paid_amount=10000),
        )

    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_change",
            cafe_context=cafe_context("strawberry-cake", child_menu_id="sandwich"),
        )

    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_change",
            cafe_context=cafe_context(
                "strawberry-cake",
                child_menu_id="sandwich",
                paid_amount=5000,
            ),
        )


@pytest.mark.asyncio
async def test_frontend_menu_snapshot_is_persisted_without_server_replacement(
    tmp_path: object,
) -> None:
    service, repository, database = await make_service(tmp_path)
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_menu_total",
            cafe_context=cafe_context("strawberry-juice"),
        )
    )
    before = await repository.get_state(started.conversation_id)
    restored = await service.snapshot(started.conversation_id)
    after = await repository.get_state(restored.conversation_id)

    assert before.scenario_data["menu_items"] == FRONTEND_MENU
    assert before.scenario_data == after.scenario_data
    await database.dispose()


def test_integrated_stage_is_not_exposed_in_current_prototype() -> None:
    from mormi_api.content import get_scenario

    with pytest.raises(KeyError):
        get_scenario("cafe_integrated")
