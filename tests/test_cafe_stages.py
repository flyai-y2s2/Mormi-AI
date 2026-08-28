from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway
from pydantic import ValidationError
from sqlalchemy import select

from mormi_api.db import (
    Database,
    DialogueClaimRecord,
    DialogueTaskOutcomeRecord,
    DialogueTurnObservationRecord,
    NoteEvidenceLinkRecord,
)
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    ExpressionLevel,
    HintLevel,
    InputKind,
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
        "price": 4500,
        "image_url": "/figma/cafe/strawberry-cake.png?v=2",
    },
    {
        "id": "sandwich",
        "name": "샌드위치",
        "price": 5000,
        "image_url": "/figma/cafe/sandwich.png?v=2",
    },
]


def cafe_context(
    mormi_menu_id: str,
    budget: int | None = None,
    child_menu_id: str | None = None,
) -> dict[str, object]:
    return {
        "menu_items": FRONTEND_MENU,
        "mormi_menu_id": mormi_menu_id,
        "child_menu_id": child_menu_id,
        "budget": budget,
    }


async def make_service(
    tmp_path: object,
    *,
    skills: tuple[str, ...] = (),
    skill_levels: dict[str, ExpressionLevel] | None = None,
) -> tuple[ConversationService, Repository, Database]:
    database_path = str(tmp_path) + f"/cafe-{uuid4().hex}.db"
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    configured_skills = {
        skill: ExpressionLevel.L2 for skill in skills
    }
    configured_skills.update(skill_levels or {})
    if configured_skills:
        await repository.save_profile(
            LearnerProfile(
                learner_id=1,
                skills={
                    skill: SkillProfile(
                        skill_id=skill,
                        highest_stable_expression_level=level,
                    )
                    for skill, level in configured_skills.items()
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


def test_queue_scenario_requires_frontend_counts() -> None:
    with pytest.raises(ValidationError):
        SessionCreate(learner_id=1, scene="cafe", scenario_id="cafe_queue")

    # Two equal lines have no shorter side to count.
    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue",
            queue_context={"left_count": 3, "right_count": 3},
        )

    for queue_context in (
        {"left_count": 0, "right_count": 2},
        {"left_count": 2, "right_count": 0},
        {"left_count": 6, "right_count": 2},
        {"left_count": 2, "right_count": 6},
    ):
        with pytest.raises(ValidationError):
            SessionCreate(
                learner_id=1,
                scene="cafe",
                scenario_id="cafe_queue",
                queue_context=queue_context,
            )

    # The demo id keeps drawing its own line-up, so it must refuse a screen's counts.
    with pytest.raises(ValidationError):
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue_demo",
            queue_context={"left_count": 2, "right_count": 5},
        )


@pytest.mark.asyncio
async def test_queue_counts_come_from_the_frontend(tmp_path: object) -> None:
    service, repository, database = await make_service(tmp_path)
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue",
            queue_context={"left_count": 2, "right_count": 5},
        )
    )

    state = await repository.get_state(started.conversation_id)
    assert state.scenario_data["left_count"] == 2
    assert state.scenario_data["right_count"] == 5
    assert state.expression_level is ExpressionLevel.L4
    assert state.hint_level is HintLevel.H0
    assert started.turn.mormi.text == "왼쪽과 오른쪽 줄에 각각 몇 명이 있어?"
    assert started.turn.input.kind is InputKind.TEXT
    assert started.turn.input.target_slots == ["left_count", "right_count"]
    assert started.turn.help_card is None
    await database.dispose()


@pytest.mark.asyncio
async def test_queue_completion_keeps_verified_counts_inside_the_shared_contract(
    tmp_path: object,
) -> None:
    service, _, database = await make_service(
        tmp_path,
        skills=("compare_quantity_in_context",),
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_queue",
            queue_context={"left_count": 1, "right_count": 5},
        )
    )

    left = await choose(service, started.conversation_id, started.turn, "1")
    right = await choose(service, started.conversation_id, left.turn, "5")
    side = await choose(service, started.conversation_id, right.turn, "left")
    completed = await choose(service, started.conversation_id, side.turn, "fewer")

    assert completed.turn.completion is not None
    facts = completed.turn.completion.verified_facts
    assert facts["left_count"] == 1
    assert facts["right_count"] == 5
    assert facts["left_count"] != facts["right_count"]
    assert all(1 <= int(facts[key]) <= 5 for key in ("left_count", "right_count"))
    await database.dispose()


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
    assert "6,000원" in started.turn.mormi.text
    assert "딸기주스" in started.turn.mormi.text
    over = await choose(service, started.conversation_id, started.turn, "sandwich")
    assert over.turn.status.value == "active"
    assert over.turn.visual.data["budget_status"] == "over"
    assert over.turn.visual.data["total"] == 9000
    state = await repository.get_state(started.conversation_id)
    assert "child_menu_id" not in state.scenario_data

    corrected = await choose(service, started.conversation_id, over.turn, "milk")
    assert corrected.turn.status.value == "completed"
    assert corrected.turn.completion is not None
    assert corrected.turn.completion.verified_facts["child_menu_id"] == "milk"
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
            cafe_context=cafe_context("americano", child_menu_id="milk"),
        )
    )

    assert "예산" not in started.turn.mormi.text

    assert started.turn.task_id == "cafe_total_calculation"
    assert started.turn.visual.data["left"] == 3000
    assert started.turn.visual.data["right"] == 2000
    assert started.turn.visual.type == "cafe_calculation"
    operation = await choose(service, started.conversation_id, started.turn, "add")
    completed = await choose(service, started.conversation_id, operation.turn, "5000")

    assert completed.turn.status.value == "completed"
    assert completed.turn.completion is not None
    assert completed.turn.completion.verified_facts["result"] == 5000
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    assert "더해서" in notes[0].text
    async with database.sessions() as db:
        observations = list(
            (
                await db.execute(
                    select(DialogueTurnObservationRecord).order_by(
                        DialogueTurnObservationRecord.created_at.asc()
                    )
                )
            ).scalars()
        )
        claims = list((await db.execute(select(DialogueClaimRecord))).scalars())
        evidence_links = list(
            (await db.execute(select(NoteEvidenceLinkRecord))).scalars()
        )
        outcomes = list(
            (
                await db.execute(
                    select(DialogueTaskOutcomeRecord).order_by(
                        DialogueTaskOutcomeRecord.task_index.asc()
                    )
                )
            ).scalars()
        )

    # The calculation note combines the operation and result supplied on two
    # turns. Both turns must remain traceable instead of citing only the last.
    calculation_observations = {
        observation.observation_id: observation
        for observation in observations
        if observation.task_id == "cafe_total_calculation"
    }
    assert {
        link.observation_id: link.source_slot_ids_json for link in evidence_links
    } == {
        observation_id: [
            next(
                claim.slot_id
                for claim in claims
                if claim.observation_id == observation_id and claim.newly_verified
            )
        ]
        for observation_id in calculation_observations
    }
    assert [outcome.task_id for outcome in outcomes] == ["cafe_total_calculation"]
    assert outcomes[0].verified_slots_json == {"operation": "addition", "result": 5000}
    await database.dispose()


@pytest.mark.asyncio
async def test_next_task_starting_at_l0_receives_matching_h3_contract(
    tmp_path: object,
) -> None:
    """A profile-based L0 start must not enter a new task without full H3 support."""

    service, repository, database = await make_service(
        tmp_path,
        skill_levels={"add_menu_prices": ExpressionLevel.L0},
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_menu_total",
            cafe_context=cafe_context("americano", child_menu_id="milk"),
        )
    )

    state = await repository.get_state(started.conversation_id)

    assert started.turn.task_id == "cafe_total_calculation"
    assert state.expression_level is ExpressionLevel.L0
    assert state.hint_level is HintLevel.H3
    assert state.task_max_hint is HintLevel.H3
    assert started.turn.input.kind is InputKind.JOINT
    assert started.turn.help_card is not None
    assert started.turn.help_card.level is HintLevel.H3
    assert started.turn.help_card.auto_open is True
    assert started.turn.visual.type == "joint_money_calculation"
    assert started.turn.visual.data["result"] == 5000
    await database.dispose()


@pytest.mark.asyncio
async def test_change_subtracts_mormi_menu_from_fixed_10000(
    tmp_path: object,
) -> None:
    """The FE change stage pays 10,000 won for Mormi's single menu."""
    service, repository, database = await make_service(tmp_path, skills=("calculate_change",))
    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="cafe",
            scenario_id="cafe_change",
            cafe_context=cafe_context("strawberry-cake"),
        )
    )

    assert started.turn.visual.data["left"] == 10000
    assert started.turn.visual.data["right"] == 4500
    assert started.turn.visual.data["payment"] == 10000
    assert started.turn.visual.data["menu_total"] == 4500
    assert "child_menu" not in started.turn.visual.data
    assert "method" not in started.turn.input.target_slots
    operation = await choose(service, started.conversation_id, started.turn, "subtract")
    completed = await choose(service, started.conversation_id, operation.turn, "5500")

    assert completed.turn.status.value == "completed"
    assert completed.turn.completion is not None
    assert completed.turn.completion.verified_facts["result"] == 5500
    assert "받아내림" not in completed.turn.note_update.text  # type: ignore[union-attr]
    assert len(await repository.list_notes(1)) == 1
    await database.dispose()


def test_change_requires_only_current_frontend_menu_context() -> None:
    request = SessionCreate(
        learner_id=1,
        scene="cafe",
        scenario_id="cafe_change",
        cafe_context=cafe_context("strawberry-cake"),
    )
    assert request.cafe_context is not None
    assert request.cafe_context.mormi_menu_id == "strawberry-cake"


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

    with pytest.raises(KeyError):
        get_scenario("home_addition_teach")


def test_menu_total_selection_help_never_mentions_a_budget() -> None:
    from mormi_api.content import create_scenario_data, get_scenario, get_task
    from mormi_api.schemas import CafeSessionContext

    context = CafeSessionContext.model_validate(cafe_context("americano"))
    data = create_scenario_data("cafe_menu_total", context)
    task_id = get_scenario("cafe_menu_total").task_ids[0]
    task = get_task(task_id, data)

    assert task.visible_facts["budget"] is None
    assert all("예산" not in hint.body for hint in task.hints.values())


def test_calculation_guidance_uses_natural_korean_without_menu_name_particles() -> None:
    from mormi_api.content import create_scenario_data, get_scenario, get_task
    from mormi_api.schemas import CafeSessionContext

    context = CafeSessionContext.model_validate(cafe_context("americano"))
    data = {
        **create_scenario_data("cafe_menu_total", context),
        "child_menu_id": "milk",
    }
    task_id = get_scenario("cafe_menu_total").task_ids[0]
    task = get_task(task_id, data)
    prompt = task.steps[ExpressionLevel.L2][0].prompt

    assert prompt == "어떤 계산을 해야 하는지 골라서 알려줄 수 있어?"
    assert "우유을" not in prompt
