from __future__ import annotations

import random
from uuid import uuid4

import pytest
from conftest import FakeGateway
from pydantic import ValidationError
from sqlalchemy import select

from mormi_api.content import (
    PARK_SCENARIO_IDS,
    SCENARIOS,
    create_scenario_data,
    generate_park_context,
    get_task,
    representative_park_context,
)
from mormi_api.db import Database, OutboxEventRecord
from mormi_api.engine import ConversationEngine
from mormi_api.outbox import STAR_NOTE_CREATED_EVENT_TYPE
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ArithmeticClaim,
    ChildResponse,
    ExpressionLevel,
    LearnerProfile,
    ParkSessionContext,
    SafetyCategory,
    SceneType,
    SessionCreate,
    SkillProfile,
    UtteranceAnalysis,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService

EXPECTED_PRIMARY = {
    "amusement_ticket_multiply": {
        "operation": "multiplication",
        "left": 3000,
        "right": 2,
        "result": 6000,
        "facts": {"ticket_price": 3000, "party_count": 2, "total_price": 6000},
    },
    "amusement_snack_divide": {
        "operation": "division",
        "left": 6000,
        "right": 3,
        "result": 2000,
        "facts": {"snack_total": 6000, "payer_count": 3, "per_person": 2000},
    },
    "amusement_pass_compare": {
        "operation": "division",
        "left": 10000,
        "right": 2000,
        "result": 5,
        "facts": {
            "single_ride_price": 2000,
            "day_pass_price": 10000,
            "break_even_rides": 5,
            "benefit_from_rides": 6,
        },
    },
}


def _request(scenario_id: str) -> SessionCreate:
    return SessionCreate(
        learner_id=1,
        scene=SceneType.AMUSEMENT_PARK,
        scenario_id=scenario_id,
    )


@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
def test_park_session_contract_needs_only_scenario_identity(
    scenario_id: str,
) -> None:
    request = _request(scenario_id)
    assert request.scenario_id == scenario_id

    with pytest.raises(ValidationError, match="amusement_park scene"):
        SessionCreate(
            learner_id=1,
            scene=SceneType.CAFE,
            scenario_id=scenario_id,
        )


@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
def test_park_catalog_generates_only_solvable_reviewed_problems(
    scenario_id: str,
) -> None:
    chooser = random.Random(20260824)
    variants: set[str] = set()
    for _ in range(200):
        context = generate_park_context(scenario_id, chooser)
        values = {fact.key: fact.value for fact in context.facts}
        variants.add(context.variant_id)
        if scenario_id == "amusement_ticket_multiply":
            assert values["total_price"] == values["ticket_price"] * values["party_count"]
        elif scenario_id == "amusement_snack_divide":
            assert values["per_person"] == values["snack_total"] // values["payer_count"]
            assert values["snack_total"] % values["payer_count"] == 0
        else:
            assert values["break_even_rides"] == (
                values["day_pass_price"] // values["single_ride_price"]
            )
            assert values["benefit_from_rides"] == values["break_even_rides"] + 1
        assert context.transfer.prompt != context.prompt
    assert len(variants) > 3


def test_unreviewed_legacy_givens_fall_back_to_the_ai_catalog() -> None:
    legacy = representative_park_context("amusement_ticket_multiply")
    unsafe_facts = [
        fact.model_copy(update={"value": 1_234}) if fact.key == "ticket_price" else fact
        for fact in legacy.facts
    ]
    context = generate_park_context(
        "amusement_ticket_multiply",
        random.Random(7),
        compatibility_context=legacy.model_copy(update={"facts": unsafe_facts}),
    )
    values = {fact.key: fact.value for fact in context.facts}

    assert values["ticket_price"] in {2_000, 3_000, 4_000, 5_000}
    assert values["ticket_price"] != 1_234
    assert values["total_price"] == values["ticket_price"] * values["party_count"]


@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
def test_park_content_uses_ai_snapshot_and_never_exposes_internal_pedagogy(
    scenario_id: str,
) -> None:
    legacy_context = representative_park_context(scenario_id).model_copy(
        update={
            "title": "구 BE 임시 제목",
            "strategy": "구 BE 임시 풀이",
            "prompt": "두 숫자를 비교하고 이유를 설명해 주세요.",
        }
    )
    data = create_scenario_data(
        scenario_id,
        park_context=legacy_context,
        rng=random.Random(20260824),
    )
    context = ParkSessionContext.model_validate(data["park_context"])
    legacy_values = {fact.key: fact.value for fact in legacy_context.facts}
    generated_values = {fact.key: fact.value for fact in context.facts}
    primary_id, transfer_id = SCENARIOS[scenario_id].task_ids
    primary = get_task(primary_id, data)
    transfer = get_task(transfer_id, data)

    # A rolling deploy preserves only the old screen's reviewed givens so the
    # old BE can match completion. All pedagogical copy and derived values are
    # rebuilt from the AI catalog.
    assert context.title != legacy_context.title
    assert context.strategy != legacy_context.strategy
    assert context.prompt != legacy_context.prompt
    if scenario_id == "amusement_ticket_multiply":
        assert generated_values["ticket_price"] == legacy_values["ticket_price"]
        assert generated_values["party_count"] == legacy_values["party_count"]
        assert generated_values["total_price"] == (
            generated_values["ticket_price"] * generated_values["party_count"]
        )
    elif scenario_id == "amusement_snack_divide":
        assert generated_values["snack_total"] == legacy_values["snack_total"]
        assert generated_values["payer_count"] == legacy_values["payer_count"]
        assert generated_values["per_person"] == (
            generated_values["snack_total"] // generated_values["payer_count"]
        )
    else:
        assert generated_values["single_ride_price"] == legacy_values["single_ride_price"]
        assert generated_values["day_pass_price"] == legacy_values["day_pass_price"]
        assert generated_values["break_even_rides"] == (
            generated_values["day_pass_price"] // generated_values["single_ride_price"]
        )
    assert primary.steps[ExpressionLevel.L4][0].prompt == context.prompt
    assert transfer.steps[ExpressionLevel.L4][0].prompt == context.transfer.prompt
    assert transfer.base_visual.data["prompt"] == context.transfer.prompt
    assert primary.arithmetic_contract is not None
    assert "skill" not in primary.visible_facts
    assert "mormi_misconception" not in primary.visible_facts
    shown_fact_keys = {fact["key"] for fact in primary.base_visual.data["facts"]}
    assert shown_fact_keys.isdisjoint(
        {"total_price", "per_person", "break_even_rides", "benefit_from_rides"}
    )
    assert set(primary.steps) == {
        ExpressionLevel.L4,
        ExpressionLevel.L3,
        ExpressionLevel.L2,
        ExpressionLevel.L0,
    }
    assert set(primary.steps[ExpressionLevel.L0][0].input.config["completion_values"]) == set(
        primary.required_slots
    )
    assert context.transfer.equation in transfer.hints["H3"].visual_data["equation"]
    assert transfer.hints["H3"].body == context.transfer.conclusion


@pytest.mark.parametrize(
    ("operation", "left", "right", "result", "truth"),
    [
        ("multiplication", 3000, 2, 6000, "true"),
        ("multiplication", 2, 3000, 6000, "true"),
        ("multiplication", 3000, 2, 5000, "false"),
        ("division", 6000, 3, 2000, "true"),
        ("division", 6000, 3, 3000, "false"),
    ],
)
def test_park_arithmetic_claims_are_checked_deterministically(
    operation: str,
    left: int,
    right: int,
    result: int,
    truth: str,
) -> None:
    scenario_id = (
        "amusement_ticket_multiply" if operation == "multiplication" else "amusement_snack_divide"
    )
    data = create_scenario_data(
        scenario_id,
        rng=random.Random(0),
    )
    # Arithmetic truth checking is independent from the randomized scenario;
    # use a reviewed fixed snapshot so the claims below stay readable.
    data["park_context"] = representative_park_context(scenario_id).model_dump(mode="json")
    task = get_task(SCENARIOS[scenario_id].task_ids[0], data)
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        arithmetic_claims=[
            ArithmeticClaim(
                left=left,
                right=right,
                operation=operation,  # type: ignore[arg-type]
                result=result,
                evidence_span="아이의 계산 설명",
                related_slot_ids=["strategy"],
                interpretation_confidence=0.95,
            )
        ],
    )

    claims = ConversationEngine._speaker_arithmetic_claims(
        task,
        "아이의 계산 설명",
        analysis,
    )
    assert len(claims) == 1
    assert claims[0].truth_status == truth


async def _make_service(
    tmp_path: object,
    scenario_id: str,
    expression_level: ExpressionLevel = ExpressionLevel.L2,
) -> tuple[ConversationService, Repository, Database]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/park-{scenario_id}-{uuid4().hex}.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    data = create_scenario_data(scenario_id, rng=random.Random(0))
    skills = {get_task(task_id, data).skill_id for task_id in SCENARIOS[scenario_id].task_ids}
    await repository.save_profile(
        LearnerProfile(
            learner_id=1,
            skills={
                skill: SkillProfile(
                    skill_id=skill,
                    highest_stable_expression_level=expression_level,
                )
                for skill in skills
            },
        )
    )
    return (
        ConversationService(
            repository,
            ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
        ),
        repository,
        database,
    )


async def _choose(
    service: ConversationService,
    conversation_id: str,
    turn: object,
    choice_id: str,
):
    return await service.respond(
        conversation_id,
        ChildResponse(
            turn_id=turn.turn_id,  # type: ignore[attr-defined]
            response_id=uuid4(),
            type="choice",
            choice_ids=[choice_id],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
async def test_park_stage_completes_with_exact_facts_transfer_and_one_note_event(
    tmp_path: object,
    scenario_id: str,
) -> None:
    service, repository, database = await _make_service(tmp_path, scenario_id)
    started = await service.create_conversation(_request(scenario_id))

    assert started.turn.input.kind.value == "choices"
    current = started
    initial_state = await repository.get_state(started.conversation_id)
    context = ParkSessionContext.model_validate(initial_state.scenario_data["park_context"])
    saw_transfer = False
    for _ in range(8):
        state = await repository.get_state(started.conversation_id)
        task_id = state.current_task_id
        if task_id.endswith("_transfer"):
            saw_transfer = True
        task = get_task(task_id, state.scenario_data)
        step = task.step_by_id(state.subgoal_id)
        assert step is not None
        target_slot = step.target_slots[0]
        expected_value = task.slots[target_slot].expected
        choice_id = (
            "strategy_correct"
            if target_slot == "strategy"
            else next(
                option_id
                for option_id, effects in step.choice_effects.items()
                if effects.get(target_slot) == expected_value
            )
        )
        current = await _choose(
            service,
            started.conversation_id,
            current.turn,
            choice_id,
        )
        if current.turn.completion is not None:
            break

    assert saw_transfer is True
    assert current.turn.completion is not None
    expected_facts = {
        fact.key: fact.value
        for fact in context.facts
        if fact.key in context.required_verified_fact_keys
    }
    assert current.turn.completion.verified_facts == expected_facts
    assert current.turn.completion.stage_completion_eligible is True
    final_state = await repository.get_state(started.conversation_id)
    primary_evidence = final_state.completed_task_slots[SCENARIOS[scenario_id].task_ids[0]]
    derived_key = {
        "amusement_ticket_multiply": "total_price",
        "amusement_snack_divide": "per_person",
        "amusement_pass_compare": "break_even_rides",
    }[scenario_id]
    assert current.turn.completion.verified_facts[derived_key] == primary_evidence["answer"]
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    async with database.sessions() as db:
        outbox = list((await db.execute(select(OutboxEventRecord))).scalars())
    assert sum(event.event_type == STAR_NOTE_CREATED_EVENT_TYPE for event in outbox) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_joint_model_completes_stage_without_child_teaching_reward(
    tmp_path: object,
) -> None:
    scenario_id = "amusement_ticket_multiply"
    service, repository, database = await _make_service(
        tmp_path,
        scenario_id,
        expression_level=ExpressionLevel.L0,
    )
    current = await service.create_conversation(_request(scenario_id))

    for _ in range(2):
        assert current.turn.input.kind.value == "joint"
        current = await service.respond(
            current.conversation_id,
            ChildResponse(
                turn_id=current.turn.turn_id,
                response_id=uuid4(),
                type="action",
                values=current.turn.input.config["completion_values"],
            ),
        )

    assert current.turn.completion is not None
    assert current.turn.completion.outcome.value == "supported"
    assert current.turn.completion.teach_reward_eligible is False
    assert current.turn.completion.stage_completion_eligible is True
    state = await repository.get_state(current.conversation_id)
    assert state.joint_performance_used is True
    await database.dispose()
