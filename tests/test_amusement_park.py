from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway
from pydantic import ValidationError
from sqlalchemy import select

from mormi_api.content import (
    PARK_SCENARIO_IDS,
    SCENARIOS,
    create_scenario_data,
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
        park_context=representative_park_context(scenario_id),
    )


@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
def test_park_session_contract_accepts_only_matching_backend_context(
    scenario_id: str,
) -> None:
    request = _request(scenario_id)
    assert request.park_context is not None

    with pytest.raises(ValidationError, match="park_context is required"):
        SessionCreate(
            learner_id=1,
            scene=SceneType.AMUSEMENT_PARK,
            scenario_id=scenario_id,
        )

    with pytest.raises(ValidationError, match="amusement_park scene"):
        SessionCreate(
            learner_id=1,
            scene=SceneType.CAFE,
            scenario_id=scenario_id,
            park_context=request.park_context,
        )

    wrong = request.park_context.model_copy(update={"stage_id": "not_this_stage"})
    with pytest.raises(ValidationError, match="stage_id"):
        SessionCreate(
            learner_id=1,
            scene=SceneType.AMUSEMENT_PARK,
            scenario_id=scenario_id,
            park_context=wrong,
        )


@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
def test_park_math_contract_rejects_inconsistent_backend_facts(
    scenario_id: str,
) -> None:
    context = representative_park_context(scenario_id)
    result_key = {
        "amusement_ticket_multiply": "total_price",
        "amusement_snack_divide": "per_person",
        "amusement_pass_compare": "break_even_rides",
    }[scenario_id]
    facts = [fact.model_copy(deep=True) for fact in context.facts]
    for fact in facts:
        if fact.key == result_key:
            fact.value += 1
    broken = context.model_copy(update={"facts": facts})

    with pytest.raises(ValidationError, match="inconsistent"):
        SessionCreate(
            learner_id=1,
            scene=SceneType.AMUSEMENT_PARK,
            scenario_id=scenario_id,
            park_context=broken,
        )


@pytest.mark.parametrize("scenario_id", sorted(PARK_SCENARIO_IDS))
def test_park_content_uses_backend_copy_and_numbers_without_replacement(
    scenario_id: str,
) -> None:
    context = representative_park_context(scenario_id)
    data = create_scenario_data(scenario_id, park_context=context)
    primary_id, transfer_id = SCENARIOS[scenario_id].task_ids
    primary = get_task(primary_id, data)
    transfer = get_task(transfer_id, data)
    expected = EXPECTED_PRIMARY[scenario_id]

    assert data["park_context"] == context.model_dump(mode="json")
    assert primary.steps[ExpressionLevel.L4][0].prompt == context.prompt
    assert transfer.steps[ExpressionLevel.L4][0].prompt == context.transfer.prompt
    assert transfer.base_visual.data["prompt"] == context.transfer.prompt
    assert primary.arithmetic_contract is not None
    assert primary.arithmetic_contract.operation == expected["operation"]
    assert primary.arithmetic_contract.left == expected["left"]
    assert primary.arithmetic_contract.right == expected["right"]
    assert primary.arithmetic_contract.result == expected["result"]
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
        park_context=representative_park_context(scenario_id),
    )
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
) -> tuple[ConversationService, Repository, Database]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/park-{scenario_id}-{uuid4().hex}.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    context = representative_park_context(scenario_id)
    data = create_scenario_data(scenario_id, park_context=context)
    skills = {get_task(task_id, data).skill_id for task_id in SCENARIOS[scenario_id].task_ids}
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
    saw_transfer = False
    for _ in range(8):
        state = await repository.get_state(started.conversation_id)
        task_id = state.current_task_id
        if task_id.endswith("_transfer"):
            saw_transfer = True
        step_id = state.subgoal_id
        choice_id = {
            "choose_answer": f"value_{EXPECTED_PRIMARY[scenario_id]['result']}",
            "choose_benefit": "value_6",
            "choose_strategy": "strategy_correct",
            "choose_transfer_answer": {
                "amusement_ticket_multiply": "value_14000",
                "amusement_snack_divide": "value_2000",
                "amusement_pass_compare": "value_4",
            }[scenario_id],
            "choose_transfer_benefit": "value_5",
        }[step_id]
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
    assert current.turn.completion.verified_facts == EXPECTED_PRIMARY[scenario_id]["facts"]
    notes = await repository.list_notes(1)
    assert len(notes) == 1
    async with database.sessions() as db:
        outbox = list((await db.execute(select(OutboxEventRecord))).scalars())
    assert sum(event.event_type == STAR_NOTE_CREATED_EVENT_TYPE for event in outbox) == 1
    await database.dispose()
