from __future__ import annotations

from conftest import FakeGateway

from mormi_api.content import QUEUE_TASK_ID, QueueSessionContext, create_scenario_data, get_task
from mormi_api.engine import ConversationEngine
from mormi_api.schemas import ExpressionLevel, HintLevel, SceneType, SessionState


def test_task_anchor_is_deterministic_and_independent_of_hint_level() -> None:
    scenario_data = create_scenario_data(
        "cafe_queue",
        queue_context=QueueSessionContext(left_count=3, right_count=5),
    )
    state = SessionState(
        learner_id=1,
        scene=SceneType.CAFE,
        scenario_id="cafe_queue",
        task_ids=[QUEUE_TASK_ID],
        scenario_data=scenario_data,
        expression_level=ExpressionLevel.L4,
    )
    engine = ConversationEngine(FakeGateway())  # type: ignore[arg-type]
    initial = engine.initial_turn(state)
    assert initial.task_anchor is not None

    state.hint_level = HintLevel.H2
    legacy_shaped_turn = initial.model_copy(update={"task_anchor": None})
    restored = engine.ensure_task_anchor(state, legacy_shaped_turn)

    assert restored.task_anchor == initial.task_anchor
    assert restored.task_anchor.anchor_id == f"{QUEUE_TASK_ID}:free_counts"
    assert restored.task_anchor.prompt == "왼쪽과 오른쪽 줄에 각각 몇 명이 있어?"
    assert restored.task_anchor.target_slots == initial.input.target_slots


def test_task_anchor_completed_items_never_include_unverified_values() -> None:
    scenario_data = create_scenario_data(
        "cafe_queue",
        queue_context=QueueSessionContext(left_count=3, right_count=5),
    )
    state = SessionState(
        learner_id=1,
        scene=SceneType.CAFE,
        scenario_id="cafe_queue",
        task_ids=[QUEUE_TASK_ID],
        scenario_data=scenario_data,
        expression_level=ExpressionLevel.L3,
        verified_slots={"left_count": 3},
        subgoal_id="short_counts",
    )
    task = get_task(QUEUE_TASK_ID, scenario_data)
    step = task.step_for(ExpressionLevel.L3, state.verified_slots)
    state.subgoal_id = step.id
    engine = ConversationEngine(FakeGateway())  # type: ignore[arg-type]
    anchor = engine._task_anchor(state, task, step.input)

    assert anchor is not None
    assert anchor.target_slots == ["right_count"]
    assert [item.slot_id for item in anchor.completed_items] == ["left_count"]
    assert all(item.slot_id != "right_count" for item in anchor.completed_items)
