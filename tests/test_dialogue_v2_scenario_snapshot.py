from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from mormi_api.dialogue_v2_life_content import LifeScenarioPackV2
from mormi_api.dialogue_v2_scenario_snapshot import (
    DialogueV3ScenarioSnapshotError,
    pin_life_scenario_runtime_v3,
    resolve_life_scenario_runtime_v3,
)
from mormi_api.schemas import (
    ExpressionLevel,
    PinnedDialogueScenarioRuntimeV3,
    PinnedDialogueTaskNoteStateV3,
    SessionState,
)


def _task_payload(
    *,
    task_id: str,
    pack_id: str,
    phase: str,
    note_enabled: bool,
) -> dict[str, object]:
    answer_target = {
        "target_kind": "fact",
        "target_id": "shorter_line",
        "ask_kind": "answer",
    }
    relation_target = {
        "target_kind": "relation",
        "target_id": "compare_lines",
        "ask_kind": "reason_or_method",
    }
    return {
        "pack_id": pack_id,
        "content_version": 1,
        "scene": "cafe",
        "scenario_id": "cafe_queue",
        "task_id": task_id,
        "stage_id": "queue",
        "phase": phase,
        "skill_id": "compare_quantity",
        "title": "짧은 줄 찾기",
        "dictionary_card_id": "dictionary.cafe.queue",
        "source_prompt": "어느 줄이 더 짧고 어떻게 알았는지 알려줄래?",
        "base_visual": {
            "type": "cafe_queue",
            "data": {"left_count": 2, "right_count": 4},
        },
        "reasoning_graph": {
            "facts": [
                {
                    "fact_id": "left_count",
                    "role": "visible_value",
                    "value": {"type": "number", "value": 2, "unit": "명"},
                    "speaker_label": "왼쪽 줄 사람 수",
                    "initially_visible": True,
                },
                {
                    "fact_id": "right_count",
                    "role": "visible_value",
                    "value": {"type": "number", "value": 4, "unit": "명"},
                    "speaker_label": "오른쪽 줄 사람 수",
                    "initially_visible": True,
                },
                {
                    "fact_id": "shorter_line",
                    "role": "final_answer",
                    "value": {"type": "choice", "choice_id": "left"},
                    "speaker_label": "더 짧은 줄",
                    "initially_visible": False,
                    "required_for_completion": True,
                    "accepted_surface_forms": ["왼쪽"],
                },
            ],
            "relations": [
                {
                    "relation_id": "compare_lines",
                    "role": "explanation",
                    "operation": "comparison",
                    "comparison_goal": "minimum",
                    "input_fact_ids": ["left_count", "right_count"],
                    "output_fact_id": "shorter_line",
                    "evaluation_mode": "open_semantic_support",
                    "speaker_label": "사람 수를 비교해 더 짧은 줄 찾기",
                    "rubric": {
                        "sufficient": ["두 줄의 사람 수를 비교한다"],
                        "partial": ["사람 수를 본다"],
                        "incorrect": ["사람이 많은 줄이 더 짧다"],
                    },
                    "required_for_completion": True,
                }
            ],
            "completion": {
                "required_fact_ids": ["shorter_line"],
                "required_relation_ids": ["compare_lines"],
            },
        },
        "initial_question": {
            "plan_id": "ask_all",
            "targets": [answer_target, relation_target],
            "reviewed_fallback": "어느 줄이 더 짧고 어떻게 알았는지 알려줄래?",
        },
        "l3_plans": [
            {
                "plan_id": "ask_line",
                "targets": [answer_target],
                "reviewed_fallback": "어느 줄이 더 짧아?",
            },
            {
                "plan_id": "ask_reason",
                "targets": [relation_target],
                "reviewed_fallback": "어떻게 알았는지 알려줄래?",
            },
        ],
        "copy_slots": [
            {
                "copy_slot": "initial_help",
                "purpose": "initial_help",
                "targets": [answer_target],
                "generation_brief": "더 짧은 줄부터 부탁한다.",
                "reviewed_fallback": "더 짧은 줄부터 알려줄래?",
            },
            {
                "copy_slot": "choose_line",
                "purpose": "l2_question",
                "targets": [answer_target],
                "generation_brief": "두 줄 중 하나를 고르게 한다.",
                "reviewed_fallback": "더 짧은 줄을 골라줄래?",
            },
            {
                "copy_slot": "choose_reason",
                "purpose": "l2_question",
                "targets": [relation_target],
                "generation_brief": "비교 방법을 고르게 한다.",
                "reviewed_fallback": "어떻게 비교할지 골라줄래?",
            },
            {
                "copy_slot": "joint_intro",
                "purpose": "l0_intro",
                "targets": [answer_target, relation_target],
                "generation_brief": "함께 비교를 시작한다.",
                "reviewed_fallback": "나랑 같이 두 줄을 비교해 볼까?",
            },
            {
                "copy_slot": "joint_action",
                "purpose": "l0_action",
                "targets": [answer_target, relation_target],
                "generation_brief": "공동 수행 행동을 말한다.",
                "reviewed_fallback": "사람 수를 같이 가리켜 보자.",
            },
        ],
        "l2_plans": [
            {
                "plan_id": "choose_line_plan",
                "target": answer_target,
                "copy_slot": "choose_line",
                "choices": [
                    {
                        "choice_id": "left",
                        "label": "왼쪽 줄",
                        "effect": {
                            "verdict": "correct",
                            "fact_updates": [
                                {
                                    "fact_id": "shorter_line",
                                    "value": {
                                        "type": "choice",
                                        "choice_id": "left",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "choice_id": "right",
                        "label": "오른쪽 줄",
                        "effect": {
                            "verdict": "incorrect",
                            "misconception_code": "reversed_comparison",
                        },
                    },
                ],
            },
            {
                "plan_id": "choose_reason_plan",
                "target": relation_target,
                "copy_slot": "choose_reason",
                "choices": [
                    {
                        "choice_id": "compare",
                        "label": "사람 수를 비교해",
                        "effect": {
                            "verdict": "correct",
                            "relation_ids": ["compare_lines"],
                        },
                    },
                    {
                        "choice_id": "guess",
                        "label": "그냥 골라",
                        "effect": {
                            "verdict": "incorrect",
                            "misconception_code": "comparison_without_evidence",
                        },
                    },
                ],
            },
        ],
        "l0_joint_plan": {
            "action_id": "joint_compare",
            "intro_copy_slot": "joint_intro",
            "action_copy_slot": "joint_action",
            "completion_values": [
                {
                    "target_kind": "fact",
                    "target_id": "shorter_line",
                    "value": {"type": "choice", "choice_id": "left"},
                },
                {
                    "target_kind": "relation",
                    "target_id": "compare_lines",
                    "satisfied": True,
                },
            ],
            "button_label": "같이 비교하기",
        },
        "help_plan": {
            "H1": {
                "level": "H1",
                "support_type": "attention",
                "answer_policy": "hidden",
                "body": "두 줄의 사람 수를 각각 확인해 보자.",
            },
            "H2": {
                "level": "H2",
                "support_type": "guided_action",
                "answer_policy": "partial",
                "body": "두 수 중 작은 수가 있는 쪽을 찾아보자.",
                "action": "두 줄의 사람 수 비교하기",
            },
            "H3": {
                "level": "H3",
                "support_type": "joint_model",
                "answer_policy": "revealed",
                "body": "2명은 4명보다 적어서 왼쪽 줄이 더 짧아.",
                "action": "왼쪽 줄을 함께 고르기",
                "revealed_fact_ids": ["shorter_line"],
                "revealed_relation_ids": ["compare_lines"],
            },
        },
        "policies": {
            "entry_expression_level": "L4",
            "note_policy": (
                "verified_child_or_coauthored" if note_enabled else "none"
            ),
            "note_relation_ids": ["compare_lines"] if note_enabled else [],
            "note_skill_id": "compare_quantity" if note_enabled else None,
            "note_context": (
                "두 줄의 사람 수를 비교해 짧은 줄을 찾는 방법"
                if note_enabled
                else None
            ),
            "reviewed_direct_fallback": (
                "2명과 4명을 비교해 왼쪽 줄이 더 짧다고 알려줬어."
                if note_enabled
                else None
            ),
            "reviewed_coauthored_note": (
                "두 줄의 사람 수를 비교해 짧은 줄을 찾았어."
                if note_enabled
                else None
            ),
            "transition_text": "사람 수를 비교하는 방법을 다시 써 보자.",
        },
    }


def _scenario_pack() -> LifeScenarioPackV2:
    return LifeScenarioPackV2.model_validate(
        {
            "pack_id": "life.cafe.queue.v1",
            "content_version": 1,
            "scene": "cafe",
            "scenario_id": "cafe_queue",
            "task_stages": [
                {
                    "task_id": "queue_primary",
                    "default_variant_id": "default",
                    "variants": {
                        "default": _task_payload(
                            task_id="queue_primary",
                            pack_id="life.cafe.queue.primary.v1",
                            phase="primary",
                            note_enabled=True,
                        )
                    },
                },
                {
                    "task_id": "queue_transfer",
                    "default_variant_id": "default",
                    "variants": {
                        "default": _task_payload(
                            task_id="queue_transfer",
                            pack_id="life.cafe.queue.transfer.v1",
                            phase="transfer",
                            note_enabled=False,
                        )
                    },
                },
            ],
            "completion_projection": [
                {
                    "output_key": "final_choice",
                    "source_task_id": "queue_transfer",
                    "source_kind": "fact",
                    "source_id": "shorter_line",
                }
            ],
        }
    )


def test_life_scenario_runtime_v3_roundtrips_and_resolves_exact_pack() -> None:
    pack = _scenario_pack()
    evidence_id = "a" * 64
    snapshot = pin_life_scenario_runtime_v3(
        pack,
        task_note_states={
            "queue_primary": PinnedDialogueTaskNoteStateV3(
                independent_relation_evidence={
                    "compare_lines": [evidence_id],
                }
            ),
            "queue_transfer": PinnedDialogueTaskNoteStateV3(),
        },
        selector_reason="native_life_pack_canary_selected",
        canary_bucket=17,
    )

    restored = PinnedDialogueScenarioRuntimeV3.model_validate(
        snapshot.model_dump(mode="json")
    )
    resolved = resolve_life_scenario_runtime_v3(restored)

    assert resolved == pack
    assert restored.active_variant_ids == {
        "queue_primary": "default",
        "queue_transfer": "default",
    }
    assert restored.task_note_states[
        "queue_primary"
    ].independent_relation_evidence == {"compare_lines": [evidence_id]}
    assert restored.task_note_states["queue_transfer"].supported_relation_ids == []
    assert "stable_copy_plans" not in restored.model_dump(mode="json")


def test_existing_single_pack_v2_session_state_still_loads_without_v3() -> None:
    state = SessionState.model_validate(
        {
            "learner_id": 1,
            "scene": "home_teach",
            "scenario_id": "home_teach",
            "task_ids": ["home_teaching"],
            "expression_level": "L4",
            "runtime_contract_version": "verdict-v1",
            "pinned_dialogue_v2": {
                "pack_id": "required-home.number-count.v2",
                "content_version": 1,
                "source_hash": "b" * 64,
                "pack_snapshot": {"legacy": True},
                "reasoning_ledger": {},
                "stable_copy_plan_schema_version": "stable-copy-plan-set-v1",
                "stable_copy_plan_compiler_version": "stable-copy-plan-compiler-v1",
                "stable_copy_plan_set_hash": "c" * 64,
                "stable_copy_plans": {
                    f"slot_{index}": {} for index in range(5)
                },
                "selector_reason": "native_pack_canary_selected",
                "canary_bucket": 3,
            },
        }
    )

    assert state.pinned_dialogue_v2 is not None
    assert state.pinned_dialogue_scenario_v3 is None
    assert SessionState.model_validate(state.model_dump(mode="json")) == state


def test_v3_rejects_unknown_fields_and_mismatched_task_scopes() -> None:
    snapshot = pin_life_scenario_runtime_v3(
        _scenario_pack(),
        selector_reason="native_life_pack_canary_selected",
        canary_bucket=None,
    )
    payload = snapshot.model_dump(mode="json")
    payload["stable_copy_plans"] = {}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PinnedDialogueScenarioRuntimeV3.model_validate(payload)

    mismatched = snapshot.model_dump(mode="json")
    mismatched["task_note_states"].pop("queue_transfer")
    with pytest.raises(ValidationError, match="task scopes must match"):
        PinnedDialogueScenarioRuntimeV3.model_validate(mismatched)


def test_v3_resolve_rejects_payload_hash_and_variant_binding_tampering() -> None:
    snapshot = pin_life_scenario_runtime_v3(
        _scenario_pack(),
        selector_reason="native_life_pack_canary_selected",
        canary_bucket=4,
    )

    changed_payload = snapshot.model_copy(deep=True)
    changed_payload.scenario_pack_snapshot["content_version"] = 2
    with pytest.raises(DialogueV3ScenarioSnapshotError, match="identity or hash"):
        resolve_life_scenario_runtime_v3(changed_payload)

    changed_variant_payload = deepcopy(snapshot.model_dump(mode="json"))
    changed_variant_payload["active_variant_ids"]["queue_transfer"] = "missing"
    changed_variant = PinnedDialogueScenarioRuntimeV3.model_validate(
        changed_variant_payload
    )
    with pytest.raises(DialogueV3ScenarioSnapshotError, match="variant is unavailable"):
        resolve_life_scenario_runtime_v3(changed_variant)


def test_v3_task_note_state_is_strict_and_task_scoped() -> None:
    with pytest.raises(ValidationError, match="both independent and supported"):
        PinnedDialogueTaskNoteStateV3(
            independent_relation_evidence={"compare_lines": ["d" * 64]},
            supported_relation_ids=["compare_lines"],
        )
    with pytest.raises(ValidationError, match="exactly one note ID"):
        PinnedDialogueTaskNoteStateV3(note_emitted=True)

    snapshot = pin_life_scenario_runtime_v3(
        _scenario_pack(),
        selector_reason="native_life_pack_canary_selected",
        canary_bucket=4,
    )
    invalid_note_payload = snapshot.model_dump(mode="json")
    invalid_note_payload["task_note_states"]["queue_transfer"][
        "supported_relation_ids"
    ] = ["not_reviewed_for_note"]
    invalid_note = PinnedDialogueScenarioRuntimeV3.model_validate(
        invalid_note_payload
    )
    with pytest.raises(DialogueV3ScenarioSnapshotError, match="unreviewed relation"):
        resolve_life_scenario_runtime_v3(invalid_note)


def test_v3_session_state_roundtrip_keeps_v2_and_v3_fields_independent() -> None:
    snapshot = pin_life_scenario_runtime_v3(
        _scenario_pack(),
        selector_reason="native_life_pack_canary_selected",
        canary_bucket=9,
    )
    state = SessionState(
        learner_id=2,
        scene="cafe",
        scenario_id="cafe_queue",
        task_ids=["queue_primary", "queue_transfer"],
        expression_level=ExpressionLevel.L4,
        pinned_dialogue_scenario_v3=snapshot,
    )

    restored = SessionState.model_validate(state.model_dump(mode="json"))

    assert restored.pinned_dialogue_v2 is None
    assert restored.pinned_dialogue_scenario_v3 == snapshot
    assert set(restored.pinned_dialogue_scenario_v3.reasoning_ledgers) == {
        "queue_primary",
        "queue_transfer",
    }
