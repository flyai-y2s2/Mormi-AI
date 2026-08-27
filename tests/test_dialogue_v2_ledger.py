from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mormi_api.dialogue_v2_content import (
    RequiredHomeTeachingPackV2,
    required_home_content_pack_v2,
)
from mormi_api.dialogue_v2_evidence import (
    GuardedUnderstandingV2,
    guard_understanding_response_v2,
)
from mormi_api.dialogue_v2_ledger import (
    DialogueV2LedgerError,
    PinnedContentSnapshotV2,
    ReasoningLedgerV2,
    apply_guarded_understanding_v2,
    apply_structured_progress_v2,
    content_pack_hash_v2,
    empty_reasoning_ledger_v2,
    pin_content_pack_v2,
    reasoning_completion_v2,
)
from mormi_api.schemas import (
    AuxiliaryUnderstandingClaimV2,
    FactUnderstandingClaimV2,
    RelationUnderstandingClaimV2,
    UnderstandingRequestV2,
    UnderstandingResponseV2,
)


def _mixed_pack() -> RequiredHomeTeachingPackV2:
    return required_home_content_pack_v2("multiply-easy-tables")


def _request(
    pack: RequiredHomeTeachingPackV2,
    child_utterance: str,
) -> UnderstandingRequestV2:
    facts = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}
    relations = {
        relation.relation_id: relation for relation in pack.reasoning_graph.relations
    }
    targets: list[dict[str, object]] = []
    for target in pack.initial_question.targets:
        payload: dict[str, object] = {
            "target_kind": target.target_kind,
            "target_id": target.target_id,
            "ask_kind": target.ask_kind,
        }
        if target.target_kind == "fact":
            payload["expected_truth"] = facts[target.target_id].value.model_dump(mode="json")
        else:
            rubric = relations[target.target_id].rubric
            payload["rubric"] = {
                "sufficient": " / ".join(rubric.sufficient),
                "partial": " / ".join(rubric.partial),
                "incorrect": " / ".join(rubric.incorrect),
            }
        targets.append(payload)

    return UnderstandingRequestV2(
        task_id=pack.task_id,
        targets=targets,  # type: ignore[arg-type]
        claimable_graph={
            "fact_ids": [fact.fact_id for fact in pack.reasoning_graph.facts],
            "relation_ids": [
                relation.relation_id for relation in pack.reasoning_graph.relations
            ],
            "open_auxiliary_claims": pack.reasoning_graph.open_auxiliary_claims,
        },
        current_turn={
            "mormi_question": pack.initial_question.reviewed_fallback,
            "asks": list(
                dict.fromkeys(
                    target.ask_kind for target in pack.initial_question.targets
                )
            ),
            "expression_level": "L4",
            "hint_level": "H0",
        },
        child_utterance=child_utterance,
    )


def _guard(
    pack: RequiredHomeTeachingPackV2,
    child_utterance: str,
    response: UnderstandingResponseV2,
) -> GuardedUnderstandingV2:
    return guard_understanding_response_v2(
        _request(pack, child_utterance),
        response,
    )


def _partial_purchase_understanding(
    pack: RequiredHomeTeachingPackV2,
    *,
    fact_claim_id: str = "claim_purchase_total",
    relation_claim_id: str = "claim_sum_item_costs",
) -> GuardedUnderstandingV2:
    evidence = "5000+3000+3000해서 11000원이야"
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        support_need="none",
        contains_learning_evidence=True,
        answer_status="missing",
        reasoning_status="partial",
        confidence="high",
        claims=[
            FactUnderstandingClaimV2(
                claim_id=fact_claim_id,
                fact_id="purchase_total",
                claim_type="intermediate_result",
                evidence_span=evidence,
                interpreted_value={"type": "money", "amount": 11_000},
                verdict="correct",
            ),
            RelationUnderstandingClaimV2(
                claim_id=relation_claim_id,
                relation_id="sum_item_costs",
                claim_type="procedure_step",
                evidence_span=evidence,
                verdict="correct",
                arithmetic_interpretation={
                    "operation": "addition",
                    "operands": [5_000, 3_000, 3_000],
                    "result": 11_000,
                    "mathematical_validity": "correct",
                },
            ),
        ],
    )
    return _guard(pack, evidence, response)


def test_pinned_snapshot_hashes_the_complete_pack_and_rejects_drift() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)

    assert snapshot.pack_id == pack.pack_id
    assert snapshot.content_hash == content_pack_hash_v2(pack)
    assert snapshot.resolve_pack() == pack

    payload = snapshot.model_dump(mode="json")
    payload["pack_payload"]["title"] = "배포 뒤 바뀐 제목"
    with pytest.raises(ValidationError, match="hash does not match"):
        PinnedContentSnapshotV2.model_validate(payload)

    # Frozen Pydantic models are shallowly frozen. Even an accidental nested
    # mutation is detected before the snapshot can drive another turn.
    snapshot.pack_payload["title"] = "메모리에서 바뀐 제목"
    with pytest.raises(DialogueV2LedgerError, match="integrity"):
        snapshot.resolve_pack()


def test_mixed_11000_is_monotonic_milestone_progress_not_completion() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    ledger = empty_reasoning_ledger_v2(snapshot)

    result = apply_guarded_understanding_v2(
        snapshot,
        ledger,
        _partial_purchase_understanding(pack),
        source_turn_id="turn_mixed_partial",
    )

    purchase = result.ledger.verified_facts["purchase_total"]
    assert purchase.canonical_value.model_dump(mode="json") == {
        "type": "money",
        "amount": 11_000,
        "currency": "KRW",
    }
    assert result.new_fact_ids == ["purchase_total"]
    assert result.new_milestone_fact_ids == ["purchase_total"]
    assert result.new_relation_ids == ["sum_item_costs"]
    assert result.has_new_canonical_progress is True
    assert result.completion.model_dump(mode="json") == {
        "required_fact_ids": ["shortage"],
        "required_relation_ids": ["calculate_shortage"],
        "remaining_fact_ids": ["shortage"],
        "remaining_relation_ids": ["calculate_shortage"],
        "complete": False,
    }
    assert result.completion_became_true is False

    # Raw child text remains in the guarded per-turn object, not duplicated in
    # the conversation ledger. Only offsets and semantic audit data are pinned.
    serialized_ledger = json.dumps(result.ledger.model_dump(mode="json"), ensure_ascii=False)
    assert "5000+3000+3000해서 11000원이야" not in serialized_ledger


def test_required_fact_and_relation_are_the_only_completion_authority() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    partial = apply_guarded_understanding_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        _partial_purchase_understanding(pack),
        source_turn_id="turn_partial",
    )
    utterance = "11000원에서 10000원을 빼면 1000원이 모자라"
    guarded = _guard(
        pack,
        utterance,
        UnderstandingResponseV2(
            utterance_class="learning_response",
            support_need="none",
            contains_learning_evidence=True,
            answer_status="complete",
            reasoning_status="sufficient",
            claims=[
                FactUnderstandingClaimV2(
                    claim_id="claim_shortage",
                    fact_id="shortage",
                    claim_type="final_answer",
                    evidence_span="1000원이 모자라",
                    interpreted_value={"type": "money", "amount": 1_000},
                    verdict="correct",
                ),
                RelationUnderstandingClaimV2(
                    claim_id="claim_calculate_shortage",
                    relation_id="calculate_shortage",
                    claim_type="procedure_step",
                    evidence_span=utterance,
                    verdict="sufficient",
                    arithmetic_interpretation={
                        "operation": "subtraction",
                        "operands": [11_000, 10_000],
                        "result": 1_000,
                        "mathematical_validity": "correct",
                    },
                ),
            ],
        ),
    )

    completed = apply_guarded_understanding_v2(
        snapshot,
        partial.ledger,
        guarded,
        source_turn_id="turn_completion",
    )

    assert completed.new_fact_ids == ["shortage"]
    assert completed.new_relation_ids == ["calculate_shortage"]
    assert completed.completion.remaining_fact_ids == []
    assert completed.completion.remaining_relation_ids == []
    assert completed.completion.complete is True
    assert completed.completion_became_true is True


def test_correct_verdict_is_not_regraded_against_value_or_arithmetic() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    utterance = "99999원이고 3에서 2를 빼면 999야"
    guarded = _guard(
        pack,
        utterance,
        UnderstandingResponseV2(
            utterance_class="learning_response",
            contains_learning_evidence=True,
            claims=[
                FactUnderstandingClaimV2(
                    claim_id="claim_value_not_regraded",
                    fact_id="shortage",
                    claim_type="final_answer",
                    evidence_span="99999원",
                    interpreted_value={"type": "money", "amount": 99_999},
                    verdict="correct",
                ),
                RelationUnderstandingClaimV2(
                    claim_id="claim_math_not_regraded",
                    relation_id="calculate_shortage",
                    claim_type="procedure_step",
                    evidence_span="3에서 2를 빼면 999야",
                    verdict="correct",
                    arithmetic_interpretation={
                        "operation": "subtraction",
                        "operands": [3, 2],
                        "result": 999,
                        "mathematical_validity": "incorrect",
                    },
                ),
            ],
        ),
    )

    result = apply_guarded_understanding_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        guarded,
        source_turn_id="turn_no_readjudication",
    )

    fact = result.ledger.verified_facts["shortage"]
    assert fact.canonical_value.model_dump(mode="json")["amount"] == 1_000
    # Sonnet's verdict advances the canonical graph without a second numeric
    # judgment, while its model-authored values remain turn-local and are not
    # copied into the durable ledger.
    persisted = result.ledger.model_dump_json()
    assert "99999" not in persisted
    assert '"result":999' not in persisted
    assert "claim_value_not_regraded" not in persisted
    assert "claim_math_not_regraded" not in persisted
    assert result.completion.complete is True


def test_incorrect_verdict_cannot_verify_truth_or_erase_prior_progress() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    initially_verified = apply_guarded_understanding_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        _partial_purchase_understanding(pack),
        source_turn_id="turn_first",
    ).ledger
    utterance = "11000원이야"
    guarded = _guard(
        pack,
        utterance,
        UnderstandingResponseV2(
            utterance_class="learning_response",
            contains_learning_evidence=True,
            claims=[
                FactUnderstandingClaimV2(
                    claim_id="claim_later_incorrect",
                    fact_id="purchase_total",
                    claim_type="intermediate_result",
                    evidence_span=utterance,
                    interpreted_value={"type": "money", "amount": 11_000},
                    verdict="incorrect",
                ),
                FactUnderstandingClaimV2(
                    claim_id="claim_true_value_marked_incorrect",
                    fact_id="shortage",
                    claim_type="final_answer",
                    evidence_span=utterance,
                    interpreted_value={"type": "money", "amount": 1_000},
                    verdict="incorrect",
                ),
            ],
        ),
    )

    result = apply_guarded_understanding_v2(
        snapshot,
        initially_verified,
        guarded,
        source_turn_id="turn_incorrect",
    )

    assert set(result.ledger.verified_facts) == {"purchase_total"}
    assert result.new_fact_ids == []
    assert result.ignored_claim_ids == [
        "claim_later_incorrect",
        "claim_true_value_marked_incorrect",
    ]
    assert result.completion.complete is False


def test_reapplying_same_semantic_evidence_is_idempotent_even_if_claim_ids_change() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    first = apply_guarded_understanding_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        _partial_purchase_understanding(pack),
        source_turn_id="turn_retry",
    )
    retried_guarded = _partial_purchase_understanding(
        pack,
        fact_claim_id="provider_retry_fact_id",
        relation_claim_id="provider_retry_relation_id",
    )

    retried = apply_guarded_understanding_v2(
        snapshot,
        first.ledger,
        retried_guarded,
        source_turn_id="turn_retry",
    )

    assert retried.ledger == first.ledger
    assert retried.new_fact_ids == []
    assert retried.new_relation_ids == []
    assert retried.new_fact_evidence_ids == []
    assert retried.new_relation_evidence_ids == []
    assert retried.completion_became_true is False


def test_auxiliary_evidence_is_preserved_without_satisfying_completion() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    utterance = "가격을 하나씩 적었고 표부터 계산할 수도 있어"
    guarded = _guard(
        pack,
        utterance,
        UnderstandingResponseV2(
            utterance_class="learning_response",
            contains_learning_evidence=True,
            reasoning_status="partial",
            claims=[
                AuxiliaryUnderstandingClaimV2(
                    claim_id="claim_aux_correct",
                    evidence_span="가격을 하나씩 적었고",
                    summary="각 가격을 따로 정리했다",
                    verdict="correct",
                ),
                AuxiliaryUnderstandingClaimV2(
                    claim_id="claim_aux_partial",
                    evidence_span="표부터 계산할 수도 있어",
                    summary="표 금액부터 계산하려는 시도",
                    verdict="partial",
                ),
                AuxiliaryUnderstandingClaimV2(
                    claim_id="claim_aux_uncertain",
                    evidence_span="계산할 수도 있어",
                    summary="의도가 불확실한 계산 언급",
                    verdict="uncertain",
                ),
            ],
        ),
    )

    first = apply_guarded_understanding_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        guarded,
        source_turn_id="turn_auxiliary",
    )
    retried = apply_guarded_understanding_v2(
        snapshot,
        first.ledger,
        guarded,
        source_turn_id="turn_auxiliary",
    )

    assert len(first.ledger.accepted_auxiliary_evidence) == 2
    assert len(first.new_auxiliary_evidence_ids) == 2
    assert first.ignored_claim_ids == ["claim_aux_uncertain"]
    assert first.has_new_canonical_progress is False
    assert first.completion.complete is False
    assert retried.ledger == first.ledger
    assert retried.new_auxiliary_evidence_ids == []


def test_auxiliary_ledger_replaces_model_summary_with_a_pii_free_code() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    child_utterance = "김민수는 서울 강남구 역삼동 123에서 가격부터 적었어"
    guarded = _guard(
        pack,
        child_utterance,
        UnderstandingResponseV2(
            utterance_class="learning_response",
            contains_learning_evidence=True,
            reasoning_status="partial",
            claims=[
                AuxiliaryUnderstandingClaimV2(
                    claim_id="김민수-01012345678",
                    evidence_span=child_utterance,
                    summary="김민수가 서울 강남구 역삼동 123에서 가격을 정리했다",
                    verdict="partial",
                    interpreted_value={"type": "text", "text": child_utterance},
                    arithmetic_interpretation={
                        "operation": "addition",
                        "operands": [10_123_456_78, 123],
                        "result": 10_123_458_01,
                        "mathematical_validity": "correct",
                    },
                )
            ],
        ),
    )

    result = apply_guarded_understanding_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        guarded,
        source_turn_id="turn_aux_private",
    )

    persisted = result.ledger.model_dump_json()
    assert len(result.ledger.accepted_auxiliary_evidence) == 1
    accepted = next(iter(result.ledger.accepted_auxiliary_evidence.values()))
    assert accepted.summary == "task_related_auxiliary_evidence"
    assert child_utterance not in persisted
    assert "김민수" not in persisted
    assert "서울 강남구 역삼동 123" not in persisted
    assert "01012345678" not in persisted
    assert "1012345801" not in persisted
    # Turn-local model IDs are explicitly excluded from apply-result dumps too.
    assert "김민수" not in result.model_dump_json()


def test_ledger_cannot_be_used_with_a_different_pinned_pack() -> None:
    mixed_snapshot = pin_content_pack_v2(_mixed_pack())
    count_snapshot = pin_content_pack_v2(required_home_content_pack_v2("number-count"))
    ledger = empty_reasoning_ledger_v2(mixed_snapshot)

    with pytest.raises(DialogueV2LedgerError, match="not bound"):
        reasoning_completion_v2(count_snapshot, ledger)

    # The public type remains round-trip safe for SessionState JSON embedding.
    assert ReasoningLedgerV2.model_validate(ledger.model_dump(mode="json")) == ledger


def test_pinned_l2_effects_use_the_same_ledger_without_an_understanding_verdict() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    ledger = empty_reasoning_ledger_v2(snapshot)
    answer_plan = next(
        plan for plan in pack.l2_plans if plan.target.target_kind == "fact"
    )
    answer_effect = next(
        choice.effect for choice in answer_plan.choices if choice.effect.verdict == "correct"
    )
    assert answer_effect.interpreted_value is not None

    answer_result = apply_structured_progress_v2(
        snapshot,
        ledger,
        fact_values={answer_effect.target_id: answer_effect.interpreted_value},
        relation_ids=[],
        source_turn_id="turn_l2_answer",
        source_kind="choice",
    )

    assert answer_result.new_fact_ids == ["shortage"]
    assert answer_result.completion.complete is False
    evidence = answer_result.ledger.verified_facts["shortage"].evidence[0]
    assert evidence.evidence_kind == "structured"
    assert evidence.source_kind == "choice"  # type: ignore[union-attr]

    method_plan = next(
        plan for plan in pack.l2_plans if plan.target.target_kind == "relation"
    )
    method_effect = next(
        choice.effect for choice in method_plan.choices if choice.effect.verdict == "correct"
    )
    completed = apply_structured_progress_v2(
        snapshot,
        answer_result.ledger,
        fact_values={},
        relation_ids=[method_effect.target_id],
        source_turn_id="turn_l2_method",
        source_kind="choice",
    )

    assert completed.new_relation_ids == ["calculate_shortage"]
    assert completed.completion.complete is True


def test_pinned_l0_joint_completion_is_atomic_idempotent_and_tamper_safe() -> None:
    pack = _mixed_pack()
    snapshot = pin_content_pack_v2(pack)
    fact_values = {
        completion.target_id: completion.value
        for completion in pack.l0_joint_plan.completion_values
        if completion.target_kind == "fact"
    }
    relation_ids = [
        completion.target_id
        for completion in pack.l0_joint_plan.completion_values
        if completion.target_kind == "relation"
    ]

    completed = apply_structured_progress_v2(
        snapshot,
        empty_reasoning_ledger_v2(snapshot),
        fact_values=fact_values,
        relation_ids=relation_ids,
        source_turn_id="turn_l0_joint",
        source_kind="joint",
    )
    retried = apply_structured_progress_v2(
        snapshot,
        completed.ledger,
        fact_values=fact_values,
        relation_ids=relation_ids,
        source_turn_id="turn_l0_joint",
        source_kind="joint",
    )

    assert completed.completion.complete is True
    assert completed.completion_became_true is True
    assert retried.ledger == completed.ledger
    assert retried.new_fact_evidence_ids == []
    assert retried.new_relation_evidence_ids == []
    assert retried.completion_became_true is False
    assert (
        ReasoningLedgerV2.model_validate(completed.ledger.model_dump(mode="json"))
        == completed.ledger
    )

    with pytest.raises(DialogueV2LedgerError, match="differs from pinned"):
        apply_structured_progress_v2(
            snapshot,
            empty_reasoning_ledger_v2(snapshot),
            fact_values={"shortage": {"type": "money", "amount": 999_999}},
            relation_ids=[],
            source_turn_id="turn_tampered_joint",
            source_kind="joint",
        )
