from __future__ import annotations

from typing import Literal

import pytest

from mormi_api.dialogue_v2_evidence import (
    EvidenceGuardViolationCodeV2,
    EvidenceMatchKindV2,
    UnderstandingEvidenceGuardError,
    admit_understanding_response_v2,
    guard_understanding_response_v2,
)
from mormi_api.schemas import (
    AuxiliaryUnderstandingClaimV2,
    FactUnderstandingClaimV2,
    RelationUnderstandingClaimV2,
    UnderstandingRequestV2,
    UnderstandingResponseV2,
)


def _request(
    child_utterance: str,
    *,
    open_auxiliary_claims: bool = True,
) -> UnderstandingRequestV2:
    return UnderstandingRequestV2(
        task_id="money-budget",
        visible_facts={"budget": 10_000},
        fact_contexts=[
            {
                "fact_id": "budget",
                "speaker_label": "예산",
                "semantic_aliases": ["쓸 수 있는 돈"],
                "visible": True,
            },
            {
                "fact_id": "purchase_total",
                "speaker_label": "전체 값",
                "semantic_aliases": [],
                "visible": False,
            },
            {
                "fact_id": "shortage",
                "speaker_label": "모자라는 돈",
                "semantic_aliases": [],
                "visible": False,
            },
        ],
        targets=[
            {
                "target_kind": "fact",
                "target_id": "shortage",
                "ask_kind": "answer",
                "expected_truth": {"type": "money", "amount": 1_000},
            }
        ],
        claimable_graph={
            "fact_ids": ["budget", "purchase_total", "shortage"],
            "relation_ids": ["sum_item_costs", "calculate_shortage"],
            "open_auxiliary_claims": open_auxiliary_claims,
        },
        current_turn={
            "mormi_question": "돈이 얼마나 모자라는지 알려줄 수 있어?",
            "asks": ["answer"],
            "expression_level": "L4",
            "hint_level": "H0",
        },
        child_utterance=child_utterance,
    )


def _with_visible_help_card(
    request: UnderstandingRequestV2,
) -> UnderstandingRequestV2:
    payload = request.model_dump(mode="json")
    payload["current_turn"]["hint_level"] = "H1"
    payload["visible_ui_elements"] = [
        {
            "element_id": "help_card.h1",
            "kind": "help_card",
            "text": "낸 돈과 쿠키 값을 주의 깊게 보자.",
            "hint_level": "H1",
        }
    ]
    return UnderstandingRequestV2.model_validate(payload)


def test_ui_reference_is_grounded_without_becoming_learning_evidence() -> None:
    child_text = "저걸 왜 주의 깊게 봐야 돼?"
    request = _with_visible_help_card(_request(child_text))
    response = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "task_question",
            "conversation_move": "task_question",
            "move_subject": "task",
            "question_focus": "reason_or_method",
            "support_need": "concept",
            "ui_reference": {
                "element_id": "help_card.h1",
                "interaction": "asks_why",
                "reference_basis": "deictic",
                "evidence_span": "저걸",
            },
        }
    )

    admission = admit_understanding_response_v2(request, response)

    assert admission.guarded.response.ui_reference is not None
    assert admission.guarded.response.claims == []
    assert admission.guarded.response.contains_learning_evidence is False


def test_invalid_ui_reference_is_dropped_without_erasing_valid_claim() -> None:
    child_text = "1000원이고 저걸 왜 봐?"
    request = _with_visible_help_card(_request(child_text))
    response = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "conversation_move": "task_question",
            "move_subject": "task",
            "question_focus": "reason_or_method",
            "support_need": "concept",
            "contains_learning_evidence": True,
            "answer_status": "complete",
            "claims": [
                {
                    "claim_kind": "fact",
                    "claim_id": "claim_shortage",
                    "fact_id": "shortage",
                    "claim_type": "final_answer",
                    "evidence_span": "1000원",
                    "interpreted_value": {"type": "money", "amount": 1000},
                    "verdict": "correct",
                }
            ],
            "ui_reference": {
                "element_id": "help_card.not_visible",
                "interaction": "asks_why",
                "reference_basis": "deictic",
                "evidence_span": "저걸",
            },
        }
    )

    admission = admit_understanding_response_v2(request, response)

    assert admission.guarded.response.ui_reference is None
    assert [claim.claim_id for claim in admission.guarded.response.claims] == [
        "claim_shortage"
    ]
    assert admission.quarantined_claims == []


@pytest.mark.parametrize(
    ("child_utterance", "interpreted_amount", "verdict"),
    [
        ("2900", 2_900, "correct"),
        ("1000", 1_000, "incorrect"),
    ],
)
def test_guard_preserves_sonnet_verdict_without_expected_value_comparison(
    child_utterance: str,
    interpreted_amount: int,
    verdict: Literal["correct", "incorrect"],
) -> None:
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        answer_status="complete",
        claims=[
            FactUnderstandingClaimV2(
                claim_id="claim_shortage",
                fact_id="shortage",
                claim_type="final_answer",
                evidence_span=child_utterance,
                interpreted_value={"type": "money", "amount": interpreted_amount},
                verdict=verdict,
            )
        ],
    )

    guarded = guard_understanding_response_v2(request, response)

    claim = guarded.response.claims[0]
    assert isinstance(claim, FactUnderstandingClaimV2)
    assert claim.verdict == verdict
    assert claim.interpreted_value.model_dump(mode="json") == {
        "type": "money",
        "amount": interpreted_amount,
        "currency": "KRW",
    }


def test_guard_preserves_arithmetic_interpretation_without_recalculation() -> None:
    child_utterance = "5000+3000+3000해서 11000원이야"
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        reasoning_status="partial",
        claims=[
            RelationUnderstandingClaimV2(
                claim_id="claim_sum",
                relation_id="sum_item_costs",
                claim_type="procedure_step",
                evidence_span=child_utterance,
                verdict="correct",
                arithmetic_interpretation={
                    "operation": "addition",
                    "operands": [5_000, 3_000, 3_000],
                    "result": 99_999,
                    "mathematical_validity": "correct",
                },
            )
        ],
    )

    guarded = guard_understanding_response_v2(request, response)

    claim = guarded.response.claims[0]
    assert isinstance(claim, RelationUnderstandingClaimV2)
    assert claim.verdict == "correct"
    assert claim.arithmetic_interpretation is not None
    assert claim.arithmetic_interpretation.result == 99_999


def test_guard_records_exact_source_boundaries() -> None:
    child_utterance = "음, 2900원이야!"
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        claims=[
            FactUnderstandingClaimV2(
                claim_id="claim_shortage",
                fact_id="shortage",
                claim_type="final_answer",
                evidence_span="2900원이야",
                interpreted_value={"type": "money", "amount": 2_900},
                verdict="correct",
            )
        ],
    )

    guarded = guard_understanding_response_v2(request, response)

    assert guarded.evidence_matches[0].model_dump(mode="json") == {
        "claim_id": "claim_shortage",
        "source_start": 3,
        "source_end": 10,
        "source_text": "2900원이야",
        "match_kind": "exact",
    }


def test_guard_allows_only_nfc_equivalence_and_maps_to_raw_boundaries() -> None:
    decomposed_ga = "\u1100\u1161"
    child_utterance = f"{decomposed_ga}라고 했어"
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        claims=[
            AuxiliaryUnderstandingClaimV2(
                claim_id="claim_auxiliary",
                evidence_span="가",
                summary="가라고 말했다",
                verdict="correct",
            )
        ],
    )

    guarded = guard_understanding_response_v2(request, response)

    match = guarded.evidence_matches[0]
    assert match.source_start == 0
    assert match.source_end == 2
    assert match.source_text == decomposed_ga
    assert match.match_kind is EvidenceMatchKindV2.UNICODE_NFC


@pytest.mark.parametrize(
    ("child_utterance", "invented_evidence"),
    [
        ("2,900원이야", "2900"),
        ("2900 원이야", "2900원"),
        ("이천구백 원이야", "2900"),
        ("둘을 합치면 돼", "둘을 더하면 돼"),
    ],
)
def test_guard_rejects_non_literal_normalization_or_paraphrase(
    child_utterance: str,
    invented_evidence: str,
) -> None:
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        claims=[
            AuxiliaryUnderstandingClaimV2(
                claim_id="claim_auxiliary",
                evidence_span=invented_evidence,
                summary="아이 발화를 해석한 요약",
                verdict="correct",
            )
        ],
    )

    with pytest.raises(UnderstandingEvidenceGuardError) as error:
        guard_understanding_response_v2(request, response)

    assert [violation.code for violation in error.value.violations] == [
        EvidenceGuardViolationCodeV2.EVIDENCE_NOT_LITERAL
    ]
    assert child_utterance not in str(error.value)


def test_guard_rejects_unknown_graph_ids_and_disabled_auxiliary_claims() -> None:
    child_utterance = "11000원이고 가격을 적었어"
    request = _request(child_utterance, open_auxiliary_claims=False)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        claims=[
            FactUnderstandingClaimV2(
                claim_id="claim_unknown_fact",
                fact_id="invented_total",
                claim_type="intermediate_result",
                evidence_span="11000원",
                interpreted_value={"type": "money", "amount": 11_000},
                verdict="correct",
            ),
            RelationUnderstandingClaimV2(
                claim_id="claim_unknown_relation",
                relation_id="invented_relation",
                claim_type="procedure_step",
                evidence_span="가격을 적었어",
                verdict="sufficient",
            ),
            AuxiliaryUnderstandingClaimV2(
                claim_id="claim_disabled_auxiliary",
                evidence_span="가격을 적었어",
                summary="가격을 별도로 기록했다",
                verdict="correct",
            ),
        ],
    )

    with pytest.raises(UnderstandingEvidenceGuardError) as error:
        guard_understanding_response_v2(request, response)

    assert [(item.claim_id, item.code) for item in error.value.violations] == [
        (
            "claim_unknown_fact",
            EvidenceGuardViolationCodeV2.UNKNOWN_FACT_ID,
        ),
        (
            "claim_unknown_relation",
            EvidenceGuardViolationCodeV2.UNKNOWN_RELATION_ID,
        ),
        (
            "claim_disabled_auxiliary",
            EvidenceGuardViolationCodeV2.AUXILIARY_CLAIMS_DISABLED,
        ),
    ]


def test_guard_accepts_claim_free_non_learning_response() -> None:
    request = _request("너 AI잖아")
    response = UnderstandingResponseV2(
        utterance_class="non_learning_safe",
        non_learning_kind="meta",
    )

    guarded = guard_understanding_response_v2(request, response)

    assert guarded.response == response
    assert guarded.evidence_matches == []


@pytest.mark.parametrize("child_utterance", ["7권", "3,000원이야", "6번이야~"])
def test_guard_rejects_bare_answer_as_sufficient_method_evidence(
    child_utterance: str,
) -> None:
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        answer_status="complete",
        reasoning_status="sufficient",
        claims=[
            RelationUnderstandingClaimV2(
                claim_id="claim_invented_method",
                relation_id="calculate_shortage",
                claim_type="procedure_step",
                evidence_span=child_utterance,
                verdict="sufficient",
            )
        ],
    )

    with pytest.raises(UnderstandingEvidenceGuardError) as error:
        guard_understanding_response_v2(request, response)

    assert [violation.code for violation in error.value.violations] == [
        EvidenceGuardViolationCodeV2.RELATION_EVIDENCE_IS_BARE_RESULT
    ]


@pytest.mark.parametrize(
    "child_utterance",
    [
        "6000나누기 2는 3000이니까",
        "6000원을 2로 나누면 3000원이야",
        "6000나누기2",
        "600+100=700",
        "네가 낸 돈에서 쿠키 값을 빼면 돼",
    ],
)
def test_guard_keeps_actual_method_evidence_for_model_adjudication(
    child_utterance: str,
) -> None:
    request = _request(child_utterance)
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        contains_learning_evidence=True,
        reasoning_status="sufficient",
        claims=[
            RelationUnderstandingClaimV2(
                claim_id="claim_actual_method",
                relation_id="calculate_shortage",
                claim_type="procedure_step",
                evidence_span=child_utterance,
                verdict="sufficient",
            )
        ],
    )

    guarded = guard_understanding_response_v2(request, response)

    assert guarded.response == response
    assert guarded.evidence_matches[0].source_text == child_utterance


def test_claim_admission_preserves_valid_answer_and_quarantines_bare_method() -> None:
    child_utterance = "600원이야"
    request = _request(child_utterance)
    response = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "contains_learning_evidence": True,
            "answer_status": "complete",
            "reasoning_status": "sufficient",
            "claims": [
                {
                    "claim_kind": "fact",
                    "claim_id": "answer",
                    "fact_id": "shortage",
                    "claim_type": "final_answer",
                    "evidence_span": child_utterance,
                    "interpreted_value": {"type": "money", "amount": 600},
                    "verdict": "correct",
                },
                {
                    "claim_kind": "relation",
                    "claim_id": "invented_method",
                    "relation_id": "calculate_shortage",
                    "claim_type": "procedure_step",
                    "evidence_span": child_utterance,
                    "verdict": "sufficient",
                },
            ],
        }
    )

    admission = admit_understanding_response_v2(request, response)

    assert [claim.claim_id for claim in admission.guarded.response.claims] == ["answer"]
    assert [match.claim_id for match in admission.guarded.evidence_matches] == ["answer"]
    assert [item.code for item in admission.quarantined_claims] == [
        EvidenceGuardViolationCodeV2.RELATION_EVIDENCE_IS_BARE_RESULT
    ]
    assert admission.needs_repair is True
