from __future__ import annotations

import json

import pytest
from conftest import FakeGateway
from pydantic import ValidationError
from sqlalchemy import select

from mormi_api.db import Database, DialogueTurnObservationRecord
from mormi_api.dialogue_v2_content import RequiredHomeTeachingPackV2
from mormi_api.dialogue_v2_copy import (
    STABLE_COPY_PLAN_COMPILER_VERSION_V2,
    STABLE_COPY_PLAN_SCHEMA_VERSION_V2,
    StableCopyArtifactMetadataV2,
    StableCopyResolutionV2,
)
from mormi_api.dialogue_v2_ledger import PinnedContentSnapshotV2, ReasoningLedgerV2
from mormi_api.dialogue_v2_life_content import (
    LIFE_MATERIALIZER_VERSION_V2,
    LifeScenarioPackV2,
    LifeTaskPackV2,
)
from mormi_api.dialogue_v2_versions import (
    DIALOGUE_V2_SNAPSHOT_COMPONENTS_V2,
    DIALOGUE_V2_SNAPSHOT_READER_CAPABILITY_V2,
    DIALOGUE_V3_SNAPSHOT_COMPONENTS_V1,
    DIALOGUE_V3_SNAPSHOT_READER_CAPABILITY_V1,
)
from mormi_api.engine import ConversationEngine
from mormi_api.llm import (
    UNDERSTANDING_V2_SYSTEM,
    ClaudeGateway,
    _internal_understanding_response,
    structured_output_schema,
)
from mormi_api.main import app
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ArithmeticInterpretationV2,
    AuxiliaryUnderstandingClaimV2,
    ChildResponse,
    ClaimableGraphContextV2,
    DialogueRuntimeContractVersion,
    FactUnderstandingClaimV2,
    ModelUnderstandingResponseV2,
    PinnedDialogueRuntimeV2,
    PinnedDialogueScenarioRuntimeV3,
    PinnedDialogueTaskNoteStateV3,
    RelationUnderstandingClaimV2,
    SessionCreate,
    SessionState,
    TurnContract,
    UnderstandingRequestV2,
    UnderstandingResponseV2,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService
from mormi_api.settings import Settings


def _schema_literal(model: type[object], property_name: str) -> str:
    schema = model.model_json_schema()  # type: ignore[attr-defined]
    return str(schema["properties"][property_name]["const"])


def test_snapshot_reader_capability_covers_every_persisted_v2_component() -> None:
    assert DIALOGUE_V2_SNAPSHOT_READER_CAPABILITY_V2 == (
        "dialogue-v2-snapshot-reader-v2"
    )
    assert (
        _schema_literal(PinnedDialogueRuntimeV2, "schema_version"),
        _schema_literal(PinnedContentSnapshotV2, "schema_version"),
        _schema_literal(RequiredHomeTeachingPackV2, "schema_version"),
        _schema_literal(ReasoningLedgerV2, "schema_version"),
        STABLE_COPY_PLAN_SCHEMA_VERSION_V2,
        STABLE_COPY_PLAN_COMPILER_VERSION_V2,
        _schema_literal(StableCopyResolutionV2, "snapshot_schema_version"),
        _schema_literal(StableCopyArtifactMetadataV2, "artifact_schema_version"),
        _schema_literal(TurnContract, "schema_version"),
    ) == DIALOGUE_V2_SNAPSHOT_COMPONENTS_V2


def test_v3_reader_is_an_explicit_superset_of_home_and_life_snapshots() -> None:
    assert DIALOGUE_V3_SNAPSHOT_READER_CAPABILITY_V1 == (
        "dialogue-v3-snapshot-reader-v1"
    )
    assert (
        *DIALOGUE_V2_SNAPSHOT_COMPONENTS_V2,
        _schema_literal(PinnedDialogueScenarioRuntimeV3, "schema_version"),
        _schema_literal(PinnedDialogueTaskNoteStateV3, "schema_version"),
        _schema_literal(LifeScenarioPackV2, "schema_version"),
        LIFE_MATERIALIZER_VERSION_V2,
        _schema_literal(LifeTaskPackV2, "schema_version"),
    ) == DIALOGUE_V3_SNAPSHOT_COMPONENTS_V1


def test_turn_contract_reader_versions_new_rows_and_accepts_legacy_rows() -> None:
    legacy_payload = {
        "turn_id": "turn_legacy",
        "scene": "home_teach",
        "scenario_id": "home_teach",
        "task_id": "home_teaching",
        "stage_id": "home_teach",
        "task_index": 0,
        "mormi": {"text": "알려줄 수 있어?", "mood": "curious"},
        "input": {"kind": "text"},
        "visual": {"type": "home_teaching"},
        "status": "active",
        "state_version": 1,
    }

    restored = TurnContract.model_validate(legacy_payload)

    assert restored.schema_version == "turn-contract-v1"
    assert restored.model_dump(mode="json")["schema_version"] == "turn-contract-v1"


def test_runtime_contract_flag_defaults_to_legacy_and_rejects_unknown_values() -> None:
    settings = Settings(_env_file=None)

    assert (
        settings.runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )
    assert (
        Settings(
            _env_file=None,
            runtime_contract_version="verdict-v1",
        ).runtime_contract_version
        is DialogueRuntimeContractVersion.VERDICT_V1
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, runtime_contract_version="future-v9")

    Settings(
        _env_file=None,
        runtime_contract_version="verdict-v1",
    ).validate_runtime_safety()


def test_understanding_prompt_defines_expression_block_without_surface_code_rules() -> None:
    for child_text in (
        "뭐라고 설명할지 모르겠어",
        "설명하기 어려워",
        "뭐라고 말해야 할지 모르겠어",
    ):
        assert child_text in UNDERSTANDING_V2_SYSTEM
    assert "support_need=expression" in UNDERSTANDING_V2_SYSTEM
    assert "계산 방법 자체를 모르겠어" in UNDERSTANDING_V2_SYSTEM


def test_legacy_state_without_runtime_contract_field_loads_as_legacy() -> None:
    payload = {
        "learner_id": 1,
        "scene": "cafe",
        "scenario_id": "cafe_queue_demo",
        "task_ids": ["cafe_queue_count"],
        "expression_level": "L4",
    }

    state = SessionState.model_validate(payload)

    assert state.runtime_contract_version is DialogueRuntimeContractVersion.LEGACY_V1
    assert (
        state.model_dump(mode="json")["runtime_contract_version"]
        == DialogueRuntimeContractVersion.LEGACY_V1.value
    )

    pinned_v2 = state.model_copy(
        update={
            "runtime_contract_version": DialogueRuntimeContractVersion.VERDICT_V1,
        }
    )
    assert SessionState.model_validate(
        pinned_v2.model_dump(mode="json")
    ).runtime_contract_version is DialogueRuntimeContractVersion.VERDICT_V1


def test_v2_understanding_contract_preserves_correct_intermediate_progress() -> None:
    evidence = "5000+3000+3000해서 11000원이야"
    request = UnderstandingRequestV2(
        task_id="home_teaching",
        visible_facts={"budget": 10_000},
        fact_contexts=[
            {
                "fact_id": "budget",
                "speaker_label": "예산",
                "semantic_aliases": [],
                "visible": True,
            },
            {
                "fact_id": "purchase_total",
                "speaker_label": "구매 합계",
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
                "rubric": {"correct": "모자란 금액을 말한다"},
                "expected_truth": {"type": "money", "amount": 1_000},
            },
            {
                "target_kind": "relation",
                "target_id": "calculate_shortage",
                "ask_kind": "reason_or_method",
                "rubric": {"sufficient": "전체 금액에서 예산을 비교한다"},
                "semantic_contract": {
                    "relation_id": "calculate_shortage",
                    "speaker_label": "전체 값에서 예산을 빼 모자라는 돈을 구한다",
                    "operation": "subtraction",
                    "input_fact_ids": ["purchase_total", "budget"],
                    "output_fact_id": "shortage",
                    "method_policy": "open_equivalent",
                },
            },
        ],
        claimable_graph=ClaimableGraphContextV2(
            fact_ids=["budget", "purchase_total", "shortage"],
            relation_ids=["sum_item_costs", "calculate_shortage"],
        ),
        current_turn={
            "mormi_question": "다 사면 얼마고, 돈이 얼마나 모자라는지 알려줄래?",
            "asks": ["answer", "reason_or_method"],
            "expression_level": "L4",
            "hint_level": "H0",
        },
        child_utterance=evidence,
    )
    response = UnderstandingResponseV2(
        utterance_class="learning_response",
        support_need="none",
        contains_learning_evidence=True,
        answer_status="missing",
        reasoning_status="partial",
        confidence="high",
        claims=[
            FactUnderstandingClaimV2(
                claim_id="claim_purchase_total",
                fact_id="purchase_total",
                claim_type="intermediate_result",
                evidence_span=evidence,
                interpreted_value={
                    "type": "money",
                    "amount": 11_000,
                    "currency": "KRW",
                },
                verdict="correct",
            ),
            RelationUnderstandingClaimV2(
                claim_id="claim_sum_item_costs",
                relation_id="sum_item_costs",
                claim_type="procedure_step",
                evidence_span=evidence,
                verdict="correct",
                arithmetic_interpretation=ArithmeticInterpretationV2(
                    operation="addition",
                    operands=[5_000, 3_000, 3_000],
                    result=11_000,
                    mathematical_validity="correct",
                ),
            ),
        ],
    )

    assert request.claimable_graph.fact_ids == ["budget", "purchase_total", "shortage"]
    assert response.answer_status.value == "missing"
    assert response.reasoning_status.value == "partial"
    relation = response.claims[1]
    assert isinstance(relation, RelationUnderstandingClaimV2)
    assert relation.arithmetic_interpretation is not None
    assert relation.arithmetic_interpretation.operands == [5_000, 3_000, 3_000]


def test_v2_understanding_contract_rejects_invalid_or_ambiguous_structure() -> None:
    with pytest.raises(ValidationError, match="at least 2 items"):
        ArithmeticInterpretationV2(
            operation="addition",
            operands=[11_000],
            result=11_000,
            mathematical_validity="correct",
        )

    duplicate = FactUnderstandingClaimV2(
        claim_id="same",
        fact_id="purchase_total",
        claim_type="intermediate_result",
        evidence_span="11000원이야",
        interpreted_value={"type": "money", "amount": 11_000},
        verdict="correct",
    )
    with pytest.raises(ValidationError, match="claim_ids must be unique"):
        UnderstandingResponseV2(
            utterance_class="learning_response",
            contains_learning_evidence=True,
            answer_status="missing",
            reasoning_status="partial",
            claims=[duplicate, duplicate],
        )

    with pytest.raises(ValidationError, match="claims require"):
        UnderstandingResponseV2(
            utterance_class="learning_response",
            contains_learning_evidence=False,
            claims=[duplicate],
        )

    with pytest.raises(ValidationError, match="explicit support_need"):
        UnderstandingResponseV2(
            utterance_class="help_request",
            support_need="none",
        )

    task_question = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "task_question",
            "question_focus": "reason_or_method",
        }
    )
    assert task_question.conversation_move.value == "task_question"
    assert task_question.move_subject.value == "task"
    assert task_question.question_focus is not None
    assert task_question.question_focus.value == "reason_or_method"

    with pytest.raises(ValidationError, match="requires question_focus"):
        UnderstandingResponseV2.model_validate(
            {"utterance_class": "task_question"}
        )
    with pytest.raises(ValidationError, match="only valid"):
        UnderstandingResponseV2.model_validate(
            {
                "utterance_class": "learning_response",
                "question_focus": "meaning",
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        UnderstandingResponseV2.model_validate(
            {
                "utterance_class": "learning_response",
                "unexpected_legacy_field": "ignored-before-v2",
            }
        )


def test_v2_understanding_backfills_additive_axes_for_legacy_json() -> None:
    legacy_meta = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "non_learning_kind": "meta",
        }
    )
    legacy_refusal = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "non_learning_kind": "refusal",
        }
    )
    legacy_play = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "non_learning_kind": "playful",
        }
    )

    assert legacy_meta.conversation_move.value == "meta_question"
    assert legacy_meta.move_subject.value == "mormi_knowledge"
    assert legacy_refusal.conversation_move.value == "refusal"
    assert legacy_refusal.move_subject.value == "participation"
    assert legacy_play.conversation_move.value == "safe_play"
    assert legacy_play.move_subject.value == "other"


def test_v2_safe_social_move_can_coexist_with_literal_learning_claim() -> None:
    mixed = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "conversation_move": "meta_question",
            "move_subject": "mormi_ai_identity",
            "contains_learning_evidence": True,
            "answer_status": "complete",
            "claims": [
                {
                    "claim_kind": "fact",
                    "claim_id": "mixed_answer",
                    "fact_id": "total_price",
                    "claim_type": "final_answer",
                    "evidence_span": "16,000원이잖아",
                    "interpreted_value": {"type": "money", "amount": 16_000},
                    "verdict": "correct",
                }
            ],
        }
    )

    assert mixed.conversation_move.value == "meta_question"
    assert mixed.move_subject.value == "mormi_ai_identity"
    assert mixed.claims[0].evidence_span == "16,000원이잖아"


def test_v2_request_mormi_answer_keeps_support_separate_from_claims() -> None:
    request = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "help_request",
            "conversation_move": "request_mormi_answer",
            "support_need": "general_help",
        }
    )

    assert request.conversation_move.value == "request_mormi_answer"
    assert request.move_subject.value == "participation"
    assert request.claims == []


def _provider_understanding_response(
    **overrides: object,
) -> ModelUnderstandingResponseV2:
    payload: dict[str, object] = {
        "utterance_class": "learning_response",
        "conversation_move": "none",
        "move_subject": "other",
        "question_focus": None,
        "support_need": "none",
        "non_learning_kind": None,
        "contains_learning_evidence": False,
        "answer_status": "not_applicable",
        "reasoning_status": "not_applicable",
        "fact_claims": [],
        "relation_claims": [],
        "auxiliary_claims": [],
        "confidence": "high",
    }
    payload.update(overrides)
    return ModelUnderstandingResponseV2.model_validate(payload)


def _provider_fact_claim() -> dict[str, object]:
    return {
        "claim_id": "claim_total",
        "target_id": "total_price",
        "claim_type": "final_answer",
        "evidence_span": "16000원",
        "verdict": "correct",
        "value_type": "money",
        "numeric_value": 16_000,
        "text_value": None,
        "boolean_value": None,
        "unit": "KRW",
        "confidence": 0.99,
    }


def test_provider_canonicalization_gives_safety_precedence() -> None:
    response = _provider_understanding_response(
        utterance_class="safety_risk",
        conversation_move="meta_question",
        move_subject="mormi_ai_identity",
        support_need="concept",
        contains_learning_evidence=True,
        answer_status="complete",
        reasoning_status="sufficient",
        # Even an incomplete provider claim must be ignored on the safety path
        # rather than converted into trusted learning evidence.
        fact_claims=[
            {
                **_provider_fact_claim(),
                "value_type": None,
                "numeric_value": None,
            }
        ],
    )

    canonical = _internal_understanding_response(response)

    assert canonical.utterance_class.value == "safety_risk"
    assert canonical.conversation_move.value == "none"
    assert canonical.move_subject.value == "other"
    assert canonical.support_need.value == "none"
    assert canonical.answer_status.value == "not_applicable"
    assert canonical.reasoning_status.value == "not_applicable"
    assert canonical.contains_learning_evidence is False
    assert canonical.claims == []


def test_provider_diagnostic_formatting_cannot_discard_correct_verdicts() -> None:
    response = _provider_understanding_response(
        contains_learning_evidence=True,
        answer_status="complete",
        reasoning_status="sufficient",
        fact_claims=[
            {
                **_provider_fact_claim(),
                "unit": "대한민국 원",
            }
        ],
        relation_claims=[
            {
                "claim_id": "claim_division",
                "target_id": "divide_equally",
                "claim_type": "procedure_step",
                "evidence_span": "6000나누기 2는 3000이니까",
                "verdict": "sufficient",
                "operation": "division",
                # Incomplete diagnostic arithmetic must not invalidate the
                # classifier's semantic verdict or literal evidence claim.
                "operands": [6_000],
                "result": 3_000,
                "mathematical_validity": "correct",
                "confidence": 0.99,
            }
        ],
    )

    canonical = _internal_understanding_response(response)

    fact = canonical.claims[0]
    relation = canonical.claims[1]
    assert isinstance(fact, FactUnderstandingClaimV2)
    assert fact.verdict == "correct"
    assert fact.interpreted_value is None
    assert isinstance(relation, RelationUnderstandingClaimV2)
    assert relation.verdict == "sufficient"
    assert relation.arithmetic_interpretation is None


def test_provider_canonicalization_resolves_legacy_and_additive_axis_conflicts() -> None:
    task_question = _internal_understanding_response(
        _provider_understanding_response(
            utterance_class="non_learning_safe",
            conversation_move="task_question",
            move_subject="mormi_ai_identity",
            question_focus=None,
            non_learning_kind="meta",
        )
    )
    meta_question = _internal_understanding_response(
        _provider_understanding_response(
            utterance_class="task_question",
            conversation_move="meta_question",
            move_subject="mormi_ai_identity",
            question_focus="meaning",
        )
    )
    answer_request = _internal_understanding_response(
        _provider_understanding_response(
            utterance_class="learning_response",
            conversation_move="request_mormi_answer",
            move_subject="task",
            support_need="none",
        )
    )

    assert task_question.utterance_class.value == "task_question"
    assert task_question.move_subject.value == "task"
    assert task_question.question_focus is not None
    assert task_question.question_focus.value == "reason_or_method"
    assert task_question.non_learning_kind is None

    assert meta_question.utterance_class.value == "non_learning_safe"
    assert meta_question.conversation_move.value == "meta_question"
    assert meta_question.move_subject.value == "mormi_ai_identity"
    assert meta_question.question_focus is None
    assert meta_question.non_learning_kind is not None
    assert meta_question.non_learning_kind.value == "meta"

    assert answer_request.utterance_class.value == "help_request"
    assert answer_request.move_subject.value == "participation"
    assert answer_request.support_need.value == "general_help"


def test_provider_canonicalization_preserves_ui_reference_as_independent_axis() -> None:
    response = _internal_understanding_response(
        _provider_understanding_response(
            utterance_class="task_question",
            conversation_move="task_question",
            move_subject="task",
            question_focus="reason_or_method",
            support_need="concept",
            ui_reference={
                "element_id": "help_card.h1",
                "interaction": "asks_why",
                "reference_basis": "deictic",
                "evidence_span": "저걸",
            },
        )
    )

    assert response.ui_reference is not None
    assert response.ui_reference.element_id == "help_card.h1"
    assert response.ui_reference.interaction.value == "asks_why"
    assert response.claims == []
    assert response.contains_learning_evidence is False


def test_provider_canonicalization_derives_learning_flag_from_actual_claims() -> None:
    claimed = _internal_understanding_response(
        _provider_understanding_response(
            utterance_class="non_learning_safe",
            conversation_move="meta_question",
            move_subject="mormi_ai_identity",
            non_learning_kind="meta",
            contains_learning_evidence=False,
            fact_claims=[_provider_fact_claim()],
        )
    )
    empty = _internal_understanding_response(
        _provider_understanding_response(
            utterance_class="learning_response",
            conversation_move="meta_question",
            move_subject="mormi_ai_identity",
            non_learning_kind="meta",
            contains_learning_evidence=True,
        )
    )

    assert claimed.contains_learning_evidence is True
    assert claimed.utterance_class.value == "learning_response"
    assert len(claimed.claims) == 1
    assert empty.contains_learning_evidence is False
    assert empty.utterance_class.value == "non_learning_safe"
    assert empty.claims == []


def test_v2_understanding_request_rejects_invalid_graph_targets() -> None:
    base = {
        "task_id": "home_teaching",
        "visible_facts": {"budget": 10_000},
        "fact_contexts": [
            {
                "fact_id": "budget",
                "speaker_label": "예산",
                "semantic_aliases": [],
                "visible": True,
            },
            {
                "fact_id": "shortage",
                "speaker_label": "모자라는 돈",
                "semantic_aliases": [],
                "visible": False,
            },
        ],
        "targets": [
            {
                "target_kind": "fact",
                "target_id": "shortage",
                "ask_kind": "answer",
                "expected_truth": {"type": "money", "amount": 1_000},
            }
        ],
        "claimable_graph": {
            "fact_ids": ["budget", "shortage"],
            "relation_ids": [],
        },
        "current_turn": {
            "mormi_question": "얼마가 모자라는지 알려줄래?",
            "asks": ["answer"],
            "expression_level": "L4",
            "hint_level": "H0",
        },
        "child_utterance": "1000원이 모자라",
    }

    assert UnderstandingRequestV2.model_validate(base).targets[0].target_id == "shortage"

    scaffold_without_relation_target = {
        **base,
        "current_turn": {
            **base["current_turn"],
            "hint_level": "H2",
            "help_scaffolded_relation_ids": ["calculate_shortage"],
        },
        "claimable_graph": {
            "fact_ids": ["budget", "shortage"],
            "relation_ids": ["calculate_shortage"],
        },
    }
    with pytest.raises(ValidationError, match="unresolved relation targets"):
        UnderstandingRequestV2.model_validate(scaffold_without_relation_target)

    h0_scaffold = {
        **base,
        "current_turn": {
            **base["current_turn"],
            "help_scaffolded_relation_ids": ["calculate_shortage"],
        },
        "targets": [
            {
                "target_kind": "relation",
                "target_id": "calculate_shortage",
                "ask_kind": "reason_or_method",
                "semantic_contract": {
                    "relation_id": "calculate_shortage",
                    "speaker_label": "예산에서 모자라는 돈을 구한다",
                    "operation": "subtraction",
                    "input_fact_ids": ["budget"],
                    "output_fact_id": "shortage",
                    "method_policy": "open_equivalent",
                },
            }
        ],
        "claimable_graph": {
            "fact_ids": ["budget", "shortage"],
            "relation_ids": ["calculate_shortage"],
        },
    }
    h0_scaffold["current_turn"]["asks"] = ["reason_or_method"]
    with pytest.raises(ValidationError, match="H2 or H3"):
        UnderstandingRequestV2.model_validate(h0_scaffold)

    invalid_fact = {
        **base,
        "visible_facts": {},
        "fact_contexts": [],
        "claimable_graph": {"fact_ids": [], "relation_ids": []},
    }
    with pytest.raises(ValidationError, match="fact target must exist"):
        UnderstandingRequestV2.model_validate(invalid_fact)

    relation_with_truth = {
        **base,
        "targets": [
            {
                "target_kind": "relation",
                "target_id": "calculate_shortage",
                "ask_kind": "reason_or_method",
                "expected_truth": {"type": "text", "text": "전체 금액에서 예산을 뺀다"},
                "semantic_contract": {
                    "relation_id": "calculate_shortage",
                    "speaker_label": "예산에서 모자라는 돈을 구한다",
                    "operation": "subtraction",
                    "input_fact_ids": ["budget"],
                    "output_fact_id": "shortage",
                    "method_policy": "open_equivalent",
                },
            }
        ],
        "claimable_graph": {
            "fact_ids": ["budget", "shortage"],
            "relation_ids": ["calculate_shortage"],
        },
        "current_turn": {
            **base["current_turn"],
            "asks": ["reason_or_method"],
        },
    }
    with pytest.raises(ValidationError, match="cannot declare expected_truth"):
        UnderstandingRequestV2.model_validate(relation_with_truth)


def test_v2_auxiliary_claim_does_not_forge_a_canonical_graph_id() -> None:
    claim = AuxiliaryUnderstandingClaimV2(
        claim_id="claim_auxiliary",
        evidence_span="가격을 먼저 하나씩 적어봤어",
        summary="각 물건 가격을 별도로 정리했다",
        verdict="correct",
    )

    assert claim.claim_kind == "auxiliary"
    payload = claim.model_dump(mode="json")
    assert "fact_id" not in payload
    assert "relation_id" not in payload


def test_v2_understanding_provider_output_keeps_a_compact_strict_schema() -> None:
    schema = structured_output_schema(ModelUnderstandingResponseV2)
    encoded = json.dumps(schema, ensure_ascii=False)

    assert len(encoded) < 8_000
    assert '"ModelFactUnderstandingClaimV2"' in encoded
    assert '"ModelRelationUnderstandingClaimV2"' in encoded
    assert '"ModelAuxiliaryUnderstandingClaimV2"' in encoded
    assert '"numeric_value"' in encoded
    assert '"conversation_move"' in encoded
    assert '"move_subject"' in encoded
    assert "conversation_move" in schema["required"]
    assert "move_subject" in schema["required"]
    assert '"FactUnderstandingClaimV2"' not in encoded
    assert '"additionalProperties": false' in encoded


def test_v2_understanding_provider_schema_cannot_mix_fact_and_relation_roles() -> None:
    base = {
        "utterance_class": "learning_response",
        "support_need": "none",
        "non_learning_kind": None,
        "contains_learning_evidence": True,
        "answer_status": "missing",
        "reasoning_status": "partial",
        "fact_claims": [],
        "relation_claims": [
            {
                "claim_id": "claim_sum",
                "target_id": "sum_item_costs",
                "claim_type": "intermediate_result",
                "evidence_span": "다 합하면 11000원이잖아",
                "verdict": "correct",
                "operation": "addition",
                "operands": [5000, 3000, 3000],
                "result": 11000,
                "mathematical_validity": "correct",
                "confidence": 0.98,
            }
        ],
        "auxiliary_claims": [],
        "confidence": "high",
    }

    with pytest.raises(ValidationError, match="procedure_step|explanation"):
        ModelUnderstandingResponseV2.model_validate(base)


@pytest.mark.asyncio
async def test_v2_understanding_converts_compact_provider_claim_to_internal_contract() -> None:
    class FakeMessages:
        async def create(self, **_: object) -> object:
            payload = {
                "utterance_class": "learning_response",
                "conversation_move": "meta_question",
                "move_subject": "mormi_ai_identity",
                "question_focus": None,
                "support_need": "none",
                "non_learning_kind": "meta",
                "contains_learning_evidence": True,
                "answer_status": "complete",
                "reasoning_status": "not_applicable",
                "fact_claims": [
                    {
                        "claim_id": "claim_1",
                        "target_id": "total_price",
                        "claim_type": "final_answer",
                        "evidence_span": "16000원",
                        "verdict": "correct",
                        "value_type": "money",
                        "numeric_value": 16000,
                        "text_value": None,
                        "boolean_value": None,
                        "unit": "KRW",
                        "confidence": 0.99,
                    }
                ],
                "relation_claims": [],
                "auxiliary_claims": [],
                "confidence": "high",
            }
            return type(
                "Message",
                (),
                {
                    "stop_reason": "end_turn",
                    "content": [type("Text", (), {"type": "text", "text": json.dumps(payload)})()],
                },
            )()

    gateway = ClaudeGateway(Settings(anthropic_api_key=None))
    gateway.client = type("Client", (), {"messages": FakeMessages()})()  # type: ignore[assignment]
    request = UnderstandingRequestV2.model_validate(
        {
            "task_id": "home_teaching",
            "visible_facts": {"unit_price": 4000, "quantity": 4},
            "fact_contexts": [
                {
                    "fact_id": "unit_price",
                    "speaker_label": "한 개 값",
                    "semantic_aliases": [],
                    "visible": True,
                },
                {
                    "fact_id": "quantity",
                    "speaker_label": "개수",
                    "semantic_aliases": [],
                    "visible": True,
                },
                {
                    "fact_id": "total_price",
                    "speaker_label": "전체 값",
                    "semantic_aliases": [],
                    "visible": False,
                },
            ],
            "targets": [
                {
                    "target_kind": "fact",
                    "target_id": "total_price",
                    "ask_kind": "answer",
                    "rubric": {},
                    "expected_truth": {"type": "money", "amount": 16000},
                }
            ],
            "claimable_graph": {
                "fact_ids": ["unit_price", "quantity", "total_price"],
                "relation_ids": [],
                "open_auxiliary_claims": True,
            },
            "current_turn": {
                "mormi_question": "전체 값이 얼마인지 알려줄 수 있어?",
                "asks": ["answer"],
                "expression_level": "L4",
                "hint_level": "H0",
            },
            "recent_history": [],
            "child_utterance": "너 AI인데 16000원이잖아",
            "guard_feedback_codes": [],
        }
    )

    result = await gateway.understand_v2(request)

    assert result.conversation_move.value == "meta_question"
    assert result.move_subject.value == "mormi_ai_identity"
    claim = result.claims[0]
    assert isinstance(claim, FactUnderstandingClaimV2)
    assert claim.verdict == "correct"
    assert claim.interpreted_value.model_dump(mode="json") == {
        "type": "money",
        "amount": 16000.0,
        "currency": "KRW",
    }


def test_v2_runtime_selector_is_not_part_of_the_public_create_or_turn_contract() -> None:
    schema = app.openapi()["components"]["schemas"]

    assert "runtime_contract_version" not in schema["SessionCreate"]["properties"]
    assert "runtime_contract_version" not in schema["TurnContract"]["properties"]


@pytest.mark.asyncio
async def test_new_conversation_pins_legacy_runtime_across_turns_and_retries(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/runtime-version.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    engine = ConversationEngine(FakeGateway())  # type: ignore[arg-type]
    service = ConversationService(repository, engine)
    request = SessionCreate(
        learner_id=1,
        scene="home_teach",
        scenario_id="home_teach",
        learning_session_id="learning-runtime-version",
        conversation_round=1,
        practice_result_id="practice-runtime-version",
        practice_summary={
            "curriculum_session_id": "add-pictures",
            "skill_id": "basic_addition",
            "question_count": 5,
            "first_try_correct_count": 4,
            "wrong_attempt_count": 1,
        },
    )

    started = await service.create_conversation(request)
    initial_state = await repository.get_state(started.conversation_id)
    assert (
        initial_state.runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )

    advanced = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="a48257f0-a54f-44c1-a520-18931f98a9df",
            type="no_response",
        ),
    )
    advanced_state = await repository.get_state(started.conversation_id)
    assert advanced.turn.state_version == 2
    assert (
        advanced_state.runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )

    retried = await service.create_conversation(request)
    assert retried.conversation_id == started.conversation_id
    assert (
        (await repository.get_state(retried.conversation_id)).runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )

    next_round = await service.create_conversation(
        request.model_copy(update={"conversation_round": 2})
    )
    assert next_round.conversation_id != started.conversation_id
    assert (
        (await repository.get_state(next_round.conversation_id)).runtime_contract_version
        is DialogueRuntimeContractVersion.LEGACY_V1
    )

    async with database.sessions() as session:
        observation = (
            await session.execute(
                select(DialogueTurnObservationRecord).where(
                    DialogueTurnObservationRecord.conversation_id
                    == started.conversation_id
                )
            )
        ).scalar_one()
    assert observation.versions_json["runtime_contract"] == "legacy-v1"

    v2_unavailable = ConversationService(
        repository,
        engine,
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
    )
    assert v2_unavailable.dialogue_runtime_capabilities == (
        DialogueRuntimeContractVersion.LEGACY_V1,
    )

    await database.dispose()
