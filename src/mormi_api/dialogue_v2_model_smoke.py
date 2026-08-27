from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from .dialogue_v2_speaker import (
    SpeakerOutputV2,
    SpeakerPlanV2,
    validate_speaker_output_v2,
)
from .llm import ModelOutputError, ModelUnavailableError
from .schemas import MoneyValueV2, UnderstandingRequestV2, UnderstandingResponseV2

_SAFE_PROVIDER_ERROR_CODES = frozenset(
    {
        "model_connection_failed",
        "structured_schema_not_strict",
        "structured_schema_too_complex",
        "structured_schema_union_limit",
        "structured_schema_optional_limit",
        "structured_schema_invalid",
        "model_effort_invalid",
        "model_temperature_invalid",
        "model_max_tokens_invalid",
        "model_request_too_large",
        "model_bad_request",
        "model_auth_failed",
        "model_forbidden",
        "model_not_found",
        "model_rate_limited",
        "model_provider_unavailable",
    }
)

_MODEL_SMOKE_STAGES = frozenset({"understanding", "speaker"})


class DialogueV2ModelSmokeStageError(RuntimeError):
    """Bound one provider failure to its PII-free deployment-smoke stage."""

    def __init__(self, stage: str, error_code: str) -> None:
        if stage not in _MODEL_SMOKE_STAGES:
            raise ValueError("unknown dialogue V2 model smoke stage")
        self.stage = stage
        self.error_code = error_code
        super().__init__(f"{stage}_{error_code}")


def safe_model_smoke_error_code(error: Exception) -> str:
    """Return a bounded diagnostic code without provider bodies or model text."""

    if isinstance(error, DialogueV2ModelSmokeStageError):
        return f"{error.stage}_{error.error_code}"
    if isinstance(error, ModelUnavailableError):
        code = str(error)
        if code in _SAFE_PROVIDER_ERROR_CODES:
            return code
        if code == "ANTHROPIC_API_KEY is not configured":
            return "model_not_configured"
        return "model_unavailable"
    if isinstance(error, ModelOutputError):
        return "model_output_invalid"
    if isinstance(error, TimeoutError):
        return "model_smoke_timeout"
    if isinstance(error, RuntimeError) and str(error) == "speaker_v2_smoke_output_invalid":
        return "speaker_output_guard_failed"
    return "model_smoke_failed"


class DialogueV2ModelSmokeGateway(Protocol):
    async def understand_v2(
        self,
        request: UnderstandingRequestV2,
    ) -> UnderstandingResponseV2: ...

    async def speak_v2(self, plan: SpeakerPlanV2) -> SpeakerOutputV2: ...


class DialogueV2ModelSmokeReport(BaseModel):
    succeeded: bool = True
    understanding_contract: str = "ok"
    speaker_contract: str = "ok"
    classifier_model: str
    speaker_model: str


def build_understanding_smoke_request_v2() -> UnderstandingRequestV2:
    """Build a PII-free request that exercises the full strict classifier schema."""

    return UnderstandingRequestV2.model_validate(
        {
            "task_id": "deployment_smoke",
            "visible_facts": {
                "unit_price": 4_000,
                "quantity": 4,
            },
            "targets": [
                {
                    "target_kind": "fact",
                    "target_id": "answer_total",
                    "ask_kind": "answer",
                    "rubric": {"correct": "전체 값을 말한다"},
                    "expected_truth": {"type": "money", "amount": 16_000},
                },
                {
                    "target_kind": "relation",
                    "target_id": "calculate_total",
                    "ask_kind": "reason_or_method",
                    "rubric": {"sufficient": "한 개 값과 개수를 이용한다"},
                },
            ],
            "claimable_graph": {
                "fact_ids": ["unit_price", "quantity", "answer_total"],
                "relation_ids": ["calculate_total"],
                "open_auxiliary_claims": True,
            },
            "current_turn": {
                "mormi_question": "전체 값과 구하는 방법을 알려줄 수 있어?",
                "asks": ["answer", "reason_or_method"],
                "expression_level": "L4",
                "hint_level": "H0",
            },
            "child_utterance": "잘 모르겠어",
        }
    )


def build_speaker_smoke_plan_v2() -> SpeakerPlanV2:
    """Build a hidden-truth-free Haiku plan and keep server reasking out of the model."""

    target_focus = [
        {
            "target_kind": "fact",
            "target_id": "answer_total",
            "speaker_label": "전체 값",
        },
        {
            "target_kind": "relation",
            "target_id": "calculate_total",
            "speaker_label": "전체 값을 구하는 방법",
        },
    ]
    return SpeakerPlanV2.model_validate(
        {
            "dialogue_act": "respond_to_help_then_reask",
            "response_signal": {
                "kind": "help_request",
                "repeat_count": 0,
            },
            "target": {
                "fact_ids": ["answer_total"],
                "relation_ids": ["calculate_total"],
                "ask_mode": "answer_and_method",
                "success_criteria_ids": ["answer_and_method"],
            },
            "target_focus": target_focus,
            "response_plan": {
                "response_mode": "explain_mormi_limit",
                "reask_mode": "remaining_targets",
                "reask_targets": target_focus,
                "card_visible": False,
                "card_event": "none",
                "hint_level": "H0",
            },
            "support": {
                "expression_level": "L4",
                "hint_level": "H0",
                "support_need": "general_help",
                "question_style_guide": "모르미의 한계를 짧게 말하고 부탁한다",
                "help_card_visible": False,
            },
            "current_question": "전체 값과 구하는 방법을 알려줄 수 있어?",
            "fallback_copy_ref": "deployment-smoke.speaker",
        }
    )


async def run_dialogue_v2_model_smoke(
    gateway: DialogueV2ModelSmokeGateway,
    *,
    classifier_model: str,
    speaker_model: str,
) -> DialogueV2ModelSmokeReport:
    """Exercise Sonnet understanding and Haiku speaking without logging model text."""

    try:
        await gateway.understand_v2(build_understanding_smoke_request_v2())
    except Exception as error:
        raise DialogueV2ModelSmokeStageError(
            "understanding",
            safe_model_smoke_error_code(error),
        ) from error
    plan = build_speaker_smoke_plan_v2()
    try:
        output = await gateway.speak_v2(plan)
    except Exception as error:
        raise DialogueV2ModelSmokeStageError(
            "speaker",
            safe_model_smoke_error_code(error),
        ) from error
    validated = validate_speaker_output_v2(
        output,
        plan,
        forbidden_values=[MoneyValueV2(amount=16_000)],
    )
    if validated is None:
        raise RuntimeError("speaker_v2_smoke_output_invalid")
    return DialogueV2ModelSmokeReport(
        classifier_model=classifier_model,
        speaker_model=speaker_model,
    )
