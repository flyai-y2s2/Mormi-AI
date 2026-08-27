from __future__ import annotations

from mormi_api.dialogue_v2_model_smoke import (
    build_speaker_smoke_plan_v2,
    build_understanding_smoke_request_v2,
    run_dialogue_v2_model_smoke,
    safe_model_smoke_error_code,
)
from mormi_api.dialogue_v2_speaker import SpeakerOutputV2, SpeakerPlanV2
from mormi_api.llm import ModelOutputError, ModelUnavailableError
from mormi_api.schemas import UnderstandingRequestV2, UnderstandingResponseV2


class _SmokeGateway:
    def __init__(self, *, speaker_text: str = "나는 네가 알려주지 않은 건 잘 몰라...") -> None:
        self.speaker_text = speaker_text
        self.understanding_requests: list[UnderstandingRequestV2] = []
        self.speaker_plans: list[SpeakerPlanV2] = []

    async def understand_v2(
        self,
        request: UnderstandingRequestV2,
    ) -> UnderstandingResponseV2:
        self.understanding_requests.append(request)
        return UnderstandingResponseV2.model_validate(
            {
                "utterance_class": "help_request",
                "conversation_move": "none",
                "move_subject": "other",
                "support_need": "general_help",
                "contains_learning_evidence": False,
                "answer_status": "not_applicable",
                "reasoning_status": "not_applicable",
                "claims": [],
                "confidence": "high",
            }
        )

    async def speak_v2(self, plan: SpeakerPlanV2) -> SpeakerOutputV2:
        self.speaker_plans.append(plan)
        return SpeakerOutputV2(text=self.speaker_text, mood="curious")


def test_model_smoke_contract_contains_no_child_or_hidden_truth_in_speaker_plan() -> None:
    request = build_understanding_smoke_request_v2()
    plan = build_speaker_smoke_plan_v2()
    plan_json = plan.model_dump_json()

    assert request.child_utterance == "잘 모르겠어"
    assert "16000" not in plan_json
    assert "child_utterance" not in plan_json
    assert plan.response_plan is not None
    assert plan.response_plan.reask_mode == "remaining_targets"


async def test_model_smoke_exercises_both_provider_roles_without_returning_text() -> None:
    gateway = _SmokeGateway()

    report = await run_dialogue_v2_model_smoke(
        gateway,
        classifier_model="claude-sonnet-4-6",
        speaker_model="claude-haiku-4-5-20251001",
    )

    assert report.succeeded is True
    assert report.understanding_contract == "ok"
    assert report.speaker_contract == "ok"
    assert len(gateway.understanding_requests) == 1
    assert len(gateway.speaker_plans) == 1
    assert "text" not in report.model_dump(mode="json")


async def test_model_smoke_rejects_a_speaker_that_guesses_hidden_truth() -> None:
    gateway = _SmokeGateway(speaker_text="전체 값은 16,000원이야.")

    try:
        await run_dialogue_v2_model_smoke(
            gateway,
            classifier_model="claude-sonnet-4-6",
            speaker_model="claude-haiku-4-5-20251001",
        )
    except RuntimeError as error:
        assert str(error) == "speaker_v2_smoke_output_invalid"
    else:  # pragma: no cover - protects the deployment safety contract
        raise AssertionError("hidden truth must fail the provider smoke")


def test_model_smoke_failure_diagnostics_are_bounded_codes() -> None:
    assert (
        safe_model_smoke_error_code(ModelUnavailableError("model_connection_failed"))
        == "model_connection_failed"
    )
    assert (
        safe_model_smoke_error_code(ModelUnavailableError("secret provider body"))
        == "model_unavailable"
    )
    assert safe_model_smoke_error_code(ModelOutputError("provider output")) == (
        "model_output_invalid"
    )
    assert safe_model_smoke_error_code(TimeoutError()) == "model_smoke_timeout"
