from __future__ import annotations

from collections.abc import Iterator

import pytest

from mormi_api.schemas import (
    NoteContextualizationContext,
    NoteContextualizationOutput,
    SpeakerContext,
    SpeakerGuardContract,
    SpeakerOutput,
    SpeakerVerification,
    UtteranceAnalysis,
)


class FakeGateway:
    def __init__(
        self,
        analyses: list[UtteranceAnalysis] | None = None,
        adjudications: list[UtteranceAnalysis] | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.adjudications = list(adjudications or [])
        self.classify_calls = 0
        self.adjudicate_calls = 0
        self.bridge_speak_calls = 0

    async def classify(self, **_: object) -> UtteranceAnalysis:
        self.classify_calls += 1
        if not self.analyses:
            raise AssertionError("No fake classification was prepared")
        return self.analyses.pop(0)

    async def adjudicate(self, **kwargs: object) -> UtteranceAnalysis:
        self.adjudicate_calls += 1
        if self.adjudications:
            return self.adjudications.pop(0)
        return kwargs["primary_analysis"]  # type: ignore[return-value]

    async def speak(self, context: SpeakerContext) -> SpeakerOutput:
        return SpeakerOutput(
            text=context.fallback_text,
            dialogue_act=context.dialogue_act,
            asked_slot_ids=context.required_slot_ids,
        )

    async def bridge_speak(self, context: SpeakerContext) -> SpeakerOutput:
        self.bridge_speak_calls += 1
        return await self.speak(context)

    async def contextualize_note(
        self,
        context: NoteContextualizationContext,
    ) -> NoteContextualizationOutput:
        return NoteContextualizationOutput(
            text=context.fallback_text,
            source_slots_used=list(context.source_fragments),
            source_spans_used=list(context.source_fragments.values()),
            fact_refs_used=[],
            meaning_preserved=True,
            self_contained=True,
            introduced_math_content=False,
        )

    async def verify_speaker(
        self,
        context: SpeakerContext,
        guard: SpeakerGuardContract,
        output: SpeakerOutput,
    ) -> SpeakerVerification:
        del guard
        return SpeakerVerification(
            approved=True,
            dialogue_act_preserved=True,
            required_focus_preserved=True,
            only_allowed_math_used=True,
            child_not_evaluated=True,
            character_consistent=True,
            meaningfully_reframed=True,
            interaction_intent_acknowledged=True,
            task_returned_without_reward=True,
            arithmetic_claim_stance_safe=True,
            help_card_state_respected=True,
            sentence_complete=True,
            joint_mode_respected=True,
            violation_codes=[],
            detected_dialogue_act=context.dialogue_act,
            detected_asked_slot_ids=context.required_slot_ids,
            question_evidence_span=context.required_question or "",
            child_expression_spans=output.used_child_expression_spans,
            reason_code="approved",
        )


@pytest.fixture
def fake_gateway() -> Iterator[FakeGateway]:
    yield FakeGateway()
