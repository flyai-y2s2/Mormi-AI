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
    def __init__(self, analyses: list[UtteranceAnalysis] | None = None) -> None:
        self.analyses = list(analyses or [])

    async def classify(self, **_: object) -> UtteranceAnalysis:
        if not self.analyses:
            raise AssertionError("No fake classification was prepared")
        return self.analyses.pop(0)

    async def speak(self, context: SpeakerContext) -> SpeakerOutput:
        return SpeakerOutput(
            text=context.fallback_text,
            dialogue_act=context.dialogue_act,
            asked_slot_ids=context.required_slot_ids,
        )

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
            detected_dialogue_act=context.dialogue_act,
            detected_asked_slot_ids=context.required_slot_ids,
            question_evidence_span=context.required_question or "",
            child_expression_spans=output.used_child_expression_spans,
            reason_code="approved",
        )


@pytest.fixture
def fake_gateway() -> Iterator[FakeGateway]:
    yield FakeGateway()
