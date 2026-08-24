from __future__ import annotations

from collections.abc import Iterator

import pytest

from mormi_api.schemas import (
    NoteContextualizationContext,
    NoteContextualizationOutput,
    SpeakerContext,
    SpeakerOutput,
    UtteranceAnalysis,
)


class FakeGateway:
    def __init__(
        self,
        analyses: list[UtteranceAnalysis] | None = None,
        adjudications: list[UtteranceAnalysis] | None = None,
        bridge_outputs: list[SpeakerOutput] | None = None,
    ) -> None:
        self.analyses = list(analyses or [])
        self.adjudications = list(adjudications or [])
        self.bridge_outputs = list(bridge_outputs or [])
        self.classify_calls = 0
        self.adjudicate_calls = 0
        self.bridge_speak_calls = 0
        self.speaker_contexts: list[SpeakerContext] = []
        self.bridge_contexts: list[SpeakerContext] = []
        self.note_contexts: list[NoteContextualizationContext] = []

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
        self.speaker_contexts.append(context)
        return self._fallback_speaker_output(context)

    @staticmethod
    def _fallback_speaker_output(context: SpeakerContext) -> SpeakerOutput:
        return SpeakerOutput(
            text=context.fallback_text,
            dialogue_act=context.dialogue_act,
            asked_slot_ids=context.required_slot_ids,
        )

    async def bridge_speak(self, context: SpeakerContext) -> SpeakerOutput:
        self.bridge_speak_calls += 1
        self.bridge_contexts.append(context)
        if self.bridge_outputs:
            output = self.bridge_outputs.pop(0)
            return output.model_copy(
                update={
                    "dialogue_act": context.dialogue_act,
                    "asked_slot_ids": list(context.required_slot_ids),
                }
            )
        return self._fallback_speaker_output(context)

    async def contextualize_note(
        self,
        context: NoteContextualizationContext,
    ) -> NoteContextualizationOutput:
        self.note_contexts.append(context)
        return NoteContextualizationOutput(
            text=context.fallback_text,
            source_slots_used=list(context.source_fragments),
            source_spans_used=list(context.source_fragments.values()),
            fact_refs_used=[],
            meaning_preserved=True,
            self_contained=True,
            introduced_math_content=False,
        )

@pytest.fixture
def fake_gateway() -> Iterator[FakeGateway]:
    yield FakeGateway()
