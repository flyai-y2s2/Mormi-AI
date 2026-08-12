from __future__ import annotations

from collections.abc import Iterator

import pytest

from mormi_api.schemas import SpeakerContext, SpeakerOutput, UtteranceAnalysis


class FakeGateway:
    def __init__(self, analyses: list[UtteranceAnalysis] | None = None) -> None:
        self.analyses = list(analyses or [])

    async def classify(self, **_: object) -> UtteranceAnalysis:
        if not self.analyses:
            raise AssertionError("No fake classification was prepared")
        return self.analyses.pop(0)

    async def speak(self, context: SpeakerContext) -> SpeakerOutput:
        return SpeakerOutput(text=context.fallback_text)


@pytest.fixture
def fake_gateway() -> Iterator[FakeGateway]:
    yield FakeGateway()
