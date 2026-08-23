from __future__ import annotations

from mormi_api.ladder_model.dataset import LadderLevel, ResponseMode
from mormi_api.ladder_model.policy import rule_recommendation


def test_non_speech_responses_use_deterministic_ladder_policy() -> None:
    assert rule_recommendation(LadderLevel.L4, ResponseMode.NO_RESPONSE) is LadderLevel.L3
    assert rule_recommendation(LadderLevel.L3, ResponseMode.NO_RESPONSE) is LadderLevel.L2
    assert rule_recommendation(LadderLevel.L2, ResponseMode.NO_RESPONSE) is LadderLevel.L0
    assert rule_recommendation(LadderLevel.L0, ResponseMode.NO_RESPONSE) is LadderLevel.L0
    assert rule_recommendation(LadderLevel.L4, ResponseMode.CHOICE) is LadderLevel.L2
    assert rule_recommendation(LadderLevel.L3, ResponseMode.SOLVE_TOGETHER) is LadderLevel.L0


def test_text_responses_are_delegated_to_speech_model() -> None:
    assert rule_recommendation(LadderLevel.L4, ResponseMode.FREE_TEXT) is None
    assert rule_recommendation(LadderLevel.L3, ResponseMode.SHORT_ANSWER) is None

