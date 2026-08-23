from __future__ import annotations

from pathlib import Path

import pytest

from mormi_api.ladder_analysis import (
    LadderAction,
    LadderEvidence,
    LevelPerformance,
    decide_ladder_adjustment,
)
from mormi_api.ladder_model.dataset import LadderLevel
from mormi_api.ladder_model.runtime import LadderModelRuntime, RuntimePrediction


def evidence(
    *,
    current: LadderLevel,
    correct: int,
    attempts: int,
    predictions: tuple[LadderLevel, ...] = (),
    valid_speech: int = 0,
    lower_rules: int = 0,
) -> LadderEvidence:
    return LadderEvidence(
        current_level=current,
        performance_by_level={current: LevelPerformance(correct=correct, attempts=attempts)},
        recent_predictions=predictions,
        valid_speech_count=valid_speech,
        lower_rule_evidence_count=lower_rules,
        completed_session_count=2,
    )


def test_upgrade_requires_mastery_attempts_and_two_recent_higher_predictions() -> None:
    decision = decide_ladder_adjustment(
        evidence(
            current=LadderLevel.L2,
            correct=9,
            attempts=10,
            predictions=(LadderLevel.L3, LadderLevel.L4),
            valid_speech=2,
        )
    )

    assert decision.action is LadderAction.UPGRADE
    assert decision.recommended_level is LadderLevel.L3
    assert decision.current_accuracy == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("correct", "attempts", "predictions", "valid_speech"),
    [
        (3, 4, (LadderLevel.L3, LadderLevel.L3), 2),
        (9, 10, (LadderLevel.L3,), 1),
        (9, 10, (LadderLevel.L2, LadderLevel.L3), 2),
    ],
)
def test_upgrade_stays_put_when_evidence_is_insufficient(
    correct: int,
    attempts: int,
    predictions: tuple[LadderLevel, ...],
    valid_speech: int,
) -> None:
    decision = decide_ladder_adjustment(
        evidence(
            current=LadderLevel.L2,
            correct=correct,
            attempts=attempts,
            predictions=predictions,
            valid_speech=valid_speech,
        )
    )

    assert decision.action in {LadderAction.MAINTAIN, LadderAction.INSUFFICIENT_EVIDENCE}
    assert decision.recommended_level is LadderLevel.L2


def test_two_consecutive_lower_predictions_adjust_down_one_step() -> None:
    decision = decide_ladder_adjustment(
        evidence(
            current=LadderLevel.L4,
            correct=8,
            attempts=10,
            predictions=(LadderLevel.L3, LadderLevel.L2),
            valid_speech=2,
        )
    )

    assert decision.action is LadderAction.ADJUST_DOWN
    assert decision.recommended_level is LadderLevel.L3


def test_low_accuracy_and_lower_evidence_adjust_down_one_step() -> None:
    decision = decide_ladder_adjustment(
        evidence(
            current=LadderLevel.L3,
            correct=2,
            attempts=4,
            predictions=(LadderLevel.L2,),
            valid_speech=1,
        )
    )

    assert decision.action is LadderAction.ADJUST_DOWN
    assert decision.recommended_level is LadderLevel.L2


def test_l2_can_only_recommend_l0_from_rule_evidence() -> None:
    without_rule = decide_ladder_adjustment(
        evidence(current=LadderLevel.L2, correct=1, attempts=4)
    )
    with_rule = decide_ladder_adjustment(
        evidence(current=LadderLevel.L2, correct=1, attempts=4, lower_rules=2)
    )

    assert without_rule.recommended_level is LadderLevel.L2
    assert with_rule.action is LadderAction.ADJUST_DOWN
    assert with_rule.recommended_level is LadderLevel.L0


def test_boundaries_never_move_past_l0_or_l4() -> None:
    top = decide_ladder_adjustment(
        evidence(
            current=LadderLevel.L4,
            correct=4,
            attempts=4,
            predictions=(LadderLevel.L4, LadderLevel.L4),
            valid_speech=2,
        )
    )
    bottom = decide_ladder_adjustment(
        evidence(current=LadderLevel.L0, correct=0, attempts=4, lower_rules=2)
    )

    assert top.recommended_level is LadderLevel.L4
    assert bottom.recommended_level is LadderLevel.L0


def test_runtime_reports_unavailable_without_loading_or_echoing_text(tmp_path: Path) -> None:
    runtime = LadderModelRuntime(tmp_path / "missing-model")

    result = runtime.predict(["민감한 아이 발화"])

    assert result.available is False
    assert result.error_code == "MODEL_NOT_FOUND"
    assert result.predictions == ()
    assert "민감한" not in repr(result)


def test_runtime_prediction_rejects_l0_from_speech_model() -> None:
    with pytest.raises(ValueError, match="L0"):
        RuntimePrediction(level=LadderLevel.L0, confidence=0.9)
