from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .ladder_model.dataset import LadderLevel
from .ladder_model.policy import ladder_rank, one_step_higher, one_step_lower


class LadderAction(StrEnum):
    UPGRADE = "UPGRADE"
    MAINTAIN = "MAINTAIN"
    ADJUST_DOWN = "ADJUST_DOWN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class LevelPerformance:
    correct: int
    attempts: int

    def __post_init__(self) -> None:
        if self.attempts < 0 or self.correct < 0 or self.correct > self.attempts:
            raise ValueError("invalid level performance")

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.attempts if self.attempts else None


@dataclass(frozen=True)
class LadderEvidence:
    current_level: LadderLevel
    performance_by_level: dict[LadderLevel, LevelPerformance] = field(default_factory=dict)
    recent_predictions: tuple[LadderLevel, ...] = ()
    valid_speech_count: int = 0
    lower_rule_evidence_count: int = 0
    completed_session_count: int = 0

    def __post_init__(self) -> None:
        if self.valid_speech_count < 0 or self.lower_rule_evidence_count < 0:
            raise ValueError("evidence counts must not be negative")
        if self.completed_session_count < 0:
            raise ValueError("completed_session_count must not be negative")


@dataclass(frozen=True)
class LadderDecision:
    action: LadderAction
    current_level: LadderLevel
    recommended_level: LadderLevel
    current_accuracy: float | None
    evidence_count: int
    reason_code: str


def decide_ladder_adjustment(evidence: LadderEvidence) -> LadderDecision:
    current = evidence.current_level
    performance = evidence.performance_by_level.get(current, LevelPerformance(0, 0))
    accuracy = performance.accuracy
    predictions = evidence.recent_predictions
    current_rank = ladder_rank(current)
    lower_predictions = [level for level in predictions if ladder_rank(level) < current_rank]
    two_recent_lower = (
        len(predictions) >= 2
        and all(ladder_rank(level) < current_rank for level in predictions[-2:])
    )
    has_lower_evidence = bool(lower_predictions) or evidence.lower_rule_evidence_count > 0

    if current is not LadderLevel.L0:
        can_descend_to_l0 = (
            current is not LadderLevel.L2 or evidence.lower_rule_evidence_count > 0
        )
        low_accuracy_with_lower = (
            accuracy is not None
            and accuracy < 0.70
            and has_lower_evidence
            and can_descend_to_l0
        )
        consecutive_lower = two_recent_lower and can_descend_to_l0
        rule_descent = (
            current is LadderLevel.L2 and evidence.lower_rule_evidence_count >= 2
        )
        if consecutive_lower or low_accuracy_with_lower or rule_descent:
            return LadderDecision(
                action=LadderAction.ADJUST_DOWN,
                current_level=current,
                recommended_level=one_step_lower(current),
                current_accuracy=accuracy,
                evidence_count=performance.attempts,
                reason_code=(
                    "TWO_LOWER_PREDICTIONS"
                    if consecutive_lower
                    else "LOW_ACCURACY_WITH_LOWER_EVIDENCE"
                    if low_accuracy_with_lower
                    else "REPEATED_RULE_DESCENT"
                ),
            )

    next_level = one_step_higher(current)
    enough_for_upgrade = (
        current is not LadderLevel.L4
        and current is not LadderLevel.L0
        and evidence.completed_session_count >= 2
        and performance.attempts >= 4
        and accuracy is not None
        and accuracy >= 0.90
        and evidence.valid_speech_count >= 2
        and len(predictions) >= 2
        and all(
            ladder_rank(level) >= ladder_rank(next_level)
            for level in predictions[-2:]
        )
    )
    if enough_for_upgrade:
        return LadderDecision(
            action=LadderAction.UPGRADE,
            current_level=current,
            recommended_level=next_level,
            current_accuracy=accuracy,
            evidence_count=performance.attempts,
            reason_code="MASTERY_AND_HIGHER_PREDICTIONS",
        )

    insufficient = evidence.completed_session_count < 2 or performance.attempts < 4
    return LadderDecision(
        action=(
            LadderAction.INSUFFICIENT_EVIDENCE if insufficient else LadderAction.MAINTAIN
        ),
        current_level=current,
        recommended_level=current,
        current_accuracy=accuracy,
        evidence_count=performance.attempts,
        reason_code="INSUFFICIENT_EVIDENCE" if insufficient else "MAINTAIN_CURRENT_LEVEL",
    )
