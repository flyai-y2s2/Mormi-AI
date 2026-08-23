from __future__ import annotations

from .dataset import LadderLevel, ResponseMode

_LOWER = {
    LadderLevel.L4: LadderLevel.L3,
    LadderLevel.L3: LadderLevel.L2,
    LadderLevel.L2: LadderLevel.L0,
    LadderLevel.L0: LadderLevel.L0,
}


def rule_recommendation(
    current_level: LadderLevel,
    response_mode: ResponseMode,
) -> LadderLevel | None:
    """Return a deterministic recommendation or delegate speech to the model."""
    if response_mode is ResponseMode.NO_RESPONSE:
        return _LOWER[current_level]
    if response_mode is ResponseMode.CHOICE:
        return LadderLevel.L2
    if response_mode is ResponseMode.SOLVE_TOGETHER:
        return LadderLevel.L0
    return None

