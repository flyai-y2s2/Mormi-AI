from __future__ import annotations

from .dataset import LadderLevel, ResponseMode

_LOWER = {
    LadderLevel.L4: LadderLevel.L3,
    LadderLevel.L3: LadderLevel.L2,
    LadderLevel.L2: LadderLevel.L0,
    LadderLevel.L0: LadderLevel.L0,
}

_HIGHER = {
    LadderLevel.L0: LadderLevel.L2,
    LadderLevel.L2: LadderLevel.L3,
    LadderLevel.L3: LadderLevel.L4,
    LadderLevel.L4: LadderLevel.L4,
}

_RANK = {
    LadderLevel.L0: 0,
    LadderLevel.L2: 1,
    LadderLevel.L3: 2,
    LadderLevel.L4: 3,
}


def one_step_lower(level: LadderLevel) -> LadderLevel:
    return _LOWER[level]


def one_step_higher(level: LadderLevel) -> LadderLevel:
    return _HIGHER[level]


def ladder_rank(level: LadderLevel) -> int:
    return _RANK[level]


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
