from __future__ import annotations

import hashlib
import hmac
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date, timedelta
from enum import StrEnum
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class LadderLevel(StrEnum):
    L0 = "L0"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


LADDER_ORDER = (LadderLevel.L0, LadderLevel.L2, LadderLevel.L3, LadderLevel.L4)


class ResponseMode(StrEnum):
    FREE_TEXT = "free_text"
    SHORT_ANSWER = "short_answer"
    CHOICE = "choice"
    NO_RESPONSE = "no_response"
    SOLVE_TOGETHER = "solve_together"


class ConceptResult(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NOT_ASSESSED = "not_assessed"


class LadderExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    learner_key: str
    learning_session_id: str
    study_date: date
    skill_id: str
    current_level: LadderLevel
    response_mode: ResponseMode
    response_text: str = Field(min_length=1)
    concept_result: ConceptResult
    attempt_count: int = Field(ge=1)
    target_level: LadderLevel
    synthetic: bool
    source: Literal["validated_session", "validated_descent", "synthetic_rubric"]
    rubric_version: Literal["ladder-label-v1"] = "ladder-label-v1"


class DatasetSplit(NamedTuple):
    train: list[LadderExample]
    validation: list[LadderExample]
    test: list[LadderExample]


def canonical_level(value: object) -> LadderLevel:
    normalized = str(value).upper()
    if normalized == "L1":
        normalized = "L2"
    return LadderLevel(normalized)


def _learner_key(value: object, salt: bytes) -> str:
    digest = hmac.new(salt, str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"anon_{digest[:12]}"


def _sample_id(*parts: object) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return f"sample_{hashlib.sha256(payload).hexdigest()[:16]}"


def _normalize_terminal_mode(level: LadderLevel, raw: object) -> ResponseMode:
    value = str(raw or "").lower()
    if level is LadderLevel.L4:
        return ResponseMode.FREE_TEXT
    if level is LadderLevel.L3:
        return ResponseMode.SHORT_ANSWER
    if level is LadderLevel.L2 or value in {"choice", "choices"}:
        return ResponseMode.CHOICE
    return ResponseMode.SOLVE_TOGETHER


def _terminal_text(mode: ResponseMode, utterance: object) -> str:
    text = str(utterance or "").strip()
    if text:
        return text
    if mode is ResponseMode.CHOICE:
        return "[CHOICE_SELECTED]"
    if mode is ResponseMode.SOLVE_TOGETHER:
        return "[SOLVE_TOGETHER]"
    return "[NO_RESPONSE]"


def _descent_levels(target: LadderLevel) -> list[tuple[LadderLevel, LadderLevel]]:
    if target is LadderLevel.L3:
        return [(LadderLevel.L4, LadderLevel.L3)]
    if target is LadderLevel.L2:
        return [
            (LadderLevel.L4, LadderLevel.L3),
            (LadderLevel.L3, LadderLevel.L2),
        ]
    if target is LadderLevel.L0:
        return [
            (LadderLevel.L4, LadderLevel.L3),
            (LadderLevel.L3, LadderLevel.L2),
            (LadderLevel.L2, LadderLevel.L0),
        ]
    return []


def _validated_examples(
    rows: Iterable[dict[str, object]],
    *,
    failed_session_ids: set[str],
    hmac_salt: bytes,
) -> list[LadderExample]:
    examples: list[LadderExample] = []
    for row in rows:
        session_id = str(row["learning_session_id"])
        if session_id in failed_session_ids:
            continue
        learner = _learner_key(row["learner_id"], hmac_salt)
        target = canonical_level(row["recommended_expression_level"])
        skill_id = str(row["curriculum_session_id"])
        study_date = date.fromisoformat(str(row["study_date"]))
        attempts = max(1, int(str(row.get("wrong_attempt_count", 0))) + 1)

        for index, (current, recommended) in enumerate(_descent_levels(target), start=1):
            examples.append(
                LadderExample(
                    sample_id=_sample_id(session_id, "descent", index),
                    learner_key=learner,
                    learning_session_id=session_id,
                    study_date=study_date,
                    skill_id=skill_id,
                    current_level=current,
                    response_mode=ResponseMode.NO_RESPONSE,
                    response_text="[NO_RESPONSE]",
                    concept_result=ConceptResult.NOT_ASSESSED,
                    attempt_count=index,
                    target_level=recommended,
                    synthetic=False,
                    source="validated_descent",
                )
            )

        mode = _normalize_terminal_mode(target, row.get("response_mode"))
        examples.append(
            LadderExample(
                sample_id=_sample_id(session_id, "terminal"),
                learner_key=learner,
                learning_session_id=session_id,
                study_date=study_date,
                skill_id=skill_id,
                current_level=target,
                response_mode=mode,
                response_text=_terminal_text(mode, row.get("utterance")),
                concept_result=(
                    ConceptResult.NOT_ASSESSED
                    if mode is ResponseMode.SOLVE_TOGETHER
                    else ConceptResult.CORRECT
                ),
                attempt_count=attempts,
                target_level=target,
                synthetic=False,
                source="validated_session",
            )
        )
    return examples


_SKILLS = ("number-count", "number-compare", "money-count", "money-price", "money-budget")


def _synthetic_text(level: LadderLevel, skill: str, index: int) -> str:
    number = 2 + (index % 8)
    money = (2 + (index % 18)) * 100
    if level is LadderLevel.L4:
        variants = {
            "number-count": f"점을 왼쪽부터 하나씩 세어 보니 모두 {number}개야.",
            "number-compare": "양쪽을 하나씩 짝지었더니 오른쪽에 하나가 남아서 오른쪽이 더 많아.",
            "money-count": f"동전 값을 차례로 더했더니 모두 {money}원이야.",
            "money-price": f"두 물건의 가격을 더해서 합계가 {money}원이야.",
            "money-budget": f"낸 돈에서 물건값을 빼면 {money}원이 남아.",
        }
        return variants[skill]
    if level is LadderLevel.L3:
        variants = {
            "number-count": f"{number}개야.",
            "number-compare": "오른쪽이 더 많아.",
            "money-count": f"{money}원이야.",
            "money-price": f"합하면 {money}원이야.",
            "money-budget": f"{money}원이 남아.",
        }
        return variants[skill]
    if level is LadderLevel.L2:
        variants = {
            "number-count": f"[CHOICE_SELECTED] 선택: {number}개",
            "number-compare": "[CHOICE_SELECTED] 선택: 오른쪽",
            "money-count": f"[CHOICE_SELECTED] 선택: {money}원",
            "money-price": f"[CHOICE_SELECTED] 선택: 합계 {money}원",
            "money-budget": f"[CHOICE_SELECTED] 선택: 남은 돈 {money}원",
        }
        return variants[skill]
    return "[SOLVE_TOGETHER]"


def _synthetic_example(
    level: LadderLevel,
    index: int,
    learner_keys: list[str],
) -> LadderExample:
    learner = learner_keys[index % len(learner_keys)]
    skill = _SKILLS[index % len(_SKILLS)]
    study_date = date(2026, 8, 17) + timedelta(days=index % 7)
    if level is LadderLevel.L4:
        current = LadderLevel.L4
        mode = ResponseMode.FREE_TEXT
        result = ConceptResult.CORRECT
    elif level is LadderLevel.L3:
        current = LadderLevel.L4 if index % 2 == 0 else LadderLevel.L3
        mode = ResponseMode.SHORT_ANSWER
        result = ConceptResult.CORRECT
    elif level is LadderLevel.L2:
        current = LadderLevel.L3 if index % 2 == 0 else LadderLevel.L2
        mode = ResponseMode.NO_RESPONSE if index % 2 == 0 else ResponseMode.CHOICE
        result = (
            ConceptResult.NOT_ASSESSED
            if mode is ResponseMode.NO_RESPONSE
            else ConceptResult.CORRECT
        )
    else:
        current = LadderLevel.L2 if index % 2 == 0 else LadderLevel.L0
        mode = ResponseMode.NO_RESPONSE if index % 2 == 0 else ResponseMode.SOLVE_TOGETHER
        result = ConceptResult.NOT_ASSESSED
    text = (
        "[NO_RESPONSE]"
        if mode is ResponseMode.NO_RESPONSE
        else _synthetic_text(level, skill, index)
    )
    session_id = f"synthetic-{level.value.lower()}-{index:04d}"
    return LadderExample(
        sample_id=_sample_id(session_id, learner),
        learner_key=learner,
        learning_session_id=session_id,
        study_date=study_date,
        skill_id=skill,
        current_level=current,
        response_mode=mode,
        response_text=text,
        concept_result=result,
        attempt_count=1 + index % 3,
        target_level=level,
        synthetic=True,
        source="synthetic_rubric",
    )


def _balance_examples(
    examples: list[LadderExample],
    *,
    target_per_level: int,
    seed: int,
    learner_keys: list[str] | None = None,
) -> list[LadderExample]:
    rng = random.Random(seed)
    learner_keys = learner_keys or sorted({example.learner_key for example in examples})
    if not learner_keys:
        raise ValueError("at least one learner is required")
    by_level: dict[LadderLevel, list[LadderExample]] = defaultdict(list)
    for example in examples:
        by_level[example.target_level].append(example)

    balanced: list[LadderExample] = []
    for level in LADDER_ORDER:
        candidates = list(by_level[level])
        rng.shuffle(candidates)
        if level is LadderLevel.L3:
            short_answers = [
                row for row in candidates if row.response_mode is ResponseMode.SHORT_ANSWER
            ]
            required_short_answers = max(1, target_per_level // 2)
            while len(short_answers) < required_short_answers:
                row = _synthetic_example(level, len(short_answers), learner_keys)
                candidates.append(row)
                short_answers.append(row)
        required_l2_choices: list[LadderExample] = []
        if level is LadderLevel.L2:
            required_choice_count = max(1, target_per_level // 2)
            required_l2_choices = [
                _synthetic_example(level, 10_001 + 2 * index, learner_keys)
                for index in range(required_choice_count)
            ]
            candidates.extend(required_l2_choices)
        while len(candidates) < target_per_level:
            candidates.append(_synthetic_example(level, len(candidates), learner_keys))
        rng.shuffle(candidates)
        selected = candidates[:target_per_level]
        if level is LadderLevel.L3:
            selected_short = [
                row for row in selected if row.response_mode is ResponseMode.SHORT_ANSWER
            ]
            if len(selected_short) < required_short_answers:
                others = [
                    row
                    for row in selected
                    if row.response_mode is not ResponseMode.SHORT_ANSWER
                ]
                selected = short_answers[:required_short_answers] + others[
                    : target_per_level - required_short_answers
                ]
        if level is LadderLevel.L2:
            required_ids = {row.sample_id for row in required_l2_choices}
            selected_required = [row for row in selected if row.sample_id in required_ids]
            if len(selected_required) < len(required_l2_choices):
                others = [row for row in selected if row.sample_id not in required_ids]
                selected = required_l2_choices + others[
                    : target_per_level - len(required_l2_choices)
                ]
        balanced.extend(selected)
    rng.shuffle(balanced)
    return balanced


def build_training_examples(
    rows: Iterable[dict[str, object]],
    *,
    failed_session_ids: set[str],
    hmac_salt: bytes,
    target_per_level: int | None = 100,
    seed: int = 20260823,
) -> list[LadderExample]:
    """Build decision-point examples from audited session summaries.

    A no-response descent becomes its own example so the current level cannot
    serve as a perfect proxy for the target label.
    """
    if not hmac_salt:
        raise ValueError("hmac_salt must not be empty")
    source_rows = list(rows)
    all_learner_keys = sorted(
        {_learner_key(row["learner_id"], hmac_salt) for row in source_rows}
    )
    synthetic_index = 0
    while len(all_learner_keys) < 16:
        candidate = _learner_key(f"synthetic-only-{synthetic_index}", hmac_salt)
        synthetic_index += 1
        if candidate not in all_learner_keys:
            all_learner_keys.append(candidate)
    all_learner_keys.sort()
    examples = _validated_examples(
        source_rows,
        failed_session_ids=failed_session_ids,
        hmac_salt=hmac_salt,
    )
    if target_per_level is None:
        return examples
    if target_per_level < 1:
        raise ValueError("target_per_level must be positive")
    return _balance_examples(
        examples,
        target_per_level=target_per_level,
        seed=seed,
        learner_keys=all_learner_keys,
    )


def split_by_learner(
    examples: Iterable[LadderExample],
    *,
    seed: int = 20260823,
) -> DatasetSplit:
    rows = list(examples)
    learners = sorted({row.learner_key for row in rows})
    if len(learners) != 16:
        raise ValueError(f"expected exactly 16 learners, found {len(learners)}")
    counts: dict[str, Counter[LadderLevel]] = {
        learner: Counter(
            row.target_level for row in rows if row.learner_key == learner
        )
        for learner in learners
    }
    global_counts = Counter(row.target_level for row in rows)

    def score(candidate: list[str]) -> float:
        groups = (candidate[:10], candidate[10:13], candidate[13:])
        fractions = (10 / 16, 3 / 16, 3 / 16)
        value = 0.0
        for group, fraction in zip(groups, fractions, strict=True):
            group_counts: Counter[LadderLevel] = Counter()
            for learner in group:
                group_counts.update(counts[learner])
            for level in LADDER_ORDER:
                if global_counts[level] and group_counts[level] == 0:
                    value += 1_000_000.0
                expected = global_counts[level] * fraction
                value += ((group_counts[level] - expected) / max(1.0, expected)) ** 2
        return value

    rng = random.Random(seed)
    best = list(learners)
    best_score = float("inf")
    for _ in range(20_000):
        candidate = list(learners)
        rng.shuffle(candidate)
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
            if best_score == 0:
                break

    train_learners = set(best[:10])
    validation_learners = set(best[10:13])
    test_learners = set(best[13:])
    return DatasetSplit(
        train=[row for row in rows if row.learner_key in train_learners],
        validation=[row for row in rows if row.learner_key in validation_learners],
        test=[row for row in rows if row.learner_key in test_learners],
    )
