from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .dataset import ConceptResult, LadderLevel, ResponseMode, canonical_level

SPEECH_LEVELS = (LadderLevel.L2, LadderLevel.L3, LadderLevel.L4)
SPEECH_LABEL_TO_ID = {level: index for index, level in enumerate(SPEECH_LEVELS)}


class SpeechExample(BaseModel):
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
    template_group: str
    synthetic: bool
    source: Literal["validated_speech", "synthetic_rubric"]
    annotation_reason: str = Field(min_length=1)
    rubric_version: Literal["ladder-speech-label-v2"] = "ladder-speech-label-v2"

    @field_validator("response_mode")
    @classmethod
    def speech_mode_only(cls, value: ResponseMode) -> ResponseMode:
        if value not in {ResponseMode.FREE_TEXT, ResponseMode.SHORT_ANSWER}:
            raise ValueError("speech model accepts text responses only")
        return value

    @field_validator("target_level")
    @classmethod
    def speech_target_only(cls, value: LadderLevel) -> LadderLevel:
        if value not in SPEECH_LEVELS:
            raise ValueError("L0 is handled by deterministic policy")
        return value


class SpeechDatasetSummary(BaseModel):
    total: int
    validated_count: int
    synthetic_count: int
    label_counts: dict[str, int]
    split_counts: dict[str, int]
    learner_counts: dict[str, int]
    exact_input_overlap_count: int
    template_overlap_count: int
    learner_overlap_count: int
    rubric_version: str = "ladder-speech-label-v2"
    seed: int


def serialize_speech_input(example: SpeechExample) -> str:
    """Return only child speech; interaction metadata belongs to the rule policy."""
    return example.response_text


def as_speech_training_record(example: SpeechExample) -> dict[str, object]:
    return {
        **example.model_dump(mode="json"),
        "input_text": serialize_speech_input(example),
        "label": SPEECH_LABEL_TO_ID[example.target_level],
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _learner_key(value: object, salt: bytes) -> str:
    digest = hmac.new(salt, str(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"anon_{digest[:12]}"


def _sample_id(*values: object) -> str:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()
    return f"speech_{digest[:16]}"


def _template_group(text: str) -> str:
    normalized = re.sub(r"\d[\d,]*", "<NUM>", text.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return f"real_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]}"


def _assign_learners(
    learner_keys: list[str],
    validated_l4_counts: Counter[str],
    seed: int,
) -> dict[str, str]:
    ranked = sorted(learner_keys, key=lambda learner: (-validated_l4_counts[learner], learner))
    anchors = ranked[:3]
    assignment = {anchors[0]: "train", anchors[1]: "validation", anchors[2]: "test"}
    remaining = [learner for learner in learner_keys if learner not in assignment]
    random.Random(seed).shuffle(remaining)
    capacities = {"train": 9, "validation": 2, "test": 2}
    for split in ("train", "validation", "test"):
        for _ in range(capacities[split]):
            assignment[remaining.pop()] = split
    if remaining:
        raise ValueError("learner assignment did not consume exactly sixteen learners")
    return assignment


_SKILLS = ("number-count", "number-compare", "money-count", "money-price", "money-budget")

_SPEECH_PATTERNS: dict[str, dict[LadderLevel, tuple[str, ...]]] = {
    "train": {
        LadderLevel.L4: (
            "하나씩 확인해서 {answer}이 되었고 그래서 이 답이 맞아.",
            "먼저 수를 세고 다음에 합쳤더니 {answer}이 나왔어.",
            "내가 순서대로 계산한 방법은 {method}이라서 답은 {answer}이야.",
        ),
        LadderLevel.L3: ("답은 {answer}이야.", "{method}로 했어.", "나는 {answer}이라고 생각해."),
        LadderLevel.L2: ("음, {answer}인가?", "아마 {answer}...", "{method}? 잘 모르겠어."),
    },
    "validation": {
        LadderLevel.L4: (
            "양쪽을 비교해 보니 {reason} 때문에 결과가 {answer}이야.",
            "내 생각에는 {method} 순서로 풀면 {answer}을 구할 수 있어.",
            "계산을 다시 확인했는데 {reason}이라서 {answer}이 맞아.",
        ),
        LadderLevel.L3: ("내 답은 {answer}.", "{answer} 같아.", "방법은 {method}야."),
        LadderLevel.L2: ("혹시 {answer}?", "{answer}일지도...", "잘 모르지만 {method}?"),
    },
    "test": {
        LadderLevel.L4: (
            "처음 값에서 차례로 따져 보니까 {reason}, 그래서 {answer}이 나와.",
            "그림을 하나씩 짝지어 확인하면 {reason}이라 답이 {answer}이야.",
            "내가 사용한 풀이는 {method}이고 그 결과 {answer}을 얻었어.",
        ),
        LadderLevel.L3: ("{answer}라고 봐.", "결과는 {answer}.", "나는 {method}를 썼어."),
        LadderLevel.L2: ("{answer}... 맞나?", "잘 모르겠는데 {answer}?", "어... {method}?"),
    },
}


def _speech_values(skill: str, index: int) -> dict[str, str]:
    number = 2 + index % 8
    money = (2 + index % 18) * 100
    if skill.startswith("money"):
        return {
            "answer": f"{money}원",
            "method": "돈의 값을 더하는 것",
            "reason": "낸 돈과 가격의 차이를 계산했기",
        }
    return {
        "answer": f"{number}개",
        "method": "왼쪽부터 하나씩 세는 것",
        "reason": "남는 쪽이 더 많기",
    }


def _synthetic_example(
    *,
    split: str,
    level: LadderLevel,
    index: int,
    learner_keys: list[str],
) -> SpeechExample:
    skill = _SKILLS[index % len(_SKILLS)]
    patterns = _SPEECH_PATTERNS[split][level]
    family = index % len(patterns)
    text = patterns[family].format(**_speech_values(skill, index))
    current = LadderLevel.L4 if index % 2 == 0 else LadderLevel.L3
    mode = ResponseMode.FREE_TEXT if index % 2 == 0 else ResponseMode.SHORT_ANSWER
    result = ConceptResult.CORRECT if index % 2 == 0 else ConceptResult.INCORRECT
    learner = learner_keys[index % len(learner_keys)]
    session_id = f"synthetic-speech-{split}-{level.value.lower()}-{index:04d}"
    return SpeechExample(
        sample_id=_sample_id(session_id, learner),
        learner_key=learner,
        learning_session_id=session_id,
        study_date=date(2026, 8, 17) + timedelta(days=index % 7),
        skill_id=skill,
        current_level=current,
        response_mode=mode,
        response_text=text,
        concept_result=result,
        attempt_count=1 + index % 3,
        target_level=level,
        template_group=f"synthetic_{split}_{level.value}_{family}",
        synthetic=True,
        source="synthetic_rubric",
        annotation_reason={
            LadderLevel.L4: "답과 이유 또는 방법을 문장으로 연결했다.",
            LadderLevel.L3: "답 또는 방법을 직접 생성했지만 설명 슬롯이 일부 비었다.",
            LadderLevel.L2: "독립 설명이 어려워 불확실한 단편만 생성했다.",
        }[level],
    )


def _write_split(path: Path, rows: list[SpeechExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(as_speech_training_record(row), ensure_ascii=False) + "\n")


def _overlap_count(splits: dict[str, list[SpeechExample]], field: str) -> int:
    total = 0
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        left_values = {getattr(row, field) for row in splits[left]}
        right_values = {getattr(row, field) for row in splits[right]}
        total += len(left_values & right_values)
    return total


def prepare_speech_dataset(
    *,
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    hmac_salt: bytes,
    train_per_level: int = 60,
    validation_per_level: int = 20,
    test_per_level: int = 20,
    seed: int = 20260823,
) -> SpeechDatasetSummary:
    if len(hmac_salt) < 16:
        raise ValueError("hmac_salt must contain at least sixteen bytes")
    source_rows = _read_jsonl(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failed = {
        str(row["learning_session_id"])
        for row in audit.get("failed_sessions", [])
        if isinstance(row, dict) and row.get("learning_session_id")
    }
    learner_keys = sorted({_learner_key(row["learner_id"], hmac_salt) for row in source_rows})
    synthetic_index = 0
    while len(learner_keys) < 16:
        candidate = _learner_key(f"synthetic-only-{synthetic_index}", hmac_salt)
        synthetic_index += 1
        if candidate not in learner_keys:
            learner_keys.append(candidate)
    learner_keys.sort()
    if len(learner_keys) != 16:
        raise ValueError(f"expected at most sixteen source learners, found {len(learner_keys)}")

    valid_l4_rows = [
        row
        for row in source_rows
        if str(row["learning_session_id"]) not in failed
        and canonical_level(row["recommended_expression_level"]) is LadderLevel.L4
        and str(row.get("utterance") or "").strip()
    ]
    l4_counts: Counter[str] = Counter(
        _learner_key(row["learner_id"], hmac_salt) for row in valid_l4_rows
    )
    assignment = _assign_learners(learner_keys, l4_counts, seed)
    split_learners = {
        split: sorted(learner for learner, assigned in assignment.items() if assigned == split)
        for split in ("train", "validation", "test")
    }
    targets = {
        "train": train_per_level,
        "validation": validation_per_level,
        "test": test_per_level,
    }
    splits: dict[str, list[SpeechExample]] = {name: [] for name in targets}
    template_owner: dict[str, str] = {}
    for row in valid_l4_rows:
        learner = _learner_key(row["learner_id"], hmac_salt)
        split = assignment[learner]
        if sum(item.target_level is LadderLevel.L4 for item in splits[split]) >= targets[split]:
            continue
        text = str(row["utterance"]).strip()
        group = _template_group(text)
        if group in template_owner and template_owner[group] != split:
            continue
        template_owner[group] = split
        session_id = str(row["learning_session_id"])
        splits[split].append(
            SpeechExample(
                sample_id=_sample_id(session_id, "validated"),
                learner_key=learner,
                learning_session_id=session_id,
                study_date=date.fromisoformat(str(row["study_date"])),
                skill_id=str(row["curriculum_session_id"]),
                current_level=LadderLevel.L4,
                response_mode=ResponseMode.FREE_TEXT,
                response_text=text,
                concept_result=ConceptResult.CORRECT,
                attempt_count=max(1, int(str(row.get("wrong_attempt_count", 0))) + 1),
                target_level=LadderLevel.L4,
                template_group=group,
                synthetic=False,
                source="validated_speech",
                annotation_reason="검증된 세션에서 답과 설명을 독립적으로 문장화했다.",
            )
        )

    for split, target_count in targets.items():
        for level in SPEECH_LEVELS:
            current_count = sum(item.target_level is level for item in splits[split])
            for index in range(current_count, target_count):
                splits[split].append(
                    _synthetic_example(
                        split=split,
                        level=level,
                        index=index,
                        learner_keys=split_learners[split],
                    )
                )
        random.Random(seed + len(split)).shuffle(splits[split])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_split(output_dir / f"{split}.jsonl", rows)
    all_rows = [row for rows in splits.values() for row in rows]
    _write_split(output_dir / "all.jsonl", all_rows)

    input_sets = {
        split: {serialize_speech_input(row) for row in rows}
        for split, rows in splits.items()
    }
    exact_overlap = sum(
        len(input_sets[left] & input_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    template_overlap = _overlap_count(splits, "template_group")
    learner_overlap = _overlap_count(splits, "learner_key")
    if exact_overlap or template_overlap or learner_overlap:
        raise ValueError("dataset split leakage detected")

    summary = SpeechDatasetSummary(
        total=len(all_rows),
        validated_count=sum(not row.synthetic for row in all_rows),
        synthetic_count=sum(row.synthetic for row in all_rows),
        label_counts=dict(sorted(Counter(row.target_level.value for row in all_rows).items())),
        split_counts={split: len(rows) for split, rows in splits.items()},
        learner_counts={split: len(values) for split, values in split_learners.items()},
        exact_input_overlap_count=exact_overlap,
        template_overlap_count=template_overlap,
        learner_overlap_count=learner_overlap,
        seed=seed,
    )
    (output_dir / "dataset-summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split-manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "rubric_version": summary.rubric_version,
                "learners": split_learners,
                "template_groups": {
                    split: sorted({row.template_group for row in rows})
                    for split, rows in splits.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
