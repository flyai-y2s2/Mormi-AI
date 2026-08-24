from __future__ import annotations

import json
from collections import defaultdict

from mormi_api.ladder_model.speech_dataset import prepare_speech_dataset


def _row(index: int, learner: int, level: str) -> dict[str, object]:
    return {
        "learner_id": f"prototype-{learner:02d}",
        "study_date": "2026-08-17",
        "curriculum_session_id": "number-count",
        "current_expression_level": level,
        "response_mode": "free_text" if level == "L4" else "short_answer",
        "utterance": f"점을 차례로 세어서 모두 {index + 2}개야.",
        "wrong_attempt_count": index % 2,
        "recommended_expression_level": level,
        "learning_session_id": f"session-{learner}-{index}",
    }


def test_speech_dataset_has_no_rule_only_labels_or_split_leakage(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for learner in range(3):
        rows.extend(_row(index, learner, "L4") for index in range(8))
    rows.extend(_row(0, learner, "L3") for learner in range(3, 16))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "failed_sessions": [
                    {"learning_session_id": f"session-{learner}-0"}
                    for learner in range(3, 16)
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = prepare_speech_dataset(
        manifest_path=manifest,
        audit_path=audit,
        output_dir=tmp_path / "speech-v2",
        hmac_salt=b"sixteen-char-salt",
        train_per_level=12,
        validation_per_level=6,
        test_per_level=6,
        seed=20260823,
    )

    assert summary.label_counts == {"L2": 24, "L3": 24, "L4": 24}
    splits = {
        name: [
            json.loads(line)
            for line in (tmp_path / "speech-v2" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for name in ("train", "validation", "test")
    }
    for rows_in_split in splits.values():
        assert {row["target_level"] for row in rows_in_split} == {"L2", "L3", "L4"}
        assert all(row["response_mode"] in {"free_text", "short_answer"} for row in rows_in_split)
        assert all("응답방식" not in row["input_text"] for row in rows_in_split)
        assert all("현재단계" not in row["input_text"] for row in rows_in_split)
        assert all("정답여부" not in row["input_text"] for row in rows_in_split)

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert {row["learner_key"] for row in splits[left]}.isdisjoint(
            {row["learner_key"] for row in splits[right]}
        )
        assert {row["template_group"] for row in splits[left]}.isdisjoint(
            {row["template_group"] for row in splits[right]}
        )
        assert {row["input_text"] for row in splits[left]}.isdisjoint(
            {row["input_text"] for row in splits[right]}
        )


def test_same_metadata_combination_has_multiple_human_labels(tmp_path) -> None:
    rows = [_row(index, learner, "L4") for learner in range(16) for index in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"failed_sessions": []}), encoding="utf-8")
    prepare_speech_dataset(
        manifest_path=manifest,
        audit_path=audit,
        output_dir=tmp_path / "speech-v2",
        hmac_salt=b"sixteen-char-salt",
        train_per_level=12,
        validation_per_level=6,
        test_per_level=6,
        seed=20260823,
    )
    train = [
        json.loads(line)
        for line in (tmp_path / "speech-v2" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    labels_by_metadata: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in train:
        labels_by_metadata[
            (row["current_level"], row["response_mode"], row["concept_result"])
        ].add(row["target_level"])
    assert max(map(len, labels_by_metadata.values())) >= 2
