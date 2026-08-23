from __future__ import annotations

import json

import pytest

from mormi_api.ladder_model.speech_trainer import (
    load_speech_jsonl,
    write_speech_confusion_csv,
)


def test_speech_loader_accepts_three_labels_and_rejects_l0(tmp_path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        "".join(
            json.dumps({"input_text": f"speech-{label}", "label": index, "target_level": label})
            + "\n"
            for index, label in enumerate(("L2", "L3", "L4"))
        ),
        encoding="utf-8",
    )
    assert [row["label"] for row in load_speech_jsonl(valid)] == [0, 1, 2]

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps({"input_text": "joint", "label": 0, "target_level": "L0"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_level"):
        load_speech_jsonl(invalid)


def test_speech_confusion_csv_uses_three_model_labels(tmp_path) -> None:
    output = tmp_path / "confusion.csv"
    write_speech_confusion_csv(output, [[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    assert output.read_text(encoding="utf-8").splitlines() == [
        "actual\\predicted,L2,L3,L4",
        "L2,2,0,0",
        "L3,0,3,0",
        "L4,0,0,4",
    ]
