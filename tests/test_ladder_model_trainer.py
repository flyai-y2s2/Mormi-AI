from __future__ import annotations

import json

import pytest

from mormi_api.ladder_model.trainer import load_labeled_jsonl, write_confusion_csv


def test_loader_returns_only_model_text_and_integer_label(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "input_text": "[현재단계=L4] 아이 응답",
                "label": 2,
                "target_level": "L3",
                "learner_key": "anon_x",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_labeled_jsonl(path) == [
        {"input_text": "[현재단계=L4] 아이 응답", "label": 2}
    ]

def test_loader_rejects_label_target_disagreement(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"input_text": "x", "label": 3, "target_level": "L3"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        load_labeled_jsonl(path)


def test_confusion_csv_has_fixed_ladder_order(tmp_path) -> None:
    output = tmp_path / "confusion.csv"
    write_confusion_csv(output, [[1, 2, 3, 4], [0, 5, 0, 0], [0, 0, 6, 0], [1, 0, 0, 7]])

    assert output.read_text(encoding="utf-8").splitlines()[:2] == [
        "actual\\predicted,L0,L2,L3,L4",
        "L0,1,2,3,4",
    ]
