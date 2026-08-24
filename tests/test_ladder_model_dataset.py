from __future__ import annotations

import json
from collections import Counter

from mormi_api.ladder_model.dataset import (
    ConceptResult,
    LadderLevel,
    ResponseMode,
    build_training_examples,
    split_by_learner,
)
from mormi_api.ladder_model.preparation import prepare_dataset


def _row(
    session_id: str,
    learner: str,
    level: str,
    *,
    mode: str,
    utterance: str | None = None,
) -> dict[str, object]:
    return {
        "learner_id": learner,
        "study_date": "2026-08-17",
        "curriculum_session_id": "number-count",
        "current_expression_level": level,
        "response_mode": mode,
        "utterance": utterance,
        "wrong_attempt_count": 1,
        "recommended_expression_level": level,
        "learning_session_id": session_id,
    }


def test_failed_sessions_are_excluded_and_no_response_is_not_missing_data() -> None:
    rows = [
        _row("ok", "testb01", "L2", mode="choice"),
        _row("failed", "testb02", "L2", mode="choice"),
    ]

    examples = build_training_examples(
        rows,
        failed_session_ids={"failed"},
        hmac_salt=b"test-salt",
        target_per_level=None,
        seed=7,
    )

    assert {example.learning_session_id for example in examples} == {"ok"}
    descent = next(
        example
        for example in examples
        if example.current_level is LadderLevel.L4
        and example.target_level is LadderLevel.L3
    )
    assert descent.response_mode is ResponseMode.NO_RESPONSE
    assert descent.response_text == "[NO_RESPONSE]"
    assert descent.concept_result is ConceptResult.NOT_ASSESSED


def test_balancing_produces_four_labels_without_current_level_copy_leakage() -> None:
    rows = [
        _row("l4", "testc01", "L4", mode="free_text", utterance="점을 세어서 세 개야."),
        _row("l2", "testb01", "L2", mode="choice"),
        _row("l0", "testg01", "L0", mode="solve_together"),
    ]

    examples = build_training_examples(
        rows,
        failed_session_ids=set(),
        hmac_salt=b"test-salt",
        target_per_level=8,
        seed=11,
    )

    assert Counter(example.target_level for example in examples) == {
        LadderLevel.L0: 8,
        LadderLevel.L2: 8,
        LadderLevel.L3: 8,
        LadderLevel.L4: 8,
    }
    assert any(example.current_level is not example.target_level for example in examples)
    assert any(
        example.synthetic
        and example.target_level is LadderLevel.L3
        and example.response_mode is ResponseMode.SHORT_ANSWER
        for example in examples
    )
    assert any(
        example.synthetic
        and example.target_level is LadderLevel.L2
        and "선택:" in example.response_text
        for example in examples
    )


def test_learner_split_has_no_overlap_and_exact_10_3_3_groups() -> None:
    rows = [
        _row(f"s{index}", f"learner-{index:02d}", "L4", mode="free_text", utterance="세 개야.")
        for index in range(16)
    ]
    examples = build_training_examples(
        rows,
        failed_session_ids=set(),
        hmac_salt=b"test-salt",
        target_per_level=None,
        seed=13,
    )

    split = split_by_learner(examples, seed=20260823)

    train = {row.learner_key for row in split.train}
    validation = {row.learner_key for row in split.validation}
    test = {row.learner_key for row in split.test}
    assert (len(train), len(validation), len(test)) == (10, 3, 3)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_balancing_keeps_all_sixteen_learners_for_group_split() -> None:
    rows = [
        _row(
            f"s{index}",
            f"learner-{index:02d}",
            "L4" if index % 2 else "L2",
            mode="free_text" if index % 2 else "choice",
            utterance="답과 이유를 설명했어." if index % 2 else None,
        )
        for index in range(16)
    ]

    examples = build_training_examples(
        rows,
        failed_session_ids=set(),
        hmac_salt=b"test-salt",
        target_per_level=20,
        seed=20260823,
    )

    assert len({row.learner_key for row in examples}) == 16
    split_by_learner(examples, seed=20260823)


def test_synthetic_supplement_keeps_learners_whose_sessions_failed_audit() -> None:
    rows = [
        _row(
            f"s{index}",
            f"learner-{index:02d}",
            "L3" if index >= 9 else "L4",
            mode="short_answer" if index >= 9 else "free_text",
            utterance="짧은 답" if index >= 9 else "답과 이유를 설명했어.",
        )
        for index in range(16)
    ]

    examples = build_training_examples(
        rows,
        failed_session_ids={f"s{index}" for index in range(9, 16)},
        hmac_salt=b"test-salt",
        target_per_level=20,
        seed=20260823,
    )

    assert len({row.learner_key for row in examples}) == 16


def test_synthetic_only_learners_fill_missing_collection_accounts() -> None:
    rows = [
        _row(
            f"s{index}",
            f"learner-{index:02d}",
            "L4",
            mode="free_text",
            utterance="답과 이유를 설명했어.",
        )
        for index in range(13)
    ]

    examples = build_training_examples(
        rows,
        failed_session_ids=set(),
        hmac_salt=b"test-salt",
        target_per_level=20,
        seed=20260823,
    )

    assert len({row.learner_key for row in examples}) == 16
    split_by_learner(examples, seed=20260823)


def test_group_split_keeps_every_label_in_validation_and_test() -> None:
    rows = []
    for learner in range(3):
        rows.extend(
            _row(
                f"l4-{learner}-{sample}",
                f"learner-{learner:02d}",
                "L4",
                mode="free_text",
                utterance="답과 이유를 설명했어.",
            )
            for sample in range(10)
        )
    for learner in range(3, 8):
        rows.extend(
            _row(
                f"l2-{learner}-{sample}",
                f"learner-{learner:02d}",
                "L2",
                mode="choice",
            )
            for sample in range(10)
        )
    rows.extend(
        _row(
            f"failed-{learner}",
            f"learner-{learner:02d}",
            "L3",
            mode="short_answer",
            utterance="짧은 답",
        )
        for learner in range(8, 16)
    )
    examples = build_training_examples(
        rows,
        failed_session_ids={f"failed-{learner}" for learner in range(8, 16)},
        hmac_salt=b"test-salt",
        target_per_level=30,
        seed=20260823,
    )

    split = split_by_learner(examples, seed=20260823)

    assert {row.target_level for row in split.validation} == set(LadderLevel)
    assert {row.target_level for row in split.test} == set(LadderLevel)


def test_prepare_dataset_writes_labeled_splits_without_account_ids(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _row(
            f"s{index}",
            f"test-user-{index:02d}",
            "L4" if index % 2 else "L2",
            mode="free_text" if index % 2 else "choice",
            utterance="세어서 답을 구했어." if index % 2 else None,
        )
        for index in range(16)
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"failed_sessions": []}), encoding="utf-8")

    summary = prepare_dataset(
        manifest_path=manifest,
        audit_path=audit,
        output_dir=tmp_path / "out",
        hmac_salt=b"local-test-salt",
        target_per_level=16,
        seed=20260823,
    )

    assert summary.total == 64
    assert summary.label_counts == {"L0": 16, "L2": 16, "L3": 16, "L4": 16}
    text = (tmp_path / "out" / "all.jsonl").read_text(encoding="utf-8")
    assert "test-user" not in text
    assert '"input_text"' in text
    assert '"label"' in text
    assert (tmp_path / "out" / "train.jsonl").exists()
    assert (tmp_path / "out" / "validation.jsonl").exists()
    assert (tmp_path / "out" / "test.jsonl").exists()
