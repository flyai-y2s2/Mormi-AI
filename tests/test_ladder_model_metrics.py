from __future__ import annotations

from mormi_api.ladder_model.metrics import evaluate_label_ids


def test_metrics_include_confusion_macro_f1_and_ordinal_errors() -> None:
    metrics = evaluate_label_ids(
        expected=[0, 1, 2, 3],
        predicted=[0, 2, 2, 0],
    )

    assert metrics.confusion_matrix == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]
    assert metrics.accuracy == 0.5
    assert metrics.mean_absolute_rank_error == 1.0
    assert metrics.severe_error_rate == 0.25
    assert round(metrics.macro_f1, 4) == 0.3333


def test_metrics_reject_empty_or_mismatched_inputs() -> None:
    for expected, predicted in (([], []), ([0], [])):
        try:
            evaluate_label_ids(expected=expected, predicted=predicted)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid metric input must fail")
