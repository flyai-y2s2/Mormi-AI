from __future__ import annotations

from pydantic import BaseModel


class LadderMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    mean_absolute_rank_error: float
    severe_error_rate: float
    confusion_matrix: list[list[int]]
    support: list[int]


def evaluate_label_ids(
    *,
    expected: list[int],
    predicted: list[int],
    class_count: int = 4,
) -> LadderMetrics:
    if not expected or len(expected) != len(predicted):
        raise ValueError("expected and predicted must have equal non-zero lengths")
    if any(value < 0 or value >= class_count for value in expected + predicted):
        raise ValueError("label id is outside the configured class range")

    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for truth, guess in zip(expected, predicted, strict=True):
        confusion[truth][guess] += 1

    f1_scores: list[float] = []
    support: list[int] = []
    for label in range(class_count):
        true_positive = confusion[label][label]
        false_positive = sum(confusion[row][label] for row in range(class_count)) - true_positive
        false_negative = sum(confusion[label]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        support.append(sum(confusion[label]))

    total = len(expected)
    absolute_errors = [abs(truth - guess) for truth, guess in zip(expected, predicted, strict=True)]
    return LadderMetrics(
        accuracy=sum(truth == guess for truth, guess in zip(expected, predicted, strict=True))
        / total,
        macro_f1=sum(f1_scores) / class_count,
        mean_absolute_rank_error=sum(absolute_errors) / total,
        severe_error_rate=sum(error >= 2 for error in absolute_errors) / total,
        confusion_matrix=confusion,
        support=support,
    )
