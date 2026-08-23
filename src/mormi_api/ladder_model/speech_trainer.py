from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .dataset import LadderLevel
from .metrics import LadderMetrics, evaluate_label_ids
from .speech_dataset import SPEECH_LABEL_TO_ID, SPEECH_LEVELS

SPEECH_ID_TO_LABEL = {index: level.value for level, index in SPEECH_LABEL_TO_ID.items()}


@dataclass(frozen=True)
class SpeechTrainConfig:
    train_path: Path
    validation_path: Path
    test_path: Path
    output_dir: Path
    base_model: str = "klue/roberta-base"
    epochs: float = 8.0
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    max_length: int = 256
    seed: int = 20260823
    fp16: bool = False


def load_speech_jsonl(path: Path) -> list[dict[str, object]]:
    expected = {level.value: index for level, index in SPEECH_LABEL_TO_ID.items()}
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get("input_text")
            label = value.get("label")
            target = value.get("target_level")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} has no input_text")
            if not isinstance(label, int) or label not in SPEECH_ID_TO_LABEL:
                raise ValueError(f"{path}:{line_number} has an invalid label")
            if target not in expected or expected[str(target)] != label:
                raise ValueError(f"{path}:{line_number} label does not match target_level")
            rows.append({"input_text": text, "label": label})
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def write_speech_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    labels = [level.value for level in SPEECH_LEVELS]
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise ValueError("speech confusion matrix must be 3x3")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_dict(metrics: LadderMetrics) -> dict[str, float]:
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "mean_absolute_rank_error": metrics.mean_absolute_rank_error,
        "severe_error_rate": metrics.severe_error_rate,
    }


def train_speech_model(config: SpeechTrainConfig) -> LadderMetrics:
    try:
        import numpy as np
        import torch
        from datasets import Dataset  # type: ignore[import-untyped]
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover - local ML environment boundary
        raise RuntimeError("install the optional analysis dependencies before training") from exc

    set_seed(config.seed)
    train_rows = load_speech_jsonl(config.train_path)
    validation_rows = load_speech_jsonl(config.validation_path)
    test_rows = load_speech_jsonl(config.test_path)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        config.base_model, use_fast=True
    )

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            tokenizer(
                batch["input_text"],
                truncation=True,
                max_length=config.max_length,
            ),
        )

    train_dataset = Dataset.from_list(train_rows).map(tokenize, batched=True)
    validation_dataset = Dataset.from_list(validation_rows).map(tokenize, batched=True)
    test_dataset = Dataset.from_list(test_rows).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=3,
        id2label=SPEECH_ID_TO_LABEL,
        label2id={value: key for key, value in SPEECH_ID_TO_LABEL.items()},
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        predicted = np.asarray(prediction.predictions).argmax(axis=-1).tolist()
        expected = np.asarray(prediction.label_ids).tolist()
        metrics = evaluate_label_ids(expected=expected, predicted=predicted, class_count=3)
        return _metric_dict(metrics)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        fp16=config.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    model_dir = config.output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    test_prediction = trainer.predict(test_dataset)
    predicted = np.asarray(test_prediction.predictions).argmax(axis=-1).tolist()
    expected = np.asarray(test_prediction.label_ids).tolist()
    metrics = evaluate_label_ids(expected=expected, predicted=predicted, class_count=3)
    (config.output_dir / "test-metrics.json").write_text(
        json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_speech_confusion_csv(
        config.output_dir / "confusion-matrix.csv",
        metrics.confusion_matrix,
    )

    weight_candidates = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    if not weight_candidates:
        raise RuntimeError("trained model weights were not saved")
    manifest = {
        "schema_version": 2,
        "model_version": "ladder-speech-klue-v2",
        "base_model": config.base_model,
        "label_order": [level.value for level in SPEECH_LEVELS],
        "rubric_version": "ladder-speech-label-v2",
        "policy_levels": [LadderLevel.L0.value],
        "seed": config.seed,
        "epochs_requested": config.epochs,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "weight_file": weight_candidates[0].name,
        "weight_sha256": _sha256(weight_candidates[0]),
        "limitations": "Synthetic speech prototype; requires expert-reviewed child speech.",
    }
    (config.output_dir / "model-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics
