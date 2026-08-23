from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import LadderLevel


@dataclass(frozen=True)
class RuntimePrediction:
    level: LadderLevel
    confidence: float

    def __post_init__(self) -> None:
        if self.level is LadderLevel.L0:
            raise ValueError("L0 is policy-only and cannot be predicted by the speech model")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True)
class RuntimeBatchResult:
    available: bool
    predictions: tuple[RuntimePrediction, ...] = ()
    error_code: str | None = None


class LadderModelRuntime:
    """Lazily loads the local speech model without retaining or logging input text."""

    def __init__(self, model_dir: Path | str) -> None:
        self._model_dir = Path(model_dir)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> str | None:
        if self._model is not None:
            return None
        if not self._model_dir.is_dir():
            return "MODEL_NOT_FOUND"
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError:
            return "MODEL_DEPENDENCY_MISSING"
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_dir, local_files_only=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_dir, local_files_only=True
            )
            self._model.eval()
            self._torch = torch
        except (OSError, ValueError):
            self._tokenizer = None
            self._model = None
            self._torch = None
            return "MODEL_LOAD_FAILED"
        return None

    def predict(self, texts: list[str]) -> RuntimeBatchResult:
        if not texts:
            return RuntimeBatchResult(available=True)
        error = self._load()
        if error is not None:
            return RuntimeBatchResult(available=False, error_code=error)
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        try:
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)
            confidence, label_ids = probabilities.max(dim=-1)
            id2label = self._model.config.id2label
            predictions = tuple(
                RuntimePrediction(
                    level=LadderLevel(str(id2label[int(label_id)]).upper()),
                    confidence=float(score),
                )
                for score, label_id in zip(confidence, label_ids, strict=True)
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            return RuntimeBatchResult(available=False, error_code="MODEL_INFERENCE_FAILED")
        return RuntimeBatchResult(available=True, predictions=predictions)
