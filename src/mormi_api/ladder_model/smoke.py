from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .dataset import LadderLevel
from .runtime import LadderModelRuntime, RuntimeBatchResult

SMOKE_TEXT = "나는 수를 세어서 답을 찾았어."
EXPECTED_MODEL_VERSION = "ladder-speech-klue-v2"


class RuntimeLike(Protocol):
    @property
    def model_version(self) -> str: ...

    def predict(self, texts: list[str]) -> RuntimeBatchResult: ...


@dataclass(frozen=True)
class ModelCheckResult:
    model_version: str
    level: LadderLevel
    confidence: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(
    model_dir: Path | str,
    *,
    runtime_factory: Callable[[Path | str], RuntimeLike] = LadderModelRuntime,
) -> ModelCheckResult:
    directory = Path(model_dir)
    manifest_path = directory.parent / "model-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("MODEL_MANIFEST_NOT_FOUND")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("MODEL_MANIFEST_INVALID") from error
    if manifest.get("model_version") != EXPECTED_MODEL_VERSION:
        raise RuntimeError("MODEL_VERSION_INVALID")
    if manifest.get("label_order") != ["L2", "L3", "L4"]:
        raise RuntimeError("MODEL_LABEL_ORDER_INVALID")
    weight_name = manifest.get("weight_file")
    expected_hash = manifest.get("weight_sha256")
    if not isinstance(weight_name, str) or not isinstance(expected_hash, str):
        raise RuntimeError("MODEL_MANIFEST_INVALID")
    weight_path = directory / weight_name
    if not weight_path.is_file():
        raise RuntimeError("MODEL_WEIGHT_NOT_FOUND")
    if _sha256(weight_path) != expected_hash.lower():
        raise RuntimeError("MODEL_CHECKSUM_MISMATCH")

    runtime = runtime_factory(directory)
    prediction = runtime.predict([SMOKE_TEXT])
    if not prediction.available or len(prediction.predictions) != 1:
        raise RuntimeError(prediction.error_code or "MODEL_INFERENCE_FAILED")
    item = prediction.predictions[0]
    return ModelCheckResult(
        model_version=runtime.model_version,
        level=item.level,
        confidence=item.confidence,
    )


def success_message(result: ModelCheckResult) -> str:
    return (
        f"MODEL_OK version={result.model_version} "
        f"level={result.level.value} confidence={result.confidence:.4f}"
    )
