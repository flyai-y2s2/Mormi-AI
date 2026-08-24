from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from mormi_api.ladder_model.dataset import LadderLevel
from mormi_api.ladder_model.runtime import RuntimeBatchResult, RuntimePrediction


def smoke_module() -> ModuleType:
    try:
        return importlib.import_module("mormi_api.ladder_model.smoke")
    except ModuleNotFoundError:
        pytest.fail("ladder runtime smoke module is not implemented")


class AvailableRuntime:
    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir)

    @property
    def model_version(self) -> str:
        return "ladder-speech-klue-v2"

    def predict(self, texts: list[str]) -> RuntimeBatchResult:
        assert texts == ["나는 수를 세어서 답을 찾았어."]
        return RuntimeBatchResult(
            available=True,
            predictions=(RuntimePrediction(level=LadderLevel.L3, confidence=0.75),),
        )


def write_model_fixture(
    root: Path,
    *,
    expected_hash: str | None = None,
    model_version: str = "ladder-speech-klue-v2",
) -> Path:
    model_dir = root / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"verified-weight")
    actual_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
    (root / "model-manifest.json").write_text(
        json.dumps(
            {
                "model_version": model_version,
                "label_order": ["L2", "L3", "L4"],
                "weight_file": "model.safetensors",
                "weight_sha256": expected_hash or actual_hash,
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def test_check_model_rejects_missing_manifest(tmp_path: Path) -> None:
    smoke = smoke_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(RuntimeError, match="MODEL_MANIFEST_NOT_FOUND"):
        smoke.check_model(model_dir, runtime_factory=AvailableRuntime)


def test_check_model_rejects_weight_checksum_mismatch(tmp_path: Path) -> None:
    smoke = smoke_module()
    model_dir = write_model_fixture(tmp_path, expected_hash="0" * 64)

    with pytest.raises(RuntimeError, match="MODEL_CHECKSUM_MISMATCH"):
        smoke.check_model(model_dir, runtime_factory=AvailableRuntime)


def test_check_model_rejects_unexpected_model_version(tmp_path: Path) -> None:
    smoke = smoke_module()
    model_dir = write_model_fixture(tmp_path, model_version="ladder-speech-v1")

    with pytest.raises(RuntimeError, match="MODEL_VERSION_INVALID"):
        smoke.check_model(model_dir, runtime_factory=AvailableRuntime)


def test_check_model_accepts_verified_bounded_prediction(tmp_path: Path) -> None:
    smoke = smoke_module()
    model_dir = write_model_fixture(tmp_path)

    result = smoke.check_model(model_dir, runtime_factory=AvailableRuntime)

    assert result.model_version == "ladder-speech-klue-v2"
    assert result.level is LadderLevel.L3
    assert result.confidence == pytest.approx(0.75)
    assert smoke.success_message(result) == (
        "MODEL_OK version=ladder-speech-klue-v2 level=L3 confidence=0.7500"
    )


def test_success_output_never_contains_smoke_input(tmp_path: Path) -> None:
    smoke = smoke_module()
    model_dir = write_model_fixture(tmp_path)

    result = smoke.check_model(model_dir, runtime_factory=AvailableRuntime)

    assert "나는 수를" not in smoke.success_message(result)
