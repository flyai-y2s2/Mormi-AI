from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from .dataset import LadderExample, build_training_examples, split_by_learner
from .training_data import as_training_record


class DatasetSummary(BaseModel):
    total: int
    validated_count: int
    synthetic_count: int
    label_counts: dict[str, int]
    response_mode_counts: dict[str, int]
    split_counts: dict[str, int]
    learner_counts: dict[str, int]
    source_manifest_sha256: str
    rubric_version: str = "ladder-label-v1"
    seed: int


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, examples: list[LadderExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(as_training_record(example), ensure_ascii=False) + "\n")


def prepare_dataset(
    *,
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    hmac_salt: bytes,
    target_per_level: int = 100,
    seed: int = 20260823,
) -> DatasetSummary:
    manifest_bytes = manifest_path.read_bytes()
    manifest_rows = _read_jsonl(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failed_session_ids = {
        str(row["learning_session_id"])
        for row in audit.get("failed_sessions", [])
        if isinstance(row, dict) and row.get("learning_session_id")
    }
    examples = build_training_examples(
        manifest_rows,
        failed_session_ids=failed_session_ids,
        hmac_salt=hmac_salt,
        target_per_level=target_per_level,
        seed=seed,
    )
    split = split_by_learner(examples, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "all.jsonl", examples)
    _write_jsonl(output_dir / "train.jsonl", split.train)
    _write_jsonl(output_dir / "validation.jsonl", split.validation)
    _write_jsonl(output_dir / "test.jsonl", split.test)

    learner_counts = {
        "train": len({row.learner_key for row in split.train}),
        "validation": len({row.learner_key for row in split.validation}),
        "test": len({row.learner_key for row in split.test}),
    }
    summary = DatasetSummary(
        total=len(examples),
        validated_count=sum(not row.synthetic for row in examples),
        synthetic_count=sum(row.synthetic for row in examples),
        label_counts=dict(sorted(Counter(row.target_level.value for row in examples).items())),
        response_mode_counts=dict(
            sorted(Counter(row.response_mode.value for row in examples).items())
        ),
        split_counts={
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        learner_counts=learner_counts,
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        seed=seed,
    )
    (output_dir / "dataset-summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split-manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "rubric_version": summary.rubric_version,
                "source_manifest_sha256": summary.source_manifest_sha256,
                "learners": {
                    "train": sorted({row.learner_key for row in split.train}),
                    "validation": sorted({row.learner_key for row in split.validation}),
                    "test": sorted({row.learner_key for row in split.test}),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary

