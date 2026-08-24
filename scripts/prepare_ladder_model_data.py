from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mormi_api.ladder_model.preparation import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare anonymized speaking-ladder train/validation/test JSONL files."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("../scripts/ladder-dataset-manifest/ladder-training-dataset.jsonl"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("../scripts/ladder-dataset-manifest/database-audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ladder-model/dataset"),
    )
    parser.add_argument("--target-per-level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    salt = os.environ.get("MORMI_LADDER_HMAC_SALT", "").encode("utf-8")
    if len(salt) < 16:
        raise SystemExit("MORMI_LADDER_HMAC_SALT must contain at least 16 characters")
    summary = prepare_dataset(
        manifest_path=args.manifest,
        audit_path=args.audit,
        output_dir=args.output,
        hmac_salt=salt,
        target_per_level=args.target_per_level,
        seed=args.seed,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
