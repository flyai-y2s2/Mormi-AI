from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mormi_api.ladder_model.speech_dataset import prepare_speech_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-resistant speech-only ladder data."
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
        default=Path("artifacts/ladder-model/dataset-v2"),
    )
    parser.add_argument("--train-per-level", type=int, default=60)
    parser.add_argument("--validation-per-level", type=int, default=20)
    parser.add_argument("--test-per-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    salt = os.environ.get("MORMI_LADDER_HMAC_SALT", "").encode("utf-8")
    if len(salt) < 16:
        raise SystemExit("MORMI_LADDER_HMAC_SALT must contain at least 16 characters")
    summary = prepare_speech_dataset(
        manifest_path=args.manifest,
        audit_path=args.audit,
        output_dir=args.output,
        hmac_salt=salt,
        train_per_level=args.train_per_level,
        validation_per_level=args.validation_per_level,
        test_per_level=args.test_per_level,
        seed=args.seed,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
