from __future__ import annotations

import argparse
import json
from pathlib import Path

from mormi_api.ladder_model.speech_trainer import SpeechTrainConfig, train_speech_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune leakage-resistant L2/L3/L4 model.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/ladder-model/dataset-v2/train.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/ladder-model/dataset-v2/validation.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("artifacts/ladder-model/dataset-v2/test.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/ladder-model/run-v2"))
    parser.add_argument("--base-model", default="klue/roberta-base")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_speech_model(
        SpeechTrainConfig(
            train_path=args.train,
            validation_path=args.validation,
            test_path=args.test,
            output_dir=args.output,
            base_model=args.base_model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            seed=args.seed,
            fp16=args.fp16,
        )
    )
    print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
