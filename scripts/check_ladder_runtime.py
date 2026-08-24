from __future__ import annotations

import argparse
from pathlib import Path

from mormi_api.ladder_model.smoke import check_model, success_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local ladder model runtime")
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check_model(args.model_dir)
    except RuntimeError as error:
        raise SystemExit(f"MODEL_ERROR code={error}") from error
    print(success_message(result))


if __name__ == "__main__":
    main()
