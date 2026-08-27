"""Exercise the production Sonnet and Haiku structured-output contracts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mormi_api.dialogue_v2_model_smoke import (  # noqa: E402
    run_dialogue_v2_model_smoke,
    safe_model_smoke_error_code,
)
from mormi_api.llm import ClaudeGateway  # noqa: E402
from mormi_api.settings import Settings  # noqa: E402


async def _run() -> int:
    settings = Settings()
    settings.validate_runtime_safety()
    gateway = ClaudeGateway(settings)
    try:
        async with asyncio.timeout(
            settings.classifier_timeout_seconds
            + settings.speaker_timeout_seconds
            + 1
        ):
            report = await run_dialogue_v2_model_smoke(
                gateway,
                classifier_model=settings.classifier_model,
                speaker_model=settings.speaker_model,
            )
    except Exception as error:
        print(
            json.dumps(
                {
                    "succeeded": False,
                    "error_type": type(error).__name__,
                    "error_code": safe_model_smoke_error_code(error),
                },
                sort_keys=True,
            )
        )
        return 1
    finally:
        if gateway.client is not None:
            await gateway.client.close()
    print(report.model_dump_json())
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
