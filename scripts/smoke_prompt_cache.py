from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mormi_api.llm import UNDERSTANDING_V2_SYSTEM, ClaudeGateway  # noqa: E402
from mormi_api.settings import get_settings  # noqa: E402


def usage_value(usage: object, field: str) -> int:
    return int(getattr(usage, field, 0) or 0)


async def main() -> None:
    settings = get_settings()
    gateway = ClaudeGateway(settings)
    if not gateway.client:
        raise SystemExit("MORMI_ANTHROPIC_API_KEY is required")

    system = gateway._system_with_prompt_cache("understanding_v2", UNDERSTANDING_V2_SYSTEM)
    if isinstance(system, str):
        raise SystemExit(
            "Enable understanding_v2 in MORMI_PROMPT_CACHING_ENABLED and "
            "MORMI_PROMPT_CACHE_STAGES"
        )

    responses = []
    for sequence in (1, 2):
        responses.append(
            await gateway.client.messages.create(
                model=settings.classifier_model,
                max_tokens=8,
                temperature=0,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": f"PII-free prompt-cache smoke request {sequence}",
                    }
                ],
            )
        )

    first_usage = responses[0].usage
    second_usage = responses[1].usage
    first_write = usage_value(first_usage, "cache_creation_input_tokens")
    first_read = usage_value(first_usage, "cache_read_input_tokens")
    second_read = usage_value(second_usage, "cache_read_input_tokens")

    print(
        "prompt_cache_smoke "
        f"first_write_tokens={first_write} first_read_tokens={first_read} "
        f"second_read_tokens={second_read}"
    )
    if first_write <= 0 and first_read <= 0:
        raise SystemExit("first call neither created nor reused a prompt cache entry")
    if second_read <= 0:
        raise SystemExit("second call did not reuse the prompt cache entry")


if __name__ == "__main__":
    asyncio.run(main())
