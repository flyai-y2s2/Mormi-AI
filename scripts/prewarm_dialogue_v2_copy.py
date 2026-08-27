from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mormi_api.copy_cache import GeneratedCopyCacheRepository  # noqa: E402
from mormi_api.db import Database  # noqa: E402
from mormi_api.dialogue_v2_copy import (  # noqa: E402
    PrewarmReportV2,
    StableCopyResolverV2,
    prewarm_required_home_copy_v2,
)
from mormi_api.llm import ClaudeGateway  # noqa: E402
from mormi_api.settings import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prewarm all stable-copy slots in the nine required V2 home packs"
    )
    parser.add_argument(
        "--database-url",
        help="override MORMI_DATABASE_URL; the migrated cache table must already exist",
    )
    return parser.parse_args()


async def _run(settings: Settings) -> PrewarmReportV2:
    database = Database(settings.database_url)
    repository = GeneratedCopyCacheRepository(
        database,
        lease_seconds=settings.stable_copy_cache_lease_seconds,
        retry_base_seconds=settings.stable_copy_cache_retry_base_seconds,
        retry_max_seconds=settings.stable_copy_cache_retry_max_seconds,
    )
    gateway = ClaudeGateway(settings)
    resolver = StableCopyResolverV2(repository, gateway, settings)
    try:
        return await prewarm_required_home_copy_v2(resolver, repository)
    finally:
        if gateway.client is not None:
            await gateway.client.close()
        await database.dispose()


def main() -> int:
    args = parse_args()
    settings = (
        Settings(database_url=args.database_url)
        if args.database_url
        else Settings()
    )
    settings.validate_runtime_safety()
    report = asyncio.run(_run(settings))
    print(report.as_json())
    return 0 if report.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
