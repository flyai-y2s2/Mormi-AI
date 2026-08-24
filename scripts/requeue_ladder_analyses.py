from __future__ import annotations

import argparse
import asyncio

from mormi_api.db import Database
from mormi_api.ladder_analysis_repository import LadderAnalysisRepository
from mormi_api.settings import get_settings


async def run(*, confirmed: bool) -> int:
    if not confirmed:
        raise SystemExit("Use --confirm to requeue model-configuration failures")
    settings = get_settings()
    database = Database(settings.database_url)
    store = LadderAnalysisRepository(
        database,
        lease_seconds=settings.ladder_analysis_lease_seconds,
    )
    try:
        return await store.requeue_model_failures()
    finally:
        await database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Requeue ladder jobs that failed only because the model was unavailable"
    )
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    count = asyncio.run(run(confirmed=args.confirm))
    print(f"requeued_ladder_analyses={count}")


if __name__ == "__main__":
    main()
