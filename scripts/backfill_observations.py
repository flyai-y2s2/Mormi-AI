"""Backfill legacy turns without guessing fields that were never collected."""

from __future__ import annotations

import asyncio

from mormi_api.db import Database
from mormi_api.repository import Repository
from mormi_api.security import TextCipher
from mormi_api.settings import get_settings


async def run() -> None:
    settings = get_settings()
    database = Database(settings.database_url)
    repository = Repository(
        database,
        TextCipher(settings.raw_data_encryption_key),
        idempotency_retention_days=settings.idempotency_retention_days,
        classifier_model=settings.classifier_model,
        speaker_model=settings.speaker_model,
    )
    try:
        count = await repository.backfill_historical_observations()
        print(f"backfilled_observations={count}")
    finally:
        await database.dispose()


if __name__ == "__main__":
    asyncio.run(run())
