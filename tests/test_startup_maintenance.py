from __future__ import annotations

import pytest

from mormi_api.main import run_startup_maintenance
from mormi_api.settings import Settings


class RecordingDatabase:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def create_schema(self) -> None:
        self.calls.append("create_schema")


class RecordingRepository:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def migrate_existing_storage_to_permanent(self) -> None:
        self.calls.append("migrate_existing_storage_to_permanent")

    async def purge_expired_raw_data(self) -> None:
        self.calls.append("purge_expired_raw_data")


@pytest.mark.asyncio
async def test_startup_maintenance_keeps_the_existing_write_order_by_default() -> None:
    calls: list[str] = []

    await run_startup_maintenance(
        RecordingDatabase(calls),  # type: ignore[arg-type]
        RecordingRepository(calls),  # type: ignore[arg-type]
        skip=False,
    )

    assert calls == [
        "create_schema",
        "migrate_existing_storage_to_permanent",
        "purge_expired_raw_data",
    ]


@pytest.mark.asyncio
async def test_read_only_startup_skips_all_startup_write_routines() -> None:
    calls: list[str] = []

    await run_startup_maintenance(
        RecordingDatabase(calls),  # type: ignore[arg-type]
        RecordingRepository(calls),  # type: ignore[arg-type]
        skip=Settings(skip_startup_maintenance=True).skip_startup_maintenance,
    )

    assert calls == []
