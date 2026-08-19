from __future__ import annotations

from types import SimpleNamespace

import pytest

from mormi_api import main
from mormi_api.settings import Settings, get_settings


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

    await main.run_startup_maintenance(
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

    await main.run_startup_maintenance(
        RecordingDatabase(calls),  # type: ignore[arg-type]
        RecordingRepository(calls),  # type: ignore[arg-type]
        skip=Settings(skip_startup_maintenance=True).skip_startup_maintenance,
    )

    assert calls == []


@pytest.mark.parametrize(
    ("skip_startup_maintenance", "expected_calls"),
    [
        (
            False,
            [
                "database",
                "gateway",
                "repository",
                "create_schema",
                "migrate_existing_storage_to_permanent",
                "purge_expired_raw_data",
                "engine",
                "service",
                "dispose",
            ],
        ),
        (True, ["database", "gateway", "repository", "engine", "service", "dispose"]),
    ],
)
async def test_lifespan_respects_the_startup_maintenance_setting(
    monkeypatch: pytest.MonkeyPatch,
    skip_startup_maintenance: bool,
    expected_calls: list[str],
) -> None:
    calls: list[str] = []

    class LifespanDatabase:
        def __init__(self, _: str) -> None:
            calls.append("database")

        async def create_schema(self) -> None:
            calls.append("create_schema")

        async def dispose(self) -> None:
            calls.append("dispose")

    class LifespanRepository:
        def __init__(self, *_: object, **__: object) -> None:
            calls.append("repository")

        async def migrate_existing_storage_to_permanent(self) -> None:
            calls.append("migrate_existing_storage_to_permanent")

        async def purge_expired_raw_data(self) -> None:
            calls.append("purge_expired_raw_data")

    class LifespanGateway:
        def __init__(self, _: Settings) -> None:
            calls.append("gateway")

    class LifespanEngine:
        def __init__(self, *_: object, **__: object) -> None:
            calls.append("engine")

    class LifespanService:
        def __init__(self, *_: object) -> None:
            calls.append("service")

    monkeypatch.setattr(main, "Database", LifespanDatabase)
    monkeypatch.setattr(main, "Repository", LifespanRepository)
    monkeypatch.setattr(main, "ClaudeGateway", LifespanGateway)
    monkeypatch.setattr(main, "ConversationEngine", LifespanEngine)
    monkeypatch.setattr(main, "ConversationService", LifespanService)
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(
            environment="development",
            database_url="sqlite+aiosqlite:///./test.db",
            skip_startup_maintenance=skip_startup_maintenance,
        ),
    )

    app = SimpleNamespace(state=SimpleNamespace())
    async with main.lifespan(app):
        assert app.state.settings.skip_startup_maintenance is skip_startup_maintenance

    assert calls == expected_calls


def test_startup_maintenance_flag_parses_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MORMI_ENVIRONMENT", "development")
    monkeypatch.setenv("MORMI_SKIP_STARTUP_MAINTENANCE", "true")
    get_settings.cache_clear()
    try:
        assert get_settings().skip_startup_maintenance is True
    finally:
        get_settings.cache_clear()


def test_production_rejects_skip_startup_maintenance() -> None:
    settings = Settings(environment="production", skip_startup_maintenance=True)

    with pytest.raises(RuntimeError, match="MORMI_SKIP_STARTUP_MAINTENANCE"):
        settings.validate_runtime_safety()
