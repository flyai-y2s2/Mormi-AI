from __future__ import annotations

from types import SimpleNamespace

import pytest

from mormi_api import main
from mormi_api.settings import Settings, get_settings


class RecordingDatabase:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.engine = RecordingEngine(calls)

    async def create_schema(self) -> None:
        self.calls.append("create_schema")


class RecordingConnection:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def __aenter__(self) -> RecordingConnection:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def run_sync(self, _: object) -> None:
        self.calls.append("require_observation_schema")


class RecordingEngine:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def connect(self) -> RecordingConnection:
        return RecordingConnection(self.calls)


class RecordingRepository:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def migrate_existing_storage_to_permanent(self) -> None:
        self.calls.append("migrate_existing_storage_to_permanent")

    async def migrate_existing_storage_to_plaintext(self) -> None:
        self.calls.append("migrate_existing_storage_to_plaintext")

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
        "require_observation_schema",
        "migrate_existing_storage_to_permanent",
        "migrate_existing_storage_to_plaintext",
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
            None,
            [
                "database",
                "gateway",
                "repository",
                "create_schema",
                "require_observation_schema",
                "migrate_existing_storage_to_permanent",
                "migrate_existing_storage_to_plaintext",
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
    skip_startup_maintenance: bool | None,
    expected_calls: list[str],
) -> None:
    calls: list[str] = []

    class LifespanDatabase:
        def __init__(self, _: str) -> None:
            calls.append("database")
            self.engine = RecordingEngine(calls)

        async def create_schema(self) -> None:
            calls.append("create_schema")

        async def dispose(self) -> None:
            calls.append("dispose")

    class LifespanRepository:
        def __init__(self, *_: object, **__: object) -> None:
            calls.append("repository")

        async def migrate_existing_storage_to_permanent(self) -> None:
            calls.append("migrate_existing_storage_to_permanent")

        async def migrate_existing_storage_to_plaintext(self) -> None:
            calls.append("migrate_existing_storage_to_plaintext")

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
    settings_kwargs = (
        {"skip_startup_maintenance": skip_startup_maintenance}
        if skip_startup_maintenance is not None
        else {}
    )
    settings = Settings(
        environment="development",
        database_url="sqlite+aiosqlite:///./test.db",
        _env_file=None,
        **settings_kwargs,
    )
    assert settings.skip_startup_maintenance is (skip_startup_maintenance is True)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    app = SimpleNamespace(state=SimpleNamespace())
    async with main.lifespan(app):
        assert app.state.settings.skip_startup_maintenance is (skip_startup_maintenance is True)

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


def test_startup_maintenance_defaults_to_false_when_the_flag_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MORMI_SKIP_STARTUP_MAINTENANCE", raising=False)

    settings = Settings(_env_file=None)

    assert settings.skip_startup_maintenance is False


def test_production_rejects_skip_startup_maintenance() -> None:
    settings = Settings(environment="production", skip_startup_maintenance=True)

    with pytest.raises(RuntimeError, match="MORMI_SKIP_STARTUP_MAINTENANCE"):
        settings.validate_runtime_safety()
