from __future__ import annotations

import asyncio
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

    async def run_sync(self, _: object) -> str:
        self.calls.append("require_observation_schema")
        return "final"


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


@pytest.mark.asyncio
async def test_raw_retention_maintenance_repeats_purge_after_startup() -> None:
    stop_event = asyncio.Event()
    calls = 0

    class RetentionRepository:
        async def purge_expired_raw_data(self) -> None:
            nonlocal calls
            calls += 1
            stop_event.set()

    await asyncio.wait_for(
        main.run_raw_retention_maintenance(
            RetentionRepository(),  # type: ignore[arg-type]
            stop_event,
            interval_seconds=0.001,
        ),
        timeout=1,
    )

    assert calls == 1


def test_transition_identity_schema_forces_v2_canary_to_zero() -> None:
    with pytest.raises(RuntimeError, match="canary must remain 0"):
        main.validate_conversation_identity_rollout(
            "transition",
            dialogue_v2_canary_percent=1,
        )

    main.validate_conversation_identity_rollout(
        "transition",
        dialogue_v2_canary_percent=0,
    )
    main.validate_conversation_identity_rollout(
        "final",
        dialogue_v2_canary_percent=100,
    )


@pytest.mark.parametrize(
    ("skip_startup_maintenance", "expected_calls"),
    [
        (
            None,
            [
                "v2_catalog",
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
        (
            True,
            ["v2_catalog", "database", "gateway", "repository", "engine", "service", "dispose"],
        ),
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
        def __init__(self, *_: object, **__: object) -> None:
            calls.append("service")

    monkeypatch.setattr(main, "Database", LifespanDatabase)
    monkeypatch.setattr(main, "Repository", LifespanRepository)
    monkeypatch.setattr(main, "ClaudeGateway", LifespanGateway)
    monkeypatch.setattr(main, "ConversationEngine", LifespanEngine)
    monkeypatch.setattr(main, "ConversationService", LifespanService)
    monkeypatch.setattr(
        main,
        "load_required_home_content_catalog_v2",
        lambda: calls.append("v2_catalog"),
    )
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


def test_stable_copy_validator_default_invalidates_pre_firewall_artifacts() -> None:
    settings = Settings(_env_file=None)

    assert settings.stable_copy_validator_version == "stable-copy-validator-v2"


def test_production_rejects_skip_startup_maintenance() -> None:
    settings = Settings(environment="production", skip_startup_maintenance=True)

    with pytest.raises(RuntimeError, match="MORMI_SKIP_STARTUP_MAINTENANCE"):
        settings.validate_runtime_safety()
