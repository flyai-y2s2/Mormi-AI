from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schemas import DialogueRuntimeContractVersion

EffortLevel = Literal["low", "medium", "high", "max"]
PromptCacheStage = Literal["understanding_v2", "speaker_v2"]
PromptCacheTtl = Literal["5m", "1h"]


def _default_prompt_cache_stages() -> frozenset[PromptCacheStage]:
    return frozenset({"understanding_v2"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MORMI_",
        extra="ignore",
    )

    environment: str = "development"
    database_url: str = "sqlite+aiosqlite:///./data/mormi.db"
    anthropic_api_key: str | None = None
    classifier_model: str = "claude-sonnet-4-6"
    classifier_effort: EffortLevel = "low"
    bridge_model: str = "claude-haiku-4-5-20251001"
    speaker_model: str = "claude-haiku-4-5-20251001"
    speaker_effort: EffortLevel = "low"
    star_note_model: str = "claude-haiku-4-5-20251001"
    prompt_caching_enabled: bool = False
    prompt_cache_ttl: PromptCacheTtl = "5m"
    prompt_cache_stages: frozenset[PromptCacheStage] = Field(
        default_factory=_default_prompt_cache_stages
    )
    # Teacher-facing summaries are not Mormi dialogue. Keep their existing
    # Sonnet model independent from the child-facing speaker selection.
    report_model: str = "claude-sonnet-4-6"
    # This is an enum-valued feature flag. Only newly created conversations
    # receive the configured value; existing conversations use their pinned
    # state snapshot.
    runtime_contract_version: DialogueRuntimeContractVersion = (
        DialogueRuntimeContractVersion.LEGACY_V1
    )
    # V2 is assigned only to new conversations backed by a native home pack or
    # a reviewed cafe/amusement scenario pack. A stable server-side hash selects
    # the configured percentage and the chosen runtime is persisted in state.
    dialogue_v2_canary_percent: int = Field(default=0, ge=0, le=100)
    dialogue_v2_canary_salt: str = Field(
        default="mormi-dialogue-v2-default",
        min_length=8,
        max_length=200,
    )
    stable_copy_model: str = "claude-sonnet-4-6"
    # Independent from legacy/V2 selection. Only newly created V2 conversations
    # enroll; the enabled flag is a turn-only emergency bypass for enrolled ones.
    session_parent_graph_enabled: bool = False
    session_parent_graph_canary_percent: int = Field(default=0, ge=0, le=100)
    session_parent_store_timeout_seconds: float = Field(default=0.5, ge=0.05, le=5)
    stable_copy_effort: EffortLevel = "low"
    stable_copy_timeout_seconds: float = Field(default=8.0, ge=0.5, le=30)
    stable_copy_prompt_version: str = Field(default="stable-copy-v1", min_length=1)
    stable_copy_schema_version: str = Field(default="stable-copy-output-v1", min_length=1)
    stable_copy_validator_version: str = Field(default="stable-copy-validator-v2", min_length=1)
    stable_copy_cache_lease_seconds: float = Field(default=30.0, ge=2, le=300)
    stable_copy_cache_retry_base_seconds: float = Field(default=2.0, ge=0.1, le=300)
    stable_copy_cache_retry_max_seconds: float = Field(default=120.0, ge=1, le=3600)
    classifier_timeout_seconds: float = Field(default=15.0, ge=0.5, le=60)
    speaker_timeout_seconds: float = Field(default=10.0, ge=0.5, le=30)
    bridge_timeout_seconds: float = Field(default=4.0, ge=0.5, le=10)
    raw_data_encryption_key: str | None = None
    service_api_key: str | None = None
    skip_startup_maintenance: bool = False
    observation_ingest_url: str | None = None
    observation_ingest_key: str | None = None
    star_note_events_enabled: bool = False
    outbox_poll_interval_seconds: float = Field(default=2.0, ge=0.1, le=60)
    outbox_batch_size: int = Field(default=20, ge=1, le=200)
    outbox_request_timeout_seconds: float = Field(default=5.0, ge=0.5, le=30)
    outbox_retry_base_seconds: float = Field(default=2.0, ge=0.1, le=300)
    outbox_retry_max_seconds: float = Field(default=300.0, ge=1, le=3600)
    outbox_lease_seconds: float = Field(default=30.0, ge=5, le=600)
    idempotency_retention_days: int = Field(default=30, ge=1, le=90)
    cors_origins: list[str] = ["http://localhost:3000"]
    show_internal_pedagogy: bool = False
    ladder_model_dir: Path | None = None
    ladder_analysis_worker_enabled: bool = False
    ladder_analysis_poll_interval_seconds: float = Field(default=2.0, ge=0.1, le=60)
    ladder_analysis_batch_size: int = Field(default=10, ge=1, le=100)
    ladder_analysis_lease_seconds: float = Field(default=60.0, ge=5, le=600)

    @field_validator(
        "anthropic_api_key",
        "raw_data_encryption_key",
        "service_api_key",
        "observation_ingest_url",
        "observation_ingest_key",
        "ladder_model_dir",
        mode="before",
    )
    @classmethod
    def empty_string_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def observation_ingest_enabled(self) -> bool:
        return bool(self.observation_ingest_url and self.observation_ingest_key)

    @property
    def ladder_analysis_enabled(self) -> bool:
        return self.ladder_analysis_worker_enabled and self.ladder_model_dir is not None

    def validate_runtime_safety(self) -> None:
        if (
            self.dialogue_v2_canary_percent
            and self.runtime_contract_version is not DialogueRuntimeContractVersion.VERDICT_V1
        ):
            raise RuntimeError(
                "MORMI_DIALOGUE_V2_CANARY_PERCENT requires "
                "MORMI_RUNTIME_CONTRACT_VERSION=verdict-v1"
            )
        if (
            self.stable_copy_cache_retry_max_seconds
            < self.stable_copy_cache_retry_base_seconds
        ):
            raise RuntimeError(
                "MORMI_STABLE_COPY_CACHE_RETRY_MAX_SECONDS must be greater than or "
                "equal to MORMI_STABLE_COPY_CACHE_RETRY_BASE_SECONDS"
            )
        if self.skip_startup_maintenance and self.environment.lower() not in {
            "local",
            "development",
            "test",
        }:
            raise RuntimeError(
                "MORMI_SKIP_STARTUP_MAINTENANCE is allowed only in local, development, or test"
            )
        if self.production and self.database_url.startswith("sqlite"):
            raise RuntimeError("A PostgreSQL database is required in production")
        if self.production and not self.service_api_key:
            raise RuntimeError("MORMI_SERVICE_API_KEY is required in production")
        if self.observation_ingest_url and not self.observation_ingest_url.startswith(
            ("http://", "https://")
        ):
            raise RuntimeError("MORMI_OBSERVATION_INGEST_URL must use http or https")
        if self.outbox_retry_max_seconds < self.outbox_retry_base_seconds:
            raise RuntimeError(
                "MORMI_OUTBOX_RETRY_MAX_SECONDS must be greater than or equal to "
                "MORMI_OUTBOX_RETRY_BASE_SECONDS"
            )
        if self.outbox_lease_seconds <= self.outbox_request_timeout_seconds:
            raise RuntimeError(
                "MORMI_OUTBOX_LEASE_SECONDS must be greater than "
                "MORMI_OUTBOX_REQUEST_TIMEOUT_SECONDS"
            )
        if self.ladder_analysis_worker_enabled and self.ladder_model_dir is None:
            raise RuntimeError(
                "MORMI_LADDER_MODEL_DIR is required when the ladder analysis worker is enabled"
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime_safety()
    return settings
