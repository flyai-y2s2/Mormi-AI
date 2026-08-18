from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MORMI_",
        extra="ignore",
    )

    environment: str = "development"
    database_url: str = "sqlite+aiosqlite:///./data/mormi.db"
    anthropic_api_key: str | None = None
    classifier_model: str = "claude-haiku-4-5-20251001"
    speaker_model: str = "claude-sonnet-4-6"
    speaker_timeout_seconds: float = Field(default=8.0, ge=0.5, le=30)
    speaker_verifier_enabled: bool = True
    speaker_verifier_timeout_seconds: float = Field(default=1.8, ge=0.2, le=10)
    raw_data_encryption_key: str | None = None
    service_api_key: str | None = None
    observation_ingest_url: str | None = None
    observation_ingest_key: str | None = None
    outbox_poll_interval_seconds: float = Field(default=2.0, ge=0.1, le=60)
    outbox_batch_size: int = Field(default=20, ge=1, le=200)
    outbox_request_timeout_seconds: float = Field(default=5.0, ge=0.5, le=30)
    outbox_retry_base_seconds: float = Field(default=2.0, ge=0.1, le=300)
    outbox_retry_max_seconds: float = Field(default=300.0, ge=1, le=3600)
    outbox_lease_seconds: float = Field(default=30.0, ge=5, le=600)
    idempotency_retention_days: int = Field(default=30, ge=1, le=90)
    cors_origins: list[str] = ["http://localhost:3000"]
    show_internal_pedagogy: bool = False

    @field_validator(
        "anthropic_api_key",
        "raw_data_encryption_key",
        "service_api_key",
        "observation_ingest_url",
        "observation_ingest_key",
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

    def validate_runtime_safety(self) -> None:
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


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime_safety()
    return settings
