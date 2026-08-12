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
    raw_data_encryption_key: str | None = None
    service_api_key: str | None = None
    idempotency_retention_days: int = Field(default=30, ge=1, le=90)
    cors_origins: list[str] = ["http://localhost:3000"]
    show_internal_pedagogy: bool = False

    @field_validator(
        "anthropic_api_key", "raw_data_encryption_key", "service_api_key", mode="before"
    )
    @classmethod
    def empty_string_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def production(self) -> bool:
        return self.environment.lower() == "production"

    def validate_runtime_safety(self) -> None:
        if self.production and not self.raw_data_encryption_key:
            raise RuntimeError("MORMI_RAW_DATA_ENCRYPTION_KEY is required in production")
        if self.production and self.database_url.startswith("sqlite"):
            raise RuntimeError("A PostgreSQL database is required in production")
        if self.production and not self.service_api_key:
            raise RuntimeError("MORMI_SERVICE_API_KEY is required in production")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_runtime_safety()
    return settings
