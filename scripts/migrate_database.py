"""Apply rollout-safe AI database migrations without discarding legacy rows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mormi_api.migrations import apply_database_migrations  # noqa: E402
from mormi_api.settings import Settings  # noqa: E402


def main() -> None:
    settings = Settings()
    raw_url = os.getenv("MORMI_DATABASE_URL", settings.database_url)
    target_revision = os.getenv("MORMI_DATABASE_MIGRATION_TARGET", "head")
    phase = apply_database_migrations(
        raw_url,
        ROOT,
        target_revision=target_revision,
    )
    print(f"conversation_identity_schema_phase={phase}")


if __name__ == "__main__":
    main()
