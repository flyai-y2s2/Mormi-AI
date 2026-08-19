"""Apply additive AI database migrations without discarding legacy rows."""

from __future__ import annotations

import os
from pathlib import Path

from mormi_api.migrations import apply_database_migrations
from mormi_api.settings import Settings


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings()
    raw_url = os.getenv("MORMI_DATABASE_URL", settings.database_url)
    apply_database_migrations(raw_url, root)


if __name__ == "__main__":
    main()
