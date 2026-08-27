from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def test_migration_script_imports_from_a_source_checkout_without_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path('scripts/migrate_database.py', run_name='import_only')",
        ],
        cwd=ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_prewarm_script_help_works_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/prewarm_dialogue_v2_copy.py", "--help"],
        cwd=ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Prewarm all stable-copy slots" in result.stdout
