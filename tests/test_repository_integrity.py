from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from mormi_api.db import Database, DialogueClaimRecord
from mormi_api.repository import _is_duplicate_response_integrity_error


@pytest.mark.asyncio
async def test_sqlite_enforces_observation_claim_foreign_key(tmp_path: object) -> None:
    """Local tests must fail on the same missing-parent write as PostgreSQL."""

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/foreign-keys.db")
    await database.create_schema()

    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        async with database.sessions() as db:
            db.add(
                DialogueClaimRecord(
                    observation_id="missing-observation",
                    slot_id="answer",
                    semantic_role="conclusion",
                    value_json="3",
                    factual=True,
                    validation_status="verified",
                    evidence_span_encrypted="plain:3개",
                    newly_verified=True,
                )
            )
            await db.commit()

    await database.dispose()


@pytest.mark.parametrize(
    "message",
    [
        "UNIQUE constraint failed: turns.conversation_id, turns.response_id",
        (
            "duplicate key value violates unique constraint "
            '"uq_observation_conversation_response"'
        ),
    ],
)
def test_duplicate_response_constraints_are_idempotency_conflicts(message: str) -> None:
    error = IntegrityError("INSERT", {}, RuntimeError(message))

    assert _is_duplicate_response_integrity_error(error) is True


def test_foreign_key_failure_is_not_misclassified_as_duplicate_response() -> None:
    error = IntegrityError(
        "INSERT",
        {},
        RuntimeError(
            "insert on dialogue_claims violates foreign key constraint "
            '"dialogue_claims_observation_id_fkey"'
        ),
    )

    assert _is_duplicate_response_integrity_error(error) is False
