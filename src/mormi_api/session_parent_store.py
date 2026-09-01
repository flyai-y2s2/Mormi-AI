"""DB-authoritative session cursors, published only at committed turn boundaries.

No request text, model output, or pedagogical decision is stored here. A failed
publication cannot undo an already committed child response. Concurrent writers
are fenced by the cursor generation AND the canonical conversation version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from .db import ConversationRecord, Database, SessionParentRecord
from .schemas import SessionState, SessionStatus, utc_now

PARENT_GRAPH_VERSION = "session-parent-v1"
ParentPhase = Literal["waiting", "completed"]


@dataclass(frozen=True)
class ParentCursor:
    conversation_id: str
    graph_version: str
    state_version: int
    turn_id: str
    phase: ParentPhase
    generation: int
    checkpoint: dict[str, Any] | None


class SessionParentStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def enroll(self, state: SessionState) -> None:
        """Enroll only the committed winner of a NEW conversation creation."""
        assert state.current_turn_id is not None
        try:
            async with self.database.sessions() as db:
                db.add(
                    SessionParentRecord(
                        conversation_id=state.conversation_id,
                        graph_version=PARENT_GRAPH_VERSION,
                        state_version=state.state_version,
                        turn_id=state.current_turn_id,
                        phase=(
                            "completed" if state.status is SessionStatus.COMPLETED else "waiting"
                        ),
                        generation=1,
                        checkpoint=None,
                    )
                )
                await db.commit()
        except IntegrityError:
            # A retried enrollment must never reset a live parent's checkpoint.
            if await self.load(state.conversation_id) is None:
                raise

    async def load(self, conversation_id: str) -> ParentCursor | None:
        async with self.database.sessions() as db:
            row = await db.get(SessionParentRecord, conversation_id)
            if row is None:
                return None
            if row.phase not in {"waiting", "completed"}:
                raise ValueError("invalid parent cursor phase")
            return ParentCursor(
                conversation_id=row.conversation_id,
                graph_version=row.graph_version,
                state_version=row.state_version,
                turn_id=row.turn_id,
                phase="completed" if row.phase == "completed" else "waiting",
                generation=row.generation,
                checkpoint=row.checkpoint,
            )

    async def publish(
        self,
        cursor: ParentCursor,
        *,
        state_version: int,
        turn_id: str,
        phase: ParentPhase,
        checkpoint: dict[str, Any],
    ) -> bool:
        """Short transaction AFTER domain commit; never hold a lock over an LLM call."""
        async with self.database.sessions() as db:
            conversation = await db.get(
                ConversationRecord, cursor.conversation_id, with_for_update=True
            )
            if (
                conversation is None
                or conversation.state_version != state_version
                or conversation.state_json.get("current_turn_id") != turn_id
                or (conversation.status == SessionStatus.COMPLETED.value) != (phase == "completed")
            ):
                return False
            result = await db.execute(
                update(SessionParentRecord)
                .where(
                    SessionParentRecord.conversation_id == cursor.conversation_id,
                    SessionParentRecord.generation == cursor.generation,
                    SessionParentRecord.graph_version == PARENT_GRAPH_VERSION,
                )
                .values(
                    state_version=state_version,
                    turn_id=turn_id,
                    phase=phase,
                    checkpoint=checkpoint,
                    generation=cursor.generation + 1,
                    updated_at=utc_now(),
                )
            )
            await db.commit()
            return bool(result.rowcount)  # type: ignore[attr-defined]
