from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from parity_support import RecordedCalls, assert_same, comparable, payload
from session_parent_support import REFERENCE_PATH, service
from sqlalchemy import select
from test_dialogue_v2_fault_parity import complete_understanding
from test_dialogue_v2_life_runtime import MENU, LifeRuntimeGateway
from test_dialogue_v2_service_routing import _home_request

import mormi_api.schemas as schemas
from mormi_api.content import representative_park_context
from mormi_api.db import Base, Database
from mormi_api.dialogue_v2_content import REQUIRED_HOME_SESSION_IDS
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, InputKind, ResponseType, SessionCreate, SessionStatus
from mormi_api.security import TextCipher
from mormi_api.session_parent_store import SessionParentStore

SCENARIOS = [
    *sorted(REQUIRED_HOME_SESSION_IDS),
    "cafe_queue",
    "cafe_budget_menu",
    "cafe_menu_total",
    "cafe_change",
    "amusement_ticket_multiply",
    "amusement_snack_divide",
    "amusement_pass_compare",
]


def request_for(scenario_id: str) -> SessionCreate:
    if scenario_id in REQUIRED_HOME_SESSION_IDS:
        request = _home_request(learning_session_id=f"parent-{scenario_id}")
        assert request.practice_summary is not None
        request.practice_summary.curriculum_session_id = scenario_id
        return request
    context: dict[str, Any]
    if scenario_id.startswith("amusement"):
        context = {
            "scene": "amusement_park",
            "park_context": representative_park_context(scenario_id),
        }
    elif scenario_id == "cafe_queue":
        context = {"scene": "cafe", "queue_context": {"left_count": 2, "right_count": 4}}
    else:
        context = {
            "scene": "cafe",
            "cafe_context": {
                "menu_items": MENU,
                "mormi_menu_id": "americano",
                "budget": 6000,
            },
        }
    return SessionCreate(
        learner_id=71,
        scenario_id=scenario_id,
        learning_session_id=f"parent-{scenario_id}",
        **context,
    )


@pytest.mark.parametrize("scenario_id", SCENARIOS)
@pytest.mark.parametrize("mode", ["supported", "joint", "direct"])
async def test_independent_service_to_parent_full_session_parity(
    scenario_id: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed = datetime(2026, 8, 31, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(tz: Any) -> datetime:
            return fixed.astimezone(tz)

    monkeypatch.setattr(schemas, "datetime", Clock)
    monkeypatch.setattr(random, "SystemRandom", lambda: random.Random(20260831))

    async def execute(reference: bool, alternate: bool = False) -> Any:
        sequence = count(1)
        monkeypatch.setattr(schemas, "uuid4", lambda: UUID(int=next(sequence)))
        database = Database(f"sqlite+aiosqlite:///{tmp_path}/{reference}-{alternate}.db")
        await database.create_schema()
        repository = Repository(database, TextCipher("synthetic"))
        calls: list[Any] = []

        class DirectGateway(LifeRuntimeGateway):
            async def understand_v2(self, request: Any) -> Any:
                self.understanding_requests.append(request)
                return complete_understanding(request)

        gateway = RecordedCalls(
            DirectGateway() if mode == "direct" else LifeRuntimeGateway(), calls, "model"
        )
        app = service(repository, gateway, reference=reference)
        trace: list[Any] = []
        try:
            envelope = await app.create_conversation(request_for(scenario_id))
            trace.append(payload(envelope))
            saved_responses = []
            for i in range(45):
                if envelope.turn.status is SessionStatus.COMPLETED:
                    break
                # New service/compiled graph each time: no in-process memory can
                # accidentally supply the session state. Also exercise rollback.
                app = service(
                    repository,
                    gateway,
                    reference=reference,
                    parent=not alternate or i % 2 == 0,
                    percent=0,
                )
                state = await repository.get_state(envelope.conversation_id)
                engine = app._engine_for_state(state)
                turn = envelope.turn
                if mode == "direct" and turn.input.kind is InputKind.TEXT:
                    data = {"type": ResponseType.TEXT, "text": "합성 정답과 충분한 설명"}
                elif turn.input.kind is InputKind.JOINT:
                    data = {
                        "type": ResponseType.ACTION,
                        "values": turn.input.config["completion_values"],
                    }
                elif turn.input.kind is InputKind.CHOICES and mode == "supported":
                    pack, _, ledger, _ = engine._resolve_state(state)
                    plan = engine._active_l2_plan(state, pack, ledger)
                    choice_id = next(
                        c.choice_id
                        for c in plan.choices
                        if c.effect.verdict == "correct" and not getattr(c, "disabled", False)
                    )
                    data = {"type": ResponseType.CHOICE, "choice_ids": [choice_id]}
                else:
                    data = {"type": ResponseType.NO_RESPONSE}
                response = ChildResponse(
                    turn_id=turn.turn_id, response_id=UUID(int=10000 + i), **data
                )
                events = [e async for e in app.respond_stream(envelope.conversation_id, response)]
                trace.append(payload(events))
                envelope = events[-1].envelope
                assert envelope is not None
                saved_responses.append((response, envelope))
                assert await app.snapshot(envelope.conversation_id) == envelope
            else:
                raise AssertionError("session did not complete")
            # An earlier response must replay its original result, NOT latest END.
            call_count = len(calls)
            for response, expected in saved_responses:
                assert await app.respond(envelope.conversation_id, response) == expected
            assert len(calls) == call_count
            rows = {}
            async with database.sessions() as db:
                for table in Base.metadata.sorted_tables:
                    if table.name == "dialogue_session_parents":
                        continue  # The only deliberately additive internal data.
                    result = await db.execute(select(table).order_by(*table.primary_key.columns))
                    rows[table.name] = [dict(r) for r in result.mappings()]
            if not reference and not alternate:
                cursor = await SessionParentStore(database).load(envelope.conversation_id)
                assert cursor is not None and cursor.phase == "completed"
                assert cursor.checkpoint is not None
            return comparable({"trace": trace, "rows": rows, "calls": calls})
        finally:
            await database.dispose()

    baseline = await execute(True)
    assert_same(baseline, await execute(False))
    assert_same(baseline, await execute(False, alternate=True))


def test_reference_service_is_frozen() -> None:
    # SHA filled from the independently read develop blob, not new service code.
    assert hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest() == (
        "9fda81073da8f7ddebb1cb969eb012f95378132a6113c00a310b4c9599a43711"
    )
