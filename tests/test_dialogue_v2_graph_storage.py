from __future__ import annotations

import random
from datetime import UTC, datetime
from itertools import count
from typing import Any
from uuid import UUID

import pytest
from conftest import FakeGateway
from parity_support import HOME, LIFE, assert_same, comparable, payload
from sqlalchemy import select
from test_dialogue_v2_life_runtime import MENU, LifeRuntimeGateway, _correct_choice_id

import mormi_api.schemas as schemas
from mormi_api.content import representative_park_context
from mormi_api.db import Base, Database
from mormi_api.dialogue_v2_life_runtime import DialogueV2LifeEngine
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    DialogueRuntimeContractVersion,
    InputKind,
    ResponseType,
    SessionCreate,
    SessionStatus,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService

SCENARIOS = [
    "cafe_queue",
    "cafe_budget_menu",
    "cafe_menu_total",
    "cafe_change",
    "amusement_ticket_multiply",
    "amusement_snack_divide",
    "amusement_pass_compare",
]


@pytest.mark.parametrize("scenario_id", SCENARIOS)
async def test_real_service_storage_and_replay_match_old_engine(
    scenario_id: str,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = datetime(2026, 8, 31, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(tz: Any) -> datetime:
            return fixed.astimezone(tz)

    monkeypatch.setattr(schemas, "datetime", Clock)
    monkeypatch.setattr(random, "SystemRandom", lambda: random.Random(20260831))

    async def execute(graph: bool, alternate: bool = False) -> Any:
        sequence = count(1)
        monkeypatch.setattr(schemas, "uuid4", lambda: UUID(int=next(sequence)))
        database = Database(f"sqlite+aiosqlite:///{tmp_path}/{graph}-{alternate}.db")
        await database.create_schema()
        repository = Repository(database, TextCipher("synthetic-parity-key"))
        gateway = LifeRuntimeGateway()
        home = (DialogueV2Engine if graph else HOME.DialogueV2Engine)(gateway)
        life = (DialogueV2LifeEngine if graph else LIFE.DialogueV2LifeEngine)(gateway)
        if graph:
            home.run_turn_stream = home._run_turn_graph
            life.run_turn_stream = life._run_turn_graph
        service = ConversationService(
            repository,
            ConversationEngine(FakeGateway()),
            v2_engine=home,
            life_v2_engine=life,
            runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
            dialogue_v2_canary_percent=100,
            dialogue_v2_canary_salt="parity",
        )
        old_home = HOME.DialogueV2Engine(gateway)
        old_life = LIFE.DialogueV2LifeEngine(gateway)
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
        request = SessionCreate(
            learner_id=801,
            scenario_id=scenario_id,
            learning_session_id=f"parity-{scenario_id}",
            **context,
        )
        trace = []
        try:
            envelope = await service.create_conversation(request)
            trace.append(payload(envelope))
            for i in range(40):
                if alternate:
                    # Each engine reads snapshots persisted by the other, including
                    # partial progress, task transitions and note-emission markers.
                    service.v2_engine = home if i % 2 == 0 else old_home
                    service.life_v2_engine = life if i % 2 == 0 else old_life
                turn = envelope.turn
                if turn.status is SessionStatus.COMPLETED:
                    break
                state = await repository.get_state(envelope.conversation_id)
                if turn.input.kind is InputKind.CHOICES:
                    response = ChildResponse(
                        turn_id=turn.turn_id,
                        response_id=UUID(int=1000 + i),
                        type=ResponseType.CHOICE,
                        choice_ids=[_correct_choice_id(life, state)],
                    )
                elif turn.input.kind is InputKind.JOINT:
                    response = ChildResponse(
                        turn_id=turn.turn_id,
                        response_id=UUID(int=1000 + i),
                        type=ResponseType.ACTION,
                        values=turn.input.config["completion_values"],
                    )
                else:
                    response = ChildResponse(
                        turn_id=turn.turn_id,
                        response_id=UUID(int=1000 + i),
                        type=ResponseType.NO_RESPONSE,
                    )
                events = [
                    event
                    async for event in service.respond_stream(envelope.conversation_id, response)
                ]
                trace.append(payload(events))
                envelope = events[-1].envelope
                assert envelope is not None
                # Idempotent replay must not produce progress, duplicate notes or advance state.
                replay = [
                    event
                    async for event in service.respond_stream(envelope.conversation_id, response)
                ]
                assert len(replay) == 1 and replay[0].replayed is True
                assert replay[0].envelope == envelope
                assert (await service.snapshot(envelope.conversation_id)).turn == envelope.turn
            else:
                raise AssertionError("scenario did not complete")
            rows = {}
            async with database.sessions() as db:
                for table in Base.metadata.sorted_tables:
                    result = await db.execute(select(table).order_by(*table.primary_key.columns))
                    rows[table.name] = [dict(row) for row in result.mappings()]
            assert len(rows["star_notes"] if "star_notes" in rows else rows["notes"]) == 1
            return comparable(
                {"trace": trace, "rows": rows, "speaker_plans": gateway.speaker_plans}
            )
        finally:
            await database.dispose()

    baseline = await execute(False)
    assert_same(baseline, await execute(True))
    assert_same(baseline, await execute(True, alternate=True))
