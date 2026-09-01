"""Optional release gate: TWO OS processes, isolated schema on a local test DB.

MORMI_TEST_POSTGRES_URL must name a dedicated local mormi_test_* database.
Never loads .env or connects to the application's production database.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue
from typing import Any
from uuid import uuid4

import pytest
from session_parent_support import service
from sqlalchemy import event, func, select
from sqlalchemy.engine import make_url
from test_dialogue_v2_fault_parity import complete_understanding
from test_dialogue_v2_runtime import RecordingV2Gateway
from test_dialogue_v2_service_routing import _home_request

from mormi_api.db import Database, DialogueTurnObservationRecord, NoteRecord
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, ResponseType
from mormi_api.security import TextCipher
from mormi_api.session_parent_store import SessionParentStore


def scoped_database(url: str, schema: str) -> Database:
    assert schema.startswith("mormi_parent_test_") and schema.replace("_", "").isalnum()
    database = Database(url)

    @event.listens_for(database.engine.sync_engine, "do_connect")
    def search_path(_: Any, __: Any, ___: Any, parameters: dict[str, Any]) -> None:
        # A transactional SET in a connect event is rolled back by pool/dialect
        # initialization. Set the startup parameter so EVERY worker/connection
        # keeps this test-only schema even across rollback and pool checkout.
        parameters["server_settings"] = {
            **parameters.get("server_settings", {}),
            "search_path": schema,
        }

    return database


def worker(
    url: str,
    schema: str,
    conversation_id: str,
    turn_id: str,
    response_id: str,
    barrier: Any,
    output: Any,
) -> None:
    async def run() -> None:
        class Gateway(RecordingV2Gateway):
            async def understand_v2(self, request: Any) -> Any:
                await asyncio.to_thread(barrier.wait, 10)
                return complete_understanding(request)

        database = scoped_database(url, schema)
        try:
            app = service(Repository(database, TextCipher("synthetic")), Gateway(), percent=0)
            result = await app.respond(
                conversation_id,
                ChildResponse(
                    turn_id=turn_id,
                    response_id=response_id,
                    type=ResponseType.TEXT,
                    text="합성 정답 설명",
                ),
            )
            output.put(("ok", result.turn.state_version))
        except Exception as error:
            output.put((type(error).__name__, None))  # No credentials/error text.
        finally:
            await database.dispose()

    asyncio.run(run())


@pytest.mark.parametrize("same_response_id", [True, False], ids=["duplicate", "competing"])
@pytest.mark.parametrize("trial", range(3))
async def test_postgres_two_processes_commit_once_and_fence_parent(
    same_response_id: bool, trial: int,
) -> None:
    raw_url = os.getenv("MORMI_TEST_POSTGRES_URL")
    if not raw_url:
        pytest.skip("release gate requires an explicitly configured local test PostgreSQL")
    url = make_url(raw_url)
    if (
        url.drivername != "postgresql+asyncpg"
        or url.host not in {"localhost", "127.0.0.1", "::1"}
        or not (url.database or "").startswith("mormi_test_")
    ):
        pytest.fail("refusing non-local/non-dedicated test database")
    pytest.importorskip("asyncpg")
    schema = f"mormi_parent_test_{uuid4().hex}"
    admin = Database(raw_url)
    database = scoped_database(raw_url, schema)
    processes = []
    try:
        async with admin.engine.begin() as connection:
            await connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        await database.create_schema()
        repository = Repository(database, TextCipher("synthetic"))
        app = service(repository, RecordingV2Gateway())
        first = await app.create_conversation(_home_request())
        initial_cursor = await SessionParentStore(database).load(first.conversation_id)
        ctx = multiprocessing.get_context("spawn")
        barrier, output = ctx.Barrier(2), ctx.Queue()
        response_id = str(uuid4())
        for _ in range(2):
            process = ctx.Process(
                target=worker,
                args=(
                    raw_url,
                    schema,
                    first.conversation_id,
                    first.turn.turn_id,
                    response_id if same_response_id else str(uuid4()),
                    barrier,
                    output,
                ),
            )
            process.start()
            processes.append(process)
        for process in processes:
            await asyncio.to_thread(process.join, 20)
            assert not process.is_alive(), "PostgreSQL worker timed out"
            assert process.exitcode == 0
        try:
            results = [output.get(timeout=2) for _ in range(2)]
        except queue.Empty:
            pytest.fail("missing PostgreSQL worker result")
        assert sorted(r[0] for r in results) == ["StaleConversationError", "ok"]
        async with database.sessions() as db:
            assert (
                await db.scalar(select(func.count()).select_from(DialogueTurnObservationRecord))
                == 1
            )
            assert await db.scalar(select(func.count()).select_from(NoteRecord)) == 1
        store = SessionParentStore(database)
        cursor = await store.load(first.conversation_id)
        assert cursor is not None and cursor.state_version == 2 and cursor.phase == "completed"
        assert initial_cursor is not None and cursor.checkpoint is not None
        assert not await store.publish(
            initial_cursor,
            state_version=2,
            turn_id=cursor.turn_id,
            phase=cursor.phase,
            checkpoint=cursor.checkpoint,
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 5)
        await database.dispose()
        # Only the unique synthetic schema created by this test is removed.
        async with admin.engine.begin() as connection:
            await connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.dispose()
