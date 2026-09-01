from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
from alembic.config import Config
from conftest import FakeGateway
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.exc import IntegrityError

from alembic import command
from mormi_api.db import (
    Base,
    ConversationRecord,
    Database,
    DataMigrationRecord,
    DialogueClaimRecord,
    DialogueTurnObservationRecord,
    OutboxEventRecord,
    TurnRecord,
)
from mormi_api.engine import ConversationEngine
from mormi_api.migrations import apply_database_migrations
from mormi_api.repository import Repository
from mormi_api.schemas import ChildResponse, SessionCreate, utc_now
from mormi_api.security import StoredTextCodec, TextCipher
from mormi_api.service import ConversationService


def _alembic_config(database_url: str) -> Config:
    root = Path(__file__).resolve().parents[1]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _legacy_fernet(secret: str, text: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return f"fernet:{Fernet(key).encrypt(text.encode('utf-8')).decode('ascii')}"


def test_stored_text_codec_writes_plaintext_and_reads_legacy_envelopes() -> None:
    secret = "legacy-encryption-key"
    codec = StoredTextCodec(secret)

    assert codec.store("점이 3개야") == "plain:점이 3개야"
    assert codec.load("plain:점이 3개야") == "점이 3개야"
    assert codec.load("이미 평문인 값") == "이미 평문인 값"
    assert codec.load(_legacy_fernet(secret, "예전 암호문")) == "예전 암호문"


def test_stored_text_codec_requires_legacy_key_only_for_old_fernet_rows() -> None:
    codec = StoredTextCodec(None)

    assert codec.load("plain:새 평문") == "새 평문"
    assert codec.load("접두사 없는 평문") == "접두사 없는 평문"
    with pytest.raises(ValueError, match="MORMI_RAW_DATA_ENCRYPTION_KEY"):
        codec.load(_legacy_fernet("missing-key", "예전 암호문"))


def test_additive_migration_preserves_legacy_conversation_and_turn_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MORMI_DATABASE_URL", raising=False)
    database_path = tmp_path / "legacy.db"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE conversations (
                conversation_id VARCHAR(100) PRIMARY KEY,
                learner_id INTEGER NOT NULL,
                learning_session_id VARCHAR(100),
                scene VARCHAR(40) NOT NULL,
                scenario_id VARCHAR(100) NOT NULL,
                state_json JSON NOT NULL,
                state_version INTEGER NOT NULL,
                status VARCHAR(40) NOT NULL,
                raw_retention_until DATETIME,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
    TurnRecord.__table__.create(engine)
    now = utc_now()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            INSERT INTO conversations (
                conversation_id, learner_id, learning_session_id, scene,
                scenario_id, state_json, state_version, status,
                raw_retention_until, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "conversation_legacy",
                7,
                "lesson_legacy",
                "home_teach",
                "home_teach",
                "{}",
                1,
                "active",
                None,
                now,
                now,
            ),
        )
        connection.execute(
            TurnRecord.__table__.insert().values(
                turn_id="turn_legacy",
                conversation_id="conversation_legacy",
                task_id="home_teaching",
                state_version=1,
                mormi_question_encrypted="encrypted",
                turn_contract={},
                expression_level="L4",
                hint_level="H0",
                created_at=now,
            )
        )

    config = _alembic_config(database_url)
    command.upgrade(config, "head")

    with engine.connect() as connection:
        assert connection.execute(
            select(ConversationRecord.conversation_id)
        ).scalar_one() == "conversation_legacy"
        assert connection.execute(
            select(ConversationRecord.conversation_round)
        ).scalar_one() == 1
        assert connection.execute(
            select(TurnRecord.turn_id)
        ).scalar_one() == "turn_legacy"
        tables = set(inspect(connection).get_table_names())
        assert "dialogue_turn_observations" in tables
        assert "dialogue_claims" in tables
        assert "dialogue_task_outcomes" in tables
        assert "note_evidence_links" in tables
        assert "ai_outbox_events" in tables
        assert "dialogue_generated_copy_cache" in tables
        observation_columns = {
            column["name"]
            for column in inspect(connection).get_columns(
                "dialogue_turn_observations"
            )
        }
        assert "adult_intervention_status" not in observation_columns
        conversation_unique_constraints = {
            constraint["name"]
            for constraint in inspect(connection).get_unique_constraints("conversations")
        }
        assert (
            "uq_conversation_learning_session_scene_scenario_round"
            in conversation_unique_constraints
        )
        assert "uq_conversation_learning_session_round" not in conversation_unique_constraints
        cache_columns = {
            column["name"]
            for column in inspect(connection).get_columns(
                "dialogue_generated_copy_cache"
            )
        }
        assert {
            "cache_key",
            "key_version",
            "status",
            "attempts",
            "available_at",
            "lease_token",
            "artifact_json",
            "artifact_sha256",
            "last_error_code",
            "ready_at",
            "created_at",
            "updated_at",
        } == cache_columns
        assert {
            index["name"]
            for index in inspect(connection).get_indexes(
                "dialogue_generated_copy_cache"
            )
        } == {"ix_dialogue_generated_copy_cache_available"}

    command.downgrade(config, "base")
    with engine.connect() as connection:
        assert connection.execute(
            select(ConversationRecord.conversation_id)
        ).scalar_one() == "conversation_legacy"
        assert connection.execute(
            select(TurnRecord.turn_id)
        ).scalar_one() == "turn_legacy"
        assert "dialogue_generated_copy_cache" not in set(
            inspect(connection).get_table_names()
        )
    engine.dispose()


def test_scenario_idempotency_migration_preserves_rows_and_rolls_back_safely(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "scenario-idempotency.db"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url)
    config = _alembic_config(database_url)
    Base.metadata.create_all(engine)
    command.stamp(config, "head")
    command.downgrade(config, "20260825_04")

    def unique_constraints() -> dict[str, tuple[str, ...]]:
        return {
            str(constraint["name"]): tuple(constraint.get("column_names") or ())
            for constraint in inspect(engine).get_unique_constraints("conversations")
            if constraint.get("name")
        }

    old_name = "uq_conversation_learning_session_round"
    new_name = "uq_conversation_learning_session_scene_scenario_round"
    assert old_name in unique_constraints()
    assert new_name not in unique_constraints()

    now = utc_now()
    first_row = {
        "conversation_id": "conversation_queue",
        "learner_id": 77,
        "learning_session_id": "cafe_visit_77",
        "conversation_round": 1,
        "scene": "cafe",
        "scenario_id": "cafe_queue",
        "state_json": {},
        "state_version": 1,
        "status": "active",
        "raw_retention_until": None,
        "created_at": now,
        "updated_at": now,
    }
    with engine.begin() as connection:
        connection.execute(ConversationRecord.__table__.insert().values(**first_row))

    command.upgrade(config, "20260826_05")
    constraints = unique_constraints()
    assert old_name in constraints
    assert constraints[new_name] == (
        "learner_id",
        "learning_session_id",
        "scene",
        "scenario_id",
        "conversation_round",
    )

    second_row = {
        **first_row,
        "conversation_id": "conversation_budget",
        "scenario_id": "cafe_budget_menu",
    }
    # Expand phase still protects the currently live visit-wide reader, so a
    # second scenario cannot be created until every rollback image is upgraded.
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(ConversationRecord.__table__.insert().values(**second_row))

    command.upgrade(config, "head")
    constraints = unique_constraints()
    assert old_name not in constraints
    assert new_name in constraints
    with engine.begin() as connection:
        connection.execute(ConversationRecord.__table__.insert().values(**second_row))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            ConversationRecord.__table__.insert().values(
                **{
                    **first_row,
                    "conversation_id": "conversation_queue_duplicate",
                }
            )
        )

    command.downgrade(config, "20260826_05")
    constraints = unique_constraints()
    assert new_name in constraints
    assert old_name in constraints
    with engine.connect() as connection:
        rows = connection.execute(
            select(
                ConversationRecord.conversation_id,
                ConversationRecord.conversation_round,
                ConversationRecord.state_json,
            ).order_by(ConversationRecord.conversation_id)
        ).all()
    assert rows == [
        ("conversation_budget", 1, {"conversation_round": 1}),
        ("conversation_queue", 2, {"conversation_round": 2}),
    ]

    command.downgrade(config, "20260825_04")
    constraints = unique_constraints()
    assert new_name not in constraints
    assert old_name in constraints
    engine.dispose()


def test_migration_stamps_complete_schema_created_by_app_startup(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup-created.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    apply_database_migrations(
        f"sqlite+aiosqlite:///{database_path}",
        Path(__file__).resolve().parents[1],
    )

    with engine.connect() as connection:
        version = connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
        assert version == "20260831_07"
    engine.dispose()


def test_migration_runner_supports_transition_then_final_on_fresh_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "fresh-expand-contract.db"
    async_url = f"sqlite+aiosqlite:///{database_path}"
    sync_url = f"sqlite:///{database_path}"
    root = Path(__file__).resolve().parents[1]

    transition_phase = apply_database_migrations(
        async_url,
        root,
        target_revision="20260826_05",
    )
    engine = create_engine(sync_url)
    with engine.connect() as connection:
        constraints = {
            str(item["name"])
            for item in inspect(connection).get_unique_constraints("conversations")
            if item.get("name")
        }
        assert connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == "20260826_05"
    assert transition_phase == "transition"
    assert "uq_conversation_learning_session_round" in constraints
    assert "uq_conversation_learning_session_scene_scenario_round" in constraints
    engine.dispose()

    final_phase = apply_database_migrations(async_url, root, target_revision="head")
    engine = create_engine(sync_url)
    with engine.connect() as connection:
        constraints = {
            str(item["name"])
            for item in inspect(connection).get_unique_constraints("conversations")
            if item.get("name")
        }
        assert connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == "20260831_07"
    assert final_phase == "final"
    assert "uq_conversation_learning_session_round" not in constraints
    assert "uq_conversation_learning_session_scene_scenario_round" in constraints
    engine.dispose()


def test_migration_repairs_missing_copy_cache_index_without_recreating_table(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup-created-without-cache-index.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "DROP INDEX ix_dialogue_generated_copy_cache_available"
        )

    apply_database_migrations(
        f"sqlite+aiosqlite:///{database_path}",
        Path(__file__).resolve().parents[1],
    )

    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == "20260831_07"
        assert {
            index["name"]
            for index in inspect(connection).get_indexes(
                "dialogue_generated_copy_cache"
            )
        } == {"ix_dialogue_generated_copy_cache_available"}
    engine.dispose()


def test_migration_upgrades_complete_unversioned_baseline_through_current_head(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "unversioned-baseline.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    config = _alembic_config(sync_url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE dialogue_generated_copy_cache")
    command.stamp(config, "20260825_03")
    command.downgrade(config, "20260823_02")
    with engine.begin() as connection:
        assert "conversation_round" not in {
            column["name"]
            for column in inspect(connection).get_columns("conversations")
        }
        assert "dialogue_generated_copy_cache" not in set(
            inspect(connection).get_table_names()
        )
        connection.exec_driver_sql("DROP TABLE alembic_version")

    apply_database_migrations(
        f"sqlite+aiosqlite:///{database_path}",
        Path(__file__).resolve().parents[1],
    )

    with engine.connect() as connection:
        version = connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
        assert version == "20260831_07"
        assert "conversation_round" in {
            column["name"]
            for column in inspect(connection).get_columns("conversations")
        }
        assert "dialogue_generated_copy_cache" in set(
            inspect(connection).get_table_names()
        )
        assert {
            index["name"]
            for index in inspect(connection).get_indexes(
                "dialogue_generated_copy_cache"
            )
        } == {"ix_dialogue_generated_copy_cache_available"}
    engine.dispose()


def test_migration_refuses_to_stamp_table_names_with_a_missing_column(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "malformed-startup-created.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE dialogue_claims")
        connection.exec_driver_sql(
            "CREATE TABLE dialogue_claims (id INTEGER PRIMARY KEY)"
        )

    with pytest.raises(RuntimeError, match="missing_column:observation_id"):
        apply_database_migrations(
            f"sqlite+aiosqlite:///{database_path}",
            Path(__file__).resolve().parents[1],
        )

    assert "alembic_version" not in set(inspect(engine).get_table_names())
    engine.dispose()


def test_migration_refuses_head_version_with_a_missing_observation_table(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "head-but-partial.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    config = _alembic_config(sync_url)
    command.stamp(config, "head")
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE dialogue_claims")

    with pytest.raises(RuntimeError, match="dialogue_claims:missing_table"):
        apply_database_migrations(
            f"sqlite+aiosqlite:///{database_path}",
            Path(__file__).resolve().parents[1],
        )

    engine.dispose()


def test_migration_refuses_head_version_with_a_missing_copy_cache_table(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "head-without-copy-cache.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    config = _alembic_config(sync_url)
    command.stamp(config, "head")
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE dialogue_generated_copy_cache")

    with pytest.raises(
        RuntimeError,
        match="dialogue_generated_copy_cache:missing_table",
    ):
        apply_database_migrations(
            f"sqlite+aiosqlite:///{database_path}",
            Path(__file__).resolve().parents[1],
        )

    engine.dispose()


def test_migration_refuses_head_version_without_conversation_round_contract(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "head-without-conversation-round.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    config = _alembic_config(sync_url)
    command.stamp(config, "head")
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE conversations")
        connection.exec_driver_sql(
            "CREATE TABLE conversations (conversation_id VARCHAR(100) PRIMARY KEY)"
        )

    with pytest.raises(
        RuntimeError,
        match="conversations:missing_column:conversation_round",
    ):
        apply_database_migrations(
            f"sqlite+aiosqlite:///{database_path}",
            Path(__file__).resolve().parents[1],
        )

    engine.dispose()


def test_migration_refuses_head_version_without_scenario_round_unique_constraint(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "head-without-round-unique.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    config = _alembic_config(sync_url)
    command.stamp(config, "head")
    with engine.begin() as connection:
        # CTAS preserves every column but intentionally drops constraints and
        # indexes, reproducing a manually stamped partial schema.
        connection.exec_driver_sql(
            "CREATE TABLE conversations_partial AS SELECT * FROM conversations"
        )
        connection.exec_driver_sql("DROP TABLE conversations")
        connection.exec_driver_sql(
            "ALTER TABLE conversations_partial RENAME TO conversations"
        )

    with pytest.raises(
        RuntimeError,
        match=(
            "conversations:missing_unique:"
            "uq_conversation_learning_session_scene_scenario_round"
        ),
    ):
        apply_database_migrations(
            f"sqlite+aiosqlite:///{database_path}",
            Path(__file__).resolve().parents[1],
        )

    engine.dispose()


def test_migration_refuses_head_version_with_obsolete_visit_wide_unique(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "head-with-obsolete-visit-unique.db"
    sync_url = f"sqlite:///{database_path}"
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    config = _alembic_config(sync_url)
    command.stamp(config, "head")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX uq_conversation_learning_session_round "
            "ON conversations (learner_id, learning_session_id, conversation_round)"
        )

    with pytest.raises(
        RuntimeError,
        match=(
            "conversations:obsolete_unique:"
            "uq_conversation_learning_session_round"
        ),
    ):
        apply_database_migrations(
            f"sqlite+aiosqlite:///{database_path}",
            Path(__file__).resolve().parents[1],
        )

    engine.dispose()


@pytest.mark.asyncio
async def test_plaintext_migration_converts_every_legacy_dialogue_evidence_once(
    tmp_path: Path,
) -> None:
    secret = "legacy-encryption-key"
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/plaintext-migration.db")
    await database.create_schema()
    repository = Repository(database, StoredTextCodec(secret))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=10,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )
    await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="1e120ef8-16a3-451f-9e73-254b63f45645",
            type="no_response",
        ),
    )

    question = "왼쪽과 오른쪽에는 각각 몇 명이 있어?"
    response = "왼쪽 세 명, 오른쪽 다섯 명"
    evidence = "오른쪽 다섯 명"
    note_evidence = "사람이 적은 줄에 서면 덜 기다려"
    async with database.sessions() as db:
        source_turn = (
            await db.execute(
                select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id)
            )
        ).scalar_one()
        observation = (
            await db.execute(
                select(DialogueTurnObservationRecord).where(
                    DialogueTurnObservationRecord.source_turn_id == started.turn.turn_id
                )
            )
        ).scalar_one()
        conversation = await db.get(ConversationRecord, started.conversation_id)
        assert conversation is not None

        source_turn.mormi_question_encrypted = _legacy_fernet(secret, question)
        source_turn.response_raw_encrypted = _legacy_fernet(secret, response)
        db.add(
            DialogueClaimRecord(
                observation_id=observation.observation_id,
                slot_id="right_count",
                semantic_role="observation",
                value_json=5,
                factual=True,
                validation_status="verified",
                evidence_span_encrypted=_legacy_fernet(secret, evidence),
                newly_verified=True,
            )
        )
        state_json = dict(conversation.state_json)
        state_json["child_note_evidence"] = {
            "reason": _legacy_fernet(secret, note_evidence)
        }
        state_json[Repository._STATE_EVIDENCE_ENCRYPTED] = True
        conversation.state_json = state_json
        await db.commit()

    await repository.migrate_existing_storage_to_plaintext()
    await repository.migrate_existing_storage_to_plaintext()

    async with database.sessions() as db:
        source_turn = (
            await db.execute(
                select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id)
            )
        ).scalar_one()
        claim = (
            await db.execute(
                select(DialogueClaimRecord).where(
                    DialogueClaimRecord.slot_id == "right_count"
                )
            )
        ).scalar_one()
        conversation = await db.get(ConversationRecord, started.conversation_id)
        migration = await db.get(
            DataMigrationRecord,
            Repository.PLAINTEXT_STORAGE_MIGRATION,
        )

        assert source_turn.mormi_question_encrypted == f"plain:{question}"
        assert source_turn.response_raw_encrypted == f"plain:{response}"
        assert claim.evidence_span_encrypted == f"plain:{evidence}"
        assert conversation is not None
        assert Repository._STATE_EVIDENCE_ENCRYPTED not in conversation.state_json
        assert conversation.state_json["child_note_evidence"] == {
            "reason": note_evidence
        }
        assert migration is not None

    raw_turns = await repository.raw_turns(started.conversation_id)
    first_answered = next(turn for turn in raw_turns if turn["response"] is not None)
    assert first_answered["question"] == question
    assert first_answered["response"] == response
    await database.dispose()


@pytest.mark.asyncio
async def test_historical_backfill_marks_missing_fields_without_reanalysis(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/backfill.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=9,
            scene="cafe",
            scenario_id="cafe_queue_demo",
        )
    )
    await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="0855ffef-3042-4ef4-85e0-22162b560c7e",
            type="no_response",
        ),
    )

    # Simulate a row created by the legacy application before observation
    # tables existed. The original turn remains untouched.
    async with database.sessions() as db:
        await db.execute(delete(OutboxEventRecord))
        await db.execute(delete(DialogueTurnObservationRecord))
        await db.commit()

    assert await repository.backfill_historical_observations() == 1
    assert await repository.backfill_historical_observations() == 0

    async with database.sessions() as db:
        observation = (
            await db.execute(select(DialogueTurnObservationRecord))
        ).scalar_one()
        outbox_count = len(list((await db.execute(select(OutboxEventRecord))).scalars()))
        turn = (
            await db.execute(
                select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id)
            )
        ).scalar_one()

    assert observation.record_origin == "historical_backfill"
    assert observation.difficulty_class == "not_collected"
    assert observation.transition_reason == "not_collected"
    assert observation.analysis_json["historical_backfill"] is True
    assert outbox_count == 0
    assert turn.response_raw_encrypted is not None
    await database.dispose()
