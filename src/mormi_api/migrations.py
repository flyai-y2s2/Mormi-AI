"""Expand-contract database migration helpers shared by operations and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from alembic.config import Config
from sqlalchemy import UniqueConstraint, create_engine, inspect
from sqlalchemy.engine import Connection, Engine, make_url

from alembic import command

from .db import Base

OBSERVATION_TABLES = {
    "dialogue_turn_observations",
    "dialogue_claims",
    "dialogue_task_outcomes",
    "note_evidence_links",
    "ai_outbox_events",
}

UNVERSIONED_BASELINE_TABLES = OBSERVATION_TABLES | {"ladder_analysis_jobs"}
CURRENT_HEAD_TABLES = UNVERSIONED_BASELINE_TABLES | {
    # Revisions 20260825_03, 20260826_05 and 20260826_06 changed the core idempotency
    # identity. A database stamped at head without the current column and
    # scenario-scoped constraint must fail before a live conversation starts.
    "conversations",
    "dialogue_generated_copy_cache",
}

SCENARIO_IDENTITY_TRANSITION_REVISION = "20260826_05"
SCENARIO_IDENTITY_FINAL_REVISION = "20260826_06"
SESSION_PARENT_REVISION = "20260831_07"
SCENARIO_IDENTITY_READER_CAPABILITY = "conversation-scenario-idempotency-reader-v1"
OLD_CONVERSATION_IDENTITY = "uq_conversation_learning_session_round"
NEW_CONVERSATION_IDENTITY = (
    "uq_conversation_learning_session_scene_scenario_round"
)
ConversationIdentitySchemaPhase = Literal["transition", "final"]
IdentityContract = Literal["compatible", "transition", "final"]


def _application_schema_issues(
    bind: Connection | Engine,
    *,
    required_tables: set[str],
    identity_contract: IdentityContract = "final",
) -> list[str]:
    """Return structural differences that make an application schema unsafe.

    Table names alone are not enough: an interrupted/manual deployment can
    leave a table present while a required column, FK, unique constraint or
    index is missing. Stamping that database as Alembic head would make later
    upgrades skip the repair and defer the failure to a child's live turn.
    """

    inspector = inspect(bind)
    existing = set(inspector.get_table_names())
    issues: list[str] = []
    for table_name in sorted(required_tables):
        if table_name not in existing:
            issues.append(f"{table_name}:missing_table")
            continue
        expected = Base.metadata.tables[table_name]
        actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
        for column_name in sorted({column.name for column in expected.columns} - actual_columns):
            issues.append(f"{table_name}:missing_column:{column_name}")

        actual_foreign_keys = {
            (
                tuple(key.get("constrained_columns") or ()),
                str(key.get("referred_table")),
                tuple(key.get("referred_columns") or ()),
            )
            for key in inspector.get_foreign_keys(table_name)
        }
        expected_foreign_keys = {
            (
                tuple(element.parent.name for element in constraint.elements),
                next(iter(constraint.elements)).column.table.name,
                tuple(element.column.name for element in constraint.elements),
            )
            for constraint in expected.foreign_key_constraints
        }
        for columns, referred_table, referred_columns in sorted(
            expected_foreign_keys - actual_foreign_keys
        ):
            issues.append(
                f"{table_name}:missing_fk:{','.join(columns)}->"
                f"{referred_table}({','.join(referred_columns)})"
            )

        actual_unique_names = {
            str(constraint["name"])
            for constraint in inspector.get_unique_constraints(table_name)
            if constraint.get("name")
        }
        expected_unique_names = {
            str(constraint.name)
            for constraint in expected.constraints
            if isinstance(constraint, UniqueConstraint) and constraint.name
        }
        for name in sorted(expected_unique_names - actual_unique_names):
            issues.append(f"{table_name}:missing_unique:{name}")

        actual_index_names = {
            str(index["name"])
            for index in inspector.get_indexes(table_name)
            if index.get("name")
        }
        expected_index_names = {str(index.name) for index in expected.indexes if index.name}
        for name in sorted(expected_index_names - actual_index_names):
            issues.append(f"{table_name}:missing_index:{name}")
        if table_name == "conversations":
            unique_index_names = {
                str(index["name"])
                for index in inspector.get_indexes(table_name)
                if index.get("name") and index.get("unique")
            }
            actual_identity_names = actual_unique_names | unique_index_names
            if (
                identity_contract == "transition"
                and OLD_CONVERSATION_IDENTITY not in actual_identity_names
            ):
                issues.append(
                    f"{table_name}:missing_unique:{OLD_CONVERSATION_IDENTITY}"
                )
            elif (
                identity_contract == "final"
                and OLD_CONVERSATION_IDENTITY in actual_identity_names
            ):
                issues.append(
                    f"{table_name}:obsolete_unique:{OLD_CONVERSATION_IDENTITY}"
                )
    return issues


def _observation_schema_issues(
    bind: Connection | Engine,
    *,
    identity_contract: IdentityContract = "compatible",
) -> list[str]:
    """Return application issues for a rolling-compatible identity phase."""

    return _application_schema_issues(
        bind,
        required_tables=CURRENT_HEAD_TABLES,
        identity_contract=identity_contract,
    )


def _unversioned_baseline_schema_issues(bind: Connection | Engine) -> list[str]:
    return _application_schema_issues(
        bind,
        required_tables=UNVERSIONED_BASELINE_TABLES,
    )


def _raise_schema_issues(issues: list[str], *, contract: str) -> None:
    if issues:
        raise RuntimeError(
            f"{contract} schema does not match the application contract; "
            "refusing to stamp or start from a silently partial migration. "
            f"issues={issues}"
        )


def conversation_identity_schema_phase(
    bind: Connection | Engine,
) -> ConversationIdentitySchemaPhase:
    inspector = inspect(bind)
    unique_names = {
        str(constraint["name"])
        for constraint in inspector.get_unique_constraints("conversations")
        if constraint.get("name")
    }
    unique_names.update(
        str(index["name"])
        for index in inspector.get_indexes("conversations")
        if index.get("name") and index.get("unique")
    )
    return (
        "transition"
        if OLD_CONVERSATION_IDENTITY in unique_names
        else "final"
    )


def require_observation_schema(
    bind: Connection | Engine,
    *,
    identity_contract: IdentityContract = "compatible",
) -> ConversationIdentitySchemaPhase:
    _raise_schema_issues(
        _observation_schema_issues(bind, identity_contract=identity_contract),
        contract=f"Application ({identity_contract} identity)",
    )
    return conversation_identity_schema_phase(bind)


def _require_unversioned_baseline_schema(bind: Connection | Engine) -> None:
    _raise_schema_issues(
        _unversioned_baseline_schema_issues(bind),
        contract="Unversioned baseline",
    )


def require_session_parent_schema(bind: Connection | Engine) -> None:
    """The optional parent is additive; turn-only readers still accept revision 06."""
    _raise_schema_issues(
        _application_schema_issues(bind, required_tables={"dialogue_session_parents"}),
        contract="Session parent",
    )


def synchronous_url(raw_url: str) -> str:
    url = make_url(raw_url)

    if url.drivername == "postgresql+asyncpg":
        url = url.set(drivername="postgresql+psycopg")

        query = dict(url.query)
        if "ssl" in query:
            query["sslmode"] = query.pop("ssl")

        url = url.set(query=query)

    elif url.drivername == "sqlite+aiosqlite":
        url = url.set(drivername="sqlite")

    return url.render_as_string(hide_password=False)


def apply_database_migrations(
    raw_url: str,
    root: Path,
    *,
    target_revision: str = "head",
) -> ConversationIdentitySchemaPhase:
    """Apply one reviewed expand/contract migration target and verify it."""

    if target_revision not in {
        "head",
        SCENARIO_IDENTITY_TRANSITION_REVISION,
        SCENARIO_IDENTITY_FINAL_REVISION,
        SESSION_PARENT_REVISION,
    }:
        raise ValueError(f"unsupported database migration target: {target_revision}")
    expected_phase: ConversationIdentitySchemaPhase = (
        "transition"
        if target_revision == SCENARIO_IDENTITY_TRANSITION_REVISION
        else "final"
    )

    config = Config(root / "alembic.ini")
    sync_url = synchronous_url(raw_url)
    config.set_main_option("sqlalchemy.url", sync_url.replace("%", "%%"))

    engine = create_engine(sync_url)
    try:
        existing = set(inspect(engine).get_table_names())
        if "conversations" not in existing:
            # A brand-new database has no legacy rows to preserve.
            Base.metadata.create_all(engine)
            if expected_phase == "transition":
                # ORM metadata describes the final schema. Re-enter the
                # revision chain at 04 so transition revision 05 can add the
                # rollback-compatible old key beside the new key.
                command.stamp(config, "20260825_04")
                command.upgrade(config, SCENARIO_IDENTITY_TRANSITION_REVISION)
            else:
                command.stamp(
                    config,
                    SESSION_PARENT_REVISION if target_revision in {"head", SESSION_PARENT_REVISION}
                    else SCENARIO_IDENTITY_FINAL_REVISION,
                )
            require_observation_schema(engine, identity_contract=expected_phase)
        elif "alembic_version" in existing or not (existing & OBSERVATION_TABLES):
            # Existing pilot databases keep every row. Revision 05 expands the
            # key while revision 06 contracts it only after every live/rollback
            # image advertises the scenario-aware reader.
            command.upgrade(config, target_revision)
            require_observation_schema(engine, identity_contract=expected_phase)
        elif OBSERVATION_TABLES.issubset(existing):
            # Older deployments call Base.metadata.create_all() during app
            # startup. If that happened before this operational command, the
            # complete baseline schema already exists but Alembic has no
            # version row. Validate only the tables owned by that baseline;
            # later additions (conversation_round and generated-copy cache)
            # must not block the stamp that allows their revisions to run.
            _require_unversioned_baseline_schema(engine)
            command.stamp(config, "20260823_02")
            command.upgrade(config, target_revision)
            require_observation_schema(engine, identity_contract=expected_phase)
        else:
            partial = sorted(existing & OBSERVATION_TABLES)
            missing = sorted(OBSERVATION_TABLES - existing)
            raise RuntimeError(
                "Partial observation schema detected; refusing to guess. "
                f"present={partial}, missing={missing}"
            )
        if target_revision in {"head", SESSION_PARENT_REVISION}:
            require_session_parent_schema(engine)
    finally:
        engine.dispose()
    return expected_phase
