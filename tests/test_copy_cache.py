from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mormi_api.copy_cache import (
    CopyCacheAcquireState,
    CopyCacheCorruptionError,
    CopyGenerationLease,
    GeneratedCopyCacheRepository,
    StableCopyCacheKeyError,
    build_stable_copy_cache_key,
)
from mormi_api.db import Database, DialogueGeneratedCopyCacheRecord


def _cache_key(
    *,
    content_revision: str = "home-required-v2-r1",
    generation_config: dict[str, object] | None = None,
    generation_plan: dict[str, object] | None = None,
) -> str:
    return build_stable_copy_cache_key(
        content_revision=content_revision,
        content_hash="a" * 64,
        copy_slot_id="multiply-easy-tables.initial-help.answer",
        locale="ko-KR",
        prompt_version="stable-copy-prompt-v1",
        schema_version="stable-copy-output-v1",
        validator_version="stable-copy-validator-v1",
        model_id="claude-sonnet-4-6",
        generation_config=generation_config
        or {"effort": "low", "temperature": 0},
        generation_plan=generation_plan
        or {
            "goal": "답 하나만 부담 없이 말할 수 있게 다시 묻는다",
            "constraints": ["모르미 1인칭", "평가 표현 금지"],
        },
    )


async def _repository(
    tmp_path: Path,
    *,
    lease_seconds: float = 10,
    retry_base_seconds: float = 3,
    retry_max_seconds: float = 30,
) -> tuple[Database, GeneratedCopyCacheRepository]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/generated-copy-cache.db")
    await database.create_schema()
    return database, GeneratedCopyCacheRepository(
        database,
        lease_seconds=lease_seconds,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def test_stable_copy_key_is_canonical_and_revision_sensitive() -> None:
    composed = _cache_key(
        generation_plan={
            "goal": "모르미가 아이에게 부탁해",
            "rules": {"tone": "편안하게", "level": 2},
        }
    )
    reordered_and_decomposed = _cache_key(
        generation_plan={
            "rules": {"level": 2, "tone": "편안하게"},
            "goal": "모르미가 아이에게 부탁해",
        }
    )

    assert composed == reordered_and_decomposed
    assert len(composed) == 64
    assert set(composed) <= set("0123456789abcdef")
    assert _cache_key(content_revision="home-required-v2-r2") != composed
    assert _cache_key(generation_config={"effort": "low", "temperature": 0.1}) != composed


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "learner_id",
        "childUtterance",
        "conversation-id",
        "recent_history",
        "evidence_span",
        "responseId",
    ],
)
def test_stable_copy_key_rejects_nested_per_child_material(
    forbidden_field: str,
) -> None:
    with pytest.raises(StableCopyCacheKeyError, match="per-child field"):
        _cache_key(
            generation_plan={
                "stable": {"instruction": "짧게 물어본다"},
                "runtime": {forbidden_field: "아동별 값"},
            }
        )


def test_cache_table_persists_no_key_material_or_child_identifiers() -> None:
    columns = set(DialogueGeneratedCopyCacheRecord.__table__.columns.keys())

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
    } == columns
    assert not {
        "learner_id",
        "conversation_id",
        "turn_id",
        "response_id",
        "child_utterance",
        "key_material_json",
    } & columns
    assert {
        index.name for index in DialogueGeneratedCopyCacheRecord.__table__.indexes
    } == {"ix_dialogue_generated_copy_cache_available"}


@pytest.mark.asyncio
async def test_cache_lease_completes_once_and_ready_artifact_is_immutable(
    tmp_path: Path,
) -> None:
    database, repository = await _repository(tmp_path)
    cache_key = _cache_key()
    started_at = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)

    first = await repository.acquire(cache_key, now=started_at)
    assert first.state is CopyCacheAcquireState.LEASED
    assert first.lease is not None
    assert first.lease.attempt == 1
    assert first.lease.lease_expires_at == started_at + timedelta(seconds=10)

    busy = await repository.acquire(cache_key, now=started_at + timedelta(seconds=1))
    assert busy.state is CopyCacheAcquireState.BUSY
    assert busy.retry_at == first.lease.lease_expires_at

    completed = await repository.complete(
        first.lease,
        artifact={
            "text": "나 한 가지만 먼저 알고 싶어... 얼마인지 알려줄 수 있어?",
            "asked_targets": ["shortage"],
        },
        now=started_at + timedelta(seconds=2),
    )
    assert completed is not None
    assert completed.ready_at == started_at + timedelta(seconds=2)

    first_read = await repository.get_ready(cache_key)
    assert first_read is not None
    assert isinstance(first_read.artifact, dict)
    first_read.artifact["text"] = "호출자가 바꾼 값"

    second_read = await repository.get_ready(cache_key)
    assert second_read is not None
    assert isinstance(second_read.artifact, dict)
    assert second_read.artifact["text"] != "호출자가 바꾼 값"

    ready = await repository.acquire(cache_key, now=started_at + timedelta(seconds=3))
    assert ready.state is CopyCacheAcquireState.READY
    assert ready.artifact == second_read
    assert ready.lease is None

    replacement = await repository.complete(
        first.lease,
        artifact={"text": "ready 이후 덮어쓰기 시도"},
        now=started_at + timedelta(seconds=4),
    )
    assert replacement is None
    assert await repository.fail(
        first.lease,
        error_code="LATE_FAILURE",
        now=started_at + timedelta(seconds=4),
    ) is None
    assert await repository.get_ready(cache_key) == second_read
    await database.dispose()


@pytest.mark.asyncio
async def test_failure_uses_exponential_backoff_before_releasing_next_lease(
    tmp_path: Path,
) -> None:
    database, repository = await _repository(tmp_path)
    cache_key = _cache_key()
    started_at = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)

    first = await repository.acquire(cache_key, now=started_at)
    assert first.lease is not None
    first_retry_at = await repository.fail(
        first.lease,
        error_code="MODEL_TIMEOUT",
        now=started_at + timedelta(seconds=1),
    )
    assert first_retry_at == started_at + timedelta(seconds=4)

    waiting = await repository.acquire(
        cache_key,
        now=started_at + timedelta(seconds=3),
    )
    assert waiting.state is CopyCacheAcquireState.BACKOFF
    assert waiting.retry_at == first_retry_at

    second = await repository.acquire(cache_key, now=first_retry_at)
    assert second.state is CopyCacheAcquireState.LEASED
    assert second.lease is not None
    assert second.lease.attempt == 2
    second_retry_at = await repository.fail(
        second.lease,
        error_code="OUTPUT_INVALID",
        now=first_retry_at,
    )
    assert second_retry_at == first_retry_at + timedelta(seconds=6)

    still_waiting = await repository.acquire(
        cache_key,
        now=second_retry_at - timedelta(microseconds=1),
    )
    assert still_waiting.state is CopyCacheAcquireState.BACKOFF

    third = await repository.acquire(cache_key, now=second_retry_at)
    assert third.state is CopyCacheAcquireState.LEASED
    assert third.lease is not None
    assert third.lease.attempt == 3
    await database.dispose()


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed_and_stale_generator_cannot_write(
    tmp_path: Path,
) -> None:
    database, repository = await _repository(tmp_path, lease_seconds=5)
    cache_key = _cache_key()
    started_at = datetime(2026, 8, 25, 11, 0, tzinfo=UTC)

    first = await repository.acquire(cache_key, now=started_at)
    assert first.lease is not None
    reclaimed = await repository.acquire(
        cache_key,
        now=started_at + timedelta(seconds=5),
    )
    assert reclaimed.state is CopyCacheAcquireState.LEASED
    assert reclaimed.lease is not None
    assert reclaimed.lease.attempt == 2
    assert reclaimed.lease.lease_token != first.lease.lease_token

    assert await repository.complete(
        first.lease,
        artifact={"text": "만료된 worker의 결과"},
        now=started_at + timedelta(seconds=6),
    ) is None
    accepted = await repository.complete(
        reclaimed.lease,
        artifact=["generic", {"text": "새 lease 결과"}],
        now=started_at + timedelta(seconds=6),
    )
    assert accepted is not None
    assert accepted.artifact == ["generic", {"text": "새 lease 결과"}]
    await database.dispose()


@pytest.mark.asyncio
async def test_invalid_json_and_raw_error_message_never_enter_the_cache(
    tmp_path: Path,
) -> None:
    database, repository = await _repository(tmp_path)
    cache_key = _cache_key()
    acquired = await repository.acquire(cache_key)
    assert acquired.lease is not None

    with pytest.raises(ValueError, match="non-finite JSON number"):
        await repository.complete(
            acquired.lease,
            artifact={"temperature": float("nan")},
        )
    with pytest.raises(ValueError, match="error_code"):
        await repository.fail(
            acquired.lease,
            error_code="아이 원문이 포함된 오류 설명",
        )

    assert await repository.get_ready(cache_key) is None
    busy = await repository.acquire(cache_key)
    assert busy.state is CopyCacheAcquireState.BUSY
    await database.dispose()


@pytest.mark.asyncio
async def test_ready_lookup_detects_storage_mutation_instead_of_serving_it(
    tmp_path: Path,
) -> None:
    database, repository = await _repository(tmp_path)
    cache_key = _cache_key()
    acquired = await repository.acquire(cache_key)
    assert acquired.lease is not None
    assert await repository.complete(
        acquired.lease,
        artifact={"text": "검수된 생성 문구"},
    ) is not None

    async with database.sessions() as db:
        record = await db.get(DialogueGeneratedCopyCacheRecord, cache_key)
        assert record is not None
        record.artifact_json = {"text": "외부에서 변경된 문구"}
        await db.commit()

    with pytest.raises(CopyCacheCorruptionError, match="digest mismatch"):
        await repository.get_ready(cache_key)
    await database.dispose()


@pytest.mark.asyncio
async def test_repository_rejects_invalid_timing_configuration_and_lease_keys() -> None:
    database = Database("sqlite+aiosqlite:///:memory:")
    with pytest.raises(ValueError, match="lease_seconds"):
        GeneratedCopyCacheRepository(
            database,
            lease_seconds=0,
            retry_base_seconds=1,
            retry_max_seconds=2,
        )
    with pytest.raises(ValueError, match="at least retry_base_seconds"):
        GeneratedCopyCacheRepository(
            database,
            lease_seconds=1,
            retry_base_seconds=3,
            retry_max_seconds=2,
        )

    invalid_lease = CopyGenerationLease(
        cache_key="not-a-digest",
        lease_token="token",
        attempt=1,
        lease_expires_at=datetime.now(UTC),
    )
    repository = GeneratedCopyCacheRepository(
        database,
        lease_seconds=1,
        retry_base_seconds=1,
        retry_max_seconds=2,
    )

    with pytest.raises(ValueError, match="cache_key"):
        await repository.complete(invalid_lease, artifact={"text": "unused"})
    await database.dispose()
