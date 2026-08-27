from __future__ import annotations

import hashlib
import json
import math
import secrets
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy.exc import IntegrityError

from .db import Database, DialogueGeneratedCopyCacheRecord
from .schemas import utc_now

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

STABLE_COPY_CACHE_KEY_VERSION = "stable-copy-key-v1"
_CACHE_KEY_NAMESPACE = "mormi-dialogue-stable-copy"
_HEX_DIGITS = frozenset("0123456789abcdef")
_ERROR_CODE_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Generation plans are reviewed, content-owned inputs.  This deny-list is a
# defence-in-depth boundary against accidentally adding per-child state to a
# permanent cache key.  A hash is pseudonymous data when its input contains an
# identifier, so hashing alone is not the privacy control.
_FORBIDDEN_KEY_MATERIAL_FIELDS = frozenset(
    {
        "address",
        "birthdate",
        "childid",
        "childname",
        "childresponse",
        "childutterance",
        "conversationid",
        "dialoguehistory",
        "email",
        "evidencespan",
        "learnerid",
        "learnername",
        "learnerprofile",
        "phonenumber",
        "rawtext",
        "recenthistory",
        "responseid",
        "turnid",
        "userid",
    }
)


class StableCopyCacheKeyError(ValueError):
    pass


class CopyCacheCorruptionError(RuntimeError):
    pass


class CopyCacheAcquireState(StrEnum):
    READY = "ready"
    LEASED = "leased"
    BUSY = "busy"
    BACKOFF = "backoff"


@dataclass(frozen=True, slots=True)
class CachedCopyArtifact:
    cache_key: str
    artifact: JsonValue
    artifact_sha256: str
    ready_at: datetime


@dataclass(frozen=True, slots=True)
class CopyGenerationLease:
    cache_key: str
    lease_token: str
    attempt: int
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class CopyCacheAcquireResult:
    state: CopyCacheAcquireState
    artifact: CachedCopyArtifact | None = None
    lease: CopyGenerationLease | None = None
    retry_at: datetime | None = None


def _compact_field_name(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


def _json_value(
    value: object,
    *,
    normalize_unicode: bool,
    reject_pii_fields: bool,
    path: str = "$",
) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value) if normalize_unicode else value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return value
    if isinstance(value, list):
        return [
            _json_value(
                item,
                normalize_unicode=normalize_unicode,
                reject_pii_fields=reject_pii_fields,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"JSON object key must be a string at {path}")
            if (
                reject_pii_fields
                and _compact_field_name(raw_key) in _FORBIDDEN_KEY_MATERIAL_FIELDS
            ):
                raise StableCopyCacheKeyError(
                    f"per-child field is forbidden in stable-copy key material: {raw_key}"
                )
            key = (
                unicodedata.normalize("NFC", raw_key)
                if normalize_unicode
                else raw_key
            )
            if key in normalized:
                raise ValueError(f"duplicate JSON key after normalization at {path}: {key}")
            normalized[key] = _json_value(
                item,
                normalize_unicode=normalize_unicode,
                reject_pii_fields=reject_pii_fields,
                path=f"{path}.{key}",
            )
        return normalized
    raise ValueError(f"value at {path} is not JSON-compatible")


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _key_component(value: str, *, name: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise StableCopyCacheKeyError(f"{name} must be non-empty without edge whitespace")
    if len(normalized) > 200:
        raise StableCopyCacheKeyError(f"{name} exceeds 200 characters")
    return normalized


def build_stable_copy_cache_key(
    *,
    content_revision: str,
    content_hash: str,
    copy_slot_id: str,
    locale: str,
    prompt_version: str,
    schema_version: str,
    validator_version: str,
    model_id: str,
    generation_config: Mapping[str, object],
    generation_plan: Mapping[str, object],
) -> str:
    """Hash only stable, content-owned inputs into a deterministic cache key.

    Per-child values are intentionally absent from the signature.  Nested
    generation-plan keys that commonly carry identifiers, raw utterances,
    evidence or dialogue history are rejected before hashing.
    """

    canonical_plan = _json_value(
        generation_plan,
        normalize_unicode=True,
        reject_pii_fields=True,
    )
    canonical_generation_config = _json_value(
        generation_config,
        normalize_unicode=True,
        reject_pii_fields=True,
    )
    _validate_sha256_hex(content_hash, name="content_hash")
    material: JsonValue = {
        "namespace": _CACHE_KEY_NAMESPACE,
        "key_version": STABLE_COPY_CACHE_KEY_VERSION,
        "content_revision": _key_component(
            content_revision,
            name="content_revision",
        ),
        "content_hash": content_hash,
        "copy_slot_id": _key_component(copy_slot_id, name="copy_slot_id"),
        "locale": _key_component(locale, name="locale"),
        "prompt_version": _key_component(prompt_version, name="prompt_version"),
        "schema_version": _key_component(schema_version, name="schema_version"),
        "validator_version": _key_component(
            validator_version,
            name="validator_version",
        ),
        "model_id": _key_component(model_id, name="model_id"),
        "generation_config": canonical_generation_config,
        "generation_plan": canonical_plan,
    }
    return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _validate_sha256_hex(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_cache_key(cache_key: str) -> None:
    _validate_sha256_hex(cache_key, name="cache_key")


def _validate_error_code(error_code: str) -> None:
    if not 1 <= len(error_code) <= 80 or any(
        character not in _ERROR_CODE_CHARACTERS for character in error_code
    ):
        raise ValueError(
            "error_code must contain only uppercase ASCII letters, digits, or underscore"
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _artifact_value(value: object) -> JsonValue:
    return _json_value(
        value,
        normalize_unicode=False,
        reject_pii_fields=False,
    )


def _artifact_digest(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


class GeneratedCopyCacheRepository:
    """Durable get-or-generate coordination for immutable stable copy.

    A lease token plus attempt number prevents an expired generator from
    overwriting a newer result.  Once a row reaches ``ready`` no repository
    method can alter its artifact.
    """

    def __init__(
        self,
        database: Database,
        *,
        lease_seconds: float,
        retry_base_seconds: float,
        retry_max_seconds: float,
    ) -> None:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if not math.isfinite(retry_base_seconds) or retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be positive")
        if (
            not math.isfinite(retry_max_seconds)
            or retry_max_seconds < retry_base_seconds
        ):
            raise ValueError("retry_max_seconds must be at least retry_base_seconds")
        self.database = database
        self.lease_seconds = lease_seconds
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds

    @staticmethod
    def _assert_key_version(record: DialogueGeneratedCopyCacheRecord) -> None:
        if record.key_version != STABLE_COPY_CACHE_KEY_VERSION:
            raise CopyCacheCorruptionError(
                f"unsupported generated-copy cache key version: {record.key_version}"
            )

    @staticmethod
    def _lease(record: DialogueGeneratedCopyCacheRecord) -> CopyGenerationLease:
        GeneratedCopyCacheRepository._assert_key_version(record)
        if record.lease_token is None:
            raise CopyCacheCorruptionError("generating cache record has no lease token")
        return CopyGenerationLease(
            cache_key=record.cache_key,
            lease_token=record.lease_token,
            attempt=record.attempts,
            lease_expires_at=_as_utc(record.available_at),
        )

    @staticmethod
    def _ready_artifact(
        record: DialogueGeneratedCopyCacheRecord,
    ) -> CachedCopyArtifact:
        GeneratedCopyCacheRepository._assert_key_version(record)
        if record.artifact_sha256 is None or record.ready_at is None:
            raise CopyCacheCorruptionError("ready cache record is missing artifact metadata")
        artifact = _artifact_value(record.artifact_json)
        if _artifact_digest(artifact) != record.artifact_sha256:
            raise CopyCacheCorruptionError("ready cache artifact digest mismatch")
        return CachedCopyArtifact(
            cache_key=record.cache_key,
            artifact=deepcopy(artifact),
            artifact_sha256=record.artifact_sha256,
            ready_at=_as_utc(record.ready_at),
        )

    async def get_ready(self, cache_key: str) -> CachedCopyArtifact | None:
        _validate_cache_key(cache_key)
        async with self.database.sessions() as db:
            record = await db.get(DialogueGeneratedCopyCacheRecord, cache_key)
            if record is None or record.status != "ready":
                return None
            return self._ready_artifact(record)

    async def acquire(
        self,
        cache_key: str,
        *,
        now: datetime | None = None,
    ) -> CopyCacheAcquireResult:
        _validate_cache_key(cache_key)
        acquired_at = _as_utc(now or utc_now())
        lease_until = acquired_at + timedelta(seconds=self.lease_seconds)
        lease_token = secrets.token_hex(24)

        async with self.database.sessions() as db:
            record = await db.get(
                DialogueGeneratedCopyCacheRecord,
                cache_key,
                with_for_update=True,
            )
            if record is None:
                record = DialogueGeneratedCopyCacheRecord(
                    cache_key=cache_key,
                    key_version=STABLE_COPY_CACHE_KEY_VERSION,
                    status="generating",
                    attempts=1,
                    available_at=lease_until,
                    lease_token=lease_token,
                    created_at=acquired_at,
                    updated_at=acquired_at,
                )
                db.add(record)
                try:
                    await db.commit()
                except IntegrityError:
                    # A concurrent creator won the unique-key race.  Read its
                    # committed state through the normal existing-row path.
                    await db.rollback()
                else:
                    return CopyCacheAcquireResult(
                        state=CopyCacheAcquireState.LEASED,
                        lease=self._lease(record),
                    )

        return await self._acquire_existing(cache_key, acquired_at=acquired_at)

    async def _acquire_existing(
        self,
        cache_key: str,
        *,
        acquired_at: datetime,
    ) -> CopyCacheAcquireResult:
        async with self.database.sessions() as db:
            record = await db.get(
                DialogueGeneratedCopyCacheRecord,
                cache_key,
                with_for_update=True,
            )
            if record is None:
                # A concurrent rollback immediately after an insert conflict is
                # not a valid steady state and is safe for the caller to retry.
                return CopyCacheAcquireResult(
                    state=CopyCacheAcquireState.BUSY,
                    retry_at=acquired_at,
                )
            self._assert_key_version(record)
            if record.status == "ready":
                return CopyCacheAcquireResult(
                    state=CopyCacheAcquireState.READY,
                    artifact=self._ready_artifact(record),
                )

            available_at = _as_utc(record.available_at)
            if available_at > acquired_at:
                state = (
                    CopyCacheAcquireState.BUSY
                    if record.status == "generating"
                    else CopyCacheAcquireState.BACKOFF
                )
                if record.status not in {"generating", "retry"}:
                    raise CopyCacheCorruptionError(
                        f"unknown generated-copy cache status: {record.status}"
                    )
                return CopyCacheAcquireResult(
                    state=state,
                    retry_at=available_at,
                )

            if record.status not in {"generating", "retry"}:
                raise CopyCacheCorruptionError(
                    f"unknown generated-copy cache status: {record.status}"
                )

            lease_until = acquired_at + timedelta(seconds=self.lease_seconds)
            record.status = "generating"
            record.attempts += 1
            record.available_at = lease_until
            record.lease_token = secrets.token_hex(24)
            record.last_error_code = None
            record.updated_at = acquired_at
            await db.commit()
            return CopyCacheAcquireResult(
                state=CopyCacheAcquireState.LEASED,
                lease=self._lease(record),
            )

    async def complete(
        self,
        lease: CopyGenerationLease,
        *,
        artifact: object,
        now: datetime | None = None,
    ) -> CachedCopyArtifact | None:
        _validate_cache_key(lease.cache_key)
        canonical_artifact = _artifact_value(artifact)
        digest = _artifact_digest(canonical_artifact)
        completed_at = _as_utc(now or utc_now())

        async with self.database.sessions() as db:
            record = await db.get(
                DialogueGeneratedCopyCacheRecord,
                lease.cache_key,
                with_for_update=True,
            )
            if (
                record is None
                or record.status != "generating"
                or record.attempts != lease.attempt
                or record.lease_token != lease.lease_token
            ):
                return None
            self._assert_key_version(record)
            record.status = "ready"
            record.artifact_json = deepcopy(canonical_artifact)
            record.artifact_sha256 = digest
            record.available_at = completed_at
            record.lease_token = None
            record.last_error_code = None
            record.ready_at = completed_at
            record.updated_at = completed_at
            await db.commit()
            return self._ready_artifact(record)

    async def fail(
        self,
        lease: CopyGenerationLease,
        *,
        error_code: str,
        now: datetime | None = None,
    ) -> datetime | None:
        _validate_cache_key(lease.cache_key)
        _validate_error_code(error_code)
        failed_at = _as_utc(now or utc_now())

        async with self.database.sessions() as db:
            record = await db.get(
                DialogueGeneratedCopyCacheRecord,
                lease.cache_key,
                with_for_update=True,
            )
            if (
                record is None
                or record.status != "generating"
                or record.attempts != lease.attempt
                or record.lease_token != lease.lease_token
            ):
                return None
            self._assert_key_version(record)
            retry_at = failed_at + timedelta(
                seconds=self._retry_delay(record.attempts)
            )
            record.status = "retry"
            record.available_at = retry_at
            record.lease_token = None
            record.last_error_code = error_code
            record.updated_at = failed_at
            await db.commit()
            return retry_at

    def _retry_delay(self, attempt: int) -> float:
        delay = self.retry_base_seconds
        for _ in range(max(attempt - 1, 0)):
            if delay >= self.retry_max_seconds:
                return self.retry_max_seconds
            delay = min(delay * 2, self.retry_max_seconds)
        return delay
