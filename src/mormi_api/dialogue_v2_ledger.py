from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .dialogue_v2_content import RequiredHomeTeachingPackV2
from .dialogue_v2_evidence import (
    EvidenceMatchKindV2,
    EvidenceMatchV2,
    GuardedUnderstandingV2,
)
from .dialogue_v2_life_content import LifeFactV2, LifeTaskPackV2
from .schemas import (
    AuxiliaryUnderstandingClaimV2,
    CanonicalValueV2,
    FactUnderstandingClaimV2,
    RelationUnderstandingClaimV2,
)

_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
_CANONICAL_VALUE_ADAPTER: TypeAdapter[CanonicalValueV2] = TypeAdapter(CanonicalValueV2)
_AUXILIARY_SUMMARY_CODE_V2 = "task_related_auxiliary_evidence"


class DialogueV2LedgerError(ValueError):
    """Raised when pinned server-owned state is internally inconsistent."""


class LedgerModelV2(BaseModel):
    """Strict immutable base for conversation-pinned V2 reasoning state."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_content_pack_json_v2(pack: RequiredHomeTeachingPackV2) -> str:
    """Return the stable byte representation used by conversation snapshots."""

    return json.dumps(
        pack.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_pack_hash_v2(pack: RequiredHomeTeachingPackV2) -> str:
    """Hash one complete pack revision, not the mutable process catalog."""

    return hashlib.sha256(canonical_content_pack_json_v2(pack).encode("utf-8")).hexdigest()


def canonical_life_task_pack_json_v2(pack: LifeTaskPackV2) -> str:
    """Stable representation for one materialized real-life task variant."""

    return json.dumps(
        pack.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def life_task_pack_hash_v2(pack: LifeTaskPackV2) -> str:
    return hashlib.sha256(
        canonical_life_task_pack_json_v2(pack).encode("utf-8")
    ).hexdigest()


class PinnedContentSnapshotV2(LedgerModelV2):
    """A complete immutable content revision selected for one conversation.

    The payload is intentionally self-contained. Resumed conversations never
    consult the process-global catalog, whose current revision may have changed
    since the child began teaching Mormi.
    """

    schema_version: Literal["pinned-content-v2"] = "pinned-content-v2"
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    curriculum_session_id: str = Field(pattern=_ID_PATTERN, max_length=100)
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    pack_payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_pinned_pack(self) -> PinnedContentSnapshotV2:
        pack = RequiredHomeTeachingPackV2.model_validate(self.pack_payload)
        if pack.pack_id != self.pack_id:
            raise ValueError("pinned content pack_id does not match its payload")
        if pack.content_version != self.content_version:
            raise ValueError("pinned content version does not match its payload")
        if pack.curriculum_session_id != self.curriculum_session_id:
            raise ValueError("pinned curriculum session does not match its payload")
        if content_pack_hash_v2(pack) != self.content_hash:
            raise ValueError("pinned content hash does not match its payload")
        return self

    def resolve_pack(self) -> RequiredHomeTeachingPackV2:
        """Revalidate and return the exact pack stored in this snapshot."""

        pack = RequiredHomeTeachingPackV2.model_validate(self.pack_payload)
        if (
            pack.pack_id != self.pack_id
            or pack.content_version != self.content_version
            or pack.curriculum_session_id != self.curriculum_session_id
            or content_pack_hash_v2(pack) != self.content_hash
        ):
            # ``frozen=True`` prevents replacing the field, while Python mappings
            # are still mutable containers. Recheck the hash on every resolution
            # so an accidental in-process nested mutation cannot affect a turn.
            raise DialogueV2LedgerError("pinned content snapshot failed integrity validation")
        return pack


class PinnedLifeTaskSnapshotV2(LedgerModelV2):
    """Immutable materialized task variant inside a scenario V3 snapshot."""

    schema_version: Literal["pinned-life-task-v2"] = "pinned-life-task-v2"
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    task_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    pack_payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_pinned_pack(self) -> PinnedLifeTaskSnapshotV2:
        pack = LifeTaskPackV2.model_validate(self.pack_payload)
        if (
            pack.pack_id != self.pack_id
            or pack.content_version != self.content_version
            or pack.task_id != self.task_id
            or life_task_pack_hash_v2(pack) != self.content_hash
        ):
            raise ValueError("pinned life task identity does not match its payload")
        return self

    def resolve_pack(self) -> LifeTaskPackV2:
        pack = LifeTaskPackV2.model_validate(self.pack_payload)
        if (
            pack.pack_id != self.pack_id
            or pack.content_version != self.content_version
            or pack.task_id != self.task_id
            or life_task_pack_hash_v2(pack) != self.content_hash
        ):
            raise DialogueV2LedgerError(
                "pinned life task snapshot failed integrity validation"
            )
        return pack


def pin_content_pack_v2(pack: RequiredHomeTeachingPackV2) -> PinnedContentSnapshotV2:
    """Copy a reviewed pack into a self-validating conversation snapshot."""

    payload = pack.model_dump(mode="json")
    return PinnedContentSnapshotV2(
        pack_id=pack.pack_id,
        content_version=pack.content_version,
        curriculum_session_id=pack.curriculum_session_id,
        content_hash=content_pack_hash_v2(pack),
        pack_payload=payload,
    )


def pin_life_task_pack_v2(pack: LifeTaskPackV2) -> PinnedLifeTaskSnapshotV2:
    return PinnedLifeTaskSnapshotV2(
        pack_id=pack.pack_id,
        content_version=pack.content_version,
        task_id=pack.task_id,
        content_hash=life_task_pack_hash_v2(pack),
        pack_payload=pack.model_dump(mode="json"),
    )


class LedgerEvidencePointerV2(LedgerModelV2):
    """Literal-evidence provenance without duplicating raw child text in state."""

    evidence_id: str = Field(pattern=_SHA256_PATTERN)
    source_turn_id: str = Field(min_length=1, max_length=100)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    match_kind: EvidenceMatchKindV2

    @model_validator(mode="after")
    def validate_source_range(self) -> LedgerEvidencePointerV2:
        if self.source_end <= self.source_start:
            raise ValueError("ledger evidence end must follow its start")
        return self


class FactVerificationEvidenceV2(LedgerEvidencePointerV2):
    evidence_kind: Literal["understanding_fact"] = "understanding_fact"
    classifier_verdict: Literal["correct"] = "correct"


class RelationVerificationEvidenceV2(LedgerEvidencePointerV2):
    evidence_kind: Literal["understanding_relation"] = "understanding_relation"
    classifier_verdict: Literal["correct", "sufficient"]


class StructuredVerificationEvidenceV2(LedgerModelV2):
    """Server-authored evidence from a pinned L2 choice or L0 joint action."""

    evidence_kind: Literal["structured"] = "structured"
    evidence_id: str = Field(pattern=_SHA256_PATTERN)
    source_turn_id: str = Field(min_length=1, max_length=100)
    source_kind: Literal["choice", "joint"]
    target_kind: Literal["fact", "relation"]
    target_id: str = Field(pattern=_ID_PATTERN, max_length=160)


FactLedgerEvidenceV2 = Annotated[
    FactVerificationEvidenceV2 | StructuredVerificationEvidenceV2,
    Field(discriminator="evidence_kind"),
]

RelationLedgerEvidenceV2 = Annotated[
    RelationVerificationEvidenceV2 | StructuredVerificationEvidenceV2,
    Field(discriminator="evidence_kind"),
]


class VerifiedFactLedgerEntryV2(LedgerModelV2):
    fact_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    canonical_value: CanonicalValueV2
    evidence: list[FactLedgerEvidenceV2] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> VerifiedFactLedgerEntryV2:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("fact ledger evidence ids must be unique")
        if any(
            isinstance(item, StructuredVerificationEvidenceV2)
            and (item.target_kind != "fact" or item.target_id != self.fact_id)
            for item in self.evidence
        ):
            raise ValueError("structured fact evidence must reference its ledger fact")
        return self


class VerifiedRelationLedgerEntryV2(LedgerModelV2):
    relation_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    evidence: list[RelationLedgerEvidenceV2] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> VerifiedRelationLedgerEntryV2:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("relation ledger evidence ids must be unique")
        if any(
            isinstance(item, StructuredVerificationEvidenceV2)
            and (item.target_kind != "relation" or item.target_id != self.relation_id)
            for item in self.evidence
        ):
            raise ValueError("structured relation evidence must reference its ledger relation")
        return self


class AcceptedAuxiliaryEvidenceV2(LedgerEvidencePointerV2):
    summary: str = Field(min_length=1, max_length=160)
    classifier_verdict: Literal["correct", "sufficient", "partial"]


class ReasoningLedgerV2(LedgerModelV2):
    """Monotonic canonical progress for one pinned teaching pack.

    Canonical fact values come only from the reviewed content snapshot. Model
    interpretations and verdicts are retained as audit evidence, but code never
    compares them to expected values or recomputes their arithmetic.
    """

    schema_version: Literal["reasoning-ledger-v2"] = "reasoning-ledger-v2"
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    content_hash: str = Field(pattern=_SHA256_PATTERN)
    verified_facts: dict[str, VerifiedFactLedgerEntryV2] = Field(
        default_factory=dict,
        max_length=100,
    )
    verified_relations: dict[str, VerifiedRelationLedgerEntryV2] = Field(
        default_factory=dict,
        max_length=100,
    )
    accepted_auxiliary_evidence: dict[str, AcceptedAuxiliaryEvidenceV2] = Field(
        default_factory=dict,
        max_length=200,
    )

    @model_validator(mode="after")
    def map_keys_must_match_entries(self) -> ReasoningLedgerV2:
        if any(key != value.fact_id for key, value in self.verified_facts.items()):
            raise ValueError("verified fact map keys must match fact ids")
        if any(
            key != value.relation_id for key, value in self.verified_relations.items()
        ):
            raise ValueError("verified relation map keys must match relation ids")
        if any(
            key != value.evidence_id
            for key, value in self.accepted_auxiliary_evidence.items()
        ):
            raise ValueError("auxiliary evidence map keys must match evidence ids")
        return self


ContentSnapshotV2 = PinnedContentSnapshotV2 | PinnedLifeTaskSnapshotV2
TaskPackV2 = RequiredHomeTeachingPackV2 | LifeTaskPackV2


def empty_reasoning_ledger_v2(snapshot: ContentSnapshotV2) -> ReasoningLedgerV2:
    """Initialize an empty ledger bound to one immutable content snapshot."""

    return ReasoningLedgerV2(
        pack_id=snapshot.pack_id,
        content_version=snapshot.content_version,
        content_hash=snapshot.content_hash,
    )


class ReasoningCompletionV2(LedgerModelV2):
    required_fact_ids: list[str]
    required_relation_ids: list[str]
    remaining_fact_ids: list[str]
    remaining_relation_ids: list[str]
    complete: bool


class ReasoningLedgerApplyResultV2(LedgerModelV2):
    ledger: ReasoningLedgerV2
    new_fact_ids: list[str]
    new_relation_ids: list[str]
    new_milestone_fact_ids: list[str]
    new_fact_evidence_ids: list[str]
    new_relation_evidence_ids: list[str]
    new_auxiliary_evidence_ids: list[str]
    # Model-authored claim IDs are turn-local correlation handles. They may
    # contain child text, so both maps are excluded from every serialization.
    claim_evidence_ids: dict[str, str] = Field(default_factory=dict, exclude=True)
    ignored_claim_ids: list[str] = Field(default_factory=list, exclude=True)
    completion: ReasoningCompletionV2
    completion_became_true: bool

    @property
    def has_new_canonical_progress(self) -> bool:
        return bool(self.new_fact_ids or self.new_relation_ids)


def _validate_ledger_binding(
    snapshot: ContentSnapshotV2,
    ledger: ReasoningLedgerV2,
    pack: TaskPackV2,
) -> None:
    expected_identity = (
        snapshot.pack_id,
        snapshot.content_version,
        snapshot.content_hash,
    )
    actual_identity = (ledger.pack_id, ledger.content_version, ledger.content_hash)
    if actual_identity != expected_identity:
        raise DialogueV2LedgerError("reasoning ledger is not bound to the pinned content")

    facts = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}
    relations = {
        relation.relation_id: relation for relation in pack.reasoning_graph.relations
    }
    if not set(ledger.verified_facts).issubset(facts):
        raise DialogueV2LedgerError("reasoning ledger contains an unknown fact")
    if not set(ledger.verified_relations).issubset(relations):
        raise DialogueV2LedgerError("reasoning ledger contains an unknown relation")
    for fact_id, entry in ledger.verified_facts.items():
        fact = facts[fact_id]
        value_matches = (
            fact.accepts_value(entry.canonical_value)
            if isinstance(fact, LifeFactV2)
            else entry.canonical_value == fact.value
        )
        if not value_matches:
            raise DialogueV2LedgerError(
                "reasoning ledger canonical fact differs from pinned content"
            )


def reasoning_completion_v2(
    snapshot: ContentSnapshotV2,
    ledger: ReasoningLedgerV2,
) -> ReasoningCompletionV2:
    """Calculate completion strictly from required graph IDs."""

    pack = snapshot.resolve_pack()
    _validate_ledger_binding(snapshot, ledger, pack)
    contract = pack.reasoning_graph.completion
    remaining_facts = [
        fact_id
        for fact_id in contract.required_fact_ids
        if fact_id not in ledger.verified_facts
    ]
    remaining_relations = [
        relation_id
        for relation_id in contract.required_relation_ids
        if relation_id not in ledger.verified_relations
    ]
    return ReasoningCompletionV2(
        required_fact_ids=list(contract.required_fact_ids),
        required_relation_ids=list(contract.required_relation_ids),
        remaining_fact_ids=remaining_facts,
        remaining_relation_ids=remaining_relations,
        complete=not remaining_facts and not remaining_relations,
    )


def _evidence_matches_by_claim(
    guarded: GuardedUnderstandingV2,
) -> dict[str, EvidenceMatchV2]:
    claim_ids = [claim.claim_id for claim in guarded.response.claims]
    matches = {match.claim_id: match for match in guarded.evidence_matches}
    if len(matches) != len(guarded.evidence_matches) or set(matches) != set(claim_ids):
        raise DialogueV2LedgerError(
            "guarded understanding must contain exactly one evidence match per claim"
        )
    return matches


def _stable_evidence_id(
    *,
    source_turn_id: str,
    match: EvidenceMatchV2,
    claim_kind: str,
    target_id: str,
    semantic_payload: object,
) -> str:
    """Create an idempotency key without including raw child text."""

    payload = json.dumps(
        {
            "source_turn_id": source_turn_id,
            "source_start": match.source_start,
            "source_end": match.source_end,
            "match_kind": match.match_kind.value,
            "claim_kind": claim_kind,
            "target_id": target_id,
            "semantic_payload": semantic_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_guarded_understanding_v2(
    snapshot: ContentSnapshotV2,
    ledger: ReasoningLedgerV2,
    guarded: GuardedUnderstandingV2,
    *,
    source_turn_id: str,
) -> ReasoningLedgerApplyResultV2:
    """Monotonically apply model verdicts admitted by the literal evidence guard.

    This function deliberately does *not* compare interpreted values with content
    truth, normalize units, or recalculate arithmetic. Interpreted values and
    arithmetic details may be absent because they are diagnostic metadata, not a
    second adjudicator. A ``correct`` fact verdict records the pinned canonical
    fact; a ``correct``/``sufficient`` relation verdict records the relation.
    Incorrect, partial, and uncertain canonical claims are observed but cannot add
    or remove verified progress.
    """

    if not source_turn_id or len(source_turn_id) > 100:
        raise DialogueV2LedgerError("source_turn_id must be a non-empty bounded id")

    pack = snapshot.resolve_pack()
    _validate_ledger_binding(snapshot, ledger, pack)
    matches = _evidence_matches_by_claim(guarded)
    facts_by_id = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}

    before_completion = reasoning_completion_v2(snapshot, ledger)
    verified_facts = dict(ledger.verified_facts)
    verified_relations = dict(ledger.verified_relations)
    auxiliary_evidence = dict(ledger.accepted_auxiliary_evidence)

    new_fact_ids: list[str] = []
    new_relation_ids: list[str] = []
    new_milestone_fact_ids: list[str] = []
    new_fact_evidence_ids: list[str] = []
    new_relation_evidence_ids: list[str] = []
    new_auxiliary_evidence_ids: list[str] = []
    claim_evidence_ids: dict[str, str] = {}
    ignored_claim_ids: list[str] = []

    for claim in guarded.response.claims:
        match = matches[claim.claim_id]
        if isinstance(claim, FactUnderstandingClaimV2):
            if claim.verdict != "correct":
                ignored_claim_ids.append(claim.claim_id)
                continue
            fact_semantic_payload: dict[str, object] = {"verdict": claim.verdict}
            evidence_id = _stable_evidence_id(
                source_turn_id=source_turn_id,
                match=match,
                claim_kind="fact",
                target_id=claim.fact_id,
                semantic_payload=fact_semantic_payload,
            )
            fact_evidence = FactVerificationEvidenceV2(
                evidence_id=evidence_id,
                source_turn_id=source_turn_id,
                source_start=match.source_start,
                source_end=match.source_end,
                match_kind=match.match_kind,
            )
            claim_evidence_ids[claim.claim_id] = evidence_id
            existing = verified_facts.get(claim.fact_id)
            if existing is None:
                fact = facts_by_id[claim.fact_id]
                verified_facts[claim.fact_id] = VerifiedFactLedgerEntryV2(
                    fact_id=claim.fact_id,
                    canonical_value=fact.value,
                    evidence=[fact_evidence],
                )
                new_fact_ids.append(claim.fact_id)
                if fact.role == "intermediate_result":
                    new_milestone_fact_ids.append(claim.fact_id)
                new_fact_evidence_ids.append(evidence_id)
            elif evidence_id not in {item.evidence_id for item in existing.evidence}:
                verified_facts[claim.fact_id] = existing.model_copy(
                    update={"evidence": [*existing.evidence, fact_evidence]}
                )
                new_fact_evidence_ids.append(evidence_id)
            continue

        if isinstance(claim, RelationUnderstandingClaimV2):
            if claim.verdict not in {"correct", "sufficient"}:
                ignored_claim_ids.append(claim.claim_id)
                continue
            relation_verdict = cast(Literal["correct", "sufficient"], claim.verdict)
            relation_semantic_payload: dict[str, object] = {
                "verdict": relation_verdict,
            }
            evidence_id = _stable_evidence_id(
                source_turn_id=source_turn_id,
                match=match,
                claim_kind="relation",
                target_id=claim.relation_id,
                semantic_payload=relation_semantic_payload,
            )
            relation_evidence = RelationVerificationEvidenceV2(
                evidence_id=evidence_id,
                source_turn_id=source_turn_id,
                source_start=match.source_start,
                source_end=match.source_end,
                match_kind=match.match_kind,
                classifier_verdict=relation_verdict,
            )
            claim_evidence_ids[claim.claim_id] = evidence_id
            existing_relation = verified_relations.get(claim.relation_id)
            if existing_relation is None:
                verified_relations[claim.relation_id] = VerifiedRelationLedgerEntryV2(
                    relation_id=claim.relation_id,
                    evidence=[relation_evidence],
                )
                new_relation_ids.append(claim.relation_id)
                new_relation_evidence_ids.append(evidence_id)
            elif evidence_id not in {
                item.evidence_id for item in existing_relation.evidence
            }:
                verified_relations[claim.relation_id] = existing_relation.model_copy(
                    update={"evidence": [*existing_relation.evidence, relation_evidence]}
                )
                new_relation_evidence_ids.append(evidence_id)
            continue

        if not isinstance(claim, AuxiliaryUnderstandingClaimV2):  # pragma: no cover
            raise DialogueV2LedgerError("unsupported guarded understanding claim")
        if claim.verdict not in {"correct", "sufficient", "partial"}:
            ignored_claim_ids.append(claim.claim_id)
            continue
        auxiliary_verdict = cast(
            Literal["correct", "sufficient", "partial"],
            claim.verdict,
        )
        # ``claim.summary`` is free-form model output and may paraphrase the
        # child's raw utterance or personal information.  Auxiliary evidence is
        # useful only as a non-completing progress marker, so persist and hash a
        # fixed server-owned code instead of model-authored prose.
        auxiliary_semantic_payload: dict[str, object] = {
            "verdict": auxiliary_verdict,
            "summary_code": _AUXILIARY_SUMMARY_CODE_V2,
        }
        evidence_id = _stable_evidence_id(
            source_turn_id=source_turn_id,
            match=match,
            claim_kind="auxiliary",
            target_id="open",
            semantic_payload=auxiliary_semantic_payload,
        )
        claim_evidence_ids[claim.claim_id] = evidence_id
        if evidence_id in auxiliary_evidence:
            continue
        auxiliary_evidence[evidence_id] = AcceptedAuxiliaryEvidenceV2(
            evidence_id=evidence_id,
            source_turn_id=source_turn_id,
            source_start=match.source_start,
            source_end=match.source_end,
            match_kind=match.match_kind,
            summary=_AUXILIARY_SUMMARY_CODE_V2,
            classifier_verdict=auxiliary_verdict,
        )
        new_auxiliary_evidence_ids.append(evidence_id)

    next_ledger = ReasoningLedgerV2(
        pack_id=ledger.pack_id,
        content_version=ledger.content_version,
        content_hash=ledger.content_hash,
        verified_facts=verified_facts,
        verified_relations=verified_relations,
        accepted_auxiliary_evidence=auxiliary_evidence,
    )
    completion = reasoning_completion_v2(snapshot, next_ledger)
    return ReasoningLedgerApplyResultV2(
        ledger=next_ledger,
        new_fact_ids=new_fact_ids,
        new_relation_ids=new_relation_ids,
        new_milestone_fact_ids=new_milestone_fact_ids,
        new_fact_evidence_ids=new_fact_evidence_ids,
        new_relation_evidence_ids=new_relation_evidence_ids,
        new_auxiliary_evidence_ids=new_auxiliary_evidence_ids,
        claim_evidence_ids=claim_evidence_ids,
        ignored_claim_ids=ignored_claim_ids,
        completion=completion,
        completion_became_true=completion.complete and not before_completion.complete,
    )


def _stable_structured_evidence_id(
    *,
    source_turn_id: str,
    source_kind: Literal["choice", "joint"],
    target_kind: Literal["fact", "relation"],
    target_id: str,
    canonical_value: CanonicalValueV2 | None = None,
) -> str:
    payload = json.dumps(
        {
            "source_turn_id": source_turn_id,
            "source_kind": source_kind,
            "target_kind": target_kind,
            "target_id": target_id,
            "canonical_value": (
                canonical_value.model_dump(mode="json")
                if canonical_value is not None
                else None
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_structured_progress_v2(
    snapshot: ContentSnapshotV2,
    ledger: ReasoningLedgerV2,
    *,
    fact_values: dict[str, CanonicalValueV2 | dict[str, Any]],
    relation_ids: list[str],
    source_turn_id: str,
    source_kind: Literal["choice", "joint"],
) -> ReasoningLedgerApplyResultV2:
    """Apply a previously resolved pinned L2 or L0 content effect.

    The caller must first resolve the submitted choice/action against the active
    pinned plan. This boundary then verifies only server-owned IDs and canonical
    values. It never synthesizes an understanding verdict and never accepts a
    client-provided label as mathematical evidence.
    """

    if not source_turn_id or len(source_turn_id) > 100:
        raise DialogueV2LedgerError("source_turn_id must be a non-empty bounded id")
    if len(relation_ids) != len(set(relation_ids)):
        raise DialogueV2LedgerError("structured relation ids must be unique")

    pack = snapshot.resolve_pack()
    _validate_ledger_binding(snapshot, ledger, pack)
    facts_by_id = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}
    relation_id_set = {
        relation.relation_id for relation in pack.reasoning_graph.relations
    }
    unknown_facts = set(fact_values) - set(facts_by_id)
    if unknown_facts:
        raise DialogueV2LedgerError("structured progress contains an unknown fact")
    unknown_relations = set(relation_ids) - relation_id_set
    if unknown_relations:
        raise DialogueV2LedgerError("structured progress contains an unknown relation")

    parsed_fact_values: dict[str, CanonicalValueV2] = {}
    for fact_id, raw_value in fact_values.items():
        value = _CANONICAL_VALUE_ADAPTER.validate_python(raw_value)
        fact = facts_by_id[fact_id]
        value_matches = (
            fact.accepts_value(value)
            if isinstance(fact, LifeFactV2)
            else value == fact.value
        )
        if not value_matches:
            raise DialogueV2LedgerError(
                "structured fact value differs from pinned canonical content"
            )
        parsed_fact_values[fact_id] = value

    before_completion = reasoning_completion_v2(snapshot, ledger)
    verified_facts = dict(ledger.verified_facts)
    verified_relations = dict(ledger.verified_relations)
    new_fact_ids: list[str] = []
    new_relation_ids: list[str] = []
    new_milestone_fact_ids: list[str] = []
    new_fact_evidence_ids: list[str] = []
    new_relation_evidence_ids: list[str] = []

    for fact_id, value in parsed_fact_values.items():
        fact = facts_by_id[fact_id]
        evidence_id = _stable_structured_evidence_id(
            source_turn_id=source_turn_id,
            source_kind=source_kind,
            target_kind="fact",
            target_id=fact_id,
            canonical_value=value,
        )
        evidence = StructuredVerificationEvidenceV2(
            evidence_id=evidence_id,
            source_turn_id=source_turn_id,
            source_kind=source_kind,
            target_kind="fact",
            target_id=fact_id,
        )
        existing = verified_facts.get(fact_id)
        if existing is None:
            verified_facts[fact_id] = VerifiedFactLedgerEntryV2(
                fact_id=fact_id,
                canonical_value=value,
                evidence=[evidence],
            )
            new_fact_ids.append(fact_id)
            if fact.role == "intermediate_result":
                new_milestone_fact_ids.append(fact_id)
            new_fact_evidence_ids.append(evidence_id)
        elif evidence_id not in {item.evidence_id for item in existing.evidence}:
            verified_facts[fact_id] = existing.model_copy(
                update={"evidence": [*existing.evidence, evidence]}
            )
            new_fact_evidence_ids.append(evidence_id)

    for relation_id in relation_ids:
        evidence_id = _stable_structured_evidence_id(
            source_turn_id=source_turn_id,
            source_kind=source_kind,
            target_kind="relation",
            target_id=relation_id,
        )
        evidence = StructuredVerificationEvidenceV2(
            evidence_id=evidence_id,
            source_turn_id=source_turn_id,
            source_kind=source_kind,
            target_kind="relation",
            target_id=relation_id,
        )
        existing_relation = verified_relations.get(relation_id)
        if existing_relation is None:
            verified_relations[relation_id] = VerifiedRelationLedgerEntryV2(
                relation_id=relation_id,
                evidence=[evidence],
            )
            new_relation_ids.append(relation_id)
            new_relation_evidence_ids.append(evidence_id)
        elif evidence_id not in {
            item.evidence_id for item in existing_relation.evidence
        }:
            verified_relations[relation_id] = existing_relation.model_copy(
                update={"evidence": [*existing_relation.evidence, evidence]}
            )
            new_relation_evidence_ids.append(evidence_id)

    next_ledger = ReasoningLedgerV2(
        pack_id=ledger.pack_id,
        content_version=ledger.content_version,
        content_hash=ledger.content_hash,
        verified_facts=verified_facts,
        verified_relations=verified_relations,
        accepted_auxiliary_evidence=ledger.accepted_auxiliary_evidence,
    )
    completion = reasoning_completion_v2(snapshot, next_ledger)
    return ReasoningLedgerApplyResultV2(
        ledger=next_ledger,
        new_fact_ids=new_fact_ids,
        new_relation_ids=new_relation_ids,
        new_milestone_fact_ids=new_milestone_fact_ids,
        new_fact_evidence_ids=new_fact_evidence_ids,
        new_relation_evidence_ids=new_relation_evidence_ids,
        new_auxiliary_evidence_ids=[],
        ignored_claim_ids=[],
        completion=completion,
        completion_became_true=completion.complete and not before_completion.complete,
    )
