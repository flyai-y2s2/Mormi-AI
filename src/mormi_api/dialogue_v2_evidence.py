from __future__ import annotations

import unicodedata
from enum import StrEnum

from pydantic import ConfigDict, Field, model_validator

from .schemas import (
    AuxiliaryUnderstandingClaimV2,
    DialogueV2Model,
    FactUnderstandingClaimV2,
    RelationUnderstandingClaimV2,
    UnderstandingRequestV2,
    UnderstandingResponseV2,
)


class EvidenceMatchKindV2(StrEnum):
    """The only equivalence rules admitted by the provenance guard."""

    EXACT = "exact"
    UNICODE_NFC = "unicode_nfc"


class EvidenceGuardViolationCodeV2(StrEnum):
    UNKNOWN_FACT_ID = "unknown_fact_id"
    UNKNOWN_RELATION_ID = "unknown_relation_id"
    AUXILIARY_CLAIMS_DISABLED = "auxiliary_claims_disabled"
    EVIDENCE_NOT_LITERAL = "evidence_not_literal"


class EvidenceMatchV2(DialogueV2Model):
    """A claim's evidence mapped back to boundaries in the original utterance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1, max_length=100)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    source_text: str = Field(min_length=1, max_length=300)
    match_kind: EvidenceMatchKindV2

    @model_validator(mode="after")
    def validate_source_boundaries(self) -> EvidenceMatchV2:
        if self.source_end <= self.source_start:
            raise ValueError("evidence source_end must be after source_start")
        if self.source_end - self.source_start != len(self.source_text):
            raise ValueError("evidence source boundaries must match source_text")
        return self


class EvidenceGuardViolationV2(DialogueV2Model):
    """Privacy-safe contract error; raw child text is deliberately omitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1, max_length=100)
    code: EvidenceGuardViolationCodeV2


class GuardedUnderstandingV2(DialogueV2Model):
    """Understanding output admitted through the schema/provenance boundary.

    The response is preserved exactly as the Sonnet understanding model returned
    it.  In particular, this type performs no expected-value comparison, arithmetic
    calculation, unit normalization, or verdict rewriting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    response: UnderstandingResponseV2
    evidence_matches: list[EvidenceMatchV2]


class UnderstandingEvidenceGuardError(ValueError):
    """Raised when an understanding result cannot safely change learning state."""

    def __init__(self, violations: list[EvidenceGuardViolationV2]) -> None:
        self.violations = tuple(violations)
        # Claim IDs are model-authored correlation handles and may contain raw
        # child text. Keep them available only to the in-memory repair flow and
        # omit them from exception/log strings.
        summary = ", ".join(violation.code.value for violation in self.violations)
        super().__init__(f"understanding evidence guard failed ({summary})")


def _literal_evidence_match(
    child_utterance: str,
    evidence_span: str,
) -> tuple[int, int, EvidenceMatchKindV2] | None:
    """Locate a contiguous source span using exact text or Unicode NFC only.

    The fallback deliberately scans source slices rather than transforming words,
    numbers, punctuation, whitespace, or units.  Scanning also lets us return
    boundaries in the unmodified child utterance when several source code points
    compose into one NFC character.
    """

    exact_start = child_utterance.find(evidence_span)
    if exact_start >= 0:
        return (
            exact_start,
            exact_start + len(evidence_span),
            EvidenceMatchKindV2.EXACT,
        )

    normalized_evidence = unicodedata.normalize("NFC", evidence_span)
    if normalized_evidence not in unicodedata.normalize("NFC", child_utterance):
        return None

    for source_start in range(len(child_utterance)):
        for source_end in range(source_start + 1, len(child_utterance) + 1):
            source_slice = child_utterance[source_start:source_end]
            if unicodedata.normalize("NFC", source_slice) == normalized_evidence:
                return source_start, source_end, EvidenceMatchKindV2.UNICODE_NFC
    return None


def guard_understanding_response_v2(
    request: UnderstandingRequestV2,
    response: UnderstandingResponseV2,
) -> GuardedUnderstandingV2:
    """Validate claim provenance without adjudicating semantic correctness.

    Pydantic has already enforced the response schema and unique claim IDs.  This
    boundary checks only server-owned graph membership and literal evidence in the
    current raw utterance.  Any failure rejects the whole understanding result so
    the caller can retry once without partially advancing state.
    """

    fact_ids = frozenset(request.claimable_graph.fact_ids)
    relation_ids = frozenset(request.claimable_graph.relation_ids)
    violations: list[EvidenceGuardViolationV2] = []
    evidence_matches: list[EvidenceMatchV2] = []

    for claim in response.claims:
        if isinstance(claim, FactUnderstandingClaimV2) and claim.fact_id not in fact_ids:
            violations.append(
                EvidenceGuardViolationV2(
                    claim_id=claim.claim_id,
                    code=EvidenceGuardViolationCodeV2.UNKNOWN_FACT_ID,
                )
            )
        elif (
            isinstance(claim, RelationUnderstandingClaimV2)
            and claim.relation_id not in relation_ids
        ):
            violations.append(
                EvidenceGuardViolationV2(
                    claim_id=claim.claim_id,
                    code=EvidenceGuardViolationCodeV2.UNKNOWN_RELATION_ID,
                )
            )
        elif (
            isinstance(claim, AuxiliaryUnderstandingClaimV2)
            and not request.claimable_graph.open_auxiliary_claims
        ):
            violations.append(
                EvidenceGuardViolationV2(
                    claim_id=claim.claim_id,
                    code=EvidenceGuardViolationCodeV2.AUXILIARY_CLAIMS_DISABLED,
                )
            )

        evidence_match = _literal_evidence_match(
            request.child_utterance,
            claim.evidence_span,
        )
        if evidence_match is None:
            violations.append(
                EvidenceGuardViolationV2(
                    claim_id=claim.claim_id,
                    code=EvidenceGuardViolationCodeV2.EVIDENCE_NOT_LITERAL,
                )
            )
            continue

        source_start, source_end, match_kind = evidence_match
        evidence_matches.append(
            EvidenceMatchV2(
                claim_id=claim.claim_id,
                source_start=source_start,
                source_end=source_end,
                source_text=request.child_utterance[source_start:source_end],
                match_kind=match_kind,
            )
        )

    if violations:
        raise UnderstandingEvidenceGuardError(violations)

    return GuardedUnderstandingV2(
        response=response.model_copy(deep=True),
        evidence_matches=evidence_matches,
    )
