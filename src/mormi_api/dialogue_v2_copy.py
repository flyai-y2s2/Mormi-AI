from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .copy_cache import (
    CachedCopyArtifact,
    CopyCacheAcquireState,
    CopyGenerationLease,
    GeneratedCopyCacheRepository,
    build_stable_copy_cache_key,
)
from .dialogue_v2_content import (
    CopySlotV2,
    RequiredHomeTeachingPackV2,
    load_required_home_content_catalog_v2,
)
from .dialogue_v2_ledger import content_pack_hash_v2
from .dialogue_v2_speaker import (
    STABLE_COPY_JOINT_ACTION_V2,
    STABLE_COPY_L0_GENERATION_BRIEF_V2,
    SpeakerAllowedFactV2,
    SpeakerTargetV2,
    StableCopyOutputV2,
    StableCopyPlanV2,
    StableCopyTransitionV2,
    stable_copy_output_violation_v2,
)
from .schemas import CanonicalValueV2, ExpressionLevel, HintLevel, NumberValueV2, utc_now
from .settings import Settings

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_KEY_DIGEST_PATTERN = r"^[0-9a-f]{16}$"
_ARTIFACT_SCHEMA_VERSION: Literal["stable-copy-cache-artifact-v1"] = (
    "stable-copy-cache-artifact-v1"
)
_STABLE_COPY_MAX_TOKENS = 220
_STABLE_COPY_TEMPERATURE = 0.25
STABLE_COPY_PLAN_SCHEMA_VERSION_V2: Literal["stable-copy-plan-set-v1"] = (
    "stable-copy-plan-set-v1"
)
STABLE_COPY_PLAN_COMPILER_VERSION_V2: Literal["stable-copy-plan-compiler-v1"] = (
    "stable-copy-plan-compiler-v1"
)
_REQUIRED_STABLE_COPY_PLAN_COUNT_V2 = 5
_SAFE_L0_MORMI_FALLBACK_V2 = "도움 카드를 보면서 나와 같이 해 줄 수 있어?"


class StableCopyGeneratorV2(Protocol):
    async def generate_stable_copy_v2(
        self,
        plan: StableCopyPlanV2,
    ) -> StableCopyOutputV2: ...


class StableCopyResolutionError(RuntimeError):
    pass


class StableCopyOutputViolationError(StableCopyResolutionError):
    """A structurally valid artifact failed the current child-output firewall."""


class StableCopyArtifactMetadataV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_schema_version: Literal["stable-copy-cache-artifact-v1"] = (
        _ARTIFACT_SCHEMA_VERSION
    )
    origin: Literal["generated", "reviewed_fallback"]
    pack_id: str = Field(min_length=1, max_length=160)
    content_version: int = Field(ge=1)
    pack_hash: str = Field(pattern=_SHA256_PATTERN)
    copy_slot: str = Field(min_length=1, max_length=160)
    plan_sha256: str = Field(pattern=_SHA256_PATTERN)
    prompt_version: str = Field(min_length=1, max_length=200)
    output_schema_version: str = Field(min_length=1, max_length=200)
    validator_version: str = Field(min_length=1, max_length=200)
    model_id: str = Field(min_length=1, max_length=200)
    effort: str = Field(min_length=1, max_length=30)
    generation_config: dict[str, str | int | float | bool]
    generated_at: datetime | None = None


class StableCopyCacheArtifactV2(BaseModel):
    """Versioned envelope stored through the generic JSON cache repository."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_schema_version: Literal["stable-copy-cache-artifact-v1"] = (
        _ARTIFACT_SCHEMA_VERSION
    )
    output: StableCopyOutputV2
    metadata: StableCopyArtifactMetadataV2


StableCopyResolutionStatusV2 = Literal[
    "pinned",
    "hit",
    "generated",
    "seeded_reviewed_fallback",
    "contended_fallback",
    "generation_fallback",
    "reviewed_fallback",
]
StableCopyPurposeV2 = Literal[
    "initial_help",
    "l2_question",
    "l0_intro",
    "l0_action",
]


class StableCopyResolutionV2(BaseModel):
    """Child-safe text plus everything needed to pin and audit its source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_schema_version: Literal["stable-copy-resolution-v1"] = (
        "stable-copy-resolution-v1"
    )
    text: str = Field(min_length=1)
    mood: Literal["curious", "listening", "thinking", "relieved", "celebrating"]
    dialogue_act: str = Field(min_length=1, max_length=100)
    asked_fact_ids: list[str] = Field(default_factory=list, max_length=8)
    asked_relation_ids: list[str] = Field(default_factory=list, max_length=8)
    status: StableCopyResolutionStatusV2
    key_digest: str = Field(pattern=_KEY_DIGEST_PATTERN)
    full_cache_key: str = Field(pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_metadata: StableCopyArtifactMetadataV2
    fallback_reason: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def key_digest_matches_full_key(self) -> StableCopyResolutionV2:
        if self.key_digest != self.full_cache_key[:16]:
            raise ValueError("stable-copy key digest must prefix the full cache key")
        return self

    def as_output(self) -> StableCopyOutputV2:
        return StableCopyOutputV2(
            text=self.text,
            mood=self.mood,
            dialogue_act=self.dialogue_act,
            asked_fact_ids=list(self.asked_fact_ids),
            asked_relation_ids=list(self.asked_relation_ids),
        )


@dataclass(frozen=True, slots=True)
class StableCopyWorkItemV2:
    plan: StableCopyPlanV2
    reviewed_fallback: str
    pack_hash: str
    output_firewall: StableCopyOutputFirewallV2


@dataclass(frozen=True, slots=True)
class StableCopyOutputFirewallV2:
    """Pack-owned output deny material kept outside model and cache payloads."""

    forbidden_values: tuple[CanonicalValueV2, ...] = ()
    forbidden_surfaces: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledStableCopyPlanSetV2:
    """Exact plan payloads pinned when one V2 conversation is created."""

    schema_version: Literal["stable-copy-plan-set-v1"]
    compiler_version: Literal["stable-copy-plan-compiler-v1"]
    plan_set_hash: str
    plans: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class PrewarmFailureV2:
    pack_id: str
    copy_slot: str
    status: str
    reason: str
    key_digest: str


@dataclass(frozen=True, slots=True)
class PrewarmReportV2:
    expected: int
    ready: int
    failures: tuple[PrewarmFailureV2, ...]

    @property
    def succeeded(self) -> bool:
        return self.ready == self.expected and not self.failures

    def as_json(self) -> str:
        return json.dumps(
            {
                "expected": self.expected,
                "ready": self.ready,
                "succeeded": self.succeeded,
                "failures": [asdict(failure) for failure in self.failures],
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def _canonical_sha256(value: object) -> str:
    material = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _plan_sha256(plan: StableCopyPlanV2) -> str:
    return _canonical_sha256(plan.model_dump(mode="json"))


def _target_for_slot(
    slot: CopySlotV2,
    *,
    opaque_ids: bool = False,
) -> SpeakerTargetV2:
    fact_ids = [
        f"fact_{index}"
        if opaque_ids
        else target.target_id
        for index, target in enumerate(
            (item for item in slot.targets if item.target_kind == "fact"),
            start=1,
        )
    ]
    relation_ids = [
        f"relation_{index}"
        if opaque_ids
        else target.target_id
        for index, target in enumerate(
            (item for item in slot.targets if item.target_kind == "relation"),
            start=1,
        )
    ]
    if fact_ids and relation_ids:
        ask_mode: Literal["answer", "reason_or_method", "answer_and_method"] = (
            "answer_and_method"
        )
    elif fact_ids:
        ask_mode = "answer"
    else:
        ask_mode = "reason_or_method"
    return SpeakerTargetV2(
        fact_ids=fact_ids,
        relation_ids=relation_ids,
        ask_mode=ask_mode,
        success_criteria_ids=[],
    )


def _transition_for_purpose(
    purpose: StableCopyPurposeV2,
) -> StableCopyTransitionV2:
    if purpose == "initial_help":
        return StableCopyTransitionV2(
            from_expression_level=ExpressionLevel.L4,
            from_hint_level=HintLevel.H0,
            to_expression_level=ExpressionLevel.L3,
            to_hint_level=HintLevel.H1,
        )
    if purpose == "l2_question":
        return StableCopyTransitionV2(
            from_expression_level=ExpressionLevel.L3,
            from_hint_level=HintLevel.H1,
            to_expression_level=ExpressionLevel.L2,
            to_hint_level=HintLevel.H2,
        )
    return StableCopyTransitionV2(
        from_expression_level=ExpressionLevel.L2,
        from_hint_level=HintLevel.H2,
        to_expression_level=ExpressionLevel.L0,
        to_hint_level=HintLevel.H3,
    )


def _dialogue_act_for_purpose(purpose: StableCopyPurposeV2) -> str:
    return {
        "initial_help": "offer_initial_help",
        "l2_question": "present_l2_choices",
        "l0_intro": "start_joint_support",
        "l0_action": "guide_joint_action",
    }[purpose]


def _visible_facts_for_slot(
    pack: RequiredHomeTeachingPackV2,
    slot: CopySlotV2,
) -> list[SpeakerAllowedFactV2]:
    if slot.purpose in {"l0_intro", "l0_action"}:
        # H3 is UI support for the child, not Mormi's knowledge. The cached
        # Mormi copy only needs to ask for joint work and receives no card
        # values, revealed answer, equation, or method.
        return []
    facts = [
        SpeakerAllowedFactV2(
            fact_id=fact.fact_id,
            value=fact.value,
            # The typed value and fact ID carry the reviewed meaning.  Reusing
            # a label such as "한 개 가격" here would introduce the unrelated
            # numeric literal 1 into a 4,000-won fact and fail the closed-world
            # speaker guard.
            speaker_text="문제 화면에 주어진 정보",
        )
        for fact in pack.reasoning_graph.facts
        if fact.initially_visible
    ]
    if "한 명" in slot.reviewed_fallback:
        # "한 명이 얼마" denotes the reviewed per-person basis, not the
        # hidden money answer.  Declare that visible linguistic unit explicitly
        # instead of weakening the numeric guard for every stable-copy plan.
        facts.append(
            SpeakerAllowedFactV2(
                fact_id="stable-context.one-person",
                value=NumberValueV2(value=1, unit="명"),
                speaker_text="한 명 기준",
            )
        )
    return facts


def build_stable_copy_output_firewall_v2(
    pack: RequiredHomeTeachingPackV2,
    slot: CopySlotV2,
) -> StableCopyOutputFirewallV2:
    """Compile hidden-answer output guards without adding truth to a model plan.

    Every Mormi utterance keeps unresolved target truth outside its authority.
    H3 may reveal that truth in the child-facing UI, but it does not make the
    card a knowledge source for the stable-copy model or its cached output.
    """

    target_pairs = {
        (target.target_kind, target.target_id) for target in slot.targets
    }
    target_fact_ids = {
        target_id
        for target_kind, target_id in target_pairs
        if target_kind == "fact"
    }
    values: list[CanonicalValueV2] = []
    surfaces: list[str] = []
    for fact in pack.reasoning_graph.facts:
        if fact.fact_id not in target_fact_ids:
            continue
        values.append(fact.value)
        surfaces.extend(fact.accepted_surface_forms)
    for l2_plan in pack.l2_plans:
        if (l2_plan.target.target_kind, l2_plan.target.target_id) not in target_pairs:
            continue
        surfaces.extend(
            choice.label
            for choice in l2_plan.choices
            if choice.effect.verdict == "correct"
        )
    return StableCopyOutputFirewallV2(
        forbidden_values=tuple(values),
        forbidden_surfaces=tuple(dict.fromkeys(surfaces)),
    )


def build_stable_copy_work_items_v2(
    pack: RequiredHomeTeachingPackV2,
) -> list[StableCopyWorkItemV2]:
    """Compile every reviewed copy slot into the PII-free speaker contract."""

    pack_hash = content_pack_hash_v2(pack)
    l2_plans = {plan.copy_slot: plan for plan in pack.l2_plans}
    items: list[StableCopyWorkItemV2] = []
    for slot in pack.copy_slots:
        choice_labels = (
            [choice.label for choice in l2_plans[slot.copy_slot].choices]
            if slot.purpose == "l2_question"
            else []
        )
        plan = StableCopyPlanV2(
            purpose=slot.purpose,
            pack_id=pack.pack_id,
            copy_slot=slot.copy_slot,
            content_version=pack.content_version,
            dialogue_act=_dialogue_act_for_purpose(slot.purpose),
            target=_target_for_slot(
                slot,
                opaque_ids=slot.purpose in {"l0_intro", "l0_action"},
            ),
            transition=_transition_for_purpose(slot.purpose),
            visible_facts=_visible_facts_for_slot(pack, slot),
            choice_labels=choice_labels,
            joint_action=(
                STABLE_COPY_JOINT_ACTION_V2
                if slot.purpose in {"l0_intro", "l0_action"}
                else None
            ),
            generation_brief=(
                STABLE_COPY_L0_GENERATION_BRIEF_V2
                if slot.purpose in {"l0_intro", "l0_action"}
                else slot.generation_brief
            ),
            reveal_policy="hidden",
        )
        items.append(
            StableCopyWorkItemV2(
                plan=plan,
                # The content slot's L0 action copy belongs to the visible UI
                # and may contain its equation or values. It must not be used
                # as a Mormi generation/caching fallback.
                reviewed_fallback=(
                    _SAFE_L0_MORMI_FALLBACK_V2
                    if slot.purpose in {"l0_intro", "l0_action"}
                    else slot.reviewed_fallback
                ),
                pack_hash=pack_hash,
                output_firewall=build_stable_copy_output_firewall_v2(pack, slot),
            )
        )
    return items


def _validated_plan_payloads_v2(
    plan_payloads: Mapping[str, object],
) -> dict[str, StableCopyPlanV2]:
    if len(plan_payloads) != _REQUIRED_STABLE_COPY_PLAN_COUNT_V2:
        raise StableCopyResolutionError(
            "pinned stable-copy plan set must contain exactly five plans"
        )
    plans: dict[str, StableCopyPlanV2] = {}
    for copy_slot, payload in plan_payloads.items():
        if not isinstance(copy_slot, str):
            raise StableCopyResolutionError("pinned stable-copy plan key must be a string")
        try:
            plan = (
                payload
                if isinstance(payload, StableCopyPlanV2)
                else StableCopyPlanV2.model_validate(payload)
            )
        except (TypeError, ValueError) as error:
            raise StableCopyResolutionError(
                f"pinned stable-copy plan is invalid: {copy_slot}"
            ) from error
        if plan.copy_slot != copy_slot:
            raise StableCopyResolutionError(
                "pinned stable-copy plan key does not match its copy slot"
            )
        plans[copy_slot] = plan
    return plans


def stable_copy_plan_set_hash_v2(
    plan_payloads: Mapping[str, object],
    *,
    pack_hash: str,
    schema_version: str = STABLE_COPY_PLAN_SCHEMA_VERSION_V2,
    compiler_version: str = STABLE_COPY_PLAN_COMPILER_VERSION_V2,
) -> str:
    """Hash the exact versioned plan set and its pinned content revision."""

    if not re.fullmatch(_SHA256_PATTERN, pack_hash):
        raise StableCopyResolutionError("stable-copy plan set has an invalid pack hash")
    _validated_plan_payloads_v2(plan_payloads)
    # Hash the payload exactly as persisted, rather than re-dumping it through
    # today's model defaults. This preserves the identity of in-flight plan
    # sets when an additive field (for example fact provenance) is introduced.
    persisted_plans = {
        copy_slot: (
            payload.model_dump(mode="json")
            if isinstance(payload, StableCopyPlanV2)
            else payload
        )
        for copy_slot, payload in sorted(plan_payloads.items())
    }
    return _canonical_sha256(
        {
            "schema_version": schema_version,
            "compiler_version": compiler_version,
            "pack_hash": pack_hash,
            "plans": persisted_plans,
        }
    )


def compile_stable_copy_plan_set_v2(
    pack: RequiredHomeTeachingPackV2,
) -> CompiledStableCopyPlanSetV2:
    """Compile all five plans once for conversation-level snapshotting."""

    work_items = build_stable_copy_work_items_v2(pack)
    if len(work_items) != _REQUIRED_STABLE_COPY_PLAN_COUNT_V2:
        raise StableCopyResolutionError(
            "V2 conversation requires exactly five stable-copy plans"
        )
    plans = {
        item.plan.copy_slot: item.plan.model_dump(mode="json") for item in work_items
    }
    if len(plans) != len(work_items):
        raise StableCopyResolutionError("stable-copy compiler produced duplicate copy slots")
    pack_hash = content_pack_hash_v2(pack)
    return CompiledStableCopyPlanSetV2(
        schema_version=STABLE_COPY_PLAN_SCHEMA_VERSION_V2,
        compiler_version=STABLE_COPY_PLAN_COMPILER_VERSION_V2,
        plan_set_hash=stable_copy_plan_set_hash_v2(plans, pack_hash=pack_hash),
        plans=plans,
    )


def validate_pinned_stable_copy_plan_set_v2(
    pack: RequiredHomeTeachingPackV2,
    *,
    pack_hash: str,
    schema_version: str,
    compiler_version: str,
    plan_set_hash: str,
    plan_payloads: Mapping[str, object],
) -> dict[str, StableCopyPlanV2]:
    """Validate a stored plan set without invoking today's plan compiler."""

    if schema_version != STABLE_COPY_PLAN_SCHEMA_VERSION_V2:
        raise StableCopyResolutionError(
            f"unsupported pinned stable-copy plan schema: {schema_version}"
        )
    if compiler_version != STABLE_COPY_PLAN_COMPILER_VERSION_V2:
        raise StableCopyResolutionError(
            f"unsupported pinned stable-copy plan compiler: {compiler_version}"
        )
    plans = _validated_plan_payloads_v2(plan_payloads)
    expected_slots = {slot.copy_slot: slot for slot in pack.copy_slots}
    if len(expected_slots) != _REQUIRED_STABLE_COPY_PLAN_COUNT_V2:
        raise StableCopyResolutionError(
            "pinned content pack must contain exactly five stable-copy slots"
        )
    if set(plans) != set(expected_slots):
        raise StableCopyResolutionError(
            "pinned stable-copy plans do not cover the pinned content slots"
        )
    expected_hash = stable_copy_plan_set_hash_v2(
        plan_payloads,
        pack_hash=pack_hash,
        schema_version=schema_version,
        compiler_version=compiler_version,
    )
    if plan_set_hash != expected_hash:
        raise StableCopyResolutionError("pinned stable-copy plan set hash mismatch")

    for copy_slot, plan in plans.items():
        slot = expected_slots[copy_slot]
        if (
            plan.pack_id != pack.pack_id
            or plan.content_version != pack.content_version
            or plan.purpose != slot.purpose
        ):
            raise StableCopyResolutionError(
                "pinned stable-copy plan does not match the pinned content slot"
            )
        expected_targets = [_target_for_slot(slot)]
        if slot.purpose in {"l0_intro", "l0_action"}:
            expected_targets.append(_target_for_slot(slot, opaque_ids=True))
        if not any(
            plan.target.fact_ids == expected_target.fact_ids
            and plan.target.relation_ids == expected_target.relation_ids
            and plan.target.ask_mode == expected_target.ask_mode
            for expected_target in expected_targets
        ):
            raise StableCopyResolutionError(
                "pinned stable-copy plan target does not match the pinned content slot"
            )
    return {copy_slot: plan.model_copy(deep=True) for copy_slot, plan in plans.items()}


class StableCopyResolverV2:
    def __init__(
        self,
        repository: GeneratedCopyCacheRepository,
        generator: StableCopyGeneratorV2,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.generator = generator
        self.settings = settings

    def _generation_config(self) -> dict[str, str | int | float | bool]:
        return {
            "effort": self.settings.stable_copy_effort,
            "max_tokens": _STABLE_COPY_MAX_TOKENS,
            "temperature": _STABLE_COPY_TEMPERATURE,
            "timeout_seconds": self.settings.stable_copy_timeout_seconds,
        }

    def _cache_key(self, plan: StableCopyPlanV2, pack_hash: str) -> str:
        return build_stable_copy_cache_key(
            content_revision=f"{plan.pack_id}:v{plan.content_version}",
            content_hash=pack_hash,
            copy_slot_id=plan.copy_slot,
            locale=plan.locale,
            prompt_version=self.settings.stable_copy_prompt_version,
            schema_version=self.settings.stable_copy_schema_version,
            validator_version=self.settings.stable_copy_validator_version,
            model_id=self.settings.stable_copy_model,
            generation_config=self._generation_config(),
            generation_plan=plan.model_dump(mode="json"),
        )

    def _metadata(
        self,
        plan: StableCopyPlanV2,
        pack_hash: str,
        *,
        origin: Literal["generated", "reviewed_fallback"],
        generated_at: datetime | None,
    ) -> StableCopyArtifactMetadataV2:
        return StableCopyArtifactMetadataV2(
            origin=origin,
            pack_id=plan.pack_id,
            content_version=plan.content_version,
            pack_hash=pack_hash,
            copy_slot=plan.copy_slot,
            plan_sha256=_plan_sha256(plan),
            prompt_version=self.settings.stable_copy_prompt_version,
            output_schema_version=self.settings.stable_copy_schema_version,
            validator_version=self.settings.stable_copy_validator_version,
            model_id=self.settings.stable_copy_model,
            effort=self.settings.stable_copy_effort,
            generation_config=self._generation_config(),
            generated_at=generated_at,
        )

    @staticmethod
    def _fallback_output(
        plan: StableCopyPlanV2,
        reviewed_fallback: str,
        output_firewall: StableCopyOutputFirewallV2 | None = None,
    ) -> StableCopyOutputV2:
        mood: Literal["curious", "listening", "thinking"]
        if plan.purpose in {"initial_help", "l2_question"}:
            mood = "curious"
        elif plan.purpose == "l0_intro":
            mood = "thinking"
        else:
            mood = "listening"
        output = StableCopyOutputV2(
            text=reviewed_fallback.strip(),
            mood=mood,
            dialogue_act=plan.dialogue_act,
            asked_fact_ids=list(plan.target.fact_ids),
            asked_relation_ids=list(plan.target.relation_ids),
        )
        firewall = output_firewall or StableCopyOutputFirewallV2()
        violation = stable_copy_output_violation_v2(
            output,
            plan,
            forbidden_values=firewall.forbidden_values,
            forbidden_surfaces=firewall.forbidden_surfaces,
        )
        if violation is not None:
            raise StableCopyResolutionError(
                f"reviewed stable-copy fallback violated its plan: {violation}"
            )
        return output

    @staticmethod
    def _result(
        output: StableCopyOutputV2,
        *,
        status: StableCopyResolutionStatusV2,
        cache_key: str,
        metadata: StableCopyArtifactMetadataV2,
        artifact_sha256: str | None,
        fallback_reason: str | None = None,
    ) -> StableCopyResolutionV2:
        return StableCopyResolutionV2(
            text=output.text.strip(),
            mood=output.mood,
            dialogue_act=output.dialogue_act,
            asked_fact_ids=list(output.asked_fact_ids),
            asked_relation_ids=list(output.asked_relation_ids),
            status=status,
            key_digest=cache_key[:16],
            full_cache_key=cache_key,
            artifact_sha256=artifact_sha256,
            artifact_metadata=metadata,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _validated_output(
        output: StableCopyOutputV2,
        plan: StableCopyPlanV2,
        output_firewall: StableCopyOutputFirewallV2,
    ) -> StableCopyOutputV2:
        violation = stable_copy_output_violation_v2(
            output,
            plan,
            forbidden_values=output_firewall.forbidden_values,
            forbidden_surfaces=output_firewall.forbidden_surfaces,
        )
        if violation is not None:
            raise StableCopyOutputViolationError(
                f"stable-copy output violated its plan: {violation}"
            )
        return output.model_copy(deep=True)

    def _from_cached_artifact(
        self,
        artifact: CachedCopyArtifact,
        *,
        plan: StableCopyPlanV2,
        pack_hash: str,
        status: Literal["hit", "generated", "seeded_reviewed_fallback"],
        output_firewall: StableCopyOutputFirewallV2,
    ) -> StableCopyResolutionV2:
        envelope = StableCopyCacheArtifactV2.model_validate(artifact.artifact)
        expected = self._metadata(
            plan,
            pack_hash,
            origin=envelope.metadata.origin,
            generated_at=envelope.metadata.generated_at,
        )
        if envelope.metadata != expected:
            raise StableCopyResolutionError("stable-copy cache metadata mismatch")
        output = self._validated_output(envelope.output, plan, output_firewall)
        return self._result(
            output,
            status=status,
            cache_key=artifact.cache_key,
            metadata=envelope.metadata,
            artifact_sha256=artifact.artifact_sha256,
        )

    def _from_pinned_snapshot(
        self,
        pinned_snapshot: StableCopyResolutionV2 | Mapping[str, object],
        *,
        plan: StableCopyPlanV2,
        pack_hash: str,
        output_firewall: StableCopyOutputFirewallV2,
    ) -> StableCopyResolutionV2:
        snapshot = (
            pinned_snapshot
            if isinstance(pinned_snapshot, StableCopyResolutionV2)
            else StableCopyResolutionV2.model_validate(pinned_snapshot)
        )
        metadata = snapshot.artifact_metadata
        if (
            metadata.pack_id != plan.pack_id
            or metadata.content_version != plan.content_version
            or metadata.pack_hash != pack_hash
            or metadata.copy_slot != plan.copy_slot
            or metadata.plan_sha256 != _plan_sha256(plan)
        ):
            raise StableCopyResolutionError("pinned stable-copy snapshot does not match plan")
        self._validated_output(snapshot.as_output(), plan, output_firewall)
        return snapshot.model_copy(update={"status": "pinned"}, deep=True)

    def _fallback_result(
        self,
        plan: StableCopyPlanV2,
        reviewed_fallback: str,
        pack_hash: str,
        cache_key: str,
        output_firewall: StableCopyOutputFirewallV2,
        *,
        status: Literal[
            "contended_fallback",
            "generation_fallback",
            "reviewed_fallback",
        ],
        reason: str,
    ) -> StableCopyResolutionV2:
        return self._result(
            self._fallback_output(plan, reviewed_fallback, output_firewall),
            status=status,
            cache_key=cache_key,
            metadata=self._metadata(
                plan,
                pack_hash,
                origin="reviewed_fallback",
                generated_at=None,
            ),
            artifact_sha256=None,
            fallback_reason=reason,
        )

    async def _mark_failure(
        self,
        lease: CopyGenerationLease,
        error_code: str,
    ) -> None:
        try:
            await self.repository.fail(lease, error_code=error_code)
        except Exception:
            # The child-safe fallback must not depend on recording retry state.
            # The lease expires and can still be reclaimed if persistence failed.
            return

    async def resolve(
        self,
        plan: StableCopyPlanV2,
        *,
        reviewed_fallback: str,
        pack_hash: str,
        output_firewall: StableCopyOutputFirewallV2,
        pinned_snapshot: StableCopyResolutionV2 | Mapping[str, object] | None = None,
    ) -> StableCopyResolutionV2:
        if (
            plan.purpose in {"l0_intro", "l0_action"}
            and not plan.is_safe_l0_generation_plan()
        ):
            # Rolling compatibility: old conversations may contain a pinned
            # L0 plan with the card answer, equation and exact joint action.
            # It may be parsed for state recovery, but never reaches cache or
            # a model. Return only the new content-free reviewed utterance.
            return self._fallback_result(
                plan,
                _SAFE_L0_MORMI_FALLBACK_V2,
                pack_hash,
                self._cache_key(plan, pack_hash),
                output_firewall,
                status="reviewed_fallback",
                reason="LEGACY_L0_MODEL_BYPASSED",
            )
        if pinned_snapshot is not None:
            try:
                return self._from_pinned_snapshot(
                    pinned_snapshot,
                    plan=plan,
                    pack_hash=pack_hash,
                    output_firewall=output_firewall,
                )
            except StableCopyOutputViolationError:
                # A conversation may have pinned text under an older, weaker
                # validator.  Keep metadata/plan mismatches fail-closed, but
                # replace child-unsafe copy with the reviewed fallback bound to
                # the same pinned pack and plan.
                return self._fallback_result(
                    plan,
                    reviewed_fallback,
                    pack_hash,
                    self._cache_key(plan, pack_hash),
                    output_firewall,
                    status="reviewed_fallback",
                    reason="PINNED_OUTPUT_GUARD_REJECTED",
                )

        cache_key = self._cache_key(plan, pack_hash)
        fallback = self._fallback_output(plan, reviewed_fallback, output_firewall)
        try:
            ready = await self.repository.get_ready(cache_key)
        except Exception:
            return self._fallback_result(
                plan,
                reviewed_fallback,
                pack_hash,
                cache_key,
                output_firewall,
                status="reviewed_fallback",
                reason="CACHE_LOOKUP_FAILED",
            )
        if ready is not None:
            try:
                return self._from_cached_artifact(
                    ready,
                    plan=plan,
                    pack_hash=pack_hash,
                    status="hit",
                    output_firewall=output_firewall,
                )
            except (StableCopyResolutionError, ValueError):
                return self._result(
                    fallback,
                    status="reviewed_fallback",
                    cache_key=cache_key,
                    metadata=self._metadata(
                        plan,
                        pack_hash,
                        origin="reviewed_fallback",
                        generated_at=None,
                    ),
                    artifact_sha256=None,
                    fallback_reason="CACHE_ARTIFACT_INVALID",
                )

        try:
            acquired = await self.repository.acquire(cache_key)
        except Exception:
            return self._fallback_result(
                plan,
                reviewed_fallback,
                pack_hash,
                cache_key,
                output_firewall,
                status="reviewed_fallback",
                reason="CACHE_ACQUIRE_FAILED",
            )
        if acquired.state is CopyCacheAcquireState.READY:
            assert acquired.artifact is not None
            try:
                return self._from_cached_artifact(
                    acquired.artifact,
                    plan=plan,
                    pack_hash=pack_hash,
                    status="hit",
                    output_firewall=output_firewall,
                )
            except (StableCopyResolutionError, ValueError):
                return self._fallback_result(
                    plan,
                    reviewed_fallback,
                    pack_hash,
                    cache_key,
                    output_firewall,
                    status="reviewed_fallback",
                    reason="CACHE_ARTIFACT_INVALID",
                )
        if acquired.state in {CopyCacheAcquireState.BUSY, CopyCacheAcquireState.BACKOFF}:
            return self._fallback_result(
                plan,
                reviewed_fallback,
                pack_hash,
                cache_key,
                output_firewall,
                status="contended_fallback",
                reason=(
                    "CACHE_BUSY"
                    if acquired.state is CopyCacheAcquireState.BUSY
                    else "CACHE_BACKOFF"
                ),
            )
        assert acquired.lease is not None
        lease = acquired.lease

        try:
            candidate = await asyncio.wait_for(
                self.generator.generate_stable_copy_v2(plan),
                timeout=self.settings.stable_copy_timeout_seconds,
            )
            candidate = StableCopyOutputV2.model_validate(candidate)
        except TimeoutError:
            await self._mark_failure(lease, "MODEL_TIMEOUT")
            return self._fallback_result(
                plan,
                reviewed_fallback,
                pack_hash,
                cache_key,
                output_firewall,
                status="generation_fallback",
                reason="MODEL_TIMEOUT",
            )
        except Exception:
            await self._mark_failure(lease, "GENERATION_FAILED")
            return self._fallback_result(
                plan,
                reviewed_fallback,
                pack_hash,
                cache_key,
                output_firewall,
                status="generation_fallback",
                reason="GENERATION_FAILED",
            )

        violation = stable_copy_output_violation_v2(
            candidate,
            plan,
            forbidden_values=output_firewall.forbidden_values,
            forbidden_surfaces=output_firewall.forbidden_surfaces,
        )
        if violation is not None:
            if plan.purpose in {"l0_intro", "l0_action"}:
                # L0 copy is deliberately reviewed-only: the child-facing text
                # may ask for joint action, but it must never learn or restate
                # anything from the fully revealed H3 card.  If a generated
                # draft crosses that boundary, persist the reviewed fallback as
                # the immutable artifact instead of retrying unsafe drafts.
                metadata = self._metadata(
                    plan,
                    pack_hash,
                    origin="reviewed_fallback",
                    generated_at=utc_now(),
                )
                envelope = StableCopyCacheArtifactV2(
                    output=fallback,
                    metadata=metadata,
                )
                try:
                    completed = await self.repository.complete(
                        lease,
                        artifact=envelope.model_dump(mode="json"),
                    )
                except Exception:
                    completed = None
                if completed is not None:
                    return self._from_cached_artifact(
                        completed,
                        plan=plan,
                        pack_hash=pack_hash,
                        status="seeded_reviewed_fallback",
                        output_firewall=output_firewall,
                    )
                try:
                    winner = await self.repository.get_ready(cache_key)
                except Exception:
                    winner = None
                if winner is not None:
                    return self._from_cached_artifact(
                        winner,
                        plan=plan,
                        pack_hash=pack_hash,
                        status="hit",
                        output_firewall=output_firewall,
                    )
            await self._mark_failure(lease, "OUTPUT_GUARD_REJECTED")
            return self._fallback_result(
                plan,
                reviewed_fallback,
                pack_hash,
                cache_key,
                output_firewall,
                status="generation_fallback",
                reason="OUTPUT_GUARD_REJECTED",
            )

        metadata = self._metadata(
            plan,
            pack_hash,
            origin="generated",
            generated_at=utc_now(),
        )
        envelope = StableCopyCacheArtifactV2(
            output=candidate,
            metadata=metadata,
        )
        try:
            completed = await self.repository.complete(
                lease,
                artifact=envelope.model_dump(mode="json"),
            )
        except Exception:
            completed = None
        if completed is not None:
            return self._from_cached_artifact(
                completed,
                plan=plan,
                pack_hash=pack_hash,
                status="generated",
                output_firewall=output_firewall,
            )

        try:
            winner = await self.repository.get_ready(cache_key)
        except Exception:
            winner = None
        if winner is not None:
            try:
                return self._from_cached_artifact(
                    winner,
                    plan=plan,
                    pack_hash=pack_hash,
                    status="hit",
                    output_firewall=output_firewall,
                )
            except (StableCopyResolutionError, ValueError):
                pass
        return self._fallback_result(
            plan,
            reviewed_fallback,
            pack_hash,
            cache_key,
            output_firewall,
            status="contended_fallback",
            reason="LEASE_LOST",
        )


async def prewarm_required_home_copy_v2(
    resolver: StableCopyResolverV2,
    repository: GeneratedCopyCacheRepository,
) -> PrewarmReportV2:
    """Generate every reviewed slot and prove a durable ready row exists."""

    catalog = load_required_home_content_catalog_v2()
    work_items = [
        item
        for pack in sorted(catalog.packs, key=lambda candidate: candidate.pack_id)
        for item in build_stable_copy_work_items_v2(pack)
    ]
    failures: list[PrewarmFailureV2] = []
    ready_count = 0
    for item in work_items:
        resolution = await resolver.resolve(
            item.plan,
            reviewed_fallback=item.reviewed_fallback,
            pack_hash=item.pack_hash,
            output_firewall=item.output_firewall,
        )
        if resolution.status not in {
            "hit",
            "generated",
            "seeded_reviewed_fallback",
        }:
            failures.append(
                PrewarmFailureV2(
                    pack_id=item.plan.pack_id,
                    copy_slot=item.plan.copy_slot,
                    status=resolution.status,
                    reason=resolution.fallback_reason or "READY_ARTIFACT_INVALID",
                    key_digest=resolution.key_digest,
                )
            )
            continue
        try:
            ready_artifact = await repository.get_ready(resolution.full_cache_key)
        except Exception:
            ready_artifact = None
        if ready_artifact is None:
            failures.append(
                PrewarmFailureV2(
                    pack_id=item.plan.pack_id,
                    copy_slot=item.plan.copy_slot,
                    status=resolution.status,
                    reason=resolution.fallback_reason or "READY_ROW_MISSING",
                    key_digest=resolution.key_digest,
                )
            )
            continue
        ready_count += 1
    return PrewarmReportV2(
        expected=len(work_items),
        ready=ready_count,
        failures=tuple(failures),
    )
