from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .dialogue_v2_content import (
    CopySlotV2,
    L2ChoicePlanV2,
    RequiredHomeTeachingPackV2,
    TargetRefV2,
    required_home_content_pack_v2,
)
from .dialogue_v2_copy import (
    StableCopyOutputFirewallV2,
    StableCopyResolutionV2,
    build_stable_copy_output_firewall_v2,
    compile_stable_copy_plan_set_v2,
    validate_pinned_stable_copy_plan_set_v2,
)
from .dialogue_v2_evidence import (
    GuardedUnderstandingV2,
    UnderstandingEvidenceGuardError,
    guard_understanding_response_v2,
)
from .dialogue_v2_ledger import (
    PinnedContentSnapshotV2,
    ReasoningLedgerApplyResultV2,
    ReasoningLedgerV2,
    RelationVerificationEvidenceV2,
    StructuredVerificationEvidenceV2,
    apply_guarded_understanding_v2,
    apply_structured_progress_v2,
    empty_reasoning_ledger_v2,
    pin_content_pack_v2,
    reasoning_completion_v2,
)
from .dialogue_v2_speaker import (
    BridgePlanV2,
    ConversationResponsePlanV2,
    SpeakerAcceptedRelationV2,
    SpeakerAllowedFactV2,
    SpeakerEvidenceV2,
    SpeakerOutputV2,
    SpeakerPlanV2,
    SpeakerResponseSignalV2,
    SpeakerSupportV2,
    SpeakerTargetFocusV2,
    SpeakerTargetV2,
    StableCopyPlanV2,
    bridge_excerpt_violation_v2,
    pii_safe_bridge_excerpt_v2,
    speaker_output_violation_v2,
    stable_copy_output_violation_v2,
    validate_bridge_output_v2,
)
from .dictionary_models import dictionary_reference
from .engine import EngineProgress, EngineTurnResult
from .llm import ModelOutputError, ModelUnavailableError, validate_note_contextualization
from .schemas import (
    CanonicalValueV2,
    ChildResponse,
    ChoiceOption,
    CompletionContract,
    CompletionOutcome,
    ConversationMoveV2,
    DialogueHistoryTurn,
    DialogueRuntimeContractVersion,
    DifficultyClass,
    ExpressionLevel,
    FactUnderstandingClaimV2,
    HelpCardContract,
    HintLevel,
    InputContract,
    InputKind,
    MormiContract,
    MoveSubjectV2,
    NoResponseKindV2,
    NoteAttribution,
    NoteContextualizationContext,
    NoteEvidence,
    NoteUpdate,
    PedagogySnapshot,
    PinnedDialogueRuntimeV2,
    RelationUnderstandingClaimV2,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SessionState,
    SessionStatus,
    SpeakerRuntimeAudit,
    SupportNeed,
    TaskAnchorContract,
    TaskRelation,
    TurnContract,
    UnderstandingRequestV2,
    UnderstandingResponseV2,
    UtteranceAnalysis,
    UtteranceClassV2,
    VisualContract,
    new_id,
)

MoodV2 = Literal["curious", "listening", "thinking", "relieved", "celebrating"]
SpeechResultV2 = tuple[str, MoodV2, str, str | None]


class DialogueV2Gateway(Protocol):
    async def understand_v2(
        self,
        request: UnderstandingRequestV2,
    ) -> UnderstandingResponseV2: ...

    async def speak_v2(self, plan: SpeakerPlanV2) -> SpeakerOutputV2: ...

    async def bridge_speak_v2(self, plan: BridgePlanV2) -> SpeakerOutputV2: ...


@dataclass(frozen=True, slots=True)
class StableCopyResolution:
    text: str
    mood: MoodV2
    dialogue_act: str
    status: Literal[
        "hit",
        "generated",
        "seeded_reviewed_fallback",
        "contended_fallback",
        "generation_fallback",
        "reviewed_fallback",
    ]
    cache_key: str | None = None
    artifact: dict[str, Any] | None = None


class StableCopyResolver(Protocol):
    async def resolve(
        self,
        plan: StableCopyPlanV2,
        *,
        reviewed_fallback: str,
        pack_hash: str,
        output_firewall: StableCopyOutputFirewallV2,
        pinned_snapshot: StableCopyResolutionV2 | Mapping[str, object] | None = None,
    ) -> StableCopyResolutionV2: ...


@dataclass(frozen=True, slots=True)
class _QuestionContract:
    targets: list[TargetRefV2]
    fallback: str
    input: InputContract
    copy_slot: CopySlotV2 | None = None


@dataclass(frozen=True, slots=True)
class _TurnSemantics:
    response: UnderstandingResponseV2
    apply_result: ReasoningLedgerApplyResultV2
    guarded: GuardedUnderstandingV2 | None
    route: Literal["main", "bridge", "safety", "stable", "understanding_fallback"]
    understanding_source: Literal[
        "sonnet_low",
        "deterministic_fallback",
        "structured_choice",
        "structured_joint",
        "explicit_no_response",
        "silence_timeout",
        "asr_empty",
    ]
    attempts: int = 0
    latency_ms: int | None = None
    guard_status: Literal[
        "not_applicable",
        "passed",
        "retry_passed",
        "failed",
    ] = "not_applicable"
    structured_correct: bool | None = None
    initial_help: bool = False
    # Static, raw-free reason code for a classifier failure. It is copied only
    # into SpeakerRuntimeAudit and never includes exception or child text.
    understanding_failure_reason: Literal[
        "understanding_timeout",
        "understanding_model_unavailable",
        "understanding_model_output_invalid",
        "understanding_evidence_guard_failed",
    ] | None = None
    # Reviewed presentation-only patch selected by an opaque life-scene
    # choice. It never changes mathematical truth or the reasoning ledger.
    visual_patch: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _UnresolvedAnswerFirewall:
    """Server-only deny material; never serialized into a speaker model plan."""

    values: tuple[CanonicalValueV2, ...]
    surfaces: tuple[str, ...]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _target_key(target: TargetRefV2) -> str:
    return f"{target.target_kind}:{target.target_id}"


def _primitive(value: CanonicalValueV2) -> str | int | float | bool:
    if value.type == "money":
        return value.amount
    if value.type == "number":
        return value.value
    if value.type == "text":
        return value.text
    if value.type == "boolean":
        return value.value
    return value.choice_id


def _same_json_value(actual: object, expected: object) -> bool:
    """Compare server-authored action values without bool/number coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_json_value(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _ask_mode(targets: list[TargetRefV2]) -> Literal[
    "answer", "reason_or_method", "answer_and_method", "none"
]:
    asks = {target.ask_kind for target in targets}
    if not asks:
        return "none"
    if asks == {"answer"}:
        return "answer"
    if asks == {"reason_or_method"}:
        return "reason_or_method"
    return "answer_and_method"


class DialogueV2Engine:
    """Verdict-authoritative V2 runtime for the nine native home packs.

    It deliberately does not call any legacy slot acceptance or arithmetic
    reconciliation method.  Free-text truth flows in one direction:
    Sonnet Low -> literal evidence guard -> monotonic reasoning ledger.
    """

    def __init__(
        self,
        gateway: DialogueV2Gateway,
        *,
        copy_resolver: StableCopyResolver | None = None,
        show_internal_pedagogy: bool = False,
        classifier_timeout_seconds: float = 15.0,
        speaker_timeout_seconds: float = 10.0,
        bridge_timeout_seconds: float = 4.0,
    ) -> None:
        self.gateway = gateway
        self.copy_resolver = copy_resolver
        self.show_internal_pedagogy = show_internal_pedagogy
        self.classifier_timeout_seconds = classifier_timeout_seconds
        self.speaker_timeout_seconds = speaker_timeout_seconds
        self.bridge_timeout_seconds = bridge_timeout_seconds

    async def initialize_state(
        self,
        state: SessionState,
        *,
        curriculum_session_id: str,
        selector_reason: str,
        canary_bucket: int | None,
    ) -> TurnContract:
        if state.runtime_contract_version is not DialogueRuntimeContractVersion.VERDICT_V1:
            raise ValueError("V2 state must be pinned to verdict-v1")
        pack = required_home_content_pack_v2(curriculum_session_id)
        snapshot = pin_content_pack_v2(pack)
        ledger = empty_reasoning_ledger_v2(snapshot)
        stable_copy_plan_set = compile_stable_copy_plan_set_v2(pack)
        state.pinned_dialogue_v2 = PinnedDialogueRuntimeV2(
            pack_id=snapshot.pack_id,
            content_version=snapshot.content_version,
            source_hash=snapshot.content_hash,
            pack_snapshot=snapshot.pack_payload,
            reasoning_ledger=ledger.model_dump(mode="json"),
            stable_copy_plan_schema_version=stable_copy_plan_set.schema_version,
            stable_copy_plan_compiler_version=stable_copy_plan_set.compiler_version,
            stable_copy_plan_set_hash=stable_copy_plan_set.plan_set_hash,
            stable_copy_plans=stable_copy_plan_set.plans,
            selector_reason=selector_reason,
            canary_bucket=canary_bucket,
        )
        state.expression_level = ExpressionLevel.L4
        state.hint_level = HintLevel.H0
        state.subgoal_id = pack.initial_question.plan_id
        state.verified_slots = {}
        turn = self._build_turn(
            state,
            pack,
            ledger,
            text=pack.initial_question.reviewed_fallback,
            mood="curious",
        )
        state.current_turn_id = turn.turn_id
        return turn

    def ensure_task_anchor(
        self,
        state: SessionState,
        turn: TurnContract,
    ) -> TurnContract:
        if turn.task_anchor is not None or turn.status is SessionStatus.COMPLETED:
            return turn
        pack, _, ledger, _ = self._resolve_state(state)
        question = self._question_contract(state, pack, ledger)
        return turn.model_copy(
            update={"task_anchor": self._task_anchor(pack, question.input)},
            deep=True,
        )

    async def run_turn_stream(
        self,
        state: SessionState,
        response: ChildResponse,
        previous_question: str,
        *,
        recent_dialogue: list[DialogueHistoryTurn] | None = None,
    ) -> AsyncIterator[EngineProgress | EngineTurnResult]:
        pack, snapshot, ledger, stable_copy_plans = self._resolve_state(state)
        next_state = state.model_copy(deep=True)
        next_state.state_version += 1
        next_state.current_turn_id = None

        if response.type is ResponseType.TEXT:
            yield EngineProgress("understanding")
            semantics = await self._understand_text(
                state,
                pack,
                snapshot,
                ledger,
                response,
                previous_question,
                recent_dialogue or [],
            )
        elif response.type is ResponseType.NO_RESPONSE:
            semantics = self._no_response_semantics(
                state,
                snapshot,
                ledger,
                response,
            )
        elif response.type in {ResponseType.CHOICE, ResponseType.FILL}:
            semantics = self._choice_semantics(
                state,
                pack,
                snapshot,
                ledger,
                response,
            )
        elif response.type is ResponseType.ACTION:
            semantics = self._joint_semantics(
                state,
                pack,
                snapshot,
                ledger,
                response,
            )
        else:
            raise ValueError("V2 home teaching received an unsupported response type")

        yield EngineProgress("planning")
        self._apply_ladder_policy(next_state, semantics)
        self._store_ledger(next_state, semantics.apply_result.ledger)
        self._sync_legacy_slots(next_state, pack, semantics.apply_result.ledger)
        self._track_note_provenance(
            next_state,
            pack,
            semantics,
            before_hint=state.hint_level,
        )

        fallback_reason: str | None
        if semantics.apply_result.completion.complete:
            self._complete_state(next_state, pack, semantics)
            # Completion is a finite product-state message, so it does not
            # spend a model call.  Attribution remains separate in NoteUpdate;
            # the learner-facing sentence is always grateful and never frames
            # Mormi as having figured the task out alone.
            text = "고마워~ 네가 도와줘서 끝까지 이해할 수 있었어!"
            mood: Literal[
                "curious", "listening", "thinking", "relieved", "celebrating"
            ] = "celebrating"
            source = "reviewed_fallback"
            fallback_reason = None
            stable_status = "not_applicable"
            stable_key_digest = None
            speaker_latency = None
            emitted_dialogue_act = self._dialogue_act(semantics)
        else:
            question = self._question_contract(next_state, pack, semantics.apply_result.ledger)
            if semantics.initial_help:
                question = self._initial_help_question_contract(pack, question)
            stable = await self._stable_fallback(
                next_state,
                pack,
                semantics.apply_result.ledger,
                question,
                stable_copy_plans=stable_copy_plans,
                before_expression=state.expression_level,
                before_hint=state.hint_level,
            )
            stable_status = stable.status if stable is not None else "not_applicable"
            stable_key_digest = (
                stable.cache_key[:16] if stable is not None and stable.cache_key else None
            )
            conversational_fallback = self._conversational_fallback(
                pack,
                semantics,
                question,
                help_card_visible=next_state.hint_level is not HintLevel.H0,
                repeat_count=next_state.concept_failures,
            )

            # Stable copy is selected by an explicit route (for example the
            # help button), never merely because ladder policy moved the next
            # input into L2 or L0.  A free-text wrong answer or reverse
            # question still deserves a contextual reaction before the
            # server-owned structured re-ask.
            if semantics.route == "understanding_fallback":
                # Sonnet supplied no trustworthy semantics, so do not call a
                # second model or change pedagogy. The reviewed current-target
                # re-ask is safe to return as an ordinary committed turn.
                text = conversational_fallback
                mood = "thinking"
                source = "deterministic_validation_fallback"
                fallback_reason = semantics.understanding_failure_reason
                speaker_latency = None
                emitted_dialogue_act = self._dialogue_act(semantics)
            elif stable is not None and semantics.route == "stable":
                text, mood = stable.text, stable.mood
                source = (
                    "stable_copy_cache"
                    if stable.status in {
                        "hit",
                        "generated",
                        "seeded_reviewed_fallback",
                    }
                    else "stable_copy_fallback"
                )
                fallback_reason = None
                speaker_latency = None
                emitted_dialogue_act = stable.dialogue_act
            elif semantics.route == "safety":
                text = self._unsafe_fallback(question.fallback)
                mood = "thinking"
                source = "v2_safety_fallback"
                fallback_reason = None
                speaker_latency = None
                emitted_dialogue_act = self._dialogue_act(semantics)
            elif semantics.route == "bridge":
                yield EngineProgress("speaking")
                started = time.perf_counter()
                text, mood, source, fallback_reason = await self._bridge(
                    next_state,
                    pack,
                    semantics,
                    question,
                    conversational_fallback,
                    response.text,
                    previous_question,
                    before_hint=state.hint_level,
                )
                speaker_latency = _elapsed_ms(started)
                emitted_dialogue_act = self._dialogue_act(semantics)
            else:
                yield EngineProgress("speaking")
                started = time.perf_counter()
                text, mood, source, fallback_reason = await self._speak(
                    next_state,
                    pack,
                    semantics,
                    question,
                    conversational_fallback,
                    previous_question,
                    before_hint=state.hint_level,
                )
                speaker_latency = _elapsed_ms(started)
                emitted_dialogue_act = self._dialogue_act(semantics)

        yield EngineProgress("validating")
        next_turn = self._build_turn(
            next_state,
            pack,
            semantics.apply_result.ledger,
            text=text,
            mood=mood,
        )
        next_state.current_turn_id = next_turn.turn_id
        analysis = self._compatibility_analysis(semantics)
        runtime = SpeakerRuntimeAudit(
            dialogue_act=emitted_dialogue_act,
            speaker_source=source,  # type: ignore[arg-type]
            verifier_status="not_required",
            fallback_reason=fallback_reason,
            speaker_latency_ms=speaker_latency,
            runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
            understanding_source=semantics.understanding_source,
            understanding_attempts=semantics.attempts,
            understanding_latency_ms=semantics.latency_ms,
            evidence_guard_status=semantics.guard_status,
            new_progress=semantics.apply_result.has_new_canonical_progress,
            newly_verified_fact_ids=semantics.apply_result.new_fact_ids,
            newly_verified_relation_ids=semantics.apply_result.new_relation_ids,
            stable_copy_status=stable_status,  # type: ignore[arg-type]
            stable_copy_key_digest=stable_key_digest,
            content_pack_id=snapshot.pack_id,
            content_version=snapshot.content_version,
            content_source_hash=snapshot.content_hash,
        )
        result = EngineTurnResult(
            state=next_state,
            analysis=analysis,
            classifier_response_category=analysis.response_category,
            turn=next_turn,
            runtime=runtime,
            accepted_claims=dict(next_state.verified_slots),
        )
        yield await self._contextualize_note_if_safe(
            result,
            response=response,
            recent_dialogue=recent_dialogue or [],
        )

    async def _contextualize_note_if_safe(
        self,
        result: EngineTurnResult,
        *,
        response: ChildResponse,
        recent_dialogue: list[DialogueHistoryTurn],
    ) -> EngineTurnResult:
        """Use Haiku for direct-note wording without making note availability LLM-bound."""

        note = result.turn.note_update
        if note is None or note.attribution is not NoteAttribution.CHILD:
            return result
        state = result.state
        if not state.raw_storage_enabled:
            return result
        pack, _, ledger, _ = self._resolve_state(state)
        source_by_turn = {
            turn.turn_id: turn.child
            for turn in recent_dialogue
            if turn.child is not None
        }
        if response.text is not None:
            source_by_turn[response.turn_id] = response.text
        fragments: dict[str, str] = {}
        for relation_id in pack.policies.note_relation_ids:
            if relation_id not in state.child_note_evidence:
                return result
            entry = ledger.verified_relations.get(relation_id)
            if entry is None:
                return result
            pointer = next(
                (
                    evidence
                    for evidence in reversed(entry.evidence)
                    if isinstance(evidence, RelationVerificationEvidenceV2)
                ),
                None,
            )
            if pointer is None:
                return result
            source = source_by_turn.get(pointer.source_turn_id)
            if source is None:
                return result
            fragment = source[pointer.source_start : pointer.source_end]
            if not fragment or bridge_excerpt_violation_v2(fragment) is not None:
                return result
            fragments[relation_id] = fragment
        context = NoteContextualizationContext(
            skill_id=pack.curriculum_session_id,
            note_context=pack.source_problem.prompt,
            source_fragments=fragments,
            reviewed_facts={
                fact.fact_id: fact.speaker_label
                for fact in pack.reasoning_graph.facts
                if fact.initially_visible or fact.fact_id in ledger.verified_facts
            },
            allowed_numbers=[
                str(_primitive(fact.value))
                for fact in pack.reasoning_graph.facts
                if isinstance(_primitive(fact.value), int | float)
                and not isinstance(_primitive(fact.value), bool)
            ],
            fallback_text=note.text,
        )
        try:
            gateway = self.gateway  # Protocol intentionally keeps this optional.
            async with asyncio.timeout(min(self.speaker_timeout_seconds, 4.0)):
                output = await gateway.contextualize_note(context)  # type: ignore[attr-defined]
            contextualized = validate_note_contextualization(output, context)
        except Exception:
            contextualized = None
        if contextualized is None:
            return result
        return EngineTurnResult(
            state=result.state,
            analysis=result.analysis,
            classifier_response_category=result.classifier_response_category,
            turn=result.turn.model_copy(
                update={
                    "note_update": note.model_copy(
                        update={"text": contextualized},
                        deep=True,
                    )
                },
                deep=True,
            ),
            runtime=result.runtime,
            accepted_claims=result.accepted_claims,
        )

    async def _understand_text(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        snapshot: PinnedContentSnapshotV2,
        ledger: ReasoningLedgerV2,
        response: ChildResponse,
        previous_question: str,
        recent_dialogue: list[DialogueHistoryTurn],
    ) -> _TurnSemantics:
        assert response.text is not None
        targets = self._targets_from_active_state(state, pack, ledger)
        request = UnderstandingRequestV2.model_validate(
            {
            "task_id": pack.task_id,
            "visible_facts": {
                fact.fact_id: _primitive(fact.value)
                for fact in pack.reasoning_graph.facts
                if fact.initially_visible
            },
            "targets": [
                {
                    "target_kind": target.target_kind,
                    "target_id": target.target_id,
                    "ask_kind": target.ask_kind,
                    "rubric": self._target_rubric(pack, target),
                    "expected_truth": self._target_truth(pack, target),
                }
                for target in targets
            ],
            "claimable_graph": {
                "fact_ids": [fact.fact_id for fact in pack.reasoning_graph.facts],
                "relation_ids": [
                    relation.relation_id for relation in pack.reasoning_graph.relations
                ],
                "open_auxiliary_claims": pack.reasoning_graph.open_auxiliary_claims,
            },
            "current_turn": {
                "mormi_question": previous_question,
                "asks": list(dict.fromkeys(target.ask_kind for target in targets)),
                "expression_level": state.expression_level,
                "hint_level": state.hint_level,
                # H2/H3 may scaffold a currently requested method.  Only its
                # reviewed relation identity crosses the understanding
                # boundary; the card body, equation and values never enter a
                # Mormi speaker plan or this request.
                "help_scaffolded_relation_ids": (
                    [
                        target.target_id
                        for target in targets
                        if target.target_kind == "relation"
                    ]
                    if state.hint_level in {HintLevel.H2, HintLevel.H3}
                    else []
                ),
            },
            "recent_history": recent_dialogue[-6:],
            "child_utterance": response.text,
            }
        )
        started = time.perf_counter()
        for attempt in (1, 2):
            try:
                async with asyncio.timeout(self.classifier_timeout_seconds):
                    candidate = await self.gateway.understand_v2(request)
            except TimeoutError:
                return self._understanding_failure_semantics(
                    snapshot,
                    ledger,
                    attempts=attempt,
                    started=started,
                    reason="understanding_timeout",
                )
            except ModelUnavailableError:
                return self._understanding_failure_semantics(
                    snapshot,
                    ledger,
                    attempts=attempt,
                    started=started,
                    reason="understanding_model_unavailable",
                )
            except ModelOutputError:
                return self._understanding_failure_semantics(
                    snapshot,
                    ledger,
                    attempts=attempt,
                    started=started,
                    reason="understanding_model_output_invalid",
                )
            route = self._route_for_understanding(candidate)
            if route in {"bridge", "safety"}:
                # Pure social turns and safety redirects do not mutate the
                # reasoning ledger. Mixed safe-social + learning turns are
                # routed through the main path below, where their literal
                # learning evidence is guarded and preserved independently.
                return _TurnSemantics(
                    response=candidate,
                    apply_result=self._unchanged_result(snapshot, ledger),
                    guarded=None,
                    route=route,
                    understanding_source="sonnet_low",
                    attempts=attempt,
                    latency_ms=_elapsed_ms(started),
                    guard_status="not_applicable",
                )
            try:
                guarded = guard_understanding_response_v2(request, candidate)
            except UnderstandingEvidenceGuardError as error:
                request = request.model_copy(
                    update={
                        "guard_feedback_codes": [
                            violation.code.value for violation in error.violations
                        ]
                    }
                )
                continue
            result = apply_guarded_understanding_v2(
                snapshot,
                ledger,
                guarded,
                source_turn_id=response.turn_id,
            )
            initial_help = (
                state.expression_level is ExpressionLevel.L4
                and state.hint_level is HintLevel.H0
                and guarded.response.support_need is SupportNeed.GENERAL_HELP
                and guarded.response.conversation_move is ConversationMoveV2.NONE
            )
            return _TurnSemantics(
                response=guarded.response,
                apply_result=result,
                guarded=guarded,
                route=(
                    "stable"
                    if initial_help
                    else route
                ),
                understanding_source="sonnet_low",
                attempts=attempt,
                latency_ms=_elapsed_ms(started),
                guard_status="passed" if attempt == 1 else "retry_passed",
                initial_help=initial_help,
            )
        return self._understanding_failure_semantics(
            snapshot,
            ledger,
            attempts=2,
            started=started,
            reason="understanding_evidence_guard_failed",
        )

    def _understanding_failure_semantics(
        self,
        snapshot: PinnedContentSnapshotV2,
        ledger: ReasoningLedgerV2,
        *,
        attempts: int,
        started: float,
        reason: Literal[
            "understanding_timeout",
            "understanding_model_unavailable",
            "understanding_model_output_invalid",
            "understanding_evidence_guard_failed",
        ],
    ) -> _TurnSemantics:
        """Turn an untrusted classifier result into a no-progress safe turn.

        The returned object contains no child text, provider prose or guessed
        intent. Its dedicated route bypasses both ladder mutation and another
        model call; the runtime later appends the reviewed current-target re-ask.
        """

        return _TurnSemantics(
            response=UnderstandingResponseV2(
                utterance_class=UtteranceClassV2.LEARNING_RESPONSE,
            ),
            apply_result=self._unchanged_result(snapshot, ledger),
            guarded=None,
            route="understanding_fallback",
            understanding_source="deterministic_fallback",
            attempts=attempts,
            latency_ms=_elapsed_ms(started),
            guard_status="failed",
            understanding_failure_reason=reason,
        )

    def _no_response_semantics(
        self,
        state: SessionState,
        snapshot: PinnedContentSnapshotV2,
        ledger: ReasoningLedgerV2,
        response: ChildResponse,
    ) -> _TurnSemantics:
        kind = response.no_response_kind or NoResponseKindV2.EXPLICIT_HELP
        support_need = (
            SupportNeed.GENERAL_HELP
            if kind is NoResponseKindV2.EXPLICIT_HELP
            else SupportNeed.EXPRESSION
            if kind is NoResponseKindV2.SILENCE_TIMEOUT
            else SupportNeed.NONE
        )
        semantic = UnderstandingResponseV2.model_validate(
            {
            "utterance_class": (
                "help_request"
                if kind is NoResponseKindV2.EXPLICIT_HELP
                else "learning_response"
            ),
            "support_need": support_need,
            }
        )
        completion = reasoning_completion_v2(snapshot, ledger)
        result = ReasoningLedgerApplyResultV2(
            ledger=ledger,
            new_fact_ids=[],
            new_relation_ids=[],
            new_milestone_fact_ids=[],
            new_fact_evidence_ids=[],
            new_relation_evidence_ids=[],
            new_auxiliary_evidence_ids=[],
            ignored_claim_ids=[],
            completion=completion,
            completion_became_true=False,
        )
        understanding_source: Literal[
            "explicit_no_response",
            "silence_timeout",
            "asr_empty",
        ]
        if kind is NoResponseKindV2.EXPLICIT_HELP:
            understanding_source = "explicit_no_response"
        elif kind is NoResponseKindV2.SILENCE_TIMEOUT:
            understanding_source = "silence_timeout"
        else:
            understanding_source = "asr_empty"
        return _TurnSemantics(
            response=semantic,
            apply_result=result,
            guarded=None,
            route=(
                "stable"
                if kind is NoResponseKindV2.EXPLICIT_HELP
                else "main"
            ),
            understanding_source=understanding_source,
            initial_help=(
                kind is NoResponseKindV2.EXPLICIT_HELP
                and state.expression_level is ExpressionLevel.L4
                and state.hint_level is HintLevel.H0
            ),
        )

    def _choice_semantics(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        snapshot: PinnedContentSnapshotV2,
        ledger: ReasoningLedgerV2,
        response: ChildResponse,
    ) -> _TurnSemantics:
        if state.expression_level is not ExpressionLevel.L2:
            raise ValueError("choice response is not valid outside L2")
        plan = self._active_l2_plan(state, pack, ledger)
        if len(response.choice_ids) != 1:
            raise ValueError("V2 L2 response requires exactly one choice")
        choice = next(
            (item for item in plan.choices if item.choice_id == response.choice_ids[0]),
            None,
        )
        if choice is None:
            raise ValueError("choice_id does not belong to the pinned V2 plan")
        effect = choice.effect
        correct = effect.verdict == "correct"
        if correct:
            relation_ids: list[str] = []
            if effect.target_kind == "relation":
                # A reviewed L2 method choice is an action by the learner, not
                # knowledge that Mormi inferred from merely seeing a help
                # card.  Home content may intentionally express a composite
                # method in that one choice (for example: add every item cost,
                # then compare the total with the budget).  In that case the
                # pack-owned note policy is the authoritative list of method
                # relations performed together by the choice.
                relation_ids = list(
                    dict.fromkeys(
                        [effect.target_id, *pack.policies.note_relation_ids]
                    )
                )
            result = apply_structured_progress_v2(
                snapshot,
                ledger,
                fact_values=(
                    {effect.target_id: effect.interpreted_value}
                    if effect.target_kind == "fact" and effect.interpreted_value is not None
                    else {}
                ),
                relation_ids=relation_ids,
                source_turn_id=response.turn_id,
                source_kind="choice",
            )
        else:
            result = self._unchanged_result(snapshot, ledger)
        semantic = UnderstandingResponseV2.model_validate(
            {
            "utterance_class": "learning_response",
            "contains_learning_evidence": correct,
            "answer_status": (
                "complete"
                if correct and effect.target_kind == "fact"
                else "incorrect"
                if not correct and effect.target_kind == "fact"
                else "not_applicable"
            ),
            "reasoning_status": (
                "sufficient"
                if correct and effect.target_kind == "relation"
                else "incorrect"
                if not correct and effect.target_kind == "relation"
                else "not_applicable"
            ),
            }
        )
        return _TurnSemantics(
            response=semantic,
            apply_result=result,
            guarded=None,
            route="main",
            understanding_source="structured_choice",
            structured_correct=correct,
        )

    def _joint_semantics(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        snapshot: PinnedContentSnapshotV2,
        ledger: ReasoningLedgerV2,
        response: ChildResponse,
    ) -> _TurnSemantics:
        if state.expression_level is not ExpressionLevel.L0 or state.hint_level is not HintLevel.H3:
            raise ValueError("joint response requires L0-H3")
        expected_values: dict[str, str | int | float | bool] = {}
        for completion in pack.l0_joint_plan.completion_values:
            key = f"{completion.target_kind}:{completion.target_id}"
            expected_values[key] = (
                _primitive(completion.value)
                if completion.target_kind == "fact"
                else True
            )
        if not _same_json_value(response.values, expected_values):
            raise ValueError("action values do not match pinned joint completion values")
        values: dict[str, CanonicalValueV2 | dict[str, Any]] = {
            completion.target_id: completion.value
            for completion in pack.l0_joint_plan.completion_values
            if completion.target_kind == "fact"
        }
        relations = [
            completion.target_id
            for completion in pack.l0_joint_plan.completion_values
            if completion.target_kind == "relation"
        ]
        # Merely displaying H3 never teaches Mormi.  Once the learner actually
        # submits the pinned joint action, however, every note relation that
        # the reviewed H3 joint model explicitly covers was performed
        # together.  Preserve those relations as structured/coauthored
        # evidence even when an older pinned L0 completion contract listed
        # only the minimum graph-completion relation.
        h3_joint_note_relations = set(pack.policies.note_relation_ids).intersection(
            pack.help_plan.H3.revealed_relation_ids
        )
        relations = list(dict.fromkeys([*relations, *sorted(h3_joint_note_relations)]))
        result = apply_structured_progress_v2(
            snapshot,
            ledger,
            fact_values=values,
            relation_ids=relations,
            source_turn_id=response.turn_id,
            source_kind="joint",
        )
        return _TurnSemantics(
            response=UnderstandingResponseV2.model_validate(
                {
                    "utterance_class": "learning_response",
                    "contains_learning_evidence": True,
                    "answer_status": "complete",
                    "reasoning_status": "sufficient",
                }
            ),
            apply_result=result,
            guarded=None,
            route="stable",
            understanding_source="structured_joint",
            structured_correct=True,
        )

    @staticmethod
    def _unchanged_result(
        snapshot: PinnedContentSnapshotV2,
        ledger: ReasoningLedgerV2,
    ) -> ReasoningLedgerApplyResultV2:
        return ReasoningLedgerApplyResultV2(
            ledger=ledger,
            new_fact_ids=[],
            new_relation_ids=[],
            new_milestone_fact_ids=[],
            new_fact_evidence_ids=[],
            new_relation_evidence_ids=[],
            new_auxiliary_evidence_ids=[],
            ignored_claim_ids=[],
            completion=reasoning_completion_v2(snapshot, ledger),
            completion_became_true=False,
        )

    @staticmethod
    def _route_for_understanding(
        response: UnderstandingResponseV2,
    ) -> Literal["main", "bridge", "safety", "stable"]:
        if response.utterance_class.value in {"system_manipulation", "safety_risk"}:
            return "safety"
        # The provider contract uses ``learning_response`` for a safe social
        # move that also carries real learning evidence.  Claims attached to
        # the legacy/pure ``non_learning_safe`` class remain fail-closed.
        if response.utterance_class is UtteranceClassV2.NON_LEARNING_SAFE:
            return "bridge"
        if response.conversation_move in {
            ConversationMoveV2.META_QUESTION,
            ConversationMoveV2.REFUSAL,
            ConversationMoveV2.SAFE_PLAY,
        } and not response.contains_learning_evidence:
            return "bridge"
        return "main"

    def _apply_ladder_policy(
        self,
        state: SessionState,
        semantics: _TurnSemantics,
    ) -> None:
        if semantics.understanding_source == "deterministic_fallback":
            # A provider/contract failure says nothing about the child's
            # understanding or expression. Preserve every L/H counter exactly.
            return
        response = semantics.response
        if semantics.apply_result.has_new_canonical_progress:
            state.expression_failures = 0
            state.concept_failures = 0
            state.vague_clarifications = 0
            state.unrelated_count = 0
            if response.conversation_move in {
                ConversationMoveV2.TASK_QUESTION,
                ConversationMoveV2.REQUEST_MORMI_ANSWER,
            }:
                self._raise_hint(state)
            return
        if response.conversation_move in {
            ConversationMoveV2.TASK_QUESTION,
            ConversationMoveV2.REQUEST_MORMI_ANSWER,
        }:
            # Reverse questions and requests that Mormi supply the answer are
            # not wrong answers. They ask the product to expose one stronger
            # reviewed scaffold, while the reasoning ledger stays untouched.
            state.vague_clarifications = 0
            self._raise_hint(state)
            return
        if response.conversation_move in {
            ConversationMoveV2.META_QUESTION,
            ConversationMoveV2.REFUSAL,
            ConversationMoveV2.SAFE_PLAY,
        } or semantics.route in {"bridge", "safety"}:
            state.unrelated_count += 1
            return
        if semantics.understanding_source == "asr_empty":
            return
        if semantics.understanding_source == "silence_timeout":
            self._lower_expression(state)
            return
        if response.support_need is SupportNeed.EXPRESSION:
            self._lower_expression(state)
            return
        if response.support_need is SupportNeed.GENERAL_HELP:
            self._lower_expression(state)
            self._raise_hint(state)
            return
        has_incorrect = semantics.structured_correct is False or any(
            getattr(claim, "verdict", None) == "incorrect"
            for claim in response.claims
        )
        if response.support_need in {SupportNeed.CONCEPT, SupportNeed.BOTH} or has_incorrect:
            state.concept_failures += 1
            self._raise_hint(state)
            return

        # Related-but-vague, uncertain, or a repeated already-verified fact:
        # keep one clarification, then change the visible response contract.
        if state.vague_clarifications == 0:
            state.vague_clarifications = 1
        else:
            state.vague_clarifications = 0
            self._lower_expression(state)

    @staticmethod
    def _raise_hint(state: SessionState) -> None:
        """Expose exactly one stronger reviewed scaffold without judging again."""

        if state.hint_level is HintLevel.H0:
            state.hint_level = HintLevel.H1
        elif state.hint_level is HintLevel.H1:
            state.hint_level = HintLevel.H2
        elif state.hint_level is HintLevel.H2:
            state.hint_level = HintLevel.H3
            state.expression_level = ExpressionLevel.L0
        if state.expression_level is ExpressionLevel.L0:
            state.hint_level = HintLevel.H3
        state.task_max_hint = max(
            state.task_max_hint,
            state.hint_level,
            key=lambda item: list(HintLevel).index(item),
        )

    @staticmethod
    def _lower_expression(state: SessionState) -> None:
        state.expression_failures += 1
        state.expression_level = state.expression_level.lower()
        if state.expression_level is ExpressionLevel.L0:
            state.hint_level = HintLevel.H3
            state.task_max_hint = HintLevel.H3

    @staticmethod
    def _track_note_provenance(
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
        *,
        before_hint: HintLevel,
    ) -> None:
        """Remember whether each note relation was taught or jointly established.

        Existing SessionState fields are reused so old snapshots remain readable.
        Only server-owned relation IDs and an opaque marker are stored; child text
        never enters this provenance record.
        """

        note_relations = set(pack.policies.note_relation_ids)
        direct_turn = (
            semantics.understanding_source == "sonnet_low"
            and before_hint in {HintLevel.H0, HintLevel.H1}
        )
        for relation_id in semantics.apply_result.new_relation_ids:
            if relation_id not in note_relations:
                continue
            if direct_turn and relation_id not in state.supported_note_slots:
                state.child_note_evidence[relation_id] = "verified_relation"
            elif relation_id not in state.child_note_evidence:
                state.supported_note_slots = list(
                    dict.fromkeys([*state.supported_note_slots, relation_id])
                )

        # A help card is never applied to the learning ledger: Mormi cannot
        # read it and suddenly understand the method.  It does, however,
        # establish that a later relation was learned with reviewed support.
        # H1 only points at visible information; H2/H3 expose enough guided
        # method content that note attribution must remain coauthored.
        if before_hint in {
            HintLevel.H2,
            HintLevel.H3,
        } or state.hint_level in {
            HintLevel.H2,
            HintLevel.H3,
        }:
            unresolved_note_relations = note_relations.difference(
                semantics.apply_result.ledger.verified_relations
            )
            state.supported_note_slots = list(
                dict.fromkeys(
                    [*state.supported_note_slots, *sorted(unresolved_note_relations)]
                )
            )

    @staticmethod
    def _complete_state(
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
    ) -> None:
        joint = semantics.understanding_source == "structured_joint"
        state.joint_performance_used = state.joint_performance_used or joint
        note_relations = set(pack.policies.note_relation_ids)
        independent = note_relations.issubset(state.child_note_evidence)
        state.all_tasks_direct = independent
        state.status = SessionStatus.COMPLETED
        state.completion_outcome = (
            CompletionOutcome.TAUGHT
            if independent and not state.joint_performance_used
            else CompletionOutcome.SUPPORTED
        )
        state.teach_reward_eligible = state.completion_outcome is CompletionOutcome.TAUGHT
        state.completed_task_slots[state.current_task_id] = dict(state.verified_slots)

    def _resolve_state(
        self,
        state: SessionState,
    ) -> tuple[
        RequiredHomeTeachingPackV2,
        PinnedContentSnapshotV2,
        ReasoningLedgerV2,
        dict[str, StableCopyPlanV2],
    ]:
        if state.runtime_contract_version is not DialogueRuntimeContractVersion.VERDICT_V1:
            raise ValueError("legacy conversation cannot enter V2 engine")
        pinned = state.pinned_dialogue_v2
        if pinned is None:
            raise ValueError("verdict-v1 conversation has no pinned V2 state")
        pack = RequiredHomeTeachingPackV2.model_validate(pinned.pack_snapshot)
        snapshot = PinnedContentSnapshotV2(
            pack_id=pinned.pack_id,
            content_version=pinned.content_version,
            curriculum_session_id=pack.curriculum_session_id,
            content_hash=pinned.source_hash,
            pack_payload=pinned.pack_snapshot,
        )
        ledger = ReasoningLedgerV2.model_validate(pinned.reasoning_ledger)
        # Both validators recheck the full payload hash and ledger binding.
        reasoning_completion_v2(snapshot, ledger)
        stable_copy_plans = validate_pinned_stable_copy_plan_set_v2(
            pack,
            pack_hash=pinned.source_hash,
            schema_version=pinned.stable_copy_plan_schema_version,
            compiler_version=pinned.stable_copy_plan_compiler_version,
            plan_set_hash=pinned.stable_copy_plan_set_hash,
            plan_payloads=pinned.stable_copy_plans,
        )
        return pack, snapshot, ledger, stable_copy_plans

    @staticmethod
    def _store_ledger(state: SessionState, ledger: ReasoningLedgerV2) -> None:
        pinned = state.pinned_dialogue_v2
        if pinned is None:  # pragma: no cover - checked by _resolve_state
            raise ValueError("missing V2 runtime state")
        state.pinned_dialogue_v2 = pinned.model_copy(
            update={"reasoning_ledger": ledger.model_dump(mode="json")},
            deep=True,
        )

    @staticmethod
    def _sync_legacy_slots(
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
    ) -> None:
        facts = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}
        relations = {
            relation.relation_id: relation for relation in pack.reasoning_graph.relations
        }
        final_id = next(
            fact.fact_id
            for fact in pack.reasoning_graph.facts
            if fact.role == "final_answer"
        )
        compatible: dict[str, str | int | float | bool] = {}
        if final_id in ledger.verified_facts:
            compatible["answer"] = _primitive(facts[final_id].value)
        verified_required_relations = [
            relation_id
            for relation_id in pack.reasoning_graph.completion.required_relation_ids
            if relation_id in ledger.verified_relations
        ]
        if verified_required_relations:
            compatible["rule"] = relations[verified_required_relations[0]].speaker_label
        state.verified_slots = compatible

    def _question_contract(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
    ) -> _QuestionContract:
        if state.status is SessionStatus.COMPLETED:
            return _QuestionContract([], "", InputContract(kind=InputKind.NONE))
        targets = self._targets_from_active_state(state, pack, ledger)
        target_keys = [_target_key(target) for target in targets]
        if state.expression_level is ExpressionLevel.L0:
            state.hint_level = HintLevel.H3
            intro = next(slot for slot in pack.copy_slots if slot.purpose == "l0_intro")
            action = next(slot for slot in pack.copy_slots if slot.purpose == "l0_action")
            completion_values: dict[str, str | int | float | bool] = {}
            for completion in pack.l0_joint_plan.completion_values:
                key = f"{completion.target_kind}:{completion.target_id}"
                completion_values[key] = (
                    _primitive(completion.value)
                    if completion.target_kind == "fact"
                    else True
                )
            state.subgoal_id = "l0.joint"
            return _QuestionContract(
                targets=targets,
                fallback=intro.reviewed_fallback,
                input=InputContract(
                    kind=InputKind.JOINT,
                    target_slots=target_keys,
                    submit_label=pack.l0_joint_plan.button_label,
                    config={
                        "text": action.reviewed_fallback,
                        "action_id": pack.l0_joint_plan.action_id,
                        "completion_values": completion_values,
                    },
                ),
                copy_slot=intro,
            )
        if state.expression_level is ExpressionLevel.L2:
            l2_plan = self._l2_plan_for_targets(pack, targets)
            slot = next(
                item for item in pack.copy_slots if item.copy_slot == l2_plan.copy_slot
            )
            state.subgoal_id = l2_plan.plan_id
            return _QuestionContract(
                targets=[l2_plan.target],
                fallback=slot.reviewed_fallback,
                input=InputContract(
                    kind=InputKind.CHOICES,
                    choices=[
                        ChoiceOption(id=choice.choice_id, label=choice.label)
                        for choice in l2_plan.choices
                    ],
                    target_slots=[_target_key(l2_plan.target)],
                    submit_label="알려주기",
                    config={"plan_id": l2_plan.plan_id},
                ),
                copy_slot=slot,
            )
        if state.expression_level is ExpressionLevel.L3:
            l3_plan = next(
                candidate
                for plan in pack.l3_plans
                for candidate in [plan]
                if _target_key(candidate.targets[0]) in target_keys
            )
            state.subgoal_id = l3_plan.plan_id
            return _QuestionContract(
                targets=list(l3_plan.targets),
                fallback=l3_plan.reviewed_fallback,
                input=InputContract(
                    kind=InputKind.TEXT,
                    target_slots=[_target_key(l3_plan.targets[0])],
                    placeholder="짧게 알려줘",
                    submit_label="알려주기",
                ),
            )
        state.subgoal_id = (
            pack.initial_question.plan_id
            if len(targets) > 1
            else next(
                plan.plan_id
                for plan in pack.l3_plans
                if plan.targets[0] == targets[0]
            )
        )
        fallback = (
            pack.initial_question.reviewed_fallback
            if len(targets) > 1
            else next(
                plan.reviewed_fallback
                for plan in pack.l3_plans
                if plan.targets[0] == targets[0]
            )
        )
        return _QuestionContract(
            targets=targets,
            fallback=fallback,
            input=InputContract(
                kind=InputKind.TEXT,
                target_slots=target_keys,
                placeholder=(
                    "답과 방법을 네 말로 알려줘"
                    if len(targets) > 1
                    else "짧게 알려줘"
                ),
                submit_label="알려주기",
            ),
        )

    @staticmethod
    def _initial_help_question_contract(
        pack: RequiredHomeTeachingPackV2,
        question: _QuestionContract,
    ) -> _QuestionContract:
        """Bind the first help transition to its immutable, cacheable copy slot."""

        slot = next(item for item in pack.copy_slots if item.purpose == "initial_help")
        targets = list(slot.targets)
        return _QuestionContract(
            targets=targets,
            fallback=slot.reviewed_fallback,
            input=question.input.model_copy(
                update={"target_slots": [_target_key(target) for target in targets]},
                deep=True,
            ),
            copy_slot=slot,
        )

    def _targets_from_active_state(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
    ) -> list[TargetRefV2]:
        completion = reasoning_completion_v2(
            self._snapshot_from_pack(pack),
            ledger,
        )
        remaining = {
            *(('fact', item) for item in completion.remaining_fact_ids),
            *(('relation', item) for item in completion.remaining_relation_ids),
        }
        ordered = [
            target
            for target in pack.initial_question.targets
            if (target.target_kind, target.target_id) in remaining
        ]
        if not ordered:
            return []
        if state.expression_level in {ExpressionLevel.L4, ExpressionLevel.L0}:
            return ordered
        l3_plans = pack.l3_plans if state.expression_level is ExpressionLevel.L3 else []
        if l3_plans:
            for l3_plan in l3_plans:
                if l3_plan.targets[0] in ordered:
                    return [l3_plan.targets[0]]
        for l2_plan in pack.l2_plans:
            if l2_plan.target in ordered:
                return [l2_plan.target]
        return [ordered[0]]

    @staticmethod
    def _snapshot_from_pack(pack: RequiredHomeTeachingPackV2) -> PinnedContentSnapshotV2:
        # Used only with a ledger whose content hash is checked by
        # reasoning_completion_v2.  Pinning the parsed payload is deterministic.
        return pin_content_pack_v2(pack)

    def _active_l2_plan(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
    ) -> L2ChoicePlanV2:
        plan = next((item for item in pack.l2_plans if item.plan_id == state.subgoal_id), None)
        if plan is None:
            plan = self._l2_plan_for_targets(
                pack,
                self._targets_from_active_state(state, pack, ledger),
            )
        return plan

    @staticmethod
    def _l2_plan_for_targets(
        pack: RequiredHomeTeachingPackV2,
        targets: list[TargetRefV2],
    ) -> L2ChoicePlanV2:
        target_keys = {(target.target_kind, target.target_id) for target in targets}
        try:
            return next(
                plan
                for plan in pack.l2_plans
                if (plan.target.target_kind, plan.target.target_id) in target_keys
            )
        except StopIteration as error:  # pragma: no cover - pack validation guarantees coverage
            raise ValueError("V2 pack has no L2 plan for the remaining target") from error

    @staticmethod
    def _target_truth(
        pack: RequiredHomeTeachingPackV2,
        target: TargetRefV2,
    ) -> CanonicalValueV2 | None:
        if target.target_kind == "relation":
            return None
        return next(
            fact.value
            for fact in pack.reasoning_graph.facts
            if fact.fact_id == target.target_id
        )

    @staticmethod
    def _target_rubric(
        pack: RequiredHomeTeachingPackV2,
        target: TargetRefV2,
    ) -> dict[str, str]:
        if target.target_kind == "fact":
            fact = next(
                item for item in pack.reasoning_graph.facts if item.fact_id == target.target_id
            )
            return {
                "correct": f"{fact.speaker_label}에 대한 수학적으로 맞는 주장",
                "partial": "역할은 맞지만 표현이 불완전한 주장",
            }
        relation = next(
            item
            for item in pack.reasoning_graph.relations
            if item.relation_id == target.target_id
        )
        return {
            "sufficient": " | ".join(relation.rubric.sufficient),
            "partial": " | ".join(relation.rubric.partial),
            "incorrect": " | ".join(relation.rubric.incorrect),
        }

    async def _stable_fallback(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
        question: _QuestionContract,
        *,
        stable_copy_plans: Mapping[str, StableCopyPlanV2],
        before_expression: ExpressionLevel,
        before_hint: HintLevel,
    ) -> StableCopyResolution | None:
        slot = question.copy_slot
        if slot is None:
            return None
        pinned = state.pinned_dialogue_v2
        if pinned is None:  # pragma: no cover
            raise ValueError("missing V2 runtime state")
        del ledger
        try:
            plan = stable_copy_plans[slot.copy_slot]
        except KeyError as error:  # pragma: no cover - plan-set validation checks coverage
            raise ValueError("pinned stable-copy plan is missing for the copy slot") from error
        transition = plan.transition
        if (
            transition.from_expression_level is not before_expression
            or transition.from_hint_level is not before_hint
            or transition.to_expression_level is not state.expression_level
            or transition.to_hint_level is not state.hint_level
        ):
            # Only the context-independent transition prewarmed for this slot
            # may reuse its artifact. Other L/H combinations need the normal
            # contextual speaker and the reviewed question as its fallback.
            return None
        resolution = StableCopyResolution(
            text=slot.reviewed_fallback,
            mood="thinking",
            dialogue_act=plan.dialogue_act,
            status="reviewed_fallback",
        )
        if slot.purpose in {"l0_intro", "l0_action"}:
            # L0/H3 card truth belongs to the child-facing UI. Mormi always
            # uses reviewed generic copy and never asks a model or old cached
            # artifact to read the card and discover its answer.
            return resolution
        if self.copy_resolver is not None:
            output_firewall = build_stable_copy_output_firewall_v2(pack, slot)
            raw = await self.copy_resolver.resolve(
                plan,
                reviewed_fallback=slot.reviewed_fallback,
                pack_hash=pinned.source_hash,
                output_firewall=output_firewall,
                pinned_snapshot=pinned.copy_snapshots.get(slot.copy_slot),
            )
            violation = stable_copy_output_violation_v2(
                raw.as_output(),
                plan,
                forbidden_values=output_firewall.forbidden_values,
                forbidden_surfaces=output_firewall.forbidden_surfaces,
            )
            if violation is not None:
                # Do not pin a cache artifact compiled under an older or weaker
                # output guard.  The reviewed pack fallback is independently
                # validated as context-free content.
                return resolution
            normalized_status = (
                "hit"
                if raw.status == "pinned"
                and raw.artifact_metadata.origin == "generated"
                else "reviewed_fallback"
                if raw.status == "pinned"
                else raw.status
            )
            resolution = StableCopyResolution(
                text=raw.text,
                mood=raw.mood,
                dialogue_act=raw.dialogue_act,
                status=normalized_status,
                cache_key=raw.full_cache_key,
                artifact=raw.model_dump(mode="json"),
            )
            copies = dict(pinned.copy_snapshots)
            copies[slot.copy_slot] = raw.model_dump(mode="json")
            state.pinned_dialogue_v2 = pinned.model_copy(
                update={"copy_snapshots": copies},
                deep=True,
            )
        return resolution

    async def _speak(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
        question: _QuestionContract,
        fallback: str,
        previous_question: str,
        *,
        before_hint: HintLevel,
    ) -> SpeechResultV2:
        response_plan = self._conversation_response_plan(
            state,
            pack,
            semantics,
            question,
            before_hint=before_hint,
        )
        private_support_route = response_plan.response_mode != "normal"
        target_aliases: dict[tuple[str, str], str] = {
            (target.target_kind, target.target_id): focus.target_id
            for target, focus in zip(
                question.targets,
                response_plan.reask_targets,
                strict=True,
            )
        }
        plan = SpeakerPlanV2(
            dialogue_act=self._dialogue_act(semantics),
            response_signal=self._speaker_response_signal(
                semantics,
                question.targets,
                repeat_count=state.concept_failures,
                target_aliases=target_aliases,
            ),
            accepted_evidence=self._speaker_evidence(semantics),
            accepted_relations=self._speaker_accepted_relations(
                state,
                pack,
                semantics,
            ),
            target=self._speaker_target(question.targets, target_aliases=target_aliases),
            target_focus=response_plan.reask_targets,
            response_plan=response_plan,
            support=SpeakerSupportV2(
                expression_level=state.expression_level,
                hint_level=state.hint_level,
                support_need=semantics.response.support_need.value,
                question_style_guide=self._question_style(state.expression_level),
                help_card_visible=state.hint_level is not HintLevel.H0,
            ),
            allowed_facts=self._allowed_facts(
                pack,
                semantics.apply_result.ledger,
                question.targets,
                include_screen=not private_support_route,
            ),
            # The product UI renders only the latest Mormi turn.  The model
            # therefore owns the conversational reaction, while the server
            # owns the active target question on every non-completed turn.
            # Keeping this target-only text out of model generation makes it
            # impossible for a valid acknowledgement-only completion to make
            # the learner's current question disappear.
            current_question=self._safe_reask_text(response_plan),
            previous_mormi_text=(None if private_support_route else previous_question),
            fallback_copy_ref=f"fallback.{state.subgoal_id}",
        )
        firewall = self._unresolved_answer_firewall(pack, question.targets)
        last_reason = "speaker_contract_rejected"
        for _ in range(2):
            try:
                async with asyncio.timeout(self.speaker_timeout_seconds):
                    output = await self.gateway.speak_v2(plan)
                violation = speaker_output_violation_v2(
                    output,
                    plan,
                    forbidden_values=firewall.values,
                    forbidden_surfaces=firewall.surfaces,
                )
                if violation is None:
                    text = output.text.strip()
                    # The product UI keeps only the active Mormi turn on
                    # screen.  On every active model-generated turn the model
                    # owns only the short reaction/acknowledgement; the server
                    # appends the reviewed, target-only reask.  This applies
                    # to ordinary partial progress as well as support/social
                    # recovery, so a valid acknowledgement-only model output
                    # can never remove the question from the screen.
                    text = self._compose_reaction_and_reask(
                        text,
                        self._safe_reask_text(response_plan),
                    )
                    return text, output.mood, "llm", None
                last_reason = violation
            except Exception as error:  # provider/schema failure uses the reviewed fallback
                last_reason = type(error).__name__
        return fallback, "thinking", "generation_fallback", last_reason

    async def _bridge(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
        question: _QuestionContract,
        fallback: str,
        child_text: str | None,
        previous_question: str,
        *,
        before_hint: HintLevel,
    ) -> SpeechResultV2:
        move = semantics.response.conversation_move
        if move is ConversationMoveV2.META_QUESTION:
            kind = "meta"
        elif move is ConversationMoveV2.REFUSAL:
            kind = "refusal"
        elif move is ConversationMoveV2.SAFE_PLAY:
            kind = (
                semantics.response.non_learning_kind.value
                if semantics.response.non_learning_kind
                else "playful"
            )
        else:
            kind = "off_topic"
        response_plan = self._conversation_response_plan(
            state,
            pack,
            semantics,
            question,
            before_hint=before_hint,
        )
        target_aliases: dict[tuple[str, str], str] = {
            (target.target_kind, target.target_id): focus.target_id
            for target, focus in zip(
                question.targets,
                response_plan.reask_targets,
                strict=True,
            )
        }
        plan = BridgePlanV2(
            interaction_kind=kind,  # type: ignore[arg-type]
            safe_child_excerpt=pii_safe_bridge_excerpt_v2(
                child_text,
                interaction_kind=kind,  # type: ignore[arg-type]
            ),
            current_question=self._safe_reask_text(response_plan),
            target=self._speaker_target(
                question.targets,
                target_aliases=target_aliases,
            ),
            target_focus=response_plan.reask_targets,
            response_plan=response_plan,
            # Pure social turns need no mathematical context. Keeping even
            # visible givens out prevents Haiku from solving the task from the
            # screen instead of asking the child to teach Mormi.
            allowed_facts=[],
            repeat_count=state.unrelated_count,
            previous_mormi_text=None,
            fallback_copy_ref=f"fallback.{state.subgoal_id}",
        )
        firewall = self._unresolved_answer_firewall(pack, question.targets)
        try:
            async with asyncio.timeout(self.bridge_timeout_seconds):
                output = await self.gateway.bridge_speak_v2(plan)
            text = validate_bridge_output_v2(
                output,
                plan,
                forbidden_values=firewall.values,
                forbidden_surfaces=firewall.surfaces,
            )
            if text is not None:
                # Haiku decides only how to acknowledge the safe interaction
                # kind.  The server-owned reask is always present because the
                # previous turn is not visible in the production UI.
                return (
                    self._compose_reaction_and_reask(
                        text,
                        self._safe_reask_text(response_plan),
                    ),
                    output.mood,
                    "bridge_llm",
                    None,
                )
            reason = "bridge_contract_rejected"
        except Exception as error:
            reason = type(error).__name__
        return fallback, "thinking", "generation_fallback", reason

    def _conversation_response_plan(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
        question: _QuestionContract,
        *,
        before_hint: HintLevel,
    ) -> ConversationResponsePlanV2:
        """Compile the social move without giving the model pedagogical authority."""

        move = semantics.response.conversation_move
        subject = semantics.response.move_subject
        has_incorrect = semantics.structured_correct is False or any(
            getattr(claim, "verdict", None) == "incorrect"
            for claim in semantics.response.claims
        )
        card_visible = state.hint_level is not HintLevel.H0

        if move is ConversationMoveV2.META_QUESTION:
            response_mode = (
                "explain_ai_role"
                if subject is MoveSubjectV2.MORMI_AI_IDENTITY
                else "explain_mormi_limit"
            )
        elif move is ConversationMoveV2.REFUSAL:
            response_mode = "respond_refusal"
        elif move is ConversationMoveV2.SAFE_PLAY:
            response_mode = "respond_safe_play"
        elif move is ConversationMoveV2.REQUEST_MORMI_ANSWER:
            response_mode = "decline_answer_and_ask"
        elif move is ConversationMoveV2.TASK_QUESTION or (
            card_visible
            and (
                has_incorrect
                or semantics.response.support_need
                in {SupportNeed.CONCEPT, SupportNeed.BOTH, SupportNeed.GENERAL_HELP}
            )
        ):
            response_mode = "redirect_to_help_card"
        elif semantics.route == "safety":
            response_mode = "safety_redirect"
        else:
            response_mode = "normal"

        if state.expression_level is ExpressionLevel.L0:
            reask_mode = "joint_action"
        elif response_mode in {
            "redirect_to_help_card",
            "decline_answer_and_ask",
        }:
            reask_mode = "help_guided_targets"
        else:
            reask_mode = "remaining_targets"

        target_aliases = (
            self._private_target_aliases(question.targets)
            if response_mode != "normal"
            else {}
        )
        return ConversationResponsePlanV2(
            response_mode=response_mode,  # type: ignore[arg-type]
            reask_mode=reask_mode,  # type: ignore[arg-type]
            reask_targets=self._speaker_target_focus(
                pack,
                question.targets,
                target_aliases=target_aliases,
            ),
            card_visible=card_visible,
            card_event=(
                "opened_or_strengthened"
                if card_visible and state.hint_level is not before_hint
                else "none"
            ),
            hint_level=state.hint_level,
        )

    @staticmethod
    def _safe_reask_text(plan: ConversationResponsePlanV2) -> str:
        """Build a target-only anchor with no problem values or solution method."""

        labels = list(dict.fromkeys(item.speaker_label for item in plan.reask_targets))
        target_text = DialogueV2Engine._join_reask_labels(labels)
        if plan.reask_mode == "joint_action":
            return f"도움 카드에 나온 순서대로 {target_text}, 나와 같이 해 줄 수 있어?"
        if plan.reask_mode == "help_guided_targets":
            if plan.response_mode == "decline_answer_and_ask":
                return f"도움 카드를 보고 다시 {target_text} 알려주면 안 될까?"
            return f"도움 카드를 보고 다시 {target_text} 알려줄 수 있어?"
        if plan.response_mode == "respond_refusal":
            return f"{target_text} 알려주면 안 될까?"
        return f"{target_text} 알려줄 수 있어?"

    @staticmethod
    def _join_reask_labels(labels: list[str]) -> str:
        """Join nominal targets without guessing a Korean case particle."""

        if not labels:
            return "지금 궁금한 것"
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{DialogueV2Engine._with_and_particle(labels[0])} {labels[1]}"
        return ", ".join(labels[:-1]) + f", 그리고 {labels[-1]}"

    @staticmethod
    def _with_and_particle(label: str) -> str:
        """Attach the child-friendly Korean connector `(이)랑` safely."""

        stripped = label.rstrip()
        if not stripped:
            return label
        last = stripped[-1]
        codepoint = ord(last)
        if 0xAC00 <= codepoint <= 0xD7A3:
            particle = "이랑" if (codepoint - 0xAC00) % 28 else "랑"
            return f"{stripped}{particle}"
        return f"{stripped}하고"

    @staticmethod
    def _compose_reaction_and_reask(reaction: str, reask: str) -> str:
        """Join model tone with the immutable active-turn question.

        This is composition, not semantic output adjudication: the model is
        never asked to prove that it remembered to reask, and the server does
        not inspect Korean meaning with a regular expression.
        """

        reaction = reaction.strip()
        return reask if not reaction else f"{reaction} {reask}"

    @staticmethod
    def _private_target_aliases(
        targets: list[TargetRefV2],
    ) -> dict[tuple[str, str], str]:
        """Replace descriptive graph IDs before a support/social model call."""

        counters = {"fact": 0, "relation": 0}
        aliases: dict[tuple[str, str], str] = {}
        for target in targets:
            counters[target.target_kind] += 1
            aliases[(target.target_kind, target.target_id)] = (
                f"{target.target_kind}_{counters[target.target_kind]}"
            )
        return aliases

    @staticmethod
    def _speaker_target(
        targets: list[TargetRefV2],
        *,
        target_aliases: dict[tuple[str, str], str] | None = None,
    ) -> SpeakerTargetV2:
        aliases = target_aliases or {}
        return SpeakerTargetV2(
            fact_ids=[
                aliases.get((target.target_kind, target.target_id), target.target_id)
                for target in targets
                if target.target_kind == "fact"
            ],
            relation_ids=[
                aliases.get((target.target_kind, target.target_id), target.target_id)
                for target in targets
                if target.target_kind == "relation"
            ],
            ask_mode=_ask_mode(targets),
            success_criteria_ids=[
                aliases.get((target.target_kind, target.target_id), target.target_id)
                for target in targets
            ],
        )

    @staticmethod
    def _unresolved_answer_firewall(
        pack: RequiredHomeTeachingPackV2,
        targets: list[TargetRefV2],
        *,
        excluded_fact_ids: set[str] | None = None,
    ) -> _UnresolvedAnswerFirewall:
        """Compile output-only deny material from the pinned content pack.

        This data never enters a model request and never changes an
        understanding verdict or the reasoning ledger.  It solely prevents a
        speaker from surfacing an unresolved target (or a correct structured
        option) before the child has supplied it.  L0 facts explicitly revealed
        by the stable-copy plan are excluded.
        """

        excluded = excluded_fact_ids or set()
        target_pairs = {
            (target.target_kind, target.target_id)
            for target in targets
            if not (
                target.target_kind == "fact" and target.target_id in excluded
            )
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
        return _UnresolvedAnswerFirewall(
            values=tuple(values),
            surfaces=tuple(dict.fromkeys(surfaces)),
        )

    @staticmethod
    def _speaker_target_focus(
        pack: RequiredHomeTeachingPackV2,
        targets: list[TargetRefV2],
        *,
        target_aliases: dict[tuple[str, str], str] | None = None,
    ) -> list[SpeakerTargetFocusV2]:
        """Resolve reviewed child-safe target meaning from the pinned pack."""

        fact_labels = {
            fact.fact_id: fact.speaker_label for fact in pack.reasoning_graph.facts
        }
        # An unresolved relation's reviewed ``speaker_label`` may contain the
        # very operation the child is meant to teach (for example, "전체 값을
        # 세 사람에게 나누는 방법").  The speaker receives only a neutral ask
        # label until the relation is verified.  The full reviewed label moves
        # to ``accepted_relations`` only after child/structured evidence.
        relation_labels = {
            relation.relation_id: {
                "counting": "세는 방법",
                "comparison": "비교하는 방법",
                "addition": "계산 방법",
                "subtraction": "계산 방법",
                "multiplication": "계산 방법",
                "division": "계산 방법",
                "selection": "고르는 방법",
            }.get(relation.operation, "계산 방법")
            for relation in pack.reasoning_graph.relations
        }
        aliases = target_aliases or {}
        return [
            SpeakerTargetFocusV2(
                target_kind=target.target_kind,
                target_id=aliases.get(
                    (target.target_kind, target.target_id),
                    target.target_id,
                ),
                speaker_label=(
                    fact_labels[target.target_id]
                    if target.target_kind == "fact"
                    else relation_labels[target.target_id]
                ),
            )
            for target in targets
        ]

    def _speaker_accepted_relations(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
    ) -> list[SpeakerAcceptedRelationV2]:
        relation_labels = {
            relation.relation_id: relation.speaker_label
            for relation in pack.reasoning_graph.relations
        }
        supported_relation_ids = set(state.supported_note_slots)
        pinned_life = state.pinned_dialogue_scenario_v3
        if pinned_life is not None and pack.task_id in pinned_life.task_note_states:
            supported_relation_ids.update(
                pinned_life.task_note_states[pack.task_id].supported_relation_ids
            )
        return [
            SpeakerAcceptedRelationV2(
                relation_id=relation_id,
                speaker_label=relation_labels[relation_id],
                source=(
                    "jointly_derived"
                    if relation_id in supported_relation_ids
                    or (
                        semantics.apply_result.ledger.verified_relations[
                            relation_id
                        ].evidence
                        and all(
                            isinstance(item, StructuredVerificationEvidenceV2)
                            for item in semantics.apply_result.ledger.verified_relations[
                                relation_id
                            ].evidence
                        )
                    )
                    else "child_verified"
                ),
            )
            for relation_id in semantics.apply_result.new_relation_ids
        ]

    @staticmethod
    def _speaker_response_signal(
        semantics: _TurnSemantics,
        targets: list[TargetRefV2],
        *,
        repeat_count: int,
        target_aliases: dict[tuple[str, str], str] | None = None,
    ) -> SpeakerResponseSignalV2:
        """Compile a raw-free conversational event from the semantic result."""

        claims = semantics.guarded.response.claims if semantics.guarded is not None else []
        attempted_fact_ids = list(
            dict.fromkeys(
                claim.fact_id
                for claim in claims
                if isinstance(claim, FactUnderstandingClaimV2)
            )
        )
        attempted_relation_ids = list(
            dict.fromkeys(
                claim.relation_id
                for claim in claims
                if isinstance(claim, RelationUnderstandingClaimV2)
            )
        )
        incorrect_fact_ids = list(
            dict.fromkeys(
                claim.fact_id
                for claim in claims
                if isinstance(claim, FactUnderstandingClaimV2)
                and claim.verdict == "incorrect"
            )
        )
        incorrect_relation_ids = list(
            dict.fromkeys(
                claim.relation_id
                for claim in claims
                if isinstance(claim, RelationUnderstandingClaimV2)
                and claim.verdict == "incorrect"
            )
        )

        # Structured L2 choices do not create model claims; their pinned target
        # is still enough to tell the speaker what kind of attempt occurred.
        if semantics.understanding_source == "structured_choice":
            attempted_fact_ids = [
                target.target_id for target in targets if target.target_kind == "fact"
            ]
            attempted_relation_ids = [
                target.target_id
                for target in targets
                if target.target_kind == "relation"
            ]
            if semantics.structured_correct is False:
                incorrect_fact_ids = list(attempted_fact_ids)
                incorrect_relation_ids = list(attempted_relation_ids)

        if semantics.response.conversation_move is ConversationMoveV2.TASK_QUESTION:
            kind = "task_question"
        elif semantics.apply_result.has_new_canonical_progress:
            kind = "new_progress"
        elif incorrect_fact_ids and incorrect_relation_ids:
            kind = "incorrect_answer_and_method"
        elif incorrect_fact_ids or semantics.structured_correct is False and attempted_fact_ids:
            kind = "incorrect_answer"
        elif incorrect_relation_ids or (
            semantics.structured_correct is False and attempted_relation_ids
        ):
            kind = "incorrect_method"
        elif semantics.response.support_need is SupportNeed.GENERAL_HELP:
            kind = "help_request"
        elif semantics.response.support_need is SupportNeed.EXPRESSION:
            kind = "expression_block"
        elif semantics.understanding_source in {"silence_timeout", "asr_empty"}:
            kind = "no_response"
        elif semantics.understanding_source in {"structured_choice", "structured_joint"}:
            kind = "structured_response"
        else:
            kind = "related_vague"

        aliases = target_aliases or {}

        def alias_ids(kind: str, ids: list[str]) -> list[str]:
            return [aliases.get((kind, target_id), target_id) for target_id in ids]

        return SpeakerResponseSignalV2(
            kind=kind,  # type: ignore[arg-type]
            question_focus=(
                semantics.response.question_focus.value
                if semantics.response.question_focus is not None
                else None
            ),
            attempted_fact_ids=alias_ids("fact", attempted_fact_ids),
            attempted_relation_ids=alias_ids("relation", attempted_relation_ids),
            incorrect_fact_ids=alias_ids("fact", incorrect_fact_ids),
            incorrect_relation_ids=alias_ids("relation", incorrect_relation_ids),
            new_fact_ids=alias_ids("fact", list(semantics.apply_result.new_fact_ids)),
            new_relation_ids=alias_ids(
                "relation",
                list(semantics.apply_result.new_relation_ids),
            ),
            repeat_count=repeat_count,
        )

    @staticmethod
    def _speaker_evidence(semantics: _TurnSemantics) -> list[SpeakerEvidenceV2]:
        guarded = semantics.guarded
        if guarded is None:
            return []
        result: list[SpeakerEvidenceV2] = []
        for claim in guarded.response.claims:
            if getattr(claim, "verdict", None) not in {"correct", "sufficient", "partial"}:
                continue
            evidence_id = semantics.apply_result.claim_evidence_ids.get(claim.claim_id)
            if evidence_id is not None:
                result.append(
                    SpeakerEvidenceV2(
                        evidence_id=evidence_id,
                        verdict=claim.verdict,
                    )
                )
        return result

    @staticmethod
    def _allowed_facts(
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
        targets: list[TargetRefV2],
        *,
        visible_only: bool = False,
        include_screen: bool = True,
    ) -> list[SpeakerAllowedFactV2]:
        unresolved = {
            target.target_id for target in targets if target.target_kind == "fact"
        }
        result: list[SpeakerAllowedFactV2] = []
        for fact in pack.reasoning_graph.facts:
            allowed = (include_screen and fact.initially_visible) or (
                not visible_only and fact.fact_id in ledger.verified_facts
            )
            if allowed and fact.fact_id not in unresolved:
                source: Literal["screen", "child_verified", "jointly_derived"]
                if include_screen and fact.initially_visible:
                    source = "screen"
                else:
                    evidence = ledger.verified_facts[fact.fact_id].evidence
                    source = (
                        "jointly_derived"
                        if evidence
                        and all(
                            isinstance(item, StructuredVerificationEvidenceV2)
                            for item in evidence
                        )
                        else "child_verified"
                    )
                result.append(
                    SpeakerAllowedFactV2(
                        fact_id=fact.fact_id,
                        value=fact.value,
                        speaker_text=fact.speaker_label,
                        source=source,
                    )
                )
        return result

    @staticmethod
    def _question_style(level: ExpressionLevel) -> str:
        return {
            ExpressionLevel.L4: "답과 방법을 한 흐름으로 부탁하되 한 번에 한 초점만 둔다.",
            ExpressionLevel.L3: "남은 한 가지를 짧고 구체적으로 부탁한다.",
            ExpressionLevel.L2: "화면의 검수된 선택지에서 골라 알려 달라고 부탁한다.",
            ExpressionLevel.L1: "화면의 검수된 선택지에서 골라 알려 달라고 부탁한다.",
            ExpressionLevel.L0: "도움 카드를 보며 공동 수행을 부탁한다.",
        }[level]

    @staticmethod
    def _dialogue_act(semantics: _TurnSemantics) -> str:
        if semantics.route == "bridge":
            return "bridge_back"
        if semantics.route == "safety":
            return "safe_redirect"
        if semantics.initial_help:
            return "offer_initial_help"
        if semantics.understanding_source == "structured_joint":
            return "complete_joint_work"
        if semantics.response.conversation_move in {
            ConversationMoveV2.TASK_QUESTION,
            ConversationMoveV2.REQUEST_MORMI_ANSWER,
        }:
            return "redirect_to_reviewed_support"
        if semantics.apply_result.has_new_canonical_progress:
            return "acknowledge_progress_then_ask"
        if semantics.structured_correct is False or any(
            getattr(claim, "verdict", None) == "incorrect"
            for claim in semantics.response.claims
        ):
            return "reask_with_support"
        if semantics.response.support_need is not SupportNeed.NONE:
            return "offer_support"
        return "clarify_then_reask"

    @staticmethod
    def _conversational_fallback(
        pack: RequiredHomeTeachingPackV2,
        semantics: _TurnSemantics,
        question: _QuestionContract,
        *,
        help_card_visible: bool,
        repeat_count: int,
    ) -> str:
        """Keep a failed model call conversational without changing pedagogy.

        The reviewed question remains the semantic anchor, but repeating it byte-for-byte
        after a wrong, vague, or unrelated response makes the dialogue feel broken.  This
        fallback contains no child claim or hidden answer and only varies the social bridge
        and the already-decided target shape.
        """

        focus = DialogueV2Engine._speaker_target_focus(pack, question.targets)
        labels = list(dict.fromkeys(item.speaker_label for item in focus))
        focus_text = DialogueV2Engine._join_reask_labels(labels)
        plain_reask = f"{focus_text} 알려주면 안 될까?"
        if question.input.kind is InputKind.JOINT:
            reask = f"도움 카드에 나온 순서대로 {focus_text}, 나와 같이 해 줄 수 있어?"
        elif help_card_visible:
            reask = f"도움 카드를 보고 다시 {focus_text} 알려주면 안 될까?"
        else:
            reask = plain_reask

        move = semantics.response.conversation_move
        if move is ConversationMoveV2.META_QUESTION:
            if semantics.response.move_subject is MoveSubjectV2.MORMI_AI_IDENTITY:
                return f"나는 AI지만, 네가 가르쳐 주지 않은 건 잘 몰라... {reask}"
            return f"나는 네가 가르쳐 주지 않은 건 잘 몰라... {reask}"
        if move is ConversationMoveV2.REFUSAL:
            refusal_reask = (
                reask if question.input.kind is InputKind.JOINT else plain_reask
            )
            return f"나 꼭 알고 싶은데... {refusal_reask}"
        if move is ConversationMoveV2.SAFE_PLAY:
            return f"나는 장난 말고 지금은 {focus_text}이 궁금해. {reask}"
        if move is ConversationMoveV2.REQUEST_MORMI_ANSWER:
            return f"나는 어떻게 하는 건지 몰라... {reask}"
        if move is ConversationMoveV2.TASK_QUESTION:
            if help_card_visible:
                return f"나도 아직 잘 모르겠어... 어? 도움 카드가 나왔어. {reask}"
            return f"나도 아직 잘 모르겠어... {reask}"

        if semantics.route == "bridge":
            return f"오, 그런 이야기도 있구나. 그런데 나는 아직 궁금해... {reask}"

        has_incorrect = semantics.structured_correct is False or any(
            getattr(claim, "verdict", None) == "incorrect"
            for claim in semantics.response.claims
        )
        if has_incorrect or semantics.response.support_need in {
            SupportNeed.CONCEPT,
            SupportNeed.BOTH,
        }:
            opening = (
                "어, 그런가? 나 아직 잘 모르겠어..."
                if repeat_count <= 1
                else "음, 내가 아직 헷갈리는 부분이 있나 봐..."
            )
            return f"{opening} {reask}"

        if semantics.response.support_need is not SupportNeed.NONE:
            return f"음, 내가 아직 잘 못 알아들었어... {reask}"
        return f"음, 내가 아직 잘 못 알아들었어... {reask}"

    @staticmethod
    def _unsafe_fallback(question: str) -> str:
        return f"그 말은 따라 하지 않을게. {question}"

    def _build_turn(
        self,
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
        *,
        text: str,
        mood: Literal["curious", "listening", "thinking", "relieved", "celebrating"],
    ) -> TurnContract:
        question = self._question_contract(state, pack, ledger)
        visual = (
            VisualContract(type="success", data={"task": pack.task_id})
            if state.status is SessionStatus.COMPLETED
            else self._base_visual(pack)
        )
        completion = None
        if state.status is SessionStatus.COMPLETED and state.completion_outcome is not None:
            completion = CompletionContract(
                outcome=state.completion_outcome,
                teach_reward_eligible=state.teach_reward_eligible,
                stage_completion_eligible=True,
                verified_facts=self._completion_facts(pack, ledger),
            )
        note_update = self._note_update(state, pack, ledger)
        pedagogy = (
            PedagogySnapshot(
                expression_level=state.expression_level,
                hint_level=state.hint_level,
                subgoal_id=state.subgoal_id,
                verified_slots=dict(state.verified_slots),
            )
            if self.show_internal_pedagogy
            else None
        )
        return TurnContract(
            turn_id=new_id("turn"),
            scene=state.scene,
            scenario_id=state.scenario_id,
            task_id=pack.task_id,
            stage_id=pack.stage_id,
            task_index=state.task_index,
            mormi=MormiContract(text=text, mood=mood),
            input=question.input,
            visual=visual,
            help_card=self._help_card(pack, state.hint_level),
            note_update=note_update,
            status=state.status,
            state_version=state.state_version,
            completion=completion,
            pedagogy=pedagogy,
            task_anchor=(
                self._task_anchor(pack, question.input)
                if state.status is SessionStatus.ACTIVE
                else None
            ),
            dictionary_ref=(
                dictionary_reference(state.dictionary_snapshots[state.current_task_id])
                if state.current_task_id in state.dictionary_snapshots
                else None
            ),
        )

    @staticmethod
    def _note_update(
        state: SessionState,
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
    ) -> NoteUpdate | None:
        if state.status is not SessionStatus.COMPLETED:
            return None
        note_relations = set(pack.policies.note_relation_ids)
        if not note_relations.issubset(ledger.verified_relations):
            return None
        independent = note_relations.issubset(state.child_note_evidence)
        return NoteUpdate(
            skill_id=pack.curriculum_session_id,
            text=pack.help_plan.H3.body,
            attribution=(
                NoteAttribution.CHILD if independent else NoteAttribution.COAUTHORED
            ),
            evidence=(
                NoteEvidence.DIRECT_EXPLANATION
                if independent
                else NoteEvidence.SUPPORTED_COMPLETION
            ),
            attribution_label=(
                "아이가 알려줌" if independent else "아이와 함께 공부함"
            ),
        )

    @staticmethod
    def _base_visual(pack: RequiredHomeTeachingPackV2) -> VisualContract:
        problem = {
            "prompt": pack.source_problem.prompt,
            "answers": list(pack.source_problem.answers),
            "visual": dict(pack.source_problem.visual),
        }
        return VisualContract(
            type="home_teaching",
            data={
                "curriculum_session_id": pack.curriculum_session_id,
                "subject": "math",
                "unit": pack.title,
                "title": pack.title,
                "problem": problem,
            },
        )

    @staticmethod
    def _help_card(
        pack: RequiredHomeTeachingPackV2,
        level: HintLevel,
    ) -> HelpCardContract | None:
        if level is HintLevel.H0:
            return None
        card = {
            HintLevel.H1: pack.help_plan.H1,
            HintLevel.H2: pack.help_plan.H2,
            HintLevel.H3: pack.help_plan.H3,
        }[level]
        problem = {
            "prompt": pack.source_problem.prompt,
            "answers": list(pack.source_problem.answers),
            "visual": dict(pack.source_problem.visual),
        }
        return HelpCardContract(
            visible=True,
            auto_open=True,
            level=level,
            body=card.body,
            visual_type=(
                "joint_reading_card"
                if level is HintLevel.H3
                else "home_practice_problem"
                if level is HintLevel.H2
                else None
            ),
            visual_data=(
                {"text": card.body}
                if level is HintLevel.H3
                else problem
                if level is HintLevel.H2
                else {}
            ),
        )

    @staticmethod
    def _task_anchor(
        pack: RequiredHomeTeachingPackV2,
        input_contract: InputContract,
    ) -> TaskAnchorContract:
        return TaskAnchorContract(
            anchor_id=f"v2.{pack.pack_id}",
            prompt=pack.source_problem.prompt,
            target_slots=list(input_contract.target_slots),
        )

    @staticmethod
    def _completion_facts(
        pack: RequiredHomeTeachingPackV2,
        ledger: ReasoningLedgerV2,
    ) -> dict[str, str | int | float | bool]:
        facts = {fact.fact_id: fact for fact in pack.reasoning_graph.facts}
        relations = {
            relation.relation_id: relation for relation in pack.reasoning_graph.relations
        }
        result: dict[str, str | int | float | bool] = {}
        for fact_id in pack.reasoning_graph.completion.required_fact_ids:
            if fact_id in ledger.verified_facts:
                result[fact_id] = _primitive(facts[fact_id].value)
        for relation_id in pack.reasoning_graph.completion.required_relation_ids:
            if relation_id in ledger.verified_relations:
                result[relation_id] = relations[relation_id].speaker_label
        final = next(
            fact for fact in pack.reasoning_graph.facts if fact.role == "final_answer"
        )
        if final.fact_id in ledger.verified_facts:
            result["answer"] = _primitive(final.value)
        required_relations = pack.reasoning_graph.completion.required_relation_ids
        if required_relations and required_relations[0] in ledger.verified_relations:
            result["rule"] = relations[required_relations[0]].speaker_label
        return result

    @staticmethod
    def _compatibility_analysis(semantics: _TurnSemantics) -> UtteranceAnalysis:
        response = semantics.response
        if semantics.understanding_source == "deterministic_fallback":
            return UtteranceAnalysis(
                safety_category=SafetyCategory.NORMAL,
                response_category=ResponseCategory.RECOGNITION_OR_INPUT_ERROR,
                difficulty_class=DifficultyClass.INPUT,
                task_relation=TaskRelation.CURRENT_TASK,
                confidence=0,
            )
        if semantics.route == "safety":
            return UtteranceAnalysis(
                safety_category=(
                    SafetyCategory.PROMPT_INJECTION
                    if response.utterance_class.value == "system_manipulation"
                    else SafetyCategory.DANGEROUS
                ),
                response_category=ResponseCategory.UNRELATED_RESPONSE,
                difficulty_class=DifficultyClass.ENGAGEMENT,
                task_relation=TaskRelation.OFF_TOPIC,
                confidence=1,
            )
        if semantics.route == "bridge":
            return UtteranceAnalysis(
                safety_category=SafetyCategory.NORMAL,
                response_category=ResponseCategory.UNRELATED_RESPONSE,
                difficulty_class=DifficultyClass.ENGAGEMENT,
                task_relation=TaskRelation.OFF_TOPIC,
                conversation_only=True,
                confidence=1,
            )
        if semantics.understanding_source == "asr_empty":
            category = ResponseCategory.RECOGNITION_OR_INPUT_ERROR
            difficulty = DifficultyClass.INPUT
        elif response.conversation_move is ConversationMoveV2.TASK_QUESTION:
            category = ResponseCategory.RELATED_VAGUE
            difficulty = DifficultyClass.UNKNOWN
        elif semantics.understanding_source == "silence_timeout":
            category = ResponseCategory.NO_RESPONSE
            difficulty = DifficultyClass.EXPRESSION
        elif response.support_need is SupportNeed.GENERAL_HELP:
            category = ResponseCategory.HELP_REQUEST
            difficulty = DifficultyClass.EXPRESSION
        elif response.support_need is SupportNeed.EXPRESSION:
            category = ResponseCategory.EXPRESSION_BLOCK
            difficulty = DifficultyClass.EXPRESSION
        elif response.support_need in {SupportNeed.CONCEPT, SupportNeed.BOTH}:
            category = ResponseCategory.CONCEPTUAL_BLOCK
            difficulty = (
                DifficultyClass.BOTH
                if response.support_need is SupportNeed.BOTH
                else DifficultyClass.CONCEPT
            )
        elif semantics.apply_result.completion.complete:
            category = ResponseCategory.CORRECT_FULL
            difficulty = DifficultyClass.UNKNOWN
        elif semantics.apply_result.has_new_canonical_progress:
            category = ResponseCategory.CORRECT_PARTIAL
            difficulty = DifficultyClass.UNKNOWN
        elif semantics.structured_correct is False or any(
            getattr(claim, "verdict", None) == "incorrect" for claim in response.claims
        ):
            category = ResponseCategory.CONCEPTUAL_ERROR
            difficulty = DifficultyClass.CONCEPT
        else:
            category = ResponseCategory.RELATED_VAGUE
            difficulty = DifficultyClass.EXPRESSION
        return UtteranceAnalysis(
            safety_category=SafetyCategory.NORMAL,
            response_category=category,
            difficulty_class=difficulty,
            task_relation=TaskRelation.CURRENT_TASK,
            confidence={"low": 0.35, "medium": 0.7, "high": 1.0}[
                response.confidence.value
            ],
        )
