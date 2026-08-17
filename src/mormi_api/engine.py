from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .content import StepDefinition, TaskDefinition, get_task
from .dictionary_models import dictionary_reference
from .llm import (
    ClaudeGateway,
    extract_numeric_values,
    validate_speaker_output,
    validate_speaker_verification,
)
from .schemas import (
    CafeMenuItem,
    ChildResponse,
    CompletionContract,
    CompletionOutcome,
    DifficultyClass,
    EntryPhase,
    EntryStance,
    ExpressionLevel,
    HelpCardContract,
    HelpCardEvent,
    HintLevel,
    InputContract,
    InputKind,
    InteractionIntent,
    LearnerProfile,
    MormiContract,
    NoteAttribution,
    NoteEvidence,
    NoteUpdate,
    PedagogicalDecision,
    PedagogySnapshot,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SessionState,
    SessionStatus,
    SkillProfile,
    SpeakerContext,
    SpeakerGuardContract,
    SpeakerOutput,
    SpeakerQuestionIntent,
    SpeakerRuntimeAudit,
    SpeakerVerification,
    SpeakerVerificationPolicy,
    SupportTrigger,
    TaskRelation,
    TurnContract,
    UtteranceAnalysis,
    VisualContract,
    new_id,
)
from .security import (
    deterministic_safety,
    safety_redirect,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineProgress:
    stage: Literal["understanding", "planning", "speaking", "validating"]


@dataclass(frozen=True)
class EngineTurnResult:
    state: SessionState
    analysis: UtteranceAnalysis
    turn: TurnContract
    runtime: SpeakerRuntimeAudit
    accepted_claims: dict[str, str | int | float | bool]


class ConversationGraphState(TypedDict, total=False):
    session: dict[str, Any]
    response: dict[str, Any]
    previous_question: str
    analysis: dict[str, Any]
    decision: dict[str, Any]
    speaker_output: dict[str, Any]
    speaker_text: str
    runtime: dict[str, Any]
    turn: dict[str, Any]


class ConversationEngine:
    def __init__(
        self,
        gateway: ClaudeGateway,
        *,
        show_internal_pedagogy: bool = False,
        speaker_timeout_seconds: float = 8.0,
        semantic_verifier_enabled: bool = True,
        verifier_timeout_seconds: float = 1.8,
    ) -> None:
        self.gateway = gateway
        self.show_internal_pedagogy = show_internal_pedagogy
        self.speaker_timeout_seconds = speaker_timeout_seconds
        self.semantic_verifier_enabled = semantic_verifier_enabled
        self.verifier_timeout_seconds = verifier_timeout_seconds
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(ConversationGraphState)
        # LangGraph's current generic overloads do not accept async bound methods
        # cleanly in mypy even though they are supported at runtime.
        builder.add_node("understand", self._understand_node)  # type: ignore[call-overload]
        builder.add_node("orchestrate", self._orchestrate_node)  # type: ignore[call-overload]
        builder.add_node("speak", self._speak_node)  # type: ignore[call-overload]
        builder.add_node(  # type: ignore[call-overload]
            "validate_and_compose",
            self._validate_and_compose_node,
        )
        builder.add_edge(START, "understand")
        builder.add_edge("understand", "orchestrate")
        builder.add_edge("orchestrate", "speak")
        builder.add_edge("speak", "validate_and_compose")
        builder.add_edge("validate_and_compose", END)
        # Canonical state is committed atomically to the encrypted application
        # database after this per-turn graph succeeds. We deliberately do not
        # checkpoint raw child text inside LangGraph.
        return builder.compile()

    async def run_turn(
        self,
        state: SessionState,
        response: ChildResponse,
        previous_question: str,
    ) -> tuple[SessionState, UtteranceAnalysis, TurnContract]:
        result: EngineTurnResult | None = None
        async for event in self.run_turn_stream(state, response, previous_question):
            if isinstance(event, EngineTurnResult):
                result = event
        if result is None:  # pragma: no cover - LangGraph always reaches END
            raise RuntimeError("Conversation graph produced no final turn")
        return result.state, result.analysis, result.turn

    async def run_turn_stream(
        self,
        state: SessionState,
        response: ChildResponse,
        previous_question: str,
    ) -> AsyncIterator[EngineProgress | EngineTurnResult]:
        """Expose safe graph milestones without leaking unvalidated model text."""

        final_values: dict[str, Any] | None = None
        node_stages: dict[str, EngineProgress] = {
            "understand": EngineProgress("understanding"),
            "orchestrate": EngineProgress("planning"),
            "speak": EngineProgress("speaking"),
            "validate_and_compose": EngineProgress("validating"),
        }
        async for mode, payload in self.graph.astream(
            {
                "session": state.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "previous_question": previous_question,
            },
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node_name in payload:
                    progress = node_stages.get(node_name)
                    if progress:
                        yield progress
            elif mode == "values":
                final_values = payload

        if final_values is None or "turn" not in final_values:
            raise RuntimeError("Conversation graph produced no final turn")
        decision = PedagogicalDecision.model_validate(final_values["decision"])
        yield EngineTurnResult(
            state=SessionState.model_validate(final_values["session"]),
            analysis=UtteranceAnalysis.model_validate(final_values["analysis"]),
            turn=TurnContract.model_validate(final_values["turn"]),
            runtime=SpeakerRuntimeAudit.model_validate(final_values["runtime"]),
            accepted_claims=decision.accepted_claims,
        )

    def initial_turn(self, state: SessionState) -> TurnContract:
        task = get_task(state.current_task_id, state.scenario_data)
        step = task.active_step(
            state.expression_level,
            state.verified_slots,
            entry_active=(
                state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE
                and state.expression_level is ExpressionLevel.L4
            ),
            targeted_followup=(state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP),
        )
        state.subgoal_id = step.id
        state.current_turn_id = new_id("turn")
        return self._turn_contract(
            state,
            task,
            text=step.prompt,
            input_contract=step.input,
            visual=task.base_visual,
            help_card=self._help_card(task, state.hint_level),
            mood="curious",
        )

    async def _understand_node(self, graph_state: ConversationGraphState) -> dict[str, Any]:
        state = SessionState.model_validate(graph_state["session"])
        response = ChildResponse.model_validate(graph_state["response"])
        task = get_task(state.current_task_id, state.scenario_data)
        previous_question = graph_state["previous_question"]

        # "잘 모르겠어" 버튼 is an expression-support request, not evidence
        # that the child holds a mathematical misconception.  It bypasses the
        # LLM because the UI action already has one unambiguous meaning.
        if response.type is ResponseType.NO_RESPONSE:
            analysis = UtteranceAnalysis(
                safety_category=SafetyCategory.NORMAL,
                response_category=ResponseCategory.HELP_REQUEST,
                difficulty_class=DifficultyClass.EXPRESSION,
                task_relation=TaskRelation.CURRENT_TASK,
                bottleneck="expression",
                confidence=1,
            )
            return {"analysis": analysis.model_dump(mode="json")}

        text = response.text or self._response_as_text(response)
        deterministic_category = deterministic_safety(text)
        if deterministic_category is not SafetyCategory.NORMAL:
            category = (
                ResponseCategory.UNRELATED_RESPONSE
                if deterministic_category is SafetyCategory.PLAYFUL_OFFTOPIC
                else ResponseCategory.RECOGNITION_OR_INPUT_ERROR
            )
            analysis = UtteranceAnalysis(
                safety_category=deterministic_category,
                response_category=category,
                difficulty_class=DifficultyClass.ENGAGEMENT,
                task_relation=TaskRelation.OFF_TOPIC,
                interaction_intent=(
                    InteractionIntent.PLAYFUL_TEASE
                    if deterministic_category is SafetyCategory.PLAYFUL_OFFTOPIC
                    else InteractionIntent.NONE
                ),
                social_grounding_span=(
                    text if deterministic_category is SafetyCategory.PLAYFUL_OFFTOPIC else ""
                ),
                confidence=1,
            )
            return {"analysis": analysis.model_dump(mode="json")}

        if response.type is ResponseType.TEXT:
            analysis = await self.gateway.classify(
                state=state,
                task=task,
                previous_question=previous_question,
                response=response,
            )
            # Deterministic safety always wins; model safety may make the result
            # stricter but can never downgrade an explicit local match.
            if analysis.safety_category is SafetyCategory.UNKNOWN:
                analysis.safety_category = SafetyCategory.NORMAL
            return {"analysis": analysis.model_dump(mode="json")}

        analysis = self._deterministic_analysis(state, task, response)
        return {"analysis": analysis.model_dump(mode="json")}

    async def _orchestrate_node(self, graph_state: ConversationGraphState) -> dict[str, Any]:
        state = SessionState.model_validate(graph_state["session"])
        response = ChildResponse.model_validate(graph_state["response"])
        analysis = UtteranceAnalysis.model_validate(graph_state["analysis"])
        task = get_task(state.current_task_id, state.scenario_data)
        decision = self._decide(
            state,
            task,
            response,
            analysis,
            graph_state["previous_question"],
        )
        return {
            "session": decision.state.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }

    async def _speak_node(self, graph_state: ConversationGraphState) -> dict[str, Any]:
        decision = PedagogicalDecision.model_validate(graph_state["decision"])
        context = decision.speaker_context
        if decision.dialogue_act in {
            "unsafe_redirect",
            "joint_mode",
            # These two turns are also rendered beside a star-note update.
            # Their reviewed fallback deliberately contains no mathematical
            # claim, so a speaker model can never add a strategy the child did
            # not teach while celebrating or moving to the next task.
            "session_complete",
            "task_transition",
        }:
            runtime = SpeakerRuntimeAudit(
                dialogue_act=decision.dialogue_act,
                speaker_source="reviewed_fallback",
                verifier_status="not_required",
            )
            return {
                "speaker_text": context.fallback_text,
                "runtime": runtime.model_dump(mode="json"),
            }
        try:
            async with asyncio.timeout(self.speaker_timeout_seconds):
                output = await self.gateway.speak(context)
            runtime = SpeakerRuntimeAudit(
                dialogue_act=decision.dialogue_act,
                speaker_source="llm",
                verifier_status=(
                    "disabled"
                    if context.verification_policy is SpeakerVerificationPolicy.SEMANTIC
                    and not self.semantic_verifier_enabled
                    else "not_required"
                ),
            )
            return {
                "speaker_output": output.model_dump(mode="json"),
                "runtime": runtime.model_dump(mode="json"),
            }
        except Exception as error:
            logger.warning(
                "speaker_fallback stage=generation error_type=%s",
                type(error).__name__,
            )
            runtime = SpeakerRuntimeAudit(
                dialogue_act=decision.dialogue_act,
                speaker_source="generation_fallback",
                verifier_status="not_required",
                fallback_reason=type(error).__name__,
            )
            return {
                "speaker_text": context.fallback_text,
                "runtime": runtime.model_dump(mode="json"),
            }

    async def _validate_and_compose_node(
        self,
        graph_state: ConversationGraphState,
    ) -> dict[str, Any]:
        decision = PedagogicalDecision.model_validate(graph_state["decision"])
        task = get_task(decision.state.current_task_id, decision.state.scenario_data)
        guard = self._speaker_guard(task, decision.state, decision.speaker_context)
        runtime = SpeakerRuntimeAudit.model_validate(graph_state["runtime"])
        text = graph_state.get("speaker_text")
        if not text and graph_state.get("speaker_output"):
            output = SpeakerOutput.model_validate(graph_state["speaker_output"])
            text = validate_speaker_output(output, decision.speaker_context, guard)
            if not text:
                runtime = runtime.model_copy(
                    update={
                        "speaker_source": "deterministic_validation_fallback",
                        "fallback_reason": "speaker_contract_rejected",
                    }
                )
            if (
                text
                and decision.speaker_context.verification_policy
                is SpeakerVerificationPolicy.SEMANTIC
                and self.semantic_verifier_enabled
            ):
                try:
                    async with asyncio.timeout(self.verifier_timeout_seconds):
                        verification = await self.gateway.verify_speaker(
                            decision.speaker_context,
                            guard,
                            output,
                        )
                    verified = SpeakerVerification.model_validate(verification)
                    if not validate_speaker_verification(
                        verified,
                        decision.speaker_context,
                        guard,
                        output,
                    ):
                        text = None
                        runtime = runtime.model_copy(
                            update={
                                "speaker_source": "semantic_verification_fallback",
                                "verifier_status": "rejected",
                                "fallback_reason": verified.reason_code,
                            }
                        )
                        logger.info(
                            "speaker_fallback stage=semantic reason=%s dialogue_act=%s",
                            verified.reason_code,
                            decision.dialogue_act,
                        )
                    else:
                        runtime = runtime.model_copy(update={"verifier_status": "approved"})
                except Exception as error:
                    text = None
                    runtime = runtime.model_copy(
                        update={
                            "speaker_source": "semantic_verification_fallback",
                            "verifier_status": "error",
                            "fallback_reason": type(error).__name__,
                        }
                    )
                    logger.warning(
                        "speaker_fallback stage=verification error_type=%s",
                        type(error).__name__,
                    )
        text = text or decision.speaker_context.fallback_text
        turn = self._turn_contract(
            decision.state,
            task,
            text=text,
            input_contract=decision.input,
            visual=decision.visual,
            help_card=decision.help_card,
            mood=decision.mood,
            note=decision.note_update,
        )
        decision.state.current_turn_id = turn.turn_id
        return {
            "session": decision.state.model_dump(mode="json"),
            "turn": turn.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
        }

    def _decide(
        self,
        state: SessionState,
        task: TaskDefinition,
        response: ChildResponse,
        analysis: UtteranceAnalysis,
        previous_question: str,
    ) -> PedagogicalDecision:
        next_state = state.model_copy(deep=True)
        next_state.state_version += 1
        next_state.current_turn_id = None
        entry_active = (
            task.entry_step is not None and state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE
        )
        open_followup_active = state.entry_phase is EntryPhase.AWAITING_OPEN_FOLLOWUP
        targeted_followup_active = state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP
        interpreted_step = task.active_step(
            state.expression_level,
            state.verified_slots,
            entry_active=entry_active,
            targeted_followup=targeted_followup_active,
        )
        interpreted_slots = {
            *interpreted_step.target_slots,
            *interpreted_step.optional_slots,
        }

        if analysis.safety_category is not SafetyCategory.NORMAL:
            if analysis.safety_category is SafetyCategory.PLAYFUL_OFFTOPIC:
                next_state.unrelated_count += 1
                return self._decision_for_current_step(
                    next_state,
                    task,
                    dialogue_act="brief_ack_and_redirect",
                    fallback=self._social_bridge_fallback(
                        InteractionIntent.PLAYFUL_TEASE,
                        next_state.unrelated_count,
                    ),
                    child_text=response.text,
                    analysis=analysis,
                    previous_question=previous_question,
                    must_reframe=True,
                )
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="unsafe_redirect",
                fallback=safety_redirect(analysis.safety_category),
                child_text=None,
                analysis=analysis,
                previous_question=previous_question,
            )

        # Conversational relation is independent of mathematical correctness.
        # Route a safely grounded social intent before any ladder or claim
        # logic so a classifier's secondary correctness label cannot turn a
        # meta remark into a concept failure or learning evidence.
        social_grounding = self._safe_social_grounding_span(
            response.text,
            analysis,
            allowed_numbers=extract_numeric_values(
                " ".join(
                    [
                        previous_question,
                        *(str(value) for value in task.visible_facts.values()),
                    ]
                )
            ),
        )
        if (
            social_grounding
            and analysis.interaction_intent is not InteractionIntent.NONE
            and analysis.task_relation
            in {TaskRelation.META_ABOUT_MORMI, TaskRelation.OFF_TOPIC}
        ):
            next_state.unrelated_count += 1
            meta_turn = analysis.task_relation is TaskRelation.META_ABOUT_MORMI
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act=(
                    "acknowledge_meta_and_reask"
                    if meta_turn
                    else "brief_ack_and_redirect"
                ),
                fallback=self._social_bridge_fallback(
                    analysis.interaction_intent,
                    next_state.unrelated_count,
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                must_reframe=True,
            )

        # Accepting a reviewed wrong guess is concept evidence, never a
        # successful conversational answer.  Conversely, rejecting it is only
        # a stance: it cannot fill answer/reason slots or enter the star note.
        if entry_active and analysis.entry_stance is EntryStance.ACCEPT_WRONG_GUESS:
            analysis.response_category = ResponseCategory.CONCEPTUAL_ERROR
            analysis.difficulty_class = DifficultyClass.CONCEPT

        if task.behavior == "budget_menu_selection" and response.choice_ids:
            menu_items = [
                CafeMenuItem.model_validate(item)
                for item in task.visible_facts.get("menu_items", [])
            ]
            menu_by_id = {item.id: item for item in menu_items}
            selected = menu_by_id.get(response.choice_ids[0])
            mormi_data = task.visible_facts.get("mormi_menu", {})
            budget = int(task.visible_facts.get("budget") or 0)
            mormi_price = int(mormi_data.get("price", 0)) if isinstance(mormi_data, dict) else 0
            if selected and mormi_price + selected.price > budget:
                total = mormi_price + selected.price
                decision = self._decision_for_current_step(
                    next_state,
                    task,
                    dialogue_act="budget_exceeded",
                    fallback="장바구니가 예산을 넘었네. 다른 메뉴로 바꿔볼까?",
                    child_text=response.text,
                    analysis=analysis,
                    previous_question=previous_question,
                )
                decision.visual = task.base_visual.model_copy(
                    update={
                        "data": {
                            **task.base_visual.data,
                            "child_pick": selected.model_dump(),
                            "total": total,
                            "budget_status": "over",
                        }
                    },
                    deep=True,
                )
                return decision

        successful_categories = {
            ResponseCategory.CORRECT_FULL,
            ResponseCategory.CORRECT_PARTIAL,
            ResponseCategory.SELF_CORRECTION,
        }
        # Completion requires two independent gates: the understanding stage
        # must call the response successful, and every accepted claim must
        # match reviewed curriculum facts.  An error analysis can therefore
        # never fill a slot even if a malformed model/client supplies a value.
        grounded_claims = (
            task.validated_claims(
                (claim.slot_id, claim.value, claim.factual) for claim in analysis.claims
            )
            if analysis.response_category in successful_categories
            else {}
        )
        if response.type is ResponseType.TEXT and response.text:
            grounded_claims = self._filter_text_explanation_claims(
                task,
                response.text,
                analysis,
                grounded_claims,
            )
        # Only claims requested by the current turn may advance the state.
        # A child can still repeat a previously verified observation or result
        # naturally. Keep that grounded fact available to the speaker for
        # acknowledgement without counting it as progress.
        accepted_claims = {
            slot_id: value
            for slot_id, value in grounded_claims.items()
            if slot_id in interpreted_slots
        }
        understood_claims = {
            slot_id: value
            for slot_id, value in grounded_claims.items()
            if state.verified_slots.get(slot_id) == value
        }
        understood_claims.update(accepted_claims)
        # A repeated fact is still understood, but it is not new learning
        # progress. Treating it as newly verified can select the same unresolved
        # step forever instead of changing the support contract.
        newly_verified = {
            slot_id: value
            for slot_id, value in accepted_claims.items()
            if next_state.verified_slots.get(slot_id) != value
        }
        next_state.verified_slots.update(newly_verified)
        if (
            entry_active
            and analysis.entry_stance is EntryStance.REJECT_WRONG_GUESS
            and not newly_verified
        ):
            next_state.entry_phase = EntryPhase.AWAITING_OPEN_FOLLOWUP
            next_state.unrelated_count = 0
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="entry_rejection_followup",
                fallback=task.step_for(
                    next_state.expression_level,
                    next_state.verified_slots,
                ).prompt,
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
            )
        if (entry_active or open_followup_active or targeted_followup_active) and newly_verified:
            next_state.entry_phase = EntryPhase.RESOLVED
        if newly_verified:
            next_state.vague_clarifications = 0
            self._record_note_provenance(
                next_state,
                task,
                response,
                analysis,
                newly_verified,
            )
            if "child_menu" in newly_verified:
                next_state.scenario_data["child_menu_id"] = newly_verified["child_menu"]
                next_state.scenario_data.pop("last_over_budget_total", None)
                next_state.scenario_data.pop("last_child_menu_id", None)
            next_state.unrelated_count = 0
            if task.complete(next_state.verified_slots):
                return self._complete_task(
                    next_state,
                    task,
                    analysis,
                    response.text,
                    previous_question,
                    accepted_claims=understood_claims,
                )
            next_step = task.step_for(
                next_state.expression_level,
                next_state.verified_slots,
            )
            # A child who supplies one useful part of an L4 answer has already
            # responded independently, regardless of whether the scene is the
            # home or the cafe. Keep the L4 credit and split only the question:
            # the next turn uses the shorter L3-shaped prompt for the missing
            # slot. Scene-specific handling here used to make cafe calculations
            # repeat the original result+method question after a correct result.
            split_l4_followup = (
                next_state.expression_level is ExpressionLevel.L4
                and next_step.id == state.subgoal_id
            )
            if (
                entry_active
                or open_followup_active
                or targeted_followup_active
                or split_l4_followup
            ):
                next_state.entry_phase = EntryPhase.AWAITING_TARGETED_FOLLOWUP
            # A partial answer is useful evidence, but the next prompt should
            # request only the missing piece instead of repeating the original
            # multi-part L4 question.
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="acknowledge_partial",
                fallback=self._partial_fallback(task, newly_verified, next_state),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                newly_verified=newly_verified,
                accepted_claims=understood_claims,
            )

        # A colloquial answer can be meaningfully related while still being
        # too incomplete to fill a canonical curriculum slot.  The classifier
        # is allowed to return correct_partial without inventing a claim.  Do
        # not erase that understanding by falling through to the unrelated
        # branch; move to a more concrete expression step instead.
        if analysis.response_category in successful_categories:
            if entry_active or open_followup_active or targeted_followup_active:
                next_state.entry_phase = EntryPhase.RESOLVED
            self._lower_until_visible_contract_changes(
                next_state,
                task,
                interpreted_step,
            )
            if next_state.expression_level is ExpressionLevel.L0:
                next_state.hint_level = HintLevel.H3
                next_state.task_max_hint = HintLevel.H3
                dialogue_act = "joint_mode"
                fallback = "같은 말만 다시 물어서 미안해. 도움 카드대로 같이 해볼까?"
            else:
                dialogue_act = "acknowledge_unstructured_partial"
                next_step = task.step_for(
                    next_state.expression_level,
                    next_state.verified_slots,
                )
                if understood_claims:
                    acknowledgement = self._younger_sibling_acknowledgement(
                        task,
                        understood_claims,
                    )
                    fallback = self._preface_question(acknowledgement, next_step.prompt)
                else:
                    fallback = self._preface_question(
                        "앗, 내가 한 번에 많이 물어봤네.",
                        next_step.prompt,
                    )
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act=dialogue_act,
                fallback=fallback,
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                newly_verified=understood_claims,
                accepted_claims=understood_claims,
            )

        category = analysis.response_category
        if category is ResponseCategory.RELATED_VAGUE:
            if entry_active or open_followup_active or targeted_followup_active:
                next_state.entry_phase = EntryPhase.RESOLVED
            next_state.expression_failures += 1
            # The first on-topic but vague reply gets one conversational
            # clarification at the same L/H level.  It is not a request for
            # conceptual help and must not open a card by itself.
            first_clarification = next_state.vague_clarifications == 0
            next_state.vague_clarifications += 1
            help_event = HelpCardEvent.NONE
            if not first_clarification:
                next_state.vague_clarifications = 0
                self._lower_until_visible_contract_changes(
                    next_state,
                    task,
                    interpreted_step,
                )
                previous_hint = next_state.hint_level
                if next_state.expression_level is ExpressionLevel.L0:
                    next_state.hint_level = HintLevel.H3
                elif next_state.hint_level is HintLevel.H0:
                    next_state.hint_level = HintLevel.H1
                help_event = self._help_card_event(
                    previous_hint,
                    next_state.hint_level,
                )
                next_state.task_max_hint = max_hint(
                    next_state.task_max_hint,
                    next_state.hint_level,
                )
            step = task.step_for(
                next_state.expression_level,
                next_state.verified_slots,
            )
            grounding = self._safe_grounding_for_step(
                task,
                next_state,
                step,
                response.text,
                analysis,
            )
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act=(
                    "clarify_vague_response" if first_clarification else "support_vague_response"
                ),
                fallback=self._support_transition_fallback(
                    SupportTrigger.RELATED_VAGUE,
                    help_event,
                    grounding,
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                support_trigger=SupportTrigger.RELATED_VAGUE,
                help_card_event=help_event,
                must_reframe=True,
            )

        if category in {
            ResponseCategory.EXPRESSION_BLOCK,
            ResponseCategory.NO_RESPONSE,
        }:
            if entry_active or open_followup_active or targeted_followup_active:
                next_state.entry_phase = EntryPhase.RESOLVED
            next_state.expression_failures += 1
            next_state.vague_clarifications = 0
            next_state.expression_level = next_state.expression_level.lower()
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="reduce_expression_load",
                fallback=self._smooth_ladder_fallback(next_state, task),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                support_trigger=SupportTrigger.EXPRESSION_BLOCK,
                must_reframe=True,
            )

        if category is ResponseCategory.HELP_REQUEST:
            if entry_active or open_followup_active or targeted_followup_active:
                next_state.entry_phase = EntryPhase.RESOLVED
            next_state.expression_failures += 1
            next_state.vague_clarifications = 0
            next_state.expression_level = next_state.expression_level.lower()
            previous_hint = next_state.hint_level
            if next_state.hint_level is HintLevel.H0:
                next_state.hint_level = HintLevel.H1
            elif next_state.hint_level is HintLevel.H1:
                next_state.hint_level = HintLevel.H2
            elif next_state.expression_level is ExpressionLevel.L0:
                next_state.hint_level = HintLevel.H3
            next_state.task_max_hint = max_hint(
                next_state.task_max_hint,
                next_state.hint_level,
            )
            joint_mode = next_state.expression_level is ExpressionLevel.L0
            if joint_mode:
                next_state.hint_level = HintLevel.H3
                next_state.task_max_hint = HintLevel.H3
            help_event = self._help_card_event(previous_hint, next_state.hint_level)
            decision = self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="joint_mode" if joint_mode else "accept_help_request",
                fallback=self._support_transition_fallback(
                    SupportTrigger.EXPLICIT_HELP_REQUEST,
                    help_event,
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                support_trigger=SupportTrigger.EXPLICIT_HELP_REQUEST,
                help_card_event=help_event,
                must_reframe=not joint_mode,
            )
            # A help request changes the amount of support, never the problem
            # the child is looking at.  Hint visuals remain available inside
            # the help-card contract while the main task visual stays stable.
            decision.visual = task.base_visual.model_copy(deep=True)
            return decision

        if category in {
            ResponseCategory.CONCEPTUAL_ERROR,
            ResponseCategory.CONCEPTUAL_BLOCK,
        } or analysis.difficulty_class in {DifficultyClass.CONCEPT, DifficultyClass.BOTH}:
            if entry_active or open_followup_active or targeted_followup_active:
                next_state.entry_phase = EntryPhase.RESOLVED
            next_state.concept_failures += 1
            next_state.vague_clarifications = 0
            previous_hint = next_state.hint_level
            next_state.task_max_hint = max_hint(
                next_state.task_max_hint, next_state.hint_level.increase()
            )
            if next_state.hint_level is HintLevel.H0:
                next_state.hint_level = HintLevel.H1
            elif next_state.hint_level is HintLevel.H1:
                next_state.hint_level = HintLevel.H2
            elif next_state.hint_level is HintLevel.H2 and next_state.expression_level.rank > 1:
                next_state.expression_level = ExpressionLevel.L1
            else:
                next_state.expression_level = ExpressionLevel.L0
                next_state.hint_level = HintLevel.H3
                next_state.task_max_hint = HintLevel.H3
            help_event = self._help_card_event(previous_hint, next_state.hint_level)
            step = task.step_for(
                next_state.expression_level,
                next_state.verified_slots,
            )
            grounding = self._safe_grounding_for_step(
                task,
                next_state,
                step,
                response.text,
                analysis,
            )
            support_trigger = (
                SupportTrigger.REPEATED_CONCEPTUAL_CONFLICT
                if next_state.concept_failures > 1
                else SupportTrigger.CONCEPTUAL_CONFLICT
            )
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="joint_mode"
                if next_state.hint_level is HintLevel.H3
                else "show_help_card",
                fallback=self._support_transition_fallback(
                    support_trigger,
                    help_event,
                    grounding,
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                support_trigger=support_trigger,
                help_card_event=help_event,
                must_reframe=next_state.hint_level is not HintLevel.H3,
            )

        if category is ResponseCategory.RECOGNITION_OR_INPUT_ERROR:
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="clarify_input",
                fallback="내가 말을 잘 못 알아들었어. 한 번만 다시 말해줘.",
                child_text=None,
                analysis=analysis,
                previous_question=previous_question,
            )

        next_state.unrelated_count += 1
        return self._decision_for_current_step(
            next_state,
            task,
            dialogue_act="context_return",
            fallback="그 얘기는 이따 하자.",
            child_text=response.text,
            analysis=analysis,
            previous_question=previous_question,
        )

    def _complete_task(
        self,
        state: SessionState,
        task: TaskDefinition,
        analysis: UtteranceAnalysis,
        child_text: str | None,
        previous_question: str,
        *,
        accepted_claims: Mapping[str, str | int | float | bool],
    ) -> PedagogicalDecision:
        note, direct = self._note_from_provenance(state, task)
        if task.note_policy != "none":
            state.all_tasks_direct = state.all_tasks_direct and direct
        # The speaker may celebrate only what the note provenance permits.
        # Feeding a catalog rule here used to make Mormi suddenly announce a
        # strategy the child never mentioned (for example, pairing dots).
        contribution = note.text if note else ""

        if state.task_index + 1 < len(state.task_ids):
            state.task_index += 1
            state.expression_level = state.task_start_levels.get(
                state.current_task_id,
                ExpressionLevel.L2,
            )
            state.task_start_level = state.expression_level
            state.hint_level = HintLevel.H0
            state.task_max_hint = HintLevel.H0
            state.subgoal_id = "initial"
            state.verified_slots = {}
            state.expression_failures = 0
            state.concept_failures = 0
            state.vague_clarifications = 0
            state.child_note_evidence = {}
            state.supported_note_slots = []
            next_task = get_task(state.current_task_id, state.scenario_data)
            state.entry_phase = (
                EntryPhase.AWAITING_ENTRY_RESPONSE
                if state.dialogue_policy_version == 2
                and next_task.entry_step is not None
                and state.expression_level is ExpressionLevel.L4
                else EntryPhase.RESOLVED
            )
            next_step = next_task.active_step(
                state.expression_level,
                state.verified_slots,
                entry_active=state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE,
            )
            fallback = self._success_then_question(contribution, next_step.prompt)
            return PedagogicalDecision(
                state=state,
                dialogue_act="task_transition",
                accepted_claims=dict(accepted_claims),
                required_question=next_step.prompt,
                input=next_step.input,
                visual=next_task.base_visual,
                help_card=None,
                note_update=note,
                mood="relieved",
                speaker_context=self._speaker_context(
                    task=next_task,
                    state=state,
                    dialogue_act="task_transition",
                    previous_question=previous_question,
                    required_question=next_step.prompt,
                    verified_facts=(
                        {"previous_contribution": contribution} if contribution else {}
                    ),
                    analysis=analysis,
                    child_text=child_text,
                    fallback=fallback,
                ),
            )

        state.status = SessionStatus.COMPLETED
        state.completion_outcome = (
            CompletionOutcome.TAUGHT if state.all_tasks_direct else CompletionOutcome.SUPPORTED
        )
        state.teach_reward_eligible = True
        fallback = self._fit_50("네가 알려준 방법으로 내가 끝까지 해냈어!")
        return PedagogicalDecision(
            state=state,
            dialogue_act="session_complete",
            accepted_claims=dict(accepted_claims),
            required_question=None,
            input=InputContract(kind=InputKind.NONE),
            visual=VisualContract(type="success", data={"task": task.id}),
            help_card=None,
            note_update=note,
            mood="celebrating",
            speaker_context=self._speaker_context(
                task=task,
                state=state,
                dialogue_act="session_complete",
                previous_question=previous_question,
                required_question=None,
                verified_facts=({"completion_contribution": contribution} if contribution else {}),
                analysis=analysis,
                child_text=child_text,
                fallback=fallback,
            ),
        )

    def _decision_for_current_step(
        self,
        state: SessionState,
        task: TaskDefinition,
        *,
        dialogue_act: str,
        fallback: str,
        child_text: str | None,
        analysis: UtteranceAnalysis,
        previous_question: str,
        newly_verified: Mapping[str, object] | None = None,
        accepted_claims: Mapping[str, str | int | float | bool] | None = None,
        support_trigger: SupportTrigger = SupportTrigger.NONE,
        help_card_event: HelpCardEvent = HelpCardEvent.NONE,
        must_reframe: bool = False,
    ) -> PedagogicalDecision:
        if state.hint_level is HintLevel.H3:
            state.expression_level = ExpressionLevel.L0
        step = task.active_step(
            state.expression_level,
            state.verified_slots,
            entry_active=(state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE),
            targeted_followup=(state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP),
        )
        state.subgoal_id = step.id
        help_card = self._help_card(task, state.hint_level)
        visual = self._visual_for(task, state.hint_level)
        verified_facts = {slot: task.slots[slot].fact_sentence for slot in (newly_verified or {})}
        # Support branches provide a complete, reason-specific fallback.
        # Never overwrite it with a generic prefix + the previous catalog
        # question; that was the source of robotic semantic repetition.
        context = self._speaker_context(
            task=task,
            state=state,
            dialogue_act=dialogue_act,
            previous_question=previous_question,
            required_question=step.prompt,
            verified_facts=verified_facts,
            analysis=analysis,
            child_text=child_text,
            fallback=self._fit_50(fallback),
            support_trigger=support_trigger,
            help_card_event=help_card_event,
            must_reframe=must_reframe,
        )
        if (
            not must_reframe
            or context.verification_policy is SpeakerVerificationPolicy.DETERMINISTIC
        ):
            context.fallback_text = self._ensure_required_question(fallback, step.prompt)
        return PedagogicalDecision(
            state=state,
            dialogue_act=dialogue_act,
            accepted_claims=dict(accepted_claims or {}),
            required_question=step.prompt,
            input=step.input,
            visual=visual,
            help_card=help_card,
            mood="thinking" if help_card else "listening",
            speaker_context=context,
        )

    def _speaker_context(
        self,
        *,
        task: TaskDefinition,
        state: SessionState,
        dialogue_act: str,
        previous_question: str,
        required_question: str | None,
        verified_facts: Mapping[str, str],
        analysis: UtteranceAnalysis,
        child_text: str | None,
        fallback: str,
        support_trigger: SupportTrigger = SupportTrigger.NONE,
        help_card_event: HelpCardEvent = HelpCardEvent.NONE,
        must_reframe: bool = False,
    ) -> SpeakerContext:
        active_step = task.active_step(
            state.expression_level,
            state.verified_slots,
            entry_active=(state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE),
            targeted_followup=(state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP),
        )
        required_slot_ids = (
            [slot for slot in active_step.target_slots if slot not in state.verified_slots]
            if required_question
            else []
        )
        required_slot_descriptions = {
            slot: task.slots[slot].description for slot in required_slot_ids
        }
        allowed_numbers = sorted(
            extract_numeric_values(" ".join([*verified_facts.values(), required_question or ""]))
        )
        expression = self._safe_grounding_span(
            child_text,
            analysis,
            allowed_numbers=set(allowed_numbers),
        )
        if dialogue_act in {
            "acknowledge_meta_and_reask",
            "brief_ack_and_redirect",
        }:
            expression = self._safe_social_grounding_span(
                child_text,
                analysis,
                allowed_numbers=set(allowed_numbers),
            )
        mode: Literal["none", "quote_safe"] = "quote_safe" if expression else "none"
        question_intent = SpeakerQuestionIntent(
            operation=task.skill_id,
            response_kind=self._speaker_response_kind(
                task,
                active_step,
                required_slot_ids,
            ),
            referents=[task.title, *allowed_numbers] if required_question else [],
            required_meanings=required_slot_descriptions,
        )
        verification_policy = self._speaker_verification_policy(
            task,
            dialogue_act,
            required_slot_ids,
            has_child_grounding=bool(expression),
        )
        return SpeakerContext(
            dialogue_act=dialogue_act,
            task_relation=analysis.task_relation,
            interaction_intent=analysis.interaction_intent,
            interaction_repeat_count=state.unrelated_count,
            support_trigger=support_trigger,
            help_card_event=help_card_event,
            must_reframe=must_reframe,
            previous_question=previous_question,
            required_question=required_question,
            verified_facts=dict(verified_facts),
            required_slot_ids=required_slot_ids,
            required_slot_descriptions=required_slot_descriptions,
            question_intent=question_intent,
            child_expression_mode=mode,
            child_expression=expression,
            allowed_numbers=allowed_numbers,
            verification_policy=verification_policy,
            fallback_text=fallback,
        )

    @staticmethod
    def _speaker_response_kind(
        task: TaskDefinition,
        step: StepDefinition,
        required_slot_ids: list[str],
    ) -> Literal[
        "answer",
        "reason_or_method",
        "answer_and_reason",
        "choice",
        "count",
        "relation",
        "action",
        "joint",
        "none",
    ]:
        if not required_slot_ids:
            return "none"
        if step.input.kind is InputKind.JOINT:
            return "joint"
        if step.input.kind is InputKind.COUNT:
            return "count"
        if step.input.kind is InputKind.EQUATION:
            return "action"
        explanation_slots = set(task.text_explanation_slots)
        required = set(required_slot_ids)
        if explanation_slots.intersection(required) and required - explanation_slots:
            return "answer_and_reason"
        if required.issubset(explanation_slots) or required.intersection(
            {"reason", "rule", "tracking", "method", "operation"}
        ):
            return "reason_or_method"
        if step.input.kind in {InputKind.CHOICES, InputKind.FILL}:
            return "choice"
        if any("관계" in task.slots[slot].description for slot in required_slot_ids):
            return "relation"
        return "answer"

    def _speaker_verification_policy(
        self,
        task: TaskDefinition,
        dialogue_act: str,
        required_slot_ids: list[str],
        *,
        has_child_grounding: bool,
    ) -> SpeakerVerificationPolicy:
        if not self.semantic_verifier_enabled:
            return SpeakerVerificationPolicy.DETERMINISTIC
        explanation_slots = set(task.text_explanation_slots)
        semantic_dialogue_acts = {
            "acknowledge_meta_and_reask",
            "acknowledge_partial",
            "acknowledge_unstructured_partial",
            "accept_help_request",
            "brief_ack_and_redirect",
            "clarify_vague_response",
            "entry_rejection_followup",
            "reduce_expression_load",
            "show_help_card",
            "support_vague_response",
        }
        if (
            has_child_grounding
            or bool(explanation_slots.intersection(required_slot_ids))
            or dialogue_act in semantic_dialogue_acts
        ):
            return SpeakerVerificationPolicy.SEMANTIC
        return SpeakerVerificationPolicy.DETERMINISTIC

    @staticmethod
    def _safe_grounding_span(
        child_text: str | None,
        analysis: UtteranceAnalysis,
        *,
        allowed_numbers: set[str],
    ) -> str | None:
        if analysis.safety_category is not SafetyCategory.NORMAL or not child_text:
            return None
        span = re.sub(r"\s+", " ", analysis.grounding_span).strip()
        source = re.sub(r"\s+", " ", child_text).strip()
        if not span or len(span) > 30 or span not in source:
            return None
        if deterministic_safety(span) is not SafetyCategory.NORMAL:
            return None
        if not extract_numeric_values(span).issubset(allowed_numbers):
            return None
        return span

    @staticmethod
    def _safe_social_grounding_span(
        child_text: str | None,
        analysis: UtteranceAnalysis,
        *,
        allowed_numbers: set[str],
    ) -> str | None:
        """Return an exact safe social quote without treating it as learning proof."""

        if (
            analysis.safety_category
            not in {
                SafetyCategory.NORMAL,
                SafetyCategory.PLAYFUL_OFFTOPIC,
            }
            or not child_text
        ):
            return None
        span = re.sub(r"\s+", " ", analysis.social_grounding_span).strip()
        source = re.sub(r"\s+", " ", child_text).strip()
        if not span or len(span) > 30 or span not in source:
            return None
        if deterministic_safety(span) not in {
            SafetyCategory.NORMAL,
            SafetyCategory.PLAYFUL_OFFTOPIC,
        }:
            return None
        if not extract_numeric_values(span).issubset(allowed_numbers):
            return None
        return span

    @staticmethod
    def _social_bridge_fallback(
        intent: InteractionIntent,
        repeat_count: int,
    ) -> str:
        """Reviewed bridge copy used only if the bounded speaker path fails."""

        repeated = repeat_count >= 2
        if intent is InteractionIntent.AUTHENTICITY_CHALLENGE:
            return (
                "나 아직 진짜 헷갈려... 이것만 알려주면 안 돼?"
                if repeated
                else "나 진짜 몰라서 물어본 거야... 조금만 알려주면 안 돼?"
            )
        if intent is InteractionIntent.PLAYFUL_TEASE:
            return (
                "응, 들었어. 나는 아직 이게 헷갈려... 알려줄래?"
                if repeated
                else "장난치는 거지? 나는 아직 이게 헷갈려... 알려줄래?"
            )
        if intent is InteractionIntent.FRUSTRATION:
            return "내가 자꾸 물어서 답답했구나... 이것만 알려줄래?"
        if intent is InteractionIntent.REFUSAL:
            return "지금은 말하기 싫구나... 이것만 같이 볼까?"
        return "그 말도 들었어. 나는 아직 이게 궁금해... 알려줄래?"

    @classmethod
    def _safe_grounding_for_step(
        cls,
        task: TaskDefinition,
        state: SessionState,
        step: StepDefinition,
        child_text: str | None,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        fact_sentences = [
            task.slots[slot].fact_sentence for slot in state.verified_slots if slot in task.slots
        ]
        visible_facts = [str(value) for value in task.visible_facts.values()]
        allowed_numbers = extract_numeric_values(
            " ".join([step.prompt, *fact_sentences, *visible_facts])
        )
        return cls._safe_grounding_span(
            child_text,
            analysis,
            allowed_numbers=allowed_numbers,
        )

    @staticmethod
    def _speaker_guard(
        task: TaskDefinition,
        state: SessionState,
        context: SpeakerContext,
    ) -> SpeakerGuardContract:
        del state  # The explicit parameter documents that the guard is task-state scoped.
        question_compact = re.sub(r"[\s\W_]", "", context.required_question or "").lower()
        forbidden: list[str] = []
        for slot_id in context.required_slot_ids:
            slot = task.slots[slot_id]
            forms = [
                str(slot.expected),
                *slot.aliases,
                *(str(value) for value in slot.accepted_values),
            ]
            for form in forms:
                compact = re.sub(r"[\s\W_]", "", form).lower()
                if len(compact) < 2 or compact in question_compact:
                    continue
                forbidden.append(form)
        return SpeakerGuardContract(
            forbidden_answer_forms=list(dict.fromkeys(forbidden)),
            child_expression_source=(
                context.child_expression if context.child_expression_mode == "quote_safe" else None
            ),
        )

    def _turn_contract(
        self,
        state: SessionState,
        task: TaskDefinition,
        *,
        text: str,
        input_contract: InputContract,
        visual: VisualContract,
        help_card: HelpCardContract | None,
        mood: str,
        note: NoteUpdate | None = None,
    ) -> TurnContract:
        turn_id = state.current_turn_id or new_id("turn")
        pedagogy = (
            PedagogySnapshot(
                expression_level=state.expression_level,
                hint_level=state.hint_level,
                subgoal_id=state.subgoal_id,
                verified_slots=state.verified_slots,
                bottleneck=None,
            )
            if self.show_internal_pedagogy
            else None
        )
        return TurnContract(
            turn_id=turn_id,
            scene=state.scene,
            scenario_id=state.scenario_id,
            task_id=task.id,
            stage_id=task.stage_id,
            task_index=state.task_index,
            mormi=MormiContract(text=self._fit_50(text), mood=mood),  # type: ignore[arg-type]
            input=input_contract,
            visual=visual,
            help_card=help_card,
            note_update=note,
            status=state.status,
            state_version=state.state_version,
            completion=(
                CompletionContract(
                    outcome=state.completion_outcome,
                    teach_reward_eligible=state.teach_reward_eligible,
                    verified_facts=self._completion_facts(state),
                )
                if state.status is SessionStatus.COMPLETED and state.completion_outcome is not None
                else None
            ),
            pedagogy=pedagogy,
            dictionary_ref=(
                dictionary_reference(state.dictionary_snapshots[task.id])
                if task.id in state.dictionary_snapshots
                else None
            ),
        )

    @staticmethod
    def _completion_facts(
        state: SessionState,
    ) -> dict[str, str | int | float | bool]:
        """Return only deterministic curriculum facts for backend sync.

        Raw child text and LLM-authored note candidates never enter this field.
        Multi-task cafe scenarios keep the selected menu in ``scenario_data``
        while the final task owns the current verified slots, so both sources
        are normalized into one small machine contract.
        """

        facts = dict(state.verified_slots)
        child_menu = facts.get("child_menu") or state.scenario_data.get("child_menu_id")
        if isinstance(child_menu, str) and child_menu:
            facts["child_menu_id"] = child_menu
        return facts

    @staticmethod
    def _deterministic_analysis(
        state: SessionState,
        task: TaskDefinition,
        response: ChildResponse,
    ) -> UtteranceAnalysis:
        step = task.active_step(
            state.expression_level,
            state.verified_slots,
            entry_active=state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE,
            targeted_followup=(state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP),
        )
        candidate_values: dict[str, object] = {}
        selected_labels: list[str] = []
        if response.type in {ResponseType.CHOICE, ResponseType.FILL}:
            choices_by_id = {choice.id: choice for choice in step.input.choices}
            if len(response.choice_ids) != 1 or any(
                choice_id not in choices_by_id for choice_id in response.choice_ids
            ):
                return UtteranceAnalysis(
                    safety_category=SafetyCategory.NORMAL,
                    response_category=ResponseCategory.RECOGNITION_OR_INPUT_ERROR,
                    difficulty_class=DifficultyClass.INPUT,
                    bottleneck="structured_input",
                    confidence=1,
                )
            for choice_id in response.choice_ids:
                selected_labels.append(choices_by_id[choice_id].label)
                candidate_values.update(step.choice_effects.get(choice_id, {}))
        elif response.type in {ResponseType.COUNT, ResponseType.EQUATION, ResponseType.ACTION}:
            candidate_values.update(response.values)

        claims = []
        interpreted_slots = {*step.target_slots, *step.optional_slots}
        for slot_id, value in candidate_values.items():
            slot = task.slots.get(slot_id)
            if slot and slot_id in interpreted_slots:
                claims.append(
                    {
                        "slot_id": slot_id,
                        "value": value,
                        "factual": slot.accepts(value),
                        "evidence_span": str(value),
                    }
                )
        if not claims and selected_labels:
            # Keep a structured, non-factual trace of which requested concept
            # was missed without ever turning the child-facing label into a
            # verified slot value.
            claims.extend(
                {
                    "slot_id": slot_id,
                    "value": selected_labels[0],
                    "factual": False,
                    "evidence_span": selected_labels[0],
                }
                for slot_id in step.target_slots
            )
        factual_count = sum(1 for claim in claims if claim["factual"])
        all_factual = bool(claims) and factual_count == len(claims)
        factual_slots = {claim["slot_id"] for claim in claims if claim["factual"]}
        if all_factual and set(step.target_slots).issubset(factual_slots):
            category = ResponseCategory.CORRECT_FULL
            difficulty = DifficultyClass.UNKNOWN
        elif factual_count:
            category = ResponseCategory.CORRECT_PARTIAL
            difficulty = DifficultyClass.UNKNOWN
        else:
            category = ResponseCategory.CONCEPTUAL_ERROR
            difficulty = DifficultyClass.CONCEPT
        return UtteranceAnalysis.model_validate(
            {
                "safety_category": "normal",
                "response_category": category,
                "difficulty_class": difficulty,
                "claims": claims,
                "misconception_tag": (
                    task.misconception_tags[0]
                    if category is ResponseCategory.CONCEPTUAL_ERROR and task.misconception_tags
                    else None
                ),
                "bottleneck": "concept" if difficulty is DifficultyClass.CONCEPT else "unknown",
                "confidence": 1,
            }
        )

    @staticmethod
    def _help_card(task: TaskDefinition, level: HintLevel) -> HelpCardContract | None:
        if level is HintLevel.H0:
            return None
        hint = task.hints[level]
        return HelpCardContract(
            level=level,
            body=hint.body,
            visual_type=hint.visual_type,
            visual_data=hint.visual_data,
        )

    @staticmethod
    def _help_card_event(before: HintLevel, after: HintLevel) -> HelpCardEvent:
        if after is HintLevel.H3:
            return HelpCardEvent.JOINT
        if before is HintLevel.H0 and after is not HintLevel.H0:
            return HelpCardEvent.OPENED
        if before is not after:
            return HelpCardEvent.STRENGTHENED
        return HelpCardEvent.NONE

    @classmethod
    def _support_transition_fallback(
        cls,
        trigger: SupportTrigger,
        event: HelpCardEvent,
        grounding: str | None = None,
    ) -> str:
        """Build a coherent reviewed line for each support reason."""

        if event is HelpCardEvent.JOINT:
            return "이제 도움 카드대로 나랑 같이 해보자."
        if trigger is SupportTrigger.RELATED_VAGUE:
            if event is HelpCardEvent.NONE:
                if grounding:
                    return cls._fit_50(
                        f"그런데 ‘{grounding}’가 어떻게 하는 건지 모르겠어... 조금만 더 알려줄래?"
                    )
                return "어떻게 하는 건지 아직 헷갈려... 조금만 더 알려줄래?"
            if grounding:
                return cls._fit_50(f"‘{grounding}’가 아직 헷갈려... 카드도 보고 조금 더 알려줄래?")
            return "카드에 도움이 나왔어. 이걸 보고 조금 더 알려줄래?"
        if trigger is SupportTrigger.EXPLICIT_HELP_REQUEST:
            if event is HelpCardEvent.OPENED:
                return "오, 도움 카드가 나왔어! 이걸 보고 다시 알려줄래?"
            if event is HelpCardEvent.STRENGTHENED:
                return "카드에 다른 도움도 나왔어. 이걸로 다시 알려줄래?"
            return "이걸 보고 나한테 다시 알려줄래?"
        if trigger in {
            SupportTrigger.CONCEPTUAL_CONFLICT,
            SupportTrigger.REPEATED_CONCEPTUAL_CONFLICT,
        }:
            if grounding:
                return cls._fit_50(
                    f"‘{grounding}’라는 거야? 나는 아직 헷갈려... 카드를 보고 다시 알려줄래?"
                )
            if event is HelpCardEvent.STRENGTHENED:
                return "나는 아직 헷갈려... 카드에 나온 다른 도움도 같이 볼까?"
            return "나는 아직 헷갈려... 도움 카드를 보고 다시 알려줄래?"
        return "어떻게 하는 건지 아직 헷갈려... 조금만 더 알려줄래?"

    @staticmethod
    def _visual_for(task: TaskDefinition, level: HintLevel) -> VisualContract:
        if level in {HintLevel.H2, HintLevel.H3}:
            hint = task.hints[level]
            if hint.visual_type:
                return VisualContract(type=hint.visual_type, data=hint.visual_data)
        return task.base_visual

    @staticmethod
    def _step_contract_signature(step: Any) -> tuple[str, str]:
        """Describe what the child can actually see and do on a step.

        Expression levels are internal metadata.  Two different levels can
        accidentally resolve to the same prompt and the same text box, which
        feels like an infinite loop to the child.  The serialized input keeps
        choices, fill text and target slots in this visible-contract check.
        """

        prompt = re.sub(r"\s+", "", step.prompt)
        return prompt, step.input.model_dump_json(exclude_none=True)

    @classmethod
    def _lower_until_visible_contract_changes(
        cls,
        state: SessionState,
        task: TaskDefinition,
        previous_step: Any,
    ) -> None:
        """Lower support until the next child interaction is visibly new.

        A targeted L4 follow-up already uses an L3-shaped text question while
        preserving the child's L4 credit.  Merely changing the stored level
        from L4 to L3 would therefore repeat the exact same interaction.  In
        that case continue to L2, where choices provide real additional help.
        """

        previous_signature = cls._step_contract_signature(previous_step)
        if state.expression_level is not ExpressionLevel.L0:
            state.expression_level = state.expression_level.lower()
        while state.expression_level is not ExpressionLevel.L0:
            candidate = task.step_for(state.expression_level, state.verified_slots)
            if cls._step_contract_signature(candidate) != previous_signature:
                return
            state.expression_level = state.expression_level.lower()

    @staticmethod
    def _partial_fallback(
        task: TaskDefinition,
        newly_verified: Mapping[str, object],
        state: SessionState,
    ) -> str:
        step = task.active_step(
            state.expression_level,
            state.verified_slots,
            entry_active=state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE,
            targeted_followup=(state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP),
        )
        acknowledgement = ConversationEngine._younger_sibling_acknowledgement(
            task,
            newly_verified,
        )
        if (
            task.skill_id == "number-count"
            and "answer" in newly_verified
            and "tracking" not in state.verified_slots
        ):
            if "answer" in newly_verified:
                acknowledgement = "아, 세 개구나!"
            return ConversationEngine._fit_50(f"{acknowledgement} {step.prompt}")
        return ConversationEngine._preface_question(acknowledgement, step.prompt)

    @staticmethod
    def _younger_sibling_acknowledgement(
        task: TaskDefinition,
        newly_verified: Mapping[str, object],
    ) -> str:
        """Acknowledge only reviewed facts in a warm younger-sibling voice."""

        if task.skill_id == "number-count" and "answer" in newly_verified:
            return "아, 세 개구나!"
        if task.skill_id == "number-compare" and "answer" in newly_verified:
            return "아, 오른쪽이 더 많구나!"
        if "result" in newly_verified and isinstance(newly_verified["result"], int):
            return f"아, {newly_verified['result']:,}원이구나!"
        return "아, 그렇구나!"

    @staticmethod
    def _ensure_required_question(fallback: str, question: str) -> str:
        normalized_fallback = re.sub(r"\s+", "", fallback)
        normalized_question = re.sub(r"\s+", "", question)
        if normalized_question in normalized_fallback:
            return ConversationEngine._fit_50(fallback)
        return ConversationEngine._preface_question(fallback, question)

    @staticmethod
    def _smooth_ladder_fallback(state: SessionState, task: TaskDefinition) -> str:
        step = task.step_for(state.expression_level, state.verified_slots)
        if state.expression_level is ExpressionLevel.L3:
            return ConversationEngine._preface_question("내가 한꺼번에 많이 물어봤네.", step.prompt)
        if state.expression_level is ExpressionLevel.L2:
            return ConversationEngine._preface_question("말로만 들으려니 헷갈려.", step.prompt)
        if state.expression_level is ExpressionLevel.L1:
            return ConversationEngine._preface_question("어디부터 볼지 몰랐네.", step.prompt)
        return ConversationEngine._fit_50("도움 카드 순서대로 나와 같이 해볼까?")

    @staticmethod
    def _success_then_question(fact: str, question: str) -> str:
        return ConversationEngine._preface_question("네가 알려줘서 알겠어.", question)

    @staticmethod
    def _response_as_text(response: ChildResponse) -> str:
        if response.choice_ids:
            return " ".join(response.choice_ids)
        return str(response.values)

    @staticmethod
    def _fit_50(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized if len(normalized) <= 50 else normalized[:49].rstrip() + "…"

    @staticmethod
    def _preface_question(preface: str, question: str) -> str:
        combined = re.sub(r"\s+", " ", f"{preface} {question}").strip()
        return combined if len(combined) <= 50 else ConversationEngine._fit_50(question)

    @staticmethod
    def _safe_child_note_text(
        task: TaskDefinition,
        child_text: str,
    ) -> str | None:
        text = re.sub(r"\s+", " ", child_text).strip()
        if not text or "?" in text or len(text) > 120:
            return None
        allowed_numbers = {
            number.replace(",", "")
            for value in task.visible_facts.values()
            for number in re.findall(r"\d[\d,]*", str(value))
        }
        allowed_numbers.update(
            number.replace(",", "")
            for slot in task.slots.values()
            for number in re.findall(r"\d[\d,]*", str(slot.expected))
        )
        if any(
            number.replace(",", "") not in allowed_numbers
            for number in re.findall(r"\d[\d,]*", text)
        ):
            return None
        if re.search(r"(틀렸|맞았|정답|오답)", text):
            return None
        compact = re.sub(r"[\s,.!?]", "", text)
        if re.fullmatch(
            r"(?:잘)?모르겠(?:어|다)?|몰라|"
            r"(?:왼쪽|오른쪽)(?:이|가)?(?:더)?"
            r"(?:커|커요|많아|많아요|많잖아|작아|적어|이야|야)?|"
            r"(?:\d[\d,]*|(?:영|일|이|삼|사|오|육|칠|팔|구|십|백|천|만|"
            r"한|두|세|네|다섯|여섯|일곱|여덟|아홉)+)"
            r"(?:원|개|명)?(?:이야|야|잖아|입니다)?",
            compact,
        ):
            return None
        return text

    @classmethod
    def _claim_evidence_text(
        cls,
        task: TaskDefinition,
        child_text: str,
        evidence_span: str,
    ) -> str | None:
        """Return only wording that is visibly present in the child turn.

        The classifier may locate the relevant clause, but it may not rewrite
        that clause for the note.  Exact-substring checking keeps an unrelated
        or incorrect clause out and prevents a model-authored strategy from
        being attributed to the child.
        """

        span = re.sub(r"\s+", " ", evidence_span).strip()
        source = re.sub(r"\s+", " ", child_text).strip()
        if not span or span not in source:
            return None
        return cls._safe_child_note_text(task, span)

    @classmethod
    def _filter_text_explanation_claims(
        cls,
        task: TaskDefinition,
        child_text: str,
        analysis: UtteranceAnalysis,
        verified: Mapping[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        """Fail closed when a method/reason claim has no explanatory source.

        This is deliberately narrower than general answer verification.  A
        result such as ``600원이야`` can still be remembered as an answer,
        while ``오른쪽이 더 많아`` cannot masquerade as the reason for a
        comparison and prematurely close the teaching task.
        """

        filtered = dict(verified)
        for slot_id in task.text_explanation_slots:
            if slot_id not in filtered:
                continue
            supported = any(
                claim.factual
                and claim.slot_id == slot_id
                and cls._claim_evidence_text(task, child_text, claim.evidence_span)
                for claim in analysis.claims
            )
            if not supported:
                filtered.pop(slot_id, None)
        return filtered

    @classmethod
    def _record_note_provenance(
        cls,
        state: SessionState,
        task: TaskDefinition,
        response: ChildResponse,
        analysis: UtteranceAnalysis,
        newly_verified: Mapping[str, object],
    ) -> None:
        note_slots = set(task.effective_note_slots)
        verified_note_slots = note_slots.intersection(newly_verified)
        if not verified_note_slots or task.note_policy == "none":
            return

        if response.type is ResponseType.TEXT and response.text:
            factual_claims = {
                claim.slot_id: claim
                for claim in analysis.claims
                if claim.factual and claim.slot_id in verified_note_slots
            }
            for slot_id, claim in factual_claims.items():
                # No inferred fallback: missing or rewritten evidence fails
                # closed.  Task completion may still proceed, but it cannot
                # produce a direct-attribution note without an exact source.
                evidence = cls._claim_evidence_text(
                    task,
                    response.text,
                    claim.evidence_span,
                )
                if evidence:
                    state.child_note_evidence[slot_id] = evidence
            return

        supported = set(state.supported_note_slots)
        supported.update(verified_note_slots)
        state.supported_note_slots = sorted(supported)

    @staticmethod
    def _note_from_provenance(
        state: SessionState,
        task: TaskDefinition,
    ) -> tuple[NoteUpdate | None, bool]:
        if task.note_policy == "none":
            return None, True

        note_slots = task.effective_note_slots
        direct_slots = set(state.child_note_evidence)
        supported_slots = set(state.supported_note_slots)
        all_direct = set(note_slots).issubset(direct_slots)

        if all_direct:
            # Keep the child's exact evidence, but add only a reviewed,
            # strategy-neutral context.  This turns fragments such as
            # "손가락 하나씩 펴면서" into a self-contained note without
            # replacing them with the curriculum's preferred method.
            pieces = list(
                dict.fromkeys(state.child_note_evidence[slot_id] for slot_id in note_slots)
            )
            evidence = " ".join(piece.rstrip(". ") for piece in pieces)
            text = f"{task.note_context}에 대해 “{evidence}”라고 배웠어."
            if task.note_direct_conclusion:
                text = f"{text} {task.note_direct_conclusion}"
            return (
                NoteUpdate(
                    skill_id=task.skill_id,
                    text=text,
                    attribution=NoteAttribution.CHILD,
                    evidence=NoteEvidence.DIRECT_EXPLANATION,
                    attribution_label="아이가 알려줌",
                ),
                True,
            )

        if set(note_slots).issubset(direct_slots | supported_slots):
            return (
                NoteUpdate(
                    skill_id=task.skill_id,
                    text=task.coauthored_note,
                    attribution=NoteAttribution.COAUTHORED,
                    evidence=NoteEvidence.SUPPORTED_COMPLETION,
                    attribution_label="아이와 같이 공부함",
                ),
                False,
            )

        # Completion alone is not evidence for a note.  If neither safe child
        # wording nor a reviewed structured interaction covers the note slots,
        # emitting the curriculum line would attribute knowledge the child
        # never expressed.
        return None, False


def max_hint(left: HintLevel, right: HintLevel) -> HintLevel:
    order = list(HintLevel)
    return order[max(order.index(left), order.index(right))]


def select_start_level(profile: LearnerProfile, skill_id: str) -> ExpressionLevel:
    skill = profile.skills.get(skill_id)
    if skill:
        return skill.highest_stable_expression_level
    return ExpressionLevel.L4


def update_skill_profile(
    profile: LearnerProfile,
    state: SessionState,
    task: TaskDefinition,
) -> LearnerProfile:
    current = profile.skills.get(task.skill_id) or SkillProfile(skill_id=task.skill_id)
    independent = state.task_max_hint is HintLevel.H0
    if independent:
        current.h0_success_streak += 1
        current.concept_mastery = min(1, current.concept_mastery + 0.08)
    else:
        current.h0_success_streak = 0
        current.concept_mastery = min(1, current.concept_mastery + 0.02)
    current.recent_max_hint = state.task_max_hint
    current.last_bottleneck = "unknown"
    if independent and state.expression_level.rank >= current.highest_stable_expression_level.rank:
        current.expression_independence = min(1, current.expression_independence + 0.06)
        if current.h0_success_streak >= 2:
            current.highest_stable_expression_level = (
                current.highest_stable_expression_level.higher()
            )
    profile.skills[task.skill_id] = current
    return profile
