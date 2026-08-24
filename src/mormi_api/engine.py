from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .content import PARK_SCENARIO_IDS, SlotDefinition, StepDefinition, TaskDefinition, get_task
from .dictionary_models import dictionary_reference
from .llm import (
    ClaudeGateway,
    extract_numeric_values,
    validate_note_contextualization,
    validate_speaker_output,
)
from .schemas import (
    CafeMenuItem,
    ChildResponse,
    CompletionContract,
    CompletionOutcome,
    DialogueHistoryTurn,
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
    NoteContextualizationContext,
    NoteContextualizationOutput,
    NoteEvidence,
    NoteUpdate,
    ParkSessionContext,
    PedagogicalDecision,
    PedagogySnapshot,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SessionState,
    SessionStatus,
    SkillProfile,
    SlotClaim,
    SpeakerArithmeticClaim,
    SpeakerContext,
    SpeakerGuardContract,
    SpeakerOutput,
    SpeakerQuantity,
    SpeakerQuestionIntent,
    SpeakerRuntimeAudit,
    SpeakerVerificationPolicy,
    SupportTrigger,
    TaskAnchorCompletedItem,
    TaskAnchorContract,
    TaskRelation,
    TurnContract,
    UnderstandingRoute,
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
    classifier_response_category: ResponseCategory
    turn: TurnContract
    runtime: SpeakerRuntimeAudit
    accepted_claims: dict[str, str | int | float | bool]


class ConversationGraphState(TypedDict, total=False):
    session: dict[str, Any]
    response: dict[str, Any]
    previous_question: str
    recent_dialogue: list[dict[str, Any]]
    analysis: dict[str, Any]
    classifier_response_category: str
    understanding_route: str
    adjudicator_used: bool
    decision: dict[str, Any]
    speaker_output: dict[str, Any]
    speaker_text: str
    note_output: dict[str, Any]
    runtime: dict[str, Any]
    turn: dict[str, Any]


class ConversationEngine:
    def __init__(
        self,
        gateway: ClaudeGateway,
        *,
        show_internal_pedagogy: bool = False,
        speaker_timeout_seconds: float = 10.0,
        bridge_timeout_seconds: float = 4.0,
    ) -> None:
        self.gateway = gateway
        self.show_internal_pedagogy = show_internal_pedagogy
        self.speaker_timeout_seconds = speaker_timeout_seconds
        self.bridge_timeout_seconds = bridge_timeout_seconds
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(ConversationGraphState)
        # LangGraph's current generic overloads do not accept async bound methods
        # cleanly in mypy even though they are supported at runtime.
        builder.add_node("understand", self._understand_node)  # type: ignore[call-overload]
        builder.add_node("orchestrate", self._orchestrate_node)  # type: ignore[call-overload]
        builder.add_node("bridge_speak", self._bridge_speak_node)  # type: ignore[call-overload]
        builder.add_node("speak", self._speak_node)  # type: ignore[call-overload]
        builder.add_node(  # type: ignore[call-overload]
            "validate_and_compose",
            self._validate_and_compose_node,
        )
        builder.add_edge(START, "understand")
        builder.add_edge("understand", "orchestrate")
        builder.add_conditional_edges(
            "orchestrate",
            self._route_after_orchestration,
            {
                "bridge_speak": "bridge_speak",
                "speak": "speak",
            },
        )
        builder.add_edge("bridge_speak", "validate_and_compose")
        builder.add_edge("speak", "validate_and_compose")
        builder.add_edge("validate_and_compose", END)
        # Canonical state is committed atomically to the access-controlled application
        # database after this per-turn graph succeeds. We deliberately do not
        # checkpoint raw child text inside LangGraph.
        return builder.compile()

    async def run_turn(
        self,
        state: SessionState,
        response: ChildResponse,
        previous_question: str,
        recent_dialogue: list[DialogueHistoryTurn] | None = None,
    ) -> tuple[SessionState, UtteranceAnalysis, TurnContract]:
        result: EngineTurnResult | None = None
        async for event in self.run_turn_stream(
            state,
            response,
            previous_question,
            recent_dialogue=recent_dialogue,
        ):
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
        *,
        recent_dialogue: list[DialogueHistoryTurn] | None = None,
    ) -> AsyncIterator[EngineProgress | EngineTurnResult]:
        """Expose safe graph milestones without leaking unvalidated model text."""

        normalized_state = state.model_copy(deep=True)
        self._normalize_expression_ladder(normalized_state)
        self._normalize_terminal_support(normalized_state)
        final_values: dict[str, Any] | None = None
        node_stages: dict[str, EngineProgress] = {
            "understand": EngineProgress("understanding"),
            "orchestrate": EngineProgress("planning"),
            "bridge_speak": EngineProgress("speaking"),
            "speak": EngineProgress("speaking"),
            "validate_and_compose": EngineProgress("validating"),
        }
        async for mode, payload in self.graph.astream(
            {
                "session": normalized_state.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "previous_question": previous_question,
                "recent_dialogue": [
                    turn.model_dump(mode="json") for turn in (recent_dialogue or [])[-6:]
                ],
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
            classifier_response_category=ResponseCategory(
                final_values["classifier_response_category"]
            ),
            turn=TurnContract.model_validate(final_values["turn"]),
            runtime=SpeakerRuntimeAudit.model_validate(final_values["runtime"]),
            accepted_claims=decision.accepted_claims,
        )

    def initial_turn(self, state: SessionState) -> TurnContract:
        task = get_task(state.current_task_id, state.scenario_data)
        self._normalize_expression_ladder(state)
        self._normalize_terminal_support(state)
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
            visual=self._visual_for(task, state.hint_level),
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
            return self._understanding_result(analysis, UnderstandingRoute.NORMAL)

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
            route = (
                UnderstandingRoute.BRIDGE
                if deterministic_category is SafetyCategory.PLAYFUL_OFFTOPIC
                else UnderstandingRoute.NORMAL
            )
            return self._understanding_result(analysis, route)

        if response.type is ResponseType.TEXT:
            recent_dialogue = [
                DialogueHistoryTurn.model_validate(turn)
                for turn in graph_state.get("recent_dialogue", [])
            ]
            analysis = await self.gateway.classify(
                state=state,
                task=task,
                previous_question=previous_question,
                response=response,
                dialogue_history=recent_dialogue,
            )
            # Deterministic safety always wins; model safety may make the result
            # stricter but can never downgrade an explicit local match.
            if analysis.safety_category is SafetyCategory.UNKNOWN:
                analysis.safety_category = SafetyCategory.NORMAL
            analysis = self._enforce_claim_provenance(task, text, analysis)
            route = self._select_understanding_route(analysis)
            return self._understanding_result(analysis, route)

        analysis = self._deterministic_analysis(state, task, response)
        return self._understanding_result(analysis, UnderstandingRoute.NORMAL)

    @staticmethod
    def _understanding_result(
        analysis: UtteranceAnalysis,
        route: UnderstandingRoute,
    ) -> dict[str, Any]:
        """Keep the understanding model's label before code reconciliation."""

        return {
            "analysis": analysis.model_dump(mode="json"),
            "classifier_response_category": analysis.response_category.value,
            "understanding_route": route.value,
            "adjudicator_used": False,
        }

    @staticmethod
    def _is_reviewed_conversation_only(analysis: UtteranceAnalysis) -> bool:
        """Return true only when no child-grounded learning evidence remains."""

        return (
            analysis.conversation_only
            and not analysis.claims
            and not analysis.arithmetic_claims
            and analysis.response_category is not ResponseCategory.HELP_REQUEST
        )

    def _select_understanding_route(
        self,
        analysis: UtteranceAnalysis,
    ) -> UnderstandingRoute:
        """Use the separate cheap bridge only for safe non-learning turns."""

        if analysis.safety_category is not SafetyCategory.NORMAL:
            return (
                UnderstandingRoute.BRIDGE
                if analysis.safety_category is SafetyCategory.PLAYFUL_OFFTOPIC
                else UnderstandingRoute.NORMAL
            )
        # Do not require an unexpected safe utterance to fit a closed intent
        # taxonomy. The primary understanding model establishes whether this
        # turn contains child-grounded learning evidence; a separate cheap
        # bridge model writes child-facing copy only after that evidence is
        # frozen. Contradictory outputs that contain a grounded learning claim
        # stay on the normal pedagogical path.
        if self._is_reviewed_conversation_only(analysis):
            return UnderstandingRoute.BRIDGE
        return UnderstandingRoute.NORMAL

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
        recent_dialogue = [
            DialogueHistoryTurn.model_validate(turn)
            for turn in graph_state.get("recent_dialogue", [])[-3:]
        ]
        speaker_context = decision.speaker_context.model_copy(
            update={
                "expression_level": decision.state.expression_level,
                "hint_level": decision.state.hint_level,
                "unresolved_focus": dict(decision.speaker_context.required_slot_descriptions),
                "recent_dialogue": recent_dialogue,
            }
        )
        decision = decision.model_copy(update={"speaker_context": speaker_context})
        return {
            "session": decision.state.model_dump(mode="json"),
            # ``_decide`` may reconcile correctness from validated slot claims.
            # Put that effective analysis back into graph state while keeping
            # the classifier's original label in its own audit field.
            "analysis": analysis.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }

    @staticmethod
    def _route_after_orchestration(graph_state: ConversationGraphState) -> str:
        """Use the cheap bridge speaker only for a safe social bridge act.

        The understanding route alone cannot bypass the pedagogical decision:
        the orchestrator must first preserve the unresolved learning focus and
        choose the state-preserving bridge dialogue act.
        """

        decision = PedagogicalDecision.model_validate(graph_state["decision"])
        if graph_state.get(
            "understanding_route"
        ) == UnderstandingRoute.BRIDGE.value and decision.dialogue_act in {
            "acknowledge_non_learning_and_reask",
            "acknowledge_task_comment_and_reask",
            "acknowledge_meta_and_reask",
            "brief_ack_and_redirect",
        }:
            return "bridge_speak"
        return "speak"

    async def _bridge_speak_node(
        self,
        graph_state: ConversationGraphState,
    ) -> dict[str, Any]:
        """Generate one cheap social bridge after learning evidence is frozen.

        The primary understanding model never writes child-facing copy.  The
        bridge receives only the already reviewed speaker contract, and the
        same structural/provenance guard used for the main speaker is applied before
        its sentence can reach a child.
        """

        decision = PedagogicalDecision.model_validate(graph_state["decision"])
        state = SessionState.model_validate(graph_state["session"])
        context = decision.speaker_context
        task = get_task(state.current_task_id, state.scenario_data)
        started_at = time.perf_counter()
        source: Literal[
            "deterministic_validation_fallback",
            "generation_fallback",
        ]
        try:
            async with asyncio.timeout(self.bridge_timeout_seconds):
                output = await self.gateway.bridge_speak(context)
            guard = self._speaker_guard(task, decision.state, context)
            validated = validate_speaker_output(output, context, guard)
            if validated is not None:
                output.text = validated
                runtime = SpeakerRuntimeAudit(
                    dialogue_act=decision.dialogue_act,
                    speaker_source="bridge_llm",
                    verifier_status="not_required",
                    speaker_latency_ms=self._elapsed_ms(started_at),
                )
                return {
                    "speaker_output": output.model_dump(mode="json"),
                    "runtime": runtime.model_dump(mode="json"),
                }
            reason = "bridge_contract_rejected"
            source = "deterministic_validation_fallback"
            logger.warning("speaker_fallback stage=bridge_validation")
        except Exception as error:
            reason = type(error).__name__
            source = "generation_fallback"
            logger.warning(
                "speaker_fallback stage=bridge_generation error_type=%s",
                reason,
            )
        runtime = SpeakerRuntimeAudit(
            dialogue_act=decision.dialogue_act,
            speaker_source=source,
            verifier_status="not_required",
            fallback_reason=reason,
            speaker_latency_ms=self._elapsed_ms(started_at),
        )
        return {
            "speaker_text": context.fallback_text,
            "runtime": runtime.model_dump(mode="json"),
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
            result: dict[str, Any] = {
                "speaker_text": context.fallback_text,
                "runtime": runtime.model_dump(mode="json"),
            }
            note_context = decision.note_contextualization
            if note_context is not None:
                try:
                    async with asyncio.timeout(min(self.speaker_timeout_seconds, 4.0)):
                        note_output = await self.gateway.contextualize_note(note_context)
                    result["note_output"] = note_output.model_dump(mode="json")
                except Exception as error:
                    # The reviewed deterministic note is always safe to show.
                    # A note-language failure must never fail or delay the
                    # completed learning turn beyond the bounded timeout.
                    logger.warning(
                        "note_contextualization_fallback error_type=%s",
                        type(error).__name__,
                    )
            return result
        started_at = time.perf_counter()
        try:
            async with asyncio.timeout(self.speaker_timeout_seconds):
                output = await self.gateway.speak(context)
            runtime = SpeakerRuntimeAudit(
                dialogue_act=decision.dialogue_act,
                speaker_source="llm",
                verifier_status="not_required",
                speaker_latency_ms=self._elapsed_ms(started_at),
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
                speaker_latency_ms=self._elapsed_ms(started_at),
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
        runtime = runtime.model_copy(
            update={
                "understanding_route": UnderstandingRoute(
                    graph_state.get(
                        "understanding_route",
                        UnderstandingRoute.NORMAL.value,
                    )
                ),
                "adjudicator_used": bool(graph_state.get("adjudicator_used", False)),
            }
        )
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
        text = text or decision.speaker_context.fallback_text
        note_update = decision.note_update
        note_context = decision.note_contextualization
        if note_update is not None and note_context is not None and graph_state.get("note_output"):
            note_output = NoteContextualizationOutput.model_validate(graph_state["note_output"])
            contextualized = validate_note_contextualization(note_output, note_context)
            if contextualized:
                note_update = note_update.model_copy(update={"text": contextualized})
            else:
                logger.info(
                    "note_contextualization_fallback reason=contract_rejected skill_id=%s",
                    note_context.skill_id,
                )
        turn = self._turn_contract(
            decision.state,
            task,
            text=text,
            input_contract=decision.input,
            visual=decision.visual,
            help_card=decision.help_card,
            mood=decision.mood,
            note=note_update,
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
        # ``_orchestrate_node`` has already established the provenance boundary
        # for open text.  Keep this exact analysis instance here: arithmetic
        # reconciliation below intentionally mutates its effective category,
        # and that corrected value must be written back to LangGraph state.
        next_state = state.model_copy(deep=True)
        self._normalize_expression_ladder(next_state)
        self._normalize_terminal_support(next_state)
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

        # Open-set conversational bridge.  The detailed intent enums remain
        # useful telemetry, but an unexpected safe utterance does not have to
        # fit one of them before it can receive a natural response.  This path
        # can never verify a slot, alter L/H, or write a star note.
        if self._is_reviewed_conversation_only(analysis):
            next_state.unrelated_count += 1
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="acknowledge_non_learning_and_reask",
                fallback=self._open_conversation_fallback(next_state.unrelated_count),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                must_reframe=True,
            )

        # Accepting a reviewed wrong guess is concept evidence, never a
        # successful conversational answer.  Conversely, rejecting it is only
        # a stance: it cannot fill answer/reason slots or enter the star note.
        accepted_wrong_entry_guess = (
            entry_active and analysis.entry_stance is EntryStance.ACCEPT_WRONG_GUESS
        )
        if accepted_wrong_entry_guess:
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

        # The model proposes claims and an initial label; code owns the final
        # correctness label.  Validate every safe, task-directed claim against
        # the reviewed slot contracts first, so an inconsistent model label
        # cannot turn a complete answer into ``correct_partial`` or block a
        # valid answer from advancing the session.
        if response.type is ResponseType.TEXT and response.text:
            self._invalidate_unverified_arithmetic_support(
                task,
                response.text,
                analysis,
            )
            self._invalidate_false_structured_arithmetic(
                task,
                response.text,
                analysis,
            )
            self._ground_text_numeric_claims(
                task,
                response.text,
                analysis,
            )
        grounded_claims = task.validated_slot_claims(analysis.claims)
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
        if not accepted_wrong_entry_guess:
            self._reconcile_response_category(
                analysis,
                task,
                interpreted_step,
                accepted_claims,
                grounded_claims,
            )
        successful_categories = {
            ResponseCategory.CORRECT_FULL,
            ResponseCategory.CORRECT_PARTIAL,
            ResponseCategory.SELF_CORRECTION,
        }
        understood_claims = {
            slot_id: value
            for slot_id, value in grounded_claims.items()
            if task.slots[slot_id].equivalent_state_value(
                state.verified_slots.get(slot_id),
                value,
            )
        }
        understood_claims.update(accepted_claims)
        # A repeated fact is still understood, but it is not new learning
        # progress. Treating it as newly verified can select the same unresolved
        # step forever instead of changing the support contract.
        newly_verified = {
            slot_id: value
            for slot_id, value in accepted_claims.items()
            if not task.slots[slot_id].equivalent_state_value(
                next_state.verified_slots.get(slot_id),
                value,
            )
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
                    # The support contract may have changed to L2 even when
                    # an authored L2 prompt accidentally duplicates the L4
                    # question.  In that case, use the reviewed choice-stage
                    # sentence so the child can hear that the response mode
                    # changed.  ``같이`` remains reserved for L0 joint work.
                    next_question = (
                        self._smooth_ladder_fallback(next_state, task)
                        if next_state.expression_level is ExpressionLevel.L2
                        else self._complete_mormi_text(next_step.prompt)
                    )
                    fallback = (
                        self._preface_question(acknowledgement, next_question)
                        if acknowledgement
                        else next_question
                    )
                else:
                    # No mathematical fact was understood, so a canned
                    # acknowledgement would pretend that Mormi learned
                    # something and make the lower-step question sound like a
                    # mechanical retry.  The reviewed step prompt is the
                    # complete safe fallback; the speaker model may still
                    # bridge to it naturally on the normal path.
                    fallback = (
                        self._smooth_ladder_fallback(next_state, task)
                        if next_state.expression_level is ExpressionLevel.L2
                        else self._complete_mormi_text(next_step.prompt)
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
                # The expression contract has just changed.  Requiring the
                # catalog question verbatim would append it to the complete
                # L2 bridge and recreate the robotic duplicate prompt that
                # this branch exists to prevent.
                must_reframe=True,
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
            previous_hint = next_state.hint_level
            self._normalize_terminal_support(next_state)
            joint_mode = next_state.expression_level is ExpressionLevel.L0
            help_event = self._help_card_event(previous_hint, next_state.hint_level)
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="joint_mode" if joint_mode else "reduce_expression_load",
                fallback=(
                    self._support_transition_fallback(
                        SupportTrigger.EXPRESSION_BLOCK,
                        help_event,
                    )
                    if joint_mode
                    else self._smooth_ladder_fallback(next_state, task)
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                support_trigger=SupportTrigger.EXPRESSION_BLOCK,
                help_card_event=help_event,
                must_reframe=not joint_mode,
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
            arithmetic_claims = self._speaker_arithmetic_claims(
                task,
                response.text,
                analysis,
            )
            help_card_visible = next_state.hint_level is not HintLevel.H0
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
                    arithmetic_claims=arithmetic_claims,
                    help_card_visible=help_card_visible,
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

    @staticmethod
    def _reconcile_response_category(
        analysis: UtteranceAnalysis,
        task: TaskDefinition,
        step: StepDefinition,
        accepted_claims: Mapping[str, str | int | float | bool],
        grounded_claims: Mapping[str, str | int | float | bool],
    ) -> None:
        """Derive effective correctness from reviewed claims, not model prose.

        Safety, input failures and explicit help requests describe interaction
        state rather than mathematical completeness, so they remain protected.
        For an assessable learning response, current-turn target coverage is the
        canonical source of ``correct_full`` versus ``correct_partial``.
        """

        protected_categories = {
            ResponseCategory.HELP_REQUEST,
            ResponseCategory.EXPRESSION_BLOCK,
            ResponseCategory.CONCEPTUAL_BLOCK,
            ResponseCategory.NO_RESPONSE,
            ResponseCategory.RECOGNITION_OR_INPUT_ERROR,
        }
        if analysis.response_category in protected_categories or (
            analysis.response_category is ResponseCategory.UNRELATED_RESPONSE
            and analysis.task_relation is TaskRelation.OFF_TOPIC
        ):
            return

        target_slots = set(step.target_slots)
        accepted_slots = set(accepted_claims)
        accepted_targets = target_slots.intersection(accepted_slots)
        accepted_optional = set(step.optional_slots).intersection(accepted_slots)
        rejected_target_claim = any(
            claim.slot_id in target_slots
            and task.slots[claim.slot_id].accepted_claim_value(claim) is None
            for claim in analysis.claims
            if claim.slot_id in task.slots
        )

        if target_slots and target_slots.issubset(accepted_slots):
            analysis.response_category = (
                ResponseCategory.SELF_CORRECTION
                if analysis.response_category is ResponseCategory.SELF_CORRECTION
                else ResponseCategory.CORRECT_FULL
            )
            analysis.difficulty_class = DifficultyClass.UNKNOWN
            analysis.misconception_tag = None
            analysis.bottleneck = "unknown"
            return

        if accepted_targets or accepted_optional:
            analysis.response_category = ResponseCategory.CORRECT_PARTIAL
            # A response can contain a verified part and a conflicting part.
            # Keep concept difficulty for analytics while still preserving and
            # acknowledging the part the child genuinely supplied.
            if rejected_target_claim:
                analysis.difficulty_class = DifficultyClass.CONCEPT
                analysis.bottleneck = "concept"
                if analysis.misconception_tag is None and task.misconception_tags:
                    analysis.misconception_tag = task.misconception_tags[0]
            elif analysis.difficulty_class not in {
                DifficultyClass.CONCEPT,
                DifficultyClass.BOTH,
            }:
                analysis.difficulty_class = DifficultyClass.UNKNOWN
                analysis.bottleneck = "unknown"
            return

        if rejected_target_claim:
            analysis.response_category = ResponseCategory.CONCEPTUAL_ERROR
            analysis.difficulty_class = DifficultyClass.CONCEPT
            analysis.bottleneck = "concept"
            if analysis.misconception_tag is None and task.misconception_tags:
                analysis.misconception_tag = task.misconception_tags[0]
            return

        # The child may naturally repeat an already verified fact while the
        # current turn asks for a missing reason or method.  It is understood
        # but makes no new progress, so keep it as a partial response.  The
        # existing no-progress policy will then lower support instead of
        # repeating the same open question forever.
        if grounded_claims:
            analysis.response_category = ResponseCategory.CORRECT_PARTIAL
            analysis.difficulty_class = DifficultyClass.UNKNOWN
            analysis.misconception_tag = None
            analysis.bottleneck = "unknown"
            return

        if analysis.response_category in {
            ResponseCategory.CORRECT_FULL,
            ResponseCategory.SELF_CORRECTION,
        }:
            analysis.response_category = ResponseCategory.RELATED_VAGUE
            analysis.difficulty_class = DifficultyClass.EXPRESSION
            analysis.misconception_tag = None
            analysis.bottleneck = "expression"

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
        note_contextualization = self._note_contextualization_context(state, task, note)
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
            ).canonical()
            state.task_start_level = state.expression_level
            state.hint_level = HintLevel.H0
            state.task_max_hint = HintLevel.H0
            self._normalize_terminal_support(state)
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
            state.subgoal_id = next_step.id
            fallback = self._task_transition_then_question(
                task,
                contribution,
                next_step.prompt,
            )
            return PedagogicalDecision(
                state=state,
                dialogue_act="task_transition",
                accepted_claims=dict(accepted_claims),
                required_question=next_step.prompt,
                input=next_step.input,
                visual=self._visual_for(next_task, state.hint_level),
                help_card=self._help_card(next_task, state.hint_level),
                note_update=note,
                note_contextualization=note_contextualization,
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
        fallback = self._complete_mormi_text("네가 알려준 방법으로 내가 끝까지 해냈어!")
        return PedagogicalDecision(
            state=state,
            dialogue_act="session_complete",
            accepted_claims=dict(accepted_claims),
            required_question=None,
            input=InputContract(kind=InputKind.NONE),
            visual=VisualContract(type="success", data={"task": task.id}),
            help_card=None,
            note_update=note,
            note_contextualization=note_contextualization,
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
        self._normalize_terminal_support(state)
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
            fallback=self._complete_mormi_text(fallback),
            support_trigger=support_trigger,
            help_card_event=help_card_event,
            help_card_visible=help_card is not None,
            must_reframe=must_reframe,
        )
        if not must_reframe:
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

    @staticmethod
    def _normalize_expression_ladder(state: SessionState) -> None:
        """Read legacy L1 state without ever emitting a new L1 turn.

        L1 and L2 were both structured support in the previous five-level
        ladder.  The canonical four-level contract keeps the public labels
        stable and maps persisted L1 values onto the choice-only L2 level.
        """

        state.expression_level = state.expression_level.canonical()
        if state.task_start_level is not None:
            state.task_start_level = state.task_start_level.canonical()
        state.task_start_levels = {
            task_id: level.canonical() for task_id, level in state.task_start_levels.items()
        }

    @staticmethod
    def _normalize_terminal_support(state: SessionState) -> None:
        """Keep the terminal expression and hint contracts as one safe pair.

        L and H remain independent above the terminal state.  L0, however,
        uses a reviewed joint-performance input whose completion values can
        finish the task, while H3 is the matching fully revealed help card.
        Allowing only one side to reach its terminal value would either expose
        a joint answer without the H3 explanation or reveal H3 while still
        asking the child for an independent response.
        """

        if (
            state.expression_level is not ExpressionLevel.L0
            and state.hint_level is not HintLevel.H3
        ):
            return
        state.expression_level = ExpressionLevel.L0
        state.hint_level = HintLevel.H3
        state.task_max_hint = HintLevel.H3

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
        help_card_visible: bool = False,
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
        # A reviewed follow-up question describes what the child should teach
        # next; it is not evidence that the child has already taught every
        # number appearing in that question.  Only child-confirmed facts and
        # grounded child arithmetic claims may enter the speaker's numerical
        # knowledge boundary.
        reviewed_numbers = sorted(
            extract_numeric_values(" ".join(verified_facts.values()))
        )
        arithmetic_claims = self._speaker_arithmetic_claims(
            task,
            child_text,
            analysis,
        )
        child_claim_numbers = sorted(
            {
                str(quantity.value)
                for claim in arithmetic_claims
                for quantity in (claim.left, claim.right, claim.claimed_result)
            }
        )
        allowed_numbers = sorted({*reviewed_numbers, *child_claim_numbers})
        expression = self._safe_grounding_span(
            child_text,
            analysis,
            allowed_numbers=set(allowed_numbers),
        )
        if dialogue_act in {
            "acknowledge_non_learning_and_reask",
            "acknowledge_task_comment_and_reask",
            "acknowledge_meta_and_reask",
            "brief_ack_and_redirect",
        }:
            expression = self._safe_social_grounding_span(
                child_text,
                analysis,
                allowed_numbers=set(allowed_numbers),
            )
        elif not set(required_slot_ids).intersection(task.text_explanation_slots):
            # Do not offer a quotable child phrase to the speaker when the
            # only unresolved focus is a reviewed canonical answer.
            expression = None
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
        # The speaker never decides pedagogy or mathematical truth.  The
        # redundant verifier model is gone, but a semantic *wording policy* is
        # still useful when the speaker needs to refer to the child's own
        # expression or ask for an open explanation.  In that case the main
        # speaker may paraphrase the reviewed question naturally.  Child-facing
        # wording policy is enforced by the prompt and offline regressions; the
        # runtime boundary only enforces dialogue act, slot ids, declared
        # verified-fact ids and exact child-quote provenance.
        verification_policy = self._speaker_verification_policy(
            task,
            required_slot_ids,
            has_child_grounding=bool(expression),
            has_unverified_arithmetic=any(
                claim.truth_status != "true" for claim in arithmetic_claims
            ),
        )
        safe_child_utterance = self._safe_child_utterance(child_text, analysis)
        return SpeakerContext(
            dialogue_act=dialogue_act,
            task_relation=analysis.task_relation,
            interaction_intent=analysis.interaction_intent,
            interaction_repeat_count=state.unrelated_count,
            support_trigger=support_trigger,
            help_card_event=help_card_event,
            expression_level=state.expression_level,
            hint_level=state.hint_level,
            unresolved_focus=required_slot_descriptions,
            must_reframe=must_reframe,
            previous_question=previous_question,
            required_question=required_question,
            verified_facts=dict(verified_facts),
            required_slot_ids=required_slot_ids,
            required_slot_descriptions=required_slot_descriptions,
            question_intent=question_intent,
            child_expression_mode=mode,
            child_expression=expression,
            safe_child_utterance=safe_child_utterance,
            arithmetic_claims=arithmetic_claims,
            help_card_visible=help_card_visible,
            child_claim_numbers=child_claim_numbers,
            allowed_numbers=allowed_numbers,
            verification_policy=verification_policy,
            fallback_text=fallback,
        )

    @staticmethod
    def _speaker_verification_policy(
        task: TaskDefinition,
        required_slot_ids: list[str],
        *,
        has_child_grounding: bool,
        has_unverified_arithmetic: bool,
    ) -> SpeakerVerificationPolicy:
        """Choose strict-copy vs natural wording without another model call.

        ``SEMANTIC`` now describes the main speaker's permitted surface form;
        it no longer schedules a verifier LLM.  Mathematical truth and state
        transitions have already been frozen by the orchestrator.
        """

        needs_open_explanation = bool(
            set(required_slot_ids).intersection(task.text_explanation_slots)
        )
        if has_child_grounding or has_unverified_arithmetic or needs_open_explanation:
            return SpeakerVerificationPolicy.SEMANTIC
        return SpeakerVerificationPolicy.DETERMINISTIC

    @staticmethod
    def _safe_child_utterance(
        child_text: str | None,
        analysis: UtteranceAnalysis,
    ) -> str | None:
        """Return bounded untrusted child text only for a normal safe turn."""

        if analysis.safety_category is not SafetyCategory.NORMAL or not child_text:
            return None
        text = re.sub(r"\s+", " ", child_text).strip()
        if not text or len(text) > 240:
            return None
        if deterministic_safety(text) is not SafetyCategory.NORMAL:
            return None
        return text

    @classmethod
    def _arithmetic_result(
        cls,
        operation: str,
        left: int,
        right: int,
    ) -> int | None:
        """Compute only the four reviewed integer operations.

        Exact division is required because every current learning contract has
        one integer answer.  Returning ``None`` makes malformed or non-exact
        relations false instead of silently rounding them.
        """

        if operation == "addition":
            return left + right
        if operation == "subtraction":
            return left - right
        if operation == "multiplication":
            return left * right
        if operation == "division" and right != 0 and left % right == 0:
            return left // right
        return None

    @classmethod
    def _speaker_arithmetic_claims(
        cls,
        task: TaskDefinition,
        child_text: str | None,
        analysis: UtteranceAnalysis,
    ) -> list[SpeakerArithmeticClaim]:
        """Attach scene meaning to exact child arithmetic evidence.

        The primary understanding model interprets the child's language. This method does not parse
        Korean; it verifies provenance, computes truth, and supplies reviewed
        labels such as ``낸 돈`` and ``거스름돈`` so Sonnet never has to infer
        what anonymous ``left`` and ``result`` fields mean.
        """

        if analysis.safety_category is not SafetyCategory.NORMAL or not child_text:
            return []
        contract = task.arithmetic_contract
        output: list[SpeakerArithmeticClaim] = []
        for claim in analysis.arithmetic_claims:
            source = cls._exact_child_evidence_text(child_text, claim.evidence_span)
            if source is None:
                continue
            direct_alignment = bool(
                contract
                and claim.operation == contract.operation
                and claim.left == contract.left
                and claim.right == contract.right
            )
            reversed_commutative_alignment = bool(
                contract
                and claim.operation == contract.operation
                and claim.operation in {"addition", "multiplication"}
                and claim.left == contract.right
                and claim.right == contract.left
            )
            aligned = direct_alignment or reversed_commutative_alignment
            computed = cls._arithmetic_result(
                claim.operation,
                claim.left,
                claim.right,
            )
            unit = contract.unit if contract and aligned else ""
            output.append(
                SpeakerArithmeticClaim(
                    operation=claim.operation,
                    source_text=source,
                    left=SpeakerQuantity(
                        value=claim.left,
                        role=(
                            contract.right_label
                            if contract and reversed_commutative_alignment
                            else contract.left_label
                            if contract and direct_alignment
                            else "첫 번째 수"
                        ),
                        unit=unit,
                    ),
                    right=SpeakerQuantity(
                        value=claim.right,
                        role=(
                            contract.left_label
                            if contract and reversed_commutative_alignment
                            else contract.right_label
                            if contract and direct_alignment
                            else "두 번째 수"
                        ),
                        unit=unit,
                    ),
                    claimed_result=SpeakerQuantity(
                        value=claim.result,
                        role=(
                            contract.result_label
                            if contract and aligned
                            else "아이가 말한 계산 결과"
                        ),
                        unit=unit,
                    ),
                    truth_status=(
                        "true"
                        if computed is not None and claim.result == computed
                        else "false"
                    ),
                    related_slot_ids=[
                        slot_id for slot_id in claim.related_slot_ids if slot_id in task.slots
                    ],
                )
            )
        return output

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

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, round((time.perf_counter() - started_at) * 1000))

    @staticmethod
    def _open_conversation_fallback(repeat_count: int) -> str:
        """Reviewed no-learning fallback independent of any intent enum."""

        if repeat_count >= 2:
            return "응, 들었어. 나는 아직 이게 궁금해... 이것만 알려주면 안 돼?"
        return "그렇구나. 나는 아직 이게 궁금해... 이것만 알려주면 안 돼?"

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
        del task, state
        return SpeakerGuardContract(
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
            mormi=MormiContract(text=self._complete_mormi_text(text), mood=mood),  # type: ignore[arg-type]
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
            task_anchor=self._task_anchor(state, task, input_contract),
            dictionary_ref=(
                dictionary_reference(state.dictionary_snapshots[task.id])
                if task.id in state.dictionary_snapshots
                else None
            ),
        )

    def ensure_task_anchor(
        self,
        state: SessionState,
        turn: TurnContract,
    ) -> TurnContract:
        """Backfill the additive anchor contract for an active legacy turn."""

        if turn.task_anchor is not None or turn.status is not SessionStatus.ACTIVE:
            return turn
        task = get_task(state.current_task_id, state.scenario_data)
        anchor = self._task_anchor(state, task, turn.input)
        return turn.model_copy(update={"task_anchor": anchor})

    @staticmethod
    def _task_anchor_label(task: TaskDefinition, slot_id: str) -> str:
        slot = task.slots[slot_id]
        if len(slot.description) <= 24:
            return slot.description
        return {
            "observation": "확인한 것",
            "conclusion": "알아낸 답",
            "operation": "계산",
            "method": "알려준 방법",
            "reason": "알려준 이유",
            "explanation": "알려준 설명",
            "selection": "고른 것",
        }[slot.semantic_role]

    def _task_anchor(
        self,
        state: SessionState,
        task: TaskDefinition,
        input_contract: InputContract,
    ) -> TaskAnchorContract | None:
        """Build the visible reminder from reviewed content and verified state only."""

        if state.status is not SessionStatus.ACTIVE or input_contract.kind is InputKind.NONE:
            return None
        step = task.step_by_id(state.subgoal_id)
        if step is None:
            step = task.active_step(
                state.expression_level,
                state.verified_slots,
                entry_active=(state.entry_phase is EntryPhase.AWAITING_ENTRY_RESPONSE),
                targeted_followup=(state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP),
            )
        target_slots = [
            slot_id
            for slot_id in input_contract.target_slots
            if slot_id not in state.verified_slots
        ]
        # L0 joint performance can intentionally revisit already verified
        # pieces. Keep the contract usable without inventing another target.
        if not target_slots:
            target_slots = list(input_contract.target_slots)
        completed_items = [
            TaskAnchorCompletedItem(
                slot_id=slot_id,
                label=self._task_anchor_label(task, slot_id),
                value=state.verified_slots[slot_id],
                display_text=task.slots[slot_id].fact_sentence,
            )
            for slot_id in task.slots
            if slot_id in state.verified_slots and slot_id not in target_slots
        ]
        return TaskAnchorContract(
            anchor_id=f"{task.id}:{state.subgoal_id}",
            prompt=step.prompt,
            completed_items=completed_items,
            target_slots=target_slots,
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

        if state.scenario_id in PARK_SCENARIO_IDS:
            raw_context = state.scenario_data.get("park_context")
            context = ParkSessionContext.model_validate(raw_context)
            values = {fact.key: fact.value for fact in context.facts}
            # This is the BE-authored and SessionCreate-validated contract.
            # Do not derive completion facts from LLM prose or child text.
            return {
                key: values[key]
                for key in context.required_verified_fact_keys
            }

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
                factual = slot.accepts(value)
                claims.append(
                    {
                        "slot_id": slot_id,
                        "value": value,
                        "factual": factual,
                        "evidence_span": str(value),
                        "supported": factual if slot.is_semantic_support else None,
                        "support_confidence": (
                            1.0 if factual and slot.is_semantic_support else None
                        ),
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
        *,
        arithmetic_claims: list[SpeakerArithmeticClaim] | None = None,
        help_card_visible: bool | None = None,
    ) -> str:
        """Build a coherent reviewed line for each support reason."""

        if event is HelpCardEvent.JOINT:
            return "이제 도움 카드대로 나랑 같이 해보자."
        if trigger is SupportTrigger.RELATED_VAGUE:
            if event is HelpCardEvent.NONE:
                if grounding:
                    return cls._complete_mormi_text(
                        f"그런데 ‘{grounding}’가 어떻게 하는 건지 모르겠어... 조금만 더 알려줄래?"
                    )
                return "어떻게 하는 건지 아직 헷갈려... 조금만 더 알려줄래?"
            if grounding:
                return cls._complete_mormi_text(
                    f"‘{grounding}’가 아직 헷갈려... 카드도 보고 조금 더 알려줄래?"
                )
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
            false_claim = next(
                (claim for claim in (arithmetic_claims or []) if claim.truth_status == "false"),
                None,
            )
            if false_claim is not None:
                return cls._false_arithmetic_claim_fallback(
                    false_claim,
                    help_card_visible=(
                        event is not HelpCardEvent.NONE
                        if help_card_visible is None
                        else help_card_visible
                    ),
                )
            if grounding:
                if help_card_visible or event is not HelpCardEvent.NONE:
                    return cls._complete_mormi_text(
                        f"‘{grounding}’이라고? 나는 아직 헷갈려... "
                        "도움 카드를 보고 다시 알려줄 수 있어?"
                    )
                return cls._complete_mormi_text(
                    f"‘{grounding}’이라고? 나는 아직 헷갈려... 조금 더 알려줄 수 있어?"
                )
            if event is HelpCardEvent.STRENGTHENED:
                return "나는 아직 헷갈려... 카드에 나온 다른 도움도 같이 볼까?"
            if help_card_visible or event is not HelpCardEvent.NONE:
                return "나는 아직 헷갈려... 도움 카드를 보고 다시 알려줄 수 있어?"
            return "나는 아직 헷갈려... 어떻게 생각한 건지 조금 더 알려줄 수 있어?"
        return "어떻게 하는 건지 아직 헷갈려... 조금만 더 알려줄래?"

    @classmethod
    def _false_arithmetic_claim_fallback(
        cls,
        claim: SpeakerArithmeticClaim,
        *,
        help_card_visible: bool,
    ) -> str:
        """Reflect one false child claim without confirming or correcting it."""

        left = cls._quantity_copy(claim.left)
        right = cls._quantity_copy(claim.right)
        result = cls._quantity_copy(claim.claimed_result)
        if claim.operation == "addition":
            relation = f"{left}과 {right}을 더하면 모두 {result}이라는 말이"
        elif claim.operation == "multiplication":
            relation = f"{left}에 {right}을 곱하면 {result}이라는 말이"
        elif claim.operation == "division":
            relation = f"{left}을 {right}으로 나누면 {result}이라는 말이"
        elif "거스름" in claim.claimed_result.role:
            relation = f"{left}에서 {right}을 빼면 거스름돈이 {result}이라는 말이"
        elif "남" in claim.claimed_result.role:
            relation = f"{left}에서 {right}을 빼면 {result}이 남는다는 말이"
        else:
            relation = f"{left}에서 {right}을 빼면 결과가 {result}이라는 말이"
        request = (
            "아직 잘 모르겠어... 도움 카드를 보고 다시 알려줄 수 있어?"
            if help_card_visible
            else "아직 잘 모르겠어... 어떻게 계산한 건지 다시 알려줄 수 있어?"
        )
        return cls._complete_mormi_text(f"{relation} {request}")

    @staticmethod
    def _quantity_copy(quantity: SpeakerQuantity) -> str:
        return f"{quantity.value:,}{quantity.unit}"

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
        if acknowledgement:
            return ConversationEngine._preface_question(acknowledgement, step.prompt)
        return ConversationEngine._complete_mormi_text(step.prompt)

    @staticmethod
    def _younger_sibling_acknowledgement(
        task: TaskDefinition,
        newly_verified: Mapping[str, object],
    ) -> str | None:
        """Acknowledge a verified fact without hard-coded example values.

        This is a last-resort deterministic line used only when speaker
        generation fails.  If code cannot express the verified fact safely,
        it returns ``None`` and lets the reviewed next question stand alone
        instead of attaching a generic ``아, 그렇구나!``.
        """

        if task.skill_id == "number-count" and "answer" in newly_verified:
            match = re.fullmatch(
                r"\s*(\d[\d,]*)\s*(?:개)?\s*",
                str(newly_verified["answer"]),
            )
            if match:
                count = int(match.group(1).replace(",", ""))
                native_count = {
                    1: "한 개",
                    2: "두 개",
                    3: "세 개",
                    4: "네 개",
                    5: "다섯 개",
                    6: "여섯 개",
                    7: "일곱 개",
                    8: "여덟 개",
                    9: "아홉 개",
                    10: "열 개",
                }.get(count, f"{count:,}개")
                return f"아, {native_count}구나!"
        if task.skill_id == "number-compare" and "answer" in newly_verified:
            side = str(newly_verified["answer"]).strip()
            if side in {"왼쪽", "오른쪽"}:
                return f"아, {side}이 더 많구나!"
        if (
            "result" in newly_verified
            and isinstance(newly_verified["result"], int)
            and not isinstance(newly_verified["result"], bool)
        ):
            return f"아, {newly_verified['result']:,}원이구나!"
        return None

    @staticmethod
    def _ensure_required_question(fallback: str, question: str) -> str:
        normalized_fallback = re.sub(r"\s+", "", fallback)
        normalized_question = re.sub(r"\s+", "", question)
        if normalized_question in normalized_fallback:
            return ConversationEngine._complete_mormi_text(fallback)
        return ConversationEngine._preface_question(fallback, question)

    @staticmethod
    def _smooth_ladder_fallback(state: SessionState, task: TaskDefinition) -> str:
        step = task.step_for(state.expression_level, state.verified_slots)
        if state.expression_level is ExpressionLevel.L2:
            # At L2 the UI itself has changed to reviewed choices.  Repeating
            # an L4/L3 prompt here made the screen look stuck even though the
            # input contract had changed.  This is one complete emergency
            # fallback, not a generic prefix attached to arbitrary questions.
            return "그럼 혹시 여기서 골라서 알려줄 수 있어?"
        if state.expression_level is ExpressionLevel.L3:
            # Do not bolt a generic excuse onto every ladder transition.  It
            # was shown even when the child had answered part of the question
            # and made unrelated sessions sound identical.  The authored
            # prompt already carries the exact remaining focus.
            return ConversationEngine._complete_mormi_text(step.prompt)
        return ConversationEngine._complete_mormi_text("도움 카드 순서대로 나와 같이 해볼까?")

    @staticmethod
    def _task_transition_then_question(
        task: TaskDefinition,
        _contribution: str,
        question: str,
    ) -> str:
        """Return a reviewed task transition without generic praise glue.

        ``transition_text`` is authored for a concrete state change such as a
        menu being selected.  If a task has no such transition, the next
        reviewed question stands on its own.  A generic acknowledgement must
        not imply that a tap, selection, or incomplete utterance taught Mormi
        something.
        """

        if task.transition_text:
            return ConversationEngine._preface_question(task.transition_text, question)
        return ConversationEngine._complete_mormi_text(question)

    @staticmethod
    def _response_as_text(response: ChildResponse) -> str:
        if response.choice_ids:
            return " ".join(response.choice_ids)
        return str(response.values)

    @staticmethod
    def _complete_mormi_text(text: str) -> str:
        """Normalize child-facing copy without imposing a character limit.

        The speaker is prompted to target 50 characters, but runtime code must
        never reject or slice an otherwise valid complete sentence by length.
        """

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _preface_question(preface: str, question: str) -> str:
        return ConversationEngine._complete_mormi_text(f"{preface} {question}")

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

    @staticmethod
    def _exact_child_evidence_text(
        child_text: str,
        evidence_span: str,
    ) -> str | None:
        """Return an exact, bounded clause from the child's own utterance.

        Open method/reason/explanation slots are semantically judged by the
        classifier.  Their evidence may legitimately contain a derived number
        that is not printed on screen (for example, ``5와 3의 차이는 2``).
        Reusing the note-specific visible-number allowlist here used to erase
        such valid explanations and leave the classifier's ``correct_partial``
        label uncorrected.  Exact-substring provenance keeps model-authored
        wording out without reducing open mathematical reasoning to a finite
        number or phrase list.
        """

        span = re.sub(r"\s+", " ", evidence_span).strip()
        source = re.sub(r"\s+", " ", child_text).strip()
        if not span or span not in source or len(span) > 120:
            return None
        if re.search(r"(틀렸|맞았|정답|오답)", span):
            return None
        return span

    @staticmethod
    def _operation_claim_is_grounded(evidence: str, value: object) -> bool:
        """Require an operation claim to be visible in the child's own words.

        The reviewed task contract may tell the understanding model which
        operation is mathematically expected, but that contract is never
        evidence that the child said it.  This small lexical boundary does not
        decide whether a method is pedagogically good; it only prevents a
        model-authored ``addition``/``subtraction`` claim from being promoted
        when the utterance contains no matching operation at all.
        """

        normalized = re.sub(r"\s+", "", str(value)).lower()
        if normalized in {"addition", "add", "+"} or re.search(
            r"(덧셈|더하기|더하|더해|더했|더하면|합치|합쳐|합하면|모으)",
            normalized,
        ):
            return bool(
                re.search(r"(덧셈|더하기|더하|더해|더했|더하면|합치|합쳐|합하면|모으)", evidence)
            )
        if normalized in {"subtraction", "subtract", "-"} or re.search(
            r"(뺄셈|빼기|빼|뺀|남|거스름|차이)", normalized
        ):
            return bool(re.search(r"(뺄셈|빼기|빼|뺀|남|거스름|차이)", evidence))
        return False

    @staticmethod
    def _canonical_claim_is_lexically_grounded(
        slot: SlotDefinition,
        evidence: str,
    ) -> bool:
        """Check provenance for a closed claim on a conversational turn.

        This is deliberately *not* a general answer parser.  Sonnet remains
        responsible for understanding creative child language.  The check is
        used only when the same model says the turn is meta/off-topic (or
        conversation-only), where a closed task claim is contradictory and
        therefore needs direct lexical evidence before it may cross the trust
        boundary.  It blocks outputs such as ``answer=right`` for "오늘 급식
        뭐야" without restricting ordinary current-task explanations.
        """

        normalized_evidence = re.sub(r"[\s,.'\"?!…]", "", evidence).lower()
        if not normalized_evidence:
            return False

        candidates = [
            slot.expected,
            *slot.aliases,
            *slot.accepted_values,
        ]
        lexical_aliases = {
            "left": ("왼쪽", "좌측"),
            "right": ("오른쪽", "우측"),
            "same": ("같아", "같다", "똑같"),
            "equal": ("같아", "같다", "똑같"),
        }
        for candidate in candidates:
            normalized_candidate = re.sub(
                r"[\s,.'\"?!…]", "", str(candidate)
            ).lower()
            if normalized_candidate and normalized_candidate in normalized_evidence:
                return True
            for alias in lexical_aliases.get(normalized_candidate, ()):
                if alias in normalized_evidence:
                    return True
        return False

    @classmethod
    def _enforce_claim_provenance(
        cls,
        task: TaskDefinition,
        child_text: str,
        analysis: UtteranceAnalysis,
    ) -> UtteranceAnalysis:
        """Freeze the trust boundary between model interpretation and state.

        The model may interpret spelling, ellipsis and creative explanations,
        but only an exact span from the *current* child utterance may become a
        learning claim.  Task answers, prior dialogue and reviewed arithmetic
        are context for interpretation, never provenance.  This makes an
        incorrect classifier output harmless before it reaches verified slots
        or the speaker.
        """

        sanitized = analysis.model_copy(deep=True)
        # These fields remain in the wire schema for compatibility, but this
        # architecture has neither a second adjudicator nor classifier-authored
        # child-facing bridge copy.
        sanitized.bridge_reply = ""
        sanitized.needs_adjudication = False
        sanitized.adjudication_reason = ""
        model_marked_conversation_only = sanitized.conversation_only

        def exact(span: str) -> str | None:
            return cls._exact_child_evidence_text(child_text, span)

        # Unsafe turns can never mutate pedagogy.  A model-authored
        # ``conversation_only`` flag is not itself a trust boundary: mixed
        # turns such as "왜 알려줘야 돼? 그래도 둘을 더하면 돼" may contain a
        # real learning fragment.  Ground every normal-turn claim first, then
        # decide whether the turn is conversation-only from what survived.
        if sanitized.safety_category is not SafetyCategory.NORMAL:
            sanitized.claims = []
            sanitized.arithmetic_claims = []
        else:
            grounded_claims: list[SlotClaim] = []
            contradictory_numeric_claim = False
            conversational_turn = model_marked_conversation_only or (
                sanitized.task_relation
                in {
                    TaskRelation.META_ABOUT_TASK,
                    TaskRelation.META_ABOUT_MORMI,
                    TaskRelation.OFF_TOPIC,
                    TaskRelation.UNKNOWN,
                }
            )
            for claim in sanitized.claims:
                slot = task.slots.get(claim.slot_id)
                evidence = exact(claim.evidence_span)
                if slot is None or evidence is None:
                    continue
                if slot.semantic_role == "operation" and not cls._operation_claim_is_grounded(
                    evidence, claim.value
                ):
                    continue

                # On a turn classified as conversational, a model-produced
                # closed task value is internally contradictory.  Preserve it
                # only when the child's exact evidence also names that value.
                # Selection controls are handled deterministically elsewhere,
                # and open method/reason slots remain Sonnet's semantic job.
                if (
                    conversational_turn
                    and not slot.is_semantic_support
                    and slot.semantic_role not in {"operation", "selection"}
                    and cls._claim_numeric_value(claim) is None
                    and not cls._canonical_claim_is_lexically_grounded(slot, evidence)
                ):
                    continue

                # For explicit numeric claims, the normalized value must be a
                # number the child actually uttered.  Unparseable Korean words
                # and typos remain available to the semantic classifier and
                # later confidence checks instead of being rejected by regex.
                if not slot.is_semantic_support:
                    numeric_value = cls._claim_numeric_value(claim)
                    evidence_values = {
                        int(value) for value in extract_numeric_values(evidence)
                    }
                    if (
                        numeric_value is not None
                        and evidence_values
                        and numeric_value not in evidence_values
                    ):
                        # The classifier copied a reviewed number that the
                        # child did not say (for example expected=1700 with
                        # exact evidence "1800원이야").  Drop the fabricated
                        # claim, but retain the observation that the child's
                        # explicit task answer was mathematically different.
                        contradictory_numeric_claim = True
                        continue
                    if (
                        conversational_turn
                        and numeric_value is not None
                        and not evidence_values
                    ):
                        continue
                claim.evidence_span = evidence
                grounded_claims.append(claim)
            sanitized.claims = grounded_claims

            if contradictory_numeric_claim:
                sanitized.response_category = ResponseCategory.CONCEPTUAL_ERROR
                sanitized.difficulty_class = DifficultyClass.CONCEPT
                sanitized.bottleneck = "concept"
                if sanitized.misconception_tag is None and task.misconception_tags:
                    sanitized.misconception_tag = task.misconception_tags[0]

            grounded_arithmetic = []
            for relation in sanitized.arithmetic_claims:
                evidence = exact(relation.evidence_span)
                if evidence is None:
                    continue
                evidence_values = {
                    int(value) for value in extract_numeric_values(evidence)
                }
                if not {relation.left, relation.right, relation.result}.issubset(
                    evidence_values
                ):
                    continue
                if not cls._operation_claim_is_grounded(evidence, relation.operation):
                    continue
                relation.evidence_span = evidence
                relation.related_slot_ids = [
                    slot_id for slot_id in relation.related_slot_ids if slot_id in task.slots
                ]
                grounded_arithmetic.append(relation)
            sanitized.arithmetic_claims = grounded_arithmetic

        sanitized.reference_resolutions = [
            resolution
            for resolution in sanitized.reference_resolutions
            if exact(resolution.source_span) is not None
        ]
        sanitized.grounding_span = exact(sanitized.grounding_span) or ""
        sanitized.social_grounding_span = exact(sanitized.social_grounding_span) or ""

        # A safe meta/off-topic utterance with no child-grounded learning
        # evidence belongs to the cheap conversational bridge.  Mixed turns
        # stay on the pedagogical path because their real learning fragment is
        # preserved above.
        if not sanitized.claims and not sanitized.arithmetic_claims and (
            model_marked_conversation_only
            or sanitized.task_relation
            in {
                TaskRelation.META_ABOUT_TASK,
                TaskRelation.META_ABOUT_MORMI,
                TaskRelation.OFF_TOPIC,
            }
        ):
            sanitized.conversation_only = True
        elif sanitized.claims or sanitized.arithmetic_claims:
            sanitized.conversation_only = False
        return sanitized

    @staticmethod
    def _claim_numeric_value(claim: SlotClaim) -> int | None:
        """Return one normalized integer that the classifier says the child claimed."""

        if isinstance(claim.value, bool) or claim.value is None:
            return None
        if isinstance(claim.value, int):
            return claim.value
        if isinstance(claim.value, float):
            return int(claim.value) if claim.value.is_integer() else None
        values = extract_numeric_values(str(claim.value))
        if len(values) != 1:
            return None
        return int(next(iter(values)))

    @classmethod
    def _ground_text_numeric_claims(
        cls,
        task: TaskDefinition,
        child_text: str,
        analysis: UtteranceAnalysis,
    ) -> None:
        """Bind numeric claims to the child's evidence before correctness checks.

        The classifier interprets Korean wording and common typos, but it may
        never replace the child's wrong number with the reviewed answer.  Clear
        numbers are checked deterministically.  Only an exact, otherwise
        unparseable evidence span can use a high-confidence model
        interpretation, which preserves support for forms such as a typo in a
        Korean number word without trusting model-authored arithmetic.
        """

        for claim in analysis.claims:
            slot = task.slots.get(claim.slot_id)
            if slot is None or slot.is_semantic_support or isinstance(slot.expected, bool):
                continue

            reviewed_numbers = extract_numeric_values(str(slot.expected))
            if not isinstance(slot.expected, (int, float)) and len(reviewed_numbers) != 1:
                continue

            evidence = cls._exact_child_evidence_text(child_text, claim.evidence_span)
            claimed_value = cls._claim_numeric_value(claim)
            if evidence is None or claimed_value is None:
                claim.factual = False
                continue

            evidence_values = {int(value) for value in extract_numeric_values(evidence)}
            if evidence_values:
                # The normalized claim must be one of the quantities the child
                # actually said.  This blocks expected=1700/evidence=1800 even
                # if the model incorrectly reports value=1700.
                if claimed_value not in evidence_values:
                    claim.factual = False
                elif slot.accepts(claim.value):
                    # Preserve an already valid reviewed form such as
                    # ``4묶음``, ``6cm`` or ``1시간``.  Those units carry task
                    # meaning and must not be collapsed to a bare integer.
                    pass
                else:
                    # Keep Korean surface wording in evidence_span, but expose
                    # a closed numeric value to the slot contract.  This is
                    # value normalization, not interpretation of the child's
                    # sentence ending or intent.
                    claim.value = claimed_value
                continue

            # Reviewed aliases such as "셋" are deterministic evidence even
            # when the general Korean-number parser intentionally stays narrow.
            if slot.accepts(evidence) and slot.accepts(claimed_value):
                claim.value = claimed_value
                continue

            if (claim.interpretation_confidence or 0) < 0.85:
                claim.factual = False
            else:
                claim.value = claimed_value

    @classmethod
    def _invalidate_false_structured_arithmetic(
        cls,
        task: TaskDefinition,
        child_text: str,
        analysis: UtteranceAnalysis,
    ) -> None:
        """Reject false arithmetic after Claude has understood the wording.

        No Korean phrase or sentence ending is interpreted here.  The model
        extracts one structured relation from the child's exact evidence;
        code performs only arithmetic and provenance checks.  A false relation
        invalidates the explanation slots it was offered to support, while a
        separately correct answer claim may still be preserved as partial
        progress.
        """

        false_slot_ids: set[str] = set()
        found_false_relation = False
        for arithmetic in analysis.arithmetic_claims:
            evidence = cls._exact_child_evidence_text(child_text, arithmetic.evidence_span)
            if evidence is None:
                continue
            computed = cls._arithmetic_result(
                arithmetic.operation,
                arithmetic.left,
                arithmetic.right,
            )
            if computed is not None and arithmetic.result == computed:
                continue
            found_false_relation = True
            false_slot_ids.update(
                slot_id
                for slot_id in arithmetic.related_slot_ids
                if slot_id in task.slots and task.slots[slot_id].is_semantic_support
            )

        if not found_false_relation:
            return

        # The classifier's prose-level label is only a proposal.  Once the
        # deterministic arithmetic check proves that the child's stated
        # relation is false, the effective observation must say so as well.
        # Leaving ``correct_full`` here used to keep state mutation safe while
        # corrupting analytics and selecting success-shaped dialogue acts.
        analysis.response_category = ResponseCategory.CONCEPTUAL_ERROR
        analysis.difficulty_class = DifficultyClass.CONCEPT
        analysis.bottleneck = "concept"
        if analysis.misconception_tag is None and task.misconception_tags:
            analysis.misconception_tag = task.misconception_tags[0]

        # If the classifier omitted related_slot_ids, fail closed only for
        # semantic claims from this response. Closed answers are still checked
        # independently against their reviewed values.
        if not false_slot_ids:
            false_slot_ids = {
                claim.slot_id
                for claim in analysis.claims
                if claim.slot_id in task.slots and task.slots[claim.slot_id].is_semantic_support
            }
        for claim in analysis.claims:
            if claim.slot_id in false_slot_ids:
                claim.factual = False
                claim.supported = False

    @classmethod
    def _invalidate_unverified_arithmetic_support(
        cls,
        task: TaskDefinition,
        child_text: str,
        analysis: UtteranceAnalysis,
    ) -> None:
        """Do not trust a number-rich explanation without a checkable relation.

        The primary understanding model remains responsible for understanding
        the child's language.  The engine only requires that a supported
        arithmetic explanation carry the structured relation promised by the
        classifier contract.  Counting numeric evidence is a provenance guard,
        not a Korean phrase parser.
        A separately correct closed answer remains available as partial
        progress even when the explanation must be re-elicited.
        """

        if task.arithmetic_contract is None:
            return

        structured_slots: set[str] = set()
        for arithmetic in analysis.arithmetic_claims:
            evidence = cls._exact_child_evidence_text(child_text, arithmetic.evidence_span)
            if evidence is None:
                continue
            structured_slots.update(arithmetic.related_slot_ids)

        for claim in analysis.claims:
            slot = task.slots.get(claim.slot_id)
            if slot is None or not slot.is_semantic_support or claim.supported is not True:
                continue
            evidence = cls._exact_child_evidence_text(child_text, claim.evidence_span)
            if evidence is None or len(extract_numeric_values(evidence)) < 3:
                continue
            if claim.slot_id not in structured_slots:
                claim.factual = False
                claim.supported = False

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
        for slot_id in task.semantic_support_slots | set(task.text_explanation_slots):
            if slot_id not in filtered:
                continue
            supported = False
            for claim in analysis.claims:
                if not claim.factual or claim.slot_id != slot_id:
                    continue
                evidence = cls._exact_child_evidence_text(child_text, claim.evidence_span)
                if claim.supported is True and evidence is not None:
                    supported = True
                    break
                # Compatibility for deterministic responses and old test
                # fixtures that predate the semantic-support contract.
                if claim.supported is None and cls._claim_evidence_text(
                    task,
                    child_text,
                    claim.evidence_span,
                ):
                    supported = True
                    break
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
                slot = task.slots[slot_id]
                evidence = (
                    cls._exact_child_evidence_text(
                        response.text,
                        claim.evidence_span,
                    )
                    if slot.is_semantic_support and claim.supported is True
                    else cls._claim_evidence_text(
                        task,
                        response.text,
                        claim.evidence_span,
                    )
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

    @staticmethod
    def _note_contextualization_context(
        state: SessionState,
        task: TaskDefinition,
        note: NoteUpdate | None,
    ) -> NoteContextualizationContext | None:
        """Build the only context an LLM may use to complete a direct note.

        Semantic method/reason fact sentences are intentionally excluded:
        those sentences can contain a curriculum-preferred strategy that the
        child never used.  The reviewed note context and canonical fact slots
        are enough to resolve scene-bound references without importing one.
        """

        if note is None or note.attribution is not NoteAttribution.CHILD:
            return None
        source_fragments = {
            slot_id: state.child_note_evidence[slot_id]
            for slot_id in task.effective_note_slots
            if slot_id in state.child_note_evidence
        }
        if not source_fragments:
            return None
        reviewed_facts = {"note_context": task.note_context}
        for slot_id in task.effective_note_slots:
            slot = task.slots[slot_id]
            if not slot.is_semantic_support and slot_id in state.verified_slots:
                reviewed_facts[f"slot:{slot_id}"] = slot.fact_sentence
        allowed_numbers = sorted(
            extract_numeric_values(
                " ".join(
                    [
                        *source_fragments.values(),
                        *reviewed_facts.values(),
                    ]
                )
            )
        )
        return NoteContextualizationContext(
            skill_id=task.skill_id,
            note_context=task.note_context,
            source_fragments=source_fragments,
            reviewed_facts=reviewed_facts,
            allowed_numbers=allowed_numbers,
            fallback_text=note.text,
        )


def max_hint(left: HintLevel, right: HintLevel) -> HintLevel:
    order = list(HintLevel)
    return order[max(order.index(left), order.index(right))]


def select_start_level(profile: LearnerProfile, skill_id: str) -> ExpressionLevel:
    skill = profile.skills.get(skill_id)
    if skill:
        return skill.highest_stable_expression_level.canonical()
    return ExpressionLevel.L4


def update_skill_profile(
    profile: LearnerProfile,
    state: SessionState,
    task: TaskDefinition,
) -> LearnerProfile:
    current = profile.skills.get(task.skill_id) or SkillProfile(skill_id=task.skill_id)
    current.highest_stable_expression_level = current.highest_stable_expression_level.canonical()
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
    profile.skills[task.skill_id] = current
    return profile
