from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .content import TaskDefinition, get_task
from .llm import ClaudeGateway, ModelOutputError, ModelUnavailableError, validate_speaker_output
from .schemas import (
    ChildResponse,
    CompletionContract,
    CompletionOutcome,
    DifficultyClass,
    ExpressionLevel,
    HelpCardContract,
    HintLevel,
    InputContract,
    InputKind,
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
    SpeakerOutput,
    TurnContract,
    UtteranceAnalysis,
    VisualContract,
    new_id,
)
from .security import (
    deterministic_safety,
    safe_child_expression,
    safety_redirect,
)


class ConversationGraphState(TypedDict, total=False):
    session: dict[str, Any]
    response: dict[str, Any]
    previous_question: str
    analysis: dict[str, Any]
    decision: dict[str, Any]
    speaker_output: dict[str, Any]
    speaker_text: str
    turn: dict[str, Any]


class ConversationEngine:
    def __init__(self, gateway: ClaudeGateway, *, show_internal_pedagogy: bool = False) -> None:
        self.gateway = gateway
        self.show_internal_pedagogy = show_internal_pedagogy
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
        output = await self.graph.ainvoke(
            {
                "session": state.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "previous_question": previous_question,
            }
        )
        return (
            SessionState.model_validate(output["session"]),
            UtteranceAnalysis.model_validate(output["analysis"]),
            TurnContract.model_validate(output["turn"]),
        )

    def initial_turn(self, state: SessionState) -> TurnContract:
        task = get_task(state.current_task_id)
        step = task.step_for(state.expression_level, state.verified_slots)
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
        task = get_task(state.current_task_id)
        previous_question = graph_state["previous_question"]

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
        task = get_task(state.current_task_id)
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
        if decision.dialogue_act in {"unsafe_redirect", "show_help_card", "joint_mode"}:
            return {"speaker_text": context.fallback_text}
        try:
            output = await self.gateway.speak(context)
            return {"speaker_output": output.model_dump(mode="json")}
        except (ModelOutputError, ModelUnavailableError):
            return {"speaker_text": context.fallback_text}

    async def _validate_and_compose_node(
        self,
        graph_state: ConversationGraphState,
    ) -> dict[str, Any]:
        decision = PedagogicalDecision.model_validate(graph_state["decision"])
        text = graph_state.get("speaker_text")
        if not text and graph_state.get("speaker_output"):
            output = SpeakerOutput.model_validate(graph_state["speaker_output"])
            text = validate_speaker_output(output, decision.speaker_context)
        text = text or decision.speaker_context.fallback_text
        task = get_task(decision.state.current_task_id)
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

        if analysis.safety_category is not SafetyCategory.NORMAL:
            if analysis.safety_category is SafetyCategory.PLAYFUL_OFFTOPIC:
                next_state.unrelated_count += 1
                if next_state.unrelated_count >= 3:
                    next_state.expression_level = next_state.expression_level.lower()
                return self._decision_for_current_step(
                    next_state,
                    task,
                    dialogue_act="context_return",
                    fallback=safety_redirect(analysis.safety_category),
                    child_text=None,
                    analysis=analysis,
                    previous_question=previous_question,
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

        newly_verified = task.validated_claims(
            (claim.slot_id, claim.value, claim.factual) for claim in analysis.claims
        )
        next_state.verified_slots.update(newly_verified)
        if newly_verified:
            next_state.unrelated_count = 0
            if (
                state.expression_level is ExpressionLevel.L4
                and state.hint_level is HintLevel.H0
                and analysis.response_category is ResponseCategory.CORRECT_FULL
                and analysis.note_candidate.strip()
            ):
                next_state.direct_note_candidate = self._safe_direct_note(
                    task,
                    analysis.note_candidate,
                    newly_verified,
                )
            if task.complete(next_state.verified_slots):
                return self._complete_task(
                    next_state,
                    task,
                    analysis,
                    response.text,
                    previous_question,
                )
            # A partial answer is useful evidence, but the next prompt should
            # request only the missing piece instead of repeating the original
            # multi-part L4 question.
            if next_state.expression_level is ExpressionLevel.L4:
                next_state.expression_level = ExpressionLevel.L3
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="acknowledge_partial",
                fallback=self._partial_fallback(task, newly_verified, next_state),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
                newly_verified=newly_verified,
            )

        category = analysis.response_category
        if category in {
            ResponseCategory.EXPRESSION_BLOCK,
            ResponseCategory.NO_RESPONSE,
        }:
            next_state.expression_failures += 1
            next_state.expression_level = next_state.expression_level.lower()
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="reduce_expression_load",
                fallback=self._smooth_ladder_fallback(next_state, task),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
            )

        if category is ResponseCategory.HELP_REQUEST:
            next_state.expression_level = next_state.expression_level.lower()
            if next_state.hint_level is HintLevel.H0:
                next_state.hint_level = HintLevel.H1
                next_state.task_max_hint = max_hint(
                    next_state.task_max_hint,
                    HintLevel.H1,
                )
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="accept_help_request",
                fallback=self._preface_question(
                    "도움 카드가 열렸네.",
                    task.step_for(
                        next_state.expression_level,
                        next_state.verified_slots,
                    ).prompt,
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
            )

        if category in {
            ResponseCategory.CONCEPTUAL_ERROR,
            ResponseCategory.CONCEPTUAL_BLOCK,
        } or analysis.difficulty_class in {DifficultyClass.CONCEPT, DifficultyClass.BOTH}:
            next_state.concept_failures += 1
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
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="joint_mode"
                if next_state.hint_level is HintLevel.H3
                else "show_help_card",
                fallback=(
                    "도움 카드 순서대로 나와 같이 해볼까?"
                    if next_state.hint_level is HintLevel.H3
                    else "도움 카드가 열렸네. 같이 살펴볼까?"
                ),
                child_text=response.text,
                analysis=analysis,
                previous_question=previous_question,
            )

        if category is ResponseCategory.RECOGNITION_OR_INPUT_ERROR:
            return self._decision_for_current_step(
                next_state,
                task,
                dialogue_act="clarify_input",
                fallback="내가 잘 못 들었어. 방금 말만 다시 들려줄래?",
                child_text=None,
                analysis=analysis,
                previous_question=previous_question,
            )

        next_state.unrelated_count += 1
        if next_state.unrelated_count >= 3:
            next_state.expression_level = next_state.expression_level.lower()
        return self._decision_for_current_step(
            next_state,
            task,
            dialogue_act="context_return",
            fallback="그 얘기도 궁금해. 먼저 지금 상황만 같이 볼까?",
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
    ) -> PedagogicalDecision:
        direct = bool(
            state.direct_note_candidate
            and state.task_start_level is ExpressionLevel.L4
            and state.task_max_hint is HintLevel.H0
        )
        state.all_tasks_direct = state.all_tasks_direct and direct
        note = NoteUpdate(
            skill_id=task.skill_id,
            text=state.direct_note_candidate or task.coauthored_note,
            attribution=NoteAttribution.CHILD if direct else NoteAttribution.COAUTHORED,
            evidence=(
                NoteEvidence.DIRECT_EXPLANATION if direct else NoteEvidence.SUPPORTED_COMPLETION
            ),
            attribution_label="아이가 알려줌" if direct else "아이와 같이 공부함",
        )
        contribution = task.slots[task.required_slots[-1]].fact_sentence

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
            state.direct_note_candidate = None
            next_task = get_task(state.current_task_id)
            next_step = next_task.step_for(state.expression_level, state.verified_slots)
            fallback = self._success_then_question(contribution, next_step.prompt)
            return PedagogicalDecision(
                state=state,
                dialogue_act="task_transition",
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
                    verified_facts=[contribution],
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
                verified_facts=[contribution],
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
    ) -> PedagogicalDecision:
        if state.hint_level is HintLevel.H3:
            state.expression_level = ExpressionLevel.L0
        step = task.step_for(state.expression_level, state.verified_slots)
        state.subgoal_id = step.id
        help_card = self._help_card(task, state.hint_level)
        visual = self._visual_for(task, state.hint_level)
        verified_facts = [task.slots[slot].fact_sentence for slot in (newly_verified or {})]
        if dialogue_act == "show_help_card":
            fallback = self._preface_question("도움 카드가 열렸네.", step.prompt)
        elif dialogue_act == "joint_mode":
            fallback = self._preface_question("도움 카드 순서대로 보자.", step.prompt)
        return PedagogicalDecision(
            state=state,
            dialogue_act=dialogue_act,
            required_question=step.prompt,
            input=step.input,
            visual=visual,
            help_card=help_card,
            mood="thinking" if help_card else "listening",
            speaker_context=self._speaker_context(
                task=task,
                state=state,
                dialogue_act=dialogue_act,
                previous_question=previous_question,
                required_question=step.prompt,
                verified_facts=verified_facts,
                analysis=analysis,
                child_text=child_text,
                fallback=self._fit_50(fallback),
            ),
        )

    def _speaker_context(
        self,
        *,
        task: TaskDefinition,
        state: SessionState,
        dialogue_act: str,
        previous_question: str,
        required_question: str | None,
        verified_facts: list[str],
        analysis: UtteranceAnalysis,
        child_text: str | None,
        fallback: str,
    ) -> SpeakerContext:
        all_factual = bool(analysis.claims) and all(claim.factual for claim in analysis.claims)
        mode, expression = safe_child_expression(
            child_text,
            analysis.safety_category,
            all_claims_factual=all_factual,
        )
        missing = [
            task.slots[slot].description for slot in task.missing_slots(state.verified_slots)
        ]
        allowed_numbers = sorted(
            set(re.findall(r"\d[\d,]*", " ".join(verified_facts + [required_question or ""])))
        )
        return SpeakerContext(
            dialogue_act=dialogue_act,
            previous_question=previous_question,
            required_question=required_question,
            verified_facts=verified_facts,
            missing_slots=missing,
            child_expression_mode=mode,  # type: ignore[arg-type]
            child_expression=expression,
            allowed_numbers=allowed_numbers,
            fallback_text=fallback,
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
                )
                if state.status is SessionStatus.COMPLETED and state.completion_outcome is not None
                else None
            ),
            pedagogy=pedagogy,
        )

    @staticmethod
    def _deterministic_analysis(
        state: SessionState,
        task: TaskDefinition,
        response: ChildResponse,
    ) -> UtteranceAnalysis:
        step = task.step_for(state.expression_level, state.verified_slots)
        candidate_values: dict[str, object] = {}
        if response.type in {ResponseType.CHOICE, ResponseType.FILL}:
            for choice_id in response.choice_ids:
                candidate_values.update(step.choice_effects.get(choice_id, {}))
        elif response.type in {ResponseType.COUNT, ResponseType.EQUATION, ResponseType.ACTION}:
            candidate_values.update(response.values)

        claims = []
        for slot_id, value in candidate_values.items():
            slot = task.slots.get(slot_id)
            if slot:
                claims.append(
                    {
                        "slot_id": slot_id,
                        "value": value,
                        "factual": slot.accepts(value),
                        "evidence_span": str(value),
                    }
                )
        factual_count = sum(1 for claim in claims if claim["factual"])
        all_factual = bool(claims) and factual_count == len(claims)
        expected_count = len(step.target_slots)
        if all_factual and factual_count >= expected_count:
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
    def _visual_for(task: TaskDefinition, level: HintLevel) -> VisualContract:
        if level in {HintLevel.H2, HintLevel.H3}:
            hint = task.hints[level]
            if hint.visual_type:
                return VisualContract(type=hint.visual_type, data=hint.visual_data)
        return task.base_visual

    @staticmethod
    def _partial_fallback(
        task: TaskDefinition,
        newly_verified: Mapping[str, object],
        state: SessionState,
    ) -> str:
        facts = [task.slots[slot].fact_sentence.rstrip(".") for slot in newly_verified]
        step = task.step_for(state.expression_level, state.verified_slots)
        prefix = " ".join(facts[:1])
        return ConversationEngine._preface_question(f"{prefix}구나.", step.prompt)

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
    def _safe_direct_note(
        task: TaskDefinition,
        candidate: str,
        newly_verified: Mapping[str, object],
    ) -> str | None:
        text = re.sub(r"\s+", " ", candidate).strip()
        if not text or "?" in text or len(text) > 120:
            return None
        if set(task.required_slots) - set(newly_verified):
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
        return text


def max_hint(left: HintLevel, right: HintLevel) -> HintLevel:
    order = list(HintLevel)
    return order[max(order.index(left), order.index(right))]


def select_start_level(
    profile: LearnerProfile, skill_id: str, practice_rate: float | None
) -> ExpressionLevel:
    skill = profile.skills.get(skill_id)
    if skill:
        return skill.highest_stable_expression_level
    if practice_rate is None:
        return ExpressionLevel.L2
    if practice_rate >= 0.9:
        return ExpressionLevel.L4
    if practice_rate >= 0.7:
        return ExpressionLevel.L3
    return ExpressionLevel.L2


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
