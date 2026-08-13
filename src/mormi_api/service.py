from __future__ import annotations

from .content import create_scenario_data, get_scenario, get_task
from .engine import ConversationEngine, select_start_level, update_skill_profile
from .repository import DuplicateResponseError, Repository
from .schemas import (
    ChildResponse,
    EntryPhase,
    ExpressionLevel,
    InputContract,
    InputKind,
    PracticeResult,
    PracticeSummary,
    ResponseType,
    SessionCreate,
    SessionEnvelope,
    SessionState,
    utc_now,
)


class ConversationService:
    def __init__(self, repository: Repository, engine: ConversationEngine) -> None:
        self.repository = repository
        self.engine = engine

    async def save_practice(self, summary: PracticeResult) -> PracticeResult:
        await self.repository.save_practice_summary(summary)
        return summary

    async def create_conversation(self, request: SessionCreate) -> SessionEnvelope:
        scenario = get_scenario(request.scenario_id)
        if scenario.scene is not request.scene:
            raise ValueError("scenario_id does not belong to the requested scene")
        if request.learning_session_id:
            existing_id = await self.repository.conversation_id_for_learning_session(
                request.learner_id,
                request.learning_session_id,
            )
            if existing_id:
                return await self.snapshot(existing_id)
        practice_summary = request.practice_summary
        if practice_summary:
            if request.practice_result_id:
                inline_result = PracticeResult(
                    **practice_summary.model_dump(),
                    practice_result_id=request.practice_result_id,
                    learner_id=request.learner_id,
                )
                await self.repository.save_practice_summary(inline_result)
                # `practice_result_id` is the existing idempotency identity for
                # a completed drill. A retry must use the first persisted
                # snapshot instead of allowing a later payload to alter the
                # teaching scenario for that same result.
                stored_result = await self.repository.get_practice_summary(
                    request.practice_result_id
                )
                if not stored_result:
                    raise RuntimeError("practice result was not persisted")
                if stored_result.learner_id != request.learner_id:
                    raise ValueError("practice result does not belong to learner_id")
                practice_summary = PracticeSummary.model_validate(stored_result.model_dump())
        elif request.practice_result_id:
            loaded_result = await self.repository.get_practice_summary(request.practice_result_id)
            if not loaded_result:
                raise ValueError(
                    "practice_result_id is unavailable; include practice_summary for MVP"
                )
            if loaded_result.learner_id != request.learner_id:
                raise ValueError("practice result does not belong to learner_id")
            practice_summary = PracticeSummary.model_validate(loaded_result.model_dump())

        profile = await self.repository.get_profile(request.learner_id)
        curriculum_session_id = (
            practice_summary.curriculum_session_id if practice_summary else None
        )
        if request.scenario_id == "home_teach" and not practice_summary:
            raise ValueError("practice_summary or practice_result_id is required for home_teach")
        scenario_data = create_scenario_data(
            request.scenario_id,
            request.cafe_context,
            queue_context=request.queue_context,
            curriculum_session_id=curriculum_session_id,
            skill_id=practice_summary.skill_id if practice_summary else None,
            practice_result_id=request.practice_result_id,
        )
        task_start_levels = {
            task_id: (
                # Drill accuracy is concept-performance evidence, not evidence
                # that the child cannot explain.  Home teaching must therefore
                # offer one independent L4-H0 teaching turn before adding any
                # expression support.
                ExpressionLevel.L4
                if request.scenario_id == "home_teach"
                else ExpressionLevel.L2
                if get_task(task_id, scenario_data).behavior
                in {"budget_menu_selection", "menu_selection"}
                else select_start_level(
                    profile,
                    get_task(task_id, scenario_data).skill_id,
                )
            )
            for task_id in scenario.task_ids
        }
        start_level = task_start_levels[scenario.task_ids[0]]
        first_task = get_task(scenario.task_ids[0], scenario_data)
        started_at = utc_now()
        state = SessionState(
            learner_id=request.learner_id,
            learning_session_id=request.learning_session_id,
            scene=request.scene,
            scenario_id=request.scenario_id,
            task_ids=scenario.task_ids,
            scenario_data=scenario_data,
            task_start_levels=task_start_levels,
            expression_level=start_level,
            task_start_level=start_level,
            dialogue_policy_version=2,
            entry_phase=(
                EntryPhase.AWAITING_ENTRY_RESPONSE
                if first_task.entry_step is not None and start_level is ExpressionLevel.L4
                else EntryPhase.RESOLVED
            ),
            raw_storage_enabled=request.conversation_storage_consent,
            retention_policy=request.retention_policy,
            raw_retention_until=request.retention_policy.expires_at(started_at),
            created_at=started_at,
            updated_at=started_at,
        )
        turn = self.engine.initial_turn(state)
        state.current_turn_id = turn.turn_id
        await self.repository.create_conversation(state, turn)
        return SessionEnvelope(conversation_id=state.conversation_id, turn=turn)

    async def respond(
        self,
        conversation_id: str,
        response: ChildResponse,
    ) -> SessionEnvelope:
        response_id = str(response.response_id)
        prior = await self.repository.response_exists(conversation_id, response_id)
        if prior:
            return SessionEnvelope(conversation_id=conversation_id, turn=prior)

        state = await self.repository.get_state(conversation_id)
        active_turn = await self.repository.active_turn(state)
        if active_turn.turn_id != response.turn_id:
            raise ValueError("turn_id is stale; use the latest turn")
        if not response_matches_input(active_turn.input.kind, response.type):
            raise ValueError(
                f"response type {response.type.value} does not match "
                f"input kind {active_turn.input.kind.value}"
            )
        validate_response_payload(active_turn.input, response)
        previous_task = get_task(state.current_task_id, state.scenario_data)
        next_state, analysis, next_turn = await self.engine.run_turn(
            state,
            response,
            active_turn.mormi.text,
        )
        try:
            await self.repository.commit_turn(
                previous_state=state,
                next_state=next_state,
                response=response,
                analysis=analysis,
                next_turn=next_turn,
                previous_question=active_turn.mormi.text,
                note=next_turn.note_update,
            )
        except DuplicateResponseError:
            prior = await self.repository.response_exists(conversation_id, response_id)
            if prior:
                return SessionEnvelope(conversation_id=conversation_id, turn=prior)
            raise

        if next_turn.note_update:
            profile = await self.repository.get_profile(state.learner_id)
            evidence_state = state.model_copy(deep=True)
            evidence_state.verified_slots = {
                slot_id: previous_task.slots[slot_id].expected
                for slot_id in previous_task.required_slots
            }
            profile = update_skill_profile(profile, evidence_state, previous_task)
            await self.repository.save_profile(profile)

        return SessionEnvelope(conversation_id=conversation_id, turn=next_turn)

    async def snapshot(self, conversation_id: str) -> SessionEnvelope:
        state = await self.repository.get_state(conversation_id)
        turn = await self.repository.active_turn(state)
        return SessionEnvelope(conversation_id=conversation_id, turn=turn)


def response_matches_input(input_kind: InputKind, response_type: ResponseType) -> bool:
    if response_type is ResponseType.NO_RESPONSE:
        return input_kind is not InputKind.NONE
    expected = {
        InputKind.TEXT: ResponseType.TEXT,
        InputKind.CHOICES: ResponseType.CHOICE,
        InputKind.FILL: ResponseType.FILL,
        InputKind.COUNT: ResponseType.COUNT,
        InputKind.EQUATION: ResponseType.EQUATION,
        InputKind.JOINT: ResponseType.ACTION,
        InputKind.BUTTON: ResponseType.ACTION,
    }
    return expected.get(input_kind) is response_type


def validate_response_payload(input_contract: InputContract, response: ChildResponse) -> None:
    """Reject structured answers that do not belong to the active turn.

    Choice IDs are server-authored opaque identifiers.  Accepting an unknown
    ID, or more than one ID for a single-choice teaching turn, would let a
    stale/malformed client bypass the reviewed choice mapping.
    """

    if response.type not in {ResponseType.CHOICE, ResponseType.FILL}:
        return
    if len(response.choice_ids) != 1:
        raise ValueError("structured teaching response requires exactly one choice_id")
    allowed_ids = {choice.id for choice in input_contract.choices if not choice.disabled}
    if response.choice_ids[0] not in allowed_ids:
        raise ValueError("choice_id does not belong to the active turn")
