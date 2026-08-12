from __future__ import annotations

from .content import get_scenario, get_task
from .engine import ConversationEngine, select_start_level, update_skill_profile
from .repository import DuplicateResponseError, Repository
from .schemas import (
    ChildResponse,
    PracticeResult,
    PracticeSummary,
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
        practice_summary = request.practice_summary
        if practice_summary:
            if request.practice_result_id:
                inline_result = PracticeResult(
                    **practice_summary.model_dump(),
                    practice_result_id=request.practice_result_id,
                    learner_id=request.learner_id,
                )
                await self.repository.save_practice_summary(inline_result)
        elif request.practice_result_id:
            loaded_result = await self.repository.get_practice_summary(request.practice_result_id)
            if not loaded_result:
                raise ValueError(
                    "practice_result_id is unavailable; include practice_summary for MVP"
                )
            if loaded_result.learner_id != request.learner_id:
                raise ValueError("practice result does not belong to learner_id")
            practice_summary = PracticeSummary.model_validate(loaded_result.model_dump())

        practice_rate = practice_summary.success_rate if practice_summary else None

        profile = await self.repository.get_profile(request.learner_id)
        task_start_levels = {
            task_id: select_start_level(
                profile,
                get_task(task_id).skill_id,
                practice_rate if index == 0 else None,
            )
            for index, task_id in enumerate(scenario.task_ids)
        }
        start_level = task_start_levels[scenario.task_ids[0]]
        started_at = utc_now()
        state = SessionState(
            learner_id=request.learner_id,
            learning_session_id=request.learning_session_id,
            scene=request.scene,
            scenario_id=request.scenario_id,
            task_ids=scenario.task_ids,
            task_start_levels=task_start_levels,
            expression_level=start_level,
            task_start_level=start_level,
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
        previous_task = get_task(state.current_task_id)
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
