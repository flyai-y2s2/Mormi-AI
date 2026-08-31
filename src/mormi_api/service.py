from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal, cast

from .content import create_scenario_data, get_scenario, get_task
from .dialogue_v2_amusement_content import materialize_amusement_scenario_v2
from .dialogue_v2_cafe_content import materialize_cafe_scenario_v2
from .dialogue_v2_life_runtime import DialogueV2LifeEngine
from .dialogue_v2_router import select_dialogue_runtime
from .dialogue_v2_runtime import DialogueV2Engine
from .dialogue_v2_versions import (
    DIALOGUE_V2_SNAPSHOT_READER_CAPABILITY_V2,
    DIALOGUE_V3_SNAPSHOT_READER_CAPABILITY_V1,
)
from .dictionary_catalog import (
    DICTIONARY_CATALOG,
    dictionary_card_envelope,
    get_dictionary_card_by_id,
)
from .dictionary_models import DictionaryCardEnvelope
from .engine import (
    ConversationEngine,
    EngineProgress,
    EngineTurnResult,
    select_start_level,
    update_skill_profile,
)
from .observability import TurnScope, turn_scope
from .repository import DuplicateResponseError, PersistenceError, Repository
from .schemas import (
    ChildResponse,
    DialogueRuntimeContractVersion,
    EntryPhase,
    ExpressionLevel,
    InputContract,
    InputKind,
    PracticeResult,
    PracticeSummary,
    ResponseType,
    SceneType,
    SessionCreate,
    SessionEnvelope,
    SessionState,
    utc_now,
)

logger = logging.getLogger(__name__)


class InvalidTurnResponseError(ValueError):
    """The client response does not belong to the current active turn."""


@dataclass(frozen=True)
class ConversationStreamEvent:
    name: Literal["accepted", "progress", "turn"]
    stage: str | None = None
    envelope: SessionEnvelope | None = None
    replayed: bool = False


class ConversationService:
    def __init__(
        self,
        repository: Repository,
        engine: ConversationEngine,
        *,
        v2_engine: DialogueV2Engine | None = None,
        life_v2_engine: DialogueV2LifeEngine | None = None,
        runtime_contract_version: DialogueRuntimeContractVersion = (
            DialogueRuntimeContractVersion.LEGACY_V1
        ),
        dialogue_v2_canary_percent: int = 0,
        dialogue_v2_canary_salt: str = "mormi-dialogue-v2-default",
    ) -> None:
        if not 0 <= dialogue_v2_canary_percent <= 100:
            raise ValueError("dialogue V2 canary percent must be between 0 and 100")
        self.repository = repository
        self.engine = engine
        self.v2_engine = v2_engine
        self.life_v2_engine = life_v2_engine
        self.runtime_contract_version = runtime_contract_version
        self.dialogue_v2_canary_percent = dialogue_v2_canary_percent
        self.dialogue_v2_canary_salt = dialogue_v2_canary_salt

    @property
    def dialogue_runtime_capabilities(
        self,
    ) -> tuple[DialogueRuntimeContractVersion, ...]:
        versions = [DialogueRuntimeContractVersion.LEGACY_V1]
        if self.v2_engine is not None or self.life_v2_engine is not None:
            versions.append(DialogueRuntimeContractVersion.VERDICT_V1)
        return tuple(versions)

    @property
    def dialogue_snapshot_reader_capabilities(self) -> tuple[str, ...]:
        """Advertise the exact pinned-copy formats this process can resume."""

        if self.v2_engine is None:
            return ()
        capabilities = [DIALOGUE_V2_SNAPSHOT_READER_CAPABILITY_V2]
        if self.life_v2_engine is not None:
            capabilities.append(DIALOGUE_V3_SNAPSHOT_READER_CAPABILITY_V1)
        return tuple(capabilities)

    def _engine_for_state(
        self,
        state: SessionState,
    ) -> ConversationEngine | DialogueV2Engine:
        """Use only the runtime version pinned in the persisted conversation."""

        if state.runtime_contract_version is DialogueRuntimeContractVersion.LEGACY_V1:
            return self.engine
        if (
            state.pinned_dialogue_v2 is not None
            and state.pinned_dialogue_scenario_v3 is not None
        ):
            raise RuntimeError("verdict-v1 conversation has ambiguous pinned runtimes")
        if state.pinned_dialogue_v2 is not None:
            if self.v2_engine is None:
                raise RuntimeError(
                    "verdict-v1 conversation cannot run because the V2 engine is unavailable"
                )
            return self.v2_engine
        if state.pinned_dialogue_scenario_v3 is not None:
            if self.life_v2_engine is None:
                raise RuntimeError(
                    "verdict-v1 life conversation cannot run because the life V2 "
                    "engine is unavailable"
                )
            return self.life_v2_engine
        raise RuntimeError("verdict-v1 conversation has no pinned dialogue runtime")

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
                request.scene,
                request.scenario_id,
                request.conversation_round,
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
            # Access the deprecated rolling-deploy field without emitting a
            # warning for every request that does not use it.
            park_context=request.__dict__.get("park_context"),
            curriculum_session_id=curriculum_session_id,
            skill_id=practice_summary.skill_id if practice_summary else None,
            practice_result_id=request.practice_result_id,
        )
        runtime_selection = select_dialogue_runtime(
            configured_version=self.runtime_contract_version,
            canary_percent=self.dialogue_v2_canary_percent,
            canary_salt=self.dialogue_v2_canary_salt,
            scene=request.scene,
            scenario_id=request.scenario_id,
            curriculum_session_id=curriculum_session_id,
            learner_id=request.learner_id,
            learning_session_id=request.learning_session_id,
            conversation_round=request.conversation_round,
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
        started_at = utc_now()
        state = SessionState(
            learner_id=request.learner_id,
            learning_session_id=request.learning_session_id,
            conversation_round=request.conversation_round,
            scene=request.scene,
            scenario_id=request.scenario_id,
            task_ids=scenario.task_ids,
            scenario_data=scenario_data,
            dictionary_catalog_version=DICTIONARY_CATALOG.catalog_version,
            dictionary_snapshots={
                task_id: get_dictionary_card_by_id(
                    get_task(task_id, scenario_data).dictionary_card_id
                ).model_copy(deep=True)
                for task_id in scenario.task_ids
            },
            task_start_levels=task_start_levels,
            expression_level=start_level,
            task_start_level=start_level,
            # Policy v3 removes deliberate wrong-guess openings for all new
            # sessions.  Legacy snapshots retain their persisted entry phase,
            # but a newly created conversation always starts from the genuine
            # L4-H0 help request.
            dialogue_policy_version=3,
            runtime_contract_version=runtime_selection.version,
            entry_phase=EntryPhase.RESOLVED,
            raw_storage_enabled=request.conversation_storage_consent,
            retention_policy=request.retention_policy,
            raw_retention_until=request.retention_policy.expires_at(started_at),
            created_at=started_at,
            updated_at=started_at,
        )
        if runtime_selection.version is DialogueRuntimeContractVersion.VERDICT_V1:
            if request.scene is SceneType.HOME_TEACH:
                if self.v2_engine is None:
                    raise RuntimeError(
                        "verdict-v1 conversation was selected but the V2 engine is "
                        "unavailable"
                    )
                if curriculum_session_id is None:  # pragma: no cover - selector guarantees it
                    raise RuntimeError("verdict-v1 selection has no curriculum session")
                turn = await self.v2_engine.initialize_state(
                    state,
                    curriculum_session_id=curriculum_session_id,
                    selector_reason=runtime_selection.reason,
                    canary_bucket=runtime_selection.bucket,
                )
            else:
                if self.life_v2_engine is None:
                    raise RuntimeError(
                        "verdict-v1 life conversation was selected but the life V2 "
                        "engine is unavailable"
                    )
                if request.scene is SceneType.CAFE:
                    scenario_pack = materialize_cafe_scenario_v2(
                        request.scenario_id,
                        scenario_data,
                    )
                elif request.scene is SceneType.AMUSEMENT_PARK:
                    scenario_pack = materialize_amusement_scenario_v2(
                        request.scenario_id,
                        scenario_data,
                        task_start_levels=task_start_levels,
                    )
                else:  # pragma: no cover - selector owns native scene eligibility
                    raise RuntimeError("verdict-v1 life selection has an invalid scene")
                turn = await self.life_v2_engine.initialize_scenario_state(
                    state,
                    scenario_pack,
                    selector_reason=runtime_selection.reason,
                    canary_bucket=runtime_selection.bucket,
                )
        else:
            turn = self.engine.initial_turn(state)
        state.current_turn_id = turn.turn_id
        persisted_conversation_id = await self.repository.create_conversation(state, turn)
        if persisted_conversation_id != state.conversation_id:
            return await self.snapshot(persisted_conversation_id)
        return SessionEnvelope(conversation_id=state.conversation_id, turn=turn)

    async def respond(
        self,
        conversation_id: str,
        response: ChildResponse,
    ) -> SessionEnvelope:
        final: SessionEnvelope | None = None
        async for event in self.respond_stream(conversation_id, response):
            if event.envelope is not None:
                final = event.envelope
        if final is None:  # pragma: no cover - stream always ends with a turn
            raise RuntimeError("Conversation response produced no final turn")
        return final

    async def respond_stream(
        self,
        conversation_id: str,
        response: ChildResponse,
    ) -> AsyncGenerator[ConversationStreamEvent, None]:
        """Run one canonical response path while exposing non-sensitive progress."""

        # Every llm_call logged below this point carries this turn's ids.
        token = turn_scope.set(TurnScope(conversation_id, response.turn_id))
        try:
            async with aclosing(self._respond_stream_scoped(conversation_id, response)) as events:
                async for event in events:
                    yield event
        finally:
            turn_scope.reset(token)

    async def _respond_stream_scoped(
        self,
        conversation_id: str,
        response: ChildResponse,
    ) -> AsyncGenerator[ConversationStreamEvent, None]:
        started = time.perf_counter()
        response_id = str(response.response_id)
        prior = await self.repository.response_exists(conversation_id, response_id)
        if prior:
            yield ConversationStreamEvent(
                name="turn",
                envelope=SessionEnvelope(conversation_id=conversation_id, turn=prior),
                replayed=True,
            )
            return

        state = await self.repository.get_state(conversation_id)
        runtime_engine = self._engine_for_state(state)
        active_turn = runtime_engine.ensure_task_anchor(
            state,
            await self.repository.active_turn(state),
        )
        if active_turn.turn_id != response.turn_id:
            raise InvalidTurnResponseError("turn_id is stale; use the latest turn")
        if not response_matches_input(active_turn.input.kind, response.type):
            raise InvalidTurnResponseError(
                f"response type {response.type.value} does not match "
                f"input kind {active_turn.input.kind.value}"
            )
        validate_response_payload(
            active_turn.input,
            response,
            exact_joint=(
                state.runtime_contract_version
                is DialogueRuntimeContractVersion.VERDICT_V1
            ),
        )
        yield ConversationStreamEvent(name="accepted", stage="accepted")
        previous_task = get_task(state.current_task_id, state.scenario_data)
        recent_dialogue = await self.repository.recent_dialogue_context(
            conversation_id,
            limit=6,
        )
        result: EngineTurnResult | None = None
        engine_events = cast(
            AsyncGenerator[EngineProgress | EngineTurnResult, None],
            runtime_engine.run_turn_stream(
                state, response, active_turn.mormi.text, recent_dialogue=recent_dialogue,
            ),
        )
        # Both engines return async generators. Explicitly close the nested
        # iterator before releasing the request scope, including on disconnect.
        async with aclosing(engine_events):
            async for engine_event in engine_events:
                if isinstance(engine_event, EngineProgress):
                    yield ConversationStreamEvent(
                        name="progress",
                        stage=engine_event.stage,
                    )
                else:
                    result = engine_event
        if result is None:  # pragma: no cover - graph always reaches END
            raise RuntimeError("Conversation graph produced no result")
        next_state, analysis, next_turn = result.state, result.analysis, result.turn
        try:
            await self.repository.commit_turn(
                previous_state=state,
                next_state=next_state,
                response=response,
                analysis=analysis,
                classifier_response_category=result.classifier_response_category,
                next_turn=next_turn,
                previous_question=active_turn.mormi.text,
                note=next_turn.note_update,
                runtime=result.runtime,
                accepted_claims=result.accepted_claims,
            )
        except DuplicateResponseError as error:
            prior = await self.repository.response_exists(conversation_id, response_id)
            if prior:
                yield ConversationStreamEvent(
                    name="turn",
                    envelope=SessionEnvelope(conversation_id=conversation_id, turn=prior),
                    replayed=True,
                )
                return
            # A uniqueness conflict without a replayable result means the DB
            # is inconsistent (for example, an orphaned observation from a
            # manual operation). Do not leak an unclassified 500 or pretend
            # the child's response succeeded.
            raise PersistenceError("duplicate_response_result_missing") from error

        if next_turn.note_update:
            profile = await self.repository.get_profile(state.learner_id)
            evidence_state = state.model_copy(deep=True)
            evidence_state.verified_slots = {
                slot_id: previous_task.slots[slot_id].expected
                for slot_id in previous_task.required_slots
            }
            profile = update_skill_profile(profile, evidence_state, previous_task)
            await self.repository.save_profile(profile)

        turn_runtime = result.runtime
        logger.info(
            "turn conversation_id=%s turn_id=%s duration_ms=%d "
            "runtime_contract=%s understanding_source=%s evidence_guard=%s "
            "new_progress=%s verified_facts=%d verified_relations=%d "
            "speaker_source=%s stable_copy_status=%s content_pack=%s "
            "verifier_status=%s fallback_reason=%s",
            conversation_id,
            response.turn_id,
            int((time.perf_counter() - started) * 1000),
            getattr(turn_runtime, "runtime_contract_version", None),
            getattr(turn_runtime, "understanding_source", None),
            getattr(turn_runtime, "evidence_guard_status", None),
            getattr(turn_runtime, "new_progress", False),
            len(getattr(turn_runtime, "newly_verified_fact_ids", ())),
            len(getattr(turn_runtime, "newly_verified_relation_ids", ())),
            getattr(turn_runtime, "speaker_source", None),
            getattr(turn_runtime, "stable_copy_status", None),
            getattr(turn_runtime, "content_pack_id", None),
            getattr(turn_runtime, "verifier_status", None),
            getattr(turn_runtime, "fallback_reason", None),
        )
        yield ConversationStreamEvent(
            name="turn",
            envelope=SessionEnvelope(conversation_id=conversation_id, turn=next_turn),
        )

    async def snapshot(self, conversation_id: str) -> SessionEnvelope:
        state = await self.repository.get_state(conversation_id)
        runtime_engine = self._engine_for_state(state)
        turn = runtime_engine.ensure_task_anchor(
            state,
            await self.repository.active_turn(state),
        )
        return SessionEnvelope(conversation_id=conversation_id, turn=turn)

    async def dictionary_card(self, conversation_id: str) -> DictionaryCardEnvelope:
        """Return the reviewed card snapshot pinned when the conversation began."""

        state = await self.repository.get_state(conversation_id)
        try:
            card = state.dictionary_snapshots[state.current_task_id]
        except KeyError as error:
            # A legacy conversation may predate dictionary snapshots. Refuse
            # today's card rather than changing trusted content mid-session.
            raise ValueError("conversation has no pinned dictionary snapshot") from error
        return dictionary_card_envelope(
            card,
            catalog_version=state.dictionary_catalog_version,
        )


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


def _same_json_value(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercion."""

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


def validate_response_payload(
    input_contract: InputContract,
    response: ChildResponse,
    *,
    exact_joint: bool = False,
) -> None:
    """Reject structured answers that do not belong to the active turn.

    Choice IDs are server-authored opaque identifiers.  Accepting an unknown
    ID, or more than one ID for a single-choice teaching turn, would let a
    stale/malformed client bypass the reviewed choice mapping.
    """

    if (
        exact_joint
        and input_contract.kind is InputKind.JOINT
        and response.type is ResponseType.ACTION
    ):
        expected = input_contract.config.get("completion_values")
        if not isinstance(expected, dict):
            raise InvalidTurnResponseError(
                "V2 joint input has no valid completion_values"
            )
        if not _same_json_value(response.values, expected):
            raise InvalidTurnResponseError(
                "V2 joint response values must exactly match the active completion_values"
            )
        return
    if response.type not in {ResponseType.CHOICE, ResponseType.FILL}:
        return
    if len(response.choice_ids) != 1:
        raise InvalidTurnResponseError(
            "structured teaching response requires exactly one choice_id"
        )
    allowed_ids = {choice.id for choice in input_contract.choices if not choice.disabled}
    if response.choice_ids[0] not in allowed_ids:
        raise InvalidTurnResponseError("choice_id does not belong to the active turn")
