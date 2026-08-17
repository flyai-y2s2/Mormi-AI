from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .content import get_task
from .db import (
    ConversationRecord,
    Database,
    DataMigrationRecord,
    DialogueClaimRecord,
    DialogueTaskOutcomeRecord,
    DialogueTurnObservationRecord,
    LearnerProfileRecord,
    NoteEvidenceLinkRecord,
    NoteRecord,
    OutboxEventRecord,
    PracticeResultRecord,
    TurnRecord,
)
from .schemas import (
    ChildResponse,
    LearnerProfile,
    NoteAttribution,
    NoteEvidence,
    NoteUpdate,
    PracticeResult,
    ResponseCategory,
    RetentionPolicy,
    SessionState,
    SpeakerRuntimeAudit,
    TurnContract,
    UtteranceAnalysis,
    new_id,
    utc_now,
)
from .security import TextCipher


class ConversationNotFoundError(KeyError):
    pass


class StaleConversationError(RuntimeError):
    pass


class DuplicateResponseError(RuntimeError):
    pass


class PersistenceError(RuntimeError):
    """A turn could not be stored for a reason other than an idempotent replay."""


_DUPLICATE_RESPONSE_CONSTRAINTS = {
    "uq_conversation_response",
    "uq_observation_conversation_response",
    "uq_observation_source_turn",
}


def _is_duplicate_response_integrity_error(error: IntegrityError) -> bool:
    """Return true only for constraints that represent the same answered turn.

    SQLAlchemy wraps PostgreSQL/asyncpg errors differently across driver
    versions, so inspect the exception chain as well as SQLite's stable message.
    Other integrity failures must not masquerade as successful idempotent
    replays.
    """

    current: BaseException | None = error.orig
    seen: set[int] = set()
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        constraint_name = getattr(current, "constraint_name", None)
        if constraint_name in _DUPLICATE_RESPONSE_CONSTRAINTS:
            return True
        current = current.__cause__ or current.__context__

    message = " ".join(messages)
    sqlite_unique_fragments = (
        "UNIQUE constraint failed: turns.conversation_id, turns.response_id",
        "UNIQUE constraint failed: dialogue_turn_observations.conversation_id, "
        "dialogue_turn_observations.response_id",
        "UNIQUE constraint failed: dialogue_turn_observations.source_turn_id",
    )
    return any(fragment in message for fragment in sqlite_unique_fragments) or any(
        constraint in message for constraint in _DUPLICATE_RESPONSE_CONSTRAINTS
    )


class Repository:
    PERMANENT_STORAGE_MIGRATION = "2026-08-permanent-raw-storage"
    _STATE_EVIDENCE_ENCRYPTED = "_child_note_evidence_encrypted"

    def __init__(
        self,
        database: Database,
        cipher: TextCipher,
        *,
        idempotency_retention_days: int = 30,
        classifier_model: str = "not_collected",
        speaker_model: str = "not_collected",
    ) -> None:
        self.database = database
        self.cipher = cipher
        self.idempotency_retention_days = idempotency_retention_days
        self.classifier_model = classifier_model
        self.speaker_model = speaker_model

    async def save_practice_summary(self, summary: PracticeResult) -> None:
        success_rate = summary.success_rate or 0
        async with self.database.sessions() as db:
            existing = await db.get(PracticeResultRecord, summary.practice_result_id)
            if existing:
                return
            db.add(
                PracticeResultRecord(
                    practice_result_id=summary.practice_result_id,
                    learner_id=summary.learner_id,
                    skill_id=summary.skill_id,
                    summary_json=summary.model_dump(mode="json"),
                    success_rate=success_rate,
                )
            )
            await db.commit()

    async def get_practice_summary(self, practice_result_id: str) -> PracticeResult | None:
        async with self.database.sessions() as db:
            record = await db.get(PracticeResultRecord, practice_result_id)
            return PracticeResult.model_validate(record.summary_json) if record else None

    async def conversation_id_for_learning_session(
        self,
        learner_id: int,
        learning_session_id: str,
    ) -> str | None:
        """Return the already-created home dialogue after a network retry."""

        async with self.database.sessions() as db:
            statement = (
                select(ConversationRecord.conversation_id)
                .where(
                    ConversationRecord.learner_id == learner_id,
                    ConversationRecord.learning_session_id == learning_session_id,
                )
                .order_by(ConversationRecord.created_at.desc())
                .limit(1)
            )
            return (await db.execute(statement)).scalar_one_or_none()

    async def create_conversation(self, state: SessionState, turn: TurnContract) -> None:
        async with self.database.sessions() as db:
            conversation = ConversationRecord(
                conversation_id=state.conversation_id,
                learner_id=state.learner_id,
                learning_session_id=state.learning_session_id,
                scene=state.scene.value,
                scenario_id=state.scenario_id,
                state_json=self._dump_state(state),
                state_version=state.state_version,
                status=state.status.value,
                raw_retention_until=state.raw_retention_until,
                created_at=state.created_at,
                updated_at=state.updated_at,
            )
            db.add(conversation)
            # No ORM relationship is declared between these persistence
            # records. Flush the FK parent explicitly instead of relying on
            # incidental unit-of-work insertion order.
            await db.flush([conversation])
            db.add(self._turn_record(state, turn))
            await db.commit()

    async def get_state(self, conversation_id: str) -> SessionState:
        async with self.database.sessions() as db:
            record = await db.get(ConversationRecord, conversation_id)
            if not record:
                raise ConversationNotFoundError(conversation_id)
            return self._load_state(record.state_json)

    async def get_profile(self, learner_id: int) -> LearnerProfile:
        async with self.database.sessions() as db:
            record = await db.get(LearnerProfileRecord, learner_id)
            if not record:
                return LearnerProfile(learner_id=learner_id)
            return LearnerProfile.model_validate(record.profile_json)

    async def save_profile(self, profile: LearnerProfile) -> None:
        profile.updated_at = utc_now()
        async with self.database.sessions() as db:
            record = await db.get(LearnerProfileRecord, profile.learner_id)
            if record:
                record.profile_json = profile.model_dump(mode="json")
                record.updated_at = profile.updated_at
            else:
                db.add(
                    LearnerProfileRecord(
                        learner_id=profile.learner_id,
                        profile_json=profile.model_dump(mode="json"),
                        updated_at=profile.updated_at,
                    )
                )
            await db.commit()

    async def response_exists(
        self,
        conversation_id: str,
        response_id: str,
    ) -> TurnContract | None:
        async with self.database.sessions() as db:
            statement = select(TurnRecord).where(
                TurnRecord.conversation_id == conversation_id,
                TurnRecord.response_id == response_id,
                or_(
                    TurnRecord.response_expires_at.is_(None),
                    TurnRecord.response_expires_at > utc_now(),
                ),
            )
            record = (await db.execute(statement)).scalar_one_or_none()
            if not record:
                return None
            if not record.result_turn_id:
                return None
            result = (
                await db.execute(
                    select(TurnRecord).where(TurnRecord.turn_id == record.result_turn_id)
                )
            ).scalar_one_or_none()
            return self._load_turn(result) if result else None

    async def active_turn(self, state: SessionState) -> TurnContract:
        if not state.current_turn_id:
            raise StaleConversationError("Conversation has no active turn")
        async with self.database.sessions() as db:
            statement = select(TurnRecord).where(TurnRecord.turn_id == state.current_turn_id)
            record = (await db.execute(statement)).scalar_one_or_none()
            if not record or record.conversation_id != state.conversation_id:
                raise StaleConversationError("Active turn was not found")
            return self._load_turn(record)

    async def commit_turn(
        self,
        *,
        previous_state: SessionState,
        next_state: SessionState,
        response: ChildResponse,
        analysis: UtteranceAnalysis,
        next_turn: TurnContract,
        previous_question: str,
        note: NoteUpdate | None,
        runtime: SpeakerRuntimeAudit,
        accepted_claims: Mapping[str, object],
    ) -> None:
        async with self.database.sessions() as db:
            conversation_record = await db.get(
                ConversationRecord,
                previous_state.conversation_id,
                with_for_update=True,
            )
            if not conversation_record:
                raise ConversationNotFoundError(previous_state.conversation_id)
            if conversation_record.state_version != previous_state.state_version:
                raise StaleConversationError(previous_state.conversation_id)

            statement = select(TurnRecord).where(TurnRecord.turn_id == response.turn_id)
            current_turn = (await db.execute(statement)).scalar_one_or_none()
            if not current_turn or current_turn.conversation_id != previous_state.conversation_id:
                raise StaleConversationError("Response does not match the active turn")
            response_id = str(response.response_id)
            if current_turn.response_id:
                if current_turn.response_id == response_id:
                    raise DuplicateResponseError(response_id)
                raise StaleConversationError("Turn was already answered")

            raw_response = response.text or self._structured_response_text(response)
            current_turn.response_id = response_id
            current_turn.response_expires_at = (
                None
                if previous_state.retention_policy is RetentionPolicy.PERMANENT
                else utc_now() + timedelta(days=self.idempotency_retention_days)
            )
            current_turn.result_turn_id = next_turn.turn_id
            current_turn.response_type = response.type.value
            current_turn.response_raw_encrypted = (
                self.cipher.encrypt(raw_response) if previous_state.raw_storage_enabled else None
            )
            current_turn.response_structured = response.model_dump(mode="json", exclude={"text"})
            current_turn.safety_category = analysis.safety_category.value
            current_turn.response_category = analysis.response_category.value

            next_state.updated_at = utc_now()
            conversation_record.state_json = self._dump_state(next_state)
            conversation_record.state_version = next_state.state_version
            conversation_record.status = next_state.status.value
            conversation_record.updated_at = next_state.updated_at
            db.add(self._turn_record(next_state, next_turn))

            try:
                observation = self._observation_record(
                    previous_state=previous_state,
                    next_state=next_state,
                    current_turn=current_turn,
                    response=response,
                    analysis=analysis,
                    next_turn=next_turn,
                    runtime=runtime,
                )
                db.add(observation)
                # Claims reference the observation. SQLAlchemy cannot infer
                # mapper ordering without an ORM relationship, so persist the
                # parent before constructing any child rows.
                await db.flush([observation])

                claim_records = self._claim_records(
                    observation.observation_id,
                    previous_state,
                    analysis,
                    accepted_claims,
                )
                db.add_all(claim_records)
                # Note provenance can span several child turns. Persist the
                # current claims before querying the complete task evidence.
                if claim_records:
                    await db.flush(claim_records)

                if note:
                    note_record = NoteRecord(
                        note_id=note.note_id,
                        conversation_id=next_state.conversation_id,
                        learner_id=next_state.learner_id,
                        skill_id=note.skill_id,
                        text=note.text,
                        attribution=note.attribution.value,
                        evidence=note.evidence.value,
                        attribution_label=note.attribution_label,
                    )
                    db.add(note_record)
                    # The evidence link has two FK parents: the already
                    # flushed observation and this note.
                    await db.flush([note_record])
                    db.add_all(
                        await self._note_evidence_link_records(
                            db,
                            previous_state=previous_state,
                            note=note,
                        )
                    )

                task_outcome = await self._task_outcome_record(
                    db,
                    previous_state=previous_state,
                    next_state=next_state,
                    observation=observation,
                    claims=claim_records,
                    note=note,
                )
                if task_outcome:
                    db.add(task_outcome)

                db.add(self._outbox_record(observation, claim_records, task_outcome))
                await db.commit()
            except IntegrityError as error:
                await db.rollback()
                if _is_duplicate_response_integrity_error(error):
                    raise DuplicateResponseError(response_id) from error
                raise PersistenceError("turn_persistence_failed") from error
            except SQLAlchemyError as error:
                await db.rollback()
                raise PersistenceError("turn_persistence_failed") from error

    def _observation_record(
        self,
        *,
        previous_state: SessionState,
        next_state: SessionState,
        current_turn: TurnRecord,
        response: ChildResponse,
        analysis: UtteranceAnalysis,
        next_turn: TurnContract,
        runtime: SpeakerRuntimeAudit,
    ) -> DialogueTurnObservationRecord:
        input_payload = current_turn.turn_contract.get("input", {})
        input_kind = (
            str(input_payload.get("kind", "not_collected"))
            if isinstance(input_payload, dict)
            else "not_collected"
        )
        current_stage = str(
            current_turn.turn_contract.get("stage_id", "not_collected")
        )
        safe_analysis = analysis.model_dump(
            mode="json",
            exclude={"claims", "grounding_span", "note_candidate"},
        )
        help_card = next_turn.help_card
        return DialogueTurnObservationRecord(
            observation_id=new_id("observation"),
            conversation_id=previous_state.conversation_id,
            learner_id=previous_state.learner_id,
            learning_session_id=previous_state.learning_session_id,
            scene=previous_state.scene.value,
            scenario_id=previous_state.scenario_id,
            task_id=previous_state.current_task_id,
            stage_id=current_stage,
            task_index=previous_state.task_index,
            subgoal_id=previous_state.subgoal_id,
            source_turn_id=current_turn.turn_id,
            result_turn_id=next_turn.turn_id,
            response_id=str(response.response_id),
            response_type=response.type.value,
            input_kind=input_kind,
            response_category=analysis.response_category.value,
            difficulty_class=analysis.difficulty_class.value,
            concept_result=self._concept_result(analysis.response_category),
            safety_category=analysis.safety_category.value,
            misconception_tag=analysis.misconception_tag,
            bottleneck=analysis.bottleneck or "unknown",
            classifier_confidence=analysis.confidence,
            expression_before=previous_state.expression_level.value,
            expression_after=next_state.expression_level.value,
            hint_before=previous_state.hint_level.value,
            hint_after=next_state.hint_level.value,
            transition_reason=runtime.dialogue_act,
            dialogue_act=runtime.dialogue_act,
            help_card_shown=help_card is not None,
            help_card_level=help_card.level.value if help_card else None,
            help_card_auto_open=bool(help_card and help_card.auto_open),
            speaker_source=runtime.speaker_source,
            verifier_status=runtime.verifier_status,
            fallback_reason=runtime.fallback_reason,
            completion_outcome=(
                next_state.completion_outcome.value
                if next_state.completion_outcome is not None
                else None
            ),
            record_origin="live",
            analysis_json=safe_analysis,
            runtime_json=runtime.model_dump(mode="json"),
            versions_json={
                "observation_schema": 1,
                "dialogue_policy": previous_state.dialogue_policy_version,
                "dictionary_catalog": previous_state.dictionary_catalog_version,
                "content": (
                    previous_state.dictionary_snapshots[previous_state.current_task_id]
                    .content_version
                    if previous_state.current_task_id in previous_state.dictionary_snapshots
                    else "not_collected"
                ),
                "classifier_model": self.classifier_model,
                "speaker_model": self.speaker_model,
            },
        )

    def _claim_records(
        self,
        observation_id: str,
        previous_state: SessionState,
        analysis: UtteranceAnalysis,
        accepted_claims: Mapping[str, object],
    ) -> list[DialogueClaimRecord]:
        task = get_task(previous_state.current_task_id, previous_state.scenario_data)
        records: list[DialogueClaimRecord] = []
        for claim in analysis.claims:
            slot = task.slots.get(claim.slot_id)
            accepted = bool(
                slot is not None
                and claim.slot_id in accepted_claims
                and accepted_claims[claim.slot_id] == claim.value
            )
            advanced_state = bool(
                accepted and previous_state.verified_slots.get(claim.slot_id) != claim.value
            )
            records.append(
                DialogueClaimRecord(
                    observation_id=observation_id,
                    slot_id=claim.slot_id,
                    semantic_role=slot.semantic_role if slot else "not_collected",
                    # Only reviewed canonical values are analytics-safe. A
                    # rejected model claim can contain arbitrary child text;
                    # its evidence remains available only in encrypted form.
                    value_json=claim.value if accepted else None,
                    factual=claim.factual,
                    validation_status="verified" if accepted else "rejected",
                    evidence_span_encrypted=(
                        self.cipher.encrypt(claim.evidence_span)
                        if previous_state.raw_storage_enabled and claim.evidence_span
                        else None
                    ),
                    newly_verified=advanced_state,
                )
            )
        return records

    async def _note_evidence_link_records(
        self,
        db: AsyncSession,
        *,
        previous_state: SessionState,
        note: NoteUpdate,
    ) -> list[NoteEvidenceLinkRecord]:
        """Link a note to every verified task turn that supplied its slots.

        A note may combine an answer from one turn with a method from a later
        turn. Linking only the completion turn loses that provenance and can
        make a teacher-facing report cite the wrong child response.
        """

        task = get_task(previous_state.current_task_id, previous_state.scenario_data)
        note_slots = set(task.effective_note_slots)
        if not note_slots:
            return []
        statement = (
            select(
                DialogueClaimRecord.observation_id,
                DialogueClaimRecord.slot_id,
            )
            .join(
                DialogueTurnObservationRecord,
                DialogueTurnObservationRecord.observation_id
                == DialogueClaimRecord.observation_id,
            )
            .where(
                DialogueTurnObservationRecord.conversation_id
                == previous_state.conversation_id,
                DialogueTurnObservationRecord.task_index == previous_state.task_index,
                DialogueClaimRecord.validation_status == "verified",
                DialogueClaimRecord.newly_verified.is_(True),
                DialogueClaimRecord.slot_id.in_(note_slots),
            )
        )
        slots_by_observation: dict[str, set[str]] = {}
        for observation_id, slot_id in (await db.execute(statement)).all():
            slots_by_observation.setdefault(observation_id, set()).add(slot_id)
        if not slots_by_observation:
            raise PersistenceError("note_evidence_missing")
        return [
            NoteEvidenceLinkRecord(
                note_id=note.note_id,
                observation_id=observation_id,
                source_slot_ids_json=sorted(slot_ids),
            )
            for observation_id, slot_ids in sorted(slots_by_observation.items())
        ]

    async def _task_outcome_record(
        self,
        db: AsyncSession,
        *,
        previous_state: SessionState,
        next_state: SessionState,
        observation: DialogueTurnObservationRecord,
        claims: list[DialogueClaimRecord],
        note: NoteUpdate | None,
    ) -> DialogueTaskOutcomeRecord | None:
        task_finished = (
            next_state.status.value == "completed"
            or next_state.task_index != previous_state.task_index
        )
        if not task_finished:
            return None
        statement = select(DialogueTurnObservationRecord).where(
            DialogueTurnObservationRecord.conversation_id
            == previous_state.conversation_id,
            DialogueTurnObservationRecord.task_index == previous_state.task_index,
        ).order_by(DialogueTurnObservationRecord.created_at.asc())
        prior = list((await db.execute(statement)).scalars())
        evidence_ids = [record.observation_id for record in prior]
        if observation.observation_id not in evidence_ids:
            evidence_ids.append(observation.observation_id)
        evidence_records = list(prior)
        if all(
            record.observation_id != observation.observation_id
            for record in evidence_records
        ):
            evidence_records.append(observation)
        bottlenecks = [
            {
                "observation_id": record.observation_id,
                "value": record.bottleneck,
                "confidence": record.classifier_confidence,
            }
            for record in evidence_records
            if record.bottleneck not in {"", "unknown", "not_collected"}
        ]
        if note is not None:
            completion_outcome = (
                "taught"
                if note.attribution is NoteAttribution.CHILD
                else "supported"
            )
        elif next_state.completion_outcome is not None:
            completion_outcome = next_state.completion_outcome.value
        else:
            # A missing note is not evidence of independent teaching. Some
            # tasks intentionally have no note policy, so keep the result
            # explicitly unknown instead of inflating it to ``taught``.
            completion_outcome = "not_collected"
        verified_slots = dict(previous_state.verified_slots)
        if next_state.task_index == previous_state.task_index:
            verified_slots.update(next_state.verified_slots)
        else:
            # The next task starts with a fresh ``verified_slots`` mapping.
            # Reconstruct only from claims that already passed the same
            # deterministic acceptance gate used by persistence and outbox;
            # raw classifier claims must never become report evidence merely
            # because a task transition happened.
            for claim in claims:
                if claim.validation_status == "verified" and claim.value_json is not None:
                    verified_slots[claim.slot_id] = claim.value_json
        return DialogueTaskOutcomeRecord(
            outcome_id=new_id("task_outcome"),
            conversation_id=previous_state.conversation_id,
            learner_id=previous_state.learner_id,
            learning_session_id=previous_state.learning_session_id,
            scene=previous_state.scene.value,
            scenario_id=previous_state.scenario_id,
            task_id=previous_state.current_task_id,
            task_index=previous_state.task_index,
            start_expression_level=(
                previous_state.task_start_level or previous_state.expression_level
            ).value,
            end_expression_level=previous_state.expression_level.value,
            start_hint_level=(prior[0] if prior else observation).hint_before,
            max_hint_level=max(
                previous_state.task_max_hint.value,
                observation.hint_after,
            ),
            completion_outcome=completion_outcome,
            verified_slots_json=verified_slots,
            bottleneck_candidates_json=bottlenecks,
            evidence_observation_ids_json=evidence_ids,
            note_id=note.note_id if note else None,
            record_origin="live",
        )

    @staticmethod
    def _concept_result(category: ResponseCategory) -> str:
        if category in {ResponseCategory.CORRECT_FULL, ResponseCategory.SELF_CORRECTION}:
            return "correct_full"
        if category is ResponseCategory.CORRECT_PARTIAL:
            return "correct_partial"
        if category in {
            ResponseCategory.CONCEPTUAL_ERROR,
            ResponseCategory.CONCEPTUAL_BLOCK,
        }:
            return "incorrect"
        return "not_assessed"

    @staticmethod
    def _outbox_record(
        observation: DialogueTurnObservationRecord,
        claims: list[DialogueClaimRecord],
        task_outcome: DialogueTaskOutcomeRecord | None,
    ) -> OutboxEventRecord:
        payload: dict[str, object] = {
            "schema_version": 1,
            "observation_id": observation.observation_id,
            "conversation_id": observation.conversation_id,
            "learner_id": observation.learner_id,
            "learning_session_id": observation.learning_session_id,
            "scene": observation.scene,
            "scenario_id": observation.scenario_id,
            "task_id": observation.task_id,
            "task_index": observation.task_index,
            "source_turn_id": observation.source_turn_id,
            "response_id": observation.response_id,
            "response_type": observation.response_type,
            "response_category": observation.response_category,
            "difficulty_class": observation.difficulty_class,
            "concept_result": observation.concept_result,
            "safety_category": observation.safety_category,
            "bottleneck": observation.bottleneck,
            "classifier_confidence": observation.classifier_confidence,
            "expression_before": observation.expression_before,
            "expression_after": observation.expression_after,
            "hint_before": observation.hint_before,
            "hint_after": observation.hint_after,
            "transition_reason": observation.transition_reason,
            "dialogue_act": observation.dialogue_act,
            "speaker_source": observation.speaker_source,
            "verifier_status": observation.verifier_status,
            "record_origin": observation.record_origin,
            "versions": observation.versions_json,
            "claims": [
                {
                    "slot_id": claim.slot_id,
                    "semantic_role": claim.semantic_role,
                    "value": claim.value_json,
                    "factual": claim.factual,
                    "validation_status": claim.validation_status,
                    "newly_verified": claim.newly_verified,
                }
                for claim in claims
            ],
        }
        if task_outcome:
            payload["task_outcome"] = {
                "outcome_id": task_outcome.outcome_id,
                "completion_outcome": task_outcome.completion_outcome,
                "max_hint_level": task_outcome.max_hint_level,
                "verified_slots": task_outcome.verified_slots_json,
                "evidence_observation_ids": task_outcome.evidence_observation_ids_json,
                "note_id": task_outcome.note_id,
            }
        return OutboxEventRecord(
            event_id=new_id("event"),
            aggregate_type="dialogue_observation",
            aggregate_id=observation.observation_id,
            event_type="mormi.dialogue.observation.recorded",
            schema_version=1,
            payload_json=payload,
        )

    async def raw_turns(self, conversation_id: str) -> list[dict[str, object]]:
        await self.purge_expired_raw_data()
        async with self.database.sessions() as db:
            conversation = await db.get(ConversationRecord, conversation_id)
            if not conversation:
                raise ConversationNotFoundError(conversation_id)
            statement = (
                select(TurnRecord)
                .where(TurnRecord.conversation_id == conversation_id)
                .order_by(TurnRecord.id.asc())
            )
            records = list((await db.execute(statement)).scalars())
            return [
                {
                    "turn_id": record.turn_id,
                    "question": self.cipher.decrypt(record.mormi_question_encrypted)
                    if record.mormi_question_encrypted
                    else None,
                    "response_id": record.response_id,
                    "response_type": record.response_type,
                    "response": self.cipher.decrypt(record.response_raw_encrypted)
                    if record.response_raw_encrypted
                    else None,
                    "structured": record.response_structured,
                    "safety_category": record.safety_category,
                    "response_category": record.response_category,
                    "expression_level": record.expression_level,
                    "hint_level": record.hint_level,
                    "created_at": record.created_at,
                }
                for record in records
            ]

    async def backfill_historical_observations(self) -> int:
        """Preserve legacy turns as explicitly incomplete observations.

        This migration never asks an LLM to reinterpret old child language and
        never fabricates claims, bottlenecks or confidence. Values that were
        not stored at the time remain ``not_collected``.
        """

        inserted = 0
        async with self.database.sessions() as db:
            answered = list(
                (
                    await db.execute(
                        select(TurnRecord)
                        .where(TurnRecord.response_id.is_not(None))
                        .order_by(TurnRecord.id.asc())
                    )
                ).scalars()
            )
            existing_ids = set(
                (
                    await db.execute(
                        select(DialogueTurnObservationRecord.source_turn_id)
                    )
                ).scalars()
            )
            for record in answered:
                if record.turn_id in existing_ids:
                    continue
                conversation = await db.get(ConversationRecord, record.conversation_id)
                if not conversation:
                    continue
                result = None
                if record.result_turn_id:
                    result = (
                        await db.execute(
                            select(TurnRecord).where(
                                TurnRecord.turn_id == record.result_turn_id
                            )
                        )
                    ).scalar_one_or_none()
                current_contract = dict(record.turn_contract)
                result_contract = dict(result.turn_contract) if result else {}
                current_input = current_contract.get("input", {})
                current_stage = current_contract.get("stage_id", "not_collected")
                task_index = current_contract.get("task_index", 0)
                response_category = record.response_category or "not_collected"
                try:
                    category = ResponseCategory(response_category)
                    concept_result = self._concept_result(category)
                except ValueError:
                    concept_result = "not_collected"
                help_payload = result_contract.get("help_card")
                help_card = help_payload if isinstance(help_payload, dict) else None
                observation = DialogueTurnObservationRecord(
                    observation_id=new_id("observation"),
                    conversation_id=record.conversation_id,
                    learner_id=conversation.learner_id,
                    learning_session_id=conversation.learning_session_id,
                    scene=conversation.scene,
                    scenario_id=conversation.scenario_id,
                    task_id=record.task_id,
                    stage_id=str(current_stage),
                    task_index=int(task_index),
                    subgoal_id="not_collected",
                    source_turn_id=record.turn_id,
                    result_turn_id=record.result_turn_id,
                    response_id=record.response_id,
                    response_type=record.response_type,
                    input_kind=(
                        str(current_input.get("kind", "not_collected"))
                        if isinstance(current_input, dict)
                        else "not_collected"
                    ),
                    response_category=response_category,
                    difficulty_class="not_collected",
                    concept_result=concept_result,
                    safety_category=record.safety_category or "not_collected",
                    misconception_tag=None,
                    bottleneck="not_collected",
                    classifier_confidence=None,
                    expression_before=record.expression_level,
                    expression_after=(
                        result.expression_level if result else record.expression_level
                    ),
                    hint_before=record.hint_level,
                    hint_after=result.hint_level if result else record.hint_level,
                    transition_reason="not_collected",
                    dialogue_act="not_collected",
                    help_card_shown=help_card is not None,
                    help_card_level=(
                        str(help_card.get("level")) if help_card else None
                    ),
                    help_card_auto_open=bool(
                        help_card and help_card.get("auto_open", False)
                    ),
                    speaker_source="not_collected",
                    verifier_status="not_collected",
                    fallback_reason=None,
                    completion_outcome=None,
                    record_origin="historical_backfill",
                    analysis_json={
                        "historical_backfill": True,
                        "missing_fields": [
                            "claims",
                            "difficulty_class",
                            "bottleneck",
                            "confidence",
                            "dialogue_act",
                            "speaker_runtime",
                        ],
                    },
                    runtime_json={"historical_backfill": True},
                    versions_json={
                        "observation_schema": 1,
                        "dialogue_policy": "not_collected",
                        "dictionary_catalog": "not_collected",
                        "content": "not_collected",
                        "classifier_model": "not_collected",
                        "speaker_model": "not_collected",
                    },
                    created_at=record.created_at,
                )
                db.add(observation)
                inserted += 1
            await db.commit()
        return inserted

    async def list_notes(self, learner_id: int) -> list[NoteUpdate]:
        async with self.database.sessions() as db:
            statement = (
                select(NoteRecord)
                .where(NoteRecord.learner_id == learner_id, NoteRecord.active.is_(True))
                .order_by(NoteRecord.created_at.desc())
            )
            records = list((await db.execute(statement)).scalars())
            return [
                NoteUpdate(
                    note_id=record.note_id,
                    skill_id=record.skill_id,
                    text=record.text,
                    attribution=NoteAttribution(record.attribution),
                    evidence=NoteEvidence(record.evidence),
                    attribution_label=record.attribution_label,
                )
                for record in records
            ]

    async def purge_expired_raw_data(self) -> None:
        now = utc_now()
        async with self.database.sessions() as db:
            await db.execute(
                update(TurnRecord)
                .where(
                    TurnRecord.response_expires_at.is_not(None),
                    TurnRecord.response_expires_at <= now,
                )
                .values(
                    response_id=None,
                    response_expires_at=None,
                    result_turn_id=None,
                )
            )
            expired = select(ConversationRecord.conversation_id).where(
                ConversationRecord.raw_retention_until.is_not(None),
                ConversationRecord.raw_retention_until <= now,
            )
            ids = list((await db.execute(expired)).scalars())
            if ids:
                await db.execute(
                    update(TurnRecord)
                    .where(TurnRecord.conversation_id.in_(ids))
                    .values(
                        response_raw_encrypted=None,
                    )
                )
                await db.execute(
                    update(ConversationRecord)
                    .where(ConversationRecord.conversation_id.in_(ids))
                    .values(raw_retention_until=None)
                )
            await db.commit()

    def _dump_state(self, state: SessionState) -> dict[str, object]:
        """Serialize state without leaving child note evidence in plaintext."""

        payload: dict[str, object] = state.model_dump(mode="json")
        evidence = state.child_note_evidence
        if evidence:
            payload["child_note_evidence"] = {
                slot_id: self.cipher.encrypt(text)
                for slot_id, text in evidence.items()
            }
            payload[self._STATE_EVIDENCE_ENCRYPTED] = True
        return payload

    def _load_state(self, payload: dict[str, object]) -> SessionState:
        """Decrypt transient note evidence before giving state to the engine."""

        data = dict(payload)
        encrypted = data.pop(self._STATE_EVIDENCE_ENCRYPTED, False)
        evidence = data.get("child_note_evidence")
        if encrypted and isinstance(evidence, dict):
            data["child_note_evidence"] = {
                str(slot_id): self.cipher.decrypt(str(value))
                for slot_id, value in evidence.items()
            }
        return SessionState.model_validate(data)

    async def migrate_existing_storage_to_permanent(self) -> None:
        """One-time upgrade for conversations created before pilot consent became mandatory."""

        async with self.database.sessions() as db:
            applied = await db.get(DataMigrationRecord, self.PERMANENT_STORAGE_MIGRATION)
            if applied:
                return

            records = list((await db.execute(select(ConversationRecord))).scalars())
            conversation_ids: list[str] = []
            for record in records:
                state_json = dict(record.state_json)
                state_json["raw_storage_enabled"] = True
                state_json["retention_policy"] = RetentionPolicy.PERMANENT.value
                state_json["raw_retention_until"] = None
                record.state_json = state_json
                record.raw_retention_until = None
                conversation_ids.append(record.conversation_id)

            if conversation_ids:
                await db.execute(
                    update(TurnRecord)
                    .where(TurnRecord.conversation_id.in_(conversation_ids))
                    .values(response_expires_at=None)
                )
            db.add(DataMigrationRecord(migration_id=self.PERMANENT_STORAGE_MIGRATION))
            await db.commit()

    def _turn_record(self, state: SessionState, turn: TurnContract) -> TurnRecord:
        return TurnRecord(
            turn_id=turn.turn_id,
            conversation_id=state.conversation_id,
            task_id=state.current_task_id,
            state_version=state.state_version,
            # The generated question is required to recover the latest screen,
            # so it is always kept encrypted. Child raw responses remain
            # consent-gated separately.
            mormi_question_encrypted=self.cipher.encrypt(turn.mormi.text),
            turn_contract=self._stored_turn_contract(turn),
            expression_level=state.expression_level.value,
            hint_level=state.hint_level.value,
        )

    @staticmethod
    def _structured_response_text(response: ChildResponse) -> str:
        if response.choice_ids:
            return ", ".join(response.choice_ids)
        return str(response.values)

    @staticmethod
    def _stored_turn_contract(turn: TurnContract) -> dict[str, object]:
        payload = turn.model_dump(mode="json")
        mormi = payload.get("mormi")
        if isinstance(mormi, dict):
            mormi["text"] = ""
        return payload

    def _load_turn(self, record: TurnRecord) -> TurnContract:
        payload = dict(record.turn_contract)
        mormi = dict(payload.get("mormi", {}))
        mormi["text"] = (
            self.cipher.decrypt(record.mormi_question_encrypted)
            if record.mormi_question_encrypted
            else "대화 보존 기간이 끝났어요."
        )
        payload["mormi"] = mormi
        return TurnContract.model_validate(payload)
