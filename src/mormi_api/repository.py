from __future__ import annotations

from datetime import timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from .db import (
    ConversationRecord,
    Database,
    LearnerProfileRecord,
    NoteRecord,
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
    SessionState,
    TurnContract,
    UtteranceAnalysis,
    utc_now,
)
from .security import TextCipher


class ConversationNotFoundError(KeyError):
    pass


class StaleConversationError(RuntimeError):
    pass


class DuplicateResponseError(RuntimeError):
    pass


class Repository:
    def __init__(
        self,
        database: Database,
        cipher: TextCipher,
        *,
        idempotency_retention_days: int = 30,
    ) -> None:
        self.database = database
        self.cipher = cipher
        self.idempotency_retention_days = idempotency_retention_days

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

    async def create_conversation(self, state: SessionState, turn: TurnContract) -> None:
        async with self.database.sessions() as db:
            db.add(
                ConversationRecord(
                    conversation_id=state.conversation_id,
                    learner_id=state.learner_id,
                    learning_session_id=state.learning_session_id,
                    scene=state.scene.value,
                    scenario_id=state.scenario_id,
                    state_json=state.model_dump(mode="json"),
                    state_version=state.state_version,
                    status=state.status.value,
                    raw_retention_until=state.raw_retention_until,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                )
            )
            db.add(self._turn_record(state, turn))
            await db.commit()

    async def get_state(self, conversation_id: str) -> SessionState:
        async with self.database.sessions() as db:
            record = await db.get(ConversationRecord, conversation_id)
            if not record:
                raise ConversationNotFoundError(conversation_id)
            return SessionState.model_validate(record.state_json)

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
            current_turn.response_expires_at = utc_now() + timedelta(
                days=self.idempotency_retention_days
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
            conversation_record.state_json = next_state.model_dump(mode="json")
            conversation_record.state_version = next_state.state_version
            conversation_record.status = next_state.status.value
            conversation_record.updated_at = next_state.updated_at
            db.add(self._turn_record(next_state, next_turn))

            if note:
                db.add(
                    NoteRecord(
                        note_id=note.note_id,
                        conversation_id=next_state.conversation_id,
                        learner_id=next_state.learner_id,
                        skill_id=note.skill_id,
                        text=note.text,
                        attribution=note.attribution.value,
                        evidence=note.evidence.value,
                        attribution_label=note.attribution_label,
                    )
                )
            try:
                await db.commit()
            except IntegrityError as error:
                await db.rollback()
                raise DuplicateResponseError(response_id) from error

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
