from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from .db import ConversationRecord, TurnRecord
from .schemas import (
    ExpressionLevel,
    HintLevel,
    ReportConversationEvidence,
    ReportEvidenceResponse,
    ReportSummaryRequest,
    ReportSummaryResponse,
    ReportTurnEvidence,
    RetentionPolicy,
    SceneType,
    SessionState,
    utc_now,
)

if TYPE_CHECKING:
    from .repository import Repository


_REPORT_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:%)?")
_REPORT_QUOTE = re.compile(r"[‘'\"“]([^’'\"”]+)[’'\"”]")
_FORBIDDEN_REPORT_LANGUAGE = re.compile(
    r"진단|장애|경계선\s*지능|치료|또래보다|평균보다|상위|하위|백분위|"
    r"ADHD|자폐|질환|정신\s*질환|심리\s*진단",
    re.IGNORECASE,
)


def numeric_tokens(text: str) -> set[str]:
    return {match.replace(",", "") for match in _REPORT_NUMBER_TOKEN.findall(text)}


def _quoted_text(text: str) -> list[str]:
    return [match.strip() for match in _REPORT_QUOTE.findall(text) if match.strip()]


def reject_forbidden_report_language(text: str) -> None:
    if _FORBIDDEN_REPORT_LANGUAGE.search(text):
        raise ValueError("forbidden report language")


def validate_report_summary(
    request: ReportSummaryRequest,
    response: ReportSummaryResponse,
) -> ReportSummaryResponse:
    """Accept only summary wording directly traceable to request facts."""

    facts = {item.evidence_id: item.statement for item in request.facts}
    for narrative in response.narratives():
        if any(ref not in facts for ref in narrative.evidence_refs):
            raise ValueError("unknown report evidence reference")
        grounded = " ".join(facts[ref] for ref in narrative.evidence_refs)
        if not numeric_tokens(narrative.text).issubset(numeric_tokens(grounded)):
            raise ValueError("ungrounded report number")
        if any(quote not in grounded for quote in _quoted_text(narrative.text)):
            raise ValueError("ungrounded report quote")
        reject_forbidden_report_language(narrative.text)
    return response


def _raw_response_is_available(state: SessionState, *, now: datetime) -> bool:
    if not state.raw_storage_enabled:
        return False
    if state.retention_policy is RetentionPolicy.PERMANENT:
        return True
    return (
        state.retention_policy in {RetentionPolicy.DAYS_30, RetentionPolicy.DAYS_90}
        and state.raw_retention_until is not None
        and state.raw_retention_until > now
    )


async def build_report_evidence(
    repository: Repository,
    learner_id: int,
    conversations: list[ConversationRecord],
    turns: list[TurnRecord],
    *,
    include_raw: bool,
) -> ReportEvidenceResponse:
    """Project existing learner-scoped records into the internal report contract."""

    states = {
        conversation.conversation_id: SessionState.model_validate(conversation.state_json)
        for conversation in conversations
    }
    now = utc_now()
    turns_by_conversation: dict[str, list[ReportTurnEvidence]] = {
        conversation.conversation_id: [] for conversation in conversations
    }
    for turn in turns:
        state = states[turn.conversation_id]
        pedagogy = turn.turn_contract.get("pedagogy")
        turns_by_conversation[turn.conversation_id].append(
            ReportTurnEvidence(
                turn_id=turn.turn_id,
                task_id=turn.task_id,
                response=(
                    repository.cipher.decrypt(turn.response_raw_encrypted)
                    if (
                        include_raw
                        and _raw_response_is_available(state, now=now)
                        and turn.response_raw_encrypted
                    )
                    else None
                ),
                response_type=turn.response_type,
                response_category=turn.response_category,
                expression_level=ExpressionLevel(turn.expression_level),
                hint_level=HintLevel(turn.hint_level),
                pedagogy=dict(pedagogy) if isinstance(pedagogy, dict) else None,
                created_at=turn.created_at,
            )
        )

    evidence_conversations: list[ReportConversationEvidence] = []
    for conversation in conversations:
        state = states[conversation.conversation_id]
        evidence_conversations.append(
            ReportConversationEvidence(
                conversation_id=conversation.conversation_id,
                learning_session_id=conversation.learning_session_id,
                scene=SceneType(conversation.scene),
                scenario_id=conversation.scenario_id,
                status=state.status,
                completion_outcome=state.completion_outcome,
                teach_reward_eligible=state.teach_reward_eligible,
                verified_slots=state.verified_slots,
                task_max_hint=state.task_max_hint,
                turns=turns_by_conversation[conversation.conversation_id],
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
        )

    profile = await repository.get_profile(learner_id)
    return ReportEvidenceResponse(
        learner_id=learner_id,
        conversations=evidence_conversations,
        skills=list(profile.skills.values()),
        notes=await repository.list_notes(learner_id),
    )
