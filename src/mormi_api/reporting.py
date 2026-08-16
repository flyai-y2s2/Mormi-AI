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


_REPORT_MAGNITUDES = {"십": 10, "백": 100, "천": 1_000, "만": 10_000, "억": 100_000_000}
_REPORT_UNITS = r"%|퍼센트|원|개|명|회|점|분|시간|시|cm|mm|m|g|kg|칸|묶음|장"
_REPORT_NUMBER_TOKEN = re.compile(
    rf"(?P<number>\d[\d,]*)(?P<magnitude>(?:\s*[십백천만억])*)\s*(?P<unit>{_REPORT_UNITS})?"
)
_REPORT_QUOTE_PAIRS = {
    "‘": "’",
    "“": "”",
    "'": "'",
    '\"': '\"',
    "『": "』",
    "「": "」",
    "《": "》",
}
_REPORT_QUOTE_CLOSERS = frozenset(_REPORT_QUOTE_PAIRS.values())
_REPORT_KOREAN_TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+")
_REPORT_PARTICLE_SUFFIXES = tuple(
    sorted(
        (
            "으로부터",
            "에게서",
            "에서는",
            "입니다",
            "됩니다",
            "합니다",
            "있습니다",
            "없습니다",
            "보입니다",
            "됩니다",
            "으로",
            "에게",
            "에서",
            "부터",
            "까지",
            "보다",
            "처럼",
            "으로",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "의",
            "와",
            "과",
            "도",
            "만",
            "로",
        ),
        key=len,
        reverse=True,
    )
)
_REPORT_ALLOWED_TOKENS = frozenset(
    {
        "최근",
        "현재",
        "이번",
        "다음",
        "활동",
        "보고",
        "요약하면",
        "전반적으로",
        "그리고",
        "또한",
        "다만",
        "따라서",
        "해당",
        "부분",
        "내용",
        "결과",
        "입니다",
        "됩니다",
        "합니다",
        "있습니다",
        "없습니다",
        "보입니다",
    }
)
_FORBIDDEN_REPORT_VOCABULARY = (
    "진단",
    "장애",
    "경계선 지능",
    "치료",
    "처방",
    "약물",
    "복약",
    "투약",
    "의료",
    "진료",
    "상담",
    "검사",
    "추천",
    "권고",
    "권장",
    "ADHD",
    "자폐",
    "질환",
    "정신 질환",
    "심리 진단",
    "상위",
    "하위",
    "백분위",
    "등수",
    "순위",
    "석차",
    "뒤처",
    "느리",
    "빠르",
    "지연",
)
_FORBIDDEN_REPORT_PATTERNS = (
    re.compile(
        "|".join(
            re.escape(term).replace(r"\ ", r"\s*") for term in _FORBIDDEN_REPORT_VOCABULARY
        ),
        re.IGNORECASE,
    ),
    re.compile(r"(?:또래|동년배|반\s*친구|학급|반)\s*(?:평균\s*)?(?:보다|대비|과\s*비교)"),
    re.compile(r"(?:학급|반)\s*평균"),
)


def numeric_tokens(text: str) -> set[str]:
    """Canonicalize complete decimal, magnitude, percent, and unit expressions."""

    tokens: set[str] = set()
    for match in _REPORT_NUMBER_TOKEN.finditer(text):
        number = int(match.group("number").replace(",", ""))
        magnitude = re.sub(r"\s+", "", match.group("magnitude"))
        for suffix in magnitude:
            number *= _REPORT_MAGNITUDES[suffix]
        unit = match.group("unit") or ""
        if unit == "퍼센트":
            unit = "%"
        tokens.add(f"{number}{unit}")
    return tokens


def _quoted_text(text: str) -> list[str]:
    """Extract only balanced supported quotes and reject malformed punctuation."""

    stack: list[tuple[str, int]] = []
    quotes: list[str] = []
    symmetric_quotes = {quote for quote, closer in _REPORT_QUOTE_PAIRS.items() if quote == closer}
    for index, character in enumerate(text):
        if character in symmetric_quotes:
            if stack and stack[-1][0] == character:
                _, start = stack.pop()
                if not (quote := text[start + 1 : index].strip()):
                    raise ValueError("empty report quote")
                quotes.append(quote)
            else:
                stack.append((character, index))
        elif character in _REPORT_QUOTE_PAIRS:
            stack.append((_REPORT_QUOTE_PAIRS[character], index))
        elif character in _REPORT_QUOTE_CLOSERS:
            if not stack or stack[-1][0] != character:
                raise ValueError("unbalanced report quote")
            _, start = stack.pop()
            if not (quote := text[start + 1 : index].strip()):
                raise ValueError("empty report quote")
            quotes.append(quote)
    if stack:
        raise ValueError("unbalanced report quote")
    return quotes


def _normalized_report_token(token: str) -> str:
    for suffix in _REPORT_PARTICLE_SUFFIXES:
        if len(token) > len(suffix) and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _report_lexical_tokens(text: str) -> set[str]:
    return {
        normalized
        for token in _REPORT_KOREAN_TOKEN.findall(text.lower())
        if (normalized := _normalized_report_token(token)) not in _REPORT_ALLOWED_TOKENS
    }


def reject_forbidden_report_language(text: str) -> None:
    if any(pattern.search(text) for pattern in _FORBIDDEN_REPORT_PATTERNS):
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
        grounded_tokens = _report_lexical_tokens(grounded)
        if not _report_lexical_tokens(narrative.text).issubset(grounded_tokens):
            raise ValueError("ungrounded report language")
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
