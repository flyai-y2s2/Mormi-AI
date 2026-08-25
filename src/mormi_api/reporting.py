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
    SpeechChangeSummaryRequest,
    SpeechChangeSummaryResponse,
    utc_now,
)

if TYPE_CHECKING:
    from .repository import Repository


_REPORT_MAGNITUDES = {"십": 10, "백": 100, "천": 1_000, "만": 10_000, "억": 100_000_000}
_REPORT_UNITS = r"%|퍼센트|원|개|명|회|점|분|시간|시|cm|mm|m|g|kg|칸|묶음|장"
_REPORT_NUMBER_TOKEN = re.compile(
    rf"(?P<number>\d[\d,]*)(?P<magnitude>(?:\s*[십백천만억])*)\s*(?P<unit>{_REPORT_UNITS})?"
)
_REPORT_NATIVE_ONES = {
    "한": 1,
    "하나": 1,
    "두": 2,
    "둘": 2,
    "세": 3,
    "셋": 3,
    "네": 4,
    "넷": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
}
_REPORT_NATIVE_TENS = {
    "열": 10,
    "스물": 20,
    "서른": 30,
    "마흔": 40,
    "쉰": 50,
    "예순": 60,
    "일흔": 70,
    "여든": 80,
    "아흔": 90,
}
_REPORT_KOREAN_NUMBERS = dict(_REPORT_NATIVE_ONES)
_REPORT_KOREAN_NUMBERS.update(_REPORT_NATIVE_TENS)
_REPORT_KOREAN_NUMBERS["스무"] = 20
for _native_tens_word, _native_tens_value in _REPORT_NATIVE_TENS.items():
    for _native_one_word, _native_one_value in _REPORT_NATIVE_ONES.items():
        if _native_one_word in {"하나", "둘", "셋", "넷"}:
            continue
        _REPORT_KOREAN_NUMBERS[f"{_native_tens_word}{_native_one_word}"] = (
            _native_tens_value + _native_one_value
        )
_REPORT_KOREAN_COUNT_UNITS = r"원|개|명|회|점|분|시간|시|칸|묶음|장|문제|가지|번"
_REPORT_KOREAN_PARTICLE = r"(?:은|는|이|가|을|를|의|만|도|씩|마다|에서|으로|와|과)?"
_REPORT_KOREAN_NUMBER_TOKEN = re.compile(
    rf"(?<![가-힣])(?:(?P<native>{'|'.join(sorted(_REPORT_KOREAN_NUMBERS, key=len, reverse=True))})"
    rf"|(?P<sino>[일이삼사오육칠팔구십백천만]+))\s*"
    rf"(?P<unit>{_REPORT_KOREAN_COUNT_UNITS})(?={_REPORT_KOREAN_PARTICLE}(?:[^가-힣]|$))"
)
_REPORT_SINO_DIGITS = {
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_REPORT_SINO_UNITS = {"십": 10, "백": 100, "천": 1_000}
_REPORT_QUOTE_PAIRS = {
    "‘": "’",
    "“": "”",
    "'": "'",
    '\"': '\"',
    "『": "』",
    "「": "」",
    "《": "》",
    "〈": "〉",
}
_REPORT_QUOTE_CLOSERS = frozenset(_REPORT_QUOTE_PAIRS.values())
_REPORT_UNSUPPORTED_QUOTE_LIKE = frozenset(
    "‹›«»⟨⟩【】〔〕〖〗〘〙〚〛⦅⦆〝〞〟〃{}<>（）［］｛｝"
)
# The only permitted multi-fact wording is this exact, ordered joiner.
REPORT_FACT_SEPARATOR = " "
_FORBIDDEN_REPORT_VOCABULARY = (
    "진단",
    "장애",
    "경계선 지능",
    "치료",
    "처방",
    "약물",
    "복용",
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
    re.compile(
        r"(?:친구|학생|아이|아동|또래|동년배|학급|반)(?:들)?\s*"
        r"(?:보다|에\s*비해|대비|과\s*비교)"
    ),
    re.compile(r"약\s*(?:을|를|은|는)?\s*(?:복용|먹|투여|처방|치료)"),
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
    for match in _REPORT_KOREAN_NUMBER_TOKEN.finditer(text):
        native = match.group("native")
        sino = match.group("sino")
        if sino == "이":
            continue
        number = _REPORT_KOREAN_NUMBERS[native] if native else _parse_sino_number(sino)
        tokens.add(f"{number}{match.group('unit')}")
    return tokens


def _parse_sino_number(word: str) -> int:
    total = 0
    section = 0
    current = 0
    for character in word:
        if character in _REPORT_SINO_DIGITS:
            current = _REPORT_SINO_DIGITS[character]
        elif character in _REPORT_SINO_UNITS:
            section += (current or 1) * _REPORT_SINO_UNITS[character]
            current = 0
        elif character == "만":
            total += (section + current or 1) * 10_000
            section = 0
            current = 0
    return total + section + current


def _quoted_text(text: str) -> list[str]:
    """Extract only balanced supported quotes and reject malformed punctuation."""

    stack: list[tuple[str, int]] = []
    quotes: list[str] = []
    symmetric_quotes = {quote for quote, closer in _REPORT_QUOTE_PAIRS.items() if quote == closer}
    for index, character in enumerate(text):
        if character in _REPORT_UNSUPPORTED_QUOTE_LIKE:
            raise ValueError("unsupported report quote punctuation")
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
        referenced_statements = [facts[ref] for ref in narrative.evidence_refs]
        concatenated_statements = REPORT_FACT_SEPARATOR.join(referenced_statements)
        is_exactly_grounded = (
            narrative.text in referenced_statements
            or narrative.text == concatenated_statements
        )
        if not is_exactly_grounded:
            raise ValueError("ungrounded report language")
    return response


def validate_speech_change_summary(
    request: SpeechChangeSummaryRequest,
    response: SpeechChangeSummaryResponse | dict[str, object],
) -> SpeechChangeSummaryResponse:
    """Allow short interpretation only when its concrete evidence stays traceable."""

    value = SpeechChangeSummaryResponse.model_validate(response)
    past = request.past.utterance.strip()
    recent = request.recent.utterance.strip()
    grounded = f"{past} {recent}"
    spans = [span.strip() for span in value.evidence_spans]
    if any(not span or (span not in past and span not in recent) for span in spans):
        raise ValueError("ungrounded speech evidence span")
    if not any(span in past for span in spans) or not any(span in recent for span in spans):
        raise ValueError("speech comparison requires evidence from both utterances")
    if not numeric_tokens(value.text).issubset(numeric_tokens(grounded)):
        raise ValueError("ungrounded speech change number")
    if any(quote not in grounded for quote in _quoted_text(value.text)):
        raise ValueError("ungrounded speech change quote")
    reject_forbidden_report_language(value.text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<!\d)[.!?。！？]+(?!\d)", value.text.strip())
        if sentence.strip()
    ]
    if not 1 <= len(sentences) <= 2 or any(
        re.search(r"[가-힣]", sentence) is None for sentence in sentences
    ):
        raise ValueError("speech change must use one or two Korean sentences")
    return value


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


def _decrypt_report_response(repository: Repository, payload: str) -> str | None:
    """Read plaintext envelopes and fail closed only for unreadable legacy ciphertext."""

    try:
        return repository.text_codec.load(payload)
    except ValueError:
        return None


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
                    _decrypt_report_response(repository, turn.response_raw_encrypted)
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
        # A note may preserve exact child wording, but NoteUpdate does not carry
        # its owning conversation.  Keep the v1 contract stable while withholding
        # notes until their consent and retention provenance can be verified.
        notes=[],
    )
