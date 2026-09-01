from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    CanonicalValueV2,
    ExpressionLevel,
    HintLevel,
    MoneyValueV2,
    NumberValueV2,
    UiReferenceInteractionV2,
)

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_REFERENCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
_ARABIC_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_UNIT = r"(?:원|개|명|분|시간|시|cm|mm|m|g|kg|칸|묶음|장|잔|권)"
_SINO_NUMBER_WITH_UNIT = re.compile(
    rf"([영공일이삼사오육칠팔구십백천만]+)\s*{_NUMBER_UNIT}"
)
_KOREAN_WORD = re.compile(r"[\uAC00-\uD7A3]+")
_KOREAN_NUMBER_ENDINGS = (
    "이었어",
    "이었다",
    "이라고",
    "이잖아",
    "입니다",
    "이에요",
    "예요",
    "이구나",
    "이네",
    "이야",
    "야",
    "이다",
    "이지",
)
_REACTION_REQUEST_ENDING = re.compile(
    r"(?:알려|말해|골라|보여|도와|같이\s*해|해\s*줄|해\s*줘)"
    r".{0,40}(?:줄래|줄\s*수\s*있어|주면\s*안\s*될까|줘|할까|해\s*볼까)"
    r"\s*[.!~…]*$"
)

_EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?82[-.\s]?)?(?:0?10|0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"
)
_SENSITIVE_NUMBER = re.compile(
    r"(?<!\d)\d{6}[-.\s]?[1-4]\d{6}(?!\d)|"
    r"(?:주민(?:등록)?번호|계좌번호|카드번호|생년월일)"
)
_SELF_INTRODUCED_NAME = re.compile(
    r"(?:내|제)\s*이름(?:은|이|은요|이요)?\s*(?:[:은는이가]\s*)?[A-Za-z가-힣]{2,30}|"
    r"(?:나는|난|저는|전)\s+[A-Za-z가-힣]{2,12}(?:이야|야|예요|이에요|입니다|라고\s*해)"
)
_ADDRESS_OR_SCHOOL = re.compile(
    r"(?:주소|거주지|사는\s*곳|우편번호)|"
    r"(?:내|우리|저희)\s*집(?:은|이|주소)?|"
    r"[\uAC00-\uD7A3]{2,20}(?:로|길)\s*\d{1,5}(?:-\d{1,5})?(?:번지|호)?|"
    r"(?:초등학교|중학교|고등학교)|\d학년\s*\d반"
)
_SYSTEM_MANIPULATION = re.compile(
    r"(?:시스템\s*(?:프롬프트|지시|메시지|규칙)|"
    r"프롬프트\s*(?:해킹|인젝션)?|"
    r"(?:지시|명령|규칙)를?\s*무시|"
    r"역할을?\s*바꾸|개발자\s*메시지|"
    r"숨겨진\s*(?:지시|정답)|jailbreak|prompt\s*injection)"
)
_UNSAFE_BRIDGE_CONTENT = re.compile(
    r"(?:자살|자해|죽고\s*싶|죽여|폭탄|성폭력|아동\s*학대)"
)

_SINO_DIGITS = {
    "영": 0,
    "공": 0,
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
_SINO_SMALL_UNITS = {"십": 10, "백": 100, "천": 1000}

_NATIVE_ONE_FORMS = {
    1: ("한", "하나"),
    2: ("두", "둘"),
    3: ("세", "셋"),
    4: ("네", "넷"),
    5: ("다섯",),
    6: ("여섯",),
    7: ("일곱",),
    8: ("여덟",),
    9: ("아홉",),
}
_NATIVE_TENS_FORMS = {
    10: ("열", "열"),
    20: ("스물", "스무"),
    30: ("서른", "서른"),
    40: ("마흔", "마흔"),
    50: ("쉰", "쉰"),
    60: ("예순", "예순"),
    70: ("일흔", "일흔"),
    80: ("여든", "여든"),
    90: ("아흔", "아흔"),
}
_NATIVE_NUMBER_WORDS: dict[str, int] = {}
for _native_value, _native_forms in _NATIVE_ONE_FORMS.items():
    for _native_form in _native_forms:
        _NATIVE_NUMBER_WORDS[_native_form] = _native_value
for _native_tens, (_native_prefix, _native_standalone) in _NATIVE_TENS_FORMS.items():
    _NATIVE_NUMBER_WORDS[_native_standalone] = _native_tens
    for _native_value, _native_forms in _NATIVE_ONE_FORMS.items():
        for _native_form in _native_forms:
            _NATIVE_NUMBER_WORDS[f"{_native_prefix}{_native_form}"] = (
                _native_tens + _native_value
            )
_NATIVE_NUMBER_PATTERN = "|".join(
    re.escape(word) for word in sorted(_NATIVE_NUMBER_WORDS, key=len, reverse=True)
)
_NATIVE_NUMBER_WITH_UNIT = re.compile(
    rf"(?<![가-힣])({_NATIVE_NUMBER_PATTERN})\s*{_NUMBER_UNIT}"
)

SpeakerMoodV2 = Literal["curious", "listening", "thinking", "relieved", "celebrating"]
AskModeV2 = Literal["answer", "reason_or_method", "answer_and_method", "none"]
BridgeInteractionKindV2 = Literal["playful", "meta", "off_topic", "refusal", "insult"]
ConversationResponseModeV2 = Literal[
    "normal",
    "explain_mormi_limit",
    "explain_ai_role",
    "decline_answer_and_ask",
    "respond_refusal",
    "respond_safe_play",
    "redirect_to_help_card",
    "safety_redirect",
]
ConversationReaskModeV2 = Literal[
    "remaining_targets",
    "help_guided_targets",
    "joint_action",
]
HelpCardEventV2 = Literal["none", "opened_or_strengthened"]
SpeakerKnowledgeSourceV2 = Literal[
    "screen",
    "child_verified",
    "jointly_derived",
]
STABLE_COPY_JOINT_ACTION_V2: Literal["follow_visible_joint_ui"] = (
    "follow_visible_joint_ui"
)
STABLE_COPY_L0_GENERATION_BRIEF_V2 = (
    "카드 내용을 되말하거나 이해했다고 하지 말고 공동 수행만 부탁한다"
)
SpeakerResponseKindV2 = Literal[
    "new_progress",
    "task_question",
    "incorrect_answer",
    "incorrect_method",
    "incorrect_answer_and_method",
    "help_request",
    "expression_block",
    "related_vague",
    "no_response",
    "structured_response",
]


class DialogueV2SpeakerModel(BaseModel):
    """Strict base for data that crosses a V2 speaker-model boundary."""

    model_config = ConfigDict(extra="forbid")


class SpeakerEvidenceV2(DialogueV2SpeakerModel):
    """Raw-free provenance for evidence already accepted into the ledger."""

    # Ledger evidence IDs may be raw SHA-256 hex and therefore begin with a digit.
    evidence_id: str = Field(pattern=_REFERENCE_PATTERN, max_length=100)
    # The Sonnet Low/ledger boundary may retain literal evidence for audit,
    # but Sonnet Low never needs arbitrary child-authored text.  Typed verified
    # facts and relation IDs provide its complete acknowledgement authority.
    text: Literal[None] = None
    verdict: Literal["correct", "sufficient", "partial", "incorrect", "uncertain"]


class SpeakerAllowedFactV2(DialogueV2SpeakerModel):
    """A server-authorized fact with an explicit, closed-world provenance.

    ``source`` is intentionally limited to the immutable screen, literal child
    evidence already accepted by the ledger, or a server-owned joint action.
    A help card is not a knowledge source: its body must never be copied into
    this contract or used to widen the speaker's mathematical authority.

    ``screen`` is the compatibility default for existing pinned plans. New
    runtime plans should always set the precise source explicitly.
    """

    fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    value: CanonicalValueV2
    speaker_text: str = Field(min_length=1, max_length=160)
    source: SpeakerKnowledgeSourceV2 = "screen"

    @model_validator(mode="after")
    def copy_cannot_smuggle_an_unrelated_number(self) -> SpeakerAllowedFactV2:
        numbers = _numeric_literals_v2(self.speaker_text)
        allowed = _canonical_numbers_v2([self.value])
        if not numbers.issubset(allowed):
            raise ValueError("speaker fact text contains an undeclared number")
        return self


class SpeakerAcceptedRelationV2(DialogueV2SpeakerModel):
    """Reviewed meaning of a relation newly accepted into the ledger.

    The provenance is part of the speaker contract for the same reason it is
    part of :class:`SpeakerAllowedFactV2`: choosing a reviewed L2 option or
    completing an L0 joint action is supported learning, not an independent
    explanation authored by the child.  ``child_verified`` remains the
    compatibility default for literal free-text evidence.
    """

    relation_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    speaker_label: str = Field(min_length=1, max_length=160)
    source: Literal["child_verified", "jointly_derived"] = "child_verified"


class SpeakerTargetV2(DialogueV2SpeakerModel):
    fact_ids: list[str] = Field(default_factory=list, max_length=8)
    relation_ids: list[str] = Field(default_factory=list, max_length=8)
    ask_mode: AskModeV2
    success_criteria_ids: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def target_shape_matches_ask_mode(self) -> SpeakerTargetV2:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("speaker target fact ids must be unique")
        if len(self.relation_ids) != len(set(self.relation_ids)):
            raise ValueError("speaker target relation ids must be unique")
        if len(self.success_criteria_ids) != len(set(self.success_criteria_ids)):
            raise ValueError("speaker success criteria ids must be unique")
        has_target = bool(self.fact_ids or self.relation_ids)
        if self.ask_mode == "none" and has_target:
            raise ValueError("ask_mode=none cannot declare a target")
        if self.ask_mode != "none" and not has_target:
            raise ValueError("an asking speaker plan needs a target")
        if self.ask_mode == "answer" and self.relation_ids:
            raise ValueError("answer ask mode cannot target a relation")
        if self.ask_mode == "reason_or_method" and self.fact_ids:
            raise ValueError("reason_or_method ask mode cannot target a fact")
        if self.ask_mode == "answer_and_method" and not (
            self.fact_ids and self.relation_ids
        ):
            raise ValueError("answer_and_method needs both fact and relation targets")
        return self


class SpeakerSupportV2(DialogueV2SpeakerModel):
    expression_level: ExpressionLevel
    hint_level: HintLevel
    support_need: Literal["expression", "concept", "both", "general_help", "none"] = (
        "none"
    )
    question_style_guide: str = Field(min_length=1, max_length=240)
    help_card_visible: bool = False


class SpeakerTargetFocusV2(DialogueV2SpeakerModel):
    """Child-safe meaning of one unresolved target, without its hidden truth."""

    target_kind: Literal["fact", "relation"]
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    speaker_label: str = Field(min_length=1, max_length=160)


class ConversationResponsePlanV2(DialogueV2SpeakerModel):
    """Deterministic conversational move compiled before surface generation.

    The plan carries only the selected response mode, the remaining reviewed
    target labels and public UI state.  It has no child-text field, expected
    truth, help-card body, or revealed mathematical content.  In particular,
    ``card_visible`` authorizes the phrase "look at the help card" only; it
    never authorizes the speaker to explain or summarize the card.

    ``revealed_relation_ids`` deliberately does not exist here. Those IDs are
    server-side note provenance and are not speaker knowledge.
    """

    response_mode: ConversationResponseModeV2
    reask_mode: ConversationReaskModeV2
    reask_targets: list[SpeakerTargetFocusV2] = Field(
        default_factory=list,
        min_length=1,
        max_length=16,
    )
    card_visible: bool = False
    card_event: HelpCardEventV2 = "none"
    hint_level: HintLevel = HintLevel.H0

    @model_validator(mode="after")
    def validate_public_support_context(self) -> ConversationResponsePlanV2:
        target_keys = [
            (target.target_kind, target.target_id) for target in self.reask_targets
        ]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("conversation reask targets must be unique")

        if self.card_visible and self.hint_level is HintLevel.H0:
            raise ValueError("a visible help card requires a non-H0 hint level")
        if not self.card_visible and self.hint_level is not HintLevel.H0:
            raise ValueError("a hidden help card cannot declare an active hint level")
        if not self.card_visible and self.card_event != "none":
            raise ValueError("a hidden help card cannot declare a card event")

        help_guided_response_modes = {
            "redirect_to_help_card",
            "decline_answer_and_ask",
        }
        if self.reask_mode == "help_guided_targets":
            if not self.card_visible:
                raise ValueError("help-guided reasking requires a visible help card")
            if self.response_mode not in help_guided_response_modes:
                raise ValueError(
                    "help-guided reasking requires a help-aware response mode"
                )
        elif (
            self.response_mode in help_guided_response_modes
            and self.reask_mode != "joint_action"
        ):
            raise ValueError(
                "help-aware response mode requires help-guided targets or joint action"
            )

        if self.reask_mode == "joint_action" and (
            not self.card_visible or self.hint_level is not HintLevel.H3
        ):
            raise ValueError("joint action requires a visible H3 help card")
        return self


class SpeakerUiReferenceSignalV2(DialogueV2SpeakerModel):
    """Minimal UI event for natural reaction, with no card prose or math."""

    referenced_kind: Literal["help_card"] = "help_card"
    interaction: UiReferenceInteractionV2
    card_event: HelpCardEventV2 = "none"


class SpeakerResponseSignalV2(DialogueV2SpeakerModel):
    """Raw-free turn meaning used to make the response conversational.

    Sonnet Low may understand an incorrect answer or method, but neither the
    child's literal text nor the incorrect value needs to cross the Sonnet Low
    boundary.  Server-owned graph IDs preserve what was attempted and what
    changed without turning an ordinary wrong answer into a model failure.
    """

    kind: SpeakerResponseKindV2
    question_focus: Literal[
        "reason_or_method",
        "meaning",
        "confirmation_or_challenge",
    ] | None = None
    attempted_fact_ids: list[str] = Field(default_factory=list, max_length=8)
    attempted_relation_ids: list[str] = Field(default_factory=list, max_length=8)
    incorrect_fact_ids: list[str] = Field(default_factory=list, max_length=8)
    incorrect_relation_ids: list[str] = Field(default_factory=list, max_length=8)
    new_fact_ids: list[str] = Field(default_factory=list, max_length=8)
    new_relation_ids: list[str] = Field(default_factory=list, max_length=8)
    repeat_count: int = Field(default=0, ge=0, le=100)
    ui_reference: SpeakerUiReferenceSignalV2 | None = None

    @model_validator(mode="after")
    def ids_are_unique_and_incorrect_ids_were_attempted(self) -> SpeakerResponseSignalV2:
        groups = (
            self.attempted_fact_ids,
            self.attempted_relation_ids,
            self.incorrect_fact_ids,
            self.incorrect_relation_ids,
            self.new_fact_ids,
            self.new_relation_ids,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("speaker response signal ids must be unique")
        if not set(self.incorrect_fact_ids).issubset(self.attempted_fact_ids):
            raise ValueError("incorrect fact ids must have been attempted")
        if not set(self.incorrect_relation_ids).issubset(self.attempted_relation_ids):
            raise ValueError("incorrect relation ids must have been attempted")
        if self.kind == "task_question" and self.question_focus is None:
            raise ValueError("task_question speaker signal requires question_focus")
        if self.kind != "task_question" and self.question_focus is not None:
            raise ValueError("question_focus is only valid for task_question")
        return self


class SpeakerPlanV2(DialogueV2SpeakerModel):
    """Closed speaker input with no field capable of carrying hidden expected truth."""

    route: Literal["main_speaker"] = "main_speaker"
    dialogue_act: str = Field(pattern=_ID_PATTERN, max_length=100)
    response_signal: SpeakerResponseSignalV2
    accepted_evidence: list[SpeakerEvidenceV2] = Field(default_factory=list, max_length=20)
    accepted_relations: list[SpeakerAcceptedRelationV2] = Field(
        default_factory=list,
        max_length=20,
    )
    target: SpeakerTargetV2
    target_focus: list[SpeakerTargetFocusV2] = Field(default_factory=list, max_length=16)
    response_plan: ConversationResponsePlanV2 | None = None
    support: SpeakerSupportV2
    allowed_facts: list[SpeakerAllowedFactV2] = Field(default_factory=list, max_length=30)
    current_question: str | None = Field(default=None, max_length=240)
    previous_mormi_text: str | None = Field(default=None, max_length=500)
    fallback_copy_ref: str = Field(pattern=_REFERENCE_PATTERN, max_length=160)

    @model_validator(mode="after")
    def validate_closed_world_plan(self) -> SpeakerPlanV2:
        evidence_ids = [item.evidence_id for item in self.accepted_evidence]
        fact_ids = [item.fact_id for item in self.allowed_facts]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("speaker evidence ids must be unique")
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("speaker allowed fact ids must be unique")
        relation_ids = [item.relation_id for item in self.accepted_relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("speaker accepted relations must be unique")
        if set(self.target.fact_ids).intersection(fact_ids):
            raise ValueError("an unresolved target fact cannot be an allowed fact")
        focus_keys = [(item.target_kind, item.target_id) for item in self.target_focus]
        target_keys = [
            *(("fact", target_id) for target_id in self.target.fact_ids),
            *(("relation", target_id) for target_id in self.target.relation_ids),
        ]
        if len(focus_keys) != len(set(focus_keys)):
            raise ValueError("speaker target focus entries must be unique")
        if set(focus_keys) != set(target_keys):
            raise ValueError("speaker target focus must describe every unresolved target")
        ui_reference = self.response_signal.ui_reference
        if ui_reference is not None:
            if not self.support.help_card_visible:
                raise ValueError("a help-card UI reference requires a visible card")
            if (
                ui_reference.card_event == "opened_or_strengthened"
                and self.support.hint_level is HintLevel.H0
            ):
                raise ValueError("a help-card event requires an active hint level")
        if self.response_plan is not None:
            response_target_keys = {
                (item.target_kind, item.target_id)
                for item in self.response_plan.reask_targets
            }
            if response_target_keys != set(target_keys):
                raise ValueError(
                    "conversation response plan must reask every unresolved target"
                )
            if (
                self.response_plan.card_visible != self.support.help_card_visible
                or self.response_plan.hint_level is not self.support.hint_level
            ):
                raise ValueError(
                    "conversation response plan must match public support context"
                )
        return self


class SpeakerOutputV2(DialogueV2SpeakerModel):
    """Natural-language surface only; the server already owns plan bookkeeping."""

    text: str
    mood: SpeakerMoodV2


class BridgePlanV2(DialogueV2SpeakerModel):
    """Small Haiku input with neither child raw text nor hidden task truth."""

    route: Literal["bridge_speaker"] = "bridge_speaker"
    dialogue_act: Literal["bridge_back"] = "bridge_back"
    interaction_kind: BridgeInteractionKindV2
    reaction_mode: ConversationResponseModeV2 = "normal"
    # Data minimization is the only complete PII allowlist: the bridge model
    # receives a semantic interaction kind, never any child-authored substring.
    safe_child_excerpt: Literal[None] = None
    current_question: str | None = Field(default=None, max_length=240)
    target: SpeakerTargetV2 = Field(default_factory=lambda: SpeakerTargetV2(ask_mode="none"))
    target_focus: list[SpeakerTargetFocusV2] = Field(default_factory=list, max_length=16)
    response_plan: ConversationResponsePlanV2 | None = None
    allowed_facts: list[SpeakerAllowedFactV2] = Field(default_factory=list, max_length=20)
    repeat_count: int = Field(default=0, ge=0, le=100)
    previous_mormi_text: str | None = Field(default=None, max_length=500)
    fallback_copy_ref: str = Field(pattern=_REFERENCE_PATTERN, max_length=160)

    @model_validator(mode="after")
    def validate_allowed_facts(self) -> BridgePlanV2:
        fact_ids = [item.fact_id for item in self.allowed_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("bridge allowed fact ids must be unique")
        if any(fact.source != "screen" for fact in self.allowed_facts):
            raise ValueError("bridge facts must come only from the immutable screen")
        if set(self.target.fact_ids).intersection(fact_ids):
            raise ValueError("an unresolved bridge target cannot be an allowed fact")
        focus_keys = {(item.target_kind, item.target_id) for item in self.target_focus}
        target_keys = {
            *(("fact", target_id) for target_id in self.target.fact_ids),
            *(("relation", target_id) for target_id in self.target.relation_ids),
        }
        if focus_keys != target_keys:
            raise ValueError("bridge target focus must describe every unresolved target")
        if self.response_plan is not None:
            response_target_keys = {
                (item.target_kind, item.target_id)
                for item in self.response_plan.reask_targets
            }
            if response_target_keys != target_keys:
                raise ValueError(
                    "bridge response plan must reask every unresolved target"
                )
        return self


class StableCopyTransitionV2(DialogueV2SpeakerModel):
    from_expression_level: ExpressionLevel
    from_hint_level: HintLevel
    to_expression_level: ExpressionLevel
    to_hint_level: HintLevel


class StableCopyPlanV2(DialogueV2SpeakerModel):
    """PII-free semantic input for one immutable generated-copy artifact.

    ``pack_id`` and ``copy_slot`` bind the request to immutable storage and
    cache metadata.  For L0, the mathematical value, relation meaning and help
    card body remain absent from every content-bearing field.
    """

    purpose: Literal["initial_help", "l2_question", "l0_intro", "l0_action"]
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    locale: Literal["ko-KR"] = "ko-KR"
    dialogue_act: str = Field(pattern=_ID_PATTERN, max_length=100)
    target: SpeakerTargetV2
    transition: StableCopyTransitionV2
    visible_facts: list[SpeakerAllowedFactV2] = Field(default_factory=list, max_length=30)
    choice_labels: list[str] = Field(default_factory=list, max_length=8)
    # Newly compiled L0 plans use a closed capability token. ``str`` remains
    # temporarily accepted so already-pinned plans from the previous compiler
    # can be read during a rolling deployment; those legacy plans are never a
    # valid generation input (see ``is_safe_l0_generation_plan``).
    joint_action: str | None = Field(default=None, min_length=1, max_length=240)
    generation_brief: str = Field(min_length=1, max_length=240)
    reveal_policy: Literal["hidden", "partial", "revealed"]

    def is_safe_l0_generation_plan(self) -> bool:
        """Whether this L0 plan contains only the new content-free contract."""

        if self.purpose not in {"l0_intro", "l0_action"}:
            return False
        expected_fact_ids = [
            f"fact_{index}" for index in range(1, len(self.target.fact_ids) + 1)
        ]
        expected_relation_ids = [
            f"relation_{index}"
            for index in range(1, len(self.target.relation_ids) + 1)
        ]
        return (
            self.joint_action == STABLE_COPY_JOINT_ACTION_V2
            and not self.visible_facts
            and self.reveal_policy == "hidden"
            and self.generation_brief == STABLE_COPY_L0_GENERATION_BRIEF_V2
            and self.target.fact_ids == expected_fact_ids
            and self.target.relation_ids == expected_relation_ids
        )

    def is_legacy_l0_pinned_plan(self) -> bool:
        """Recognize the exact unsafe shape emitted by the previous compiler."""

        if self.purpose not in {"l0_intro", "l0_action"}:
            return False
        return (
            self.joint_action not in {None, STABLE_COPY_JOINT_ACTION_V2}
            and bool(self.visible_facts)
            and self.reveal_policy == "revealed"
            and self.generation_brief != STABLE_COPY_L0_GENERATION_BRIEF_V2
            and not self.is_safe_l0_generation_plan()
        )

    @model_validator(mode="after")
    def validate_purpose_contract(self) -> StableCopyPlanV2:
        fact_ids = [item.fact_id for item in self.visible_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("stable-copy visible fact ids must be unique")
        if any(fact.source != "screen" for fact in self.visible_facts):
            raise ValueError("stable-copy facts must come only from the immutable screen")
        if len(self.choice_labels) != len(set(self.choice_labels)):
            raise ValueError("stable-copy choice labels must be unique")
        if self.purpose == "l2_question":
            if len(self.choice_labels) < 2:
                raise ValueError("L2 stable copy needs the reviewed choice labels")
            if self.joint_action is not None:
                raise ValueError("L2 stable copy cannot contain a joint action")
            if self.transition.to_expression_level is not ExpressionLevel.L2:
                raise ValueError("L2 stable copy must lead to L2")
        elif self.choice_labels:
            raise ValueError("only L2 stable copy may receive choice labels")

        if self.purpose in {"l0_intro", "l0_action"}:
            if (
                self.transition.to_expression_level is not ExpressionLevel.L0
                or self.transition.to_hint_level is not HintLevel.H3
            ):
                raise ValueError("L0 stable copy must lead to L0-H3")
            if not (
                self.is_safe_l0_generation_plan()
                or self.is_legacy_l0_pinned_plan()
            ):
                raise ValueError(
                    "L0 stable copy must match the safe generation or legacy pinned shape"
                )
        elif self.joint_action is not None:
            raise ValueError("non-L0 stable copy cannot contain a joint action")

        if self.purpose in {"initial_help", "l2_question"} and set(
            self.target.fact_ids
        ).intersection(fact_ids):
            raise ValueError("hidden target truth cannot enter pre-answer stable copy")
        return self


class StableCopyOutputV2(DialogueV2SpeakerModel):
    text: str
    mood: SpeakerMoodV2
    dialogue_act: str = Field(pattern=_ID_PATTERN, max_length=100)
    asked_fact_ids: list[str] = Field(default_factory=list, max_length=8)
    asked_relation_ids: list[str] = Field(default_factory=list, max_length=8)


def _parse_sino_number_v2(word: str) -> int | None:
    total = 0
    section = 0
    digit: int | None = None
    for char in word:
        if char in _SINO_DIGITS:
            digit = _SINO_DIGITS[char]
        elif char in _SINO_SMALL_UNITS:
            section += (digit if digit is not None else 1) * _SINO_SMALL_UNITS[char]
            digit = None
        elif char == "만":
            section += digit or 0
            total += (section or 1) * 10_000
            section = 0
            digit = None
        else:
            return None
    return total + section + (digit or 0)


def _normalized_number_v2(value: int | float | Decimal) -> str:
    normalized = Decimal(str(value)).normalize()
    return format(normalized, "f")


def _strip_korean_number_ending_v2(word: str) -> str:
    for ending in _KOREAN_NUMBER_ENDINGS:
        if word.endswith(ending) and len(word) > len(ending):
            return word[: -len(ending)]
    return word


def _unitless_korean_numbers_v2(text: str) -> set[str]:
    """Recognize standalone Korean cardinal words without treating prose as numbers.

    A unit-bearing form is handled by the dedicated patterns above.  For a
    unitless form we require the entire whitespace-delimited Korean token to be
    a cardinal (optionally followed by a small copular ending).  This catches
    answer-shaped text such as ``천`` and ``천이야`` without interpreting the
    ``이`` in ordinary words such as ``이거`` as the number two.
    """

    numbers: set[str] = set()
    for match in _KOREAN_WORD.finditer(text):
        original_word = match.group(0)
        word = _strip_korean_number_ending_v2(original_word)
        if (
            word
            and word != "만"
            and any(char in _SINO_SMALL_UNITS or char == "만" for char in word)
            and all(
                char in _SINO_DIGITS
                or char in _SINO_SMALL_UNITS
                or char == "만"
                for char in word
            )
        ):
            value = _parse_sino_number_v2(word)
            if value is not None:
                numbers.add(_normalized_number_v2(value))
            continue
        if len(word) == 1 and word in _SINO_DIGITS:
            has_copular_ending = word != original_word
            before = text[: match.start()].rstrip()
            after = text[match.end() :].lstrip()
            answer_shaped = (
                has_copular_ending
                or bool(re.search(r"(?:정답|답)(?:은|은요|이|이야)?\s*$", before))
                or not after
                or after.startswith((".", "!", "?", ","))
            )
            if answer_shaped:
                numbers.add(_normalized_number_v2(_SINO_DIGITS[word]))
            continue
        native = _NATIVE_NUMBER_WORDS.get(word)
        # Contracted attributive forms such as "한" and "세" are common prose.
        # Only standalone cardinal forms are safe enough to firewall without a
        # following unit.
        if native is not None and word not in {"한", "두", "세", "네", "스무"}:
            numbers.add(_normalized_number_v2(native))
    return numbers


def _numeric_literals_v2(text: str) -> set[str]:
    numbers = {
        _normalized_number_v2(Decimal(match.group(0).replace(",", "")))
        for match in _ARABIC_NUMBER.finditer(text)
    }
    for match in _SINO_NUMBER_WITH_UNIT.finditer(text):
        value = _parse_sino_number_v2(match.group(1))
        if value is not None:
            numbers.add(_normalized_number_v2(value))
    for match in _NATIVE_NUMBER_WITH_UNIT.finditer(text):
        numbers.add(_normalized_number_v2(_NATIVE_NUMBER_WORDS[match.group(1)]))
    numbers.update(_unitless_korean_numbers_v2(text))
    return numbers


def _canonical_numbers_v2(values: list[CanonicalValueV2]) -> set[str]:
    numbers: set[str] = set()
    for value in values:
        if isinstance(value, MoneyValueV2):
            numbers.add(_normalized_number_v2(value.amount))
        elif isinstance(value, NumberValueV2):
            numbers.add(_normalized_number_v2(value.value))
    return numbers


def _normalized_surface_v2(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _contains_surface_v2(text: str, surface: str) -> bool:
    normalized_text = _normalized_surface_v2(text)
    normalized_surface = _normalized_surface_v2(surface)
    if not normalized_surface:
        return False
    if normalized_surface.isascii() and normalized_surface.isalnum():
        return re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_surface)}(?![a-z0-9])",
            normalized_text,
        ) is not None
    if any(char.isdigit() for char in normalized_surface):
        prefix = r"(?<!\d)" if normalized_surface[0].isdigit() else ""
        suffix = r"(?!\d)" if normalized_surface[-1].isdigit() else ""
        return re.search(
            f"{prefix}{re.escape(normalized_surface)}{suffix}",
            normalized_text,
        ) is not None
    return normalized_surface in normalized_text


def _canonical_text_surfaces_v2(values: Iterable[CanonicalValueV2]) -> set[str]:
    surfaces: set[str] = set()
    for value in values:
        if value.type == "text":
            surfaces.add(value.text)
        elif value.type == "choice":
            # The internal ID is not normally child-facing, but rejecting it
            # also prevents an adapter or model from surfacing an opaque answer.
            surfaces.add(value.choice_id)
            surfaces.update(
                {
                    "right": {"오른쪽", "오른 쪽", "우측", "오른편", "오른 편"},
                    "left": {"왼쪽", "왼 쪽", "좌측", "왼편", "왼 편"},
                    "same": {"같아", "똑같아", "동일해"},
                    "equal": {"같아", "똑같아", "동일해"},
                }.get(value.choice_id.casefold(), set())
            )
    return surfaces


def unresolved_answer_surface_violation_v2(
    text: str,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    """Firewall speaker text; this never judges or mutates a child's verdict."""

    values = list(forbidden_values)
    forbidden_numbers = _canonical_numbers_v2(values)
    if _numeric_literals_v2(text).intersection(forbidden_numbers):
        return "unresolved_number_surface"
    surfaces = {*forbidden_surfaces, *_canonical_text_surfaces_v2(values)}
    if any(_contains_surface_v2(text, surface) for surface in surfaces):
        return "unresolved_answer_surface"
    return None


def bridge_excerpt_violation_v2(text: str) -> str | None:
    """Return why child text must not cross the small bridge-model boundary."""

    normalized = unicodedata.normalize("NFKC", text)
    if _EMAIL.search(normalized):
        return "email"
    if _PHONE.search(normalized):
        return "phone"
    if _SENSITIVE_NUMBER.search(normalized):
        return "sensitive_number"
    if _SELF_INTRODUCED_NAME.search(normalized):
        return "self_introduced_name"
    if _ADDRESS_OR_SCHOOL.search(normalized):
        return "address_or_school"
    if _SYSTEM_MANIPULATION.search(normalized):
        return "system_manipulation"
    if _UNSAFE_BRIDGE_CONTENT.search(normalized):
        return "unsafe_content"
    return None


def pii_safe_bridge_excerpt_v2(
    child_text: str | None,
    *,
    interaction_kind: BridgeInteractionKindV2,
) -> str | None:
    """Apply data minimization at the Haiku boundary.

    No detector can prove that arbitrary child text is free from an unknown
    name or address.  The interaction kind is sufficient for a generic bridge,
    so no raw child substring is forwarded.
    """

    # A regex denylist can help reject unsafe model output, but cannot prove
    # that an arbitrary child's text contains no unknown name or address.
    # Therefore no raw excerpt crosses this model boundary at all.
    del child_text, interaction_kind
    return None


def _dynamic_output_violation_v2(
    output: SpeakerOutputV2,
    *,
    allowed_numbers: set[str],
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    """Validate only boundaries that the language model cannot own.

    The server already owns the dialogue act, targets, evidence provenance and
    allowed model context.  Requiring the model to echo those IDs made fluent
    Korean fail closed for bookkeeping mistakes.  In particular, grammatical
    counters such as ``한 명`` were mistaken for a new mathematical claim.
    Hidden unresolved truth and privacy/safety remain server-side boundaries.
    """

    if not output.text.strip():
        return "empty_text"
    if len(output.text.splitlines()) > 2:
        return "too_many_lines"
    answer_violation = unresolved_answer_surface_violation_v2(
        output.text,
        forbidden_values=forbidden_values,
        forbidden_surfaces=forbidden_surfaces,
    )
    if answer_violation is not None:
        return answer_violation
    if not _numeric_literals_v2(output.text).issubset(allowed_numbers):
        return "number_not_allowed"
    normalized = unicodedata.normalize("NFKC", output.text).strip()
    if (
        "?" in normalized
        or "？" in normalized
        or _REACTION_REQUEST_ENDING.search(normalized)
    ):
        return "question_not_allowed"
    return None


def _numbers_from_reviewed_surfaces_v2(surfaces: Iterable[str | None]) -> set[str]:
    numbers: set[str] = set()
    for surface in surfaces:
        if surface:
            numbers.update(_numeric_literals_v2(surface))
    return numbers


def _stable_copy_output_violation_v2(
    output: StableCopyOutputV2,
    *,
    dialogue_act: str,
    target: SpeakerTargetV2,
    allowed_numbers: set[str],
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    if not output.text.strip():
        return "empty_text"
    if len(output.text.splitlines()) > 2:
        return "too_many_lines"
    if output.dialogue_act != dialogue_act:
        return "dialogue_act_mismatch"
    if len(output.asked_fact_ids) != len(set(output.asked_fact_ids)):
        return "duplicate_asked_fact_id"
    if len(output.asked_relation_ids) != len(set(output.asked_relation_ids)):
        return "duplicate_asked_relation_id"
    if set(output.asked_fact_ids) != set(target.fact_ids):
        return "asked_fact_ids_mismatch"
    if set(output.asked_relation_ids) != set(target.relation_ids):
        return "asked_relation_ids_mismatch"
    answer_violation = unresolved_answer_surface_violation_v2(
        output.text,
        forbidden_values=forbidden_values,
        forbidden_surfaces=forbidden_surfaces,
    )
    if answer_violation is not None:
        return answer_violation
    if not _numeric_literals_v2(output.text).issubset(allowed_numbers):
        return "number_not_allowed"
    return None


def speaker_output_violation_v2(
    output: SpeakerOutputV2,
    plan: SpeakerPlanV2,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    privacy_violation = bridge_excerpt_violation_v2(output.text)
    if privacy_violation is not None:
        return f"private_or_unsafe_output:{privacy_violation}"
    allowed_numbers = _canonical_numbers_v2(
        [item.value for item in plan.allowed_facts]
    )
    allowed_numbers.update(
        _numbers_from_reviewed_surfaces_v2(
            [
                plan.current_question,
                plan.previous_mormi_text,
                *(item.speaker_label for item in plan.target_focus),
                *(item.speaker_label for item in plan.accepted_relations),
            ]
        )
    )
    return _dynamic_output_violation_v2(
        output,
        allowed_numbers=allowed_numbers,
        forbidden_values=forbidden_values,
        forbidden_surfaces=forbidden_surfaces,
    )


def validate_speaker_output_v2(
    output: SpeakerOutputV2,
    plan: SpeakerPlanV2,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    return (
        output.text.strip()
        if speaker_output_violation_v2(
            output,
            plan,
            forbidden_values=forbidden_values,
            forbidden_surfaces=forbidden_surfaces,
        )
        is None
        else None
    )


def bridge_output_violation_v2(
    output: SpeakerOutputV2,
    plan: BridgePlanV2,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    privacy_violation = bridge_excerpt_violation_v2(output.text)
    if privacy_violation is not None:
        return f"private_or_unsafe_output:{privacy_violation}"
    allowed_numbers = _canonical_numbers_v2(
        [item.value for item in plan.allowed_facts]
    )
    allowed_numbers.update(
        _numbers_from_reviewed_surfaces_v2(
            [
                plan.current_question,
                plan.previous_mormi_text,
                *(item.speaker_label for item in plan.target_focus),
            ]
        )
    )
    return _dynamic_output_violation_v2(
        output,
        allowed_numbers=allowed_numbers,
        forbidden_values=forbidden_values,
        forbidden_surfaces=forbidden_surfaces,
    )


def validate_bridge_output_v2(
    output: SpeakerOutputV2,
    plan: BridgePlanV2,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    return (
        output.text.strip()
        if bridge_output_violation_v2(
            output,
            plan,
            forbidden_values=forbidden_values,
            forbidden_surfaces=forbidden_surfaces,
        )
        is None
        else None
    )


def stable_copy_output_violation_v2(
    output: StableCopyOutputV2,
    plan: StableCopyPlanV2,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    # Choice labels are visible UI data but are deliberately not numeric copy
    # authority.  An L2 introduction must not leak which candidate to select.
    if plan.purpose == "l2_question" and any(
        _contains_surface_v2(output.text, label) for label in plan.choice_labels
    ):
        return "choice_label_repeated"
    allowed_numbers = _canonical_numbers_v2([item.value for item in plan.visible_facts])
    return _stable_copy_output_violation_v2(
        output,
        dialogue_act=plan.dialogue_act,
        target=plan.target,
        allowed_numbers=allowed_numbers,
        forbidden_values=forbidden_values,
        forbidden_surfaces=forbidden_surfaces,
    )


def validate_stable_copy_output_v2(
    output: StableCopyOutputV2,
    plan: StableCopyPlanV2,
    *,
    forbidden_values: Iterable[CanonicalValueV2] = (),
    forbidden_surfaces: Iterable[str] = (),
) -> str | None:
    return (
        output.text.strip()
        if stable_copy_output_violation_v2(
            output,
            plan,
            forbidden_values=forbidden_values,
            forbidden_surfaces=forbidden_surfaces,
        )
        is None
        else None
    )
