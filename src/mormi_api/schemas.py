from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dictionary_models import DictionaryCard, DictionaryReference


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SceneType(StrEnum):
    HOME_TEACH = "home_teach"
    CAFE = "cafe"
    AMUSEMENT_PARK = "amusement_park"


class RetentionPolicy(StrEnum):
    NO_RAW = "no_raw"
    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    PERMANENT = "permanent"

    @property
    def days(self) -> int | None:
        return {
            RetentionPolicy.NO_RAW: None,
            RetentionPolicy.DAYS_30: 30,
            RetentionPolicy.DAYS_90: 90,
            RetentionPolicy.PERMANENT: None,
        }[self]

    def expires_at(self, started_at: datetime) -> datetime | None:
        return started_at + timedelta(days=self.days) if self.days is not None else None


class ExpressionLevel(StrEnum):
    """Expression-support levels exposed by the dialogue contract.

    ``L1`` remains parseable only for conversations and profiles created
    before the four-level ladder migration. New dialogue content and new
    transitions use ``L4 -> L3 -> L2 -> L0`` without renumbering the labels.
    """

    L4 = "L4"
    L3 = "L3"
    L2 = "L2"
    L1 = "L1"
    L0 = "L0"

    def canonical(self) -> ExpressionLevel:
        """Map a legacy L1 value onto the canonical selection level."""

        return ExpressionLevel.L2 if self is ExpressionLevel.L1 else self

    def lower(self) -> ExpressionLevel:
        order = (
            ExpressionLevel.L4,
            ExpressionLevel.L3,
            ExpressionLevel.L2,
            ExpressionLevel.L0,
        )
        current = self.canonical()
        return order[min(order.index(current) + 1, len(order) - 1)]

    def higher(self) -> ExpressionLevel:
        order = (
            ExpressionLevel.L4,
            ExpressionLevel.L3,
            ExpressionLevel.L2,
            ExpressionLevel.L0,
        )
        current = self.canonical()
        return order[max(order.index(current) - 1, 0)]

    @property
    def rank(self) -> int:
        return {"L4": 4, "L3": 3, "L2": 2, "L1": 2, "L0": 0}[self.value]


class HintLevel(StrEnum):
    H0 = "H0"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"

    def increase(self) -> HintLevel:
        order = list(HintLevel)
        return order[min(order.index(self) + 1, len(order) - 1)]


class ResponseType(StrEnum):
    TEXT = "text"
    CHOICE = "choice"
    FILL = "fill"
    COUNT = "count"
    EQUATION = "equation"
    ACTION = "action"
    NO_RESPONSE = "no_response"


class NoResponseKindV2(StrEnum):
    """Server-visible meaning of a response with no child transcript.

    Existing clients send only ``type=no_response`` for the explicit help
    button.  That legacy shape is therefore interpreted as ``explicit_help``;
    silence and ASR-empty events must opt in explicitly so they cannot
    accidentally open the same help path.
    """

    EXPLICIT_HELP = "explicit_help"
    SILENCE_TIMEOUT = "silence_timeout"
    ASR_EMPTY = "asr_empty"


class ResponseCategory(StrEnum):
    CORRECT_FULL = "correct_full"
    CORRECT_PARTIAL = "correct_partial"
    # The response is on-topic but too underspecified to support a factual
    # claim (for example, "잘 세봐" when a counting method was requested).
    # It is neither a help request nor evidence of a misconception.
    RELATED_VAGUE = "related_vague"
    EXPRESSION_BLOCK = "expression_block"
    CONCEPTUAL_ERROR = "conceptual_error"
    CONCEPTUAL_BLOCK = "conceptual_block"
    NO_RESPONSE = "no_response"
    RECOGNITION_OR_INPUT_ERROR = "recognition_or_input_error"
    UNRELATED_RESPONSE = "unrelated_response"
    HELP_REQUEST = "help_request"
    SELF_CORRECTION = "self_correction"


class SupportTrigger(StrEnum):
    NONE = "none"
    RELATED_VAGUE = "related_vague"
    EXPLICIT_HELP_REQUEST = "explicit_help_request"
    EXPRESSION_BLOCK = "expression_block"
    CONCEPTUAL_CONFLICT = "conceptual_conflict"
    REPEATED_CONCEPTUAL_CONFLICT = "repeated_conceptual_conflict"


class HelpCardEvent(StrEnum):
    NONE = "none"
    OPENED = "opened"
    STRENGTHENED = "strengthened"
    JOINT = "joint"


class EntryStance(StrEnum):
    """Legacy v2 interpretation of a reviewed wrong-guess entry.

    This is deliberately separate from mathematical claims.  Rejecting
    Mormi's guess is useful conversational evidence, but it is not by itself
    an answer, an explanation, or star-note material.
    """

    NOT_APPLICABLE = "not_applicable"
    REJECT_WRONG_GUESS = "reject_wrong_guess"
    ACCEPT_WRONG_GUESS = "accept_wrong_guess"
    UNCLEAR = "unclear"


class EntryPhase(StrEnum):
    """Legacy entry state plus the reusable split-L4 follow-up state."""

    AWAITING_ENTRY_RESPONSE = "awaiting_entry_response"
    AWAITING_OPEN_FOLLOWUP = "awaiting_open_followup"
    AWAITING_TARGETED_FOLLOWUP = "awaiting_targeted_followup"
    RESOLVED = "resolved"


class DifficultyClass(StrEnum):
    EXPRESSION = "expression"
    CONCEPT = "concept"
    BOTH = "both"
    INPUT = "input"
    ENGAGEMENT = "engagement"
    UNKNOWN = "unknown"


class TaskRelation(StrEnum):
    """How a safe utterance relates to the current learning conversation."""

    CURRENT_TASK = "current_task"
    META_ABOUT_TASK = "meta_about_task"
    META_ABOUT_MORMI = "meta_about_mormi"
    OFF_TOPIC = "off_topic"
    UNKNOWN = "unknown"


class InteractionIntent(StrEnum):
    """Legacy/broad social telemetry, never a required routing taxonomy.

    New safe utterances do not need a new enum member.  The understanding
    model can use ``OTHER_SAFE_SOCIAL`` or ``NONE`` and describe the meaning
    in ``conversation_summary`` instead.  Pedagogical state changes must not
    depend on a perfect choice from this finite list.
    """

    NONE = "none"
    AUTHENTICITY_CHALLENGE = "authenticity_challenge"
    PLAYFUL_TEASE = "playful_tease"
    FRUSTRATION = "frustration"
    REFUSAL = "refusal"
    OTHER_SAFE_SOCIAL = "other_safe_social"


class SemanticAssessment(StrEnum):
    """Meaning-level status for one requested part of the child's response."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class UnderstandingRoute(StrEnum):
    """The graph path selected after the fast first-pass understanding."""

    NORMAL = "normal"
    ADJUDICATE = "adjudicate"
    BRIDGE = "bridge"


class SafetyCategory(StrEnum):
    NORMAL = "normal"
    UNKNOWN = "unknown"
    PLAYFUL_OFFTOPIC = "playful_offtopic"
    SEXUAL = "sexual"
    PERSONAL_DATA = "personal_data"
    PROMPT_INJECTION = "prompt_injection"
    ABUSIVE = "abusive"
    DANGEROUS = "dangerous"


class SpeakerVerificationPolicy(StrEnum):
    """How much validation a generated character line requires.

    Deterministic validation always runs.  ``semantic`` adds an independent
    low-latency audit only for turns where a paraphrased question or a child
    expression could subtly change the pedagogical meaning.
    """

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


class DialogueRuntimeContractVersion(StrEnum):
    """Conversation-pinned dialogue implementation contract.

    Existing snapshots predate this field and therefore load as ``LEGACY_V1``.
    A service deployment may opt new conversations into ``VERDICT_V1``, but it
    must never reinterpret a conversation that already has a pinned version.
    """

    LEGACY_V1 = "legacy-v1"
    VERDICT_V1 = "verdict-v1"


class CompletionOutcome(StrEnum):
    TAUGHT = "taught"
    SUPPORTED = "supported"
    BRIGHT_EXIT = "bright_exit"


class InputKind(StrEnum):
    TEXT = "text"
    CHOICES = "choices"
    FILL = "fill"
    COUNT = "count"
    EQUATION = "equation"
    JOINT = "joint"
    BUTTON = "button"
    NONE = "none"


class NoteAttribution(StrEnum):
    CHILD = "child"
    COAUTHORED = "coauthored"


class NoteEvidence(StrEnum):
    DIRECT_EXPLANATION = "direct_explanation"
    SUPPORTED_COMPLETION = "supported_completion"


class PracticeAttempt(BaseModel):
    item_id: str = Field(min_length=1, max_length=100)
    correct: bool
    # Practice summaries are service telemetry, not dialogue transcripts.
    # Keep only a non-linguistic answer value (for example, a count, equation
    # result, or selected choice IDs). Free-text child utterances belong only
    # in ChildResponse, where consent-controlled plaintext storage applies.
    response: int | float | list[str] | None = None
    misconception_tag: str | None = Field(default=None, max_length=80)
    latency_ms: int | None = Field(default=None, ge=0, le=600_000)


class PracticeSummary(BaseModel):
    curriculum_session_id: str | None = Field(default=None, min_length=1, max_length=60)
    skill_id: str = Field(min_length=1, max_length=100)
    attempts: list[PracticeAttempt] = Field(default_factory=list, max_length=50)
    question_count: int = Field(default=0, ge=0, le=50)
    first_try_correct_count: int = Field(default=0, ge=0, le=50)
    wrong_attempt_count: int = Field(default=0, ge=0, le=200)
    earned_reward: int = Field(default=0, ge=0)
    misconception_tags: list[str] = Field(default_factory=list, max_length=30)
    completed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def derive_compact_counts(self) -> PracticeSummary:
        if self.attempts:
            self.question_count = self.question_count or len(self.attempts)
            self.first_try_correct_count = self.first_try_correct_count or sum(
                1 for attempt in self.attempts if attempt.correct
            )
            if not self.misconception_tags:
                self.misconception_tags = sorted(
                    {
                        attempt.misconception_tag
                        for attempt in self.attempts
                        if attempt.misconception_tag
                    }
                )
        if self.first_try_correct_count > self.question_count:
            raise ValueError("first_try_correct_count cannot exceed question_count")
        return self

    @property
    def success_rate(self) -> float | None:
        if self.question_count:
            return self.first_try_correct_count / self.question_count
        if self.attempts:
            return sum(1 for attempt in self.attempts if attempt.correct) / len(self.attempts)
        return None


class PracticeResult(PracticeSummary):
    """A persisted practice summary with its ownership and idempotency identity."""

    practice_result_id: str = Field(default_factory=lambda: new_id("practice"))
    learner_id: int = Field(ge=1)


class CafeMenuItem(BaseModel):
    """A menu snapshot supplied by the frontend through the trusted BFF."""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=40)
    price: int = Field(ge=0, le=100_000)
    image_url: str | None = Field(default=None, max_length=500)


class CafeSessionContext(BaseModel):
    """Frontend-owned café facts frozen for one independent conversation."""

    menu_items: list[CafeMenuItem] = Field(min_length=2, max_length=20)
    mormi_menu_id: str = Field(min_length=1, max_length=64)
    budget: int | None = Field(default=None, ge=0, le=100_000)

    @model_validator(mode="after")
    def validate_menu_references(self) -> CafeSessionContext:
        menu_ids = [item.id for item in self.menu_items]
        if len(menu_ids) != len(set(menu_ids)):
            raise ValueError("menu item ids must be unique")
        if self.mormi_menu_id not in menu_ids:
            raise ValueError("mormi_menu_id must reference menu_items")
        return self


QUEUE_MIN_COUNT = 1
QUEUE_MAX_COUNT = 5


class QueueSessionContext(BaseModel):
    """Frontend-owned queue facts frozen for one AI conversation.

    The screen draws the two lines before the conversation starts, so Mormi has
    to talk about the counts the child is actually looking at. Letting this side
    draw its own would put different numbers in the speech bubble and the scene.
    """

    left_count: int = Field(ge=QUEUE_MIN_COUNT, le=QUEUE_MAX_COUNT)
    right_count: int = Field(ge=QUEUE_MIN_COUNT, le=QUEUE_MAX_COUNT)

    @model_validator(mode="after")
    def lines_must_differ(self) -> QueueSessionContext:
        if self.left_count == self.right_count:
            raise ValueError("left_count and right_count must differ")
        return self


class ParkFact(BaseModel):
    """One AI-catalog fact frozen in an amusement-park conversation."""

    key: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=50)
    value: int = Field(ge=0, le=1_000_000)
    unit: str = Field(default="", max_length=20)


class ParkTransfer(BaseModel):
    prompt: str = Field(min_length=1, max_length=160)
    equation: str = Field(min_length=1, max_length=100)
    conclusion: str = Field(min_length=1, max_length=160)


PARK_SCENARIO_STAGE: dict[str, str] = {
    "amusement_ticket_multiply": "ticket",
    "amusement_snack_divide": "snack_split",
    "amusement_pass_compare": "pass_break_even",
}

PARK_REQUIRED_FACT_KEYS: dict[str, set[str]] = {
    "amusement_ticket_multiply": {"ticket_price", "party_count", "total_price"},
    "amusement_snack_divide": {"snack_total", "payer_count", "per_person"},
    "amusement_pass_compare": {
        "single_ride_price",
        "day_pass_price",
        "break_even_rides",
        "benefit_from_rides",
    },
}


class ParkSessionContext(BaseModel):
    """AI-owned, reviewed amusement content frozen for one conversation."""

    theme_id: Literal["amusement_park"]
    variant_id: str = Field(default="legacy_external", min_length=1, max_length=120)
    stage_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=100)
    mission: str = Field(min_length=1, max_length=200)
    skill: str = Field(min_length=1, max_length=100)
    strategy: str = Field(min_length=1, max_length=200)
    mormi_misconception: str = Field(default="", max_length=200)
    prompt: str = Field(min_length=1, max_length=200)
    facts: list[ParkFact] = Field(min_length=3, max_length=12)
    required_verified_fact_keys: list[str] = Field(min_length=3, max_length=12)
    transfer: ParkTransfer

    @model_validator(mode="after")
    def validate_unique_contract(self) -> ParkSessionContext:
        keys = [fact.key for fact in self.facts]
        if len(keys) != len(set(keys)):
            raise ValueError("park fact keys must be unique")
        if len(self.required_verified_fact_keys) != len(
            set(self.required_verified_fact_keys)
        ):
            raise ValueError("required park fact keys must be unique")
        if not set(self.required_verified_fact_keys).issubset(keys):
            raise ValueError("required park fact keys must reference facts")
        return self


class SessionCreate(BaseModel):
    learner_id: int = Field(ge=1)
    scene: SceneType
    scenario_id: str = Field(min_length=1, max_length=100)
    learning_session_id: str | None = Field(default=None, max_length=100)
    # A learning session may intentionally open more than one teaching dialogue.
    # Retries reuse the same round; an explicit restart increments it.
    conversation_round: int = Field(default=1, ge=1)
    practice_result_id: str | None = Field(default=None, max_length=100)
    practice_summary: PracticeSummary | None = None
    cafe_context: CafeSessionContext | None = None
    queue_context: QueueSessionContext | None = None
    # Rolling-deploy compatibility only. The AI catalog owns amusement-park
    # copy, answers, hints and transfer problems. Only reviewed givens may be
    # preserved temporarily so an older BE can complete an in-flight visit.
    park_context: ParkSessionContext | None = Field(
        default=None,
        description=(
            "Deprecated rolling-deploy input. Only reviewed numeric givens may be "
            "preserved; all copy, answers and pedagogy are rebuilt by Mormi-AI."
        ),
        deprecated=True,
    )
    # 파일럿 참여자는 사전에 원문 저장 동의를 완료한다. 별도 필드를 보내지
    # 않는 호출도 질문·아이 원문·선택 응답을 평문으로 영구 보존한다.
    conversation_storage_consent: bool = True
    retention_policy: RetentionPolicy = RetentionPolicy.PERMANENT

    @model_validator(mode="after")
    def validate_storage_policy(self) -> SessionCreate:
        if self.conversation_storage_consent and self.retention_policy is RetentionPolicy.NO_RAW:
            raise ValueError("consented raw storage requires a retention_policy")
        if (
            not self.conversation_storage_consent
            and self.retention_policy is not RetentionPolicy.NO_RAW
        ):
            raise ValueError("retention_policy must be no_raw without storage consent")
        if self.scenario_id == "home_teach":
            if not self.learning_session_id or not self.learning_session_id.strip():
                raise ValueError("learning_session_id is required for home_teach")
            if not self.practice_result_id or not self.practice_result_id.strip():
                raise ValueError("practice_result_id is required for home_teach")
        menu_scenarios = {"cafe_budget_menu", "cafe_menu_total", "cafe_change"}
        if self.scenario_id in menu_scenarios and self.cafe_context is None:
            raise ValueError("cafe_context is required for menu scenarios")
        if self.cafe_context is not None and self.scenario_id not in menu_scenarios:
            raise ValueError("cafe_context is not used by this scenario")
        # `cafe_queue_demo` keeps drawing its own line-up so demos and tests can
        # start a queue conversation without a screen behind them.
        if self.scenario_id == "cafe_queue" and self.queue_context is None:
            raise ValueError("queue_context is required for cafe_queue")
        if self.queue_context is not None and self.scenario_id != "cafe_queue":
            raise ValueError("queue_context is not used by this scenario")
        if self.scenario_id in PARK_SCENARIO_STAGE:
            if self.scene is not SceneType.AMUSEMENT_PARK:
                raise ValueError("amusement scenarios require amusement_park scene")
        elif self.__dict__.get("park_context") is not None:
            raise ValueError("park_context is not used by this scenario")
        if self.scenario_id == "cafe_budget_menu" and (
            self.cafe_context is None or self.cafe_context.budget is None
        ):
            raise ValueError("budget is required for cafe_budget_menu")
        if self.scenario_id == "cafe_budget_menu" and self.cafe_context is not None:
            menu_by_id = {item.id: item for item in self.cafe_context.menu_items}
            mormi_price = menu_by_id[self.cafe_context.mormi_menu_id].price
            budget = self.cafe_context.budget
            assert budget is not None
            if not any(
                item.id != self.cafe_context.mormi_menu_id and mormi_price + item.price <= budget
                for item in self.cafe_context.menu_items
            ):
                raise ValueError("budget must allow at least one child menu")
        return self


class ChildResponse(BaseModel):
    turn_id: str = Field(min_length=1, max_length=100)
    response_id: UUID
    type: ResponseType
    text: str | None = Field(default=None, max_length=300)
    choice_ids: list[str] = Field(default_factory=list, max_length=20)
    values: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    no_response_kind: NoResponseKindV2 | None = None
    asr_confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: int | None = Field(default=None, ge=0, le=600_000)

    @model_validator(mode="after")
    def validate_payload(self) -> ChildResponse:
        if self.type is ResponseType.TEXT and not (self.text and self.text.strip()):
            raise ValueError("text response requires non-empty text")
        if self.type in {ResponseType.CHOICE, ResponseType.FILL} and not self.choice_ids:
            raise ValueError("choice/fill response requires choice_ids")
        if (
            self.type in {ResponseType.COUNT, ResponseType.EQUATION, ResponseType.ACTION}
            and not self.values
        ):
            raise ValueError(f"{self.type.value} response requires values")
        if self.type is ResponseType.NO_RESPONSE:
            # Backward compatibility: the only no-response event emitted by
            # the current FE is the explicit "잘 모르겠어" button.
            if self.no_response_kind is None:
                self.no_response_kind = NoResponseKindV2.EXPLICIT_HELP
        elif self.no_response_kind is not None:
            raise ValueError("no_response_kind is valid only for no_response")
        return self


class SlotClaim(BaseModel):
    slot_id: str
    value: str | int | float | bool | None = None
    factual: bool = False
    evidence_span: str = ""
    # For canonical values the classifier must normalize what the child
    # actually said, not copy the reviewed answer.  This confidence is used
    # only when a Korean number or typo cannot be parsed deterministically;
    # clear Arabic/Korean quantities are always checked against evidence in
    # code regardless of the model's confidence.
    interpretation_confidence: float | None = Field(default=None, ge=0, le=1)
    # Open-ended method/reason/explanation slots cannot be enumerated as a
    # finite list of canonical answer codes. The classifier instead states
    # whether the child's exact evidence semantically supports the reviewed
    # slot contract. Closed-value slots leave this field unset.
    supported: bool | None = None
    support_confidence: float | None = Field(default=None, ge=0, le=1)


class ArithmeticClaim(BaseModel):
    """One arithmetic relation understood from the child's own words.

    Claude extracts the linguistic meaning; deterministic code only checks
    whether the resulting numbers and operation are mathematically true for
    the reviewed task.  This keeps Korean wording out of the orchestrator.
    """

    left: int
    right: int
    operation: Literal["addition", "subtraction", "multiplication", "division"]
    result: int
    evidence_span: str = ""
    related_slot_ids: list[str] = Field(default_factory=list)
    interpretation_confidence: float = Field(default=0, ge=0, le=1)


class SpeakerQuantity(BaseModel):
    """A quantity with the reviewed scene role shown to the speaker.

    ``value`` alone is not enough for natural Korean.  The role tells the
    speaker whether the number is the money paid, a price, a count, or the
    result the child claimed.  It is descriptive context, never proof that
    the child's claim is true.
    """

    value: int
    role: str
    unit: str = ""


class SpeakerArithmeticClaim(BaseModel):
    """A provenance-preserving arithmetic claim for natural reflection."""

    operation: Literal["addition", "subtraction", "multiplication", "division"]
    source_text: str
    left: SpeakerQuantity
    right: SpeakerQuantity
    claimed_result: SpeakerQuantity
    truth_status: Literal["true", "false", "unknown"]
    related_slot_ids: list[str] = Field(default_factory=list)


class DialogueHistoryTurn(BaseModel):
    """A bounded, plaintext dialogue excerpt used only for current-session context."""

    turn_id: str
    mormi: str
    child: str | None = None
    response_type: ResponseType | None = None
    response_category: ResponseCategory | None = None


class ReferenceResolution(BaseModel):
    """An auditable resolution of a pronoun or reference in the child's words."""

    source_span: str
    resolved_to: str
    confidence: float = Field(default=0, ge=0, le=1)
    evidence_turn_ids: list[str] = Field(default_factory=list)


class UtteranceAnalysis(BaseModel):
    safety_category: SafetyCategory = SafetyCategory.UNKNOWN
    response_category: ResponseCategory = ResponseCategory.RECOGNITION_OR_INPUT_ERROR
    difficulty_class: DifficultyClass = DifficultyClass.UNKNOWN
    # These fields describe the conversational job of a safe utterance.  They
    # never verify a mathematical claim or change a ladder by themselves.
    task_relation: TaskRelation = TaskRelation.UNKNOWN
    interaction_intent: InteractionIntent = InteractionIntent.NONE
    # Open-set conversational understanding.  A safe utterance that contains
    # no answer/reason/method/help request can be handled conversationally
    # without forcing its exact intent into a growing enum.  The free-text
    # summary is audit context only; it can never verify a learning slot.
    conversation_only: bool = False
    conversation_summary: str = Field(default="", max_length=160)
    # Deprecated wire-compatibility field.  The primary understanding model
    # never writes child-facing copy; a separate lightweight bridge speaker
    # handles safe conversation-only turns.
    bridge_reply: str = Field(default="", max_length=120)
    entry_stance: EntryStance = EntryStance.NOT_APPLICABLE
    answer_status: SemanticAssessment = SemanticAssessment.NOT_APPLICABLE
    reason_status: SemanticAssessment = SemanticAssessment.NOT_APPLICABLE
    claims: list[SlotClaim] = Field(default_factory=list)
    arithmetic_claims: list[ArithmeticClaim] = Field(default_factory=list)
    reference_resolutions: list[ReferenceResolution] = Field(default_factory=list)
    # Deprecated wire-compatibility fields.  Runtime has one primary semantic
    # understanding pass and represents uncertainty directly instead of
    # invoking a second adjudicator.
    needs_adjudication: bool = False
    adjudication_reason: str = Field(default="", max_length=160)
    misconception_tag: str | None = None
    bottleneck: str = "unknown"
    # A short, exact substring of the child's response that is safe and useful
    # for a natural acknowledgement or clarification.  The orchestrator checks
    # that it really occurs in the raw response before a speaker can quote it.
    grounding_span: str = ""
    # A separate exact substring for a safe social bridge.  Keeping this out
    # of grounding_span prevents a meta remark from becoming learning proof.
    social_grounding_span: str = ""
    note_candidate: str = ""
    confidence: float = Field(default=0, ge=0, le=1)


class UtteranceClassV2(StrEnum):
    """Top-level routing class produced by the V2 understanding model."""

    LEARNING_RESPONSE = "learning_response"
    TASK_QUESTION = "task_question"
    HELP_REQUEST = "help_request"
    NON_LEARNING_SAFE = "non_learning_safe"
    SYSTEM_MANIPULATION = "system_manipulation"
    SAFETY_RISK = "safety_risk"


class QuestionFocusV2(StrEnum):
    """Minimal intent needed to answer a child question without changing pedagogy."""

    REASON_OR_METHOD = "reason_or_method"
    MEANING = "meaning"
    CONFIRMATION_OR_CHALLENGE = "confirmation_or_challenge"


class ConversationMoveV2(StrEnum):
    """Conversation action carried independently from mathematical evidence.

    ``utterance_class`` remains on the wire so conversations pinned before this
    additive contract stay readable.  This axis preserves the child's social
    or question move even when the same utterance also contains a valid claim.
    """

    NONE = "none"
    TASK_QUESTION = "task_question"
    META_QUESTION = "meta_question"
    REQUEST_MORMI_ANSWER = "request_mormi_answer"
    REFUSAL = "refusal"
    SAFE_PLAY = "safe_play"


class MoveSubjectV2(StrEnum):
    """Small subject vocabulary needed to plan a natural conversational reply."""

    TASK = "task"
    MORMI_KNOWLEDGE = "mormi_knowledge"
    MORMI_AI_IDENTITY = "mormi_ai_identity"
    PARTICIPATION = "participation"
    OTHER = "other"


def _backfill_conversation_axes_v2(data: Any) -> Any:
    """Derive additive axes when reading a legacy understanding JSON object."""

    if not isinstance(data, dict):
        return data
    payload = dict(data)
    utterance_class = payload.get("utterance_class")
    non_learning_kind = payload.get("non_learning_kind")
    move = payload.get("conversation_move")
    utterance_value = getattr(utterance_class, "value", utterance_class)
    non_learning_value = getattr(non_learning_kind, "value", non_learning_kind)
    if move is None:
        if utterance_value == UtteranceClassV2.TASK_QUESTION.value:
            move = ConversationMoveV2.TASK_QUESTION.value
        elif utterance_value == UtteranceClassV2.NON_LEARNING_SAFE.value:
            if non_learning_value == NonLearningKindV2.META.value:
                move = ConversationMoveV2.META_QUESTION.value
            elif non_learning_value == NonLearningKindV2.REFUSAL.value:
                move = ConversationMoveV2.REFUSAL.value
            else:
                move = ConversationMoveV2.SAFE_PLAY.value
        else:
            move = ConversationMoveV2.NONE.value
        payload["conversation_move"] = move
    if payload.get("move_subject") is None:
        move_value = getattr(move, "value", move)
        payload["move_subject"] = {
            ConversationMoveV2.TASK_QUESTION.value: MoveSubjectV2.TASK.value,
            ConversationMoveV2.META_QUESTION.value: MoveSubjectV2.MORMI_KNOWLEDGE.value,
            ConversationMoveV2.REQUEST_MORMI_ANSWER.value: MoveSubjectV2.PARTICIPATION.value,
            ConversationMoveV2.REFUSAL.value: MoveSubjectV2.PARTICIPATION.value,
        }.get(move_value, MoveSubjectV2.OTHER.value)
    return payload


class NonLearningKindV2(StrEnum):
    PLAYFUL = "playful"
    META = "meta"
    OFF_TOPIC = "off_topic"
    REFUSAL = "refusal"
    INSULT = "insult"


class SupportNeed(StrEnum):
    """Why the child needs support, independently of mathematical correctness."""

    EXPRESSION = "expression"
    CONCEPT = "concept"
    BOTH = "both"
    GENERAL_HELP = "general_help"
    NONE = "none"


class AnswerStatusV2(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class ReasoningStatusV2(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    MISSING = "missing"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class UnderstandingConfidenceV2(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ArithmeticValidityV2(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


class DialogueV2Model(BaseModel):
    """Strict base for contracts exchanged with the V2 model runtime."""

    model_config = ConfigDict(extra="forbid")


class ArithmeticInterpretationV2(DialogueV2Model):
    """N-ary arithmetic meaning extracted from one exact evidence span."""

    operation: Literal["addition", "subtraction", "multiplication", "division"]
    operands: list[int | float] = Field(min_length=2, max_length=16)
    result: int | float
    mathematical_validity: ArithmeticValidityV2


class MoneyValueV2(DialogueV2Model):
    type: Literal["money"] = "money"
    amount: int | float
    currency: str = Field(default="KRW", min_length=3, max_length=3)


class NumberValueV2(DialogueV2Model):
    type: Literal["number"] = "number"
    value: int | float
    unit: str | None = Field(default=None, max_length=30)


class TextValueV2(DialogueV2Model):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=160)


class BooleanValueV2(DialogueV2Model):
    type: Literal["boolean"] = "boolean"
    value: bool


class ChoiceValueV2(DialogueV2Model):
    type: Literal["choice"] = "choice"
    choice_id: str = Field(min_length=1, max_length=160)


CanonicalValueV2 = Annotated[
    MoneyValueV2 | NumberValueV2 | TextValueV2 | BooleanValueV2 | ChoiceValueV2,
    Field(discriminator="type"),
]


class FactUnderstandingClaimV2(DialogueV2Model):
    claim_kind: Literal["fact"] = "fact"
    claim_id: str = Field(min_length=1, max_length=100)
    fact_id: str = Field(min_length=1, max_length=160)
    claim_type: Literal["final_answer", "intermediate_result"]
    evidence_span: str = Field(min_length=1, max_length=300)
    interpreted_value: CanonicalValueV2
    verdict: Literal["correct", "partial", "incorrect", "uncertain"]
    confidence: float | None = Field(default=None, ge=0, le=1)


class RelationUnderstandingClaimV2(DialogueV2Model):
    claim_kind: Literal["relation"] = "relation"
    claim_id: str = Field(min_length=1, max_length=100)
    relation_id: str = Field(min_length=1, max_length=160)
    claim_type: Literal["procedure_step", "explanation"]
    evidence_span: str = Field(min_length=1, max_length=300)
    verdict: Literal["correct", "sufficient", "partial", "incorrect", "uncertain"]
    arithmetic_interpretation: ArithmeticInterpretationV2 | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class AuxiliaryUnderstandingClaimV2(DialogueV2Model):
    """Useful task-related evidence that has no reviewed canonical graph ID."""

    claim_kind: Literal["auxiliary"] = "auxiliary"
    claim_id: str = Field(min_length=1, max_length=100)
    claim_type: Literal["auxiliary"] = "auxiliary"
    evidence_span: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=160)
    verdict: Literal["correct", "sufficient", "partial", "uncertain"]
    interpreted_value: CanonicalValueV2 | None = None
    arithmetic_interpretation: ArithmeticInterpretationV2 | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


UnderstandingClaimV2 = Annotated[
    FactUnderstandingClaimV2
    | RelationUnderstandingClaimV2
    | AuxiliaryUnderstandingClaimV2,
    Field(discriminator="claim_kind"),
]


class UnderstandingTargetV2(DialogueV2Model):
    """One reviewed fact or relation the current question may ask for."""

    target_kind: Literal["fact", "relation"]
    target_id: str = Field(min_length=1, max_length=160)
    ask_kind: Literal["answer", "reason_or_method"]
    rubric: dict[str, str] = Field(default_factory=dict)
    expected_truth: CanonicalValueV2 | None = None


class ClaimableGraphContextV2(DialogueV2Model):
    fact_ids: list[str] = Field(default_factory=list, max_length=100)
    relation_ids: list[str] = Field(default_factory=list, max_length=100)
    open_auxiliary_claims: bool = True

    @model_validator(mode="after")
    def ids_must_be_unique(self) -> ClaimableGraphContextV2:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("claimable fact_ids must be unique")
        if len(self.relation_ids) != len(set(self.relation_ids)):
            raise ValueError("claimable relation_ids must be unique")
        return self


class UnderstandingTurnContextV2(DialogueV2Model):
    mormi_question: str = Field(min_length=1)
    asks: list[Literal["answer", "reason_or_method"]] = Field(min_length=1, max_length=2)
    expression_level: ExpressionLevel
    hint_level: HintLevel
    # Server-owned semantic scope of a visible H2/H3 scaffold.  This carries
    # no card prose, equation, numeric truth, or child text.  It lets the
    # understanding model interpret a later short confirmation such as
    # "응" as coauthored support for one reviewed relation without exposing
    # that help content to either Mormi speaker model.
    help_scaffolded_relation_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_help_scaffold_scope(self) -> UnderstandingTurnContextV2:
        if len(self.help_scaffolded_relation_ids) != len(
            set(self.help_scaffolded_relation_ids)
        ):
            raise ValueError("help scaffold relation ids must be unique")
        if self.help_scaffolded_relation_ids and self.hint_level not in {
            HintLevel.H2,
            HintLevel.H3,
        }:
            raise ValueError("help scaffold relations require an H2 or H3 card")
        return self


class UnderstandingRequestV2(DialogueV2Model):
    """Bounded, server-owned input for the V2 semantic understanding pass."""

    task_id: str = Field(min_length=1, max_length=160)
    visible_facts: dict[str, str | int | float | bool] = Field(default_factory=dict)
    targets: list[UnderstandingTargetV2] = Field(min_length=1, max_length=20)
    claimable_graph: ClaimableGraphContextV2
    current_turn: UnderstandingTurnContextV2
    recent_history: list[DialogueHistoryTurn] = Field(default_factory=list, max_length=6)
    child_utterance: str = Field(min_length=1, max_length=300)
    # Populated only for the single contract-repair retry.  Codes contain no
    # raw child text and ask the same understanding model to repair provenance
    # without inviting a second semantic adjudication.
    guard_feedback_codes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def targets_must_reference_the_claimable_graph(self) -> UnderstandingRequestV2:
        target_keys = [(target.target_kind, target.target_id) for target in self.targets]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("understanding targets must be unique")

        fact_ids = set(self.claimable_graph.fact_ids)
        relation_ids = set(self.claimable_graph.relation_ids)
        for target in self.targets:
            if target.target_kind == "fact" and target.target_id not in fact_ids:
                raise ValueError("fact target must exist in claimable_graph.fact_ids")
            if target.target_kind == "relation":
                if target.target_id not in relation_ids:
                    raise ValueError(
                        "relation target must exist in claimable_graph.relation_ids"
                    )
                if target.expected_truth is not None:
                    raise ValueError("relation target cannot declare expected_truth")

        if set(self.current_turn.asks) != {target.ask_kind for target in self.targets}:
            raise ValueError("current_turn asks must match target ask kinds")
        target_relation_ids = {
            target.target_id
            for target in self.targets
            if target.target_kind == "relation"
        }
        if not set(self.current_turn.help_scaffolded_relation_ids).issubset(
            target_relation_ids
        ):
            raise ValueError(
                "help scaffold relations must be unresolved relation targets"
            )
        return self


class UnderstandingResponseV2(DialogueV2Model):
    """Semantic result whose verdicts are not re-graded by deterministic code."""

    utterance_class: UtteranceClassV2
    # Additive conversational axes.  A before-validator derives them from the
    # legacy fields when an already-pinned response JSON does not contain them.
    conversation_move: ConversationMoveV2 = ConversationMoveV2.NONE
    move_subject: MoveSubjectV2 = MoveSubjectV2.OTHER
    question_focus: QuestionFocusV2 | None = None
    support_need: SupportNeed = SupportNeed.NONE
    non_learning_kind: NonLearningKindV2 | None = None
    contains_learning_evidence: bool = False
    answer_status: AnswerStatusV2 = AnswerStatusV2.NOT_APPLICABLE
    reasoning_status: ReasoningStatusV2 = ReasoningStatusV2.NOT_APPLICABLE
    claims: list[UnderstandingClaimV2] = Field(default_factory=list, max_length=30)
    confidence: UnderstandingConfidenceV2 = UnderstandingConfidenceV2.MEDIUM

    @model_validator(mode="before")
    @classmethod
    def backfill_additive_conversation_axes(cls, data: Any) -> Any:
        return _backfill_conversation_axes_v2(data)

    @model_validator(mode="after")
    def validate_semantic_contract(self) -> UnderstandingResponseV2:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("understanding claim_ids must be unique")
        if self.claims and not self.contains_learning_evidence:
            raise ValueError("claims require contains_learning_evidence=true")
        if (
            self.utterance_class is UtteranceClassV2.HELP_REQUEST
            and self.support_need is SupportNeed.NONE
        ):
            raise ValueError("help_request requires an explicit support_need")
        if (
            self.conversation_move is ConversationMoveV2.TASK_QUESTION
            and self.question_focus is None
        ):
            raise ValueError("task_question requires question_focus")
        if (
            self.conversation_move is not ConversationMoveV2.TASK_QUESTION
            and self.question_focus is not None
        ):
            raise ValueError("question_focus is only valid for task_question")
        return self


class ModelFactUnderstandingClaimV2(DialogueV2Model):
    """Compact provider-facing fact claim with structurally valid role fields."""

    claim_id: str
    target_id: str
    claim_type: Literal["final_answer", "intermediate_result"]
    evidence_span: str
    verdict: Literal["correct", "partial", "incorrect", "uncertain"]
    value_type: Literal["money", "number", "text", "boolean", "choice"] | None
    numeric_value: float | None
    text_value: str | None
    boolean_value: bool | None
    unit: str | None
    confidence: float | None


class ModelRelationUnderstandingClaimV2(DialogueV2Model):
    """Compact provider-facing relation claim with no fact-role alternatives."""

    claim_id: str
    target_id: str
    claim_type: Literal["procedure_step", "explanation"]
    evidence_span: str
    verdict: Literal["correct", "sufficient", "partial", "incorrect", "uncertain"]
    operation: Literal["addition", "subtraction", "multiplication", "division"] | None
    operands: list[float]
    result: float | None
    mathematical_validity: ArithmeticValidityV2 | None
    confidence: float | None


class ModelAuxiliaryUnderstandingClaimV2(DialogueV2Model):
    """Provider-facing progress that cannot claim a reviewed graph target."""

    claim_id: str
    evidence_span: str
    verdict: Literal["correct", "sufficient", "partial", "uncertain"]
    confidence: float | None


class ModelUnderstandingResponseV2(DialogueV2Model):
    """Small JSON grammar compiled by Anthropic, converted to the internal contract."""

    utterance_class: UtteranceClassV2
    conversation_move: ConversationMoveV2 = ConversationMoveV2.NONE
    move_subject: MoveSubjectV2 = MoveSubjectV2.OTHER
    question_focus: QuestionFocusV2 | None
    support_need: SupportNeed
    non_learning_kind: NonLearningKindV2 | None
    contains_learning_evidence: bool
    answer_status: AnswerStatusV2
    reasoning_status: ReasoningStatusV2
    fact_claims: list[ModelFactUnderstandingClaimV2]
    relation_claims: list[ModelRelationUnderstandingClaimV2]
    auxiliary_claims: list[ModelAuxiliaryUnderstandingClaimV2]
    confidence: UnderstandingConfidenceV2

    @model_validator(mode="before")
    @classmethod
    def backfill_additive_conversation_axes(cls, data: Any) -> Any:
        return _backfill_conversation_axes_v2(data)


class ChoiceOption(BaseModel):
    id: str
    label: str
    image_url: str | None = None
    disabled: bool | None = None


class InputContract(BaseModel):
    kind: InputKind
    placeholder: str | None = None
    choices: list[ChoiceOption] = Field(default_factory=list)
    target_slots: list[str] = Field(default_factory=list)
    submit_label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class HelpCardContract(BaseModel):
    visible: bool = True
    auto_open: bool = True
    level: HintLevel
    title: str = "도움 카드"
    body: str
    visual_type: str | None = None
    visual_data: dict[str, Any] = Field(default_factory=dict)


class VisualContract(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class MormiContract(BaseModel):
    # 50 characters is an authoring target, not a runtime rejection boundary.
    # A complete sentence is safer and more natural than truncated child-facing copy.
    text: str = Field(min_length=1)
    mood: Literal["curious", "listening", "thinking", "relieved", "celebrating"]
    max_lines: Literal[1, 2] = 2

    @model_validator(mode="after")
    def two_lines_maximum(self) -> MormiContract:
        if len(self.text.splitlines()) > 2:
            raise ValueError("Mormi text must fit in at most two lines")
        return self


class NoteUpdate(BaseModel):
    note_id: str = Field(default_factory=lambda: new_id("note"))
    skill_id: str
    text: str
    attribution: NoteAttribution
    evidence: NoteEvidence
    attribution_label: str


class CompletionContract(BaseModel):
    outcome: CompletionOutcome
    teach_reward_eligible: bool
    # A jointly modelled L0/H3 ending still completes and unlocks the life
    # stage, even though it does not qualify as a child-teaching reward.
    stage_completion_eligible: bool = False
    # LLM 요약이 아니라 오케스트레이터가 슬롯 정의로 검증한 값만 담는다.
    # Spring BE가 카페 단계 완료를 안전하게 동기화할 때 사용한다.
    verified_facts: dict[str, str | int | float | bool] = Field(default_factory=dict)


class PedagogySnapshot(BaseModel):
    expression_level: ExpressionLevel
    hint_level: HintLevel
    subgoal_id: str
    verified_slots: dict[str, str | int | float | bool]
    bottleneck: str | None = None


class TaskAnchorCompletedItem(BaseModel):
    """One child-provided fact that the current task no longer asks for."""

    slot_id: str
    label: str
    value: str | int | float | bool
    # Reviewed context for clients that cannot render an internal canonical
    # value such as ``right`` on its own. This sentence is exposed only after
    # the corresponding value has been verified from the child's response.
    display_text: str


class TaskAnchorContract(BaseModel):
    """Stable, non-LLM reminder of what the child should answer right now."""

    anchor_id: str
    title: str = "지금 모르미에게 알려줄 것"
    # Keep the same copy contract as Mormi's speech: 50 characters is an
    # authoring target, not a runtime rejection boundary.  The anchor repeats
    # reviewed task copy, so rejecting a complete sentence here can make an
    # otherwise valid teaching session fail before its first turn.
    prompt: str = Field(min_length=1)
    completed_items: list[TaskAnchorCompletedItem] = Field(default_factory=list)
    target_slots: list[str] = Field(min_length=1)


class TurnContract(BaseModel):
    # Persisted turn JSON predates this explicit reader version.  The default
    # keeps those rows readable while every newly stored turn carries the
    # version that the aggregate V2 snapshot capability promises to decode.
    schema_version: Literal["turn-contract-v1"] = "turn-contract-v1"
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    scene: SceneType
    scenario_id: str
    task_id: str
    stage_id: str
    task_index: int
    mormi: MormiContract
    input: InputContract
    visual: VisualContract
    help_card: HelpCardContract | None = None
    note_update: NoteUpdate | None = None
    status: SessionStatus
    state_version: int
    completion: CompletionContract | None = None
    pedagogy: PedagogySnapshot | None = None
    # Optional only for backward compatibility with turns persisted before
    # this contract existed. Every newly generated active turn includes it.
    task_anchor: TaskAnchorContract | None = None
    # The card itself is pinned in SessionState. Turns expose only its stable
    # identity so clients never derive dictionary copy from help-card text.
    dictionary_ref: DictionaryReference | None = None


class PinnedDialogueRuntimeV2(BaseModel):
    """Immutable content identity plus monotonic V2 dialogue state.

    The full reviewed pack and generated-copy selections are pinned in the
    conversation JSON so a deploy or cache refresh cannot change an in-flight
    teaching round.  Dedicated V2 modules validate the nested pack and ledger
    payloads before using them; legacy readers simply preserve this additive
    field.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pinned-dialogue-runtime-v2"] = (
        "pinned-dialogue-runtime-v2"
    )
    pack_id: str = Field(min_length=1, max_length=160)
    content_version: int = Field(ge=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pack_snapshot: dict[str, Any]
    reasoning_ledger: dict[str, Any]
    # Stable-copy plans are compiled once from the pinned pack when the
    # conversation is created.  Keeping the exact five payloads (rather than
    # rebuilding them with the process's current compiler) makes an in-flight
    # conversation independent from later compiler-rule deployments.
    stable_copy_plan_schema_version: Literal["stable-copy-plan-set-v1"]
    stable_copy_plan_compiler_version: Literal["stable-copy-plan-compiler-v1"]
    stable_copy_plan_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stable_copy_plans: dict[str, dict[str, Any]] = Field(
        min_length=5,
        max_length=5,
    )
    copy_snapshots: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selector_reason: str = Field(min_length=1, max_length=100)
    canary_bucket: int | None = Field(default=None, ge=0, le=99)


ScenarioTaskIdV3 = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9_.-]*$", max_length=120),
]
ScenarioVariantIdV3 = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=100),
]
ScenarioEvidenceIdV3 = Annotated[
    str,
    Field(pattern=r"^[a-f0-9]{64}$"),
]


class PinnedDialogueTaskNoteStateV3(BaseModel):
    """Task-scoped, raw-free provenance for one life-scene note policy.

    Evidence IDs point at the task ledger; neither child text nor a model
    paraphrase is persisted here.  Keeping support and joint-performance state
    beside the task prevents a primary note from leaking into a transfer task.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["task-note-state-v3"] = "task-note-state-v3"
    independent_relation_evidence: dict[
        ScenarioTaskIdV3,
        list[ScenarioEvidenceIdV3],
    ] = Field(default_factory=dict, max_length=20)
    supported_relation_ids: list[ScenarioTaskIdV3] = Field(
        default_factory=list,
        max_length=20,
    )
    joint_performance_used: bool = False
    note_emitted: bool = False
    emitted_note_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_note_provenance(self) -> PinnedDialogueTaskNoteStateV3:
        if len(self.supported_relation_ids) != len(set(self.supported_relation_ids)):
            raise ValueError("task note supported relation IDs must be unique")
        if set(self.independent_relation_evidence).intersection(
            self.supported_relation_ids
        ):
            raise ValueError(
                "task note relation cannot be both independent and supported"
            )
        if any(
            not evidence_ids or len(evidence_ids) != len(set(evidence_ids))
            for evidence_ids in self.independent_relation_evidence.values()
        ):
            raise ValueError(
                "independent task note evidence IDs must be non-empty and unique"
            )
        if self.note_emitted != (self.emitted_note_id is not None):
            raise ValueError("emitted task note state must include exactly one note ID")
        return self


class PinnedDialogueScenarioRuntimeV3(BaseModel):
    """Immutable multi-task life-scene snapshot selected for one conversation.

    This is deliberately separate from :class:`PinnedDialogueRuntimeV2`.
    Existing home snapshots keep their exact schema, while a V3-capable reader
    can pin a complete café or amusement scenario and isolate every task's
    variant, reasoning ledger, and note provenance. Life copy is assembled from
    reviewed templates, so generated stable-copy cache state does not belong in
    this boundary.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pinned-dialogue-scenario-runtime-v3"] = (
        "pinned-dialogue-scenario-runtime-v3"
    )
    scenario_pack_id: str = Field(
        pattern=r"^[a-z][a-z0-9_.-]*$",
        max_length=160,
    )
    scenario_content_version: int = Field(ge=1)
    scenario_source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario_pack_snapshot: dict[str, Any]
    active_variant_ids: dict[ScenarioTaskIdV3, ScenarioVariantIdV3] = Field(
        min_length=1,
        max_length=4,
    )
    reasoning_ledgers: dict[ScenarioTaskIdV3, dict[str, Any]] = Field(
        min_length=1,
        max_length=4,
    )
    task_note_states: dict[
        ScenarioTaskIdV3,
        PinnedDialogueTaskNoteStateV3,
    ] = Field(min_length=1, max_length=4)
    selector_reason: str = Field(min_length=1, max_length=100)
    canary_bucket: int | None = Field(default=None, ge=0, le=99)

    @model_validator(mode="after")
    def task_scopes_must_match(self) -> PinnedDialogueScenarioRuntimeV3:
        task_ids = set(self.active_variant_ids)
        if set(self.reasoning_ledgers) != task_ids or set(self.task_note_states) != task_ids:
            raise ValueError(
                "scenario variant, ledger, and note state task scopes must match"
            )
        return self


class SessionState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: new_id("conversation"))
    learner_id: int
    learning_session_id: str | None = None
    conversation_round: int = Field(default=1, ge=1)
    scene: SceneType
    scenario_id: str
    task_ids: list[str]
    scenario_data: dict[str, Any] = Field(default_factory=dict)
    dictionary_catalog_version: int | None = None
    # Immutable-by-conversation snapshots keep reference material stable when
    # the catalog is deployed while a child is still in the same lesson.
    dictionary_snapshots: dict[str, DictionaryCard] = Field(default_factory=dict)
    task_start_levels: dict[str, ExpressionLevel] = Field(default_factory=dict)
    task_index: int = 0
    expression_level: ExpressionLevel
    hint_level: HintLevel = HintLevel.H0
    subgoal_id: str = "initial"
    # v1 predates entry turns, v2 used conditional wrong guesses, and v3 starts
    # every new session from a genuine help request. Persisted values remain
    # readable so deployments do not break conversations already in progress.
    dialogue_policy_version: int = Field(default=1, ge=1)
    # Runtime implementation is pinned once at conversation creation. Missing
    # values in pre-V2 JSON snapshots deliberately resolve to the legacy path.
    runtime_contract_version: DialogueRuntimeContractVersion = (
        DialogueRuntimeContractVersion.LEGACY_V1
    )
    pinned_dialogue_v2: PinnedDialogueRuntimeV2 | None = None
    # Life-scene conversations use a separate snapshot format. Keeping this
    # additive field optional preserves every legacy and single-pack V2 state.
    pinned_dialogue_scenario_v3: PinnedDialogueScenarioRuntimeV3 | None = None
    entry_phase: EntryPhase = EntryPhase.RESOLVED
    verified_slots: dict[str, str | int | float | bool] = Field(default_factory=dict)
    # Evidence must survive task changes. In particular, an amusement-park
    # transfer task must not erase the primary answer that BE later records.
    completed_task_slots: dict[str, dict[str, str | int | float | bool]] = Field(
        default_factory=dict
    )
    status: SessionStatus = SessionStatus.ACTIVE
    current_turn_id: str | None = None
    state_version: int = 1
    expression_failures: int = 0
    concept_failures: int = 0
    # One clarification is allowed for a safe, related-but-vague response.
    # A repeated vague response must change the visible support contract so
    # the child can never be trapped in the same free-text question.
    vague_clarifications: int = 0
    unrelated_count: int = 0
    task_start_level: ExpressionLevel | None = None
    task_max_hint: HintLevel = HintLevel.H0
    # Safe, fact-checked child wording is tracked per note slot.  Structured
    # choices/fills are tracked separately so provenance, wording and
    # attribution cannot be conflated at completion.
    child_note_evidence: dict[str, str] = Field(default_factory=dict)
    supported_note_slots: list[str] = Field(default_factory=list)
    all_tasks_direct: bool = True
    joint_performance_used: bool = False
    raw_storage_enabled: bool = True
    retention_policy: RetentionPolicy = RetentionPolicy.PERMANENT
    raw_retention_until: datetime | None = None
    completion_outcome: CompletionOutcome | None = None
    teach_reward_eligible: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def current_task_id(self) -> str:
        return self.task_ids[self.task_index]


class SpeakerQuestionIntent(BaseModel):
    """Meaning selected by code while leaving surface wording to the speaker."""

    operation: str = ""
    response_kind: Literal[
        "answer",
        "reason_or_method",
        "answer_and_reason",
        "choice",
        "count",
        "relation",
        "action",
        "joint",
        "none",
    ] = "none"
    referents: list[str] = Field(default_factory=list)
    required_meanings: dict[str, str] = Field(default_factory=dict)


class SpeakerGuardContract(BaseModel):
    """Closed-world provenance kept outside the speaker prompt."""

    child_expression_source: str | None = None


class NoteContextualizationContext(BaseModel):
    """Closed-world input for turning child evidence into a standalone note.

    The child wording remains the source of the mathematical idea.  Reviewed
    context may only resolve omitted subjects, demonstratives and scene-bound
    references such as ``둘이`` or ``오른쪽``.
    """

    skill_id: str
    note_context: str
    source_fragments: dict[str, str] = Field(min_length=1)
    reviewed_facts: dict[str, str] = Field(default_factory=dict)
    allowed_numbers: list[str] = Field(default_factory=list)
    fallback_text: str


class NoteContextualizationOutput(BaseModel):
    """Auditable note rewrite produced under the closed-world contract."""

    text: str
    source_slots_used: list[str] = Field(default_factory=list)
    source_spans_used: list[str] = Field(default_factory=list)
    fact_refs_used: list[str] = Field(default_factory=list)
    meaning_preserved: bool = False
    self_contained: bool = False
    introduced_math_content: bool = False


class SpeakerContext(BaseModel):
    dialogue_act: str
    task_relation: TaskRelation = TaskRelation.UNKNOWN
    interaction_intent: InteractionIntent = InteractionIntent.NONE
    interaction_repeat_count: int = 0
    support_trigger: SupportTrigger = SupportTrigger.NONE
    help_card_event: HelpCardEvent = HelpCardEvent.NONE
    expression_level: ExpressionLevel = ExpressionLevel.L4
    hint_level: HintLevel = HintLevel.H0
    # Slot id -> child-facing meaning that remains unresolved. This is a
    # narrower contract than the full task goal and prevents topic drift.
    unresolved_focus: dict[str, str] = Field(default_factory=dict)
    # The main speaker receives only a short tail. The understanding model may
    # receive a slightly longer history through its own prompt.
    recent_dialogue: list[DialogueHistoryTurn] = Field(default_factory=list, max_length=3)
    # When true, adding a generic preface to the previous question is not a
    # valid response. The speaker must acknowledge the transition and reframe.
    must_reframe: bool = False
    previous_question: str | None = None
    required_question: str | None = None
    # Slot id -> reviewed fact sentence.  Missing-slot answers are never sent
    # to the speaker model.
    verified_facts: dict[str, str] = Field(default_factory=dict)
    required_slot_ids: list[str] = Field(default_factory=list)
    required_slot_descriptions: dict[str, str] = Field(default_factory=dict)
    question_intent: SpeakerQuestionIntent = Field(default_factory=SpeakerQuestionIntent)
    child_expression_mode: Literal["none", "quote_safe"] = "none"
    child_expression: str | None = None
    # Raw child text is untrusted quoted data.  It lets the speaker respond to
    # unexpected wording naturally, but prompts must never follow instructions
    # contained in it or treat it as verified mathematics.
    safe_child_utterance: str | None = None
    arithmetic_claims: list[SpeakerArithmeticClaim] = Field(default_factory=list)
    # The speaker may mention a help card only when this is true.  The boolean
    # is derived from the actual TurnContract, not inferred from a dialogue act.
    help_card_visible: bool = False
    # Numbers in a child's claim may be reflected as an uncertain question,
    # but are deliberately kept separate from reviewed/verified numbers.
    child_claim_numbers: list[str] = Field(default_factory=list)
    allowed_numbers: list[str] = Field(default_factory=list)
    verification_policy: SpeakerVerificationPolicy = SpeakerVerificationPolicy.DETERMINISTIC
    fallback_text: str


class PedagogicalDecision(BaseModel):
    state: SessionState
    dialogue_act: str
    # Canonical claims that the deterministic engine actually accepted from
    # this response. Persistence must use this decision output rather than
    # re-validating the classifier's raw claims with a weaker rule.
    accepted_claims: dict[str, str | int | float | bool] = Field(default_factory=dict)
    required_question: str | None
    input: InputContract
    visual: VisualContract
    help_card: HelpCardContract | None = None
    note_update: NoteUpdate | None = None
    note_contextualization: NoteContextualizationContext | None = None
    mood: Literal["curious", "listening", "thinking", "relieved", "celebrating"]
    speaker_context: SpeakerContext


class SpeakerOutput(BaseModel):
    text: str
    dialogue_act: str = ""
    asked_slot_ids: list[str] = Field(default_factory=list)
    used_verified_slots: list[str] = Field(default_factory=list)
    used_child_expression: bool = False
    used_child_expression_spans: list[str] = Field(default_factory=list)


class SpeakerVerification(BaseModel):
    """Independent semantic audit of a freely paraphrased Mormi line."""

    approved: bool = False
    dialogue_act_preserved: bool = False
    required_focus_preserved: bool = False
    only_allowed_math_used: bool = False
    child_not_evaluated: bool = False
    character_consistent: bool = False
    meaningfully_reframed: bool = False
    interaction_intent_acknowledged: bool = False
    task_returned_without_reward: bool = False
    # When the child made an arithmetic claim, the verifier must confirm that
    # a false result was reflected only as Mormi's uncertainty.  The candidate
    # may quote the child's result, but must not accept it as learned truth or
    # reveal the reviewed correct answer.
    arithmetic_claim_stance_safe: bool = False
    help_card_state_respected: bool = False
    sentence_complete: bool = False
    joint_mode_respected: bool = False
    violation_codes: list[str] = Field(default_factory=list)
    detected_dialogue_act: str = ""
    detected_asked_slot_ids: list[str] = Field(default_factory=list)
    question_evidence_span: str = ""
    unverified_claim_spans: list[str] = Field(default_factory=list)
    answer_leak_spans: list[str] = Field(default_factory=list)
    child_evaluation_spans: list[str] = Field(default_factory=list)
    child_expression_spans: list[str] = Field(default_factory=list)
    false_claim_confirmation_spans: list[str] = Field(default_factory=list)
    reason_code: Literal[
        "approved",
        "wrong_focus",
        "answer_leak",
        "unverified_claim",
        "child_evaluation",
        "character_break",
        "other",
    ] = "other"


class SpeakerRuntimeAudit(BaseModel):
    """Non-linguistic audit trail for the line that reached the child.

    Generated candidate text is already stored as the next turn's plaintext
    question.  This object records *how* that text was selected without
    duplicating the child's or Mormi's raw language in analytics columns.
    """

    dialogue_act: str
    speaker_source: Literal[
        "reviewed_fallback",
        "llm",
        "bridge_llm",
        "stable_copy_cache",
        "stable_copy_fallback",
        "v2_safety_fallback",
        "classifier_bridge",
        "generation_fallback",
        "deterministic_validation_fallback",
        "semantic_verification_fallback",
    ]
    verifier_status: Literal[
        "not_required",
        "disabled",
        "approved",
        "rejected",
        "error",
    ] = "not_required"
    fallback_reason: str | None = Field(default=None, max_length=120)
    understanding_route: UnderstandingRoute = UnderstandingRoute.NORMAL
    adjudicator_used: bool = False
    speaker_latency_ms: int | None = Field(default=None, ge=0)
    verifier_latency_ms: int | None = Field(default=None, ge=0)
    runtime_contract_version: DialogueRuntimeContractVersion = (
        DialogueRuntimeContractVersion.LEGACY_V1
    )
    understanding_source: Literal[
        "legacy",
        "sonnet_medium",
        "sonnet_low",
        "deterministic_fallback",
        "structured_choice",
        "structured_joint",
        "explicit_no_response",
        "silence_timeout",
        "asr_empty",
    ] = "legacy"
    understanding_attempts: int = Field(default=0, ge=0, le=2)
    understanding_latency_ms: int | None = Field(default=None, ge=0)
    evidence_guard_status: Literal[
        "not_applicable",
        "passed",
        "retry_passed",
        "failed",
    ] = "not_applicable"
    new_progress: bool = False
    newly_verified_fact_ids: list[str] = Field(default_factory=list)
    newly_verified_relation_ids: list[str] = Field(default_factory=list)
    stable_copy_status: Literal[
        "not_applicable",
        "hit",
        "generated",
        "contended_fallback",
        "generation_fallback",
        "reviewed_fallback",
    ] = "not_applicable"
    stable_copy_key_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{16}$",
    )
    content_pack_id: str | None = Field(default=None, max_length=160)
    content_version: int | None = Field(default=None, ge=1)
    content_source_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class SessionEnvelope(BaseModel):
    conversation_id: str
    turn: TurnContract


class SessionSnapshot(BaseModel):
    state: SessionState
    turn: TurnContract


class StoredTurn(BaseModel):
    turn_id: str
    conversation_id: str
    task_id: str
    mormi_question: str
    response: ChildResponse | None = None
    safety_category: SafetyCategory | None = None
    created_at: datetime


class SkillProfile(BaseModel):
    skill_id: str
    highest_stable_expression_level: ExpressionLevel = ExpressionLevel.L2
    h0_success_streak: int = 0
    recent_max_hint: HintLevel = HintLevel.H0
    frequent_hint_types: list[str] = Field(default_factory=list)
    concept_mastery: float = Field(default=0.5, ge=0, le=1)
    expression_independence: float = Field(default=0.5, ge=0, le=1)
    last_bottleneck: str = "unknown"


class LearnerProfile(BaseModel):
    learner_id: int
    skills: dict[str, SkillProfile] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_configured: bool
    database: str
    environment: str
    runtime_contract_version: DialogueRuntimeContractVersion
    dialogue_v2_canary_percent: int = Field(ge=0, le=100)
    dialogue_runtime_capabilities: list[DialogueRuntimeContractVersion] = Field(
        default_factory=lambda: [DialogueRuntimeContractVersion.LEGACY_V1]
    )
    dialogue_snapshot_reader_capabilities: list[str] = Field(default_factory=list)
    conversation_identity_reader_capabilities: list[str] = Field(default_factory=list)
    conversation_identity_schema_phase: Literal[
        "transition", "final", "unchecked"
    ] = "unchecked"


class SkillProfilesResponse(BaseModel):
    learner_id: int
    skills: list[SkillProfile]


class StarNotesResponse(BaseModel):
    learner_id: int
    notes: list[NoteUpdate]


class ReportTurnEvidence(BaseModel):
    turn_id: str
    task_id: str
    response: str | None = None
    response_type: str | None = None
    response_category: str | None = None
    expression_level: ExpressionLevel
    hint_level: HintLevel
    pedagogy: dict[str, Any] | None = None
    created_at: datetime


class ReportConversationEvidence(BaseModel):
    conversation_id: str
    learning_session_id: str | None
    scene: SceneType
    scenario_id: str
    status: SessionStatus
    completion_outcome: CompletionOutcome | None
    teach_reward_eligible: bool
    verified_slots: dict[str, str | int | float | bool]
    task_max_hint: HintLevel
    turns: list[ReportTurnEvidence]
    created_at: datetime
    updated_at: datetime


class LadderLevelPerformance(BaseModel):
    correct: int = Field(ge=0)
    attempts: int = Field(ge=0)

    @model_validator(mode="after")
    def correct_must_not_exceed_attempts(self) -> LadderLevelPerformance:
        if self.correct > self.attempts:
            raise ValueError("correct must not exceed attempts")
        return self


class LadderAnalysisCreateRequest(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    learner_id: int = Field(ge=1)
    skill_id: str = Field(min_length=1, max_length=100)
    trigger_session_id: str = Field(min_length=1, max_length=100)
    session_ids: tuple[str, str]
    current_level: ExpressionLevel
    performance_by_level: dict[ExpressionLevel, LadderLevelPerformance]
    lower_rule_evidence_count: int = Field(default=0, ge=0)


class LadderAnalysisCreateResponse(BaseModel):
    analysis_id: str
    status: str


class LadderAnalysisApprovalRequest(BaseModel):
    learner_id: int = Field(ge=1)
    recommendation_version: int = Field(ge=1)


class LadderRecommendationEvidence(BaseModel):
    analysis_id: str
    learner_id: int
    skill_id: str
    trigger_session_id: str
    session_ids: list[str]
    current_level: ExpressionLevel
    recommended_level: ExpressionLevel
    action: str = Field(
        pattern=r"^(UPGRADE|MAINTAIN|ADJUST_DOWN|INSUFFICIENT_EVIDENCE)$"
    )
    current_accuracy: float | None = Field(default=None, ge=0, le=1)
    evidence_count: int = Field(ge=0)
    reason_code: str
    recent_predictions: list[dict[str, str | float]] = Field(default_factory=list)
    model_version: str
    recommendation_version: int
    status: str
    approved: bool
    analyzed_at: datetime


class ReportEvidenceResponse(BaseModel):
    learner_id: int
    conversations: list[ReportConversationEvidence]
    skills: list[SkillProfile]
    notes: list[NoteUpdate]
    ladder_recommendations: list[LadderRecommendationEvidence] = Field(default_factory=list)


class ReportFact(BaseModel):
    evidence_id: str = Field(pattern=r"^[a-z]+:[A-Za-z0-9_.:-]+$")
    category: Literal["concept", "explanation", "life", "improved", "observe"]
    statement: str = Field(min_length=1, max_length=240)


class ReportNarrative(BaseModel):
    text: str = Field(min_length=1, max_length=160)
    evidence_refs: list[str] = Field(min_length=1, max_length=5)


class ReportSummaryRequest(BaseModel):
    learner_label: str = Field(min_length=1, max_length=40)
    facts: list[ReportFact] = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> ReportSummaryRequest:
        evidence_ids = [fact.evidence_id for fact in self.facts]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("report evidence ids must be unique")
        return self


class ReportSummaryResponse(BaseModel):
    concept_performance: ReportNarrative
    explanation_change: ReportNarrative
    life_transfer: ReportNarrative
    improved_point: ReportNarrative
    observe_point: ReportNarrative

    def narratives(self) -> list[ReportNarrative]:
        return [
            self.concept_performance,
            self.explanation_change,
            self.life_transfer,
            self.improved_point,
            self.observe_point,
        ]


class SpeechChangeSample(BaseModel):
    utterance: str = Field(min_length=1, max_length=500)
    expression_level: str | None = Field(default=None, pattern=r"^L[0-4]$")
    hint_level: str | None = Field(default=None, pattern=r"^H[0-4]$")


class SpeechChangeSummaryRequest(BaseModel):
    domain_label: str = Field(min_length=1, max_length=40)
    past: SpeechChangeSample
    recent: SpeechChangeSample


class SpeechChangeSummaryResponse(BaseModel):
    text: str = Field(min_length=1, max_length=220)
    evidence_spans: list[str] = Field(min_length=2, max_length=6)


class ConflictDetail(BaseModel):
    code: Literal["stale_turn"] = "stale_turn"
    message: str
    conversation_id: str
    turn_id: str
    state_version: int


class ConflictResponse(BaseModel):
    detail: ConflictDetail
