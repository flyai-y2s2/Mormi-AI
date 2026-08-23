from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from .dictionary_models import DictionaryCard, DictionaryReference


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SceneType(StrEnum):
    HOME_TEACH = "home_teach"
    CAFE = "cafe"


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
    META_ABOUT_MORMI = "meta_about_mormi"
    OFF_TOPIC = "off_topic"
    UNKNOWN = "unknown"


class InteractionIntent(StrEnum):
    """Social intent kept separate from mathematical correctness."""

    NONE = "none"
    AUTHENTICITY_CHALLENGE = "authenticity_challenge"
    PLAYFUL_TEASE = "playful_tease"
    FRUSTRATION = "frustration"
    REFUSAL = "refusal"
    OTHER_SAFE_SOCIAL = "other_safe_social"


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


class SessionCreate(BaseModel):
    learner_id: int = Field(ge=1)
    scene: SceneType
    scenario_id: str = Field(min_length=1, max_length=100)
    learning_session_id: str | None = Field(default=None, max_length=100)
    practice_result_id: str | None = Field(default=None, max_length=100)
    practice_summary: PracticeSummary | None = None
    cafe_context: CafeSessionContext | None = None
    queue_context: QueueSessionContext | None = None
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
    operation: Literal["addition", "subtraction"]
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

    operation: Literal["addition", "subtraction"]
    source_text: str
    left: SpeakerQuantity
    right: SpeakerQuantity
    claimed_result: SpeakerQuantity
    truth_status: Literal["true", "false", "unknown"]
    related_slot_ids: list[str] = Field(default_factory=list)


class UtteranceAnalysis(BaseModel):
    safety_category: SafetyCategory = SafetyCategory.UNKNOWN
    response_category: ResponseCategory = ResponseCategory.RECOGNITION_OR_INPUT_ERROR
    difficulty_class: DifficultyClass = DifficultyClass.UNKNOWN
    # These fields describe the conversational job of a safe utterance.  They
    # never verify a mathematical claim or change a ladder by themselves.
    task_relation: TaskRelation = TaskRelation.UNKNOWN
    interaction_intent: InteractionIntent = InteractionIntent.NONE
    entry_stance: EntryStance = EntryStance.NOT_APPLICABLE
    claims: list[SlotClaim] = Field(default_factory=list)
    arithmetic_claims: list[ArithmeticClaim] = Field(default_factory=list)
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


class SessionState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: new_id("conversation"))
    learner_id: int
    learning_session_id: str | None = None
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
    entry_phase: EntryPhase = EntryPhase.RESOLVED
    verified_slots: dict[str, str | int | float | bool] = Field(default_factory=dict)
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
    """Validation-only facts that are never sent to the Sonnet speaker."""

    forbidden_answer_forms: list[str] = Field(default_factory=list)
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


class ConflictDetail(BaseModel):
    code: Literal["stale_turn"] = "stale_turn"
    message: str
    conversation_id: str
    turn_id: str
    state_version: int


class ConflictResponse(BaseModel):
    detail: ConflictDetail
