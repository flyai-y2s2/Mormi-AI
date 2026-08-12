from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


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

    @property
    def days(self) -> int | None:
        return {
            RetentionPolicy.NO_RAW: None,
            RetentionPolicy.DAYS_30: 30,
            RetentionPolicy.DAYS_90: 90,
        }[self]

    def expires_at(self, started_at: datetime) -> datetime | None:
        return started_at + timedelta(days=self.days) if self.days is not None else None


class ExpressionLevel(StrEnum):
    L4 = "L4"
    L3 = "L3"
    L2 = "L2"
    L1 = "L1"
    L0 = "L0"

    def lower(self) -> ExpressionLevel:
        order = list(ExpressionLevel)
        return order[min(order.index(self) + 1, len(order) - 1)]

    def higher(self) -> ExpressionLevel:
        order = list(ExpressionLevel)
        return order[max(order.index(self) - 1, 0)]

    @property
    def rank(self) -> int:
        return {"L4": 4, "L3": 3, "L2": 2, "L1": 1, "L0": 0}[self.value]


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
    EXPRESSION_BLOCK = "expression_block"
    CONCEPTUAL_ERROR = "conceptual_error"
    CONCEPTUAL_BLOCK = "conceptual_block"
    NO_RESPONSE = "no_response"
    RECOGNITION_OR_INPUT_ERROR = "recognition_or_input_error"
    UNRELATED_RESPONSE = "unrelated_response"
    HELP_REQUEST = "help_request"
    SELF_CORRECTION = "self_correction"


class DifficultyClass(StrEnum):
    EXPRESSION = "expression"
    CONCEPT = "concept"
    BOTH = "both"
    INPUT = "input"
    ENGAGEMENT = "engagement"
    UNKNOWN = "unknown"


class SafetyCategory(StrEnum):
    NORMAL = "normal"
    UNKNOWN = "unknown"
    PLAYFUL_OFFTOPIC = "playful_offtopic"
    SEXUAL = "sexual"
    PERSONAL_DATA = "personal_data"
    PROMPT_INJECTION = "prompt_injection"
    ABUSIVE = "abusive"
    DANGEROUS = "dangerous"


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
    response: str | int | float | list[str] | None = None
    misconception_tag: str | None = Field(default=None, max_length=80)
    latency_ms: int | None = Field(default=None, ge=0, le=600_000)


class PracticeSummary(BaseModel):
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


class SessionCreate(BaseModel):
    learner_id: int = Field(ge=1)
    scene: SceneType
    scenario_id: str = Field(min_length=1, max_length=100)
    learning_session_id: str | None = Field(default=None, max_length=100)
    practice_result_id: str | None = Field(default=None, max_length=100)
    practice_summary: PracticeSummary | None = None
    conversation_storage_consent: bool = False
    retention_policy: RetentionPolicy = RetentionPolicy.NO_RAW

    @model_validator(mode="after")
    def validate_storage_policy(self) -> SessionCreate:
        if self.conversation_storage_consent and self.retention_policy is RetentionPolicy.NO_RAW:
            raise ValueError("consented raw storage requires a finite retention_policy")
        if (
            not self.conversation_storage_consent
            and self.retention_policy is not RetentionPolicy.NO_RAW
        ):
            raise ValueError("retention_policy must be no_raw without storage consent")
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


class UtteranceAnalysis(BaseModel):
    safety_category: SafetyCategory = SafetyCategory.UNKNOWN
    response_category: ResponseCategory = ResponseCategory.RECOGNITION_OR_INPUT_ERROR
    difficulty_class: DifficultyClass = DifficultyClass.UNKNOWN
    claims: list[SlotClaim] = Field(default_factory=list)
    misconception_tag: str | None = None
    bottleneck: str = "unknown"
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
    text: str = Field(max_length=50)
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


class PedagogySnapshot(BaseModel):
    expression_level: ExpressionLevel
    hint_level: HintLevel
    subgoal_id: str
    verified_slots: dict[str, str | int | float | bool]
    bottleneck: str | None = None


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


class SessionState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: new_id("conversation"))
    learner_id: int
    learning_session_id: str | None = None
    scene: SceneType
    scenario_id: str
    task_ids: list[str]
    task_start_levels: dict[str, ExpressionLevel] = Field(default_factory=dict)
    task_index: int = 0
    expression_level: ExpressionLevel
    hint_level: HintLevel = HintLevel.H0
    subgoal_id: str = "initial"
    verified_slots: dict[str, str | int | float | bool] = Field(default_factory=dict)
    status: SessionStatus = SessionStatus.ACTIVE
    current_turn_id: str | None = None
    state_version: int = 1
    expression_failures: int = 0
    concept_failures: int = 0
    unrelated_count: int = 0
    task_start_level: ExpressionLevel | None = None
    task_max_hint: HintLevel = HintLevel.H0
    direct_note_candidate: str | None = None
    all_tasks_direct: bool = True
    raw_storage_enabled: bool = False
    retention_policy: RetentionPolicy = RetentionPolicy.NO_RAW
    raw_retention_until: datetime | None = None
    completion_outcome: CompletionOutcome | None = None
    teach_reward_eligible: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def current_task_id(self) -> str:
        return self.task_ids[self.task_index]


class SpeakerContext(BaseModel):
    dialogue_act: str
    previous_question: str | None = None
    required_question: str | None = None
    verified_facts: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    child_expression_mode: Literal["none", "context_only", "quote_safe"] = "none"
    child_expression: str | None = None
    allowed_numbers: list[str] = Field(default_factory=list)
    fallback_text: str


class PedagogicalDecision(BaseModel):
    state: SessionState
    dialogue_act: str
    required_question: str | None
    input: InputContract
    visual: VisualContract
    help_card: HelpCardContract | None = None
    note_update: NoteUpdate | None = None
    mood: Literal["curious", "listening", "thinking", "relieved", "celebrating"]
    speaker_context: SpeakerContext


class SpeakerOutput(BaseModel):
    text: str
    used_verified_slots: list[str] = Field(default_factory=list)
    used_child_expression: bool = False


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


class ConflictDetail(BaseModel):
    code: Literal["stale_turn"] = "stale_turn"
    message: str
    conversation_id: str
    turn_id: str
    state_version: int


class ConflictResponse(BaseModel):
    detail: ConflictDetail
