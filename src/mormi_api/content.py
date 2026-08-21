from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from .copy_quality import validate_child_facing_math_copy
from .schemas import (
    QUEUE_MAX_COUNT,
    QUEUE_MIN_COUNT,
    CafeMenuItem,
    CafeSessionContext,
    ChoiceOption,
    ExpressionLevel,
    HintLevel,
    InputContract,
    InputKind,
    QueueSessionContext,
    SceneType,
    SlotClaim,
    VisualContract,
)

_VAGUE_OR_UNREVIEWED_COPY = re.compile(
    r"(어떤 방법이 맞을까|지금 상황|지금 장면|퍼진 넓이|느낌으로|"
    r"눈대중|한눈에 대충|색만 보기|크기만 보기|모양만 보기)"
)

# 모르미는 아이의 사고를 평가하는 교사가 아니라, 자신이 헷갈리는
# 지점을 아이에게 묻는 동생이다. 이 표현들은 문법적으로 자연스러워도
# 질문의 권력관계를 교사-학생으로 되돌리므로 콘텐츠 등록 단계에서 막는다.
_TEACHER_EVALUATION_COPY = re.compile(
    r"(왜\s+.+(?:라고\s+)?생각했어|왜\s+그렇게\s+생각|어떻게\s+알았어|"
    r"어떻게\s+[^?]*(?:했어|셌어|찾았어|읽었어|비교했어)|까닭은\s+무엇|"
    r"이유를\s*(?:말|설명)|설명해\s*봐|말해\s*봐)"
)

# 도움 카드는 모르미의 대사가 아니라, 아이가 문제를 계속 풀 수 있도록
# 화면이 제공하는 검수된 발판이다.  이 표현들은 문법적으로는 성립해도
# 무엇을 보고 무엇을 해야 하는지 특정하지 않아 힌트로 기능하지 않는다.
_AMBIGUOUS_HELP_COPY = re.compile(
    r"(큰\s*값부터|다음\s*돈|이어\s*더해|차례로\s*더하면\s*쉬워|"
    r"그것을|이것을|그다음\s*것|위의\s*것|아래의\s*것)"
)

HelpSupportType = Literal["attention", "guided_action", "joint_model"]
HelpAnswerPolicy = Literal["hidden", "partial", "revealed"]
HelpMethodPolicy = Literal["open_methods", "target_method"]
SlotEvaluationMode = Literal["auto", "canonical_value", "semantic_support"]
HelpSupportMode = Literal[
    "attention",
    "guided_equation",
    "guided_highlight",
    "guided_manipulation",
    "guided_sequence",
    "guided_choice",
    "joint_model",
]
HelpSkill = Literal[
    "counting",
    "comparison",
    "place_value",
    "addition",
    "subtraction",
    "budget",
    "queue",
    "selection",
    "grouping",
    "pattern",
    "time",
    "measurement",
    "geometry",
    "data",
]

_HELP_CARD_PROFILE: dict[HintLevel, tuple[HelpSupportType, HelpAnswerPolicy]] = {
    HintLevel.H1: ("attention", "hidden"),
    HintLevel.H2: ("guided_action", "partial"),
    HintLevel.H3: ("joint_model", "revealed"),
}

_GUIDED_SUPPORT_MODES = {
    "guided_equation",
    "guided_highlight",
    "guided_manipulation",
    "guided_sequence",
    "guided_choice",
}

_HELP_SKILL_SUPPORT_MODES: dict[HelpSkill, set[HelpSupportMode]] = {
    "counting": {"guided_sequence", "guided_manipulation", "guided_highlight"},
    "comparison": {"guided_choice", "guided_highlight", "guided_manipulation"},
    "place_value": {"guided_equation", "guided_manipulation"},
    "addition": {"guided_equation", "guided_manipulation", "guided_sequence"},
    "subtraction": {"guided_equation", "guided_manipulation", "guided_sequence"},
    "budget": {"guided_equation", "guided_choice"},
    "queue": {"guided_choice", "guided_sequence"},
    "selection": {"guided_choice", "guided_equation"},
    "grouping": {"guided_manipulation", "guided_equation", "guided_sequence"},
    "pattern": {"guided_highlight", "guided_equation", "guided_sequence"},
    "time": {"guided_highlight", "guided_sequence"},
    "measurement": {"guided_manipulation", "guided_choice", "guided_highlight"},
    "geometry": {"guided_choice", "guided_manipulation", "guided_highlight"},
    "data": {"guided_choice", "guided_manipulation", "guided_highlight"},
}

_DEFAULT_GUIDED_SUPPORT_MODE: dict[HelpSkill, HelpSupportMode] = {
    "counting": "guided_sequence",
    "comparison": "guided_choice",
    "place_value": "guided_equation",
    "addition": "guided_equation",
    "subtraction": "guided_equation",
    "budget": "guided_equation",
    "queue": "guided_sequence",
    "selection": "guided_choice",
    "grouping": "guided_manipulation",
    "pattern": "guided_sequence",
    "time": "guided_sequence",
    "measurement": "guided_manipulation",
    "geometry": "guided_manipulation",
    "data": "guided_highlight",
}

# These lessons explicitly practise one named representation or procedure.
# Other lessons remain open-method: the help card may offer one route, but the
# classifier must still accept any mathematically valid child explanation.
_TARGET_METHOD_HOME_ITEMS = {
    "number-count",
    "number-make-ten",
    "number-place-value",
    "add-place",
    "add-make-ten",
    "sub-place",
    "sub-borrow",
    "multiply-groups",
    "multiply-addition",
    "multiply-easy-tables",
    "multiply-tables",
    "divide-share",
    "divide-group",
    "pattern-repeat",
    "pattern-number",
    "pattern-unknown",
    "clock-basic",
    "clock-quarter",
    "time-duration",
    "time-calendar",
    "measure-compare",
    "measure-ruler",
    "geometry-compose",
    "geometry-position",
    "data-classify",
    "data-chart",
}


class SlotDefinition(BaseModel):
    id: str
    description: str
    # Content declares the semantic job of a slot.  The classifier therefore
    # reasons from a shared contract instead of guessing from a slot name or
    # memorizing one lesson's sample wording.
    semantic_role: Literal[
        "observation",
        "conclusion",
        "operation",
        "method",
        "reason",
        "explanation",
        "selection",
    ]
    expected: str | int | float | bool
    aliases: list[str] = Field(default_factory=list)
    accepted_values: list[str | int | float | bool] = Field(default_factory=list)
    preserve_value: bool = False
    fact_sentence: str
    # ``auto`` maps observable/result slots to deterministic values and maps
    # method/reason/explanation slots to grounded semantic support.
    evaluation_mode: SlotEvaluationMode = "auto"

    @property
    def resolved_evaluation_mode(self) -> Literal["canonical_value", "semantic_support"]:
        if self.evaluation_mode != "auto":
            return self.evaluation_mode
        if self.semantic_role in {"method", "reason", "explanation"}:
            return "semantic_support"
        return "canonical_value"

    @property
    def is_semantic_support(self) -> bool:
        return self.resolved_evaluation_mode == "semantic_support"

    def accepts(self, value: object) -> bool:
        if value == self.expected:
            return True
        # Children and the classifier can write the same amount either as
        # ``6,000원`` or ``6000원``.  Commas are presentation, not meaning, so
        # they must never decide whether a reviewed numeric fact is accepted.
        # Keep the normalization deliberately narrow: units and surrounding
        # words still have to match an explicit alias below.
        normalized = (
            str(value)
            .strip()
            .lower()
            .replace(" ", "")
            .replace(",", "")
        )
        # Language understanding already happened in the classifier. Compare
        # its structured numeric value with the reviewed numeric answer even
        # when curriculum copy stores that answer as ``"1,200원"``. Do not
        # parse Korean sentence endings here; the orchestrator only normalizes
        # closed values and units.
        def closed_numeric(item: object) -> float | None:
            if isinstance(item, bool):
                return None
            if isinstance(item, (int, float)):
                return float(item)
            match = re.fullmatch(
                r"([+-]?\d+(?:\.\d+)?)(?:원|개|명)?",
                str(item).strip().replace(" ", "").replace(",", ""),
            )
            return float(match.group(1)) if match is not None else None

        expected_numeric = closed_numeric(self.expected)
        value_numeric = closed_numeric(value)
        if expected_numeric is not None and value_numeric == expected_numeric:
            return True
        candidates = [
            str(self.expected),
            *self.aliases,
            *(str(item) for item in self.accepted_values),
        ]
        return normalized in {
            item.strip().lower().replace(" ", "").replace(",", "")
            for item in candidates
        }

    def canonical(self, value: object) -> str | int | float | bool:
        if self.preserve_value and isinstance(value, (str, int, float, bool)):
            return value
        return self.expected

    def accepted_claim_value(self, claim: SlotClaim) -> str | int | float | bool | None:
        """Return the reviewed state value for one classifier claim.

        Semantic slots store only satisfaction, not a synthetic method code.
        ``supported=None`` remains a compatibility path for deterministic
        button/fill claims and pre-refactor fixtures.
        """

        if not claim.factual:
            return None
        if self.is_semantic_support:
            if claim.supported is True:
                return True
            if claim.supported is None and self.accepts(claim.value):
                return True
            return None
        if self.accepts(claim.value):
            return self.canonical(claim.value)
        return None

    def equivalent_state_value(self, left: object, right: object) -> bool:
        if self.is_semantic_support:
            # Older conversations persisted representative method codes.
            # Any non-null legacy value still means this slot was satisfied.
            return left is not None and right is not None
        return left == right


class StepDefinition(BaseModel):
    id: str
    # Dialogue copy targets 50 characters, but complete reviewed sentences may exceed it.
    prompt: str = Field(min_length=1)
    target_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    input: InputContract
    choice_effects: dict[str, dict[str, str | int | float | bool]] = Field(default_factory=dict)
    fallback_text: str = Field(min_length=1)


class HelpPlanStep(BaseModel):
    body: str = Field(min_length=1, max_length=50)
    support_type: HelpSupportType
    answer_policy: HelpAnswerPolicy
    support_mode: HelpSupportMode
    fact_refs: list[str] = Field(min_length=1)
    action: str | None = Field(default=None, min_length=1, max_length=50)


class HintDefinition(HelpPlanStep):
    level: HintLevel
    visual_type: str | None = None
    visual_data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_help_contract(self) -> HintDefinition:
        expected_support, expected_answer_policy = _HELP_CARD_PROFILE[self.level]
        if self.support_type != expected_support:
            raise ValueError(f"{self.level}: support_type must be {expected_support}")
        if self.answer_policy != expected_answer_policy:
            raise ValueError(f"{self.level}: answer_policy must be {expected_answer_policy}")
        if self.level is HintLevel.H1 and self.support_mode != "attention":
            raise ValueError("H1: support_mode must direct attention")
        if self.level is HintLevel.H2 and self.support_mode not in _GUIDED_SUPPORT_MODES:
            raise ValueError("H2: support_mode must provide one stronger guided support")
        if self.level is HintLevel.H3 and self.support_mode != "joint_model":
            raise ValueError("H3: support_mode must be a complete joint model")
        if self.level in {HintLevel.H2, HintLevel.H3} and not self.action:
            raise ValueError(f"{self.level}: supported help needs one concrete action")
        if _AMBIGUOUS_HELP_COPY.search(self.body):
            raise ValueError("help-card copy contains an ambiguous action or reference")
        return self


def reviewed_help_card(
    level: HintLevel,
    body: str,
    *,
    support_mode: HelpSupportMode,
    fact_refs: Sequence[str],
    action: str | None = None,
    visual_type: str | None = None,
    visual_data: Mapping[str, Any] | None = None,
) -> HintDefinition:
    """Build one help card from the fixed H1/H2/H3 pedagogical contract."""

    support_type, answer_policy = _HELP_CARD_PROFILE[level]
    return HintDefinition(
        level=level,
        body=body,
        support_type=support_type,
        answer_policy=answer_policy,
        support_mode=support_mode,
        fact_refs=list(fact_refs),
        action=action,
        visual_type=visual_type,
        visual_data=dict(visual_data or {}),
    )


class ArithmeticValidationContract(BaseModel):
    """Reviewed arithmetic truth for one task instance.

    The classifier may understand a child's wording, but it must not be the
    only authority deciding whether an explicit numerical relation is true.
    Keeping the concrete operands and result beside the task lets the engine
    reject a fluent but false explanation without prescribing one wording or
    one calculation strategy.
    """

    operation: Literal["addition", "subtraction"]
    left: int
    right: int
    result: int
    left_label: str = "첫 번째 수"
    right_label: str = "두 번째 수"
    result_label: str = "계산 결과"
    unit: str = ""

    @model_validator(mode="after")
    def validate_result(self) -> ArithmeticValidationContract:
        expected = (
            self.left + self.right
            if self.operation == "addition"
            else self.left - self.right
        )
        if self.result != expected:
            raise ValueError("arithmetic validation contract has an inconsistent result")
        return self


class TaskDefinition(BaseModel):
    id: str
    dictionary_card_id: str
    scene: SceneType
    stage_id: str
    skill_id: str
    help_skills: list[HelpSkill] = Field(min_length=1)
    help_method_policy: HelpMethodPolicy
    accepted_methods: list[str] = Field(min_length=1)
    title: str
    goal: str
    visible_facts: dict[str, Any]
    arithmetic_contract: ArithmeticValidationContract | None = None
    slots: dict[str, SlotDefinition]
    required_slots: list[str]
    steps: dict[ExpressionLevel, list[StepDefinition]]
    hints: dict[HintLevel, HintDefinition]
    base_visual: VisualContract
    misconception_tags: list[str]
    coauthored_note: str
    # Reviewed, strategy-neutral context for a direct star-note entry.  Code
    # wraps the child's factual wording with this context so a short phrase is
    # understandable on its own without importing the curriculum's model
    # strategy.
    note_context: str = ""
    # Optional reviewed conclusion that may be appended only when every note
    # slot was directly grounded in the child's text.  It may restate those
    # verified slots, but must not introduce a new strategy.
    note_direct_conclusion: str = ""
    # Slots whose evidence belongs in the star note.  This can be narrower
    # than the completion contract: a concrete answer may finish a task while
    # the child's generalizable method is the only note-worthy contribution.
    note_slots: list[str] = Field(default_factory=list)
    # Free-text claims for these slots need an exact, note-worthy explanatory
    # span.  A bare result may fill an answer slot, but cannot satisfy a method
    # or reason slot even if the classifier overclaims it.
    text_explanation_slots: list[str] = Field(default_factory=list)
    # Compatibility fields for conversations snapshotted under dialogue policy
    # v2.  New v3 sessions never activate a wrong-guess entry and start from
    # the genuine L4 help request instead.
    entry_mode: Literal["wrong_guess", "incomplete_attempt", "genuine_question"] = (
        "genuine_question"
    )
    entry_step: StepDefinition | None = None
    behavior: str = "teaching"
    note_policy: str = "stage"
    transition_text: str | None = None

    @model_validator(mode="after")
    def validate_help_plan(self) -> TaskDefinition:
        step_groups = list(self.steps.values())
        if self.entry_step is not None:
            step_groups.append([self.entry_step])
        for steps in step_groups:
            for step in steps:
                unknown_slots = set(step.target_slots) - set(self.slots)
                if unknown_slots:
                    raise ValueError(
                        f"{self.id}/{step.id}: unknown target slots {sorted(unknown_slots)}"
                    )
                declared_input_slots = [*step.target_slots, *step.optional_slots]
                if (
                    len(declared_input_slots) != len(set(declared_input_slots))
                    or set(declared_input_slots) != set(step.input.target_slots)
                ):
                    raise ValueError(
                        f"{self.id}/{step.id}: required/optional and input target slots "
                        "must match"
                    )
        required_levels = {HintLevel.H1, HintLevel.H2, HintLevel.H3}
        if set(self.hints) != required_levels:
            raise ValueError("every task needs exactly one H1, H2 and H3 help card")
        reviewed_refs = set(self.visible_facts) | set(self.slots)
        visible_refs = set(self.visible_facts)
        allowed_guided_modes: set[HelpSupportMode] = set()
        for help_skill in self.help_skills:
            allowed_guided_modes.update(_HELP_SKILL_SUPPORT_MODES[help_skill])
        normalized_methods = {
            re.sub(r"\s+", "", method) for method in self.accepted_methods if method.strip()
        }
        if len(normalized_methods) != len(self.accepted_methods):
            raise ValueError(f"{self.id}: accepted help methods must be non-empty and unique")
        if self.hints[HintLevel.H2].support_mode not in allowed_guided_modes:
            raise ValueError(
                f"{self.id}: H2 support mode does not match declared help skills"
            )
        bodies: set[str] = set()
        for level in (HintLevel.H1, HintLevel.H2, HintLevel.H3):
            hint = self.hints[level]
            if hint.level is not level:
                raise ValueError(f"{self.id}: hint key and declared level must match")
            missing_refs = set(hint.fact_refs) - reviewed_refs
            if missing_refs:
                raise ValueError(f"{self.id}/{level}: unknown fact refs {sorted(missing_refs)}")
            if hint.answer_policy != "revealed":
                hidden_answer_refs = {
                    ref
                    for ref in hint.fact_refs
                    if ref not in visible_refs
                    and ref in self.slots
                    and self.slots[ref].semantic_role == "conclusion"
                }
                if hidden_answer_refs:
                    raise ValueError(
                        f"{self.id}/{level}: unrevealed help references final answer slots"
                    )
            normalized_body = re.sub(r"\s+", "", hint.body)
            if normalized_body in bodies:
                raise ValueError(f"{self.id}: different help levels need different visible copy")
            bodies.add(normalized_body)
        joint_steps = [
            step
            for step in self.steps.get(ExpressionLevel.L0, [])
            if step.input.kind is InputKind.JOINT
        ]
        if not joint_steps:
            raise ValueError(f"{self.id}: H3 needs an L0 joint-performance step")
        joint_completion_slots = {
            slot_id
            for step in joint_steps
            for slot_id in (step.input.config.get("completion_values") or {})
        }
        missing_completion_slots = set(self.required_slots) - joint_completion_slots
        if missing_completion_slots:
            raise ValueError(
                f"{self.id}: L0 cannot complete required slots "
                f"{sorted(missing_completion_slots)}"
            )
        return self

    def step_for(
        self,
        level: ExpressionLevel,
        verified_slots: Mapping[str, object],
    ) -> StepDefinition:
        level_steps = self.steps[level]
        for step in level_steps:
            if any(slot not in verified_slots for slot in step.target_slots):
                return step
        return level_steps[-1]

    def active_step(
        self,
        level: ExpressionLevel,
        verified_slots: Mapping[str, object],
        *,
        entry_active: bool,
        targeted_followup: bool = False,
    ) -> StepDefinition:
        """Resolve the exact prompt whose response is being interpreted."""

        if self.entry_step is not None and entry_active:
            return self.entry_step
        # A substantive but incomplete response to the entry sequence keeps
        # the child's L4 credit.  Only the next question is split so it asks
        # for the one missing idea instead of repeating a two-part prompt.
        if targeted_followup and level is ExpressionLevel.L4:
            return self.step_for(ExpressionLevel.L3, verified_slots)
        return self.step_for(level, verified_slots)

    def step_by_id(self, step_id: str) -> StepDefinition | None:
        """Resolve a persisted subgoal without depending on today's ladder state."""

        if self.entry_step is not None and self.entry_step.id == step_id:
            return self.entry_step
        for steps in self.steps.values():
            for step in steps:
                if step.id == step_id:
                    return step
        return None

    def missing_slots(self, verified_slots: Mapping[str, object]) -> list[str]:
        return [slot for slot in self.required_slots if slot not in verified_slots]

    def complete(self, verified_slots: Mapping[str, object]) -> bool:
        return not self.missing_slots(verified_slots)

    @property
    def effective_note_slots(self) -> list[str]:
        return self.note_slots or self.required_slots

    def validated_slot_claims(
        self,
        claims: Iterable[SlotClaim],
    ) -> dict[str, str | int | float | bool]:
        """Validate classifier claims under each slot's evaluation contract."""

        verified: dict[str, str | int | float | bool] = {}
        for claim in claims:
            slot = self.slots.get(claim.slot_id)
            if slot is None:
                continue
            value = slot.accepted_claim_value(claim)
            if value is not None:
                verified[claim.slot_id] = value
        return verified

    @property
    def semantic_support_slots(self) -> set[str]:
        return {
            slot_id
            for slot_id, slot in self.slots.items()
            if slot.is_semantic_support
        }


class ScenarioDefinition(BaseModel):
    id: str
    scene: SceneType
    title: str
    task_ids: list[str]


class HomeHelpPlan(BaseModel):
    """Three explicit help levels required by every home-teaching item."""

    H1: HelpPlanStep
    H2: HelpPlanStep
    H3: HelpPlanStep

    @model_validator(mode="after")
    def validate_distinct_support(self) -> HomeHelpPlan:
        steps = {
            HintLevel.H1: self.H1,
            HintLevel.H2: self.H2,
            HintLevel.H3: self.H3,
        }
        bodies = tuple(step.body for step in steps.values())
        normalized = {re.sub(r"\s+", "", body) for body in bodies}
        if len(normalized) != len(bodies):
            raise ValueError("H1, H2 and H3 help cards must have distinct jobs and copy")
        if any(_AMBIGUOUS_HELP_COPY.search(body) for body in bodies):
            raise ValueError("help-card copy contains an ambiguous action or reference")
        for level, step in steps.items():
            expected_support, expected_answer_policy = _HELP_CARD_PROFILE[level]
            if step.support_type != expected_support:
                raise ValueError(f"{level}: support_type must be {expected_support}")
            if step.answer_policy != expected_answer_policy:
                raise ValueError(f"{level}: answer_policy must be {expected_answer_policy}")
        if self.H1.support_mode != "attention":
            raise ValueError("H1 must direct attention")
        if self.H2.support_mode not in _GUIDED_SUPPORT_MODES or not self.H2.action:
            raise ValueError("H2 must provide one concrete guided support")
        if self.H3.support_mode != "joint_model" or not self.H3.action:
            raise ValueError("H3 must provide one complete joint model")
        return self

    def step_for(self, level: HintLevel) -> HelpPlanStep:
        if level is HintLevel.H1:
            return self.H1
        if level is HintLevel.H2:
            return self.H2
        if level is HintLevel.H3:
            return self.H3
        raise ValueError("H0 does not have a help-card body")


_LEGACY_HOME_HELP_SKILLS: dict[str, list[HelpSkill]] = {
    "number-count": ["counting"],
    "number-compare": ["counting", "comparison"],
    "money-count": ["addition"],
    "number-make-ten": ["counting", "addition"],
    "number-place-value": ["place_value", "addition"],
    "add-pictures": ["counting", "addition"],
    "money-price": ["addition"],
    "add-place": ["place_value", "addition"],
    "add-make-ten": ["addition"],
    "sub-pictures": ["counting", "subtraction"],
    "money-budget": ["subtraction"],
    "sub-place": ["place_value", "subtraction"],
    "sub-borrow": ["place_value", "subtraction"],
    "multiply-groups": ["counting", "grouping"],
    "multiply-addition": ["addition", "grouping"],
    "money-mission": ["addition", "subtraction"],
    "multiply-easy-tables": ["counting", "grouping"],
    "multiply-tables": ["addition", "grouping"],
    "divide-share": ["grouping"],
    "divide-group": ["grouping"],
    "pattern-repeat": ["pattern"],
    "pattern-number": ["comparison", "pattern"],
    "pattern-unknown": ["subtraction", "pattern"],
    "clock-basic": ["time"],
    "clock-quarter": ["counting", "time"],
    "time-duration": ["time"],
    "time-calendar": ["counting", "time"],
    "measure-compare": ["comparison", "measurement"],
    "measure-ruler": ["measurement"],
    "measure-weight-capacity": ["comparison", "measurement"],
    "geometry-shapes": ["counting", "geometry"],
    "geometry-compose": ["geometry"],
    "geometry-position": ["geometry"],
    "data-classify": ["data"],
    "data-chart": ["comparison", "data"],
    "data-chance": ["comparison", "data"],
}


class HomeTeachingSpec(BaseModel):
    """Reviewed teaching content for one frontend curriculum session."""

    id: str
    dictionary_card_id: str
    subject: str
    unit: str
    title: str
    help_skills: list[HelpSkill] = Field(min_length=1)
    help_method_policy: HelpMethodPolicy
    accepted_methods: list[str] = Field(min_length=1)
    # ``misconception_prompt`` is retained only so conversations snapshotted
    # before content v2 can still resume.  New catalog entries use l4_prompt.
    content_version: int = Field(default=1, ge=1)
    entry_mode: Literal["wrong_guess", "incomplete_attempt", "genuine_question"] = (
        "genuine_question"
    )
    entry_prompt: str | None = Field(default=None, min_length=1)
    l4_prompt: str | None = Field(default=None, min_length=1)
    misconception_prompt: str | None = Field(default=None, min_length=1)
    learned_line: str = Field(max_length=120)
    note_context: str = Field(min_length=1, max_length=80)
    fill_before: str
    fill_after: str
    fill_correct: str
    fill_options: list[str] = Field(min_length=2, max_length=6)
    short_prompt: str = Field(min_length=1)
    short_correct: str
    short_options: list[str] = Field(min_length=2, max_length=6)
    help_plan: HomeHelpPlan
    # Reviewed alternatives let the classifier recognize mathematically valid
    # child explanations without forcing one textbook strategy or wording.
    valid_explanations: list[str] = Field(default_factory=list)
    misconception: str
    sample_problem: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_help_copy(cls, raw: Any) -> Any:
        """Resume old snapshots while removing the old dead-field contract.

        Catalog v1 stored ``hint`` plus a list whose first item was never
        rendered.  Existing conversations may still contain that shape, so
        convert it at the boundary instead of keeping two live sources.
        """

        if not isinstance(raw, Mapping):
            return raw
        data = dict(raw)
        content_version = int(data.get("content_version", 1))
        if "help_skills" not in data and content_version <= 6:
            legacy_id = data.get("id")
            if isinstance(legacy_id, str) and legacy_id in _LEGACY_HOME_HELP_SKILLS:
                data["help_skills"] = _LEGACY_HOME_HELP_SKILLS[legacy_id]
        legacy_id = data.get("id")
        if "help_method_policy" not in data and content_version <= 6:
            data["help_method_policy"] = (
                "target_method" if legacy_id in _TARGET_METHOD_HOME_ITEMS else "open_methods"
            )
        if "accepted_methods" not in data and content_version <= 6:
            candidates = (
                [data.get("short_correct"), data.get("learned_line")]
                if data.get("help_method_policy") == "target_method"
                else [*(data.get("valid_explanations") or []), data.get("learned_line")]
            )
            data["accepted_methods"] = list(
                dict.fromkeys(item for item in candidates if isinstance(item, str) and item.strip())
            )
        if "help_plan" in data:
            return data
        legacy_cards = data.pop("help_cards", None)
        legacy_hint = data.pop("hint", None)
        legacy_lines = data.pop("help_lines", None)
        learned_line = data.get("learned_line")
        sample_problem = data.get("sample_problem")
        if isinstance(legacy_cards, Mapping):
            legacy_hint = legacy_cards.get("attention")
            legacy_lines = [
                legacy_cards.get("guided_action"),
                legacy_cards.get("joint_model"),
            ]
        if not isinstance(legacy_hint, str) or not legacy_hint.strip():
            return raw
        if not isinstance(legacy_lines, list) or not legacy_lines:
            return raw
        guided = legacy_lines[-1]
        joint = learned_line
        if not isinstance(guided, str) or not isinstance(joint, str):
            return raw
        # Some legacy entries reused the exact H1 copy at H2.  A snapshot must
        # remain resumable even then, so use its reviewed first line when that
        # gives the two levels distinct visible support.
        if re.sub(r"\s+", "", legacy_hint) == re.sub(r"\s+", "", guided):
            first_line = legacy_lines[0]
            if isinstance(first_line, str) and first_line.strip():
                guided = first_line
        sample_ref = "sample_problem"
        answer_ref = "sample_answer"
        if not isinstance(sample_problem, Mapping):
            return raw
        correct = sample_problem.get("correct")
        normalized_correct = re.sub(r"[\s,]", "", str(correct))
        if normalized_correct not in re.sub(r"[\s,]", "", joint):
            joint = f"이 문제의 답은 {correct}이야."
        legacy_skills = data.get("help_skills")
        first_skill = (
            legacy_skills[0] if isinstance(legacy_skills, list) and legacy_skills else None
        )
        guided_mode = _DEFAULT_GUIDED_SUPPORT_MODE.get(
            cast(HelpSkill, first_skill), "guided_highlight"
        )
        data["help_plan"] = {
            "H1": {
                "body": legacy_hint,
                "support_type": "attention",
                "answer_policy": "hidden",
                "support_mode": "attention",
                "fact_refs": [sample_ref],
            },
            "H2": {
                "body": guided,
                "support_type": "guided_action",
                "answer_policy": "partial",
                "support_mode": guided_mode,
                "fact_refs": [sample_ref],
                "action": guided,
            },
            "H3": {
                "body": joint,
                "support_type": "joint_model",
                "answer_policy": "revealed",
                "support_mode": "joint_model",
                "fact_refs": [sample_ref, answer_ref],
                "action": "완성된 문장을 함께 읽기",
            },
        }
        return data

    @property
    def hint(self) -> str:
        """Read-only compatibility projection for pre-v4 callers."""

        return self.help_plan.H1.body

    @property
    def help_lines(self) -> list[str]:
        """Read-only compatibility projection; no catalog field is discarded."""

        return [self.help_plan.H2.body, self.help_plan.H3.body]

    @property
    def effective_l4_prompt(self) -> str:
        prompt = self.l4_prompt or self.misconception_prompt
        if not prompt:
            raise ValueError(f"{self.id}: l4_prompt is required")
        return prompt

    @model_validator(mode="after")
    def validate_turn_coherence_contract(self) -> HomeTeachingSpec:
        prompt = self.sample_problem.get("prompt")
        correct = self.sample_problem.get("correct")
        answers = self.sample_problem.get("answers")
        visual = self.sample_problem.get("visual")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("sample_problem.prompt is required")
        if not prompt.rstrip().endswith("?"):
            raise ValueError("sample_problem.prompt must be a complete child-facing question")
        if not isinstance(answers, list) or len(answers) < 2:
            raise ValueError("sample_problem.answers needs at least two choices")
        if correct not in answers:
            raise ValueError("sample_problem.correct must be one of answers")
        if not isinstance(visual, Mapping) or not visual.get("type"):
            raise ValueError("sample_problem.visual.type is required")
        normalized_methods = {
            re.sub(r"\s+", "", method) for method in self.accepted_methods if method.strip()
        }
        if len(normalized_methods) != len(self.accepted_methods):
            raise ValueError("accepted_methods must be non-empty and unique")
        allowed_guided_modes: set[HelpSupportMode] = set()
        for help_skill in self.help_skills:
            allowed_guided_modes.update(_HELP_SKILL_SUPPORT_MODES[help_skill])
        if self.help_plan.H2.support_mode not in allowed_guided_modes:
            raise ValueError("H2 support mode does not match declared help skills")
        if self.short_prompt.strip() == "어떤 방법이 맞을까?":
            raise ValueError("short_prompt must name the current mathematical action")
        if re.search(r"그\s*부분.*(?:기억|확인)|네가\s*말한\s*데까지", self.short_prompt):
            raise ValueError("short_prompt must not sound like a system status report")
        l4_prompt = self.effective_l4_prompt
        if l4_prompt.strip() == self.short_prompt.strip():
            raise ValueError("L4 and L2 prompts must not collapse into the same request")
        if not l4_prompt.rstrip().endswith("?"):
            raise ValueError("l4_prompt must be a complete child-facing question")
        if self.content_version >= 2:
            if self.entry_mode == "wrong_guess":
                if not self.entry_prompt:
                    raise ValueError("wrong_guess entry needs entry_prompt")
                if self.entry_prompt.strip() == l4_prompt.strip():
                    raise ValueError("wrong_guess entry and L4 follow-up must differ")
                if re.search(r"(라고\s*했어|이라고\s*했어|보고)", self.entry_prompt):
                    raise ValueError("wrong_guess entry must sound like a present guess")
            elif self.entry_prompt is not None:
                raise ValueError("only wrong_guess entries may add a pre-evaluation turn")
        if self.entry_prompt and not self.entry_prompt.rstrip().endswith("?"):
            raise ValueError("entry_prompt must be a complete child-facing question")
        if not self.short_prompt.rstrip().endswith("?"):
            raise ValueError("short_prompt must be a complete child-facing question")
        if self.short_correct not in self.short_options:
            raise ValueError("short_correct must be one of short_options")
        if self.fill_correct not in self.fill_options:
            raise ValueError("fill_correct must be one of fill_options")
        if len(set(self.short_options)) != len(self.short_options):
            raise ValueError("short_options must be unique")
        if len(set(self.fill_options)) != len(self.fill_options):
            raise ValueError("fill_options must be unique")
        child_facing_copy = [
            l4_prompt,
            *([self.entry_prompt] if self.entry_prompt else []),
            self.short_prompt,
            self.learned_line,
            self.note_context,
            self.fill_before,
            self.fill_after,
            self.help_plan.H1.body,
            self.help_plan.H2.body,
            self.help_plan.H3.body,
            *self.short_options,
            *self.fill_options,
            *self.valid_explanations,
            prompt,
            *(str(answer) for answer in answers),
        ]
        if any(_VAGUE_OR_UNREVIEWED_COPY.search(text) for text in child_facing_copy):
            raise ValueError("child-facing copy contains a vague or unreviewed phrase")
        validate_child_facing_math_copy(child_facing_copy)
        mormi_questions = [
            l4_prompt,
            *([self.entry_prompt] if self.entry_prompt else []),
            self.short_prompt,
            prompt,
        ]
        if any(_TEACHER_EVALUATION_COPY.search(text) for text in mormi_questions):
            raise ValueError(
                "Mormi copy must expose its own confusion instead of evaluating the child"
            )
        if any(len(text) > 45 for text in (*self.short_options, *self.fill_options)):
            raise ValueError("choice labels must fit one readable option")
        normalized_answer = re.sub(r"[\s,]", "", str(correct))
        normalized_joint_model = re.sub(r"[\s,]", "", self.help_plan.H3.body)
        if normalized_answer not in normalized_joint_model:
            raise ValueError("H3 joint model must state the current problem's answer")
        return self


def _load_home_teaching_catalog() -> dict[str, HomeTeachingSpec]:
    path = Path(__file__).with_name("home_teaching_catalog.json")
    entries = [HomeTeachingSpec.model_validate(item) for item in json.loads(path.read_text())]
    if any(entry.entry_mode == "wrong_guess" for entry in entries):
        raise ValueError(
            "current home teaching catalog must start with genuine help requests; "
            "wrong_guess is legacy snapshot compatibility only"
        )
    catalog = {entry.id: entry for entry in entries}
    if len(catalog) != len(entries):
        raise ValueError("home teaching catalog contains duplicate ids")
    return catalog


HOME_TEACHING_CATALOG = _load_home_teaching_catalog()


def option(identifier: str, label: str, image_url: str | None = None) -> ChoiceOption:
    return ChoiceOption(id=identifier, label=label, image_url=image_url)


def text_input(*slots: str, placeholder: str = "모르미에게 알려줘") -> InputContract:
    return InputContract(kind=InputKind.TEXT, placeholder=placeholder, target_slots=list(slots))


def choice_input(slots: list[str], choices: list[ChoiceOption]) -> InputContract:
    return InputContract(kind=InputKind.CHOICES, target_slots=slots, choices=choices)


def menu_items_json(items: Sequence[CafeMenuItem]) -> list[dict[str, str | int | None]]:
    return [item.model_dump() for item in items]


QUEUE_TASK = TaskDefinition(
    id="cafe_queue_3_vs_5",
    dictionary_card_id="dictionary.cafe.cafe-queue",
    scene=SceneType.CAFE,
    stage_id="queue",
    skill_id="compare_quantity_in_context",
    help_skills=["counting", "comparison", "queue"],
    help_method_policy="open_methods",
    accepted_methods=[
        "각 줄의 사람 수를 세고 두 수를 비교하기",
        "이미 확인한 두 줄의 사람 수를 직접 비교하기",
    ],
    title="줄 서기",
    goal="두 줄을 세고 사람이 적은 줄을 고른다.",
    visible_facts={"left_count": 3, "right_count": 5, "same_cashier_speed": True},
    slots={
        "left_count": SlotDefinition(
            id="left_count",
            description="왼쪽 줄 사람 수",
            semantic_role="observation",
            expected=3,
            aliases=["3명", "세명", "세 명"],
            fact_sentence="왼쪽 줄에는 3명이 있어.",
        ),
        "right_count": SlotDefinition(
            id="right_count",
            description="오른쪽 줄 사람 수",
            semantic_role="observation",
            expected=5,
            aliases=["5명", "다섯명", "다섯 명"],
            fact_sentence="오른쪽 줄에는 5명이 있어.",
        ),
        "smaller_number": SlotDefinition(
            id="smaller_number",
            description="3과 5 중 작은 수",
            semantic_role="conclusion",
            expected=3,
            aliases=["3", "삼", "셋"],
            fact_sentence="3은 5보다 작아.",
        ),
        "final_choice": SlotDefinition(
            id="final_choice",
            description="내 앞에 기다리는 사람이 적어 차례가 빨리 오는 줄",
            semantic_role="selection",
            expected="left",
            aliases=["왼쪽", "왼쪽줄", "왼쪽 줄"],
            fact_sentence="왼쪽 줄에서는 내 차례가 더 빨리 와.",
        ),
        "reason": SlotDefinition(
            id="reason",
            description="앞에 기다리는 사람이 적으면 내 차례가 빨리 오는 이유",
            semantic_role="reason",
            expected="fewer_people",
            aliases=[
                "앞에사람이적어서",
                "앞에 사람이 적어서",
                "내앞에3명이기다려서",
                "내 앞에 3명이 기다려서",
            ],
            fact_sentence="앞에 기다리는 사람이 적으면 내 차례가 더 빨리 와.",
        ),
    },
    required_slots=["left_count", "right_count", "final_choice", "reason"],
    steps={
        ExpressionLevel.L4: [
            StepDefinition(
                id="free_counts",
                prompt="왼쪽과 오른쪽 줄에 각각 몇 명이 있어?",
                target_slots=["left_count", "right_count"],
                input=text_input("left_count", "right_count"),
                fallback_text="왼쪽과 오른쪽 줄에 각각 몇 명이 있어?",
            ),
            StepDefinition(
                id="free_comparison",
                prompt="어느 줄에 서야 빨리 갈지 모르겠어... 줄이랑 이유도 알려줄 수 있어?",
                target_slots=["final_choice", "reason"],
                input=text_input("final_choice", "reason"),
                fallback_text="어느 줄에 서야 빨리 갈지 모르겠어... 줄이랑 이유도 알려줄 수 있어?",
            ),
        ],
        ExpressionLevel.L3: [
            StepDefinition(
                id="short_counts",
                prompt="왼쪽과 오른쪽 줄에 각각 몇 명이 있어?",
                target_slots=["left_count", "right_count"],
                input=text_input("left_count", "right_count", placeholder="사람 수만 짧게 알려줘"),
                fallback_text="내가 한꺼번에 물어봤네. 사람 수만 알려줘.",
            ),
            StepDefinition(
                id="short_choice",
                prompt="어느 줄에서 내 차례가 더 빨리 올까?",
                target_slots=["final_choice"],
                input=text_input("final_choice", placeholder="왼쪽 또는 오른쪽"),
                fallback_text="내 차례가 빨리 올 줄만 알려줘.",
            ),
            StepDefinition(
                id="short_reason",
                prompt="나는 왜 그 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
                target_slots=["reason"],
                input=text_input("reason", placeholder="이유만 짧게 알려줘"),
                fallback_text="나는 왜 내 차례가 더 빨리 오는지 헷갈려... 알려줄 수 있어?",
            ),
        ],
        ExpressionLevel.L2: [
            StepDefinition(
                id="choose_left_count",
                prompt="왼쪽 줄에는 몇 명이 있어?",
                target_slots=["left_count"],
                input=choice_input(
                    ["left_count"], [option("2", "2명"), option("3", "3명"), option("4", "4명")]
                ),
                choice_effects={
                    "2": {"left_count": 2},
                    "3": {"left_count": 3},
                    "4": {"left_count": 4},
                },
                fallback_text="말로 말하기 어려우면 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_right_count",
                prompt="오른쪽 줄에는 몇 명이 있어?",
                target_slots=["right_count"],
                input=choice_input(
                    ["right_count"], [option("3", "3명"), option("4", "4명"), option("5", "5명")]
                ),
                choice_effects={
                    "3": {"right_count": 3},
                    "4": {"right_count": 4},
                    "5": {"right_count": 5},
                },
                fallback_text="오른쪽 줄 사람 수도 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_side",
                prompt="내 차례가 더 빨리 올 줄은 어느 쪽이야?",
                target_slots=["final_choice"],
                input=choice_input(
                    ["final_choice"], [option("left", "왼쪽 줄"), option("right", "오른쪽 줄")]
                ),
                choice_effects={
                    "left": {"final_choice": "left"},
                    "right": {"final_choice": "right"},
                },
                fallback_text="내 차례가 빨리 올 줄을 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_reason",
                prompt="나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 같이 골라 볼까?",
                target_slots=["reason"],
                input=choice_input(
                    ["reason"],
                    [
                        option("fewer", "내 앞에 3명이 기다려서"),
                        option("more", "내 앞에 5명이 기다려서"),
                    ],
                ),
                choice_effects={
                    "fewer": {"reason": "fewer_people"},
                    "more": {"reason": "more_people"},
                },
                fallback_text="내 차례가 빨리 오는 이유를 같이 골라 보자.",
            ),
        ],
        ExpressionLevel.L1: [
            StepDefinition(
                id="guided_count",
                prompt="사람을 한 명씩 눌러 두 줄을 같이 세어 볼까?",
                target_slots=["left_count", "right_count"],
                input=InputContract(
                    kind=InputKind.COUNT,
                    target_slots=["left_count", "right_count"],
                    config={
                        "left_person_ids": ["l1", "l2", "l3"],
                        "right_person_ids": ["r1", "r2", "r3", "r4", "r5"],
                    },
                ),
                fallback_text="내가 어디부터 볼지 몰랐네. 같이 세어 보자.",
            ),
            StepDefinition(
                id="guided_compare",
                prompt="3은 5보다 어떻게 돼?",
                target_slots=["smaller_number"],
                input=choice_input(
                    ["smaller_number"], [option("smaller", "작아"), option("larger", "커")]
                ),
                choice_effects={"smaller": {"smaller_number": 3}, "larger": {"smaller_number": 5}},
                fallback_text="3과 5를 놓고 관계부터 같이 보자.",
            ),
            StepDefinition(
                id="guided_map",
                prompt="3명이 있는 줄은 어느 쪽이야?",
                target_slots=["final_choice"],
                input=choice_input(
                    ["final_choice"], [option("left", "왼쪽"), option("right", "오른쪽")]
                ),
                choice_effects={
                    "left": {"final_choice": "left"},
                    "right": {"final_choice": "right"},
                },
                fallback_text="3명이 있는 줄을 장면에서 같이 찾아보자.",
            ),
            StepDefinition(
                id="guided_reason",
                prompt="나는 왜 왼쪽 줄이 더 빠른지 헷갈려... 같이 골라 볼까?",
                target_slots=["reason"],
                input=choice_input(
                    ["reason"],
                    [
                        option("fewer", "내 앞에 3명이 기다려서"),
                        option("more", "내 앞에 5명이 기다려서"),
                    ],
                ),
                choice_effects={
                    "fewer": {"reason": "fewer_people"},
                    "more": {"reason": "more_people"},
                },
                fallback_text="마지막 이유도 같이 이어 보자.",
            ),
        ],
        ExpressionLevel.L0: [
            StepDefinition(
                id="joint_performance",
                prompt="도움 카드 순서대로 나와 같이 해볼까?",
                target_slots=["left_count", "right_count", "final_choice", "reason"],
                input=InputContract(
                    kind=InputKind.JOINT,
                    target_slots=["left_count", "right_count", "final_choice", "reason"],
                    config={
                        "steps": ["count_left", "count_right", "compare", "choose_queue"],
                        "completion_values": {
                            "left_count": 3,
                            "right_count": 5,
                            "final_choice": "left",
                            "reason": "fewer_people",
                        },
                    },
                ),
                fallback_text="도움 카드 순서대로 나와 같이 해볼까?",
            )
        ],
    },
    hints={
        HintLevel.H1: reviewed_help_card(
            HintLevel.H1,
            body="두 줄에 있는 사람 수를 각각 확인해 보자.",
            support_mode="attention",
            fact_refs=["left_count", "right_count"],
            visual_type=None,
        ),
        HintLevel.H2: reviewed_help_card(
            HintLevel.H2,
            body="왼쪽 3명과 오른쪽 5명 중 작은 수를 찾아보자.",
            support_mode="guided_choice",
            fact_refs=["left_count", "right_count"],
            action="3과 5 중 작은 수 고르기",
            visual_type="number_cards",
            visual_data={"cards": [3, 5], "neutral_style": True},
        ),
        HintLevel.H3: reviewed_help_card(
            HintLevel.H3,
            body="왼쪽 3명, 오른쪽 5명이라 왼쪽 줄에서 덜 기다려.",
            support_mode="joint_model",
            fact_refs=["left_count", "right_count", "final_choice", "reason"],
            action="두 줄을 함께 세고 사람이 적은 줄 고르기",
            visual_type="joint_steps",
            visual_data={"steps": ["한 명씩 세기", "3과 5 비교하기", "사람이 적은 줄 찾기"]},
        ),
    },
    base_visual=VisualContract(
        type="cafe_queues",
        data={"left_people": 3, "right_people": 5, "show_counts": False},
    ),
    misconception_tags=[
        "double_counting",
        "more_people_is_faster",
        "larger_is_smaller",
        "relation_mapping_error",
    ],
    coauthored_note=(
        "각 줄의 사람을 세고, 앞에 기다리는 사람이 적은 줄에 서면 내 차례가 더 빨리 와."
    ),
    note_context="왼쪽 줄과 오른쪽 줄의 사람 수를 비교하는 방법",
)


def calculation_task(
    *,
    task_id: str,
    title: str,
    skill_id: str,
    left: int,
    right: int,
    operation: Literal["addition", "subtraction"],
    result: int,
    scene: SceneType = SceneType.CAFE,
    stage_id: str | None = None,
    left_label: str | None = None,
    right_label: str | None = None,
    result_label: str | None = None,
) -> TaskDefinition:
    symbol = "+" if operation == "addition" else "-"
    method = "carry" if operation == "addition" else "regroup"
    method_label = "올림" if operation == "addition" else "받아내림"
    operation_phrase = "더해" if operation == "addition" else "빼서"
    operation_label = "더하기" if operation == "addition" else "빼기"
    place_action = "더해" if operation == "addition" else "빼"
    resolved_left_label = left_label or ("첫 번째 금액" if operation == "addition" else "낸 돈")
    resolved_right_label = right_label or (
        "두 번째 금액" if operation == "addition" else "사용한 금액"
    )
    resolved_result_label = result_label or (
        "전체 금액" if operation == "addition" else "거스름돈"
    )
    return TaskDefinition(
        id=task_id,
        dictionary_card_id=(
            "dictionary.cafe.cafe-menu-total"
            if operation == "addition"
            else "dictionary.cafe.cafe-change"
        ),
        scene=scene,
        stage_id=stage_id or ("home_teach" if scene is SceneType.HOME_TEACH else "calculation"),
        skill_id=skill_id,
        help_skills=[
            "place_value",
            "addition" if operation == "addition" else "subtraction",
        ],
        help_method_policy="target_method",
        accepted_methods=[f"같은 자리끼리 계산하고 {method_label}하기"],
        title=title,
        goal=f"{left:,}{symbol}{right:,}을 계산하고 {method_label} 방법을 설명한다.",
        visible_facts={"left": left, "right": right, "operation": operation},
        arithmetic_contract=ArithmeticValidationContract(
            operation=operation,
            left=left,
            right=right,
            result=result,
            left_label=resolved_left_label,
            right_label=resolved_right_label,
            result_label=resolved_result_label,
            unit="원",
        ),
        slots={
            "operation": SlotDefinition(
                id="operation",
                description="필요한 계산 종류",
                semantic_role="operation",
                expected=operation,
                aliases=["더하기" if operation == "addition" else "빼기"],
                fact_sentence=(f"{left:,}원과 {right:,}원은 {operation_phrase} 계산해."),
            ),
            "result": SlotDefinition(
                id="result",
                description="계산 결과",
                semantic_role="conclusion",
                expected=result,
                aliases=[str(result), f"{result:,}", f"{result:,}원"],
                fact_sentence=f"계산 결과는 {result:,}원이야.",
            ),
            "method": SlotDefinition(
                id="method",
                description=f"{method_label}이 필요한 자리 계산 방법",
                semantic_role="method",
                expected=method,
                aliases=[method_label, f"{method_label}해"],
                fact_sentence=f"자리값을 맞추고 {method_label}해서 계산해.",
            ),
        },
        required_slots=["operation", "result", "method"],
        steps={
            ExpressionLevel.L4: [
                StepDefinition(
                    id="free_explanation",
                    prompt="나 모두 얼마인지랑 어떻게 계산하는지 헷갈려... 알려줄 수 있어?",
                    target_slots=["operation", "result", "method"],
                    input=text_input("operation", "result", "method"),
                    fallback_text="결과와 계산 방법을 네 말로 알려줘.",
                )
            ],
            ExpressionLevel.L3: [
                StepDefinition(
                    id="short_result",
                    prompt="계산한 값은 얼마야?",
                    target_slots=["result"],
                    input=text_input("result", placeholder="금액만 알려줘"),
                    fallback_text="내가 많이 물어봤네. 금액부터 알려줘.",
                ),
                StepDefinition(
                    id="short_operation",
                    prompt="두 금액을 더해야 해, 빼야 해?",
                    target_slots=["operation"],
                    input=text_input("operation", placeholder="더하기 또는 빼기"),
                    fallback_text="어떤 계산인지부터 짧게 알려줘.",
                ),
                StepDefinition(
                    id="short_method",
                    prompt=(
                        f"나 자리 계산에서 {method_label}을 어떻게 하는지 "
                        "헷갈려... 알려줄 수 있어?"
                    ),
                    target_slots=["method"],
                    input=text_input("method", placeholder="방법만 짧게 알려줘"),
                    fallback_text=f"{method_label} 방법만 짧게 알려줘.",
                ),
            ],
            ExpressionLevel.L2: [
                StepDefinition(
                    id="choose_operation",
                    prompt="어떤 계산을 해야 할까?",
                    target_slots=["operation"],
                    input=choice_input(
                        ["operation"], [option("add", "더하기"), option("subtract", "빼기")]
                    ),
                    choice_effects={
                        "add": {"operation": "addition"},
                        "subtract": {"operation": "subtraction"},
                    },
                    fallback_text="말 대신 필요한 계산을 같이 골라 보자.",
                ),
                StepDefinition(
                    id="choose_result",
                    prompt="계산한 값은 어느 쪽이야?",
                    target_slots=["result"],
                    input=choice_input(
                        ["result"],
                        [
                            option(str(result - 1000), f"{result - 1000:,}원"),
                            option(str(result), f"{result:,}원"),
                            option(str(result + 1000), f"{result + 1000:,}원"),
                        ],
                    ),
                    choice_effects={
                        str(result - 1000): {"result": result - 1000},
                        str(result): {"result": result},
                        str(result + 1000): {"result": result + 1000},
                    },
                    fallback_text="계산한 금액도 같이 골라 보자.",
                ),
                StepDefinition(
                    id="choose_method",
                    prompt="자리 계산에서 무엇을 해야 할까?",
                    target_slots=["method"],
                    input=choice_input(
                        ["method"],
                        [option(method, method_label), option("ignore", "그대로 계산하기")],
                    ),
                    choice_effects={method: {"method": method}, "ignore": {"method": "ignore"}},
                    fallback_text="자리 계산 방법도 같이 골라 보자.",
                ),
            ],
            ExpressionLevel.L1: [
                StepDefinition(
                    id="guided_equation",
                    prompt="세로식 빈칸을 한 자리씩 같이 채워볼까?",
                    target_slots=["operation", "result", "method"],
                    input=InputContract(
                        kind=InputKind.EQUATION,
                        target_slots=["operation", "result", "method"],
                        config={
                            "left": left,
                            "right": right,
                            "operation": operation,
                            "places": ["만", "천", "백", "십", "일"],
                        },
                    ),
                    fallback_text="내가 어디부터 볼지 몰랐네. 한 자리씩 보자.",
                )
            ],
            ExpressionLevel.L0: [
                StepDefinition(
                    id="joint_equation",
                    prompt="도움 카드 순서대로 세로식을 같이 채울까?",
                    target_slots=["operation", "result", "method"],
                    input=InputContract(
                        kind=InputKind.JOINT,
                        target_slots=["operation", "result", "method"],
                        config={
                            "left": left,
                            "right": right,
                            "operation": operation,
                            "result": result,
                            "completion_values": {
                                "operation": operation,
                                "result": result,
                                "method": method,
                            },
                        },
                    ),
                    fallback_text="도움 카드 순서대로 세로식을 같이 채울까?",
                )
            ],
        },
        hints={
            HintLevel.H1: reviewed_help_card(
                HintLevel.H1,
                body="두 수에서 같은 자리의 숫자를 확인해 보자.",
                support_mode="attention",
                fact_refs=["left", "right", "operation"],
            ),
            HintLevel.H2: reviewed_help_card(
                HintLevel.H2,
                body=f"같은 자리끼리 {place_action}서 빈칸을 채워 보자.",
                support_mode="guided_equation",
                fact_refs=["left", "right", "operation"],
                action=f"같은 자리끼리 {place_action}서 식의 빈칸 채우기",
                visual_type="place_value_equation",
                visual_data={"left": left, "right": right, "operation": operation},
            ),
            HintLevel.H3: reviewed_help_card(
                HintLevel.H3,
                body=f"{left:,}{symbol}{right:,}={result:,}이야. 같은 자리끼리 계산해.",
                support_mode="joint_model",
                fact_refs=["left", "right", "operation", "result", "method"],
                action="완성된 세로식을 함께 확인하기",
                visual_type="joint_equation_steps",
                visual_data={
                    "left": left,
                    "right": right,
                    "operation": operation,
                    "result": result,
                },
            ),
        },
        base_visual=VisualContract(
            type="vertical_equation",
            data={"left": left, "right": right, "operation": operation, "result_hidden": True},
        ),
        misconception_tags=[f"{method}_omission", "place_value_error", "operation_confusion"],
        coauthored_note=f"자리값을 맞추고 {method_label}해서 계산하면 {result:,}원이야.",
        note_context=f"{left:,}과 {right:,}을 {operation_label}로 계산하는 방법",
    )


KOREAN_COUNTS = {
    1: ["1명", "한명", "한 명"],
    2: ["2명", "두명", "두 명"],
    3: ["3명", "세명", "세 명"],
    4: ["4명", "네명", "네 명"],
    5: ["5명", "다섯명", "다섯 명"],
}

# QueueSessionContext와 동적 콘텐츠가 같은 1~5 계약을 공유한다.
# 둘이 어긋나면 화면이 그린 줄을 말로 못 옮기고 KeyError 로 죽는다.
MAX_QUEUE_COUNT = max(KOREAN_COUNTS)
if set(KOREAN_COUNTS) != set(range(QUEUE_MIN_COUNT, QUEUE_MAX_COUNT + 1)):
    raise RuntimeError("KOREAN_COUNTS must cover the complete queue count contract")

# 숫자를 소리 내어 읽었을 때 마지막 음절에 받침이 있는지.
# 일(ㄹ) 삼(ㅁ) 육(ㄱ) 칠(ㄹ) 팔(ㄹ) 은 받침이 있고, 이 사 오 구 는 없다.
# 0으로 끝나면 십·백·천 처럼 받침이 있는 소리로 읽는다.
_DIGIT_HAS_FINAL = {0: True, 1: True, 2: False, 3: True, 4: False,
                    5: False, 6: True, 7: True, 8: True, 9: False}


def has_final_consonant(value: str | int) -> bool:
    """마지막 글자에 받침이 있는지. 숫자는 읽는 소리를 기준으로 판단한다."""
    text = str(value).rstrip()
    if not text:
        return False
    last = text[-1]
    if last.isdigit():
        return _DIGIT_HAS_FINAL[int(last)]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28 != 0
    return False


def particle(value: str | int, after_final: str, after_vowel: str) -> str:
    """받침에 맞는 조사를 고른다. 예: particle(2, "과", "와") -> "와".

    아이가 읽는 문장이라 '아메리카노을' 같은 어색한 조사가 그대로 노출되면 안 된다.
    메뉴 이름과 숫자는 방문마다 달라지므로 조사를 문장에 박아 둘 수 없다.
    """
    return after_final if has_final_consonant(value) else after_vowel


def _nearby_count_options(value: int) -> list[ChoiceOption]:
    values = sorted({max(1, value - 1), value, min(MAX_QUEUE_COUNT, value + 1)})
    return [option(str(item), f"{item}명") for item in values]


def queue_task(
    *,
    task_id: str,
    stage_id: str,
    left: int,
    right: int,
    note_policy: str = "stage",
) -> TaskDefinition:
    if not (
        QUEUE_MIN_COUNT <= left <= QUEUE_MAX_COUNT
        and QUEUE_MIN_COUNT <= right <= QUEUE_MAX_COUNT
    ):
        raise ValueError(
            f"queue counts must be between {QUEUE_MIN_COUNT} and {QUEUE_MAX_COUNT}"
        )
    if left == right:
        raise ValueError("queue counts must differ")
    task = QUEUE_TASK.model_copy(deep=True)
    smaller = min(left, right)
    larger = max(left, right)
    side = "left" if left < right else "right"
    side_label = "왼쪽" if side == "left" else "오른쪽"
    task.id = task_id
    task.stage_id = stage_id
    task.title = "줄 서기"
    task.visible_facts = {
        "left_count": left,
        "right_count": right,
        "same_cashier_speed": True,
    }
    task.slots["left_count"] = SlotDefinition(
        id="left_count",
        description="왼쪽 줄 사람 수",
        semantic_role="observation",
        expected=left,
        aliases=KOREAN_COUNTS[left],
        fact_sentence=f"왼쪽 줄에는 {left}명이 있어.",
    )
    task.slots["right_count"] = SlotDefinition(
        id="right_count",
        description="오른쪽 줄 사람 수",
        semantic_role="observation",
        expected=right,
        aliases=KOREAN_COUNTS[right],
        fact_sentence=f"오른쪽 줄에는 {right}명이 있어.",
    )
    task.slots["smaller_number"] = SlotDefinition(
        id="smaller_number",
        description=f"{left}{particle(left, '과', '와')} {right} 중 작은 수",
        semantic_role="conclusion",
        expected=smaller,
        aliases=[str(smaller)],
        fact_sentence=f"{smaller}{particle(smaller, '이', '가')} 더 작은 수야.",
    )
    task.slots["final_choice"] = SlotDefinition(
        id="final_choice",
        description="내 앞에 기다리는 사람이 적어 차례가 빨리 오는 줄",
        semantic_role="selection",
        expected=side,
        aliases=[side_label, f"{side_label}줄", f"{side_label} 줄"],
        fact_sentence=f"{side_label} 줄에서는 내 차례가 더 빨리 와.",
    )
    task.slots["reason"] = SlotDefinition(
        id="reason",
        description="앞에 기다리는 사람이 적으면 내 차례가 빨리 오는 이유",
        semantic_role="reason",
        expected="fewer_people",
        aliases=[
            "앞에사람이적어서",
            "앞에 사람이 적어서",
            f"내앞에{smaller}명이기다려서",
            f"내 앞에 {smaller}명이 기다려서",
        ],
        fact_sentence="앞에 기다리는 사람이 적으면 내 차례가 더 빨리 와.",
    )
    task.steps[ExpressionLevel.L2][0].input = choice_input(
        ["left_count"], _nearby_count_options(left)
    )
    task.steps[ExpressionLevel.L2][0].choice_effects = {
        choice.id: {"left_count": int(choice.id)}
        for choice in task.steps[ExpressionLevel.L2][0].input.choices
    }
    task.steps[ExpressionLevel.L2][1].input = choice_input(
        ["right_count"], _nearby_count_options(right)
    )
    task.steps[ExpressionLevel.L2][1].choice_effects = {
        choice.id: {"right_count": int(choice.id)}
        for choice in task.steps[ExpressionLevel.L2][1].input.choices
    }
    task.steps[ExpressionLevel.L1][0].input.config = {
        "left_person_ids": [f"l{index}" for index in range(1, left + 1)],
        "right_person_ids": [f"r{index}" for index in range(1, right + 1)],
    }
    task.steps[ExpressionLevel.L1][1].prompt = (
        f"{left}{particle(left, '과', '와')} {right} 중 더 작은 수는 뭐야?"
    )
    task.steps[ExpressionLevel.L1][1].input = choice_input(
        ["smaller_number"],
        [option(str(left), str(left)), option(str(right), str(right))],
    )
    task.steps[ExpressionLevel.L1][1].choice_effects = {
        str(left): {"smaller_number": left},
        str(right): {"smaller_number": right},
    }
    task.steps[ExpressionLevel.L1][2].prompt = f"{smaller}명이 있는 줄은 어느 쪽이야?"
    task.steps[ExpressionLevel.L3][2].prompt = (
        f"나는 왜 {side_label} 줄이 더 빠른지 헷갈려... 알려줄 수 있어?"
    )
    task.steps[ExpressionLevel.L3][2].fallback_text = (
        "나는 왜 내 차례가 더 빨리 오는지 헷갈려... 알려줄 수 있어?"
    )
    for level, step_index in (
        (ExpressionLevel.L2, 3),
        (ExpressionLevel.L1, 3),
    ):
        reason_step = task.steps[level][step_index]
        reason_step.prompt = (
            f"나는 왜 {side_label} 줄이 더 빠른지 헷갈려... 같이 골라 볼까?"
        )
        reason_step.input = choice_input(
            ["reason"],
            [
                option("fewer", f"내 앞에 {smaller}명이 기다려서"),
                option("more", f"내 앞에 {larger}명이 기다려서"),
            ],
        )
        reason_step.choice_effects = {
            "fewer": {"reason": "fewer_people"},
            "more": {"reason": "more_people"},
        }
    task.steps[ExpressionLevel.L0][0].input.config["completion_values"] = {
        "left_count": left,
        "right_count": right,
        "final_choice": side,
        "reason": "fewer_people",
    }
    task.hints[HintLevel.H2] = reviewed_help_card(
        HintLevel.H2,
        body=f"왼쪽 {left}명과 오른쪽 {right}명 중 작은 수를 찾아보자.",
        support_mode="guided_choice",
        fact_refs=["left_count", "right_count"],
        action=f"{left}{particle(left, '과', '와')} {right} 중 작은 수 고르기",
        visual_type="number_cards",
        visual_data={"cards": [left, right], "neutral_style": True},
    )
    task.hints[HintLevel.H3] = reviewed_help_card(
        HintLevel.H3,
        body=f"왼쪽 {left}명, 오른쪽 {right}명이라 {side_label} 줄에서 덜 기다려.",
        support_mode="joint_model",
        fact_refs=["left_count", "right_count", "final_choice", "reason"],
        action="두 줄을 함께 세고 사람이 적은 줄 고르기",
        visual_type="joint_steps",
        visual_data={"steps": ["한 명씩 세기", "두 수 비교하기", "사람이 적은 줄 찾기"]},
    )
    task.base_visual = VisualContract(
        type="cafe_queues",
        data={"left_people": left, "right_people": right, "show_counts": False},
    )
    task.note_context = f"왼쪽 {left}명과 오른쪽 {right}명의 줄을 비교하는 방법"
    task.note_policy = note_policy
    task.transition_text = "사람이 적은 줄을 찾았구나."
    return task


def menu_selection_task(
    *,
    task_id: str,
    stage_id: str,
    menu_items: Sequence[CafeMenuItem],
    mormi_menu: CafeMenuItem,
    budget: int | None,
    auto_total: bool,
    behavior: str,
    note_policy: str,
) -> TaskDefinition:
    valid_ids: list[str | int | float | bool] = [
        item.id
        for item in menu_items
        if item.id != mormi_menu.id
        and (budget is None or not auto_total or mormi_menu.price + item.price <= budget)
    ]
    if not valid_ids:
        raise ValueError("menu task needs at least one selectable reviewed menu")
    suggested_menu = next(item for item in menu_items if item.id == valid_ids[0])
    suggested_total = mormi_menu.price + suggested_menu.price
    choices = [
        ChoiceOption(
            id=item.id,
            label=f"{item.name} {item.price:,}원",
            image_url=item.image_url,
            disabled=item.id == mormi_menu.id,
        )
        for item in menu_items
    ]
    input_contract = InputContract(
        kind=InputKind.CHOICES,
        target_slots=["child_menu"],
        choices=choices,
        config={
            "component": "cafe_menu_picker",
            "budget": budget,
            "mormi_menu_id": mormi_menu.id,
            "auto_total": auto_total,
            "allow_same_menu": False,
        },
    )
    if budget is not None:
        prompt = f"{budget:,}원 안에서 고르자. 나는 {mormi_menu.name}, 너는 뭘 고를래?"
        fallback = f"예산은 {budget:,}원이야. 네 메뉴 하나를 골라줄래?"
    else:
        prompt = (
            f"나는 {mormi_menu.name}{particle(mormi_menu.name, '을', '를')} 골랐어. "
            "너는 뭘 고를래?"
        )
        fallback = "계산할 메뉴를 하나 골라줄래?"
    if len(prompt) > 50:
        prompt = fallback
    step = StepDefinition(
        id="pick_menu",
        prompt=prompt,
        target_slots=["child_menu"],
        input=input_contract,
        choice_effects={item.id: {"child_menu": item.id} for item in menu_items},
        fallback_text=fallback,
    )
    joint_step = StepDefinition(
        id="joint_menu_pick",
        prompt="도움 카드와 같이 예산에 맞는 메뉴를 담아 볼까?",
        target_slots=["child_menu"],
        input=InputContract(
            kind=InputKind.JOINT,
            target_slots=["child_menu"],
            config={
                "component": "cafe_menu_picker",
                "budget": budget,
                "mormi_menu_id": mormi_menu.id,
                "suggested_menu_id": suggested_menu.id,
                "completion_values": {"child_menu": suggested_menu.id},
            },
        ),
        fallback_text="도움 카드와 같이 메뉴를 하나 담아 볼까?",
    )
    return TaskDefinition(
        id=task_id,
        dictionary_card_id=(
            "dictionary.cafe.cafe-budget-menu"
            if budget is not None
            else "dictionary.cafe.cafe-menu-total"
        ),
        scene=SceneType.CAFE,
        stage_id=stage_id,
        skill_id="choose_within_budget" if budget is not None else "choose_menu_for_calculation",
        help_skills=["budget", "selection"] if budget is not None else ["selection"],
        help_method_policy="open_methods",
        accepted_methods=(
            [
                "두 메뉴의 합계를 구해 예산과 비교하기",
                "남은 예산 안에서 고를 수 있는 메뉴 찾기",
            ]
            if budget is not None
            else ["메뉴판에서 원하는 메뉴 하나 고르기"]
        ),
        title="예산 안에서 메뉴 고르기" if budget is not None else "계산할 메뉴 고르기",
        goal=(
            "두 메뉴가 예산 안에 들어오도록 고른다."
            if budget is not None
            else "계산할 메뉴를 하나 고른다."
        ),
        visible_facts={
            "budget": budget,
            "mormi_menu": mormi_menu.model_dump(),
            "menu_items": menu_items_json(menu_items),
            "auto_total": auto_total,
        },
        slots={
            "child_menu": SlotDefinition(
                id="child_menu",
                description="아이가 고른 메뉴",
                semantic_role="selection",
                expected=valid_ids[0],
                accepted_values=valid_ids,
                preserve_value=True,
                fact_sentence="아이도 메뉴를 하나 골랐어.",
            )
        },
        required_slots=["child_menu"],
        steps={
            ExpressionLevel.L4: [step.model_copy(deep=True)],
            ExpressionLevel.L3: [step.model_copy(deep=True)],
            ExpressionLevel.L2: [step.model_copy(deep=True)],
            ExpressionLevel.L1: [step.model_copy(deep=True)],
            ExpressionLevel.L0: [joint_step],
        },
        hints={
            HintLevel.H1: reviewed_help_card(
                HintLevel.H1,
                body=(
                    "예산과 모르미가 고른 메뉴 가격을 확인해 보자."
                    if budget is not None
                    else "모르미가 고른 메뉴를 먼저 확인해 보자."
                ),
                support_mode="attention",
                fact_refs=(
                    ["budget", "mormi_menu"] if budget is not None else ["mormi_menu"]
                ),
            ),
            HintLevel.H2: reviewed_help_card(
                HintLevel.H2,
                body=(
                    "예산에서 모르미 메뉴값을 빼고 남은 돈을 보자."
                    if budget is not None
                    else "메뉴판에서 다른 메뉴 하나를 골라 보자."
                ),
                support_mode="guided_equation" if budget is not None else "guided_choice",
                fact_refs=(
                    ["budget", "mormi_menu"]
                    if budget is not None
                    else ["mormi_menu", "menu_items"]
                ),
                action=(
                    "예산에서 모르미 메뉴값을 빼기"
                    if budget is not None
                    else "메뉴판에서 다른 메뉴 하나 고르기"
                ),
                visual_type="budget_meter" if budget is not None else "cafe_menu_focus",
                visual_data=(
                    {"budget": budget, "mormi_price": mormi_menu.price}
                    if budget is not None
                    else {"mormi_menu": mormi_menu.model_dump()}
                ),
            ),
            HintLevel.H3: reviewed_help_card(
                HintLevel.H3,
                body=(
                    f"두 메뉴는 {suggested_total:,}원이라 {budget:,}원 안에서 살 수 있어."
                    if budget is not None
                    else "서로 다른 메뉴 두 개를 함께 장바구니에 담아 보자."
                ),
                support_mode="joint_model",
                fact_refs=["mormi_menu", "menu_items", "child_menu"],
                action="검수된 두 메뉴를 장바구니에 함께 담기",
                visual_type="budget_menu_help" if budget is not None else "cafe_menu_focus",
                visual_data=(
                    {
                        "budget": budget,
                        "mormi_menu": mormi_menu.model_dump(),
                        "suggested_menu": suggested_menu.model_dump(),
                        "total": suggested_total,
                    }
                    if budget is not None
                    else {
                        "mormi_menu": mormi_menu.model_dump(),
                        "suggested_menu": suggested_menu.model_dump(),
                    }
                ),
            ),
        },
        base_visual=VisualContract(
            type="cafe_menu",
            data={
                "menu_items": menu_items_json(menu_items),
                "budget": budget,
                "mormi_pick": mormi_menu.model_dump(),
                "child_pick": None,
                "auto_total": auto_total,
                "budget_status": "pending",
            },
        ),
        misconception_tags=(
            ["budget_exceeded", "price_comparison_error"] if budget is not None else []
        ),
        coauthored_note="메뉴 가격을 더한 금액이 예산보다 크면 다른 메뉴를 골라야 해.",
        note_context=(
            f"{budget:,}원 안에서 두 메뉴를 고르는 방법"
            if budget is not None
            else "메뉴판에서 메뉴를 고르는 방법"
        ),
        behavior=behavior,
        note_policy=note_policy,
        transition_text="네 메뉴도 골랐구나.",
    )


def simple_calculation_task(
    *,
    task_id: str,
    stage_id: str,
    title: str,
    left: int,
    right: int,
    operation: Literal["addition", "subtraction"],
    left_label: str,
    right_label: str,
    behavior: str,
    note_policy: str,
    coauthored_note: str,
    context: Mapping[str, Any],
) -> TaskDefinition:
    result = left + right if operation == "addition" else left - right
    symbol = "+" if operation == "addition" else "-"
    operation_label = "더하기" if operation == "addition" else "빼기"
    distractors = sorted({max(0, result - 1000), result, result + 1000})
    result_choices = [option(str(value), f"{value:,}원") for value in distractors]
    operation_choices = [option("add", "더하기"), option("subtract", "빼기")]
    operation_effects: dict[str, dict[str, str | int | float | bool]] = {
        "add": {"operation": "addition"},
        "subtract": {"operation": "subtraction"},
    }
    result_effects: dict[str, dict[str, str | int | float | bool]] = {
        str(value): {"result": value} for value in distractors
    }

    l4 = StepDefinition(
        id="free_calculation",
        prompt=(
            "나 두 메뉴가 모두 얼마인지랑 어떻게 계산하는지 헷갈려... 알려줄 수 있어?"
            if operation == "addition"
            else "나 거스름돈이 얼마인지랑 어떻게 계산하는지 헷갈려... 알려줄 수 있어?"
        ),
        target_slots=["operation", "result"],
        input=text_input("operation", "result", placeholder="값과 계산 방법을 알려줘"),
        fallback_text="계산한 값과 어떤 계산인지 알려줘.",
    )
    l3 = [
        StepDefinition(
            id="short_result",
            prompt=(
                "그럼 두 메뉴는 모두 얼마야?"
                if operation == "addition"
                else "그럼 거스름돈은 얼마야?"
            ),
            target_slots=["result"],
            input=text_input("result", placeholder="금액만 알려줘"),
            fallback_text="내가 많이 물어봤네. 금액부터 알려줘.",
        ),
        StepDefinition(
            id="short_operation",
            prompt="그 금액은 어떻게 계산한 건지 알려줄 수 있어?",
            target_slots=["operation"],
            input=text_input("operation", placeholder="더하기 또는 빼기"),
            fallback_text="나는 계산하는 방법이 아직 헷갈려... 알려줄 수 있어?",
        ),
    ]
    l2 = [
        StepDefinition(
            id="choose_operation",
            prompt="어떤 계산을 해야 할까?",
            target_slots=["operation"],
            input=choice_input(["operation"], operation_choices),
            choice_effects=operation_effects,
            fallback_text="필요한 계산을 같이 골라 보자.",
        ),
        StepDefinition(
            id="choose_result",
            prompt="계산한 금액은 어느 쪽이야?",
            target_slots=["result"],
            input=choice_input(["result"], result_choices),
            choice_effects=result_effects,
            fallback_text="계산한 금액도 같이 골라 보자.",
        ),
    ]
    fill_result = InputContract(
        kind=InputKind.FILL,
        target_slots=["result"],
        choices=result_choices,
        config={"expression": f"{left:,} {symbol} {right:,} = □"},
    )
    l1 = [
        StepDefinition(
            id="guided_operation",
            prompt=(
                "두 메뉴 가격은 어떤 계산으로 합칠까?"
                if operation == "addition"
                else "거스름돈을 구하려면 어떤 계산을 할까?"
            ),
            target_slots=["operation"],
            input=choice_input(["operation"], operation_choices),
            choice_effects=operation_effects,
            fallback_text="두 금액 사이 계산 기호부터 골라 보자.",
        ),
        StepDefinition(
            id="guided_result",
            prompt=f"{left:,} {symbol} {right:,}의 빈칸은 얼마야?",
            target_slots=["result"],
            input=fill_result,
            choice_effects=result_effects,
            fallback_text="가로식의 빈칸을 같이 채워 보자.",
        ),
    ]
    joint = StepDefinition(
        id="joint_calculation",
        prompt="도움 카드 순서대로 계산을 같이 해볼까?",
        target_slots=["operation", "result"],
        input=InputContract(
            kind=InputKind.JOINT,
            target_slots=["operation", "result"],
            config={
                "left": left,
                "right": right,
                "operation": operation,
                "result": result,
                "completion_values": {"operation": operation, "result": result},
            },
        ),
        fallback_text="도움 카드 순서대로 계산을 같이 해볼까?",
    )
    return TaskDefinition(
        id=task_id,
        dictionary_card_id=(
            "dictionary.cafe.cafe-menu-total"
            if operation == "addition"
            else "dictionary.cafe.cafe-change"
        ),
        scene=SceneType.CAFE,
        stage_id=stage_id,
        skill_id="add_menu_prices" if operation == "addition" else "calculate_change",
        help_skills=["addition" if operation == "addition" else "subtraction"],
        help_method_policy="open_methods",
        accepted_methods=(
            ["두 메뉴 가격을 더해 합계 구하기"]
            if operation == "addition"
            else ["낸 돈에서 메뉴값을 빼 거스름돈 구하기"]
        ),
        title=title,
        goal=f"{left:,}{symbol}{right:,}을 생활 맥락에서 계산한다.",
        visible_facts={"left": left, "right": right, "operation": operation, **dict(context)},
        arithmetic_contract=ArithmeticValidationContract(
            operation=operation,
            left=left,
            right=right,
            result=result,
            left_label=left_label,
            right_label=right_label,
            result_label="전체 금액" if operation == "addition" else "거스름돈",
            unit="원",
        ),
        slots={
            "operation": SlotDefinition(
                id="operation",
                description="필요한 계산 종류",
                semantic_role="operation",
                expected=operation,
                aliases=[operation_label],
                fact_sentence=f"{operation_label}로 계산해.",
            ),
            "result": SlotDefinition(
                id="result",
                description="계산 결과",
                semantic_role="conclusion",
                expected=result,
                aliases=[str(result), f"{result:,}", f"{result}원", f"{result:,}원"],
                fact_sentence=f"계산 결과는 {result:,}원이야.",
            ),
        },
        required_slots=["operation", "result"],
        steps={
            ExpressionLevel.L4: [l4],
            ExpressionLevel.L3: l3,
            ExpressionLevel.L2: l2,
            ExpressionLevel.L1: l1,
            ExpressionLevel.L0: [joint],
        },
        hints={
            HintLevel.H1: reviewed_help_card(
                HintLevel.H1,
                body=(
                    "두 메뉴 가격이 각각 얼마인지 확인해 보자."
                    if operation == "addition"
                    else "낸 돈과 메뉴값이 각각 얼마인지 확인해 보자."
                ),
                support_mode="attention",
                fact_refs=["left", "right", "operation"],
            ),
            HintLevel.H2: reviewed_help_card(
                HintLevel.H2,
                body=f"{left:,}{symbol}{right:,}=□ 식의 빈칸을 채워 보자.",
                support_mode="guided_equation",
                fact_refs=["left", "right", "operation"],
                action="두 금액을 식에 넣고 빈칸 채우기",
                visual_type="money_calculation",
                visual_data={
                    "left": left,
                    "right": right,
                    "operation": operation,
                    "result_hidden": True,
                },
            ),
            HintLevel.H3: reviewed_help_card(
                HintLevel.H3,
                body=(
                    f"{left:,}원에 {right:,}원을 더하면 모두 {result:,}원이야."
                    if operation == "addition"
                    else f"{left:,}원에서 {right:,}원을 빼면 {result:,}원이 남아."
                ),
                support_mode="joint_model",
                fact_refs=["left", "right", "operation", "result"],
                action="완성된 계산식을 함께 확인하기",
                visual_type="joint_money_calculation",
                visual_data={
                    "left": left,
                    "right": right,
                    "operation": operation,
                    "result": result,
                },
            ),
        },
        base_visual=VisualContract(
            type="cafe_calculation",
            data={
                "left": left,
                "right": right,
                "operation": operation,
                "result_hidden": True,
                **dict(context),
            },
        ),
        misconception_tags=["operation_confusion", "calculation_error"],
        coauthored_note=coauthored_note,
        note_context=f"{left:,}원과 {right:,}원을 {operation_label}로 계산하는 방법",
        behavior=behavior,
        note_policy=note_policy,
        transition_text=f"계산하면 {result:,}원이구나.",
    )


def home_teaching_task(
    spec: HomeTeachingSpec,
    *,
    skill_id: str,
) -> TaskDefinition:
    """Build a deterministic teaching task from reviewed curriculum content.

    The LLM may understand a child's paraphrase and phrase Mormi's reaction,
    but it never invents the target rule, choices, help cards, or note text.
    """

    expected_rule = spec.learned_line

    raw_sample = dict(spec.sample_problem)
    expected_answer = raw_sample.get("correct")
    sample_answers = raw_sample.get("answers")
    if not isinstance(expected_answer, (str, int, float, bool)):
        raise ValueError(f"{spec.id}: sample_problem.correct is required")
    if not isinstance(sample_answers, list) or len(sample_answers) < 2:
        raise ValueError(f"{spec.id}: sample_problem.answers needs at least two choices")

    answer_choices = [
        option(f"answer_{index}", str(label)) for index, label in enumerate(sample_answers)
    ]
    answer_effects: dict[str, dict[str, str | int | float | bool]] = {
        f"answer_{index}": {"answer": expected_answer} if label == expected_answer else {}
        for index, label in enumerate(sample_answers)
    }

    short_choices = [
        option(f"short_{index}", label) for index, label in enumerate(spec.short_options)
    ]
    short_effects: dict[str, dict[str, str | int | float | bool]] = {
        f"short_{index}": {"rule": expected_rule} if label == spec.short_correct else {}
        for index, label in enumerate(spec.short_options)
    }
    fill_choices = [option(f"fill_{index}", label) for index, label in enumerate(spec.fill_options)]
    fill_effects: dict[str, dict[str, str | int | float | bool]] = {
        f"fill_{index}": {"rule": expected_rule} if label == spec.fill_correct else {}
        for index, label in enumerate(spec.fill_options)
    }
    fill_before = spec.fill_before.strip()
    fill_after = spec.fill_after.strip()
    # Korean postpositions attach to the word that fills the blank.  Joining
    # every fragment with spaces produced visibly broken copy such as
    # ``남은 □ 을 더해``.  Keep the blank and its postposition together.
    if re.match(r"^(?:으로|로|을|를|이|가|은|는|의|와|과)(?:\s|$)", fill_after):
        blank_and_after = f"□{fill_after}"
    else:
        blank_and_after = " ".join(part for part in ("□", fill_after) if part)
    sentence_frame = " ".join(part for part in (fill_before, blank_and_after) if part)
    sample = raw_sample
    sample.pop("correct", None)

    arithmetic_contract: ArithmeticValidationContract | None = None
    visual = sample.get("visual")
    if isinstance(visual, dict) and visual.get("type") == "money":
        amounts = visual.get("amounts")
        labels = visual.get("labels")
        item_labels = (
            [str(label).strip() for label in labels if str(label).strip()]
            if isinstance(labels, list)
            else []
        )
        if isinstance(amounts, list) and amounts and all(
            isinstance(amount, int) and not isinstance(amount, bool) for amount in amounts
        ):
            if "subtraction" in spec.help_skills and isinstance(visual.get("paid"), int):
                paid = int(visual["paid"])
                spent = sum(int(amount) for amount in amounts)
                arithmetic_contract = ArithmeticValidationContract(
                    operation="subtraction",
                    left=paid,
                    right=spent,
                    result=paid - spent,
                    left_label="낸 돈",
                    right_label=(
                        f"{item_labels[0]}값" if len(item_labels) == 1 else "물건값"
                    ),
                    result_label="남는 돈",
                    unit="원",
                )
            elif "addition" in spec.help_skills and len(amounts) >= 2:
                left = int(amounts[0])
                right = sum(int(amount) for amount in amounts[1:])
                arithmetic_contract = ArithmeticValidationContract(
                    operation="addition",
                    left=left,
                    right=right,
                    result=left + right,
                    left_label=(
                        f"{item_labels[0]} 가격" if item_labels else "첫 번째 물건값"
                    ),
                    right_label=(
                        f"{item_labels[1]} 가격"
                        if len(item_labels) == 2
                        else "나머지 물건값"
                    ),
                    result_label="전체 금액",
                    unit="원",
                )

    task = TaskDefinition(
        id=HOME_TEACH_TASK_ID,
        dictionary_card_id=spec.dictionary_card_id,
        scene=SceneType.HOME_TEACH,
        stage_id="home_teach",
        skill_id=skill_id,
        help_skills=spec.help_skills,
        help_method_policy=spec.help_method_policy,
        accepted_methods=spec.accepted_methods,
        title=spec.title,
        goal=f"반복한 {spec.title}의 핵심 방법을 모르미에게 가르친다.",
        visible_facts={
            "curriculum_session_id": spec.id,
            "target_rule": expected_rule,
            "sample_answer": expected_answer,
            "sample_problem": sample,
        },
        arithmetic_contract=arithmetic_contract,
        slots={
            # The concrete answer is useful partial evidence: when a child only
            # corrects Mormi's answer, preserve it and ask only for the method.
            # It is intentionally not required for completion because the
            # teaching goal and star-note evidence are the general rule.
            "answer": SlotDefinition(
                id="answer",
                description=f"화면의 {spec.title} 예시 문제 답",
                semantic_role="conclusion",
                expected=expected_answer,
                aliases=[str(expected_answer).replace(",", "")],
                fact_sentence=f"이 문제의 답은 {expected_answer}이야.",
            ),
            "rule": SlotDefinition(
                id="rule",
                description=f"{spec.title}를 해결하는 사실이 맞는 설명 또는 검수된 방법",
                semantic_role="explanation",
                expected=expected_rule,
                aliases=list(
                    dict.fromkeys(
                        [
                            expected_rule.rstrip(".!?"),
                            spec.short_correct,
                            *spec.valid_explanations,
                        ]
                    )
                ),
                fact_sentence=expected_rule,
            ),
        },
        required_slots=["rule"],
        steps={
            ExpressionLevel.L4: [
                StepDefinition(
                    id="free_explanation",
                    prompt=spec.effective_l4_prompt,
                    target_slots=["rule"],
                    optional_slots=["answer"],
                    input=text_input(
                        "answer",
                        "rule",
                        placeholder="답과 방법을 네 말로 알려줘",
                    ),
                    fallback_text=spec.effective_l4_prompt,
                )
            ],
            ExpressionLevel.L3: [
                StepDefinition(
                    id="short_answer",
                    prompt=str(sample["prompt"]),
                    target_slots=["answer"],
                    input=text_input("answer", placeholder="답만 짧게 알려줘"),
                    fallback_text="내가 한꺼번에 물어봤네. 답부터 알려줘.",
                ),
                StepDefinition(
                    id="short_explanation",
                    prompt=spec.short_prompt,
                    target_slots=["rule"],
                    input=text_input("rule", placeholder="방법만 짧게 알려줘"),
                    fallback_text="내가 길게 물어봤네. 방법만 짧게 알려줘.",
                ),
            ],
            ExpressionLevel.L2: [
                StepDefinition(
                    id="choose_answer",
                    prompt=str(sample["prompt"]),
                    target_slots=["answer"],
                    input=choice_input(["answer"], answer_choices),
                    choice_effects=answer_effects,
                    fallback_text="말로 어렵다면 답부터 같이 골라 보자.",
                ),
                StepDefinition(
                    id="choose_method",
                    prompt=spec.short_prompt,
                    target_slots=["rule"],
                    input=choice_input(["rule"], short_choices),
                    choice_effects=short_effects,
                    fallback_text="말로 어렵다면 필요한 방법을 같이 골라 보자.",
                ),
            ],
            ExpressionLevel.L1: [
                StepDefinition(
                    id="guided_answer",
                    prompt=str(sample["prompt"]),
                    target_slots=["answer"],
                    input=choice_input(["answer"], answer_choices),
                    choice_effects=answer_effects,
                    fallback_text="화면을 보며 답부터 하나 골라 보자.",
                ),
                StepDefinition(
                    id="complete_rule",
                    prompt=sentence_frame,
                    target_slots=["rule"],
                    input=InputContract(
                        kind=InputKind.FILL,
                        target_slots=["rule"],
                        choices=fill_choices,
                        config={"sentence": sentence_frame},
                    ),
                    choice_effects=fill_effects,
                    fallback_text="도움 카드 문장의 빈칸을 같이 채워 보자.",
                ),
            ],
            ExpressionLevel.L0: [
                StepDefinition(
                    id="joint_reading",
                    prompt="도움 카드 문장을 나와 같이 읽어볼까?",
                    target_slots=["rule"],
                    input=InputContract(
                        kind=InputKind.JOINT,
                        target_slots=["rule"],
                        config={
                            "text": spec.help_plan.H3.body,
                            "completion_values": {"rule": expected_rule},
                        },
                    ),
                    fallback_text="도움 카드 문장을 나와 같이 읽어볼까?",
                )
            ],
        },
        hints={
            HintLevel.H1: reviewed_help_card(
                HintLevel.H1,
                body=spec.help_plan.H1.body,
                support_mode=spec.help_plan.H1.support_mode,
                fact_refs=spec.help_plan.H1.fact_refs,
                action=spec.help_plan.H1.action,
            ),
            HintLevel.H2: reviewed_help_card(
                HintLevel.H2,
                body=spec.help_plan.H2.body,
                support_mode=spec.help_plan.H2.support_mode,
                fact_refs=spec.help_plan.H2.fact_refs,
                action=spec.help_plan.H2.action,
                visual_type="home_practice_problem",
                visual_data=sample,
            ),
            HintLevel.H3: reviewed_help_card(
                HintLevel.H3,
                body=spec.help_plan.H3.body,
                support_mode=spec.help_plan.H3.support_mode,
                fact_refs=spec.help_plan.H3.fact_refs,
                action=spec.help_plan.H3.action,
                visual_type="joint_reading_card",
                visual_data={"text": spec.help_plan.H3.body},
            ),
        },
        base_visual=VisualContract(
            type="home_teaching",
            data={
                "curriculum_session_id": spec.id,
                "subject": spec.subject,
                "unit": spec.unit,
                "title": spec.title,
                "problem": sample,
            },
        ),
        misconception_tags=[spec.misconception],
        coauthored_note=expected_rule,
        note_context=spec.note_context,
        note_slots=["rule"],
        text_explanation_slots=["rule"],
    )
    if spec.id == "number-count":
        _configure_number_count_task(
            task,
            expected_answer=expected_answer,
            answer_choices=answer_choices,
            answer_effects=answer_effects,
            l4_prompt=spec.effective_l4_prompt,
            short_prompt=spec.short_prompt,
            short_options=spec.short_options,
            short_correct=spec.short_correct,
        )
    elif spec.id == "number-compare":
        _configure_number_compare_task(
            task,
            expected_answer=expected_answer,
            answer_choices=answer_choices,
            answer_effects=answer_effects,
            spec=spec,
        )
    _configure_home_entry(task, spec)
    return task


def _configure_home_entry(task: TaskDefinition, spec: HomeTeachingSpec) -> None:
    """Rebuild a legacy v2 wrong-guess step for persisted snapshots only.

    Current catalog content uses genuine questions.  Keeping this parser lets
    an already-open v2 conversation finish without invalidating its stored
    scenario, while ConversationService guarantees that v3 sessions never
    activate this step.
    """

    task.entry_mode = spec.entry_mode
    if spec.content_version < 2 or spec.entry_mode != "wrong_guess":
        task.entry_step = None
        return
    if not spec.entry_prompt:  # guarded by HomeTeachingSpec validation
        raise ValueError(f"{spec.id}: wrong_guess entry_prompt is required")
    target_slots = list(task.required_slots)
    optional_slots = [slot_id for slot_id in task.slots if slot_id not in target_slots]
    # Preserve the reviewed L4 field order expected by the UI (for example,
    # answer before method) while stance itself remains outside all slots.
    input_slots = list(task.steps[ExpressionLevel.L4][0].input.target_slots)
    task.entry_step = StepDefinition(
        id="entry_check",
        prompt=spec.entry_prompt,
        target_slots=target_slots,
        optional_slots=optional_slots,
        input=text_input(
            *input_slots,
            placeholder="모르미에게 네 생각을 알려줘",
        ),
        fallback_text=spec.entry_prompt,
    )


def _configure_number_count_task(
    task: TaskDefinition,
    *,
    expected_answer: str | int | float | bool,
    answer_choices: list[ChoiceOption],
    answer_effects: dict[str, dict[str, str | int | float | bool]],
    l4_prompt: str,
    short_prompt: str,
    short_options: list[str],
    short_correct: str,
) -> None:
    """Accept multiple sound counting strategies without forcing one script.

    ``tracking`` means a child described a usable way to count each visible
    object once.  Pointing is one option, not the only correct answer.  Everyday
    explanations such as "하나, 둘, 셋 하면서 세어" therefore satisfy the
    same method goal and keep their original wording as direct note evidence.
    """

    task.slots = {
        "answer": SlotDefinition(
            id="answer",
            description="화면에 보이는 점의 전체 개수",
            semantic_role="conclusion",
            expected=expected_answer,
            aliases=["3개", "세개", "세 개", "셋", "하나둘셋", "하나,둘,셋"],
            fact_sentence="점은 모두 3개야.",
        ),
        "tracking": SlotDefinition(
            id="tracking",
            description=(
                "화면에서 셀 대상을 구분하고 각 대상을 한 번씩 세는 타당한 행동. "
                "수 이름 말하기, 순서대로 세기, 가리키기, 손가락 펴기 등 서로 다른 "
                "올바른 전략을 인정한다. 단순히 개수 결과만 반복한 것은 방법이 아니며, "
                "특정 문구나 하나의 전략을 강요하지 않는다."
            ),
            semantic_role="method",
            expected="count_each_once",
            accepted_values=["point_each_dot", "one_by_one_order"],
            aliases=[
                "하나씩 세기",
                "한 개씩 세기",
                "순서대로 세기",
                "차례대로 세기",
                "하나둘셋",
                "하나, 둘, 셋",
                "하나 둘 셋 하고 세기",
                "하나 둘 셋 하면서 세기",
                "하나 둘 셋 하고 세면 돼",
                "하나 둘 셋 하면서 세면 돼",
                "수를 하나씩 말하면서 세기",
                "하나씩 가리키기",
                "한 개씩 가리키기",
                "점마다 가리키기",
                "손가락으로 가리키기",
                "하나씩 누르기",
                "점마다 누르기",
                "하나씩 짚기",
                "손가락을 하나씩 펴며 세기",
                "손가락 하나씩 펴기",
            ],
            fact_sentence="점을 하나씩 세는 방법을 알려줬어.",
        ),
    }
    task.required_slots = ["answer", "tracking"]
    task.note_slots = ["tracking"]
    task.text_explanation_slots = ["tracking"]
    task.steps = {
        ExpressionLevel.L4: [
            StepDefinition(
                id="free_count_and_method",
                prompt=l4_prompt,
                target_slots=["answer", "tracking"],
                input=text_input(
                    "answer",
                    "tracking",
                    placeholder="네가 센 수나 방법을 알려줘",
                ),
                fallback_text=l4_prompt,
            )
        ],
        ExpressionLevel.L3: [
            StepDefinition(
                id="short_count",
                prompt="지금 점이 몇 개야?",
                target_slots=["answer"],
                input=text_input(
                    "answer",
                    placeholder="센 수를 짧게 알려줘",
                ),
                fallback_text="내가 한꺼번에 물어봤네. 점이 몇 개인지 먼저 알려줘.",
            ),
            StepDefinition(
                id="short_tracking",
                prompt=short_prompt,
                target_slots=["tracking"],
                input=text_input(
                    "tracking",
                    placeholder="네가 센 방법을 알려줘",
                ),
                fallback_text=short_prompt,
            ),
        ],
        ExpressionLevel.L2: [
            StepDefinition(
                id="choose_count",
                prompt="지금 점이 몇 개야?",
                target_slots=["answer"],
                input=choice_input(["answer"], answer_choices),
                choice_effects=answer_effects,
                fallback_text="말로만 들으려니 헷갈려. 점이 몇 개인지 골라 볼까?",
            ),
            StepDefinition(
                id="choose_tracking",
                prompt="나 점을 셀 때 뭘 해야 할지 헷갈려... 같이 골라 볼까?",
                target_slots=["tracking"],
                input=choice_input(
                    ["tracking"],
                    [
                        option(f"tracking_{index}", label)
                        for index, label in enumerate(short_options)
                    ],
                ),
                choice_effects={
                    f"tracking_{index}": (
                        {"tracking": "count_each_once"} if label == short_correct else {}
                    )
                    for index, label in enumerate(short_options)
                },
                fallback_text="나 점을 셀 때 뭘 해야 할지 헷갈려... 같이 골라 볼까?",
            ),
        ],
        ExpressionLevel.L1: [
            StepDefinition(
                id="guided_count",
                prompt="지금 점이 몇 개야?",
                target_slots=["answer"],
                input=choice_input(["answer"], answer_choices),
                choice_effects=answer_effects,
                fallback_text="화면을 보며 센 수부터 골라 보자.",
            ),
            StepDefinition(
                id="complete_tracking",
                prompt="점을 하나씩 보면서 □.",
                target_slots=["tracking"],
                input=InputContract(
                    kind=InputKind.FILL,
                    target_slots=["tracking"],
                    choices=[
                        option("say_one_number", "하나, 둘, 셋 하고 세어"),
                        option("count_twice", "같은 점을 두 번 세어"),
                        option("skip_dots", "점을 건너뛰며 세어"),
                    ],
                    config={"sentence": "점을 하나씩 보면서 □."},
                ),
                choice_effects={
                    "say_one_number": {"tracking": "count_each_once"},
                    "count_twice": {},
                    "skip_dots": {},
                },
                fallback_text="도움 카드 문장의 빈칸을 같이 채워 보자.",
            ),
        ],
        ExpressionLevel.L0: [
            StepDefinition(
                id="joint_counting",
                prompt="도움 카드 문장을 나와 같이 읽어볼까?",
                target_slots=["answer", "tracking"],
                input=InputContract(
                    kind=InputKind.JOINT,
                    target_slots=["answer", "tracking"],
                    config={
                        "text": task.coauthored_note,
                        "completion_values": {
                            "answer": expected_answer,
                            "tracking": "count_each_once",
                        },
                    },
                ),
                fallback_text="도움 카드 문장을 나와 같이 읽어볼까?",
            )
        ],
    }


def _configure_number_compare_task(
    task: TaskDefinition,
    *,
    expected_answer: str | int | float | bool,
    answer_choices: list[ChoiceOption],
    answer_effects: dict[str, dict[str, str | int | float | bool]],
    spec: HomeTeachingSpec,
) -> None:
    """Separate the comparison conclusion from the child's reason.

    A short conclusion such as ``오른쪽`` is useful, but it is not yet a
    note-worthy explanation.  Conversely, correctly stating the two counts is
    a legitimate way to explain the comparison; the child must not be forced
    to recite a single strategy such as one-to-one pairing.
    """

    guided_answer_choices = [
        choice for choice in answer_choices if choice.label in {"왼쪽", "오른쪽"}
    ]
    guided_answer_effects = {
        choice.id: answer_effects[choice.id] for choice in guided_answer_choices
    }

    task.slots = {
        "answer": SlotDefinition(
            id="answer",
            description="점이 더 많은 쪽",
            semantic_role="conclusion",
            expected=expected_answer,
            aliases=["오른쪽", "오른쪽이 더 많아", "오른쪽이 커", "5개인 쪽"],
            fact_sentence="오른쪽에 점이 더 많아.",
        ),
        "reason": SlotDefinition(
            id="reason",
            description=(
                "왼쪽 3개와 오른쪽 5개를 세거나 3과 5를 비교해 오른쪽이 더 많음을 설명한 근거"
            ),
            semantic_role="reason",
            expected="count_comparison",
            aliases=[
                *spec.valid_explanations,
                "왼쪽은 3개고 오른쪽은 5개",
                "왼쪽 3개 오른쪽 5개",
                "왼쪽은 세 개고 오른쪽은 다섯 개",
                "3보다 5가 커",
                "5가 3보다 커",
            ],
            fact_sentence="왼쪽은 3개, 오른쪽은 5개라서 오른쪽이 더 많아.",
        ),
    }
    task.required_slots = ["answer", "reason"]
    task.note_slots = ["answer", "reason"]
    task.text_explanation_slots = ["reason"]
    task.note_direct_conclusion = "그래서 오른쪽에 점이 더 많다는 걸 알았어."
    task.steps = {
        ExpressionLevel.L4: [
            StepDefinition(
                id="free_comparison_and_reason",
                prompt=spec.effective_l4_prompt,
                target_slots=["answer", "reason"],
                input=text_input(
                    "answer",
                    "reason",
                    placeholder="어느 쪽인지와 까닭을 알려줘",
                ),
                fallback_text=spec.effective_l4_prompt,
            )
        ],
        ExpressionLevel.L3: [
            StepDefinition(
                id="short_comparison",
                prompt="왼쪽과 오른쪽 중 어느 쪽에 점이 더 많아?",
                target_slots=["answer"],
                input=text_input("answer", placeholder="어느 쪽인지 짧게 알려줘"),
                fallback_text="내가 한꺼번에 물어봤네. 어느 쪽이 더 많아?",
            ),
            StepDefinition(
                id="short_reason",
                prompt="나 3이랑 5를 어떻게 비교할지 헷갈려... 알려줄 수 있어?",
                target_slots=["reason"],
                input=text_input("reason", placeholder="3이랑 5를 보고 알려줘"),
                fallback_text="나 3이랑 5를 어떻게 비교할지 헷갈려... 알려줄 수 있어?",
            ),
        ],
        ExpressionLevel.L2: [
            StepDefinition(
                id="choose_comparison",
                prompt="왼쪽과 오른쪽 중 어느 쪽에 점이 더 많아?",
                target_slots=["answer"],
                input=choice_input(["answer"], answer_choices),
                choice_effects=answer_effects,
                fallback_text="말로 어렵다면 어느 쪽인지 같이 골라 보자.",
            ),
            StepDefinition(
                id="choose_reason",
                prompt="나 3이랑 5를 보고도 헷갈려... 같이 골라 볼까?",
                target_slots=["reason"],
                input=choice_input(
                    ["reason"],
                    [
                        option("counts_right", "왼쪽은 3개, 오른쪽은 5개라서"),
                        option("counts_left", "왼쪽은 5개, 오른쪽은 3개라서"),
                        option("counts_same", "두 쪽 모두 5개라서"),
                    ],
                ),
                choice_effects={
                    "counts_right": {"reason": "count_comparison"},
                    "counts_left": {},
                    "counts_same": {},
                },
                fallback_text="나 두 쪽에서 센 수가 헷갈려... 같이 골라 볼까?",
            ),
        ],
        ExpressionLevel.L1: [
            StepDefinition(
                id="guided_comparison",
                prompt="점이 5개인 쪽은 어느 쪽이야?",
                target_slots=["answer"],
                input=choice_input(["answer"], guided_answer_choices),
                choice_effects=guided_answer_effects,
                fallback_text="점이 5개인 쪽부터 같이 찾아보자.",
            ),
            StepDefinition(
                id="complete_comparison",
                prompt="왼쪽은 3개, 오른쪽은 5개라서 □.",
                target_slots=["reason"],
                input=InputContract(
                    kind=InputKind.FILL,
                    target_slots=["reason"],
                    choices=[
                        option("right_more", "오른쪽이 더 많아"),
                        option("left_more", "왼쪽이 더 많아"),
                        option("same", "두 쪽이 똑같아"),
                    ],
                    config={"sentence": "왼쪽은 3개, 오른쪽은 5개라서 □."},
                ),
                choice_effects={
                    "right_more": {"reason": "count_comparison"},
                    "left_more": {},
                    "same": {},
                },
                fallback_text="센 수를 문장에 넣어 같이 마무리해 보자.",
            ),
        ],
        ExpressionLevel.L0: [
            StepDefinition(
                id="joint_comparison",
                prompt="도움 카드 문장을 나와 같이 읽어볼까?",
                target_slots=["answer", "reason"],
                input=InputContract(
                    kind=InputKind.JOINT,
                    target_slots=["answer", "reason"],
                    config={
                        "text": task.coauthored_note,
                        "completion_values": {
                            "answer": expected_answer,
                            "reason": "count_comparison",
                        },
                    },
                ),
                fallback_text="도움 카드 문장을 나와 같이 읽어볼까?",
            )
        ],
    }


QUEUE_TASK_ID = "cafe_queue"
HOME_TEACH_TASK_ID = "home_teaching"
BUDGET_MENU_TASK_ID = "cafe_budget_menu_pick"
TOTAL_MENU_PICK_TASK_ID = "cafe_total_menu_pick"
TOTAL_CALC_TASK_ID = "cafe_total_calculation"
CHANGE_TASK_ID = "cafe_change"
CAFE_CHANGE_PAYMENT_AMOUNT = 10_000
MENU_SCENARIO_IDS = {"cafe_budget_menu", "cafe_menu_total", "cafe_change"}

SCENARIOS: dict[str, ScenarioDefinition] = {
    "home_teach": ScenarioDefinition(
        id="home_teach",
        scene=SceneType.HOME_TEACH,
        title="반복한 내용을 모르미에게 가르치기",
        task_ids=[HOME_TEACH_TASK_ID],
    ),
    "cafe_queue": ScenarioDefinition(
        id="cafe_queue",
        scene=SceneType.CAFE,
        title="1단계 줄 서기",
        task_ids=[QUEUE_TASK_ID],
    ),
    "cafe_queue_demo": ScenarioDefinition(
        id="cafe_queue_demo",
        scene=SceneType.CAFE,
        title="1단계 줄 서기(호환 ID)",
        task_ids=[QUEUE_TASK_ID],
    ),
    "cafe_budget_menu": ScenarioDefinition(
        id="cafe_budget_menu",
        scene=SceneType.CAFE,
        title="2단계 예산 안에서 메뉴 고르기",
        task_ids=[BUDGET_MENU_TASK_ID],
    ),
    "cafe_menu_total": ScenarioDefinition(
        id="cafe_menu_total",
        scene=SceneType.CAFE,
        title="3단계 메뉴값 계산하기",
        task_ids=[TOTAL_MENU_PICK_TASK_ID, TOTAL_CALC_TASK_ID],
    ),
    "cafe_change": ScenarioDefinition(
        id="cafe_change",
        scene=SceneType.CAFE,
        title="4단계 거스름돈 받기",
        task_ids=[CHANGE_TASK_ID],
    ),
}


def create_scenario_data(
    scenario_id: str,
    cafe_context: CafeSessionContext | None = None,
    rng: Any | None = None,
    *,
    queue_context: QueueSessionContext | None = None,
    curriculum_session_id: str | None = None,
    skill_id: str | None = None,
    practice_result_id: str | None = None,
) -> dict[str, Any]:
    chooser = rng or random.SystemRandom()
    data: dict[str, Any] = {}
    if scenario_id in {"cafe_queue", "cafe_queue_demo"}:
        if queue_context is not None:
            # The screen already drew the lines. Mormi must count what the child sees.
            data.update(
                left_count=queue_context.left_count,
                right_count=queue_context.right_count,
            )
        else:
            left = chooser.choice(range(1, 6))
            right = chooser.choice([value for value in range(1, 6) if value != left])
            data.update(left_count=left, right_count=right)
    if scenario_id in MENU_SCENARIO_IDS:
        if cafe_context is None:
            raise ValueError("cafe_context is required for menu scenarios")
        data.update(cafe_context.model_dump(mode="json", exclude_none=True))
    if scenario_id == "home_teach":
        if not curriculum_session_id:
            raise ValueError("curriculum_session_id is required for home_teach")
        try:
            spec = HOME_TEACHING_CATALOG[curriculum_session_id]
        except KeyError as error:
            raise ValueError(
                f"unsupported home curriculum_session_id: {curriculum_session_id}"
            ) from error
        data.update(
            curriculum_session_id=spec.id,
            skill_id=skill_id or spec.id,
            home_teaching_spec=spec.model_dump(mode="json"),
        )
        if practice_result_id:
            data["practice_result_id"] = practice_result_id
    return data


def _menu_items_from_data(data: Mapping[str, Any]) -> tuple[CafeMenuItem, ...]:
    raw_items = data.get("menu_items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("scenario_data.menu_items is required")
    return tuple(CafeMenuItem.model_validate(item) for item in raw_items)


def _menu_from_data(
    data: Mapping[str, Any],
    key: str,
    menu_items: Sequence[CafeMenuItem],
    *,
    default_to_first: bool = False,
) -> CafeMenuItem:
    menu_by_id = {item.id: item for item in menu_items}
    selected_id = data.get(key)
    if selected_id is None and default_to_first:
        return menu_items[0]
    try:
        return menu_by_id[str(selected_id)]
    except KeyError as error:
        raise ValueError(f"scenario_data.{key} must reference menu_items") from error


def get_task(task_id: str, scenario_data: Mapping[str, Any] | None = None) -> TaskDefinition:
    data = scenario_data or {}
    left_count = int(data.get("left_count", 3))
    right_count = int(data.get("right_count", 5))
    if task_id == QUEUE_TASK_ID:
        return queue_task(task_id=task_id, stage_id="queue", left=left_count, right=right_count)
    if task_id == HOME_TEACH_TASK_ID:
        raw_spec = data.get("home_teaching_spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("scenario_data.home_teaching_spec is required")
        skill_id = data.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("scenario_data.skill_id is required")
        return home_teaching_task(HomeTeachingSpec.model_validate(raw_spec), skill_id=skill_id)
    menu_items = _menu_items_from_data(data)
    mormi_menu = _menu_from_data(data, "mormi_menu_id", menu_items)
    if task_id == BUDGET_MENU_TASK_ID:
        budget = int(data["budget"])
        return menu_selection_task(
            task_id=task_id,
            stage_id="budget_menu",
            menu_items=menu_items,
            mormi_menu=mormi_menu,
            budget=budget,
            auto_total=True,
            behavior="budget_menu_selection",
            note_policy="stage",
        )
    if task_id == TOTAL_MENU_PICK_TASK_ID:
        return menu_selection_task(
            task_id=task_id,
            stage_id="menu_total",
            menu_items=menu_items,
            mormi_menu=mormi_menu,
            budget=None,
            auto_total=False,
            behavior="menu_selection",
            note_policy="none",
        )
    if task_id == TOTAL_CALC_TASK_ID:
        # Conversation creation inspects every future task before the child has
        # chosen a menu. The placeholder is used only to derive static task
        # metadata; once this task is reached, the engine has stored the real
        # child_menu_id in scenario_data.
        child_menu = _menu_from_data(
            data,
            "child_menu_id",
            menu_items,
            default_to_first=True,
        )
        return simple_calculation_task(
            task_id=task_id,
            stage_id="menu_total",
            title="메뉴값 계산하기",
            left=mormi_menu.price,
            right=child_menu.price,
            operation="addition",
            left_label=mormi_menu.name,
            right_label=child_menu.name,
            behavior="menu_total",
            note_policy="stage",
            coauthored_note="두 메뉴의 전체 가격은 각 메뉴 가격을 더해서 구해.",
            context={
                "budget": None,
                "mormi_menu": mormi_menu.model_dump(),
                "child_menu": child_menu.model_dump(),
            },
        )
    if task_id == CHANGE_TASK_ID:
        return simple_calculation_task(
            task_id=task_id,
            stage_id="change",
            title="거스름돈 받기",
            left=CAFE_CHANGE_PAYMENT_AMOUNT,
            right=mormi_menu.price,
            operation="subtraction",
            left_label="낸 돈",
            right_label="메뉴 값",
            behavior="change",
            note_policy="stage",
            coauthored_note="거스름돈은 낸 돈에서 메뉴 값을 빼서 구해.",
            context={
                "payment": CAFE_CHANGE_PAYMENT_AMOUNT,
                "menu_total": mormi_menu.price,
                "mormi_menu": mormi_menu.model_dump(),
            },
        )
    raise KeyError(f"Unknown task: {task_id}")


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as error:
        raise KeyError(f"Unknown scenario: {scenario_id}") from error


def validate_content() -> None:
    validation_context = CafeSessionContext(
        menu_items=[
            CafeMenuItem(id="sample-a", name="메뉴 A", price=2000),
            CafeMenuItem(id="sample-b", name="메뉴 B", price=3000),
        ],
        mormi_menu_id="sample-a",
        budget=10000,
    )
    for scenario in SCENARIOS.values():
        if scenario.id == "home_teach":
            for spec in HOME_TEACHING_CATALOG.values():
                scenario_data = create_scenario_data(
                    scenario.id,
                    curriculum_session_id=spec.id,
                    skill_id=spec.id,
                )
                get_task(HOME_TEACH_TASK_ID, scenario_data)
            continue
        scenario_data = create_scenario_data(
            scenario.id,
            validation_context if scenario.id in MENU_SCENARIO_IDS else None,
        )
        for task_id in scenario.task_ids:
            task_data = scenario_data
            if task_id == TOTAL_CALC_TASK_ID:
                task_data = {**scenario_data, "child_menu_id": "sample-b"}
            get_task(task_id, task_data)
    task_ids = {
        task_id
        for scenario in SCENARIOS.values()
        if scenario.id != "home_teach"
        for task_id in scenario.task_ids
    }
    sample_data = {
        **create_scenario_data("cafe_menu_total", validation_context),
        "child_menu_id": "sample-b",
    }
    tasks_to_validate = [
        get_task(
            task_id,
            sample_data if task_id != QUEUE_TASK_ID else {},
        )
        for task_id in task_ids
    ]
    tasks_to_validate.extend(
        home_teaching_task(spec, skill_id=spec.id) for spec in HOME_TEACHING_CATALOG.values()
    )
    for task in tasks_to_validate:
        if set(task.required_slots) - set(task.slots):
            raise ValueError(f"{task.id}: required slot is undefined")
        for slot in task.slots.values():
            expected_mode = (
                "semantic_support"
                if slot.semantic_role in {"method", "reason", "explanation"}
                else "canonical_value"
            )
            if slot.resolved_evaluation_mode != expected_mode:
                raise ValueError(
                    f"{task.id}/{slot.id}: {slot.semantic_role} must use {expected_mode}"
                )
            if slot.is_semantic_support and slot.preserve_value:
                raise ValueError(
                    f"{task.id}/{slot.id}: semantic support cannot preserve a model value"
                )
        if task.note_policy != "none":
            if not task.effective_note_slots:
                raise ValueError(f"{task.id}: note-producing task needs note slots")
            if set(task.effective_note_slots) - set(task.required_slots):
                raise ValueError(f"{task.id}: note slots must be required completion slots")
            if not task.note_context.strip():
                raise ValueError(f"{task.id}: note-producing task needs reviewed note context")
        if set(task.text_explanation_slots) - set(task.required_slots):
            raise ValueError(f"{task.id}: explanation slots must be required completion slots")
        for level in ExpressionLevel:
            if level not in task.steps or not task.steps[level]:
                raise ValueError(f"{task.id}: missing steps for {level}")
        reviewed_steps = [item for steps in task.steps.values() for item in steps]
        if task.entry_step is not None:
            reviewed_steps.append(task.entry_step)
        for step in reviewed_steps:
            if set(step.target_slots) - set(task.slots):
                raise ValueError(f"{task.id}/{step.id}: target slot is undefined")
            if set(step.optional_slots) - set(task.slots):
                raise ValueError(f"{task.id}/{step.id}: optional slot is undefined")
            if _TEACHER_EVALUATION_COPY.search(step.prompt) or _TEACHER_EVALUATION_COPY.search(
                step.fallback_text
            ):
                raise ValueError(
                    f"{task.id}/{step.id}: Mormi must ask from its own confusion, not evaluate"
                )
            if step.input.kind in {InputKind.CHOICES, InputKind.FILL}:
                visible_choice_ids = {choice.id for choice in step.input.choices}
                if set(step.choice_effects) != visible_choice_ids:
                    raise ValueError(
                        f"{task.id}/{step.id}: every visible choice needs one reviewed effect"
                    )
                correct_choice_ids = {
                    choice_id
                    for choice_id, effects in step.choice_effects.items()
                    if effects
                    and set(step.target_slots).issubset(effects)
                    and all(
                        task.slots[slot_id].accepts(value) for slot_id, value in effects.items()
                    )
                }
                if not correct_choice_ids:
                    raise ValueError(
                        f"{task.id}/{step.id}: structured input needs a reviewed correct choice"
                    )
            if step.input.kind is InputKind.JOINT:
                completion_values = step.input.config.get("completion_values")
                if not isinstance(completion_values, Mapping):
                    raise ValueError(f"{task.id}/{step.id}: joint input requires completion_values")
                if set(step.target_slots) - set(completion_values):
                    raise ValueError(
                        f"{task.id}/{step.id}: completion_values must cover target slots"
                    )
        for hint_level in (HintLevel.H1, HintLevel.H2, HintLevel.H3):
            if hint_level not in task.hints:
                raise ValueError(f"{task.id}: missing {hint_level} hint")


validate_content()
