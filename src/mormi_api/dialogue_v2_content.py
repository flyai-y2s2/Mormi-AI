from __future__ import annotations

import json
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .copy_quality import validate_child_facing_math_copy
from .schemas import (
    BooleanValueV2,
    CanonicalValueV2,
    ChoiceValueV2,
    ExpressionLevel,
    HintLevel,
    MoneyValueV2,
    NumberValueV2,
)

CAFE_REQUIRED_HOME_SESSION_IDS = frozenset(
    {
        "number-count",
        "number-compare",
        "money-count",
        "money-price",
        "money-budget",
    }
)

# These four sessions prepare concepts used at the amusement park.  They are
# not a BE unlock condition: the current product unlocks the park after the
# cafe journey.  Keeping that distinction in the content contract prevents AI
# copy from promising an unlock that the AI service does not own.
AMUSEMENT_PREPARATION_HOME_SESSION_IDS = frozenset(
    {
        "multiply-groups",
        "divide-share",
        "divide-group",
        "multiply-easy-tables",
    }
)

REQUIRED_HOME_SESSION_IDS = (
    CAFE_REQUIRED_HOME_SESSION_IDS | AMUSEMENT_PREPARATION_HOME_SESSION_IDS
)

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_TEACHER_OR_EVALUATION_COPY = re.compile(
    r"(맞아|틀렸|정답|오답|잘했|다시\s*생각|왜\s*그렇게\s*생각|"
    r"설명해\s*봐|말해\s*봐|확인했어|기억했어|네가\s*말한\s*데까지)"
)
_JOINT_WORDS = re.compile(r"((?<!똑)같이|함께)")
_MORMI_FIRST_PERSON = re.compile(
    r"(?:^|[\s,(])(?:나(?:는|랑|와|도|에게|한테|를|의)?|"
    r"내(?:가|게|것|꺼)?|모르미)(?=[\s,.;!?~]|$)"
)
_HELP_REQUEST_COPY = re.compile(r"(알려줄|도와줄|해\s*줄|\?)")
_ARABIC_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_H0_OPERATION_NOTATION = re.compile(
    r"\d[\d,]*(?:원|개|명|장|잔|권)?\s*[+×÷-]\s*\d"
)
_H0_OPERATION_LANGUAGE = re.compile(r"(더하|더해|빼기|빼면|곱하|나누|차이를\s*찾)")
_ARITHMETIC_EQUATION = re.compile(
    r"((?:\d[\d,]*(?:\.\d+)?\s*[+×÷-]\s*)+\d[\d,]*(?:\.\d+)?)"
    r"\s*=\s*(\d[\d,]*(?:\.\d+)?)"
)


class ContentPackV2Model(BaseModel):
    """Strict base for immutable, human-reviewed V2 content artifacts."""

    model_config = ConfigDict(extra="forbid")


class RenderedFactBindingV2(ContentPackV2Model):
    """Binds a graph fact to the exact H0 visual field that represents it."""

    fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    exposure: Literal["explicit_given", "perceptual_target"] = "explicit_given"
    source_path: str = Field(
        pattern=r"^visual(?:\.[A-Za-z][A-Za-z0-9_]*|\.\d+)+$",
        max_length=180,
    )
    label_path: str | None = Field(
        default=None,
        pattern=r"^visual(?:\.[A-Za-z][A-Za-z0-9_]*|\.\d+)+$",
        max_length=180,
    )
    rendered_label: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_label_binding(self) -> RenderedFactBindingV2:
        if (self.label_path is None) != (self.rendered_label is None):
            raise ValueError("rendered fact label_path and rendered_label must be paired")
        return self


class SourceProblemV2(ContentPackV2Model):
    """The AI-owned teaching example; it need not copy the last FE drill."""

    prompt: str = Field(min_length=1, max_length=220)
    answers: list[str] = Field(min_length=2, max_length=8)
    correct: str = Field(min_length=1, max_length=100)
    visual: dict[str, Any]
    rendered_facts: list[RenderedFactBindingV2] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_problem(self) -> SourceProblemV2:
        if not self.prompt.rstrip().endswith("?"):
            raise ValueError("source problem prompt must be a complete question")
        if len(self.answers) != len(set(self.answers)):
            raise ValueError("source problem answers must be unique")
        if self.correct not in self.answers:
            raise ValueError("source problem correct must be one of answers")
        if not isinstance(self.visual.get("type"), str):
            raise ValueError("source problem visual.type is required")
        binding_keys = [
            (binding.fact_id, binding.source_path) for binding in self.rendered_facts
        ]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("source problem rendered fact bindings must be unique")
        visual_copy = json.dumps(self.visual, ensure_ascii=False)
        if _H0_OPERATION_NOTATION.search(visual_copy) or _H0_OPERATION_LANGUAGE.search(
            visual_copy
        ):
            raise ValueError("H0 visual cannot reveal an operation or equation")
        return self


class ContentFactV2(ContentPackV2Model):
    fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    role: Literal[
        "visible_condition",
        "visible_value",
        "intermediate_result",
        "final_answer",
    ]
    value: CanonicalValueV2
    speaker_label: str = Field(min_length=1, max_length=100)
    initially_visible: bool
    required_for_completion: bool = False
    accepted_surface_forms: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_visibility_and_aliases(self) -> ContentFactV2:
        if self.role in {"intermediate_result", "final_answer"} and self.initially_visible:
            raise ValueError("derived facts cannot be initially visible")
        if self.required_for_completion and not self.accepted_surface_forms:
            raise ValueError("required facts need accepted surface forms")
        if len(self.accepted_surface_forms) != len(set(self.accepted_surface_forms)):
            raise ValueError("fact surface forms must be unique")
        if isinstance(self.value, MoneyValueV2 | NumberValueV2):
            expected = _decimal_value(self.value)
            for surface in self.accepted_surface_forms:
                numbers = _surface_numbers(surface)
                if numbers and numbers != [expected]:
                    raise ValueError("numeric fact surface form must equal canonical truth")
        return self


class RelationRubricV2(ContentPackV2Model):
    sufficient: list[str] = Field(min_length=1, max_length=12)
    partial: list[str] = Field(min_length=1, max_length=12)
    incorrect: list[str] = Field(min_length=1, max_length=12)


class ContentRelationV2(ContentPackV2Model):
    relation_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    role: Literal["procedure_step", "explanation"]
    operation: Literal[
        "counting",
        "comparison",
        "addition",
        "subtraction",
        "multiplication",
        "division",
    ]
    input_fact_ids: list[str] = Field(min_length=1, max_length=16)
    output_fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    evaluation_mode: Literal["exact_semantic", "open_semantic_support"]
    speaker_label: str = Field(min_length=1, max_length=120)
    rubric: RelationRubricV2
    required_for_completion: bool = False


def _decimal_value(value: CanonicalValueV2) -> Decimal:
    if isinstance(value, MoneyValueV2):
        return Decimal(str(value.amount))
    if isinstance(value, NumberValueV2):
        return Decimal(str(value.value))
    raise ValueError("arithmetic relation facts must use typed numeric values")


def _numeric_dimension(value: CanonicalValueV2) -> tuple[str, str | None]:
    if isinstance(value, MoneyValueV2):
        return "money", value.currency
    if isinstance(value, NumberValueV2):
        return "number", value.unit
    raise ValueError("arithmetic relation facts must use typed numeric values")


def _surface_numbers(value: object) -> list[Decimal]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [Decimal(str(value))]
    if isinstance(value, str):
        return [
            Decimal(match.group(0).replace(",", ""))
            for match in _ARABIC_NUMBER.finditer(value)
        ]
    if isinstance(value, list):
        return [number for item in value for number in _surface_numbers(item)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _surface_numbers(item)]
    return []


def _validate_authored_equations(text: str) -> None:
    for match in _ARITHMETIC_EQUATION.finditer(text):
        expression, rendered_result = match.groups()
        operators = re.findall(r"[+×÷-]", expression)
        if not operators or len(set(operators)) != 1:
            raise ValueError("authored equations must use one explicit operation")
        operands = [
            Decimal(token.replace(",", ""))
            for token in re.split(r"[+×÷-]", expression)
        ]
        operator = operators[0]
        if operator == "+":
            expected = sum(operands, start=Decimal(0))
        elif operator == "×":
            expected = Decimal(1)
            for operand in operands:
                expected *= operand
        elif len(operands) != 2:
            raise ValueError("subtraction and division equations must be binary")
        elif operator == "-":
            expected = operands[0] - operands[1]
        else:
            if operands[1] == 0:
                raise ValueError("authored equation cannot divide by zero")
            expected = operands[0] / operands[1]
        if expected != Decimal(rendered_result.replace(",", "")):
            raise ValueError("child-facing authored equation is mathematically invalid")


def _resolve_visual_path(visual: dict[str, Any], source_path: str) -> object:
    current: object = visual
    for part in source_path.split(".")[1:]:
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"rendered fact path does not exist: {source_path}")
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise ValueError(f"rendered fact path does not exist: {source_path}")
            current = current[index]
        else:
            raise ValueError(f"rendered fact path does not exist: {source_path}")
    return current


def _validate_rendered_value(
    fact: ContentFactV2,
    rendered_value: object,
    *,
    source_path: str,
) -> None:
    if isinstance(fact.value, MoneyValueV2 | NumberValueV2):
        expected = _decimal_value(fact.value)
        if expected not in _surface_numbers(rendered_value):
            raise ValueError(
                f"rendered fact {fact.fact_id} does not show its canonical value at "
                f"{source_path}"
            )
    elif isinstance(fact.value, BooleanValueV2):
        if rendered_value is not fact.value.value:
            raise ValueError(f"rendered fact {fact.fact_id} does not show its boolean value")
    elif not isinstance(rendered_value, str) or not rendered_value.strip():
        # Text and choice facts use human-readable visual labels. Their semantic
        # identity is reviewed, while numeric values above are checked exactly.
        raise ValueError(f"rendered fact {fact.fact_id} needs a non-empty visual label")


def _validate_relation_math(
    relation: ContentRelationV2,
    facts: dict[str, ContentFactV2],
) -> None:
    inputs = [facts[fact_id].value for fact_id in relation.input_fact_ids]
    output = facts[relation.output_fact_id].value
    operation = relation.operation

    if len(relation.input_fact_ids) != len(set(relation.input_fact_ids)):
        raise ValueError(f"relation {relation.relation_id} inputs must be unique")
    if operation == "counting":
        if not isinstance(output, NumberValueV2):
            raise ValueError("counting relation output must be a typed number")
        return
    numeric_inputs = [_decimal_value(value) for value in inputs]
    output_number = _decimal_value(output) if operation != "comparison" else None

    if operation == "comparison":
        if len(inputs) != 2 or not isinstance(output, ChoiceValueV2):
            raise ValueError("comparison needs two numeric inputs and one choice output")
        if _numeric_dimension(inputs[0]) != _numeric_dimension(inputs[1]):
            raise ValueError("comparison inputs must use the same numeric dimension")
        expected_choice = (
            "left"
            if numeric_inputs[0] > numeric_inputs[1]
            else "right"
            if numeric_inputs[0] < numeric_inputs[1]
            else "same"
        )
        if output.choice_id != expected_choice:
            raise ValueError(f"relation {relation.relation_id} comparison result is invalid")
        return

    assert output_number is not None
    if operation in {"addition", "subtraction"}:
        if operation == "subtraction" and len(inputs) != 2:
            raise ValueError("subtraction relation must have exactly two ordered inputs")
        dimensions = {_numeric_dimension(value) for value in [*inputs, output]}
        if len(dimensions) != 1:
            raise ValueError(f"{operation} inputs and output must share a dimension")
        expected = (
            sum(numeric_inputs, start=Decimal(0))
            if operation == "addition"
            else numeric_inputs[0] - numeric_inputs[1]
        )
        if output_number != expected:
            raise ValueError(f"relation {relation.relation_id} arithmetic result is invalid")
        return

    if len(inputs) != 2:
        raise ValueError(f"{operation} relation must have exactly two ordered inputs")
    left_dimension = _numeric_dimension(inputs[0])
    right_dimension = _numeric_dimension(inputs[1])
    output_dimension = _numeric_dimension(output)

    if operation == "multiplication":
        if left_dimension[0] == "money" and right_dimension[0] == "number":
            if output_dimension != left_dimension:
                raise ValueError("money multiplied by a count must produce the same currency")
        elif left_dimension[0] == right_dimension[0] == output_dimension[0] == "number":
            pass
        else:
            raise ValueError("unsupported multiplication dimensions")
        if output_number != numeric_inputs[0] * numeric_inputs[1]:
            raise ValueError(f"relation {relation.relation_id} arithmetic result is invalid")
        return

    if numeric_inputs[1] == 0:
        raise ValueError("division relation cannot divide by zero")
    if left_dimension[0] == "money" and right_dimension[0] == "number":
        if output_dimension != left_dimension:
            raise ValueError("money divided by a count must produce the same currency")
    elif left_dimension[0] == right_dimension[0] == "money":
        if left_dimension != right_dimension or output_dimension[0] != "number":
            raise ValueError("money divided by money must produce a number")
    elif left_dimension[0] == right_dimension[0] == output_dimension[0] == "number":
        pass
    else:
        raise ValueError("unsupported division dimensions")
    if output_number * numeric_inputs[1] != numeric_inputs[0]:
        raise ValueError(f"relation {relation.relation_id} arithmetic result is invalid")


def _validate_relation_rubric(
    relation: ContentRelationV2,
    facts: dict[str, ContentFactV2],
) -> None:
    allowed_numbers = {
        _decimal_value(facts[fact_id].value)
        for fact_id in [*relation.input_fact_ids, relation.output_fact_id]
        if isinstance(facts[fact_id].value, MoneyValueV2 | NumberValueV2)
    }
    lines = [
        *relation.rubric.sufficient,
        *relation.rubric.partial,
        *relation.rubric.incorrect,
    ]
    for line in lines:
        if not set(_surface_numbers(line)).issubset(allowed_numbers):
            raise ValueError(
                f"relation {relation.relation_id} rubric contains an undeclared value"
            )
        _validate_authored_equations(line)


class CompletionContractV2(ContentPackV2Model):
    required_fact_ids: list[str] = Field(min_length=1, max_length=20)
    required_relation_ids: list[str] = Field(min_length=1, max_length=20)


class ReasoningGraphV2(ContentPackV2Model):
    facts: list[ContentFactV2] = Field(min_length=2, max_length=50)
    relations: list[ContentRelationV2] = Field(min_length=1, max_length=50)
    completion: CompletionContractV2
    open_auxiliary_claims: bool = True

    @model_validator(mode="after")
    def validate_graph(self) -> ReasoningGraphV2:
        fact_ids = [fact.fact_id for fact in self.facts]
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("reasoning graph fact ids must be unique")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("reasoning graph relation ids must be unique")

        facts_by_id = {fact.fact_id: fact for fact in self.facts}
        fact_id_set = set(fact_ids)
        for relation in self.relations:
            if not set(relation.input_fact_ids).issubset(fact_id_set):
                raise ValueError("relation inputs must reference graph facts")
            if relation.output_fact_id not in fact_id_set:
                raise ValueError("relation output must reference a graph fact")
            if relation.output_fact_id in relation.input_fact_ids:
                raise ValueError("relation cannot consume its own output")
            _validate_relation_math(relation, facts_by_id)
            _validate_relation_rubric(relation, facts_by_id)

        required_facts = {fact.fact_id for fact in self.facts if fact.required_for_completion}
        required_relations = {
            relation.relation_id
            for relation in self.relations
            if relation.required_for_completion
        }
        if set(self.completion.required_fact_ids) != required_facts:
            raise ValueError("completion fact ids must equal required graph facts")
        if set(self.completion.required_relation_ids) != required_relations:
            raise ValueError("completion relation ids must equal required graph relations")
        if len(self.completion.required_fact_ids) != len(
            set(self.completion.required_fact_ids)
        ):
            raise ValueError("completion fact ids must be unique")
        if len(self.completion.required_relation_ids) != len(
            set(self.completion.required_relation_ids)
        ):
            raise ValueError("completion relation ids must be unique")

        # A pack may offer alternative relations to the same output, but no
        # authored dependency path may contain a cycle.
        dependencies: dict[str, set[str]] = {fact_id: set() for fact_id in fact_ids}
        for relation in self.relations:
            dependencies[relation.output_fact_id].update(relation.input_fact_ids)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(fact_id: str) -> None:
            if fact_id in visiting:
                raise ValueError("reasoning graph must be acyclic")
            if fact_id in visited:
                return
            visiting.add(fact_id)
            for dependency in dependencies[fact_id]:
                visit(dependency)
            visiting.remove(fact_id)
            visited.add(fact_id)

        for fact_id in fact_ids:
            visit(fact_id)
        return self


class TargetRefV2(ContentPackV2Model):
    target_kind: Literal["fact", "relation"]
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    ask_kind: Literal["answer", "reason_or_method"]


class QuestionPlanV2(ContentPackV2Model):
    plan_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    targets: list[TargetRefV2] = Field(min_length=1, max_length=4)
    reviewed_fallback: str = Field(min_length=1, max_length=220)


class CopySlotV2(ContentPackV2Model):
    """A semantic slot whose generated text lives in the durable copy cache."""

    copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    purpose: Literal["initial_help", "l2_question", "l0_intro", "l0_action"]
    targets: list[TargetRefV2] = Field(min_length=1, max_length=4)
    generation_brief: str = Field(min_length=1, max_length=240)
    reviewed_fallback: str = Field(min_length=1, max_length=220)


class ChoiceEffectV2(ContentPackV2Model):
    verdict: Literal["correct", "incorrect"]
    target_kind: Literal["fact", "relation"]
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    interpreted_value: CanonicalValueV2 | None = None
    misconception_code: str | None = Field(default=None, pattern=_ID_PATTERN, max_length=120)

    @model_validator(mode="after")
    def validate_effect(self) -> ChoiceEffectV2:
        if self.verdict == "correct":
            if self.misconception_code is not None:
                raise ValueError("correct choice cannot declare a misconception")
            if self.target_kind == "fact" and self.interpreted_value is None:
                raise ValueError("correct fact choice needs interpreted_value")
            if self.target_kind == "relation" and self.interpreted_value is not None:
                raise ValueError("relation choice cannot declare interpreted_value")
        else:
            if self.interpreted_value is not None:
                raise ValueError("incorrect choice cannot verify a value")
            if self.misconception_code is None:
                raise ValueError("incorrect choice needs misconception_code")
        return self


class L2ChoiceV2(ContentPackV2Model):
    choice_id: str = Field(pattern=_ID_PATTERN, max_length=100)
    label: str = Field(min_length=1, max_length=80)
    effect: ChoiceEffectV2


class L2ChoicePlanV2(ContentPackV2Model):
    plan_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    target: TargetRefV2
    copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    choices: list[L2ChoiceV2] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def validate_choices(self) -> L2ChoicePlanV2:
        ids = [choice.choice_id for choice in self.choices]
        if len(ids) != len(set(ids)):
            raise ValueError("L2 choice ids must be unique")
        if sum(choice.effect.verdict == "correct" for choice in self.choices) != 1:
            raise ValueError("L2 plan needs exactly one correct choice")
        for choice in self.choices:
            if (
                choice.effect.target_kind,
                choice.effect.target_id,
            ) != (self.target.target_kind, self.target.target_id):
                raise ValueError("L2 effects must target the plan target")
        return self


class JointFactCompletionV2(ContentPackV2Model):
    target_kind: Literal["fact"] = "fact"
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    value: CanonicalValueV2


class JointRelationCompletionV2(ContentPackV2Model):
    target_kind: Literal["relation"] = "relation"
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    satisfied: Literal[True] = True


JointCompletionV2 = Annotated[
    JointFactCompletionV2 | JointRelationCompletionV2,
    Field(discriminator="target_kind"),
]


class L0JointPlanV2(ContentPackV2Model):
    action_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    intro_copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    action_copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    completion_values: list[JointCompletionV2] = Field(min_length=2, max_length=20)
    button_label: str = Field(min_length=1, max_length=30)


class HelpCardV2(ContentPackV2Model):
    level: HintLevel
    support_type: Literal["attention", "guided_action", "joint_model"]
    answer_policy: Literal["hidden", "partial", "revealed"]
    body: str = Field(min_length=1, max_length=240)
    action: str | None = Field(default=None, min_length=1, max_length=160)
    revealed_fact_ids: list[str] = Field(default_factory=list, max_length=20)
    revealed_relation_ids: list[str] = Field(default_factory=list, max_length=20)


class HelpPlanV2(ContentPackV2Model):
    H1: HelpCardV2
    H2: HelpCardV2
    H3: HelpCardV2

    @model_validator(mode="after")
    def validate_levels(self) -> HelpPlanV2:
        expected = {
            HintLevel.H1: (self.H1, "attention", "hidden"),
            HintLevel.H2: (self.H2, "guided_action", "partial"),
            HintLevel.H3: (self.H3, "joint_model", "revealed"),
        }
        for level, (card, support_type, answer_policy) in expected.items():
            if card.level is not level:
                raise ValueError("help card level does not match its key")
            if card.support_type != support_type or card.answer_policy != answer_policy:
                raise ValueError("help card support profile does not match its level")
        if self.H1.revealed_fact_ids or self.H1.revealed_relation_ids:
            raise ValueError("H1 cannot reveal derived facts or relations")
        return self


class JourneyLinkV2(ContentPackV2Model):
    role: Literal["cafe_required", "amusement_preparation"]
    direct_transfer_scenario_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def park_link_is_not_an_unlock_contract(self) -> JourneyLinkV2:
        if self.role == "cafe_required" and self.direct_transfer_scenario_ids:
            raise ValueError("cafe required home packs do not map to park scenarios")
        return self


class ContentPoliciesV2(ContentPackV2Model):
    ladder_policy_version: Literal["existing"] = "existing"
    entry_expression_level: Literal[ExpressionLevel.L4] = ExpressionLevel.L4
    entry_hint_level: Literal[HintLevel.H0] = HintLevel.H0
    completion_policy: Literal["all_required_graph_targets"] = (
        "all_required_graph_targets"
    )
    note_policy: Literal["verified_child_or_coauthored"] = (
        "verified_child_or_coauthored"
    )
    note_relation_ids: list[str] = Field(min_length=1, max_length=10)
    dictionary_card_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    taught_reward_requires_independent_evidence: Literal[True] = True
    l0_completion_outcome: Literal["supported"] = "supported"


class RequiredHomeTeachingPackV2(ContentPackV2Model):
    schema_version: Literal["content-pack-v2"] = "content-pack-v2"
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    migration_source_content_version: int = Field(
        ge=1,
        description=(
            "One-time provenance for the legacy sample used during migration; "
            "V2 content does not track or synchronize to later legacy edits."
        ),
    )
    curriculum_session_id: str = Field(pattern=_ID_PATTERN, max_length=80)
    task_id: Literal["home_teaching"] = "home_teaching"
    stage_id: Literal["home_teach"] = "home_teach"
    title: str = Field(min_length=1, max_length=100)
    journey: JourneyLinkV2
    source_problem: SourceProblemV2
    reasoning_graph: ReasoningGraphV2
    initial_question: QuestionPlanV2
    l3_plans: list[QuestionPlanV2] = Field(min_length=2, max_length=4)
    copy_slots: list[CopySlotV2] = Field(min_length=5, max_length=8)
    l2_plans: list[L2ChoicePlanV2] = Field(min_length=2, max_length=4)
    l0_joint_plan: L0JointPlanV2
    help_plan: HelpPlanV2
    policies: ContentPoliciesV2

    @model_validator(mode="after")
    def validate_pack_contract(self) -> RequiredHomeTeachingPackV2:
        facts = {fact.fact_id: fact for fact in self.reasoning_graph.facts}
        relations = {
            relation.relation_id: relation for relation in self.reasoning_graph.relations
        }
        required_targets = {
            *(('fact', fact_id) for fact_id in self.reasoning_graph.completion.required_fact_ids),
            *(
                ('relation', relation_id)
                for relation_id in self.reasoning_graph.completion.required_relation_ids
            ),
        }

        def target_key(target: TargetRefV2) -> tuple[str, str]:
            if target.target_kind == "fact" and target.target_id not in facts:
                raise ValueError("question target must reference a graph fact")
            if target.target_kind == "relation" and target.target_id not in relations:
                raise ValueError("question target must reference a graph relation")
            return target.target_kind, target.target_id

        initial_target_keys = [target_key(target) for target in self.initial_question.targets]
        initial_targets = set(initial_target_keys)
        if len(initial_target_keys) != len(initial_targets):
            raise ValueError("initial question targets must be unique")
        if initial_targets != required_targets:
            raise ValueError("initial question must ask every completion target")

        all_plan_ids = [
            self.initial_question.plan_id,
            *(plan.plan_id for plan in self.l3_plans),
            *(plan.plan_id for plan in self.l2_plans),
        ]
        if len(all_plan_ids) != len(set(all_plan_ids)):
            raise ValueError("question plan ids must be unique")

        l3_target_keys = [
            target_key(target)
            for plan in self.l3_plans
            for target in plan.targets
        ]
        if any(len(plan.targets) != 1 for plan in self.l3_plans):
            raise ValueError("each L3 plan must ask one semantic target")
        if len(l3_target_keys) != len(set(l3_target_keys)) or set(
            l3_target_keys
        ) != required_targets:
            raise ValueError("L3 needs exactly one plan for every completion target")

        copy_slots = {slot.copy_slot: slot for slot in self.copy_slots}
        if len(copy_slots) != len(self.copy_slots):
            raise ValueError("copy slots must be unique")
        if sum(slot.purpose == "initial_help" for slot in self.copy_slots) != 1:
            raise ValueError("pack needs exactly one initial_help copy slot")
        if sum(slot.purpose == "l0_intro" for slot in self.copy_slots) != 1:
            raise ValueError("pack needs exactly one l0_intro copy slot")
        if sum(slot.purpose == "l0_action" for slot in self.copy_slots) != 1:
            raise ValueError("pack needs exactly one l0_action copy slot")

        copy_target_keys: dict[str, list[tuple[str, str]]] = {}
        for slot in self.copy_slots:
            keys = [target_key(target) for target in slot.targets]
            if len(keys) != len(set(keys)):
                raise ValueError("copy slot targets must be unique")
            copy_target_keys[slot.copy_slot] = keys
            if slot.purpose in {"l0_intro", "l0_action"}:
                if set(keys) != required_targets:
                    raise ValueError(
                        f"{slot.purpose} copy slot must cover every completion target"
                    )
            elif slot.purpose == "initial_help":
                if (
                    len(keys) != 1
                    or keys[0] not in required_targets
                    or slot.targets[0].ask_kind != "answer"
                ):
                    raise ValueError(
                        "initial help must narrow the first request to one answer target"
                    )
            elif len(keys) != 1 or keys[0] not in required_targets:
                raise ValueError("L2 copy slot must reference one completion target")

        l2_copy_slots = [
            slot for slot in self.copy_slots if slot.purpose == "l2_question"
        ]
        l2_copy_targets = [copy_target_keys[slot.copy_slot][0] for slot in l2_copy_slots]
        if len(l2_copy_targets) != len(set(l2_copy_targets)) or set(
            l2_copy_targets
        ) != required_targets:
            raise ValueError("L2 needs exactly one copy slot for every completion target")

        l2_target_keys: list[tuple[str, str]] = []
        for plan in self.l2_plans:
            key = target_key(plan.target)
            l2_target_keys.append(key)
            copy_slot = copy_slots.get(plan.copy_slot)
            if copy_slot is None or copy_slot.purpose != "l2_question":
                raise ValueError("L2 plan must reference an l2_question copy slot")
            if copy_target_keys[plan.copy_slot] != [key]:
                raise ValueError("L2 copy slot target must match its choice plan")
            for choice in plan.choices:
                effect = choice.effect
                if effect.verdict == "correct" and effect.target_kind == "fact":
                    assert effect.interpreted_value is not None
                    if effect.interpreted_value != facts[effect.target_id].value:
                        raise ValueError("correct L2 fact value must equal canonical graph truth")
        if len(l2_target_keys) != len(set(l2_target_keys)) or set(
            l2_target_keys
        ) != required_targets:
            raise ValueError("L2 needs exactly one plan for every completion target")

        intro_slot = copy_slots.get(self.l0_joint_plan.intro_copy_slot)
        action_slot = copy_slots.get(self.l0_joint_plan.action_copy_slot)
        if intro_slot is None or intro_slot.purpose != "l0_intro":
            raise ValueError("L0 plan must reference an l0_intro copy slot")
        if action_slot is None or action_slot.purpose != "l0_action":
            raise ValueError("L0 plan must reference an l0_action copy slot")

        joint_targets: set[tuple[str, str]] = set()
        for completion in self.l0_joint_plan.completion_values:
            key = (completion.target_kind, completion.target_id)
            if key in joint_targets:
                raise ValueError("L0 completion targets must be unique")
            joint_targets.add(key)
            if isinstance(completion, JointFactCompletionV2):
                if completion.target_id not in facts:
                    raise ValueError("L0 fact completion must reference a graph fact")
                if completion.value != facts[completion.target_id].value:
                    raise ValueError("L0 fact completion must use canonical graph truth")
            elif completion.target_id not in relations:
                raise ValueError("L0 relation completion must reference a graph relation")
        if joint_targets != required_targets:
            raise ValueError("L0 completion must cover every required graph target")

        for card in (self.help_plan.H1, self.help_plan.H2, self.help_plan.H3):
            if not set(card.revealed_fact_ids).issubset(facts):
                raise ValueError("help card fact refs must reference graph facts")
            if not set(card.revealed_relation_ids).issubset(relations):
                raise ValueError("help card relation refs must reference graph relations")

        h3_targets = {
            *(('fact', fact_id) for fact_id in self.help_plan.H3.revealed_fact_ids),
            *(
                ('relation', relation_id)
                for relation_id in self.help_plan.H3.revealed_relation_ids
            ),
        }
        if not required_targets.issubset(h3_targets):
            raise ValueError("H3 must reveal every required graph target")
        if set(self.help_plan.H2.revealed_fact_ids) & set(
            self.reasoning_graph.completion.required_fact_ids
        ):
            raise ValueError("H2 cannot reveal a required final fact")
        if set(self.help_plan.H2.revealed_relation_ids) & set(
            self.reasoning_graph.completion.required_relation_ids
        ):
            raise ValueError("H2 cannot reveal a required relation")

        if not set(self.policies.note_relation_ids).issubset(relations):
            raise ValueError("note relation ids must reference graph relations")
        if not set(self.policies.note_relation_ids).issubset(
            self.help_plan.H3.revealed_relation_ids
        ):
            raise ValueError("every note relation must be covered by the H3 joint model")

        final_facts = [fact for fact in facts.values() if fact.role == "final_answer"]
        if len(final_facts) != 1:
            raise ValueError("required home pack needs exactly one final answer fact")
        final_fact = final_facts[0]
        self._validate_source_problem(facts, final_fact)
        self._validate_answer_contract(final_fact)

        self._validate_copy()
        return self

    def _validate_source_problem(
        self,
        facts: dict[str, ContentFactV2],
        final_fact: ContentFactV2,
    ) -> None:
        visible_fact_ids = {
            fact.fact_id for fact in facts.values() if fact.initially_visible
        }
        explicit_fact_ids = {
            binding.fact_id
            for binding in self.source_problem.rendered_facts
            if binding.exposure == "explicit_given"
        }
        if explicit_fact_ids != visible_fact_ids:
            raise ValueError("source explicit facts must equal every initially visible fact")
        visible_numbers = {
            _decimal_value(fact.value)
            for fact in facts.values()
            if fact.initially_visible
            and isinstance(fact.value, MoneyValueV2 | NumberValueV2)
        }
        if not set(_surface_numbers(self.source_problem.prompt)).issubset(visible_numbers):
            raise ValueError("source prompt can mention only initially visible numeric facts")
        counting_outputs = {
            relation.output_fact_id
            for relation in self.reasoning_graph.relations
            if relation.operation == "counting"
        }
        for binding in self.source_problem.rendered_facts:
            if binding.fact_id not in facts:
                raise ValueError("source rendered fact must reference a graph fact")
            if (
                binding.exposure == "perceptual_target"
                and binding.fact_id not in counting_outputs
            ):
                raise ValueError(
                    "perceptual targets are reserved for values encoded by counting visuals"
                )
            rendered_value = _resolve_visual_path(
                self.source_problem.visual,
                binding.source_path,
            )
            _validate_rendered_value(
                facts[binding.fact_id],
                rendered_value,
                source_path=binding.source_path,
            )
            if binding.label_path is not None:
                rendered_label = _resolve_visual_path(
                    self.source_problem.visual,
                    binding.label_path,
                )
                if rendered_label != binding.rendered_label:
                    raise ValueError(
                        f"rendered fact {binding.fact_id} label does not match its role"
                    )

        if self.source_problem.correct not in final_fact.accepted_surface_forms:
            raise ValueError("source correct label must be accepted by final answer fact")
        if isinstance(final_fact.value, MoneyValueV2 | NumberValueV2):
            expected = _decimal_value(final_fact.value)
            correct_numbers = _surface_numbers(self.source_problem.correct)
            if correct_numbers != [expected]:
                raise ValueError("source correct numeric value must equal canonical truth")
            for surface in final_fact.accepted_surface_forms:
                numbers = _surface_numbers(surface)
                if numbers and numbers != [expected]:
                    raise ValueError("numeric answer surface form must equal canonical truth")

            if expected not in _surface_numbers(self.help_plan.H3.body):
                raise ValueError("H3 must state the canonical final numeric value")
        elif not any(
            surface in self.help_plan.H3.body
            for surface in final_fact.accepted_surface_forms
        ):
            raise ValueError("H3 must state an accepted canonical final answer")

        self._validate_help_truth(facts)

    def _validate_answer_contract(self, final_fact: ContentFactV2) -> None:
        answer_plan = next(
            (
                plan
                for plan in self.l2_plans
                if (plan.target.target_kind, plan.target.target_id)
                == ("fact", final_fact.fact_id)
            ),
            None,
        )
        if answer_plan is None:
            raise ValueError("L2 needs a final-answer choice plan")
        if [choice.label for choice in answer_plan.choices] != self.source_problem.answers:
            raise ValueError("source answer options must equal the ordered L2 answer choices")
        correct_choices = [
            choice for choice in answer_plan.choices if choice.effect.verdict == "correct"
        ]
        if len(correct_choices) != 1 or correct_choices[0].label != self.source_problem.correct:
            raise ValueError("source correct label must equal the correct L2 answer choice")

    def _validate_help_truth(self, facts: dict[str, ContentFactV2]) -> None:
        visible_fact_ids = {
            fact.fact_id for fact in facts.values() if fact.initially_visible
        }
        relations = {
            relation.relation_id: relation for relation in self.reasoning_graph.relations
        }
        relation_markers = {
            "counting": re.compile(r"(세(?:면|어|기|는)|가리키|표시)"),
            "comparison": re.compile(r"(비교|더\s*많|짝지)"),
            "addition": re.compile(r"(\+|더하|더해|더하면|합치)"),
            "subtraction": re.compile(r"(-|빼|차이)"),
            "multiplication": re.compile(r"(×|곱|번\s*더)"),
            "division": re.compile(r"(÷|나누|씩)"),
        }

        for card in (self.help_plan.H1, self.help_plan.H2, self.help_plan.H3):
            text = " ".join(part for part in (card.body, card.action) if part)
            allowed_fact_ids = visible_fact_ids | set(card.revealed_fact_ids)
            allowed_numbers = {
                _decimal_value(facts[fact_id].value)
                for fact_id in allowed_fact_ids
                if isinstance(facts[fact_id].value, MoneyValueV2 | NumberValueV2)
            }
            mentioned_numbers = set(_surface_numbers(text))
            if not mentioned_numbers.issubset(allowed_numbers):
                raise ValueError(f"{card.level.value} help copy contains an undeclared value")

            for fact_id in card.revealed_fact_ids:
                value = facts[fact_id].value
                if isinstance(value, MoneyValueV2 | NumberValueV2):
                    if _decimal_value(value) not in mentioned_numbers:
                        raise ValueError(
                            f"{card.level.value} help copy omits a declared revealed fact"
                        )
                elif isinstance(value, ChoiceValueV2):
                    aliases = facts[fact_id].accepted_surface_forms
                    if not any(alias in text for alias in aliases):
                        raise ValueError(
                            f"{card.level.value} help copy omits a declared choice fact"
                        )

            for relation_id in card.revealed_relation_ids:
                operation = relations[relation_id].operation
                if not relation_markers[operation].search(text):
                    raise ValueError(
                        f"{card.level.value} help copy omits its declared relation"
                    )

    def _validate_copy(self) -> None:
        mormi_lines = [
            self.initial_question.reviewed_fallback,
            *(plan.reviewed_fallback for plan in self.l3_plans),
            *(slot.reviewed_fallback for slot in self.copy_slots),
        ]
        if any(_TEACHER_OR_EVALUATION_COPY.search(line) for line in mormi_lines):
            raise ValueError("Mormi copy cannot evaluate or interrogate the child")
        if not self.initial_question.reviewed_fallback.startswith("나 "):
            raise ValueError("initial question must expose Mormi's first-person confusion")
        if not self.initial_question.reviewed_fallback.rstrip().endswith("?"):
            raise ValueError("initial question must end as a help request")

        for slot in self.copy_slots:
            if slot.purpose == "l2_question":
                if "골라서 알려줄 수 있어?" not in slot.reviewed_fallback:
                    raise ValueError("L2 fallback must ask the child to choose and teach Mormi")
                if _JOINT_WORDS.search(slot.reviewed_fallback):
                    raise ValueError("L2 is selection, not joint performance")
            elif slot.purpose in {"l0_intro", "l0_action"}:
                if not _JOINT_WORDS.search(slot.reviewed_fallback):
                    raise ValueError("L0 fallback must make joint performance explicit")

        help_lines = [
            line
            for card in (self.help_plan.H1, self.help_plan.H2, self.help_plan.H3)
            for line in (card.body, card.action)
            if line is not None
        ]
        if any(_MORMI_FIRST_PERSON.search(line) for line in help_lines):
            raise ValueError("help cards are neutral system scaffolds, not Mormi speech")
        if any(_HELP_REQUEST_COPY.search(line) for line in help_lines):
            raise ValueError("help cards must give neutral actions instead of asking the child")

        self._validate_static_question_copy()
        for line in [*mormi_lines, *help_lines]:
            _validate_authored_equations(line)

        validate_child_facing_math_copy(
            [
                self.source_problem.prompt,
                *self.source_problem.answers,
                *mormi_lines,
                *help_lines,
                *(choice.label for plan in self.l2_plans for choice in plan.choices),
            ]
        )

    def _validate_static_question_copy(self) -> None:
        facts = {fact.fact_id: fact for fact in self.reasoning_graph.facts}
        visible_numbers = {
            _decimal_value(fact.value)
            for fact in facts.values()
            if fact.initially_visible
            and isinstance(fact.value, MoneyValueV2 | NumberValueV2)
        }
        context_free_lines = [
            self.initial_question.reviewed_fallback,
            *(plan.reviewed_fallback for plan in self.l3_plans),
            *(
                slot.reviewed_fallback
                for slot in self.copy_slots
                if slot.purpose in {"initial_help", "l2_question"}
            ),
        ]
        for line in context_free_lines:
            if not set(_surface_numbers(line)).issubset(visible_numbers):
                raise ValueError("context-free question copy cannot reveal a hidden value")
            for fact in facts.values():
                if fact.initially_visible:
                    continue
                for surface in fact.accepted_surface_forms:
                    if not _surface_numbers(surface) and surface in line:
                        raise ValueError(
                            "context-free question copy cannot reveal a hidden value"
                        )


class RequiredHomeContentCatalogV2(ContentPackV2Model):
    schema_version: Literal["required-home-content-catalog-v2"] = (
        "required-home-content-catalog-v2"
    )
    catalog_version: int = Field(ge=1)
    packs: list[RequiredHomeTeachingPackV2] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def validate_catalog_scope(self) -> RequiredHomeContentCatalogV2:
        session_ids = [pack.curriculum_session_id for pack in self.packs]
        pack_ids = [pack.pack_id for pack in self.packs]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("catalog curriculum session ids must be unique")
        if len(pack_ids) != len(set(pack_ids)):
            raise ValueError("catalog pack ids must be unique")
        if set(session_ids) != REQUIRED_HOME_SESSION_IDS:
            raise ValueError("catalog must contain the nine required home sessions")
        for pack in self.packs:
            expected_role = (
                "cafe_required"
                if pack.curriculum_session_id in CAFE_REQUIRED_HOME_SESSION_IDS
                else "amusement_preparation"
            )
            if pack.journey.role != expected_role:
                raise ValueError("journey role does not match required session group")
        return self

    def by_session_id(self) -> dict[str, RequiredHomeTeachingPackV2]:
        return {pack.curriculum_session_id: pack for pack in self.packs}


@lru_cache(maxsize=1)
def load_required_home_content_catalog_v2() -> RequiredHomeContentCatalogV2:
    path = Path(__file__).with_name("dialogue_v2_required_home_catalog.json")
    return RequiredHomeContentCatalogV2.model_validate_json(path.read_text(encoding="utf-8"))


def required_home_content_pack_v2(session_id: str) -> RequiredHomeTeachingPackV2:
    try:
        return load_required_home_content_catalog_v2().by_session_id()[session_id]
    except KeyError as exc:
        raise KeyError(f"unknown required home V2 content pack: {session_id}") from exc


def canonical_catalog_json_v2() -> str:
    """Stable serialization used by review tooling and future cache keys."""

    catalog = load_required_home_content_catalog_v2()
    return json.dumps(
        catalog.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
