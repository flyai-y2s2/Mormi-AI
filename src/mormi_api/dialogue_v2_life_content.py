from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .dialogue_v2_content import (
    ContentPackV2Model,
    CopySlotV2,
    QuestionPlanV2,
    RelationRubricV2,
    TargetRefV2,
)
from .schemas import (
    CanonicalValueV2,
    ChoiceValueV2,
    ExpressionLevel,
    HintLevel,
    MoneyValueV2,
    NumberValueV2,
    SceneType,
    VisualContract,
)

LIFE_SCENARIO_PACK_SCHEMA_V2: Literal["life-scenario-pack-v2"] = (
    "life-scenario-pack-v2"
)
LIFE_TASK_PACK_SCHEMA_V2: Literal["life-task-pack-v2"] = "life-task-pack-v2"
LIFE_MATERIALIZER_VERSION_V2: Literal["life-materializer-v2"] = (
    "life-materializer-v2"
)

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_OPAQUE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_life_scenario_json_v2(pack: LifeScenarioPackV2) -> str:
    return _canonical_json(pack.model_dump(mode="json"))


def life_scenario_hash_v2(pack: LifeScenarioPackV2) -> str:
    return hashlib.sha256(
        canonical_life_scenario_json_v2(pack).encode("utf-8")
    ).hexdigest()


def _value_key(value: CanonicalValueV2) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _number(value: CanonicalValueV2) -> Decimal:
    if isinstance(value, MoneyValueV2):
        return Decimal(str(value.amount))
    if isinstance(value, NumberValueV2):
        return Decimal(str(value.value))
    raise ValueError("life arithmetic facts must use typed numbers")


def _dimension(value: CanonicalValueV2) -> tuple[str, str | None]:
    if isinstance(value, MoneyValueV2):
        return "money", value.currency
    if isinstance(value, NumberValueV2):
        return "number", value.unit
    raise ValueError("life arithmetic facts must use typed numbers")


class LifeFactV2(ContentPackV2Model):
    fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    role: Literal[
        "visible_condition",
        "visible_value",
        "intermediate_result",
        "final_answer",
        "selection",
    ]
    value: CanonicalValueV2
    accepted_values: list[CanonicalValueV2] = Field(default_factory=list, max_length=20)
    speaker_label: str = Field(min_length=1, max_length=100)
    initially_visible: bool
    required_for_completion: bool = False
    accepted_surface_forms: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_fact(self) -> LifeFactV2:
        if self.role in {"intermediate_result", "final_answer", "selection"} and (
            self.initially_visible
        ):
            raise ValueError("derived life facts cannot be initially visible")
        values = [self.value, *self.accepted_values]
        if len({_value_key(value) for value in values}) != len(values):
            raise ValueError("life fact accepted values must be unique")
        if len(self.accepted_surface_forms) != len(set(self.accepted_surface_forms)):
            raise ValueError("life fact surface forms must be unique")
        if self.required_for_completion and not self.accepted_surface_forms:
            raise ValueError("required life facts need reviewed surface forms")
        return self

    def accepts_value(self, value: CanonicalValueV2) -> bool:
        key = _value_key(value)
        return key in {_value_key(item) for item in [self.value, *self.accepted_values]}


class LifeRelationV2(ContentPackV2Model):
    relation_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    role: Literal["procedure_step", "explanation", "selection_rule"]
    operation: Literal[
        "counting",
        "comparison",
        "addition",
        "subtraction",
        "multiplication",
        "division",
        "selection",
    ]
    comparison_goal: Literal["maximum", "minimum"] | None = None
    input_fact_ids: list[str] = Field(min_length=1, max_length=20)
    output_fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    evaluation_mode: Literal["exact_semantic", "open_semantic_support"]
    speaker_label: str = Field(min_length=1, max_length=160)
    rubric: RelationRubricV2
    required_for_completion: bool = False


class LifeCompletionContractV2(ContentPackV2Model):
    required_fact_ids: list[str] = Field(min_length=1, max_length=20)
    required_relation_ids: list[str] = Field(default_factory=list, max_length=20)


class LifeReasoningGraphV2(ContentPackV2Model):
    facts: list[LifeFactV2] = Field(min_length=1, max_length=60)
    relations: list[LifeRelationV2] = Field(default_factory=list, max_length=60)
    completion: LifeCompletionContractV2
    open_auxiliary_claims: bool = True

    @model_validator(mode="after")
    def validate_graph(self) -> LifeReasoningGraphV2:
        facts = {fact.fact_id: fact for fact in self.facts}
        relations = {relation.relation_id: relation for relation in self.relations}
        if len(facts) != len(self.facts) or len(relations) != len(self.relations):
            raise ValueError("life graph IDs must be unique")
        if set(self.completion.required_fact_ids) != {
            fact.fact_id for fact in self.facts if fact.required_for_completion
        }:
            raise ValueError("life completion facts must match required facts")
        if set(self.completion.required_relation_ids) != {
            relation.relation_id
            for relation in self.relations
            if relation.required_for_completion
        }:
            raise ValueError("life completion relations must match required relations")
        if len(self.completion.required_fact_ids) != len(
            set(self.completion.required_fact_ids)
        ) or len(self.completion.required_relation_ids) != len(
            set(self.completion.required_relation_ids)
        ):
            raise ValueError("life completion targets must be unique")
        for relation in self.relations:
            if not set(relation.input_fact_ids).issubset(facts):
                raise ValueError("life relation inputs must reference graph facts")
            if relation.output_fact_id not in facts:
                raise ValueError("life relation output must reference a graph fact")
            if relation.output_fact_id in relation.input_fact_ids:
                raise ValueError("life relation cannot consume its own output")
            self._validate_relation(relation, facts)
        return self

    @staticmethod
    def _validate_relation(
        relation: LifeRelationV2,
        facts: dict[str, LifeFactV2],
    ) -> None:
        inputs = [facts[fact_id].value for fact_id in relation.input_fact_ids]
        output = facts[relation.output_fact_id].value
        if len(inputs) != len(relation.input_fact_ids):
            raise ValueError("life relation inputs must be unique")
        if relation.operation in {"counting", "selection"}:
            if relation.comparison_goal is not None:
                raise ValueError("comparison_goal belongs only to comparison relations")
            return
        values = [_number(value) for value in inputs]
        if relation.operation == "comparison":
            if len(inputs) != 2 or not isinstance(output, ChoiceValueV2):
                raise ValueError("life comparison needs two values and a choice output")
            if _dimension(inputs[0]) != _dimension(inputs[1]):
                raise ValueError("life comparison inputs must share a dimension")
            if relation.comparison_goal is None:
                raise ValueError("life comparison must declare maximum or minimum")
            if values[0] == values[1]:
                expected = "same"
            elif relation.comparison_goal == "maximum":
                expected = "left" if values[0] > values[1] else "right"
            else:
                expected = "left" if values[0] < values[1] else "right"
            if output.choice_id != expected:
                raise ValueError("life comparison output contradicts reviewed facts")
            return
        output_number = _number(output)
        if relation.operation in {"addition", "subtraction"}:
            if relation.operation == "subtraction" and len(values) != 2:
                raise ValueError("life subtraction must be binary")
            if len({_dimension(value) for value in [*inputs, output]}) != 1:
                raise ValueError("life addition/subtraction dimensions must match")
            expected_number = (
                sum(values, start=Decimal(0))
                if relation.operation == "addition"
                else values[0] - values[1]
            )
        elif relation.operation == "multiplication":
            if len(values) != 2:
                raise ValueError("life multiplication must be binary")
            expected_number = values[0] * values[1]
        else:
            if len(values) != 2 or values[1] == 0:
                raise ValueError("life division needs two values and a nonzero divisor")
            expected_number = values[0] / values[1]
        if output_number != expected_number:
            raise ValueError("life relation arithmetic contradicts reviewed facts")


class LifeFactUpdateV2(ContentPackV2Model):
    fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    value: CanonicalValueV2


class LifeChoiceEffectV2(ContentPackV2Model):
    verdict: Literal["correct", "incorrect"]
    fact_updates: list[LifeFactUpdateV2] = Field(default_factory=list, max_length=8)
    relation_ids: list[str] = Field(default_factory=list, max_length=8)
    misconception_code: str | None = Field(default=None, pattern=_ID_PATTERN, max_length=120)
    visual_patch: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_effect(self) -> LifeChoiceEffectV2:
        if self.verdict == "correct":
            if not self.fact_updates and not self.relation_ids:
                raise ValueError("correct life choice must apply reviewed progress")
            if self.misconception_code is not None:
                raise ValueError("correct life choice cannot declare a misconception")
        elif self.misconception_code is None:
            raise ValueError("incorrect life choice needs a misconception code")
        if len({item.fact_id for item in self.fact_updates}) != len(self.fact_updates):
            raise ValueError("life choice fact updates must be unique")
        if len(self.relation_ids) != len(set(self.relation_ids)):
            raise ValueError("life choice relations must be unique")
        return self


class LifeChoiceV2(ContentPackV2Model):
    choice_id: str = Field(pattern=_OPAQUE_ID_PATTERN, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    image_url: str | None = Field(default=None, max_length=500)
    disabled: bool = False
    effect: LifeChoiceEffectV2


class LifeL2ChoicePlanV2(ContentPackV2Model):
    plan_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    target: TargetRefV2
    copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    choices: list[LifeChoiceV2] = Field(min_length=2, max_length=20)
    input_config: dict[str, Any] = Field(default_factory=dict)
    submit_label: str = Field(default="알려주기", min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_choices(self) -> LifeL2ChoicePlanV2:
        ids = [choice.choice_id for choice in self.choices]
        if len(ids) != len(set(ids)):
            raise ValueError("life choice IDs must be unique")
        if not any(
            choice.effect.verdict == "correct" and not choice.disabled
            for choice in self.choices
        ):
            raise ValueError("life L2 plan needs one enabled correct choice")
        return self


class LifeJointFactCompletionV2(ContentPackV2Model):
    target_kind: Literal["fact"] = "fact"
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    value: CanonicalValueV2


class LifeJointRelationCompletionV2(ContentPackV2Model):
    target_kind: Literal["relation"] = "relation"
    target_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    satisfied: Literal[True] = True


LifeJointCompletionV2 = Annotated[
    LifeJointFactCompletionV2 | LifeJointRelationCompletionV2,
    Field(discriminator="target_kind"),
]


class LifeL0JointPlanV2(ContentPackV2Model):
    action_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    intro_copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    action_copy_slot: str = Field(pattern=_ID_PATTERN, max_length=160)
    completion_values: list[LifeJointCompletionV2] = Field(min_length=1, max_length=20)
    button_label: str = Field(min_length=1, max_length=30)
    input_config: dict[str, Any] = Field(default_factory=dict)


class LifeHelpCardV2(ContentPackV2Model):
    level: HintLevel
    support_type: Literal["attention", "guided_action", "joint_model"]
    answer_policy: Literal["hidden", "partial", "revealed"]
    body: str = Field(min_length=1, max_length=240)
    action: str | None = Field(default=None, min_length=1, max_length=160)
    visual_type: str | None = Field(default=None, min_length=1, max_length=100)
    visual_data: dict[str, Any] = Field(default_factory=dict)
    revealed_fact_ids: list[str] = Field(default_factory=list, max_length=20)
    revealed_relation_ids: list[str] = Field(default_factory=list, max_length=20)


class LifeHelpPlanV2(ContentPackV2Model):
    H1: LifeHelpCardV2
    H2: LifeHelpCardV2
    H3: LifeHelpCardV2

    @model_validator(mode="after")
    def validate_help(self) -> LifeHelpPlanV2:
        expected = {
            HintLevel.H1: (self.H1, "attention", "hidden"),
            HintLevel.H2: (self.H2, "guided_action", "partial"),
            HintLevel.H3: (self.H3, "joint_model", "revealed"),
        }
        for level, (card, support, policy) in expected.items():
            if (
                card.level is not level
                or card.support_type != support
                or card.answer_policy != policy
            ):
                raise ValueError("life help card profile does not match its level")
        if self.H1.revealed_fact_ids or self.H1.revealed_relation_ids:
            raise ValueError("life H1 cannot reveal derived truth")
        return self


class LifeTaskPoliciesV2(ContentPackV2Model):
    entry_expression_level: Literal[ExpressionLevel.L4, ExpressionLevel.L2]
    entry_hint_level: Literal[HintLevel.H0] = HintLevel.H0
    note_policy: Literal["none", "verified_child_or_coauthored"]
    note_relation_ids: list[str] = Field(default_factory=list, max_length=10)
    note_skill_id: str | None = Field(default=None, pattern=_ID_PATTERN, max_length=120)
    note_context: str | None = Field(default=None, min_length=1, max_length=200)
    reviewed_direct_fallback: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
    )
    reviewed_coauthored_note: str | None = Field(default=None, min_length=1, max_length=240)
    transition_text: str | None = Field(default=None, min_length=1, max_length=220)
    stable_copy_mode: Literal["reviewed_template_only"] = "reviewed_template_only"

    @model_validator(mode="after")
    def validate_note_policy(self) -> LifeTaskPoliciesV2:
        note_values = (
            self.note_relation_ids,
            self.note_skill_id,
            self.note_context,
            self.reviewed_direct_fallback,
            self.reviewed_coauthored_note,
        )
        if self.note_policy == "none" and any(note_values):
            raise ValueError("note-disabled life task cannot carry note material")
        if self.note_policy != "none" and (
            not self.note_relation_ids
            or self.note_skill_id is None
            or self.note_context is None
            or self.reviewed_direct_fallback is None
            or self.reviewed_coauthored_note is None
        ):
            raise ValueError("note-enabled life task needs relation, skill and reviewed copy")
        return self


class LifeTaskPackV2(ContentPackV2Model):
    schema_version: Literal["life-task-pack-v2"] = LIFE_TASK_PACK_SCHEMA_V2
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    scene: Literal[SceneType.CAFE, SceneType.AMUSEMENT_PARK]
    scenario_id: str = Field(pattern=_ID_PATTERN, max_length=100)
    task_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    stage_id: str = Field(pattern=_ID_PATTERN, max_length=100)
    phase: Literal["single", "selection", "calculation", "primary", "transfer"]
    skill_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    title: str = Field(min_length=1, max_length=120)
    dictionary_card_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    source_prompt: str = Field(min_length=1, max_length=240)
    base_visual: VisualContract
    reasoning_graph: LifeReasoningGraphV2
    initial_question: QuestionPlanV2
    l3_plans: list[QuestionPlanV2] = Field(min_length=1, max_length=8)
    copy_slots: list[CopySlotV2] = Field(min_length=4, max_length=12)
    l2_plans: list[LifeL2ChoicePlanV2] = Field(min_length=1, max_length=8)
    l0_joint_plan: LifeL0JointPlanV2
    help_plan: LifeHelpPlanV2
    policies: LifeTaskPoliciesV2

    @model_validator(mode="after")
    def validate_task(self) -> LifeTaskPackV2:
        facts = {fact.fact_id: fact for fact in self.reasoning_graph.facts}
        relations = {
            relation.relation_id: relation for relation in self.reasoning_graph.relations
        }
        required = {
            *(("fact", fact_id) for fact_id in self.reasoning_graph.completion.required_fact_ids),
            *(
                ("relation", relation_id)
                for relation_id in self.reasoning_graph.completion.required_relation_ids
            ),
        }

        def key(target: TargetRefV2) -> tuple[str, str]:
            if target.target_kind == "fact" and target.target_id not in facts:
                raise ValueError("life question target must reference a fact")
            if target.target_kind == "relation" and target.target_id not in relations:
                raise ValueError("life question target must reference a relation")
            return target.target_kind, target.target_id

        if {key(target) for target in self.initial_question.targets} != required:
            raise ValueError("life initial question must cover required targets")
        l3_targets = [key(plan.targets[0]) for plan in self.l3_plans]
        if any(len(plan.targets) != 1 for plan in self.l3_plans) or (
            len(l3_targets) != len(set(l3_targets)) or set(l3_targets) != required
        ):
            raise ValueError("life L3 needs one plan per required target")
        slots = {slot.copy_slot: slot for slot in self.copy_slots}
        if len(slots) != len(self.copy_slots):
            raise ValueError("life copy slots must be unique")
        purposes = [slot.purpose for slot in self.copy_slots]
        if purposes.count("initial_help") != 1 or purposes.count("l0_intro") != 1 or (
            purposes.count("l0_action") != 1
        ):
            raise ValueError("life task needs one initial-help and two L0 copy slots")
        l2_targets: list[tuple[str, str]] = []
        for plan in self.l2_plans:
            target_key = key(plan.target)
            l2_targets.append(target_key)
            slot = slots.get(plan.copy_slot)
            if slot is None or slot.purpose != "l2_question" or (
                len(slot.targets) != 1 or key(slot.targets[0]) != target_key
            ):
                raise ValueError("life L2 plan and copy slot must share one target")
            for choice in plan.choices:
                for update in choice.effect.fact_updates:
                    fact = facts.get(update.fact_id)
                    if fact is None or not fact.accepts_value(update.value):
                        raise ValueError("life choice fact update is outside reviewed truth")
                if not set(choice.effect.relation_ids).issubset(relations):
                    raise ValueError("life choice relation update is unknown")
                if choice.effect.verdict == "correct":
                    if target_key[0] == "fact" and not any(
                        update.fact_id == target_key[1]
                        for update in choice.effect.fact_updates
                    ):
                        raise ValueError("correct life choice must update its fact target")
                    if target_key[0] == "relation" and target_key[1] not in (
                        choice.effect.relation_ids
                    ):
                        raise ValueError("correct life choice must update its relation target")
        if len(l2_targets) != len(set(l2_targets)) or set(l2_targets) != required:
            raise ValueError("life L2 needs one plan per required target")
        joint_targets: set[tuple[str, str]] = set()
        for completion in self.l0_joint_plan.completion_values:
            target_key = (completion.target_kind, completion.target_id)
            joint_targets.add(target_key)
            if isinstance(completion, LifeJointFactCompletionV2):
                fact = facts.get(completion.target_id)
                if fact is None or not fact.accepts_value(completion.value):
                    raise ValueError("life L0 fact value is outside reviewed truth")
            elif completion.target_id not in relations:
                raise ValueError("life L0 relation target is unknown")
        if joint_targets != required or len(joint_targets) != len(
            self.l0_joint_plan.completion_values
        ):
            raise ValueError("life L0 completion must cover required targets exactly")
        if not set(self.policies.note_relation_ids).issubset(relations):
            raise ValueError("life note relations must reference the task graph")
        for card in (self.help_plan.H1, self.help_plan.H2, self.help_plan.H3):
            if not set(card.revealed_fact_ids).issubset(facts) or not set(
                card.revealed_relation_ids
            ).issubset(relations):
                raise ValueError("life help card contains an unknown graph reference")
        h3_targets = {
            *(("fact", fact_id) for fact_id in self.help_plan.H3.revealed_fact_ids),
            *(("relation", relation_id) for relation_id in self.help_plan.H3.revealed_relation_ids),
        }
        if not required.issubset(h3_targets):
            raise ValueError("life H3 must reveal all required targets")
        return self


class LifePriorFactVariantSelectorV2(ContentPackV2Model):
    source_task_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    fact_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    value_to_variant_id: dict[str, str] = Field(min_length=1, max_length=20)


class LifeTaskStageV2(ContentPackV2Model):
    task_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    default_variant_id: str = Field(pattern=_OPAQUE_ID_PATTERN, max_length=100)
    variants: dict[str, LifeTaskPackV2] = Field(min_length=1, max_length=20)
    selector: LifePriorFactVariantSelectorV2 | None = None

    @model_validator(mode="after")
    def validate_variants(self) -> LifeTaskStageV2:
        if self.default_variant_id not in self.variants:
            raise ValueError("life task default variant is missing")
        if any(pack.task_id != self.task_id for pack in self.variants.values()):
            raise ValueError("life task variants must share their task ID")
        if self.selector is not None and not set(
            self.selector.value_to_variant_id.values()
        ).issubset(self.variants):
            raise ValueError("life task selector references an unknown variant")
        return self


class LifeCompletionProjectionV2(ContentPackV2Model):
    output_key: str = Field(pattern=_ID_PATTERN, max_length=100)
    source_task_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    source_kind: Literal["fact", "relation_constant"]
    source_id: str = Field(pattern=_ID_PATTERN, max_length=120)
    relation_value: str | int | float | bool | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> LifeCompletionProjectionV2:
        if (self.source_kind == "relation_constant") != (
            self.relation_value is not None
        ):
            raise ValueError("relation projection must declare its reviewed value")
        return self


class LifeScenarioPackV2(ContentPackV2Model):
    schema_version: Literal["life-scenario-pack-v2"] = LIFE_SCENARIO_PACK_SCHEMA_V2
    materializer_version: Literal["life-materializer-v2"] = LIFE_MATERIALIZER_VERSION_V2
    pack_id: str = Field(pattern=_ID_PATTERN, max_length=160)
    content_version: int = Field(ge=1)
    scene: Literal[SceneType.CAFE, SceneType.AMUSEMENT_PARK]
    scenario_id: str = Field(pattern=_ID_PATTERN, max_length=100)
    task_stages: list[LifeTaskStageV2] = Field(min_length=1, max_length=4)
    completion_projection: list[LifeCompletionProjectionV2] = Field(
        min_length=1,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_scenario(self) -> LifeScenarioPackV2:
        task_ids = [stage.task_id for stage in self.task_stages]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("life scenario task IDs must be unique")
        if any(
            pack.scene is not self.scene or pack.scenario_id != self.scenario_id
            for stage in self.task_stages
            for pack in stage.variants.values()
        ):
            raise ValueError("life task variants must match their scenario")
        stages = {stage.task_id: stage for stage in self.task_stages}
        for index, stage in enumerate(self.task_stages):
            if stage.selector is None:
                continue
            if stage.selector.source_task_id not in task_ids[:index]:
                raise ValueError("life variant selector must use an earlier task")
            source_stage = stages[stage.selector.source_task_id]
            for source_pack in source_stage.variants.values():
                facts = {
                    fact.fact_id: fact for fact in source_pack.reasoning_graph.facts
                }
                if stage.selector.fact_id not in facts:
                    raise ValueError("life selector source fact is missing")
        output_keys = [projection.output_key for projection in self.completion_projection]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("life completion projection keys must be unique")
        for projection in self.completion_projection:
            projection_stage = stages.get(projection.source_task_id)
            if projection_stage is None:
                raise ValueError("life completion projection source task is missing")
            for pack in projection_stage.variants.values():
                if projection.source_kind == "fact" and projection.source_id not in {
                    fact.fact_id for fact in pack.reasoning_graph.facts
                }:
                    raise ValueError("life completion projection fact is missing")
                if projection.source_kind == "relation_constant" and (
                    projection.source_id
                    not in {
                        relation.relation_id
                        for relation in pack.reasoning_graph.relations
                    }
                ):
                    raise ValueError("life completion projection relation is missing")
        return self

    def stage_by_task_id(self, task_id: str) -> LifeTaskStageV2:
        return next(stage for stage in self.task_stages if stage.task_id == task_id)
