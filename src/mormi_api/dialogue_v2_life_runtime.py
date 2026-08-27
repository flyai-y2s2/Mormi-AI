# mypy: disable-error-code="override"

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, cast

from .dialogue_v2_ledger import (
    PinnedLifeTaskSnapshotV2,
    ReasoningLedgerApplyResultV2,
    ReasoningLedgerV2,
    RelationVerificationEvidenceV2,
    apply_structured_progress_v2,
    empty_reasoning_ledger_v2,
    pin_life_task_pack_v2,
    reasoning_completion_v2,
)
from .dialogue_v2_life_content import (
    LifeL2ChoicePlanV2,
    LifeScenarioPackV2,
    LifeTaskPackV2,
)
from .dialogue_v2_runtime import (
    DialogueV2Engine,
    StableCopyResolution,
    _primitive,
    _QuestionContract,
    _target_key,
    _TurnSemantics,
)
from .dialogue_v2_scenario_snapshot import (
    pin_life_scenario_runtime_v3,
    resolve_life_scenario_runtime_v3,
)
from .dialogue_v2_speaker import StableCopyPlanV2, bridge_excerpt_violation_v2
from .dictionary_models import dictionary_reference
from .engine import EngineTurnResult
from .llm import validate_note_contextualization
from .schemas import (
    CanonicalValueV2,
    ChildResponse,
    ChoiceOption,
    CompletionContract,
    CompletionOutcome,
    DialogueHistoryTurn,
    DialogueRuntimeContractVersion,
    ExpressionLevel,
    HelpCardContract,
    HintLevel,
    InputContract,
    InputKind,
    MormiContract,
    NoteAttribution,
    NoteContextualizationContext,
    NoteEvidence,
    NoteUpdate,
    PedagogySnapshot,
    PinnedDialogueScenarioRuntimeV3,
    PinnedDialogueTaskNoteStateV3,
    SessionState,
    SessionStatus,
    TaskAnchorContract,
    TurnContract,
    UnderstandingResponseV2,
    VisualContract,
    new_id,
)


def _value_lookup_key(value: CanonicalValueV2) -> str:
    primitive = _primitive(value)
    if isinstance(primitive, bool):
        return "true" if primitive else "false"
    return str(primitive)


def _merge_visual_data(
    visual: VisualContract,
    patch: Mapping[str, object] | None,
) -> VisualContract:
    if not patch:
        return visual.model_copy(deep=True)
    return visual.model_copy(
        update={"data": {**visual.data, **dict(patch)}},
        deep=True,
    )


class DialogueV2LifeEngine(DialogueV2Engine):
    """Multi-task V2 runtime for reviewed cafe and amusement scenario packs.

    The semantic understanding, literal evidence guard, ladder policy and
    Haiku speaker are inherited from the home engine. This class replaces only
    the content snapshot, structured-choice, presentation, transition and
    completion boundaries that are genuinely scene-specific.
    """

    async def initialize_scenario_state(
        self,
        state: SessionState,
        scenario_pack: LifeScenarioPackV2,
        *,
        selector_reason: str,
        canary_bucket: int | None,
    ) -> TurnContract:
        if state.runtime_contract_version is not DialogueRuntimeContractVersion.VERDICT_V1:
            raise ValueError("life V2 state must be pinned to verdict-v1")
        if state.scene is not scenario_pack.scene or state.scenario_id != (
            scenario_pack.scenario_id
        ):
            raise ValueError("materialized life scenario does not match session identity")
        expected_task_ids = [stage.task_id for stage in scenario_pack.task_stages]
        if state.task_ids != expected_task_ids:
            raise ValueError("materialized life scenario task order does not match session")

        active_variants = {
            stage.task_id: stage.default_variant_id
            for stage in scenario_pack.task_stages
        }
        ledgers: dict[str, dict[str, Any]] = {}
        note_states: dict[str, PinnedDialogueTaskNoteStateV3] = {}
        for stage in scenario_pack.task_stages:
            pack = stage.variants[stage.default_variant_id]
            snapshot = pin_life_task_pack_v2(pack)
            ledgers[stage.task_id] = empty_reasoning_ledger_v2(snapshot).model_dump(
                mode="json"
            )
            note_states[stage.task_id] = PinnedDialogueTaskNoteStateV3()

        state.pinned_dialogue_v2 = None
        state.pinned_dialogue_scenario_v3 = pin_life_scenario_runtime_v3(
            scenario_pack,
            active_variant_ids=active_variants,
            reasoning_ledgers=ledgers,
            task_note_states=note_states,
            selector_reason=selector_reason,
            canary_bucket=canary_bucket,
        )
        state.task_index = 0
        pack, _, ledger, _ = self._resolve_state(state)
        state.expression_level = pack.policies.entry_expression_level
        state.task_start_level = pack.policies.entry_expression_level
        state.hint_level = HintLevel.H0
        state.task_max_hint = HintLevel.H0
        state.subgoal_id = pack.initial_question.plan_id
        state.verified_slots = {}
        state.completed_task_slots = {}
        state.status = SessionStatus.ACTIVE
        initial_question = self._question_contract(state, pack, ledger)
        turn = self._build_turn(
            state,
            pack,
            ledger,
            text=initial_question.fallback,
            mood="curious",
        )
        state.current_turn_id = turn.turn_id
        return turn

    def _resolve_scenario(
        self,
        state: SessionState,
    ) -> tuple[LifeScenarioPackV2, PinnedDialogueScenarioRuntimeV3]:
        pinned = state.pinned_dialogue_scenario_v3
        if pinned is None:
            raise ValueError("verdict-v1 life conversation has no scenario snapshot")
        scenario = resolve_life_scenario_runtime_v3(pinned)
        if (
            scenario.scene is not state.scene
            or scenario.scenario_id != state.scenario_id
        ):
            raise ValueError("pinned life scenario failed identity validation")
        if [stage.task_id for stage in scenario.task_stages] != state.task_ids:
            raise ValueError("pinned life scenario task order changed")
        return scenario, pinned

    def _resolve_state(
        self,
        state: SessionState,
    ) -> tuple[
        LifeTaskPackV2,
        PinnedLifeTaskSnapshotV2,
        ReasoningLedgerV2,
        dict[str, StableCopyPlanV2],
    ]:
        if state.runtime_contract_version is not DialogueRuntimeContractVersion.VERDICT_V1:
            raise ValueError("legacy conversation cannot enter life V2 engine")
        scenario, pinned = self._resolve_scenario(state)
        task_id = state.current_task_id
        stage = scenario.stage_by_task_id(task_id)
        try:
            variant_id = pinned.active_variant_ids[task_id]
            pack = stage.variants[variant_id]
            ledger = ReasoningLedgerV2.model_validate(pinned.reasoning_ledgers[task_id])
        except KeyError as error:
            raise ValueError("pinned life task runtime is incomplete") from error
        snapshot = pin_life_task_pack_v2(pack)
        reasoning_completion_v2(snapshot, ledger)
        return pack, snapshot, ledger, {}

    def _store_ledger(self, state: SessionState, ledger: ReasoningLedgerV2) -> None:
        pinned = state.pinned_dialogue_scenario_v3
        if pinned is None:
            raise ValueError("missing life scenario runtime")
        ledgers = dict(pinned.reasoning_ledgers)
        ledgers[state.current_task_id] = ledger.model_dump(mode="json")
        state.pinned_dialogue_scenario_v3 = pinned.model_copy(
            update={"reasoning_ledgers": ledgers},
            deep=True,
        )

    def _snapshot_from_pack(self, pack: LifeTaskPackV2) -> PinnedLifeTaskSnapshotV2:
        return pin_life_task_pack_v2(pack)

    def _targets_from_active_state(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        ledger: ReasoningLedgerV2,
    ) -> list[Any]:
        completion = reasoning_completion_v2(pin_life_task_pack_v2(pack), ledger)
        remaining = {
            *(("fact", item) for item in completion.remaining_fact_ids),
            *(("relation", item) for item in completion.remaining_relation_ids),
        }
        ordered = [
            target
            for target in pack.initial_question.targets
            if (target.target_kind, target.target_id) in remaining
        ]
        if state.expression_level in {ExpressionLevel.L4, ExpressionLevel.L0}:
            return ordered
        plans = pack.l3_plans if state.expression_level is ExpressionLevel.L3 else []
        for l3_plan in plans:
            if l3_plan.targets[0] in ordered:
                return [l3_plan.targets[0]]
        for l2_plan in pack.l2_plans:
            if l2_plan.target in ordered:
                return [l2_plan.target]
        return ordered[:1]

    def _active_l2_plan(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        ledger: ReasoningLedgerV2,
    ) -> LifeL2ChoicePlanV2:
        plan = next(
            (item for item in pack.l2_plans if item.plan_id == state.subgoal_id),
            None,
        )
        if plan is not None:
            return plan
        return self._l2_plan_for_targets(
            pack,
            self._targets_from_active_state(state, pack, ledger),
        )

    @staticmethod
    def _l2_plan_for_targets(
        pack: LifeTaskPackV2,
        targets: list[Any],
    ) -> LifeL2ChoicePlanV2:
        target_keys = {
            (target.target_kind, target.target_id) for target in targets
        }
        try:
            return next(
                plan
                for plan in pack.l2_plans
                if (plan.target.target_kind, plan.target.target_id) in target_keys
            )
        except StopIteration as error:
            raise ValueError("life pack has no L2 plan for the remaining target") from error

    def _question_contract(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        ledger: ReasoningLedgerV2,
    ) -> _QuestionContract:
        if state.status is SessionStatus.COMPLETED:
            return _QuestionContract([], "", InputContract(kind=InputKind.NONE))
        targets = self._targets_from_active_state(state, pack, ledger)
        target_keys = [_target_key(target) for target in targets]
        if state.expression_level is ExpressionLevel.L0:
            state.hint_level = HintLevel.H3
            intro = next(slot for slot in pack.copy_slots if slot.purpose == "l0_intro")
            action = next(slot for slot in pack.copy_slots if slot.purpose == "l0_action")
            completion_values = {
                f"{completion.target_kind}:{completion.target_id}": (
                    _primitive(completion.value)
                    if completion.target_kind == "fact"
                    else True
                )
                for completion in pack.l0_joint_plan.completion_values
            }
            state.subgoal_id = "l0.joint"
            return _QuestionContract(
                targets=targets,
                fallback=intro.reviewed_fallback,
                input=InputContract(
                    kind=InputKind.JOINT,
                    target_slots=target_keys,
                    submit_label=pack.l0_joint_plan.button_label,
                    config={
                        **pack.l0_joint_plan.input_config,
                        "text": action.reviewed_fallback,
                        "action_id": pack.l0_joint_plan.action_id,
                        "completion_values": completion_values,
                    },
                ),
                copy_slot=intro,
            )
        if state.expression_level is ExpressionLevel.L2:
            l2_plan = self._l2_plan_for_targets(pack, targets)
            slot = next(
                item for item in pack.copy_slots if item.copy_slot == l2_plan.copy_slot
            )
            state.subgoal_id = l2_plan.plan_id
            return _QuestionContract(
                targets=[l2_plan.target],
                fallback=slot.reviewed_fallback,
                input=InputContract(
                    kind=InputKind.CHOICES,
                    choices=[
                        ChoiceOption(
                            id=choice.choice_id,
                            label=choice.label,
                            image_url=choice.image_url,
                            disabled=choice.disabled,
                        )
                        for choice in l2_plan.choices
                    ],
                    target_slots=[_target_key(l2_plan.target)],
                    submit_label=l2_plan.submit_label,
                    config={"plan_id": l2_plan.plan_id, **l2_plan.input_config},
                ),
                copy_slot=slot,
            )
        if state.expression_level is ExpressionLevel.L3:
            l3_plan = next(
                candidate
                for candidate in pack.l3_plans
                if _target_key(candidate.targets[0]) in target_keys
            )
            state.subgoal_id = l3_plan.plan_id
            return _QuestionContract(
                targets=list(l3_plan.targets),
                fallback=l3_plan.reviewed_fallback,
                input=InputContract(
                    kind=InputKind.TEXT,
                    target_slots=[_target_key(l3_plan.targets[0])],
                    placeholder="짧게 알려줘",
                    submit_label="알려주기",
                ),
            )
        state.subgoal_id = (
            pack.initial_question.plan_id
            if len(targets) > 1
            else next(
                plan.plan_id for plan in pack.l3_plans if plan.targets[0] == targets[0]
            )
        )
        fallback = (
            pack.initial_question.reviewed_fallback
            if len(targets) > 1
            else next(
                plan.reviewed_fallback
                for plan in pack.l3_plans
                if plan.targets[0] == targets[0]
            )
        )
        return _QuestionContract(
            targets=targets,
            fallback=fallback,
            input=InputContract(
                kind=InputKind.TEXT,
                target_slots=target_keys,
                placeholder=(
                    "답과 방법을 네 말로 알려줘"
                    if len(targets) > 1
                    else "짧게 알려줘"
                ),
                submit_label="알려주기",
            ),
        )

    def _choice_semantics(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        snapshot: PinnedLifeTaskSnapshotV2,
        ledger: ReasoningLedgerV2,
        response: ChildResponse,
    ) -> _TurnSemantics:
        if state.expression_level is not ExpressionLevel.L2:
            raise ValueError("life choice response is not valid outside L2")
        plan = self._active_l2_plan(state, pack, ledger)
        if len(response.choice_ids) != 1:
            raise ValueError("life L2 response requires exactly one choice")
        choice = next(
            (item for item in plan.choices if item.choice_id == response.choice_ids[0]),
            None,
        )
        if choice is None or choice.disabled:
            raise ValueError("choice_id is unavailable in the pinned life plan")
        effect = choice.effect
        correct = effect.verdict == "correct"
        result = (
            apply_structured_progress_v2(
                snapshot,
                ledger,
                fact_values={
                    update.fact_id: update.value for update in effect.fact_updates
                },
                relation_ids=list(effect.relation_ids),
                source_turn_id=response.turn_id,
                source_kind="choice",
            )
            if correct
            else self._unchanged_result(snapshot, ledger)
        )
        target_is_fact = plan.target.target_kind == "fact"
        semantic = UnderstandingResponseV2.model_validate(
            {
                "utterance_class": "learning_response",
                "contains_learning_evidence": correct,
                "answer_status": (
                    "complete"
                    if correct and target_is_fact
                    else "incorrect"
                    if not correct and target_is_fact
                    else "not_applicable"
                ),
                "reasoning_status": (
                    "sufficient"
                    if correct and not target_is_fact
                    else "incorrect"
                    if not correct and not target_is_fact
                    else "not_applicable"
                ),
            }
        )
        return _TurnSemantics(
            response=semantic,
            apply_result=result,
            guarded=None,
            route="main",
            understanding_source="structured_choice",
            structured_correct=correct,
            visual_patch=dict(effect.visual_patch) or None,
        )

    @staticmethod
    def _unchanged_result(
        snapshot: PinnedLifeTaskSnapshotV2,
        ledger: ReasoningLedgerV2,
    ) -> ReasoningLedgerApplyResultV2:
        return ReasoningLedgerApplyResultV2(
            ledger=ledger,
            new_fact_ids=[],
            new_relation_ids=[],
            new_milestone_fact_ids=[],
            new_fact_evidence_ids=[],
            new_relation_evidence_ids=[],
            new_auxiliary_evidence_ids=[],
            ignored_claim_ids=[],
            completion=reasoning_completion_v2(snapshot, ledger),
            completion_became_true=False,
        )

    def _apply_ladder_policy(
        self,
        state: SessionState,
        semantics: _TurnSemantics,
    ) -> None:
        super()._apply_ladder_policy(state, semantics)
        if semantics.visual_patch is not None:
            patches = dict(state.scenario_data.get("v2_life_visual_patches", {}))
            patches[state.current_task_id] = dict(semantics.visual_patch)
            state.scenario_data["v2_life_visual_patches"] = patches

    def _sync_legacy_slots(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        ledger: ReasoningLedgerV2,
    ) -> None:
        relations = {
            relation.relation_id: relation for relation in pack.reasoning_graph.relations
        }
        compatible: dict[str, str | int | float | bool] = {
            fact_id: _primitive(entry.canonical_value)
            for fact_id, entry in ledger.verified_facts.items()
        }
        compatible.update(
            {
                relation_id: relations[relation_id].speaker_label
                for relation_id in ledger.verified_relations
            }
        )
        state.verified_slots = compatible

    def _track_note_provenance(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        semantics: _TurnSemantics,
        *,
        before_hint: HintLevel,
    ) -> None:
        if pack.policies.note_policy == "none":
            return
        pinned = state.pinned_dialogue_scenario_v3
        if pinned is None:
            raise ValueError("missing life scenario note state")
        note_state = pinned.task_note_states[pack.task_id]
        independent = dict(note_state.independent_relation_evidence)
        supported = set(note_state.supported_relation_ids)
        direct_turn = (
            semantics.understanding_source == "sonnet_low"
            and before_hint in {HintLevel.H0, HintLevel.H1}
        )
        new_evidence = set(semantics.apply_result.new_relation_evidence_ids)
        for relation_id in pack.policies.note_relation_ids:
            entry = semantics.apply_result.ledger.verified_relations.get(relation_id)
            turn_evidence = (
                [
                    evidence.evidence_id
                    for evidence in entry.evidence
                    if evidence.evidence_id in new_evidence
                ]
                if entry is not None
                else []
            )
            if not turn_evidence:
                continue
            if direct_turn and relation_id not in supported:
                independent[relation_id] = list(dict.fromkeys(turn_evidence))
            elif relation_id not in independent:
                supported.add(relation_id)
        # Help-card content never verifies a relation or teaches Mormi by
        # itself.  Once H2/H3 has been visible, though, a later verified method
        # was learned with reviewed support and its star-note attribution must
        # remain coauthored.
        note_relation_ids = set(pack.policies.note_relation_ids)
        if before_hint in {
            HintLevel.H2,
            HintLevel.H3,
        } or state.hint_level in {
            HintLevel.H2,
            HintLevel.H3,
        }:
            supported.update(
                note_relation_ids.difference(
                    semantics.apply_result.ledger.verified_relations
                )
            )
        note_states = dict(pinned.task_note_states)
        note_states[pack.task_id] = note_state.model_copy(
            update={
                "independent_relation_evidence": independent,
                "supported_relation_ids": sorted(supported),
                "joint_performance_used": (
                    note_state.joint_performance_used
                    or semantics.understanding_source == "structured_joint"
                ),
            },
            deep=True,
        )
        state.pinned_dialogue_scenario_v3 = pinned.model_copy(
            update={"task_note_states": note_states},
            deep=True,
        )

    def _complete_state(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        semantics: _TurnSemantics,
    ) -> None:
        scenario, pinned = self._resolve_scenario(state)
        joint = semantics.understanding_source == "structured_joint"
        state.joint_performance_used = state.joint_performance_used or joint
        note_state = pinned.task_note_states[pack.task_id]
        note_relations = set(pack.policies.note_relation_ids)
        independent = bool(note_relations) and note_relations.issubset(
            note_state.independent_relation_evidence
        )
        task_direct = (
            independent
            if pack.policies.note_policy != "none"
            else semantics.understanding_source == "sonnet_low"
        )
        state.all_tasks_direct = state.all_tasks_direct and task_direct and not joint
        state.completed_task_slots[pack.task_id] = dict(state.verified_slots)
        if state.task_index + 1 >= len(state.task_ids):
            state.status = SessionStatus.COMPLETED
            state.completion_outcome = (
                CompletionOutcome.TAUGHT
                if state.all_tasks_direct and not state.joint_performance_used
                else CompletionOutcome.SUPPORTED
            )
            state.teach_reward_eligible = (
                state.completion_outcome is CompletionOutcome.TAUGHT
            )
            return

        next_task_id = state.task_ids[state.task_index + 1]
        next_stage = scenario.stage_by_task_id(next_task_id)
        next_variant_id = next_stage.default_variant_id
        if next_stage.selector is not None:
            source_task_id = next_stage.selector.source_task_id
            source_ledger = ReasoningLedgerV2.model_validate(
                pinned.reasoning_ledgers[source_task_id]
            )
            try:
                value = source_ledger.verified_facts[
                    next_stage.selector.fact_id
                ].canonical_value
                next_variant_id = next_stage.selector.value_to_variant_id[
                    _value_lookup_key(value)
                ]
            except KeyError as error:
                raise ValueError("life task variant selector has no verified mapping") from error
        next_pack = next_stage.variants[next_variant_id]
        active_variants = dict(pinned.active_variant_ids)
        active_variants[next_task_id] = next_variant_id
        ledgers = dict(pinned.reasoning_ledgers)
        ledgers[next_task_id] = empty_reasoning_ledger_v2(
            pin_life_task_pack_v2(next_pack)
        ).model_dump(mode="json")
        state.pinned_dialogue_scenario_v3 = pinned.model_copy(
            update={
                "active_variant_ids": active_variants,
                "reasoning_ledgers": ledgers,
            },
            deep=True,
        )
        state.task_index += 1
        state.status = SessionStatus.ACTIVE
        state.expression_level = next_pack.policies.entry_expression_level
        state.task_start_level = next_pack.policies.entry_expression_level
        state.hint_level = HintLevel.H0
        state.task_max_hint = HintLevel.H0
        state.subgoal_id = next_pack.initial_question.plan_id
        state.verified_slots = {}
        state.expression_failures = 0
        state.concept_failures = 0
        state.vague_clarifications = 0
        state.unrelated_count = 0
        patches = dict(state.scenario_data.get("v2_life_visual_patches", {}))
        patches.pop(pack.task_id, None)
        state.scenario_data["v2_life_visual_patches"] = patches
        state.scenario_data["v2_life_transition_from"] = pack.task_id

    async def _stable_fallback(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        ledger: ReasoningLedgerV2,
        question: _QuestionContract,
        *,
        stable_copy_plans: Mapping[str, StableCopyPlanV2],
        before_expression: ExpressionLevel,
        before_hint: HintLevel,
    ) -> StableCopyResolution | None:
        del state, pack, ledger, stable_copy_plans, before_expression, before_hint
        if question.copy_slot is None:
            return None
        return StableCopyResolution(
            text=question.copy_slot.reviewed_fallback,
            mood="thinking",
            dialogue_act={
                "initial_help": "offer_initial_help",
                "l2_question": "ask_structured_choice",
                "l0_intro": "enter_joint_work",
                "l0_action": "enter_joint_work",
            }[question.copy_slot.purpose],
            status="reviewed_fallback",
        )

    def _build_turn(
        self,
        state: SessionState,
        completed_pack: LifeTaskPackV2,
        completed_ledger: ReasoningLedgerV2,
        *,
        text: str,
        mood: Literal["curious", "listening", "thinking", "relieved", "celebrating"],
    ) -> TurnContract:
        transitioned = state.current_task_id != completed_pack.task_id
        # A verified relation is progress, not yet a finished learning event.
        # Emitting its reviewed note early could reveal the still-unresolved
        # answer and would detach the note from the task-transition outcome.
        note_update = (
            self._note_update(state, completed_pack, completed_ledger)
            if transitioned or state.status is SessionStatus.COMPLETED
            else None
        )
        if transitioned:
            pack, _, ledger, _ = self._resolve_state(state)
            transition = completed_pack.policies.transition_text or (
                "다음 것도 궁금해..."
            )
            state.scenario_data.pop("v2_life_transition_from", None)
        else:
            pack, ledger = completed_pack, completed_ledger

        question = self._question_contract(state, pack, ledger)
        if transitioned:
            text = f"{transition} {question.fallback}"
            mood = "curious"
        if state.status is SessionStatus.COMPLETED:
            visual = VisualContract(type="success", data={"task": pack.task_id})
        else:
            patch = state.scenario_data.get("v2_life_visual_patches", {}).get(
                pack.task_id
            )
            visual = _merge_visual_data(pack.base_visual, patch)
        completion = None
        if state.status is SessionStatus.COMPLETED and state.completion_outcome is not None:
            completion = CompletionContract(
                outcome=state.completion_outcome,
                teach_reward_eligible=state.teach_reward_eligible,
                stage_completion_eligible=True,
                verified_facts=self._completion_facts(state),
            )
        pedagogy = (
            PedagogySnapshot(
                expression_level=state.expression_level,
                hint_level=state.hint_level,
                subgoal_id=state.subgoal_id,
                verified_slots=dict(state.verified_slots),
            )
            if self.show_internal_pedagogy
            else None
        )
        return TurnContract(
            turn_id=new_id("turn"),
            scene=state.scene,
            scenario_id=state.scenario_id,
            task_id=pack.task_id,
            stage_id=pack.stage_id,
            task_index=state.task_index,
            mormi=MormiContract(text=text, mood=mood),
            input=question.input,
            visual=visual,
            help_card=self._help_card(pack, state.hint_level),
            note_update=note_update,
            status=state.status,
            state_version=state.state_version,
            completion=completion,
            pedagogy=pedagogy,
            task_anchor=(
                self._task_anchor(pack, question.input)
                if state.status is SessionStatus.ACTIVE
                else None
            ),
            dictionary_ref=(
                dictionary_reference(state.dictionary_snapshots[state.current_task_id])
                if state.current_task_id in state.dictionary_snapshots
                else None
            ),
        )

    def _note_update(
        self,
        state: SessionState,
        pack: LifeTaskPackV2,
        ledger: ReasoningLedgerV2,
    ) -> NoteUpdate | None:
        if pack.policies.note_policy == "none":
            return None
        if not set(pack.policies.note_relation_ids).issubset(
            ledger.verified_relations
        ):
            return None
        pinned = state.pinned_dialogue_scenario_v3
        if pinned is None:
            raise ValueError("missing life scenario note state")
        note_state = pinned.task_note_states[pack.task_id]
        if note_state.note_emitted:
            return None
        independent = set(pack.policies.note_relation_ids).issubset(
            note_state.independent_relation_evidence
        ) and not note_state.joint_performance_used
        note = NoteUpdate(
            skill_id=cast(str, pack.policies.note_skill_id),
            text=cast(
                str,
                pack.policies.reviewed_direct_fallback
                if independent
                else pack.policies.reviewed_coauthored_note,
            ),
            attribution=(
                NoteAttribution.CHILD if independent else NoteAttribution.COAUTHORED
            ),
            evidence=(
                NoteEvidence.DIRECT_EXPLANATION
                if independent
                else NoteEvidence.SUPPORTED_COMPLETION
            ),
            attribution_label=(
                "아이가 알려줌" if independent else "아이와 함께 공부함"
            ),
        )
        note_states = dict(pinned.task_note_states)
        note_states[pack.task_id] = note_state.model_copy(
            update={"note_emitted": True, "emitted_note_id": note.note_id},
            deep=True,
        )
        state.pinned_dialogue_scenario_v3 = pinned.model_copy(
            update={"task_note_states": note_states},
            deep=True,
        )
        return note

    @staticmethod
    def _help_card(
        pack: LifeTaskPackV2,
        level: HintLevel,
    ) -> HelpCardContract | None:
        if level is HintLevel.H0:
            return None
        card = {
            HintLevel.H1: pack.help_plan.H1,
            HintLevel.H2: pack.help_plan.H2,
            HintLevel.H3: pack.help_plan.H3,
        }[level]
        return HelpCardContract(
            visible=True,
            auto_open=True,
            level=level,
            body=card.body,
            visual_type=card.visual_type,
            visual_data=dict(card.visual_data),
        )

    @staticmethod
    def _task_anchor(
        pack: LifeTaskPackV2,
        input_contract: InputContract,
    ) -> TaskAnchorContract:
        return TaskAnchorContract(
            anchor_id=f"v2.{pack.pack_id}",
            prompt=pack.source_prompt,
            target_slots=list(input_contract.target_slots),
        )

    def _completion_facts(
        self,
        state: SessionState,
    ) -> dict[str, str | int | float | bool]:
        scenario, pinned = self._resolve_scenario(state)
        result: dict[str, str | int | float | bool] = {}
        for projection in scenario.completion_projection:
            stage = scenario.stage_by_task_id(projection.source_task_id)
            pack = stage.variants[pinned.active_variant_ids[stage.task_id]]
            ledger = ReasoningLedgerV2.model_validate(
                pinned.reasoning_ledgers[stage.task_id]
            )
            if projection.source_kind == "fact":
                entry = ledger.verified_facts.get(projection.source_id)
                if entry is not None:
                    result[projection.output_key] = _primitive(entry.canonical_value)
                    continue
                fact = next(
                    item
                    for item in pack.reasoning_graph.facts
                    if item.fact_id == projection.source_id
                )
                if not fact.initially_visible:
                    raise ValueError("life completion fact was not verified")
                result[projection.output_key] = _primitive(fact.value)
            else:
                if projection.source_id not in ledger.verified_relations:
                    raise ValueError("life completion relation was not verified")
                result[projection.output_key] = cast(
                    str | int | float | bool,
                    projection.relation_value,
                )
        return result

    async def _contextualize_note_if_safe(
        self,
        result: EngineTurnResult,
        *,
        response: ChildResponse,
        recent_dialogue: list[DialogueHistoryTurn],
    ) -> EngineTurnResult:
        note = result.turn.note_update
        if note is None or note.attribution is not NoteAttribution.CHILD:
            return result
        state = result.state
        if not state.raw_storage_enabled:
            return result
        scenario, pinned = self._resolve_scenario(state)
        source_task_id = next(
            (
                task_id
                for task_id, note_state in pinned.task_note_states.items()
                if note_state.emitted_note_id == note.note_id
            ),
            None,
        )
        if source_task_id is None:
            return result
        stage = scenario.stage_by_task_id(source_task_id)
        pack = stage.variants[pinned.active_variant_ids[source_task_id]]
        ledger = ReasoningLedgerV2.model_validate(
            pinned.reasoning_ledgers[source_task_id]
        )
        note_state = pinned.task_note_states[source_task_id]
        source_by_turn = {
            turn.turn_id: turn.child
            for turn in recent_dialogue
            if turn.child is not None
        }
        if response.text is not None:
            source_by_turn[response.turn_id] = response.text
        fragments: dict[str, str] = {}
        for relation_id in pack.policies.note_relation_ids:
            evidence_ids = note_state.independent_relation_evidence.get(relation_id, [])
            entry = ledger.verified_relations.get(relation_id)
            if entry is None:
                return result
            pointer = next(
                (
                    item
                    for item in entry.evidence
                    if item.evidence_id in evidence_ids
                    and isinstance(item, RelationVerificationEvidenceV2)
                ),
                None,
            )
            if pointer is None:
                return result
            source = source_by_turn.get(pointer.source_turn_id)
            if source is None:
                return result
            fragment = source[pointer.source_start : pointer.source_end]
            if not fragment or bridge_excerpt_violation_v2(fragment) is not None:
                return result
            fragments[relation_id] = fragment
        context = NoteContextualizationContext(
            skill_id=cast(str, pack.policies.note_skill_id),
            note_context=cast(str, pack.policies.note_context),
            source_fragments=fragments,
            reviewed_facts={
                fact.fact_id: fact.speaker_label
                for fact in pack.reasoning_graph.facts
                if fact.initially_visible or fact.fact_id in ledger.verified_facts
            },
            allowed_numbers=[
                str(_primitive(fact.value))
                for fact in pack.reasoning_graph.facts
                if isinstance(_primitive(fact.value), int | float)
            ],
            fallback_text=cast(str, pack.policies.reviewed_direct_fallback),
        )
        try:
            gateway = cast(Any, self.gateway)
            async with asyncio.timeout(min(self.speaker_timeout_seconds, 4.0)):
                output = await gateway.contextualize_note(context)
            text = validate_note_contextualization(output, context)
        except Exception:
            text = None
        if text is None:
            return result
        updated_note = note.model_copy(update={"text": text}, deep=True)
        return replace(
            result,
            turn=result.turn.model_copy(update={"note_update": updated_note}, deep=True),
        )
