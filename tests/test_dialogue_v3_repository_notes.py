from __future__ import annotations

import hashlib
from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy import select

from mormi_api.content import (
    CHANGE_TASK_ID,
    TOTAL_CALC_TASK_ID,
    create_scenario_data,
)
from mormi_api.db import (
    Database,
    DialogueTaskOutcomeRecord,
    DialogueTurnObservationRecord,
    NoteEvidenceLinkRecord,
    NoteRecord,
    OutboxEventRecord,
)
from mormi_api.dialogue_v2_cafe_content import create_cafe_scenario_pack_v2
from mormi_api.dialogue_v2_ledger import (
    ReasoningLedgerV2,
    RelationVerificationEvidenceV2,
    apply_structured_progress_v2,
    empty_reasoning_ledger_v2,
    pin_life_task_pack_v2,
)
from mormi_api.dialogue_v2_life_content import LifeScenarioPackV2, LifeTaskPackV2
from mormi_api.dialogue_v2_scenario_snapshot import (
    pin_life_scenario_runtime_v3,
    resolve_life_scenario_runtime_v3,
)
from mormi_api.outbox import DIALOGUE_OBSERVATION_EVENT_TYPE, STAR_NOTE_CREATED_EVENT_TYPE
from mormi_api.repository import PersistenceError, Repository, StaleConversationError
from mormi_api.schemas import (
    CafeMenuItem,
    CafeSessionContext,
    ChildResponse,
    CompletionOutcome,
    DialogueRuntimeContractVersion,
    ExpressionLevel,
    HintLevel,
    InputContract,
    InputKind,
    MormiContract,
    NoteAttribution,
    NoteEvidence,
    NoteUpdate,
    PinnedDialogueTaskNoteStateV3,
    ResponseCategory,
    ResponseType,
    SafetyCategory,
    SessionState,
    SessionStatus,
    SpeakerRuntimeAudit,
    TaskRelation,
    TurnContract,
    UtteranceAnalysis,
)
from mormi_api.security import TextCipher

MENU = [
    CafeMenuItem(id="americano", name="아메리카노", price=3000),
    CafeMenuItem(id="milk", name="우유", price=2000),
    CafeMenuItem(id="cake", name="케이크", price=4500),
]


def _context(mormi_menu_id: str) -> CafeSessionContext:
    child_menu_id = "milk" if mormi_menu_id != "milk" else "americano"
    return CafeSessionContext(
        menu_items=MENU,
        mormi_menu_id=mormi_menu_id,
        child_menu_id=child_menu_id,
    )


def _transition_scenario() -> LifeScenarioPackV2:
    """Put the note-enabled calculation before a synthetic note-free follow-up."""

    original = create_cafe_scenario_pack_v2(
        "cafe_menu_total",
        cafe_context=_context("americano"),
    )
    payload = deepcopy(original.model_dump(mode="json"))
    calculation = payload["task_stages"][0]
    follow_up = deepcopy(calculation)
    follow_up["task_id"] = "repository_note_follow_up"
    follow_up["variants"]["default"]["task_id"] = "repository_note_follow_up"
    follow_up["variants"]["default"]["pack_id"] = "cafe.repository-note-follow-up.v2"
    follow_up["variants"]["default"]["policies"].update(
        {
            "note_policy": "none",
            "note_relation_ids": [],
            "note_skill_id": None,
            "note_context": None,
            "reviewed_direct_fallback": None,
            "reviewed_coauthored_note": None,
        }
    )
    payload["task_stages"] = [calculation, follow_up]
    return LifeScenarioPackV2.model_validate(payload)


def _active_pack(scenario: LifeScenarioPackV2, task_id: str) -> LifeTaskPackV2:
    stage = scenario.stage_by_task_id(task_id)
    return stage.variants[stage.default_variant_id]


def _pinned_scenario(
    scenario: LifeScenarioPackV2,
    *,
    completed_task_id: str | None = None,
    source_turn_id: str | None = None,
    note: NoteUpdate | None = None,
    independent: bool = False,
):
    active_variants = {stage.task_id: stage.default_variant_id for stage in scenario.task_stages}
    ledgers: dict[str, dict[str, object]] = {}
    note_states: dict[str, PinnedDialogueTaskNoteStateV3] = {}
    for stage in scenario.task_stages:
        pack = stage.variants[active_variants[stage.task_id]]
        snapshot = pin_life_task_pack_v2(pack)
        ledger = empty_reasoning_ledger_v2(snapshot)
        note_state = PinnedDialogueTaskNoteStateV3()
        if stage.task_id == completed_task_id:
            assert source_turn_id is not None
            assert note is not None
            fact_values = {
                completion.target_id: completion.value
                for completion in pack.l0_joint_plan.completion_values
                if completion.target_kind == "fact"
            }
            relation_ids = [
                completion.target_id
                for completion in pack.l0_joint_plan.completion_values
                if completion.target_kind == "relation"
            ]
            ledger = apply_structured_progress_v2(
                snapshot,
                ledger,
                fact_values=fact_values,
                relation_ids=relation_ids,
                source_turn_id=source_turn_id,
                source_kind="joint",
            ).ledger
            if independent:
                verified_relations = dict(ledger.verified_relations)
                evidence_by_relation: dict[str, list[str]] = {}
                for relation_id in pack.policies.note_relation_ids:
                    evidence_id = hashlib.sha256(
                        f"{source_turn_id}:{relation_id}".encode()
                    ).hexdigest()
                    verified_relations[relation_id] = verified_relations[relation_id].model_copy(
                        update={
                            "evidence": [
                                RelationVerificationEvidenceV2(
                                    evidence_id=evidence_id,
                                    source_turn_id=source_turn_id,
                                    source_start=0,
                                    source_end=5,
                                    match_kind="exact",
                                    classifier_verdict="sufficient",
                                )
                            ]
                        }
                    )
                    evidence_by_relation[relation_id] = [evidence_id]
                ledger = ledger.model_copy(update={"verified_relations": verified_relations})
                note_state = PinnedDialogueTaskNoteStateV3(
                    independent_relation_evidence=evidence_by_relation,
                    note_emitted=True,
                    emitted_note_id=note.note_id,
                )
            else:
                note_state = PinnedDialogueTaskNoteStateV3(
                    supported_relation_ids=list(pack.policies.note_relation_ids),
                    joint_performance_used=True,
                    note_emitted=True,
                    emitted_note_id=note.note_id,
                )
        ledgers[stage.task_id] = ledger.model_dump(mode="json")
        note_states[stage.task_id] = note_state
    return pin_life_scenario_runtime_v3(
        scenario,
        active_variant_ids=active_variants,
        reasoning_ledgers=ledgers,
        task_note_states=note_states,
        selector_reason="repository_v3_test",
        canary_bucket=0,
    )


def _turn(
    state: SessionState,
    pack: LifeTaskPackV2,
    *,
    note: NoteUpdate | None = None,
    input_kind: InputKind = InputKind.JOINT,
) -> TurnContract:
    return TurnContract(
        scene=state.scene,
        scenario_id=state.scenario_id,
        task_id=state.current_task_id,
        stage_id=pack.stage_id,
        task_index=state.task_index,
        mormi=MormiContract(text="나와 같이 해볼래?", mood="curious"),
        input=InputContract(
            kind=(InputKind.NONE if state.status is SessionStatus.COMPLETED else input_kind),
            target_slots=["relation"],
        ),
        visual=pack.base_visual.model_copy(deep=True),
        note_update=note,
        status=state.status,
        state_version=state.state_version,
    )


async def _persist_completed_task(
    tmp_path: object,
    *,
    scenario: LifeScenarioPackV2,
    scenario_data: dict[str, object],
    source_task_id: str,
    terminal: bool,
    independent: bool = False,
) -> tuple[Repository, Database, SessionState, ChildResponse, NoteUpdate, TurnContract]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/{source_task_id}.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    source_pack = _active_pack(scenario, source_task_id)
    note = NoteUpdate(
        note_id=f"note-{source_task_id}",
        skill_id=source_pack.policies.note_skill_id or "missing",
        text=(
            source_pack.policies.reviewed_direct_fallback
            if independent
            else source_pack.policies.reviewed_coauthored_note
        )
        or "같이 공부했어.",
        attribution=(NoteAttribution.CHILD if independent else NoteAttribution.COAUTHORED),
        evidence=(
            NoteEvidence.DIRECT_EXPLANATION if independent else NoteEvidence.SUPPORTED_COMPLETION
        ),
        attribution_label=("아이가 알려줌" if independent else "아이와 함께 공부함"),
    )
    prior_pinned = _pinned_scenario(scenario)
    previous_state = SessionState(
        learner_id=31,
        learning_session_id=f"session-{source_task_id}",
        scene=scenario.scene,
        scenario_id=scenario.scenario_id,
        task_ids=[stage.task_id for stage in scenario.task_stages],
        scenario_data=scenario_data,
        task_index=0,
        expression_level=(ExpressionLevel.L4 if independent else ExpressionLevel.L0),
        hint_level=(HintLevel.H0 if independent else HintLevel.H3),
        task_start_level=ExpressionLevel.L4,
        task_max_hint=HintLevel.H3,
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
        pinned_dialogue_scenario_v3=prior_pinned,
    )
    current_turn = _turn(
        previous_state,
        source_pack,
        input_kind=(InputKind.TEXT if independent else InputKind.JOINT),
    )
    previous_state.current_turn_id = current_turn.turn_id
    await repository.create_conversation(previous_state, current_turn)

    completed_pinned = _pinned_scenario(
        scenario,
        completed_task_id=source_task_id,
        source_turn_id=current_turn.turn_id,
        note=note,
        independent=independent,
    )
    next_index = 0 if terminal else 1
    next_status = SessionStatus.COMPLETED if terminal else SessionStatus.ACTIVE
    next_state = previous_state.model_copy(deep=True)
    next_state.task_index = next_index
    next_state.status = next_status
    next_state.state_version = 2
    next_state.pinned_dialogue_scenario_v3 = completed_pinned
    next_state.completion_outcome = (
        (CompletionOutcome.TAUGHT if independent else CompletionOutcome.SUPPORTED)
        if terminal
        else None
    )
    next_state.teach_reward_eligible = bool(terminal and independent)
    next_state.joint_performance_used = not independent
    next_pack = _active_pack(scenario, next_state.current_task_id)
    next_state.expression_level = next_pack.policies.entry_expression_level
    next_state.hint_level = HintLevel.H0
    next_turn = _turn(next_state, next_pack, note=note)
    next_state.current_turn_id = next_turn.turn_id
    response = ChildResponse(
        turn_id=current_turn.turn_id,
        response_id=uuid4(),
        type=(ResponseType.TEXT if independent else ResponseType.ACTION),
        text=("낸 돈에서 메뉴값을 빼면 돼" if independent else None),
        values=({} if independent else {"joint": True}),
    )
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        task_relation=TaskRelation.CURRENT_TASK,
        confidence=1,
    )
    source_ledger = ReasoningLedgerV2.model_validate(
        completed_pinned.reasoning_ledgers[source_task_id]
    )
    runtime = SpeakerRuntimeAudit(
        dialogue_act="complete_task",
        speaker_source="reviewed_fallback",
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
        understanding_source=("sonnet_low" if independent else "structured_joint"),
        new_progress=True,
        newly_verified_fact_ids=sorted(source_ledger.verified_facts),
        newly_verified_relation_ids=sorted(source_ledger.verified_relations),
        content_pack_id=source_pack.pack_id,
        content_version=source_pack.content_version,
        content_source_hash=source_ledger.content_hash,
    )
    await repository.commit_turn(
        previous_state=previous_state,
        next_state=next_state,
        response=response,
        analysis=analysis,
        classifier_response_category=ResponseCategory.CORRECT_FULL,
        next_turn=next_turn,
        previous_question=current_turn.mormi.text,
        note=note,
        runtime=runtime,
        accepted_claims={},
    )
    return repository, database, previous_state, response, note, next_turn


async def _assert_single_note_transaction(
    repository: Repository,
    database: Database,
    previous_state: SessionState,
    response: ChildResponse,
    note: NoteUpdate,
    next_turn: TurnContract,
    *,
    expected_relation_id: str,
    expected_task_index: int,
) -> None:
    replay = await repository.response_exists(
        previous_state.conversation_id,
        str(response.response_id),
    )
    assert replay is not None
    assert replay.turn_id == next_turn.turn_id

    async with database.sessions() as db:
        notes = list((await db.execute(select(NoteRecord))).scalars())
        links = list((await db.execute(select(NoteEvidenceLinkRecord))).scalars())
        observations = list((await db.execute(select(DialogueTurnObservationRecord))).scalars())
        outcomes = list((await db.execute(select(DialogueTaskOutcomeRecord))).scalars())
        outbox = list((await db.execute(select(OutboxEventRecord))).scalars())

    assert [record.note_id for record in notes] == [note.note_id]
    assert len(links) == 1
    assert links[0].observation_id == observations[0].observation_id
    assert links[0].source_slot_ids_json == [expected_relation_id]
    assert observations[0].task_index == expected_task_index
    observation = observations[0]
    pinned = previous_state.pinned_dialogue_scenario_v3
    assert pinned is not None
    source_task_id = previous_state.current_task_id
    source_variant_id = pinned.active_variant_ids[source_task_id]
    scenario = resolve_life_scenario_runtime_v3(pinned)
    source_pack = scenario.stage_by_task_id(source_task_id).variants[source_variant_id]
    source_snapshot = pin_life_task_pack_v2(source_pack)
    persisted_state = await repository.get_state(previous_state.conversation_id)
    persisted_pinned = persisted_state.pinned_dialogue_scenario_v3
    assert persisted_pinned is not None
    result_task_id = next_turn.task_id
    result_variant_id = persisted_pinned.active_variant_ids[result_task_id]
    result_scenario = resolve_life_scenario_runtime_v3(persisted_pinned)
    result_pack = result_scenario.stage_by_task_id(result_task_id).variants[result_variant_id]
    result_snapshot = pin_life_task_pack_v2(result_pack)

    assert observation.task_id == source_task_id
    assert observation.runtime_json["observation_runtime_schema"] == (
        "life-v3-observation-runtime-v1"
    )
    assert observation.runtime_json["source_task_id"] == source_task_id
    assert observation.runtime_json["source_task_index"] == expected_task_index
    assert observation.runtime_json["source_variant_id"] == source_variant_id
    assert observation.runtime_json["source_content_pack_id"] == source_pack.pack_id
    assert observation.runtime_json["source_content_hash"] == source_snapshot.content_hash
    assert observation.runtime_json["result_task_id"] == result_task_id
    assert observation.runtime_json["result_variant_id"] == result_variant_id
    assert observation.runtime_json["result_content_pack_id"] == result_pack.pack_id
    assert observation.runtime_json["result_content_hash"] == result_snapshot.content_hash
    assert observation.runtime_json["task_transitioned"] is (source_task_id != result_task_id)
    assert (
        expected_relation_id in observation.runtime_json["reasoning_ledger_verified_relation_ids"]
    )
    assert observation.runtime_json["note_id"] == note.note_id
    assert observation.runtime_json["note_source_task_id"] == source_task_id
    assert observation.runtime_json["note_attribution"] == note.attribution.value
    assert observation.runtime_json["note_evidence"] == note.evidence.value
    assert observation.runtime_json["note_relation_ids"] == [expected_relation_id]
    if note.attribution is NoteAttribution.CHILD:
        assert observation.runtime_json["note_independent_relation_ids"] == [expected_relation_id]
        assert observation.runtime_json["note_supported_relation_ids"] == []
    else:
        assert observation.runtime_json["note_independent_relation_ids"] == []
        assert observation.runtime_json["note_supported_relation_ids"] == [expected_relation_id]
    assert observation.versions_json["dialogue_content_pack"] == source_pack.pack_id
    assert observation.versions_json["dialogue_content_source_hash"] == (
        source_snapshot.content_hash
    )
    assert observation.versions_json["dialogue_scenario_pack"] == (pinned.scenario_pack_id)
    assert observation.versions_json["dialogue_scenario_source_hash"] == (
        pinned.scenario_source_hash
    )
    assert observation.versions_json["dialogue_task_variant"] == source_variant_id
    assert observation.versions_json["observation_runtime_schema"] == (
        "life-v3-observation-runtime-v1"
    )
    assert observation.versions_json["reasoning_ledger_schema"] == ("reasoning-ledger-v2")
    assert len(outcomes) == 1
    assert outcomes[0].note_id == note.note_id
    assert expected_relation_id in outcomes[0].verified_slots_json
    assert outcomes[0].verified_slots_json[expected_relation_id]
    observation_events = [
        event for event in outbox if event.event_type == DIALOGUE_OBSERVATION_EVENT_TYPE
    ]
    note_events = [event for event in outbox if event.event_type == STAR_NOTE_CREATED_EVENT_TYPE]
    assert len(observation_events) == 1
    assert len(note_events) == 1
    assert note_events[0].aggregate_id == note.note_id
    assert note_events[0].payload_json["task_index"] == expected_task_index
    assert note_events[0].payload_json["evidence_links"] == [
        {
            "observation_id": links[0].observation_id,
            "source_slot_ids": [expected_relation_id],
        }
    ]


@pytest.mark.asyncio
async def test_v3_transition_note_uses_completed_source_task_ledger_once(
    tmp_path: object,
) -> None:
    scenario = _transition_scenario()
    context = _context("americano")
    scenario_data = create_scenario_data("cafe_menu_total", context)
    scenario_data["child_menu_id"] = "milk"
    repository, database, previous, response, note, next_turn = await _persist_completed_task(
        tmp_path,
        scenario=scenario,
        scenario_data=scenario_data,
        source_task_id=TOTAL_CALC_TASK_ID,
        terminal=False,
    )

    await _assert_single_note_transaction(
        repository,
        database,
        previous,
        response,
        note,
        next_turn,
        expected_relation_id="add_menu_prices",
        expected_task_index=0,
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_v3_terminal_note_and_duplicate_commit_remain_exactly_once(
    tmp_path: object,
) -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_change",
        cafe_context=_context("cake"),
    )
    scenario_data = create_scenario_data("cafe_change", _context("cake"))
    repository, database, previous, response, note, next_turn = await _persist_completed_task(
        tmp_path,
        scenario=scenario,
        scenario_data=scenario_data,
        source_task_id=CHANGE_TASK_ID,
        terminal=True,
        independent=True,
    )

    with pytest.raises(StaleConversationError):
        await repository.commit_turn(
            previous_state=previous,
            next_state=await repository.get_state(previous.conversation_id),
            response=response,
            analysis=UtteranceAnalysis(),
            classifier_response_category=ResponseCategory.CORRECT_FULL,
            next_turn=next_turn,
            previous_question="duplicate",
            note=note,
            runtime=SpeakerRuntimeAudit(
                dialogue_act="duplicate",
                speaker_source="reviewed_fallback",
            ),
            accepted_claims={},
        )

    await _assert_single_note_transaction(
        repository,
        database,
        previous,
        response,
        note,
        next_turn,
        expected_relation_id="subtract_menu_price",
        expected_task_index=0,
    )
    await database.dispose()


def test_v3_observation_rejects_scenario_hash_as_source_task_hash() -> None:
    scenario = create_cafe_scenario_pack_v2(
        "cafe_change",
        cafe_context=_context("cake"),
    )
    source_pack = _active_pack(scenario, CHANGE_TASK_ID)
    pinned = _pinned_scenario(scenario)
    previous_state = SessionState(
        learner_id=32,
        learning_session_id="session-bad-observation-hash",
        scene=scenario.scene,
        scenario_id=scenario.scenario_id,
        task_ids=[CHANGE_TASK_ID],
        scenario_data=create_scenario_data("cafe_change", _context("cake")),
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
        pinned_dialogue_scenario_v3=pinned,
        expression_level=ExpressionLevel.L4,
    )
    next_state = previous_state.model_copy(deep=True)
    next_state.state_version += 1
    next_turn = _turn(next_state, source_pack, input_kind=InputKind.TEXT)

    with pytest.raises(
        PersistenceError,
        match="v3_observation_content_identity_mismatch",
    ):
        Repository._observation_runtime_metadata(
            previous_state=previous_state,
            next_state=next_state,
            next_turn=next_turn,
            note=None,
            runtime=SpeakerRuntimeAudit(
                dialogue_act="ask_answer_and_method",
                speaker_source="llm",
                runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
                understanding_source="sonnet_low",
                content_pack_id=source_pack.pack_id,
                content_version=source_pack.content_version,
                # This is the aggregate scenario hash, not the active task
                # variant hash. Persistence must not silently accept it.
                content_source_hash=pinned.scenario_source_hash,
            ),
        )
