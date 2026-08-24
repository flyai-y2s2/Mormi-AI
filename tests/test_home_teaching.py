from __future__ import annotations

from uuid import uuid4

import pytest
from conftest import FakeGateway
from sqlalchemy import select

from mormi_api.content import HOME_TEACHING_CATALOG, HomeTeachingSpec, home_teaching_task
from mormi_api.db import (
    ConversationRecord,
    Database,
    DialogueClaimRecord,
    DialogueTaskOutcomeRecord,
    DialogueTurnObservationRecord,
    NoteEvidenceLinkRecord,
    OutboxEventRecord,
    TurnRecord,
)
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    DifficultyClass,
    EntryPhase,
    ExpressionLevel,
    HintLevel,
    InputKind,
    LearnerProfile,
    PracticeResult,
    ResponseCategory,
    SafetyCategory,
    SessionCreate,
    SessionEnvelope,
    SessionState,
    SkillProfile,
    SlotClaim,
    SpeakerContext,
    SpeakerOutput,
    SpeakerVerificationPolicy,
    UtteranceAnalysis,
)
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService

CURRENT_FRONTEND_HOME_SESSION_IDS = {
    "number-count",
    "number-compare",
    "money-count",
    "number-make-ten",
    "number-place-value",
    "add-pictures",
    "money-price",
    "add-place",
    "add-make-ten",
    "sub-pictures",
    "money-budget",
    "sub-place",
    "sub-borrow",
    "multiply-groups",
    "multiply-addition",
    "money-mission",
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
    "measure-weight-capacity",
    "geometry-shapes",
    "geometry-compose",
    "geometry-position",
    "data-classify",
    "data-chart",
    "data-chance",
}

CONTENT_VERSION_8_HOME_SESSION_IDS = {
    "number-count",
    "number-compare",
    "money-count",
    "number-make-ten",
    "number-place-value",
    "add-pictures",
    "money-price",
    "sub-borrow",
    "multiply-addition",
    "multiply-tables",
    "clock-basic",
    "clock-quarter",
    "time-calendar",
    "measure-weight-capacity",
    "geometry-compose",
    "data-chance",
}


def test_home_catalog_covers_current_frontend_curriculum() -> None:
    assert set(HOME_TEACHING_CATALOG) == CURRENT_FRONTEND_HOME_SESSION_IDS
    assert {
        spec.id for spec in HOME_TEACHING_CATALOG.values() if spec.content_version == 8
    } == CONTENT_VERSION_8_HOME_SESSION_IDS
    assert all(spec.content_version in {7, 8} for spec in HOME_TEACHING_CATALOG.values())
    assert all(spec.content_version >= 2 for spec in HOME_TEACHING_CATALOG.values())
    # Dialogue copy should aim for 50 characters, but a complete reviewed
    # sentence may exceed it. Runtime contracts must not reject or truncate it.
    assert all(spec.effective_l4_prompt.strip() for spec in HOME_TEACHING_CATALOG.values())
    assert all(spec.entry_mode != "wrong_guess" for spec in HOME_TEACHING_CATALOG.values())
    assert all(spec.entry_prompt is None for spec in HOME_TEACHING_CATALOG.values())


def test_number_count_copy_sounds_like_a_younger_sibling_asking_for_help() -> None:
    spec = HOME_TEACHING_CATALOG["number-count"]

    assert spec.sample_problem["prompt"] == "지금 점이 몇 개야?"
    assert spec.entry_prompt is None
    assert spec.effective_l4_prompt == "점이 몇 개인지랑 어떻게 세는지 알려주면 안 될까?"
    assert spec.short_prompt == ("나는 가끔 점을 세다가 헷갈려. 어떻게 세는지 알려주면 안 될까?")
    assert all(
        phrase
        not in " ".join(
            [
                spec.sample_problem["prompt"],
                spec.effective_l4_prompt,
                spec.short_prompt,
            ]
        )
        for phrase in ("그 부분은 기억했어", "네가 말한 데까지", "점을 하나 놓칠 때가 있어")
    )


def test_number_count_support_does_not_force_pointing_as_the_only_method() -> None:
    spec = HOME_TEACHING_CATALOG["number-count"]
    task = home_teaching_task(spec, skill_id=spec.id)

    assert spec.short_correct == "점을 하나씩 보며 하나, 둘, 셋 하고 세기"
    assert "가리키" not in " ".join(
        [spec.learned_line, spec.hint, *spec.help_lines, *spec.short_options]
    )
    assert ExpressionLevel.L1 not in task.steps
    short_method = task.steps[ExpressionLevel.L3][1]
    choice_method = task.steps[ExpressionLevel.L2][1]
    assert short_method.prompt != choice_method.prompt
    assert short_method.input.kind is InputKind.TEXT
    assert choice_method.input.kind is InputKind.CHOICES
    reviewed_choice = next(
        choice for choice in choice_method.input.choices if choice.label == spec.short_correct
    )
    assert choice_method.choice_effects[reviewed_choice.id] == {"tracking": "count_each_once"}


def test_every_home_task_declares_semantic_roles_instead_of_relying_on_slot_names() -> None:
    for spec in HOME_TEACHING_CATALOG.values():
        task = home_teaching_task(spec, skill_id=spec.id)
        assert all(slot.semantic_role for slot in task.slots.values())
        assert all(
            slot.resolved_evaluation_mode
            == (
                "semantic_support"
                if slot.semantic_role in {"method", "reason", "explanation"}
                else "canonical_value"
            )
            for slot in task.slots.values()
        )
        assert task.slots["answer"].semantic_role == "conclusion"
        if spec.id == "number-count":
            assert task.slots["tracking"].semantic_role == "method"
        elif spec.id == "number-compare":
            assert task.slots["reason"].semantic_role == "reason"
        else:
            assert task.slots["rule"].semantic_role == "explanation"


@pytest.mark.asyncio
@pytest.mark.parametrize("curriculum_session_id", sorted(CURRENT_FRONTEND_HOME_SESSION_IDS))
async def test_every_frontend_home_session_can_create_its_first_turn(
    curriculum_session_id: str,
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/{curriculum_session_id}-first-turn.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )

    started = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id=f"session_{curriculum_session_id}",
            practice_result_id=f"practice_{curriculum_session_id}",
            practice_summary={
                "curriculum_session_id": curriculum_session_id,
                "skill_id": curriculum_session_id,
                "question_count": 5,
                # Drill mistakes are concept evidence.  They must never skip
                # the child's first independent teaching opportunity.
                "first_try_correct_count": 2,
                "wrong_attempt_count": 3,
                "earned_reward": 1000,
                "attempts": [
                    {
                        "item_id": f"{curriculum_session_id}:{index}",
                        "correct": index < 2,
                        "latency_ms": 1000,
                    }
                    for index in range(5)
                ],
            },
        )
    )

    state = await repository.get_state(started.conversation_id)
    spec = HOME_TEACHING_CATALOG[curriculum_session_id]
    assert started.turn.visual.data["curriculum_session_id"] == curriculum_session_id
    assert started.turn.status.value == "active"
    assert started.turn.mormi.text == spec.effective_l4_prompt
    assert started.turn.visual.data["problem"]["prompt"] == spec.sample_problem["prompt"]
    assert started.turn.input.kind is InputKind.TEXT
    if curriculum_session_id == "number-count":
        expected_target_slots = ["answer", "tracking"]
    elif curriculum_session_id == "number-compare":
        expected_target_slots = ["answer", "reason"]
    else:
        expected_target_slots = ["answer", "rule"]
    assert started.turn.input.target_slots == expected_target_slots
    assert started.turn.input.choices == []
    assert started.turn.help_card is None
    assert state.expression_level is ExpressionLevel.L4
    assert state.hint_level is HintLevel.H0
    assert state.dialogue_policy_version == 3
    assert state.entry_phase is EntryPhase.RESOLVED
    await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("curriculum_session_id", sorted(CURRENT_FRONTEND_HOME_SESSION_IDS))
async def test_every_home_support_path_persists_repeated_help_requests(
    curriculum_session_id: str,
    tmp_path: object,
) -> None:
    """Every reviewed home task must survive repeated L/H support turns.

    This exercises observation and outbox writes with SQLite foreign keys
    enabled at every support level. L0 deliberately remains in joint mode when
    the child keeps requesting help, so this test checks durability rather than
    forcing an artificial completion.
    """

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/{curriculum_session_id}-support-path.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    envelope = await service.create_conversation(
        SessionCreate(
            learner_id=1,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id=f"support_{curriculum_session_id}",
            practice_result_id=f"practice_support_{curriculum_session_id}",
            practice_summary={
                "curriculum_session_id": curriculum_session_id,
                "skill_id": curriculum_session_id,
                "question_count": 1,
                "first_try_correct_count": 0,
                "wrong_attempt_count": 1,
                "earned_reward": 0,
                "attempts": [
                    {
                        "item_id": f"{curriculum_session_id}:support",
                        "correct": False,
                        "latency_ms": 1000,
                    }
                ],
            },
        )
    )

    for _ in range(8):
        envelope = await service.respond(
            envelope.conversation_id,
            ChildResponse(
                turn_id=envelope.turn.turn_id,
                response_id=uuid4(),
                type="no_response",
            ),
        )

    async with database.sessions() as db:
        observations = list((await db.execute(select(DialogueTurnObservationRecord))).scalars())
        outbox_events = list((await db.execute(select(OutboxEventRecord))).scalars())
    assert len(observations) == 8
    assert len(outbox_events) == 8
    await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("curriculum_session_id", sorted(CURRENT_FRONTEND_HOME_SESSION_IDS))
async def test_every_home_l4_answer_only_preserves_credit_and_asks_the_missing_idea(
    curriculum_session_id: str,
    tmp_path: object,
) -> None:
    spec = HOME_TEACHING_CATALOG[curriculum_session_id]
    task = home_teaching_task(spec, skill_id=spec.id)
    expected_answer = task.slots["answer"].expected
    answer_text = str(expected_answer)
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value=expected_answer,
                factual=True,
                evidence_span=answer_text,
            )
        ],
        confidence=1,
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/{curriculum_session_id}-answer-only.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway([analysis])),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=2,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id=f"session_{curriculum_session_id}_answer_only",
            practice_result_id=f"practice_{curriculum_session_id}_answer_only",
            practice_summary={
                "curriculum_session_id": curriculum_session_id,
                "skill_id": curriculum_session_id,
                "question_count": 5,
                "first_try_correct_count": 3,
            },
        )
    )

    followed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="0317accf-6a3c-40a4-a1e2-b35484f66405",
            type="text",
            text=answer_text,
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots["answer"] == task.slots["answer"].canonical(expected_answer)
    assert state.expression_level is ExpressionLevel.L4
    assert state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP
    assert state.hint_level is HintLevel.H0
    assert followed.turn.input.target_slots == task.missing_slots(state.verified_slots)
    assert followed.turn.visual == started.turn.visual
    assert followed.turn.help_card is None
    await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("curriculum_session_id", "child_text", "claim_value", "expected_answer"),
    [
        ("money-price", "1200", 1200, "1,200원"),
        ("money-price", "1,200원", 1200, "1,200원"),
        ("money-price", "1200원이지", 1200, "1,200원"),
        ("money-budget", "200", 200, "200원"),
        ("money-count", "600원이지", 600, "600원"),
    ],
)
async def test_home_structured_numeric_answer_advances_without_repeating(
    curriculum_session_id: str,
    child_text: str,
    claim_value: int,
    expected_answer: str,
    tmp_path: object,
) -> None:
    """A model-interpreted amount is compared as a value, not Korean copy."""

    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value=claim_value,
                factual=True,
                evidence_span=child_text,
                interpretation_confidence=0.99,
            )
        ],
        confidence=0.9,
    )
    database = Database(
        f"sqlite+aiosqlite:///{tmp_path}/{curriculum_session_id}-{child_text}-claimless.db"
    )
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway([analysis])),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=22,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id=f"session_{curriculum_session_id}_claimless",
            practice_result_id=f"practice_{curriculum_session_id}_claimless",
            practice_summary={
                "curriculum_session_id": curriculum_session_id,
                "skill_id": curriculum_session_id,
                "question_count": 5,
                "first_try_correct_count": 3,
            },
        )
    )

    followed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id=uuid4(),
            type="text",
            text=child_text,
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots["answer"] == expected_answer
    assert state.expression_level is ExpressionLevel.L4
    assert state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP
    assert state.hint_level is HintLevel.H0
    assert followed.turn.input.target_slots == ["rule"]
    assert followed.turn.help_card is None
    assert followed.turn.visual == started.turn.visual
    await database.dispose()


def test_structured_numeric_claim_must_be_grounded_in_child_evidence() -> None:
    spec = HOME_TEACHING_CATALOG["money-price"]
    task = home_teaching_task(spec, skill_id=spec.id)

    for child_text, evidence_span in (("1300", "1300"), ("700원과 500원", "700원과 500원")):
        analysis = UtteranceAnalysis(
            safety_category=SafetyCategory.NORMAL,
            response_category=ResponseCategory.CORRECT_PARTIAL,
            claims=[
                SlotClaim(
                    slot_id="answer",
                    value=1200,
                    factual=True,
                    evidence_span=evidence_span,
                    interpretation_confidence=0.99,
                )
            ],
        )
        ConversationEngine._ground_text_numeric_claims(
            task,
            child_text,
            analysis,
        )
        assert analysis.claims[0].factual is False


def test_closed_numeric_slot_compares_structured_values_not_display_copy() -> None:
    spec = HOME_TEACHING_CATALOG["money-price"]
    answer = home_teaching_task(spec, skill_id=spec.id).slots["answer"]

    assert answer.expected == "1,200원"
    assert answer.accepts(1200)
    assert answer.accepts("1200원")
    assert not answer.accepts(1300)


def test_every_home_support_step_keeps_question_and_choices_in_one_context() -> None:
    from mormi_api.content import home_teaching_task

    for spec in HOME_TEACHING_CATALOG.values():
        task = home_teaching_task(spec, skill_id=spec.id)
        l2_answer, l2_method = task.steps[ExpressionLevel.L2]
        expected_answers = [str(answer) for answer in spec.sample_problem["answers"]]

        assert ExpressionLevel.L1 not in task.steps
        assert l2_answer.input.kind is InputKind.CHOICES
        assert l2_method.input.kind is InputKind.CHOICES

        if spec.id == "number-compare":
            assert "어느 쪽" in l2_answer.prompt
            assert [choice.label for choice in l2_answer.input.choices] == expected_answers
            assert "왜 오른쪽" in l2_method.prompt
            assert "3" not in l2_method.prompt
            assert "5" not in l2_method.prompt
            assert "같이 골라 볼까?" not in l2_method.prompt
            assert [choice.label for choice in l2_method.input.choices] == [
                "왼쪽은 3개, 오른쪽은 5개라서",
                "왼쪽은 5개, 오른쪽은 3개라서",
                "두 쪽 모두 5개라서",
            ]
            continue

        if spec.id == "number-count":
            assert l2_answer.prompt == spec.sample_problem["prompt"]
            assert [choice.label for choice in l2_answer.input.choices] == expected_answers
            # The preceding targeted follow-up already uses short_prompt with
            # a text box.  L2 must therefore sound different and expose real
            # choice support instead of visually repeating the same turn.
            assert l2_method.prompt != spec.short_prompt
            assert "같이" not in l2_method.prompt
            assert "골라서 알려줄 수 있어?" in l2_method.prompt
            assert l2_method.input.kind is InputKind.CHOICES
            continue

        assert l2_answer.prompt == spec.sample_problem["prompt"]
        assert [choice.label for choice in l2_answer.input.choices] == expected_answers
        assert l2_method.prompt == spec.short_prompt
        assert [choice.label for choice in l2_method.input.choices] == spec.short_options
        assert spec.short_prompt != "어떤 방법이 맞을까?"


@pytest.mark.asyncio
async def test_home_first_turn_still_probes_l4_with_an_old_lower_profile(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/home-old-profile.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    await repository.save_profile(
        LearnerProfile(
            learner_id=12,
            skills={
                "number-count": SkillProfile(
                    skill_id="number-count",
                    highest_stable_expression_level=ExpressionLevel.L0,
                )
            },
        )
    )
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=12,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_old_profile",
            practice_result_id="practice_old_profile",
            practice_summary={
                "curriculum_session_id": "number-count",
                "skill_id": "number-count",
                "question_count": 5,
                "first_try_correct_count": 1,
            },
        )
    )

    assert started.turn.input.kind is InputKind.TEXT
    assert started.turn.help_card is None
    state = await repository.get_state(started.conversation_id)
    assert state.expression_level is ExpressionLevel.L4
    assert state.hint_level is HintLevel.H0
    await database.dispose()


async def _start_money_count_conversation(
    tmp_path: object,
    analyses: list[UtteranceAnalysis],
    suffix: str,
) -> tuple[Database, Repository, ConversationService, SessionEnvelope]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/entry-{suffix}.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway(analyses)),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=31,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id=f"entry-session-{suffix}",
            practice_result_id=f"entry-practice-{suffix}",
            practice_summary={
                "curriculum_session_id": "money-count",
                "skill_id": "money-count",
                "question_count": 5,
                "first_try_correct_count": 5,
            },
        )
    )
    return database, repository, service, started


@pytest.mark.asyncio
async def test_persistence_uses_engine_accepted_claims_not_raw_classifier_claims(
    tmp_path: object,
) -> None:
    """A speaker-safe rejection must also remain rejected in analytics.

    The classifier deliberately overclaims a comparison reason from a bare
    conclusion. The engine accepts only the answer; persistence must not apply
    a second, weaker validation rule that upgrades the reason for reports.
    """

    child_text = "오른쪽이 더 많아"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span=child_text,
            ),
            SlotClaim(
                slot_id="reason",
                value="count_comparison",
                factual=True,
                evidence_span=child_text,
            ),
        ],
        confidence=1,
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/accepted-claim-contract.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway([analysis])),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=32,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="accepted-claim-contract",
            practice_result_id="accepted-claim-practice",
            practice_summary={
                "curriculum_session_id": "number-compare",
                "skill_id": "number-compare",
                "question_count": 1,
                "first_try_correct_count": 1,
            },
        )
    )

    next_turn = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id=uuid4(),
            type="text",
            text=child_text,
        ),
    )
    assert next_turn.turn.status.value == "active"
    assert next_turn.turn.input.target_slots == ["reason"]

    async with database.sessions() as db:
        claims = {
            claim.slot_id: claim
            for claim in (await db.execute(select(DialogueClaimRecord))).scalars()
        }
        outbox = (await db.execute(select(OutboxEventRecord))).scalar_one()
    assert claims["answer"].validation_status == "verified"
    assert claims["answer"].value_json == "오른쪽"
    assert claims["answer"].newly_verified is True
    assert claims["reason"].validation_status == "rejected"
    assert claims["reason"].value_json is None
    assert claims["reason"].newly_verified is False
    outbox_claims = {claim["slot_id"]: claim for claim in outbox.payload_json["claims"]}
    assert outbox_claims["reason"]["validation_status"] == "rejected"
    assert outbox_claims["reason"]["value"] is None
    await database.dispose()


@pytest.mark.asyncio
async def test_genuine_l4_full_answer_can_complete_in_one_response(
    tmp_path: object,
) -> None:
    child_text = "600원이야. 500원과 100원을 더했어"
    rule_span = "500원과 100원을 더했어"
    expected_rule = HOME_TEACHING_CATALOG["money-count"].learned_line
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="600원",
                factual=True,
                evidence_span="600원이야",
            ),
            SlotClaim(
                slot_id="rule",
                value=expected_rule,
                factual=True,
                evidence_span=rule_span,
            ),
        ],
        confidence=1,
    )
    database, _, service, started = await _start_money_count_conversation(
        tmp_path, [analysis], "full-answer"
    )

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="531dbf5d-6038-4658-971d-39d719d807ea",
            type="text",
            text=child_text,
        ),
    )

    assert completed.turn.status.value == "completed"
    assert completed.turn.note_update is not None
    assert rule_span in completed.turn.note_update.text
    async with database.sessions() as db:
        observation = (await db.execute(select(DialogueTurnObservationRecord))).scalar_one()
        claims = list((await db.execute(select(DialogueClaimRecord))).scalars())
        outcome = (await db.execute(select(DialogueTaskOutcomeRecord))).scalar_one()
        evidence_link = (await db.execute(select(NoteEvidenceLinkRecord))).scalar_one()
        outboxes = list((await db.execute(select(OutboxEventRecord))).scalars())
        observation_outbox = next(
            event for event in outboxes if event.event_type == "mormi.dialogue.observation.recorded"
        )
        star_note_outbox = next(
            event for event in outboxes if event.event_type == "mormi.star_note.created"
        )

    assert observation.response_category == "correct_full"
    assert observation.concept_result == "correct_full"
    assert observation.expression_before == "L4"
    assert observation.hint_before == "H0"
    assert observation.dialogue_act == "session_complete"
    assert observation.record_origin == "live"
    assert observation.analysis_json.get("claims") is None
    assert observation.analysis_json.get("grounding_span") is None
    assert observation.analysis_json.get("social_grounding_span") is None
    assert {claim.slot_id for claim in claims} == {"answer", "rule"}
    assert all(claim.validation_status == "verified" for claim in claims)
    assert all(
        claim.evidence_span_encrypted and claim.evidence_span_encrypted.startswith("plain:")
        for claim in claims
    )
    assert outcome.completion_outcome == "taught"
    assert outcome.evidence_observation_ids_json == [observation.observation_id]
    assert outcome.note_id == completed.turn.note_update.note_id
    assert evidence_link.observation_id == observation.observation_id
    assert observation_outbox.aggregate_id == observation.observation_id
    assert observation_outbox.payload_json["observation_version"] == 1
    assert observation_outbox.payload_json["stage"] == observation.stage_id
    assert observation_outbox.payload_json["bottleneck_candidate"] == observation.bottleneck
    assert observation_outbox.payload_json["help_used"] is observation.help_card_shown
    assert observation_outbox.payload_json["fallback_occurred"] is False
    assert observation_outbox.payload_json["completion_outcome"] == (observation.completion_outcome)
    assert observation_outbox.payload_json["observed_at"].removesuffix("+00:00") == (
        observation.created_at.isoformat()
    )
    assert observation_outbox.payload_json["claims"][0].get("evidence_span") is None
    assert star_note_outbox.aggregate_id == completed.turn.note_update.note_id
    assert star_note_outbox.payload_json["note_id"] == completed.turn.note_update.note_id
    assert star_note_outbox.payload_json["text"] == completed.turn.note_update.text
    assert star_note_outbox.payload_json["attribution"] == "child"
    assert star_note_outbox.payload_json["note_version"] == 1
    assert star_note_outbox.payload_json["evidence_links"] == [
        {
            "observation_id": observation.observation_id,
            "source_slot_ids": evidence_link.source_slot_ids_json,
        }
    ]
    assert "response" not in star_note_outbox.payload_json
    assert "evidence_span" not in star_note_outbox.payload_json
    await database.dispose()


@pytest.mark.asyncio
async def test_persistence_keeps_classifier_category_but_reports_reconciled_result(
    tmp_path: object,
) -> None:
    child_text = "600원이야. 500원과 100원을 더했어"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        # Reproduce the pilot failure: the classifier says partial although it
        # supplied valid evidence for every slot requested by the L4 turn.
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="600원",
                factual=True,
                evidence_span="600원이야",
            ),
            SlotClaim(
                slot_id="rule",
                factual=True,
                supported=True,
                support_confidence=0.98,
                evidence_span="500원과 100원을 더했어",
            ),
        ],
        confidence=0.91,
    )
    database, _, service, started = await _start_money_count_conversation(
        tmp_path, [analysis], "reconciled-category"
    )

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="741dbf5d-6038-4658-971d-39d719d807ea",
            type="text",
            text=child_text,
        ),
    )

    assert completed.turn.status.value == "completed"
    async with database.sessions() as db:
        observation = (await db.execute(select(DialogueTurnObservationRecord))).scalar_one()
        answered_turn = (
            await db.execute(select(TurnRecord).where(TurnRecord.response_id.is_not(None)))
        ).scalar_one()

    assert answered_turn.response_category == "correct_full"
    assert observation.response_category == "correct_full"
    assert observation.concept_result == "correct_full"
    assert observation.analysis_json["classifier_response_category"] == "correct_partial"
    assert observation.analysis_json["effective_response_category"] == "correct_full"
    assert observation.analysis_json["response_category_reconciled"] is True
    await database.dispose()


@pytest.mark.asyncio
async def test_genuine_l4_answer_only_keeps_l4_and_asks_only_for_method(
    tmp_path: object,
) -> None:
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="600원",
                factual=True,
                evidence_span="600원이야",
            )
        ],
        confidence=1,
    )
    database, repository, service, started = await _start_money_count_conversation(
        tmp_path, [analysis], "answer-only"
    )

    followed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="541dbf5d-6038-4658-971d-39d719d807ea",
            type="text",
            text="600원이야",
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots == {"answer": "600원"}
    assert state.expression_level is ExpressionLevel.L4
    assert state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP
    assert followed.turn.mormi.text.endswith(HOME_TEACHING_CATALOG["money-count"].short_prompt)
    assert followed.turn.input.target_slots == ["rule"]
    assert followed.turn.help_card is None
    await database.dispose()


@pytest.mark.asyncio
async def test_genuine_l4_method_only_keeps_l4_and_asks_only_for_answer(
    tmp_path: object,
) -> None:
    child_text = "그건 간단한 덧셈이야"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="rule",
                factual=True,
                supported=True,
                support_confidence=0.99,
                evidence_span="덧셈이야",
            )
        ],
        grounding_span="덧셈이야",
        confidence=1,
    )

    class MethodFirstSpeakerGateway(FakeGateway):
        speaker_context: SpeakerContext | None = None

        async def speak(self, context: SpeakerContext) -> SpeakerOutput:
            self.speaker_context = context
            return SpeakerOutput(
                text="아, 덧셈을 하면 되는구나~ 그럼 모두 얼마인지 알려줄 수 있어?",
                dialogue_act=context.dialogue_act,
                asked_slot_ids=context.required_slot_ids,
                used_verified_slots=["rule"],
            )

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/method-first-money-price.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    gateway = MethodFirstSpeakerGateway([analysis])
    service = ConversationService(
        repository,
        ConversationEngine(gateway),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=32,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="method-first-money-price",
            practice_result_id="method-first-money-price-practice",
            practice_summary={
                "curriculum_session_id": "money-price",
                "skill_id": "money-price",
                "question_count": 5,
                "first_try_correct_count": 5,
            },
        )
    )
    original_visual = started.turn.visual

    followed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="e7cc6862-b91d-4dd6-978f-56ef272bbb7a",
            type="text",
            text=child_text,
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots == {"rule": True}
    assert state.expression_level is ExpressionLevel.L4
    assert state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP
    assert state.hint_level is HintLevel.H0
    assert followed.turn.status.value == "active"
    assert gateway.speaker_context is not None
    assert gateway.speaker_context.required_slot_ids == ["answer"]
    assert gateway.speaker_context.safe_child_utterance == child_text
    assert gateway.speaker_context.verified_facts.keys() == {"rule"}
    assert followed.turn.mormi.text == (
        "아, 덧셈을 하면 되는구나~ 그럼 모두 얼마인지 알려줄 수 있어?"
    )
    assert followed.turn.input.kind is InputKind.TEXT
    assert followed.turn.input.target_slots == ["answer"]
    assert followed.turn.visual == original_visual
    assert followed.turn.help_card is None
    await database.dispose()


@pytest.mark.asyncio
async def test_method_first_followup_skips_redundant_verifier_and_preserves_acknowledgement(
    tmp_path: object,
) -> None:
    child_text = "그건 간단한 덧셈이야"
    expected_text = "아, 덧셈을 하면 되는구나~ 그럼 모두 얼마인지 알려줄 수 있어?"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="rule",
                factual=True,
                supported=True,
                support_confidence=0.99,
                evidence_span="덧셈이야",
            )
        ],
        grounding_span="덧셈이야",
        confidence=1,
    )

    class VerificationTimeoutGateway(FakeGateway):
        speaker_context: SpeakerContext | None = None

        async def speak(self, context: SpeakerContext) -> SpeakerOutput:
            self.speaker_context = context
            return SpeakerOutput(
                text=expected_text,
                dialogue_act=context.dialogue_act,
                asked_slot_ids=context.required_slot_ids,
                used_verified_slots=["rule"],
            )

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/method-first-timeout.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    gateway = VerificationTimeoutGateway([analysis])
    service = ConversationService(
        repository,
        ConversationEngine(gateway),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=33,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="method-first-timeout",
            practice_result_id="method-first-timeout-practice",
            practice_summary={
                "curriculum_session_id": "money-price",
                "skill_id": "money-price",
                "question_count": 5,
                "first_try_correct_count": 5,
            },
        )
    )

    followed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="2b460314-bf6b-4aa5-aad0-812aa10d2164",
            type="text",
            text=child_text,
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots == {"rule": True}
    assert gateway.speaker_context is not None
    assert gateway.speaker_context.verification_policy is SpeakerVerificationPolicy.DETERMINISTIC
    assert followed.turn.mormi.text == expected_text
    assert followed.turn.input.target_slots == ["answer"]
    assert followed.turn.help_card is None
    async with database.sessions() as db:
        observation = (
            await db.execute(
                select(DialogueTurnObservationRecord).where(
                    DialogueTurnObservationRecord.response_id
                    == "2b460314-bf6b-4aa5-aad0-812aa10d2164"
                )
            )
        ).scalar_one()
        assert observation.runtime_json["verifier_status"] == "not_required"
        assert isinstance(observation.runtime_json["speaker_latency_ms"], int)
        assert observation.runtime_json["verifier_latency_ms"] is None
    await database.dispose()


@pytest.mark.asyncio
async def test_no_response_from_genuine_l4_lowers_one_step_and_opens_first_help_card(
    tmp_path: object,
) -> None:
    database, repository, service, started = await _start_money_count_conversation(
        tmp_path, [], "entry-help"
    )
    original_visual = started.turn.visual

    supported = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="561dbf5d-6038-4658-971d-39d719d807ea",
            type="no_response",
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.expression_level is ExpressionLevel.L3
    assert state.hint_level is HintLevel.H1
    assert state.entry_phase is EntryPhase.RESOLVED
    assert supported.turn.visual == original_visual
    assert supported.turn.help_card is not None
    assert supported.turn.input.kind is InputKind.TEXT
    await database.dispose()


@pytest.mark.asyncio
async def test_unsafe_text_keeps_the_genuine_l4_question_and_state(
    tmp_path: object,
) -> None:
    database, repository, service, started = await _start_money_count_conversation(
        tmp_path, [], "entry-safety"
    )

    redirected = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="571dbf5d-6038-4658-971d-39d719d807ea",
            type="text",
            text="씨발",
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.entry_phase is EntryPhase.RESOLVED
    assert state.expression_level is ExpressionLevel.L4
    assert state.hint_level is HintLevel.H0
    assert state.verified_slots == {}
    assert redirected.turn.input == started.turn.input
    assert redirected.turn.visual == started.turn.visual
    assert HOME_TEACHING_CATALOG["money-count"].effective_l4_prompt in redirected.turn.mormi.text
    await database.dispose()


def test_v2_snapshot_can_still_render_an_already_open_wrong_guess() -> None:
    current = HOME_TEACHING_CATALOG["money-count"].model_dump(mode="json")
    current["entry_mode"] = "wrong_guess"
    current["entry_prompt"] = "저금통에 500원이 있었어. 100원을 더 넣으면 510원일까?"
    legacy_spec = HomeTeachingSpec.model_validate(current)
    scenario_data = {
        "curriculum_session_id": legacy_spec.id,
        "skill_id": legacy_spec.id,
        "home_teaching_spec": legacy_spec.model_dump(mode="json"),
    }
    state = SessionState(
        learner_id=31,
        scene="home_teach",
        scenario_id="home_teach",
        task_ids=["home_teaching"],
        scenario_data=scenario_data,
        expression_level=ExpressionLevel.L4,
        dialogue_policy_version=2,
        entry_phase=EntryPhase.AWAITING_ENTRY_RESPONSE,
    )

    turn = ConversationEngine(FakeGateway()).initial_turn(state)  # type: ignore[arg-type]

    assert turn.mormi.text == legacy_spec.entry_prompt
    assert turn.help_card is None


def test_legacy_home_snapshot_resumes_without_new_entry_turn() -> None:
    current = HOME_TEACHING_CATALOG["money-count"].model_dump(mode="json")
    current.pop("content_version")
    current.pop("entry_mode")
    current.pop("entry_prompt")
    old_l4 = current.pop("l4_prompt")
    current["misconception_prompt"] = old_l4

    legacy_spec = HomeTeachingSpec.model_validate(current)
    task = home_teaching_task(legacy_spec, skill_id=legacy_spec.id)

    assert legacy_spec.content_version == 1
    assert task.entry_step is None
    assert task.steps[ExpressionLevel.L4][0].prompt == old_l4


@pytest.mark.asyncio
async def test_saved_practice_result_generates_home_scenario_and_stores_raw_turn(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/home-teaching.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    child_text = "돈의 개수가 아니라 적힌 값을 모두 더해야 해"
    analyses = [
        UtteranceAnalysis(
            safety_category=SafetyCategory.NORMAL,
            response_category=ResponseCategory.CORRECT_PARTIAL,
            difficulty_class=DifficultyClass.UNKNOWN,
            claims=[
                SlotClaim(
                    slot_id="rule",
                    factual=True,
                    supported=True,
                    support_confidence=1,
                    evidence_span=child_text,
                )
            ],
            note_candidate=child_text,
            confidence=1,
        ),
        UtteranceAnalysis(
            safety_category=SafetyCategory.NORMAL,
            response_category=ResponseCategory.CORRECT_FULL,
            difficulty_class=DifficultyClass.UNKNOWN,
            claims=[
                SlotClaim(
                    slot_id="answer",
                    value="600원",
                    factual=True,
                    evidence_span="600원",
                )
            ],
            confidence=1,
        ),
    ]
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway(analyses)),  # type: ignore[arg-type]
    )

    practice = PracticeResult(
        practice_result_id="practice_home_money_count",
        learner_id=7,
        curriculum_session_id="money-count",
        skill_id="money_count",
        question_count=5,
        first_try_correct_count=5,
    )
    await service.save_practice(practice)
    started = await service.create_conversation(
        SessionCreate(
            learner_id=7,
            learning_session_id="session_home_7",
            scene="home_teach",
            scenario_id="home_teach",
            practice_result_id=practice.practice_result_id,
        )
    )

    state = await repository.get_state(started.conversation_id)
    assert state.raw_storage_enabled is True
    assert state.retention_policy.value == "permanent"
    assert state.raw_retention_until is None
    assert state.scenario_data["curriculum_session_id"] == "money-count"
    assert state.scenario_data["practice_result_id"] == practice.practice_result_id
    assert state.current_task_id == "home_teaching"
    assert started.turn.mormi.text == HOME_TEACHING_CATALOG["money-count"].effective_l4_prompt
    assert started.turn.visual.data["curriculum_session_id"] == "money-count"

    after_method = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="f48fd07e-3e4c-4752-873a-54d7e42cb068",
            type="text",
            text=child_text,
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert after_method.turn.status.value == "active"
    assert after_method.turn.input.target_slots == ["answer"]
    assert state.verified_slots == {"rule": True}

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=after_method.turn.turn_id,
            response_id="a6f59963-417a-41eb-8de9-3a497f9aadba",
            type="text",
            text="600원",
        ),
    )

    assert completed.turn.status.value == "completed"
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.text == (
        f"{HOME_TEACHING_CATALOG['money-count'].note_context}에 대해 “{child_text}”라고 배웠어."
    )
    assert completed.turn.note_update.attribution.value == "child"
    transcript = await repository.raw_turns(started.conversation_id)
    assert transcript[0]["question"] == started.turn.mormi.text
    assert transcript[0]["response"] == child_text
    assert transcript[1]["question"] == after_method.turn.mormi.text
    assert transcript[1]["response"] == "600원"
    async with database.sessions() as db:
        answered = (
            await db.execute(select(TurnRecord).where(TurnRecord.turn_id == started.turn.turn_id))
        ).scalar_one()
        assert answered.response_expires_at is None
    await database.dispose()


@pytest.mark.asyncio
async def test_concrete_answer_is_preserved_and_only_the_method_is_asked_next(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/home-partial-answer.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    child_method = "하나, 둘, 셋 하면서 세면 돼"
    analyses = [
        UtteranceAnalysis(
            safety_category=SafetyCategory.NORMAL,
            response_category=ResponseCategory.CORRECT_PARTIAL,
            difficulty_class=DifficultyClass.UNKNOWN,
            claims=[
                SlotClaim(
                    slot_id="answer",
                    value="3",
                    factual=True,
                    evidence_span="3개",
                )
            ],
            confidence=1,
        ),
        UtteranceAnalysis(
            safety_category=SafetyCategory.NORMAL,
            response_category=ResponseCategory.CORRECT_FULL,
            difficulty_class=DifficultyClass.UNKNOWN,
            claims=[
                SlotClaim(
                    slot_id="tracking",
                    # The production classifier may return a reviewed method
                    # alias rather than the slot's canonical value.  The
                    # repository must persist the same canonical value used by
                    # the engine so note provenance remains linkable.
                    value="one_by_one_order",
                    factual=True,
                    evidence_span=child_method,
                )
            ],
            # Even a conflicting model-written candidate must not replace the
            # child's safe, fact-checked wording in the note.
            note_candidate="점을 하나씩 가리키며 마지막 수를 말하면 돼",
            confidence=1,
        ),
    ]
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway(analyses)),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=10,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_number_count_partial",
            practice_result_id="practice_number_count_partial",
            practice_summary={
                "curriculum_session_id": "number-count",
                "skill_id": "number-count",
                "question_count": 5,
                "first_try_correct_count": 2,
            },
        )
    )

    after_answer = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="09de4381-f4cb-46ea-9479-095f2a32a96d",
            type="text",
            text="3개",
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots["answer"] == "3"
    # The first substantive answer was produced independently at L4.  Splitting
    # the remaining method question does not erase that expression credit.
    assert state.expression_level is ExpressionLevel.L4
    assert state.entry_phase is EntryPhase.AWAITING_TARGETED_FOLLOWUP
    assert state.hint_level is HintLevel.H0
    assert after_answer.turn.mormi.text == (
        "아, 세 개구나! 나는 가끔 점을 세다가 헷갈려. 어떻게 세는지 알려주면 안 될까?"
    )
    assert after_answer.turn.input.kind is InputKind.TEXT
    assert after_answer.turn.input.target_slots == ["tracking"]
    assert after_answer.turn.help_card is None

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=after_answer.turn.turn_id,
            response_id="b5fa9e9d-23c2-4258-8505-2c6598ab9e38",
            type="text",
            text=child_method,
        ),
    )
    assert completed.turn.status.value == "completed"
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.text == (
        f"{HOME_TEACHING_CATALOG['number-count'].note_context}에 대해 “{child_method}”라고 배웠어."
    )
    assert child_method in completed.turn.note_update.text
    assert completed.turn.note_update.attribution.value == "child"
    assert completed.turn.note_update.evidence.value == "direct_explanation"
    assert completed.turn.note_update.text != HOME_TEACHING_CATALOG["number-count"].learned_line
    async with database.sessions() as db:
        record = await db.get(ConversationRecord, started.conversation_id)
        assert record is not None
        assert "_child_note_evidence_encrypted" not in record.state_json
        assert record.state_json["child_note_evidence"]["tracking"] == child_method
        tracking_claim = (
            await db.execute(
                select(DialogueClaimRecord).where(DialogueClaimRecord.slot_id == "tracking")
            )
        ).scalar_one()
        assert tracking_claim.validation_status == "verified"
        assert tracking_claim.value_json is True
        assert tracking_claim.newly_verified is True
        note_link = (
            await db.execute(
                select(NoteEvidenceLinkRecord).where(
                    NoteEvidenceLinkRecord.observation_id == tracking_claim.observation_id
                )
            )
        ).scalar_one()
        assert note_link.source_slot_ids_json == ["tracking"]
        star_note_outbox = (
            await db.execute(
                select(OutboxEventRecord).where(
                    OutboxEventRecord.event_type == "mormi.star_note.created"
                )
            )
        ).scalar_one()
        assert star_note_outbox.payload_json["text"] == completed.turn.note_update.text
        assert star_note_outbox.payload_json["evidence_links"] == [
            {
                "observation_id": tracking_claim.observation_id,
                "source_slot_ids": ["tracking"],
            }
        ]
    restored = await repository.get_state(started.conversation_id)
    assert restored.child_note_evidence["tracking"] == child_method
    await database.dispose()


@pytest.mark.asyncio
async def test_safe_child_phrase_can_ground_a_natural_targeted_followup(
    tmp_path: object,
) -> None:
    child_text = "3개야. 차근차근 세어봐"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_PARTIAL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="3",
                factual=True,
                evidence_span="3개야",
            )
        ],
        grounding_span="차근차근 세어봐",
        confidence=1,
    )

    class GroundedSpeakerGateway(FakeGateway):
        speaker_context: SpeakerContext | None = None

        async def speak(self, context: SpeakerContext) -> SpeakerOutput:
            self.speaker_context = context
            return SpeakerOutput(
                text="아, 세 개구나! ‘차근차근 세어봐’는 어떻게 하는 거야?",
                dialogue_act=context.dialogue_act,
                asked_slot_ids=context.required_slot_ids,
                used_verified_slots=["answer"],
                used_child_expression=True,
                used_child_expression_spans=["차근차근 세어봐"],
            )

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/grounded-followup.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    gateway = GroundedSpeakerGateway([analysis])
    service = ConversationService(
        repository,
        ConversationEngine(gateway),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=10,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_grounded_followup",
            practice_result_id="practice_grounded_followup",
            practice_summary={
                "curriculum_session_id": "number-count",
                "skill_id": "number-count",
                "question_count": 5,
                "first_try_correct_count": 5,
            },
        )
    )

    after_answer = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="5d0496e3-2762-4d3a-a2ae-1cbdf49ead06",
            type="text",
            text=child_text,
        ),
    )
    state = await repository.get_state(started.conversation_id)

    assert state.verified_slots["answer"] == "3"
    assert "tracking" not in state.verified_slots
    assert gateway.speaker_context is not None
    assert gateway.speaker_context.child_expression == "차근차근 세어봐"
    assert gateway.speaker_context.verification_policy is SpeakerVerificationPolicy.SEMANTIC
    assert gateway.speaker_context.required_slot_ids == ["tracking"]
    assert after_answer.turn.mormi.text == ("아, 세 개구나! ‘차근차근 세어봐’는 어떻게 하는 거야?")
    assert after_answer.turn.input.target_slots == ["tracking"]
    await database.dispose()


@pytest.mark.asyncio
async def test_number_compare_accepts_counting_reason_without_forcing_pairing(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/compare-own-words.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    child_text = "왼쪽은 세 개고 오른쪽은 다섯 개잖아"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span=child_text,
            ),
            SlotClaim(
                slot_id="reason",
                value="count_comparison",
                factual=True,
                evidence_span=child_text,
            ),
        ],
        note_candidate="양쪽 점을 하나씩 짝지으면 오른쪽이 더 많아",
        confidence=1,
    )
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway([analysis])),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=13,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_compare_own_words",
            practice_result_id="practice_compare_own_words",
            practice_summary={
                "curriculum_session_id": "number-compare",
                "skill_id": "number-compare",
                "question_count": 5,
                "first_try_correct_count": 5,
            },
        )
    )

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="edce6159-a2e3-46b4-a913-0296f0574b10",
            type="text",
            text=child_text,
        ),
    )

    assert completed.turn.status.value == "completed"
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.text == (
        f"{HOME_TEACHING_CATALOG['number-compare'].note_context}에 대해 "
        f"“{child_text}”라고 배웠어. 그래서 오른쪽에 점이 더 많다는 걸 알았어."
    )
    assert child_text in completed.turn.note_update.text
    assert completed.turn.note_update.attribution.value == "child"
    assert "짝" not in completed.turn.note_update.text
    await database.dispose()


@pytest.mark.asyncio
async def test_completion_speaker_cannot_invent_an_unprovided_strategy(
    tmp_path: object,
) -> None:
    child_text = "왼쪽은 세 개고 오른쪽은 다섯 개잖아"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[
            SlotClaim(
                slot_id="answer",
                value="오른쪽",
                factual=True,
                evidence_span=child_text,
            ),
            SlotClaim(
                slot_id="reason",
                value="count_comparison",
                factual=True,
                evidence_span=child_text,
            ),
        ],
        confidence=1,
    )

    class InventingGateway(FakeGateway):
        speak_called = False

        async def speak(self, context: SpeakerContext) -> SpeakerOutput:
            self.speak_called = True
            return SpeakerOutput(text="양쪽 점을 하나씩 짝지으면 알 수 있어!")

    database = Database(f"sqlite+aiosqlite:///{tmp_path}/completion-speaker.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    gateway = InventingGateway([analysis])
    service = ConversationService(
        repository,
        ConversationEngine(gateway),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=14,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_no_invented_completion",
            practice_result_id="practice_no_invented_completion",
            practice_summary={
                "curriculum_session_id": "number-compare",
                "skill_id": "number-compare",
                "question_count": 5,
                "first_try_correct_count": 5,
            },
        )
    )

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="5bc84233-30c5-469e-a426-15f7d3101fb7",
            type="text",
            text=child_text,
        ),
    )

    assert gateway.speak_called is False
    assert "짝" not in completed.turn.mormi.text
    assert completed.turn.mormi.text == "네가 알려준 방법으로 내가 끝까지 해냈어!"
    await database.dispose()


def test_number_compare_declares_multiple_reviewed_valid_explanations() -> None:
    from mormi_api.content import home_teaching_task

    spec = HOME_TEACHING_CATALOG["number-compare"]
    task = home_teaching_task(spec, skill_id=spec.id)

    assert len(spec.valid_explanations) >= 2
    assert all(task.slots["reason"].accepts(value) for value in spec.valid_explanations)


@pytest.mark.asyncio
async def test_home_text_turn_rejects_an_unrelated_choice_payload(tmp_path: object) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/home-input-contract.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=11,
            scene="home_teach",
            scenario_id="home_teach",
            learning_session_id="session_input_contract",
            practice_result_id="practice_input_contract",
            practice_summary={
                "curriculum_session_id": "number-count",
                "skill_id": "number-count",
                "question_count": 5,
                "first_try_correct_count": 3,
            },
        )
    )

    with pytest.raises(ValueError, match="does not match input kind text"):
        await service.respond(
            started.conversation_id,
            ChildResponse(
                turn_id=started.turn.turn_id,
                response_id="406eb34e-0801-4a82-9bd1-538a9ec03364",
                type="choice",
                choice_ids=["answer_0"],
            ),
        )

    restored = await service.snapshot(started.conversation_id)
    assert restored.turn.turn_id == started.turn.turn_id
    await database.dispose()


@pytest.mark.asyncio
async def test_home_teaching_rejects_unknown_curriculum_session(tmp_path: object) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/unknown-home.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="unsupported home curriculum_session_id"):
        await service.create_conversation(
            SessionCreate(
                learner_id=1,
                scene="home_teach",
                scenario_id="home_teach",
                learning_session_id="session_unknown_home",
                practice_result_id="practice_unknown_home",
                practice_summary={
                    "curriculum_session_id": "invented-by-client",
                    "skill_id": "unknown",
                    "question_count": 5,
                    "first_try_correct_count": 3,
                },
            )
        )
    await database.dispose()


@pytest.mark.asyncio
async def test_inline_home_practice_starts_full_turn_and_retries_keep_first_snapshot(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/inline-home-teaching.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )

    request = SessionCreate(
        learner_id=7,
        scene="home_teach",
        scenario_id="home_teach",
        learning_session_id="session_inline_home_7",
        practice_result_id="practice_inline_home_7",
        practice_summary={
            "curriculum_session_id": "money-count",
            "skill_id": "money_count",
            "question_count": 5,
            "first_try_correct_count": 3,
            "wrong_attempt_count": 2,
        },
    )
    started = await service.create_conversation(request)

    assert started.turn.scene.value == "home_teach"
    assert started.turn.scenario_id == "home_teach"
    assert started.turn.task_id == "home_teaching"
    assert started.turn.mormi.text
    assert started.turn.input.kind.value in {"text", "choices", "fill", "joint"}
    assert started.turn.visual.data["curriculum_session_id"] == "money-count"

    # A duplicated completion context can reach the API again after a network
    # retry. The repository's existing practice-result idempotency rule keeps
    # the first snapshot canonical instead of changing this result to another
    # curriculum item.
    retry = SessionCreate(
        learner_id=7,
        scene="home_teach",
        scenario_id="home_teach",
        learning_session_id="session_inline_home_7",
        practice_result_id="practice_inline_home_7",
        practice_summary={
            "curriculum_session_id": "number-count",
            "skill_id": "number_count",
            "question_count": 5,
            "first_try_correct_count": 5,
        },
    )
    retried = await service.create_conversation(retry)
    retry_state = await repository.get_state(retried.conversation_id)
    stored = await repository.get_practice_summary("practice_inline_home_7")

    assert retried.conversation_id == started.conversation_id
    assert retry_state.scenario_data["curriculum_session_id"] == "money-count"
    assert retried.turn.visual.data["curriculum_session_id"] == "money-count"
    assert stored is not None
    assert stored.curriculum_session_id == "money-count"
    assert stored.skill_id == "money_count"
    await database.dispose()


def test_home_teaching_dialogue_copy_may_exceed_fifty_characters() -> None:
    current = HOME_TEACHING_CATALOG["number-count"].model_dump(mode="json")
    l4_question = (
        "나는 가끔 점을 세다가 어디까지 셌는지 헷갈릴 때가 있어... "
        "혹시 네가 세는 방법을 조금 더 자세히 알려줄 수 있어?"
    )
    short_question = (
        "한꺼번에 물어봐서 미안해... 세는 방법은 그다음에 물어볼게. "
        "색칠된 점이 몇 개인지만 먼저 짧게 알려주면 안 될까?"
    )
    current["l4_prompt"] = l4_question
    current["short_prompt"] = short_question

    spec = HomeTeachingSpec.model_validate(current)

    assert all(len(text) > 50 for text in (l4_question, short_question))
    assert spec.l4_prompt == l4_question
    assert spec.short_prompt == short_question
