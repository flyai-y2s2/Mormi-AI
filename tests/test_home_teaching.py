from __future__ import annotations

import pytest
from conftest import FakeGateway

from mormi_api.content import HOME_TEACHING_CATALOG
from mormi_api.db import Database
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import (
    ChildResponse,
    DifficultyClass,
    PracticeResult,
    ResponseCategory,
    SafetyCategory,
    SessionCreate,
    SlotClaim,
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


def test_home_catalog_covers_current_frontend_curriculum() -> None:
    assert set(HOME_TEACHING_CATALOG) == CURRENT_FRONTEND_HOME_SESSION_IDS
    assert all(len(spec.misconception_prompt) <= 50 for spec in HOME_TEACHING_CATALOG.values())


@pytest.mark.asyncio
async def test_saved_practice_result_generates_home_scenario_and_stores_raw_turn(
    tmp_path: object,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/home-teaching.db")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    expected_rule = HOME_TEACHING_CATALOG["money-count"].learned_line
    child_text = "돈의 개수가 아니라 적힌 값을 모두 더해야 해"
    analysis = UtteranceAnalysis(
        safety_category=SafetyCategory.NORMAL,
        response_category=ResponseCategory.CORRECT_FULL,
        difficulty_class=DifficultyClass.UNKNOWN,
        claims=[SlotClaim(slot_id="rule", value=expected_rule, factual=True)],
        note_candidate=child_text,
        confidence=1,
    )
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway([analysis])),  # type: ignore[arg-type]
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
            conversation_storage_consent=True,
            retention_policy="30_days",
        )
    )

    state = await repository.get_state(started.conversation_id)
    assert state.scenario_data["curriculum_session_id"] == "money-count"
    assert state.scenario_data["practice_result_id"] == practice.practice_result_id
    assert state.current_task_id == "home_teaching"
    assert started.turn.mormi.text == HOME_TEACHING_CATALOG["money-count"].misconception_prompt
    assert started.turn.visual.data["curriculum_session_id"] == "money-count"

    completed = await service.respond(
        started.conversation_id,
        ChildResponse(
            turn_id=started.turn.turn_id,
            response_id="f48fd07e-3e4c-4752-873a-54d7e42cb068",
            type="text",
            text=child_text,
        ),
    )

    assert completed.turn.status.value == "completed"
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.text == child_text
    assert completed.turn.note_update.attribution.value == "child"
    transcript = await repository.raw_turns(started.conversation_id)
    assert transcript[0]["question"] == started.turn.mormi.text
    assert transcript[0]["response"] == child_text
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

    assert retry_state.scenario_data["curriculum_session_id"] == "money-count"
    assert retried.turn.visual.data["curriculum_session_id"] == "money-count"
    assert stored is not None
    assert stored.curriculum_session_id == "money-count"
    assert stored.skill_id == "money_count"
    await database.dispose()
