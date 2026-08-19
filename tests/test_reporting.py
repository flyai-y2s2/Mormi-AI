from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import event

from mormi_api.db import ConversationRecord, Database, NoteRecord, TurnRecord
from mormi_api.llm import ModelUnavailableError
from mormi_api.main import app
from mormi_api.reporting import validate_report_summary
from mormi_api.repository import Repository
from mormi_api.schemas import (
    CompletionOutcome,
    ExpressionLevel,
    HintLevel,
    LearnerProfile,
    ReportFact,
    ReportNarrative,
    ReportSummaryRequest,
    ReportSummaryResponse,
    RetentionPolicy,
    SceneType,
    SessionState,
    SessionStatus,
    SkillProfile,
    utc_now,
)
from mormi_api.security import TextCipher
from mormi_api.settings import Settings


def fact(evidence_id: str, statement: str, *, category: str = "concept") -> ReportFact:
    return ReportFact(evidence_id=evidence_id, category=category, statement=statement)


def summary_request(*, facts: list[ReportFact]) -> ReportSummaryRequest:
    return ReportSummaryRequest(learner_label="학습자", facts=facts)


def summary_response(
    *,
    concept_text: str,
    evidence_refs: list[str] | None = None,
) -> ReportSummaryResponse:
    return ReportSummaryResponse(
        concept_performance=ReportNarrative(
            text=concept_text,
            evidence_refs=evidence_refs or ["domain:money"],
        ),
        explanation_change=ReportNarrative(
            text=concept_text,
            evidence_refs=evidence_refs or ["domain:money"],
        ),
        life_transfer=ReportNarrative(
            text=concept_text,
            evidence_refs=evidence_refs or ["domain:money"],
        ),
        improved_point=ReportNarrative(
            text=concept_text,
            evidence_refs=evidence_refs or ["domain:money"],
        ),
        observe_point=ReportNarrative(
            text=concept_text,
            evidence_refs=evidence_refs or ["domain:money"],
        ),
    )


def five_narrative_response() -> ReportSummaryResponse:
    return ReportSummaryResponse(
        concept_performance=ReportNarrative(
            text="개념 수행은 60%입니다.", evidence_refs=["concept:performance"]
        ),
        explanation_change=ReportNarrative(
            text="설명은 차근차근 세기입니다.", evidence_refs=["explanation:counting"]
        ),
        life_transfer=ReportNarrative(
            text="생활 장면에서 3개를 골랐습니다.", evidence_refs=["life:selection"]
        ),
        improved_point=ReportNarrative(
            text="도움 요청 뒤 혼자 답했습니다.", evidence_refs=["improved:independent"]
        ),
        observe_point=ReportNarrative(
            text="다음 활동도 관찰합니다.", evidence_refs=["observe:next"]
        ),
    )


def five_narrative_request() -> ReportSummaryRequest:
    return summary_request(
        facts=[
            fact("concept:performance", "개념 수행은 60%입니다."),
            fact("explanation:counting", "설명은 차근차근 세기입니다.", category="explanation"),
            fact("life:selection", "생활 장면에서 3개를 골랐습니다.", category="life"),
            fact("improved:independent", "도움 요청 뒤 혼자 답했습니다.", category="improved"),
            fact("observe:next", "다음 활동도 관찰합니다.", category="observe"),
        ]
    )


def test_summary_rejects_unknown_evidence_and_numbers() -> None:
    request = summary_request(facts=[fact("domain:money", "최근 독립 수행률은 60%입니다.")])
    response = summary_response(
        concept_text="최근 독립 수행률은 90%입니다.",
        evidence_refs=["domain:missing"],
    )

    with pytest.raises(ValueError):
        validate_report_summary(request, response)
    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(concept_text="최근 독립 수행률은 90%입니다."),
        )


def test_summary_rejects_diagnosis_and_peer_comparison() -> None:
    request = summary_request(facts=[fact("domain:money", "최근 상태는 관찰 중입니다.")])

    for text in ("경계선 지능이 의심됩니다.", "또래보다 뒤처집니다."):
        with pytest.raises(ValueError):
            validate_report_summary(request, summary_response(concept_text=text))


def test_summary_rejects_quote_not_present_in_referenced_fact() -> None:
    request = summary_request(facts=[fact("domain:money", "최근 상태는 관찰 중입니다.")])

    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(concept_text="‘독립 수행’이라고 말할 수 있습니다."),
        )


@pytest.mark.parametrize(
    "quote",
    [
        ("‘", "’"),
        ("\"", "\""),
        ("『", "』"),
        ("「", "」"),
        ("《", "》"),
        ("〈", "〉"),
    ],
)
def test_summary_rejects_unrecognized_or_unbalanced_quote_delimiters(
    quote: tuple[str, str],
) -> None:
    request = summary_request(facts=[fact("domain:money", "최근 상태는 관찰 중입니다.")])
    opening, closing = quote

    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(concept_text=f"{opening}독립 수행{closing}이라고 말할 수 있습니다."),
        )
    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(concept_text=f"{opening}최근 상태는 관찰 중입니다."),
        )
    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(concept_text=f"최근 상태는 관찰 중입니다.{closing}"),
        )


@pytest.mark.parametrize(
    "text",
    [
        "【최근 상태는 관찰 중입니다.】",
        "⟨최근 상태는 관찰 중입니다.⟩",
        "〈최근 상태는 관찰 중입니다.",
    ],
)
def test_summary_rejects_unsupported_or_unmatched_quote_like_punctuation(text: str) -> None:
    request = summary_request(facts=[fact("domain:money", "최근 상태는 관찰 중입니다.")])

    with pytest.raises(ValueError):
        validate_report_summary(request, summary_response(concept_text=text))


@pytest.mark.parametrize(
    ("fact_statement", "summary_text"),
    [
        ("최근 금액은 5천 원입니다.", "최근 금액은 5만 원입니다."),
        ("최근 성공률은 60%입니다.", "최근 성공률은 60입니다."),
        ("최근 선택은 3개입니다.", "최근 선택은 3명입니다."),
    ],
)
def test_summary_rejects_changed_numeric_magnitude_percent_or_unit(
    fact_statement: str,
    summary_text: str,
) -> None:
    request = summary_request(facts=[fact("domain:money", fact_statement)])

    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(concept_text=summary_text),
        )


@pytest.mark.parametrize(
    "text",
    [
        "반 친구보다 느립니다.",
        "동년배보다 뒤처집니다.",
        "학급 평균보다 낮습니다.",
        "반 평균보다 빠릅니다.",
        "상위권입니다.",
        "진단이 필요합니다.",
        "약물 처방을 추천합니다.",
        "심리 치료를 권장합니다.",
    ],
)
def test_summary_rejects_peer_ranking_and_medical_language(text: str) -> None:
    request = summary_request(facts=[fact("domain:money", "최근 상태는 관찰 중입니다.")])

    with pytest.raises(ValueError):
        validate_report_summary(request, summary_response(concept_text=text))


@pytest.mark.parametrize("statement", ["친구보다 잘합니다.", "약을 복용합니다."])
def test_summary_rejects_forbidden_language_even_when_it_is_evidence(statement: str) -> None:
    request = summary_request(facts=[fact("domain:money", statement)])

    with pytest.raises(ValueError):
        validate_report_summary(request, summary_response(concept_text=statement))


@pytest.mark.parametrize(
    "statement",
    ["약을 복용합니다.", "약물 치료를 받습니다.", "투약이 필요합니다.", "처방을 권장합니다."],
)
def test_summary_retains_contextual_medication_rejection(statement: str) -> None:
    request = summary_request(facts=[fact("domain:money", statement)])

    with pytest.raises(ValueError):
        validate_report_summary(request, summary_response(concept_text=statement))


def test_summary_rejects_invented_cause_or_interpretation() -> None:
    request = summary_request(facts=[fact("domain:money", "독립 수행률은 60%입니다.")])

    for text in ("독립 수행률은 연습 덕분에 60%입니다.", "독립 수행률은 이해가 좋아서 60%입니다."):
        with pytest.raises(ValueError):
            validate_report_summary(request, summary_response(concept_text=text))


def test_summary_accepts_independently_grounded_five_narratives() -> None:
    request = five_narrative_request()
    response = five_narrative_response()

    assert validate_report_summary(request, response) == response


@pytest.mark.parametrize(
    "statement",
    [
        "개념 수행(기초)은 60%입니다.",
        "개념 수행[기초]은 60%입니다.",
        "약속을 지켰습니다.",
    ],
)
def test_summary_accepts_exact_nonquote_or_nonmedical_evidence(statement: str) -> None:
    request = summary_request(facts=[fact("concept:performance", statement)])
    response = summary_response(
        concept_text=statement,
        evidence_refs=["concept:performance"],
    )

    assert validate_report_summary(request, response) == response


def test_summary_rejects_reporting_word_paraphrase() -> None:
    request = summary_request(facts=[fact("concept:performance", "개념 수행은 60%입니다.")])

    response = summary_response(
        concept_text="최근 개념 수행은 60%입니다.",
        evidence_refs=["concept:performance"],
    )

    with pytest.raises(ValueError):
        validate_report_summary(request, response)


def test_summary_rejects_relation_swapped_from_referenced_fact() -> None:
    request = summary_request(
        facts=[
            fact(
                "concept:independence",
                "문제 해결은 혼자 했습니다. 설명은 도움을 받아 했습니다.",
            )
        ]
    )

    with pytest.raises(ValueError):
        validate_report_summary(
            request,
            summary_response(
                concept_text="도움을 받아 문제 해결을 했습니다.",
                evidence_refs=["concept:independence"],
            ),
        )


def test_summary_accepts_exact_ordered_concatenation_of_referenced_facts() -> None:
    first = "문제 해결은 혼자 했습니다."
    second = "설명은 도움을 받아 했습니다."
    request = summary_request(
        facts=[
            fact("concept:independence", first),
            fact("explanation:support", second, category="explanation"),
        ]
    )
    response = summary_response(
        concept_text=f"{first} {second}",
        evidence_refs=["concept:independence", "explanation:support"],
    )

    assert validate_report_summary(request, response) == response


def test_summary_request_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        summary_request(
            facts=[
                fact("concept:performance", "개념 수행은 60%입니다."),
                fact("concept:performance", "다른 사실입니다."),
            ]
        )


def test_summary_route_returns_sanitized_validation_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UngroundedGateway:
        async def summarize_report(self, body: ReportSummaryRequest) -> ReportSummaryResponse:
            return summary_response(concept_text="아이의 비밀 90% 결과입니다.")

    monkeypatch.setattr(
        app.state,
        "settings",
        Settings(service_api_key="shared-secret"),
        raising=False,
    )
    monkeypatch.setattr(app.state, "gateway", UngroundedGateway(), raising=False)

    response = TestClient(app).post(
        "/v1/internal/report-summaries",
        headers={"X-Mormi-Service-Key": "shared-secret"},
        json={
            "learner_label": "학습자",
            "facts": [
                {
                    "evidence_id": "domain:money",
                    "category": "concept",
                    "statement": "최근 독립 수행률은 60%입니다.",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {"code": "report_summary_ungrounded", "issues": []}
    }


def test_summary_route_returns_sanitized_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableGateway:
        async def summarize_report(self, body: ReportSummaryRequest) -> ReportSummaryResponse:
            raise ModelUnavailableError("model_connection_failed")

    monkeypatch.setattr(
        app.state,
        "settings",
        Settings(service_api_key="shared-secret"),
        raising=False,
    )
    monkeypatch.setattr(app.state, "gateway", UnavailableGateway(), raising=False)

    response = TestClient(app).post(
        "/v1/internal/report-summaries",
        headers={"X-Mormi-Service-Key": "shared-secret"},
        json={
            "learner_label": "학습자",
            "facts": [
                {
                    "evidence_id": "domain:money",
                    "category": "concept",
                    "statement": "최근 독립 수행률은 60%입니다.",
                }
            ],
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {"code": "model_connection_failed", "issues": []}
    }


async def reporting_repository(tmp_path: object) -> Repository:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/reporting.db")
    await database.create_schema()
    return Repository(database, TextCipher("test-encryption-key"))


async def seed_completed_conversation(
    repository: Repository,
    *,
    learner_id: int,
    response_text: str,
    raw_storage_enabled: bool = True,
    retention_policy: RetentionPolicy = RetentionPolicy.PERMANENT,
    raw_retention_until: datetime | None = None,
) -> str:
    conversation_id = f"conversation_{learner_id}"
    now = utc_now()
    state = SessionState(
        conversation_id=conversation_id,
        learner_id=learner_id,
        learning_session_id=f"session_{learner_id}",
        scene=SceneType.CAFE,
        scenario_id="cafe_queue_demo",
        task_ids=["compare_quantity_in_context"],
        expression_level=ExpressionLevel.L2,
        status=SessionStatus.COMPLETED,
        completion_outcome=CompletionOutcome.TAUGHT,
        teach_reward_eligible=True,
        verified_slots={"fewer": "left"},
        task_max_hint=HintLevel.H1,
        raw_storage_enabled=raw_storage_enabled,
        retention_policy=retention_policy,
        raw_retention_until=raw_retention_until,
        created_at=now,
        updated_at=now,
    )
    async with repository.database.sessions() as db:
        db.add(
            ConversationRecord(
                conversation_id=conversation_id,
                learner_id=learner_id,
                learning_session_id=state.learning_session_id,
                scene=state.scene.value,
                scenario_id=state.scenario_id,
                state_json=state.model_dump(mode="json"),
                state_version=state.state_version,
                status=state.status.value,
                raw_retention_until=state.raw_retention_until,
                created_at=now,
                updated_at=now,
            )
        )
        db.add(
            TurnRecord(
                turn_id=f"turn_{learner_id}",
                conversation_id=conversation_id,
                task_id="compare_quantity_in_context",
                state_version=state.state_version,
                turn_contract={
                    "pedagogy": {
                        "expression_level": "L2",
                        "hint_level": "H1",
                        "subgoal_id": "compare",
                        "verified_slots": {"fewer": "left"},
                    }
                },
                response_id=f"response_{learner_id}",
                response_type="text",
                response_raw_encrypted=repository.cipher.encrypt(response_text),
                response_category="correct_full",
                expression_level="L2",
                hint_level="H1",
                created_at=now,
            )
        )
        await db.commit()
    return conversation_id


async def seed_direct_note(
    repository: Repository,
    *,
    conversation_id: str,
    learner_id: int,
    text: str,
) -> None:
    async with repository.database.sessions() as db:
        db.add(
            NoteRecord(
                note_id=f"note_{learner_id}",
                conversation_id=conversation_id,
                learner_id=learner_id,
                skill_id="compare_quantity_in_context",
                text=text,
                attribution="child",
                evidence="direct_explanation",
                attribution_label="아이의 직접 설명",
                active=True,
                created_at=utc_now(),
            )
        )
        await db.commit()


@pytest.mark.asyncio
async def test_report_evidence_returns_only_the_requested_learner(tmp_path: object) -> None:
    repository = await reporting_repository(tmp_path)
    first = await seed_completed_conversation(
        repository, learner_id=11, response_text="두 개가 더 적어요"
    )
    await seed_completed_conversation(repository, learner_id=12, response_text="다섯 개예요")

    evidence = await repository.report_evidence(11, include_raw=True)

    assert [row.conversation_id for row in evidence.conversations] == [first]
    assert evidence.conversations[0].turns[0].response == "두 개가 더 적어요"
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_omits_raw_text_when_not_allowed(tmp_path: object) -> None:
    repository = await reporting_repository(tmp_path)
    await seed_completed_conversation(repository, learner_id=11, response_text="두 개가 더 적어요")

    evidence = await repository.report_evidence(11, include_raw=False)

    assert evidence.conversations[0].turns[0].response is None
    assert evidence.conversations[0].turns[0].response_category == "correct_full"
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_keeps_structured_turns_when_raw_key_is_unavailable(
    tmp_path: object,
) -> None:
    repository = await reporting_repository(tmp_path)
    repository_without_key = Repository(repository.database, TextCipher(None))
    await seed_completed_conversation(
        repository_without_key,
        learner_id=11,
        response_text="두 개가 더 적어요",
    )
    await repository_without_key.save_profile(
        LearnerProfile(
            learner_id=11,
            skills={
                "compare_quantity_in_context": SkillProfile(
                    skill_id="compare_quantity_in_context",
                    concept_mastery=0.75,
                )
            },
        )
    )

    evidence = await repository_without_key.report_evidence(11, include_raw=True)

    conversation = evidence.conversations[0]
    turn = conversation.turns[0]
    assert turn.response is None
    assert turn.response_category == "correct_full"
    assert turn.expression_level is ExpressionLevel.L2
    assert turn.hint_level is HintLevel.H1
    assert turn.pedagogy is not None
    assert conversation.verified_slots
    assert conversation.task_max_hint is HintLevel.H1
    assert [skill.skill_id for skill in evidence.skills] == ["compare_quantity_in_context"]
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_omits_expired_raw_text_without_mutating_the_record(
    tmp_path: object,
) -> None:
    repository = await reporting_repository(tmp_path)
    await seed_completed_conversation(
        repository,
        learner_id=11,
        response_text="두 개가 더 적어요",
        retention_policy=RetentionPolicy.DAYS_30,
        raw_retention_until=utc_now() - timedelta(seconds=1),
    )

    evidence = await repository.report_evidence(11, include_raw=True)

    assert evidence.conversations[0].turns[0].response is None
    async with repository.database.sessions() as db:
        stored = await db.get(TurnRecord, 1)
        assert stored is not None
        assert stored.response_raw_encrypted is not None
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_omits_raw_text_when_stored_state_has_no_consent(
    tmp_path: object,
) -> None:
    repository = await reporting_repository(tmp_path)
    await seed_completed_conversation(
        repository,
        learner_id=11,
        response_text="두 개가 더 적어요",
        raw_storage_enabled=False,
        retention_policy=RetentionPolicy.NO_RAW,
    )

    evidence = await repository.report_evidence(11, include_raw=True)

    assert evidence.conversations[0].turns[0].response is None
    assert evidence.conversations[0].turns[0].response_category == "correct_full"
    await repository.database.dispose()


@pytest.mark.parametrize(
    (
        "include_raw",
        "raw_storage_enabled",
        "retention_policy",
        "retention_expired",
        "expected_turn_response",
    ),
    [
        pytest.param(
            False,
            True,
            RetentionPolicy.PERMANENT,
            False,
            None,
            id="raw-not-requested",
        ),
        pytest.param(
            True,
            False,
            RetentionPolicy.NO_RAW,
            False,
            None,
            id="raw-consent-revoked",
        ),
        pytest.param(
            True,
            True,
            RetentionPolicy.DAYS_30,
            True,
            None,
            id="raw-retention-expired",
        ),
        pytest.param(
            True,
            True,
            RetentionPolicy.PERMANENT,
            False,
            "양쪽 점을 하나씩 짝지으면 오른쪽이 더 많아",
            id="raw-requested-and-retained",
        ),
    ],
)
@pytest.mark.asyncio
async def test_report_evidence_never_returns_direct_note_wording(
    tmp_path: object,
    *,
    include_raw: bool,
    raw_storage_enabled: bool,
    retention_policy: RetentionPolicy,
    retention_expired: bool,
    expected_turn_response: str | None,
) -> None:
    repository = await reporting_repository(tmp_path)
    child_wording = "양쪽 점을 하나씩 짝지으면 오른쪽이 더 많아"
    note_text = f"점의 수를 비교하는 방법에 대해 “{child_wording}”라고 배웠어."
    conversation_id = await seed_completed_conversation(
        repository,
        learner_id=11,
        response_text=child_wording,
        raw_storage_enabled=raw_storage_enabled,
        retention_policy=retention_policy,
        raw_retention_until=utc_now() - timedelta(seconds=1) if retention_expired else None,
    )
    await seed_direct_note(
        repository,
        conversation_id=conversation_id,
        learner_id=11,
        text=note_text,
    )
    await repository.save_profile(
        LearnerProfile(
            learner_id=11,
            skills={
                "compare_quantity_in_context": SkillProfile(
                    skill_id="compare_quantity_in_context",
                    concept_mastery=0.75,
                )
            },
        )
    )

    evidence = await repository.report_evidence(11, include_raw=include_raw)

    assert evidence.notes == []
    assert note_text not in evidence.model_dump_json()
    assert evidence.conversations[0].turns[0].response == expected_turn_response
    assert evidence.conversations[0].turns[0].response_category == "correct_full"
    assert [skill.skill_id for skill in evidence.skills] == ["compare_quantity_in_context"]
    assert (await repository.list_notes(11))[0].text == note_text
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_is_read_only_and_does_not_query_notes(tmp_path: object) -> None:
    repository = await reporting_repository(tmp_path)
    child_wording = "두 줄을 하나씩 짝지어 보면 왼쪽이 더 적어"
    conversation_id = await seed_completed_conversation(
        repository,
        learner_id=11,
        response_text=child_wording,
    )
    await seed_direct_note(
        repository,
        conversation_id=conversation_id,
        learner_id=11,
        text=f"점의 수를 비교하는 방법에 대해 “{child_wording}”라고 배웠어.",
    )
    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(repository.database.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        evidence = await repository.report_evidence(11, include_raw=True)
    finally:
        event.remove(
            repository.database.engine.sync_engine,
            "before_cursor_execute",
            record_statement,
        )

    assert evidence.notes == []
    assert statements
    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
    assert all("FROM notes" not in statement for statement in statements)
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_route_requires_the_shared_key_and_isolates_the_learner(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = await reporting_repository(tmp_path)
    first = await seed_completed_conversation(
        repository, learner_id=11, response_text="두 개가 더 적어요"
    )
    await seed_completed_conversation(repository, learner_id=12, response_text="다섯 개예요")
    monkeypatch.setattr(
        app.state,
        "settings",
        Settings(service_api_key="shared-secret"),
        raising=False,
    )
    monkeypatch.setattr(app.state, "repository", repository, raising=False)

    client = TestClient(app)
    rejected = client.get(
        "/v1/internal/learners/11/report-evidence?include_raw=true",
        headers={"X-Mormi-Service-Key": "different-secret"},
    )
    accepted = client.get(
        "/v1/internal/learners/11/report-evidence?include_raw=true",
        headers={"X-Mormi-Service-Key": "shared-secret"},
    )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["learner_id"] == 11
    assert [row["conversation_id"] for row in accepted.json()["conversations"]] == [first]
    await repository.database.dispose()


@pytest.mark.asyncio
async def test_report_evidence_route_fails_closed_when_the_service_key_is_not_configured(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = await reporting_repository(tmp_path)
    monkeypatch.setattr(app.state, "settings", Settings(service_api_key=None), raising=False)
    monkeypatch.setattr(app.state, "repository", repository, raising=False)

    response = TestClient(app).get(
        "/v1/internal/learners/11/report-evidence?include_raw=false",
        headers={"X-Mormi-Service-Key": "shared-secret"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "service_key_not_configured", "issues": []}}
    await repository.database.dispose()
