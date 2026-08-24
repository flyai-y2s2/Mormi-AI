# 주간 발화 사다리 분석 에이전트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 파인튜닝된 발화 단계 모델의 턴별 예측과 BE의 주간 학습 통계를 LangGraph로 집계해 근거가 있는 다음 주 발화 단계 추천을 생성한다.

**Architecture:** 내부 전용 요청이 주차·skill별 정답 통계를 전달하면 AI 서비스가 보존 가능한 자기 발화 원장을 조회하고 모델 추론을 수행한다. `collect → predict → aggregate → recommend → validate` 그래프는 추천을 한 단계로 제한하며 모든 결과에 모델·정책 버전과 근거 turn ID를 남긴다.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, Pydantic v2, SQLAlchemy/Alembic, ladder model artifact, pytest

**Spec:** `docs/superpowers/specs/2026-08-23-ladder-analysis-agent-design.md`

## Global Constraints

- 분석 API는 기존 internal reporting service key를 요구한다.
- 한 주의 승인 단계는 분석 중 바꾸지 않는다.
- 모델 오류나 10개 미만의 근거는 승급으로 대체하지 않고 `INSUFFICIENT_EVIDENCE`가 된다.
- 정답률 90%, 독립 수행률 70%, 다음 단계 이상 예측률 70%를 모두 만족해야 한 단계 승급을 추천한다.
- 추천 숫자와 단계는 결정형 코드가 계산하며 생성형 LLM이 만들지 않는다.
- 안전·인식 실패·시스템 오류와 retention이 끝난 원문은 분모에서 제외한다.

---

### Task 1: 주간 분석 API 계약

**Files:**
- Modify: `src/mormi_api/ladder_analysis/contracts.py`
- Test: `tests/test_weekly_ladder_contracts.py`

**Interfaces:**
- Produces: `WeeklySkillMetrics`, `WeeklyLadderAnalysisRequest`, `WeeklyLadderRecommendation`, `WeeklyLadderAnalysisResponse`, `RecommendationAction`

- [ ] **Step 1: Write failing validation tests**

```python
def test_week_must_be_seven_days() -> None:
    with pytest.raises(ValidationError, match="seven days"):
        WeeklyLadderAnalysisRequest(
            learner_id=1,
            week_start=date(2026, 8, 17),
            week_end=date(2026, 8, 22),
            skills=[],
        )


def test_accuracy_requires_matching_counts() -> None:
    with pytest.raises(ValidationError):
        WeeklySkillMetrics(
            skill_id="money-count",
            current_level="L2",
            eligible_count=10,
            correct_count=11,
            independent_count=7,
        )
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_weekly_ladder_contracts.py -q`

Expected: FAIL because weekly contracts are absent.

- [ ] **Step 3: Implement exact contracts**

`WeeklySkillMetrics` contains `skill_id`, `current_level`, `eligible_count`,
`correct_count`, `independent_count`, and excluded safety/system counts. Derived properties
calculate accuracy and independent rate without accepting client-supplied ratios.

`WeeklyLadderRecommendation` contains the exact response fields in design section 7.2 and
never includes child utterance text.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_weekly_ladder_contracts.py -q`

Expected: PASS.

```bash
git add src/mormi_api/ladder_analysis/contracts.py tests/test_weekly_ladder_contracts.py
git commit -m "feat: define weekly ladder analysis contracts"
```

### Task 2: 분석 실행과 턴 예측 감사 저장소

**Files:**
- Modify: `src/mormi_api/db.py`
- Modify: `src/mormi_api/repository.py`
- Create: `alembic/versions/20260823_01_ladder_analysis_runs.py`
- Test: `tests/test_ladder_analysis_repository.py`

**Interfaces:**
- Produces: `LadderAnalysisRunRecord`, `LadderTurnPredictionRecord`, `Repository.create_ladder_analysis_run(...)`, `Repository.complete_ladder_analysis_run(...)`, `Repository.list_ladder_evidence(...)`

- [ ] **Step 1: Write failing repository tests**

```python
async def test_completed_run_is_idempotent_for_week_skill_and_versions(repository) -> None:
    first = await repository.create_ladder_analysis_run(request_fixture())
    second = await repository.create_ladder_analysis_run(request_fixture())
    assert second.analysis_id == first.analysis_id


async def test_prediction_audit_stores_no_raw_utterance(repository) -> None:
    await repository.complete_ladder_analysis_run(run_fixture(), [prediction_fixture()])
    row = await repository.get_ladder_prediction("turn_1", "ladder-klue-v1")
    assert not hasattr(row, "child_utterance")
```

- [ ] **Step 2: Run and verify schema failure**

Run: `python -m pytest tests/test_ladder_analysis_repository.py -q`

Expected: FAIL because records and repository methods are absent.

- [ ] **Step 3: Add migration and ORM records**

Create `ladder_analysis_runs` with a unique key over learner, week start, skill, model version,
and policy version. Store counts, ratios, action, current/recommended level, status and timestamps.

Create `ladder_turn_predictions` with analysis ID, source turn ID, predicted level, four numeric
scores and model version. Do not add raw question or child utterance columns.

- [ ] **Step 4: Implement repository methods and retention-aware evidence query**

`list_ladder_evidence` joins conversation, turn and observation records, enforces learner and
date bounds, and returns only turns whose raw text is available under the existing retention
policy. Decode in memory, run inference, then discard text before persistence.

- [ ] **Step 5: Run focused tests and migration check**

Run: `python -m pytest tests/test_ladder_analysis_repository.py -q`

Expected: PASS.

Run: `python -m alembic upgrade head`

Expected: new tables and unique constraints are created on a disposable SQLite/PostgreSQL test DB.

- [ ] **Step 6: Commit**

```bash
git add src/mormi_api/db.py src/mormi_api/repository.py alembic/versions/20260823_01_ladder_analysis_runs.py tests/test_ladder_analysis_repository.py
git commit -m "feat: persist ladder analysis audits"
```

### Task 3: 결정형 주간 추천 정책

**Files:**
- Create: `src/mormi_api/ladder_analysis/policy.py`
- Test: `tests/test_ladder_recommendation_policy.py`

**Interfaces:**
- Produces: `recommend_weekly_level(metrics, predictions, policy=POLICY_V1) -> WeeklyLadderRecommendation`

- [ ] **Step 1: Write the policy matrix as failing parameterized tests**

```python
@pytest.mark.parametrize(
    ("eligible", "correct", "independent", "next_predictions", "expected"),
    [
        (9, 9, 9, 9, "INSUFFICIENT_EVIDENCE"),
        (10, 8, 8, 8, "HOLD"),
        (10, 9, 6, 8, "HOLD"),
        (10, 9, 7, 6, "HOLD"),
        (10, 9, 7, 7, "PROMOTE"),
    ],
)
def test_promotion_gate(eligible, correct, independent, next_predictions, expected) -> None:
    result = recommend_weekly_level(
        metrics_fixture(eligible, correct, independent),
        predictions_fixture(eligible, next_predictions),
    )
    assert result.action.value == expected
```

Add tests that L0 promotes only to L2, L2 only to L3, L3 only to L4, and L4 never promotes.

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_ladder_recommendation_policy.py -q`

Expected: FAIL because the policy does not exist.

- [ ] **Step 3: Implement versioned policy constants**

```python
POLICY_V1 = WeeklyLadderPolicy(
    version="weekly-ladder-v1",
    minimum_eligible=10,
    minimum_accuracy=Decimal("0.90"),
    minimum_independent_rate=Decimal("0.70"),
    minimum_next_level_rate=Decimal("0.70"),
    review_down_rate=Decimal("0.60"),
)
```

Calculate every ratio from integer counts, include the evidence turn IDs contributing to the
next-level rate, and reject a recommendation that moves more than one canonical step.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_ladder_recommendation_policy.py -q`

Expected: PASS.

```bash
git add src/mormi_api/ladder_analysis/policy.py tests/test_ladder_recommendation_policy.py
git commit -m "feat: add weekly ladder recommendation policy"
```

### Task 4: LangGraph 분석 에이전트

**Files:**
- Create: `src/mormi_api/ladder_analysis/agent.py`
- Test: `tests/test_ladder_analysis_agent.py`

**Interfaces:**
- Consumes: `Repository`, `LadderPredictor`, `recommend_weekly_level`
- Produces: `WeeklyLadderAnalysisAgent.run(request) -> WeeklyLadderAnalysisResponse`

- [ ] **Step 1: Write failing graph-path tests**

```python
async def test_agent_runs_all_five_nodes_in_order(agent, progress_spy) -> None:
    response = await agent.run(promotable_request())
    assert progress_spy.names == [
        "collect_evidence",
        "predict_utterances",
        "aggregate_week",
        "recommend_level",
        "validate_recommendation",
    ]
    assert response.recommendations[0].action is RecommendationAction.PROMOTE


async def test_model_failure_returns_unavailable_without_rule_substitution(agent) -> None:
    agent.predictor.raise_on_predict = ModelArtifactError("missing")
    response = await agent.run(promotable_request())
    assert response.status == "analysis_unavailable"
    assert response.recommendations == []
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/test_ladder_analysis_agent.py -q`

Expected: FAIL because agent is absent.

- [ ] **Step 3: Implement the state and five nodes**

Use a typed graph state containing request, decoded ephemeral evidence, predictions, aggregates,
recommendations and issues. Do not use a LangGraph checkpointer for child raw text. Persist only
the prediction audit and validated recommendation after the graph completes.

- [ ] **Step 4: Enforce output validation**

`validate_recommendation` must recalculate all ratios, canonicalize legacy L1, verify one-step
movement, ensure every evidence turn belongs to the learner/week/skill and replace any invalid
result with `analysis_unavailable`; it must not repair numbers supplied by prior nodes.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_ladder_analysis_agent.py tests/test_ladder_recommendation_policy.py -q`

Expected: PASS.

```bash
git add src/mormi_api/ladder_analysis/agent.py tests/test_ladder_analysis_agent.py
git commit -m "feat: orchestrate weekly ladder analysis"
```

### Task 5: 설정과 내부 분석 API

**Files:**
- Modify: `src/mormi_api/settings.py`
- Modify: `src/mormi_api/main.py`
- Modify: `.env.example`
- Modify: `docs/API_SPEC.md`
- Test: `tests/test_ladder_analysis_api.py`
- Create: `tests/test_settings.py`

**Interfaces:**
- Produces: `POST /v1/internal/ladder-analyses`

- [ ] **Step 1: Write failing auth, disabled and success tests**

```python
async def test_ladder_analysis_requires_internal_reporting_key(client) -> None:
    response = await client.post("/v1/internal/ladder-analyses", json=request_json())
    assert response.status_code == 401


async def test_ladder_analysis_fails_closed_without_artifact(authorized_client) -> None:
    response = await authorized_client.post(
        "/v1/internal/ladder-analyses", json=request_json()
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "ladder_analysis_unavailable"
```

- [ ] **Step 2: Run and verify route absence**

Run: `python -m pytest tests/test_ladder_analysis_api.py tests/test_settings.py -q`

Expected: FAIL because settings and route are absent.

- [ ] **Step 3: Add settings**

Add `ladder_analysis_enabled: bool = False`, `ladder_model_path: Path | None`,
`ladder_model_version: str`, and strict production validation: enabled requires an absolute,
existing artifact path and internal reporting service key.

- [ ] **Step 4: Wire one predictor and agent during lifespan**

Load and verify the artifact once at startup when enabled. The POST handler authenticates with
`InternalReportingAuth`, delegates to the agent, maps unavailable analysis to 503, and never
returns raw utterances.

- [ ] **Step 5: Document the exact request/response and run checks**

Run: `python -m pytest tests/test_ladder_analysis_api.py tests/test_settings.py -q`

Expected: PASS.

Run: `python -m pytest tests/test_reporting.py tests/test_ladder_analysis_repository.py tests/test_ladder_analysis_agent.py tests/test_ladder_analysis_api.py -q`

Expected: PASS.

Run: `python -m ruff check src/mormi_api tests`

Expected: no new violations.

- [ ] **Step 6: Commit**

```bash
git add src/mormi_api/settings.py src/mormi_api/main.py .env.example docs/API_SPEC.md tests/test_ladder_analysis_api.py tests/test_settings.py
git commit -m "feat: expose internal ladder analysis API"
```

### Task 6: AI 전체 회귀 검증

**Files:**
- Modify only if a regression is proven by tests from prior tasks.

**Interfaces:**
- Produces: a reviewed AI commit range ready for BE integration

- [ ] **Step 1: Run the complete suite**

Run: `python -m pytest -q`

Expected: all tests PASS; existing deprecation warnings may remain documented but no new warning is accepted.

- [ ] **Step 2: Run static checks**

Run: `python -m ruff check src tests scripts`

Expected: clean.

Run: `python -m mypy src/mormi_api`

Expected: success.

- [ ] **Step 3: Review privacy boundaries**

Search the new API responses, DB records, logs and test snapshots for `child_utterance`,
`mormi_question`, learner names and actual IDs. Raw text may exist only in ephemeral graph state
and the already-authorized source store.

- [ ] **Step 4: Commit any evidence-only documentation update**

If no code changed during verification, do not create an empty commit. Record the exact commands,
test counts, artifact hash and known prototype limitations in the PR description.
