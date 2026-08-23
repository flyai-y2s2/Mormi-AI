# 교사 발화 단계 승인 및 적용 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AI의 주간 발화 단계 추천을 진단 리포트에 표시하고 교사가 승인한 단계만 다음 주 학습 시작 단계로 적용한다.

**Architecture:** Spring BE가 정답 통계를 계산하고 AI 내부 분석 API를 호출한 뒤 추천과 감사 정보를 자기 DB에 저장한다. 교사 결정과 유효 주차의 단일 기준은 BE이며, 대화 시작 시 BE가 서버 간 요청으로 승인 단계를 전달하고 AI가 그 단계만 우선 적용한다.

**Tech Stack:** Java 21, Spring Boot, PostgreSQL/Flyway, FastAPI/Pydantic, Next.js/React/TypeScript, Node test runner, Gradle

**Spec:** `Mormi-AI/docs/superpowers/specs/2026-08-23-ladder-analysis-agent-design.md`

## Global Constraints

- 추천은 자동 적용하지 않는다. `APPROVED` 결정과 다음 주 유효 시작일이 모두 있어야 한다.
- 한 추천은 한 단계만 올리거나 내릴 수 있다.
- 브라우저는 learner 소유권, 현재 단계, 추천 숫자, 모델 버전과 유효 시작일을 직접 정하지 않는다.
- AI 분석 실패는 리포트의 나머지 내용을 실패시키지 않고 `분석 기록 필요`로 표시한다.
- 현재 주의 승인 단계는 변경하지 않고 승인 다음 주 월요일부터 적용한다.
- 기존 홈 가르치기의 L4 시작은 승인 단계가 없을 때의 기본값으로 유지한다.

---

### Task 1: BE의 AI 분석 클라이언트 계약

**Files:**
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/DiagnosticReportDtos.java`
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/ReportAiClient.java`
- Test: `Mormi-BE/src/test/java/com/mormi/backend/report/ReportAiClientTest.java`

**Interfaces:**
- Produces: `ReportAiClient.analyzeLadder(LadderAnalysisRequest) -> LadderAnalysisResponse`

- [ ] **Step 1: Write failing request serialization and response tests**

```java
@Test
void sendsOnlyAggregatedCorrectnessAndReceivesVersionedRecommendation() {
    LadderAnalysisResponse response = client.analyzeLadder(requestFor(
            "money-count", "L2", 12, 11, 9));

    assertThat(response.recommendations()).singleElement().satisfies(item -> {
        assertThat(item.action()).isEqualTo("PROMOTE");
        assertThat(item.recommendedLevel()).isEqualTo("L3");
        assertThat(item.modelVersion()).isEqualTo("ladder-klue-v1");
    });
    assertThat(server.takeRequest().getHeader("X-Mormi-Reporting-Service-Key"))
            .isEqualTo("report-key");
}
```

- [ ] **Step 2: Run and verify compile failure**

Run: `./gradlew test --tests com.mormi.backend.report.ReportAiClientTest`

Expected: FAIL because ladder request/response types and method are absent.

- [ ] **Step 3: Add package-private wire records and client call**

Serialize snake_case fields expected by AI. POST to `/v1/internal/ladder-analyses` using the
existing reporting key, bounded connect/read timeouts and no redirects. Translate AI 503 to an
empty unavailable result; do not fabricate a recommendation.

- [ ] **Step 4: Run tests and commit in Mormi-BE**

Run: `./gradlew test --tests com.mormi.backend.report.ReportAiClientTest`

Expected: PASS.

```bash
git add src/main/java/com/mormi/backend/report/DiagnosticReportDtos.java src/main/java/com/mormi/backend/report/ReportAiClient.java src/test/java/com/mormi/backend/report/ReportAiClientTest.java
git commit -m "feat: call ladder analysis service"
```

### Task 2: 추천·결정·적용 단계 저장

**Files:**
- Create: `Mormi-BE/src/main/resources/db/migration/V16__ladder_recommendations.sql`
- Create: `Mormi-BE/src/main/java/com/mormi/backend/report/LadderRecommendation.java`
- Create: `Mormi-BE/src/main/java/com/mormi/backend/report/LadderRecommendationRepository.java`
- Create: `Mormi-BE/src/main/java/com/mormi/backend/report/LearnerSkillLadderAssignment.java`
- Create: `Mormi-BE/src/main/java/com/mormi/backend/report/LearnerSkillLadderAssignmentRepository.java`
- Test: `Mormi-BE/src/test/java/com/mormi/backend/report/LadderRecommendationIntegrationTest.java`

**Interfaces:**
- Produces: immutable recommendation audit rows and effective-dated assignments

- [ ] **Step 1: Write failing persistence tests**

```java
@Test
void approval_creates_assignment_for_next_week_only() {
    LadderRecommendation recommendation = savePending("L2", "L3", LocalDate.of(2026, 8, 17));
    recommendation.approve(educatorId, clock.instant());
    assignmentRepository.save(recommendation.toAssignment());

    assertThat(assignmentRepository.findEffective(learnerId, "money-count", LocalDate.of(2026, 8, 23)))
            .isEmpty();
    assertThat(assignmentRepository.findEffective(learnerId, "money-count", LocalDate.of(2026, 8, 24)))
            .get().extracting(LearnerSkillLadderAssignment::getLevel).isEqualTo("L3");
}
```

- [ ] **Step 2: Run and verify missing migration/entities**

Run: `./gradlew test --tests com.mormi.backend.report.LadderRecommendationIntegrationTest`

Expected: FAIL.

- [ ] **Step 3: Add exact schema**

`ladder_recommendations` stores learner, skill, source week, current/recommended levels, action,
integer evidence counts, decimal rates, evidence turn IDs, model/policy versions, status,
reviewer, decision reason and timestamps. Unique key: learner + skill + week + model + policy.

`learner_skill_ladder_assignments` stores learner, skill, level, effective week start,
recommendation ID and created time. Unique key: learner + skill + effective week start.

Add check constraints for canonical levels, statuses and Monday effective dates.

- [ ] **Step 4: Implement domain transitions**

Only `PENDING` may become `APPROVED` or `REJECTED`; repeating the same decision is idempotent and
the opposite second decision returns conflict. `toAssignment()` exists only for approved rows and
sets `effective_week_start = source_week_start.plusWeeks(1)`.

- [ ] **Step 5: Run tests and commit**

Run: `./gradlew test --tests com.mormi.backend.report.LadderRecommendationIntegrationTest`

Expected: PASS.

```bash
git add src/main/resources/db/migration/V16__ladder_recommendations.sql src/main/java/com/mormi/backend/report/LadderRecommendation.java src/main/java/com/mormi/backend/report/LadderRecommendationRepository.java src/main/java/com/mormi/backend/report/LearnerSkillLadderAssignment.java src/main/java/com/mormi/backend/report/LearnerSkillLadderAssignmentRepository.java src/test/java/com/mormi/backend/report/LadderRecommendationIntegrationTest.java
git commit -m "feat: persist ladder recommendations and approvals"
```

### Task 3: 리포트 생성에 주간 추천 연결

**Files:**
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/DiagnosticReportDtos.java`
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/DiagnosticReportService.java`
- Create: `Mormi-BE/src/main/java/com/mormi/backend/report/LadderRecommendationService.java`
- Test: `Mormi-BE/src/test/java/com/mormi/backend/report/DiagnosticReportServiceTest.java`
- Test: `Mormi-BE/src/test/java/com/mormi/backend/report/LadderRecommendationServiceTest.java`

**Interfaces:**
- Produces: `LadderRecommendationView` in each supported report domain

- [ ] **Step 1: Write failing recommendation service tests**

```java
@Test
void current_report_requests_analysis_once_and_reuses_persisted_result() {
    service.recommendationsFor(learnerId, weekStart, analysisFixture());
    service.recommendationsFor(learnerId, weekStart, analysisFixture());

    verify(reportAiClient, times(1)).analyzeLadder(any());
    assertThat(repository.findAll()).hasSize(1);
}


@Test
void unavailable_analysis_keeps_report_available() {
    when(reportAiClient.analyzeLadder(any())).thenThrow(new ReportAiUnavailableException());
    DiagnosticReport report = diagnosticReportService.current(learnerId, weekStart);
    assertThat(report.modes()).isNotEmpty();
    assertThat(report.ladderRecommendations()).allMatch(view -> view.action().equals("UNAVAILABLE"));
}
```

- [ ] **Step 2: Run and verify failure**

Run: `./gradlew test --tests '*LadderRecommendationServiceTest' --tests '*DiagnosticReportServiceTest'`

Expected: FAIL.

- [ ] **Step 3: Build skill metrics from authoritative BE evidence**

Use `LearningTaskOutcome` and `LearningObservation` for the selected week. Derive eligible,
correct and independent integer counts; exclude system failures and non-learning safety records.
Do not let FE-provided report filters change ownership or source counts.

- [ ] **Step 4: Call AI, validate and persist**

Reject unknown skill IDs, non-canonical levels, ratios inconsistent with returned counts,
multi-step recommendations and mismatched week/model fields before persistence. Persist unavailable
status separately from HOLD so the UI can distinguish no recommendation from maintain.

- [ ] **Step 5: Extend the diagnostic DTO and run tests**

Expose current/recommended level, action, counts/rates, status, recommendation ID, evidence refs,
model/policy versions and decision status. Never expose actual raw utterances in this object; the
existing speech-evidence endpoint remains the controlled source for quotes.

Run: `./gradlew test --tests '*LadderRecommendationServiceTest' --tests '*DiagnosticReportServiceTest'`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/main/java/com/mormi/backend/report/DiagnosticReportDtos.java src/main/java/com/mormi/backend/report/DiagnosticReportService.java src/main/java/com/mormi/backend/report/LadderRecommendationService.java src/test/java/com/mormi/backend/report/DiagnosticReportServiceTest.java src/test/java/com/mormi/backend/report/LadderRecommendationServiceTest.java
git commit -m "feat: add ladder recommendations to diagnostic reports"
```

### Task 4: 교사 승인 API

**Files:**
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/LocalReportAdminController.java`
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/LocalReportAdminDtos.java`
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/report/LadderRecommendationService.java`
- Test: `Mormi-BE/src/test/java/com/mormi/backend/report/LocalReportAdminControllerTest.java`

**Interfaces:**
- Produces: `POST /v1/local-report-admin/learners/{learnerId}/ladder-recommendations/{recommendationId}/decision`

- [ ] **Step 1: Write failing authorization, approval and conflict tests**

```java
@Test
void approve_requires_admin_guard_and_uses_server_side_recommendation() throws Exception {
    mvc.perform(post(path)
            .header("X-Local-Report-Admin-Key", adminKey)
            .contentType(APPLICATION_JSON)
            .content("{\"decision\":\"APPROVE\"}"))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.applied_level").value("L3"))
        .andExpect(jsonPath("$.effective_week_start").value("2026-08-24"));
}
```

Also prove a body containing `recommended_level`, `effective_week_start`, learner ID or model
version is rejected as an unknown field, and cross-learner recommendation IDs return 404.

- [ ] **Step 2: Run and verify failure**

Run: `./gradlew test --tests com.mormi.backend.report.LocalReportAdminControllerTest`

Expected: FAIL because the endpoint is absent.

- [ ] **Step 3: Implement narrow decision body**

Accept only `decision=APPROVE|KEEP` and optional teacher note up to 500 characters. Load all other
values from the stored recommendation. `APPROVE` creates the assignment; `KEEP` marks rejected and
creates no assignment.

- [ ] **Step 4: Run tests and commit**

Run: `./gradlew test --tests com.mormi.backend.report.LocalReportAdminControllerTest`

Expected: PASS.

```bash
git add src/main/java/com/mormi/backend/report/LocalReportAdminController.java src/main/java/com/mormi/backend/report/LocalReportAdminDtos.java src/main/java/com/mormi/backend/report/LadderRecommendationService.java src/test/java/com/mormi/backend/report/LocalReportAdminControllerTest.java
git commit -m "feat: let teachers decide ladder recommendations"
```

### Task 5: 승인 단계의 다음 주 대화 적용

**Files:**
- Modify: `Mormi-AI/src/mormi_api/schemas.py`
- Modify: `Mormi-AI/src/mormi_api/service.py`
- Modify: `Mormi-AI/tests/test_service.py`
- Modify: `Mormi-BE/src/main/java/com/mormi/backend/dialogue/DialogueService.java`
- Create: `Mormi-BE/src/main/java/com/mormi/backend/report/LearnerSkillLadderAssignmentService.java`
- Test: `Mormi-BE/src/test/java/com/mormi/backend/dialogue/DialogueServiceTest.java`

**Interfaces:**
- Produces: server-only `approved_start_levels: dict[str, ExpressionLevel]` on AI `SessionCreate`

- [ ] **Step 1: Write failing AI precedence tests**

```python
async def test_approved_skill_level_overrides_home_default_l4(service) -> None:
    envelope = await service.create_conversation(
        session_create(approved_start_levels={"money-count": "L3"})
    )
    assert envelope.turn.pedagogy.expression_level is ExpressionLevel.L3


async def test_no_approval_preserves_home_default_l4(service) -> None:
    envelope = await service.create_conversation(session_create())
    assert envelope.turn.pedagogy.expression_level is ExpressionLevel.L4
```

- [ ] **Step 2: Implement and test the AI contract**

Canonicalize values, reject L1 and unknown skill keys, and use an approved level before the
existing home L4/profile fallback. The request remains service-key protected.

Run: `python -m pytest tests/test_service.py -q`

Expected: PASS.

- [ ] **Step 3: Write failing BE effective-date tests**

```java
@Test
void startHomeTeaching_passesOnlyAssignmentEffectiveToday() {
    clock.setInstant(Instant.parse("2026-08-24T01:00:00Z"));
    saveAssignment("money-count", "L3", LocalDate.of(2026, 8, 24));

    dialogueService.startHomeTeaching(learnerId, sessionId, resumeRequest());

    assertThat(aiRequest().path("approved_start_levels").path("money-count").asString())
            .isEqualTo("L3");
}
```

Also prove an assignment effective tomorrow is omitted and a browser request cannot override it.

- [ ] **Step 4: Resolve skill and add the server-owned field**

Resolve home skill from the learning session/practice result and life skill from the validated
scenario catalog. Query the latest assignment effective on or before today and insert it only in
the BE-created AI request map.

- [ ] **Step 5: Run focused tests and commit separately in AI and BE**

Run AI: `python -m pytest tests/test_service.py -q`

Run BE: `./gradlew test --tests com.mormi.backend.dialogue.DialogueServiceTest`

Expected: both PASS.

Commit AI contract first, then BE usage so each repository history is understandable.

### Task 6: 교사 리포트 추천 카드와 승인 UI

**Files:**
- Modify: `Mormi-FE/mormi-web/app/api-client.ts`
- Modify: `Mormi-FE/mormi-web/app/report/numeric-report-live-model.ts`
- Modify: `Mormi-FE/mormi-web/app/report/NumericReportPreview.tsx`
- Modify: `Mormi-FE/mormi-web/app/api/local-report-admin/[...path]/route.ts`
- Modify: `Mormi-FE/mormi-web/app/globals.css`
- Modify: `Mormi-FE/mormi-web/tests/numeric-report-live-model.test.mts`
- Modify: `Mormi-FE/mormi-web/tests/numeric-report-preview.dom.test.mjs`
- Modify: `Mormi-FE/mormi-web/tests/local-report-admin-proxy.test.mjs`

**Interfaces:**
- Consumes: `DiagnosticReportDto.ladder_recommendations`
- Produces: accessible approve/keep controls through the existing server-side admin proxy

- [ ] **Step 1: Read the installed Next.js guides required by `AGENTS.md`**

Read these installed guides before changing the proxy or component:

- `Mormi-FE/mormi-web/node_modules/next/dist/docs/01-app/01-getting-started/15-route-handlers.md`
- `Mormi-FE/mormi-web/node_modules/next/dist/docs/01-app/01-getting-started/05-server-and-client-components.md`
- `Mormi-FE/mormi-web/node_modules/next/dist/docs/01-app/02-guides/authentication.md`

- [ ] **Step 2: Write failing model and DOM tests**

```typescript
it("renders model-backed evidence instead of the old fixed L2 copy", () => {
  const model = buildNumericLiveReport(reportWithPromotionRecommendation(), []);
  assert.equal(model.domains.HOME[0].ladderStart, "L3");
  assert.match(model.domains.HOME[0].ladderRule, /정답률 92%/);
});
```

The DOM test must assert visible current/recommended level, counts, model analysis label, approve
and keep buttons; `UNAVAILABLE` must show `분석 기록이 더 필요해요` and no approval button.

- [ ] **Step 3: Run and verify failure**

Run from `Mormi-FE/mormi-web`:

`node --experimental-strip-types --test tests/numeric-report-live-model.test.mts`

`node --experimental-transform-types --import ./tests/ts-resolver.mjs --test tests/numeric-report-preview.dom.test.mjs tests/local-report-admin-proxy.test.mjs`

Expected: focused assertions fail because recommendation fields are not mapped/rendered.

- [ ] **Step 4: Extend strict DTO types and remove fixed recommendation copy**

Add exact union types for levels, actions and decisions. `numeric-report-live-model.ts` must map
BE evidence; delete the fixed `hasSpeech ? "L2"` recommendation. Unknown or inconsistent payloads
render unavailable rather than guessing.

- [ ] **Step 5: Add decision controls**

POST only `{decision:"APPROVE"}` or `{decision:"KEEP"}` through the existing HttpOnly teacher
session proxy. Disable controls while pending, show the server-returned effective week, and refresh
the report after success. Do not optimistic-update the applied level.

- [ ] **Step 6: Extend the proxy without exposing the admin key**

Forward POST request bodies only for the exact ladder-decision path, preserve the server-side key,
disable redirects and reject oversized or unknown JSON. Add valid-cookie, no-cookie, tampered-cookie
and upstream-error tests.

- [ ] **Step 7: Run FE tests and commit**

Run: `npm test`

Expected: all tests PASS.

Run: `npm run lint`

Expected: clean.

```bash
git add app/api-client.ts app/report/numeric-report-live-model.ts app/report/NumericReportPreview.tsx app/api/local-report-admin/[...path]/route.ts app/globals.css tests/numeric-report-live-model.test.mts tests/numeric-report-preview.dom.test.mjs tests/local-report-admin-proxy.test.mjs
git commit -m "feat: add teacher ladder recommendation approval"
```

### Task 7: 전체 계약 및 회귀 검증

**Files:**
- Modify only when a preceding focused test proves a regression.

**Interfaces:**
- Produces: one end-to-end verified prototype across AI, BE and FE

- [ ] **Step 1: Run AI verification**

Run: `python -m pytest -q`

Run: `python -m ruff check src tests scripts`

Run: `python -m mypy src/mormi_api`

Expected: all pass.

- [ ] **Step 2: Run BE verification**

Run: `./gradlew test --no-daemon`

Expected: BUILD SUCCESSFUL.

- [ ] **Step 3: Run FE verification**

Run: `npm test`

Run: `npm run lint`

Run: `npm run build`

Expected: all pass.

- [ ] **Step 4: Run the fixed acceptance scenario**

Use an L2 learner with 12 eligible turns, 11 correct, 9 independent and 9 model predictions at
L3 or higher. Verify the report shows L3 recommendation, learning still starts L2 before approval,
approval says next Monday, and a conversation created next Monday starts L3. Verify a fluent but
conceptually wrong L4 fixture remains L4 and receives concept help rather than expression downgrade.

- [ ] **Step 5: Record limitations**

Document that the artifact uses 16 simulated learners and one week, does not establish real-child
generalization, and requires a larger consented pilot before automatic or high-stakes use.
