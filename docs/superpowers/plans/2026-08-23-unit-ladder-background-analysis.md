# Unit Ladder Background Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 같은 소단원 반복학습을 최근 두 번 연속 완료하면 발화 사다리 분석을 백그라운드 실행하고, 실제 발화와 교사 승인형 승급·유지·하향 추천을 주간 리포트에 표시한다.

**Architecture:** Mormi-BE는 완료 후 이벤트로 Mormi-AI에 멱등 분석 작업을 등록한다. Mormi-AI는 DB 작업 큐와 지연 로드 모델 실행기로 소단원 결과를 저장하고 내부 리포트 근거에 노출한다. Mormi-BE는 발화·추천을 학습자 범위 안에서 조립하고 승인 요청을 AI 프로필 변경으로 전달하며, Mormi-FE는 실제 발화와 추천/승인 상태를 표시한다.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy async, PyTorch/Transformers, Java 21, Spring Boot 4, JPA, React 19, Next.js 16, TypeScript 5

**Spec:** `docs/superpowers/specs/2026-08-23-unit-ladder-background-analysis-design.md`

## Global Constraints

- 지원 단계 순서는 `L0 < L2 < L3 < L4`이며 L1은 읽기 시 L2로 정규화한다.
- 모델 입력에는 발화 원문만 넣고 현재 단계·응답 방식·정답 여부·소단원 ID를 직렬화하지 않는다.
- `no_response`, `choice`, `solve_together`는 규칙으로만 처리하고 L0는 모델이 예측하지 않는다.
- 분석은 자동 실행하지만 시작 단계는 교사 승인 전까지 변경하지 않는다.
- 하향과 승급은 한 단계씩만 적용한다.
- 발화 원문과 비밀키는 신규 분석 테이블이나 로그에 기록하지 않는다.
- 기존 Mormi-AI 작업 트리의 `uv.lock`과 Mormi-FE의 사용자 변경 파일은 건드리지 않는다.

---

### Task 1: 소단원 추천 판정기와 모델 런타임

**Files:**
- Create: `src/mormi_api/ladder_model/runtime.py`
- Create: `src/mormi_api/ladder_analysis.py`
- Modify: `src/mormi_api/ladder_model/policy.py`
- Test: `tests/test_ladder_analysis.py`

**Interfaces:**
- Consumes: `rule_recommendation(current_level, response_mode)`와 `run-v2/model` 저장 모델.
- Produces: `LadderEvidence`, `LadderDecision`, `decide_ladder_adjustment(...)`, `LadderModelRuntime.predict(texts)`.

- [ ] 최근 두 세션, 단계별 정답/시도 수, 최근 판정 목록을 입력하는 실패 테스트를 작성한다.
- [ ] `90% + 4회 이상 + 최근 두 판정이 다음 단계 이상`이면 한 단계 `UPGRADE`가 되는지 실패를 확인한다.
- [ ] 최근 두 판정이 연속 하위 단계이거나 `정답률 < 70% + 하위 근거`이면 한 단계 `ADJUST_DOWN`이 되는지 실패를 확인한다.
- [ ] L4 상향 금지, L0 모델 예측 금지, 근거 부족, 유지, 한 단계 제한 테스트를 추가하고 실패를 확인한다.
- [ ] 순수 판정 함수를 최소 구현해 테스트를 통과시킨다.
- [ ] 로컬 모델을 지연 로드하고 배치 추론 결과의 단계·신뢰도를 반환하는 런타임을 구현한다.
- [ ] 모델 경로/의존성 누락 시 원문을 로그에 남기지 않고 명시적 unavailable 결과를 반환하는 테스트를 통과시킨다.
- [ ] Run: `uv run --extra analysis pytest -q tests/test_ladder_policy_v2.py tests/test_ladder_analysis.py`
- [ ] Expected: selected policy and decision tests exit 0 with no failures.
- [ ] Commit: `feat: add unit ladder decision engine`

### Task 2: AI 분석 작업 저장소와 백그라운드 작업자

**Files:**
- Modify: `src/mormi_api/db.py`
- Create: `src/mormi_api/ladder_analysis_repository.py`
- Create: `src/mormi_api/ladder_analysis_worker.py`
- Modify: `src/mormi_api/settings.py`
- Modify: `src/mormi_api/migrations.py`
- Test: `tests/test_ladder_analysis_worker.py`
- Test: `tests/test_startup_maintenance.py`

**Interfaces:**
- Consumes: Task 1의 `LadderModelRuntime`과 `decide_ladder_adjustment`.
- Produces: `LadderAnalysisRecord`, `enqueue`, `claim_pending`, `complete`, `fail`, `latest_for_learner`, `LadderAnalysisWorker.run_forever()`.

- [ ] 멱등 작업 키를 두 번 등록해도 한 행만 생기는 실패 테스트를 작성한다.
- [ ] `pending → running → completed/failed` 상태 전이와 재시작 후 pending 재처리 실패 테스트를 작성한다.
- [ ] 학습자·소단원·트리거 세션·단계별 집계·결과·버전·시각만 저장하고 원문 필드가 없음을 검증한다.
- [ ] PostgreSQL과 SQLite 테스트에서 안전하게 작업을 선점하는 저장소를 최소 구현한다.
- [ ] `MORMI_LADDER_MODEL_DIR`, worker 활성화, poll interval, batch size 설정을 추가한다.
- [ ] 작업자가 기존 turn 원문을 보존 정책 안에서 읽고 Task 1 판정기로 결과를 저장하도록 구현한다.
- [ ] 오류 코드는 제한된 값으로 저장하고 다른 작업을 계속 처리하는 테스트를 통과시킨다.
- [ ] Run: `uv run --extra analysis pytest -q tests/test_ladder_analysis_worker.py tests/test_startup_maintenance.py`
- [ ] Expected: worker persistence and startup tests exit 0 with no failures.
- [ ] Commit: `feat: persist and process ladder analysis jobs`

### Task 3: AI 내부 API, 리포트 근거, 교사 승인

**Files:**
- Modify: `src/mormi_api/schemas.py`
- Modify: `src/mormi_api/repository.py`
- Modify: `src/mormi_api/reporting.py`
- Modify: `src/mormi_api/main.py`
- Modify: `src/mormi_api/engine.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_service.py`
- Test: `tests/test_home_teaching.py`

**Interfaces:**
- Produces: `POST /v1/internal/ladder-analyses`, `POST /v1/internal/ladder-analyses/{analysis_id}/approve`, `ReportEvidenceResponse.ladder_recommendations`.

- [ ] 서비스 키 없이는 등록·승인 API가 실패하고 다른 학습자 분석을 승인할 수 없는 테스트를 작성한다.
- [ ] 등록 API가 202와 동일 분석 ID를 멱등 반환하는 실패 테스트를 작성한다.
- [ ] 승인 시 최신 추천 버전만 `highest_stable_expression_level`에 한 단계 적용되는 실패 테스트를 작성한다.
- [ ] `update_skill_profile`이 숙달도·연속 성공은 갱신하지만 시작 단계를 자동 승급하지 않는 실패 테스트를 작성한다.
- [ ] 스키마·엔드포인트·저장소 연결을 최소 구현한다.
- [ ] lifespan에서 worker를 시작하고 종료 시 정상 정리한다.
- [ ] 내부 리포트 근거에 학습자별 최신 소단원 분석만 포함한다.
- [ ] Run: `uv run --extra analysis pytest -q tests/test_reporting.py tests/test_service.py tests/test_home_teaching.py tests/test_ladder_analysis_worker.py`
- [ ] Expected: internal API, profile, reporting, and worker tests exit 0 with no failures.
- [ ] Commit: `feat: expose and approve ladder recommendations`

### Task 4: BE의 연속 두 세션 트리거와 AI 작업 등록

**Files:**
- Modify: `src/main/java/com/mormi/backend/session/LearningSessionRepository.java`
- Modify: `src/main/java/com/mormi/backend/session/LearningSessionService.java`
- Create: `src/main/java/com/mormi/backend/session/LadderAnalysisTrigger.java`
- Create: `src/main/java/com/mormi/backend/session/LadderAnalysisTriggerService.java`
- Modify: `src/main/java/com/mormi/backend/report/ReportAiClient.java`
- Test: `src/test/java/com/mormi/backend/session/LadderAnalysisTriggerServiceTest.java`
- Test: `src/test/java/com/mormi/backend/report/ReportAiClientTest.java`

**Interfaces:**
- Consumes: Task 3의 분석 등록 API.
- Produces: 완료 후 커밋 이벤트와 AI 작업 등록 요청 DTO.

- [ ] 가장 최근 완료 세션 두 건이 동일 소단원일 때만 이벤트가 생성되는 실패 테스트를 작성한다.
- [ ] 다른 소단원, 완료 1건, 동일 완료 요청 재전송에서는 등록하지 않는 실패 테스트를 작성한다.
- [ ] 두 세션의 drill 시도를 단계별 정답/시도 수로 정규화하고 L1을 L2로 합산하는 실패 테스트를 작성한다.
- [ ] 학습 완료 트랜잭션 이후 비동기로 등록하며 호출 실패가 완료 응답을 깨뜨리지 않는 테스트를 작성한다.
- [ ] AI 등록 클라이언트에 멱등 키, 학습자, 소단원, 두 세션 ID, 단계별 집계를 전송하도록 구현한다.
- [ ] Run: `./gradlew test --tests '*LadderAnalysisTriggerServiceTest' --tests '*ReportAiClientTest'`
- [ ] Expected: selected Spring tests finish with `BUILD SUCCESSFUL`.
- [ ] Commit: `feat: enqueue ladder analysis after repeated unit practice`

### Task 5: BE 리포트 추천·승인과 실제 발화 폴백

**Files:**
- Modify: `src/main/java/com/mormi/backend/report/DiagnosticReportDtos.java`
- Modify: `src/main/java/com/mormi/backend/report/DiagnosticReportService.java`
- Modify: `src/main/java/com/mormi/backend/report/DiagnosticReportController.java`
- Modify: `src/main/java/com/mormi/backend/report/LocalReportAdminController.java`
- Modify: `src/main/java/com/mormi/backend/report/ReportAiClient.java`
- Test: `src/test/java/com/mormi/backend/report/DiagnosticReportServiceTest.java`
- Test: `src/test/java/com/mormi/backend/report/LocalReportAdminControllerTest.java`

**Interfaces:**
- Produces: `DiagnosticReport.ladderRecommendations`, 최근 발화 단독 가능 `SpeechEvidence`, 교사 승인 프록시 API.

- [ ] 강한 검증 발화 쌍이 기존처럼 우선 선택되는 테스트를 유지한다.
- [ ] 강한 쌍이 없어도 동일 소단원의 검증된 실제 응답에서 과거·최근을 반환하는 실패 테스트를 작성한다.
- [ ] 한 건이면 `past=null`, `recent=실제 발화`를 반환하고 원문 동의가 없으면 unavailable인 테스트를 작성한다.
- [ ] 현재 주차와 선택 소단원에 맞는 최신 추천만 리포트에 포함하는 실패 테스트를 작성한다.
- [ ] 승인 API가 학습자 범위·분석 버전을 검증하고 AI 승인 API로 전달하도록 구현한다.
- [ ] AI 요약 실패 시 정답률·예측 단계·추천 상태로 검증 가능한 기본 `AI가 본 변화` 문장을 만든다.
- [ ] Run: `./gradlew test --tests '*DiagnosticReportServiceTest' --tests '*LocalReportAdminControllerTest'`
- [ ] Expected: selected report and controller tests finish with `BUILD SUCCESSFUL`.
- [ ] Commit: `feat: add speech and ladder evidence to teacher reports`

### Task 6: FE 소단원 추천 카드, 발화 표시, 교사 승인

**Files:**
- Modify: `app/api-client.ts`
- Modify: `app/local-report-admin-client.ts`
- Modify: `app/report/numeric-report-live-model.ts`
- Modify: `app/report/NumericReportPreview.tsx`
- Modify: `app/report/ReportDashboard.tsx`
- Modify: `app/globals.css`
- Test: `tests/numeric-report-live-model.test.mts`
- Test: `tests/numeric-report-preview.dom.test.mjs`
- Test: `tests/api-client.test.mjs`

**Interfaces:**
- Consumes: Task 5의 추천, 발화, 승인 API.
- Produces: 최하단 `발화 사다리 분석` 카드와 교사 승인 상태.

- [ ] 작업 전에 `node_modules/next/dist/docs/`의 현재 데이터 패칭·클라이언트 컴포넌트 관련 가이드를 읽는다.
- [ ] 최근 발화 한 건만 있을 때 실제 원문을 표시하는 실패 DOM 테스트를 작성한다.
- [ ] UPGRADE/MAINTAIN/ADJUST_DOWN/INSUFFICIENT_EVIDENCE 카드 문구와 단계 화살표 실패 테스트를 작성한다.
- [ ] 추천이 없을 때 자동 분석 조건을 안내하고 팝업을 만들지 않는 테스트를 작성한다.
- [ ] 승급·하향에만 `이 단계로 적용` 버튼을 표시하고 성공 뒤 적용 상태를 보여주는 실패 테스트를 작성한다.
- [ ] DTO, 라이브 모델, 화면과 승인 요청을 최소 구현한다.
- [ ] 좁은 화면과 A4 리포트에서 카드가 넘치지 않도록 기존 디자인 토큰으로 스타일링한다.
- [ ] Run: `node --experimental-strip-types --test tests/numeric-report-live-model.test.mts`
- [ ] Expected: the live-model test process reports zero failures.
- [ ] Run: `node --experimental-transform-types --import ./tests/ts-resolver.mjs --test tests/numeric-report-preview.dom.test.mjs tests/api-client.test.mjs`
- [ ] Expected: the DOM and API client test process reports zero failures.
- [ ] Commit: `feat: show and approve unit ladder recommendations`

### Task 7: 통합 검증과 운영 설정

**Files:**
- Modify: `docs/LADDER_MODEL_LOCAL_TRAINING.md`
- Create: `docs/LADDER_ANALYSIS_OPERATIONS.md`

- [ ] AI에 `MORMI_LADDER_MODEL_DIR`와 worker 설정 예시를 추가하고 모델 가중치를 Git에 넣지 않는다.
- [ ] BE↔AI 서비스 키와 분석 등록/승인 경로를 운영 설정에 기록한다.
- [ ] 로컬에서 같은 소단원 두 번 완료 → pending → completed → 리포트 표시 → 교사 승인 → 다음 대화 시작 단계 반영 흐름을 검증한다.
- [ ] Run AI: `uv run --extra analysis pytest -q`
- [ ] Expected: the full AI suite exits 0 with no failures.
- [ ] Run AI checks: `uv run ruff check src tests scripts && uv run mypy src/mormi_api`
- [ ] Expected: Ruff reports all checks passed and mypy reports no issues.
- [ ] Run BE: `./gradlew test`
- [ ] Expected: the complete backend suite finishes with `BUILD SUCCESSFUL`.
- [ ] Run FE: `npm test && npm run lint`
- [ ] Expected: all frontend tests/build checks pass and ESLint exits 0.
- [ ] 실제 발화·비밀키가 로그 및 분석 테이블에 없는지 확인한다.
- [ ] Commit: `docs: document ladder analysis operations`
