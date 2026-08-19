# Mormi AI Dialogue Service

경계선지능 아동이 AI 동생 모르미를 가르치며 기초 수학을 복습하고, 카페 같은 생활 장면에 적용하도록 돕는 독립 AI 대화 서비스입니다.

이 저장소는 화면을 직접 렌더링하거나 일반 서비스 백엔드를 대신하지 않습니다.
다음 교육적 결정을 담당하고 Spring 백엔드가 프론트엔드에 전달할 수 있는
`TurnContract`를 반환합니다.

- 집 반복학습 결과로 검수된 시나리오를 고르고 첫 가르치기는 항상 `L4-H0`에서 시작
- 모든 새 집 가르치기는 오개념 낚시 없이 모르미의 진짜 L4 도움 요청으로 시작
- 현재 집 커리큘럼 36개 세션의 검수된 가르치기 시나리오 생성
- 카페의 줄 서기, 예산 메뉴 선택, 메뉴값 덧셈, 거스름돈 진행
- 발화사다리 `L4~L0`와 힌트사다리 `H0~H3`를 독립적으로 조절
- 도움 카드 자동 공개
- 도움 카드와 분리된 검수·버전 고정 궁금해사전 40개 제공
- 발화 이해 LLM, 결정형 오케스트레이터, 모르미 화자 LLM 분리
- 모든 화자 대사의 결정형 검증과 위험 턴의 조건부 의미 검증
- 검증된 대사만 전송하는 SSE 진행·대사 스트리밍
- 자유문장은 LLM, 선택·빈칸·조작은 검수 ID 기반 결정형 판정 후 동일 오케스트레이터 사용
- 아이 근거를 보존한 직접 별노트와 검수된 공동 별노트를 구분해 생성
- 학습자별 안정 발화 단계와 최근 힌트 의존도 저장
- 모르미 질문과 아이 원문 발화·선택 기록을 DB에 평문 저장

## 서비스 책임 경계

이 저장소는 다음 AI·교육 로직을 소유합니다.

- 아이 발화 이해와 사실 슬롯·누락 슬롯·안전 유형 분류
- 결정형 발화사다리·힌트사다리 및 세션 진행
- 도움 카드와 시각자료 계약 생성
- 궁금해사전의 개념·예시·전용 시각자료 카탈로그와 버전 스냅샷 제공
- 모르미 대사 생성과 출력 안전 검증
- 별노트의 원문 근거·맥락 보충·귀속 및 학습 프로필 변경값 계산
- 대화 중 활성 상태와 평문 턴 기록

일반 서비스 백엔드인 [`Mormi-BE`](https://github.com/flyai-y2s2/Mormi-BE)는 회원·인증, 전체 학습 진도, 반복학습 원본, 보상·장소 해금 등 서비스 데이터를 담당합니다. 장기적으로 별노트와 학습자 프로필의 영구 원장은 BE가 소유하고, 이 서비스는 생성·갱신 결과를 반환하는 구조를 기준으로 합니다.

운영 요청 경로는 반드시 **인증된 Spring BE → Mormi-AI**입니다. Spring BE가 인증된
학습자 ID, 원문 저장 동의와 반복 결과를 채우고 `X-Mormi-Service-Key`로 이 API를
호출합니다. Next.js/FE의 직접 BFF 호출은 로컬 개발·계약 테스트용으로만 두며 운영
경로로 사용하지 않습니다. 반복 결과의 `practice_summary`에는 선택 ID·정오·오개념
태그 같은 구조 데이터만 넣고, 아이의 자유 발화 원문을 넣지 않습니다.

## 핵심 원칙

```text
아이 응답
  → Claude Haiku 발화 이해
  → 코드 안전 게이트
  → 결정형 교육 오케스트레이터
  → Claude Sonnet 모르미 화자
  → 코드 출력 검증
  → 위험 턴만 Claude Haiku 의미 검증
  → TurnContract
```

- **통제는 코드, 언어는 LLM**: LLM이 정답, 진도, L/H 전환, 힌트, 별노트 귀속을 결정하지 않습니다.
- **두 축을 분리**: 표현이 어렵다면 `L`만 낮추고, 개념이 어렵다면 `H`만 높입니다.
- **힌트의 주체는 도움 카드**: 모르미는 카드를 함께 보자고 요청할 뿐, 스스로 정답을 가르치지 않습니다.
- **세 콘텐츠의 역할 분리**: 궁금해사전은 믿고 참고하는 개념 자료, 도움 카드는 현재
  문제를 푸는 단계별 발판, 별노트는 아이가 실제로 알려 준 근거의 기록입니다.
  서로의 문구를 복사해 만들지 않습니다.
- **사전은 런타임 생성 금지**: 궁금해사전은 LLM이 대화 중 만들지 않습니다. 검수된
  버전 카탈로그를 대화 시작 시 스냅샷으로 고정하고, 이후 배포가 있어도 진행 중인
  대화에는 같은 카드가 보입니다.
- **도움카드 공통 계약**: 모든 콘텐츠는 `H1=주의`, `H2=구체적 행동·표상`,
  `H3=공동 수행·정답 마무리`를 선언하며, 단계가 올라갈수록 지원이 실제로 증가해야 합니다.
  수식·강조·탭 조작·순서 배열 중 학습 기능에 맞는 수단을 쓰며 이미지만을 강제하지 않습니다.
- **풀이 다양성 보존**: 콘텐츠는 `open_methods`와 `target_method`를 구분합니다.
  열린 과제에서 도움카드가 제시한 경로는 한 가지 발판일 뿐, 아이의 다른 타당한 풀이도
  발화 이해 AI가 의미 기준으로 인정합니다.
- **부분 성공 보존**: 한 응답에서 맞은 슬롯은 기억하고 빠진 것만 다시 묻습니다.
- **무진전 반복 차단**: 이미 기억한 사실만 되풀이되면 같은 질문을 반복하지 않고
  표현 지원을 한 단계 높여 최종적으로 도움 카드와 공동 수행에 도달합니다.
- **비유도형 진입**: 모르미가 일부러 틀린 답을 제시하지 않고, 자신이 헷갈리는
  수학 행동을 아이에게 묻습니다. v2 오개념 진입 필드는 기존 스냅샷 호환에만 사용합니다.
- **별노트 근거 보존**: 직접 노트는 사실로 검증된 아이 원문 구절을 그대로 품고,
  검수된 문제 맥락만 덧붙입니다. 아이가 말하지 않은 풀이 전략은 넣지 않습니다.
- **자연스러운 하강**: “내가 한꺼번에 많이 물어봤네”처럼 질문 조정의 책임을 모르미가 집니다.
- **동생다운 화자**: 모르미는 교사처럼 퀴즈를 내거나 상태를 보고하지 않고,
  “아, 세 개구나!”, “나 3이랑 5를 어떻게 비교할지 헷갈려...”처럼 자신이 모르는
  지점을 털어놓고 도움을 청합니다. 아이에게 생각의 근거를 입증시키지 않습니다.
- **안전한 자연스러움**: 코드는 검증된 사실, 빠진 슬롯, 질문 목적과 금지 답을
  정하고 Sonnet은 그 범위 안에서 말투를 만듭니다. 아이의 안전한 원문 구절은
  분류기가 정확한 인용 범위로 지정한 경우에만 자연스러운 되물음에 사용할 수 있습니다.
- **조건부 의미 검증**: 부분 답변, 설명 요청, 아이 표현을 되받는 턴만 Haiku가
  질문 의미와 캐릭터를 재검증합니다. 단순 안전 대응과 검수 문구에는 추가 호출을
  하지 않으며, 시간 초과나 거절 시 교육 진행은 유지하고 검수 문구로 대체합니다.
- **원문 기록 분리**: 원문은 접근이 통제된 대화 기록에 평문 저장하며 학습 상태에는 검증된 사실만 저장합니다.
- **완료 사실 연동**: `completion.verified_facts`에는 코드가 검증한 슬롯만 담아
  Spring BE가 카페 진행을 동기화하며, 원문 발화나 LLM 추측은 넣지 않습니다.

## 기술 구성

- Python 3.12
- FastAPI + Pydantic
- LangGraph
- Anthropic Claude Haiku / Sonnet
- SQLAlchemy async
- 개발 SQLite / 운영 PostgreSQL
- pytest, Ruff, mypy

## 로컬 실행

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn mormi_api.main:app --reload
```

배포 환경에서 일반 `pip` 설치가 필요하면 다음을 사용합니다.

```bash
pip install -r requirements.txt
```

- Swagger UI: `http://localhost:8000/docs`
- 상태 확인: `GET http://localhost:8000/health`

자유 발화를 처리하려면 `.env`에 `MORMI_ANTHROPIC_API_KEY`를 등록해야 합니다. 선택·조작 응답과 결정형 테스트는 키 없이도 동작합니다.

## 주요 API

| Method | Path | 역할 |
|---|---|---|
| POST | `/v1/practice-results` | 집 반복학습 결과 저장 |
| POST | `/v1/conversations` | 가르치기/카페 대화 시작 |
| POST | `/v1/conversations/{conversation_id}/responses` | 발화·선택·조작 응답 제출 |
| POST | `/v1/conversations/{conversation_id}/responses/stream` | SSE 진행 상태와 검증된 다음 턴 스트리밍 |
| GET | `/v1/conversations/{conversation_id}` | 최신 상태와 턴 복구 |
| GET | `/v1/content/dictionary-cards/{curriculum_session_id}` | 현재 승인된 궁금해사전 카드 조회 |
| GET | `/v1/conversations/{conversation_id}/dictionary-card` | 대화에 고정된 궁금해사전 카드 조회 |
| GET | `/v1/learners/{learner_id}/skill-profiles` | 학습자별 L/H 근거 조회 |
| GET | `/v1/learners/{learner_id}/star-notes` | 별노트 조회 |
| GET | `/v1/conversations/{conversation_id}/transcript` | 보호된 원문 질문·응답 기록 조회 |

현재 프로토타입은 `cafe_queue`, `cafe_budget_menu`, `cafe_menu_total`,
`cafe_change`의 4개 독립 시나리오를 지원합니다. 5단계 통합 시나리오는 아직 공개하지
않습니다. 줄 인원은 AI가 세션 시작 시 정하고, 메뉴판·모르미 메뉴·예산은 프론트가
`cafe_context`로 전달합니다. 이 값들은 세션 상태에 보존되어 재시도나 복구 때
바뀌지 않습니다.

집 가르치기는 `scenario_id=home_teach`로 시작합니다. AI가 반복 결과의
`curriculum_session_id`를 검수된 카탈로그에서 찾아 발화사다리 질문, 도움 카드,
시각자료, 별노트용 중립 맥락과 공동 별노트 문장을 생성합니다. 현재 FE 커리큘럼 36개 세션 ID를 모두
지원하며, 클라이언트가 정답이나 가르치기 문장을 임의로 보내지는 않습니다.
반복학습 오답은 개념 수행 기록으로만 사용하며 아이의 설명 능력을 미리 낮춰
판정하지 않습니다. 따라서 집 가르치기의 첫 턴은 정답률과 무관하게 자유 설명
`L4-H0`, 텍스트 입력, 도움 카드 없음으로 시작합니다. 모르미는 틀린 답을 먼저
제시하지 않고 “나 모두 얼마인지랑 돈을 어떻게 더하는지 헷갈려. 알려줄 수 있어?”처럼
자신의 실제 도움 요청을 말합니다. 아이가 첫 발화에서 답과 이유까지 설명하면 추가
형식 질문 없이 바로 완료할 수 있습니다. 답만 설명한 경우에도 L4 수행으로 인정하고,
이미 말한 답은 되묻지 않은 채 빠진 이유나 방법만 한 번 더 묻습니다. `wrong_guess`
필드와 판정은 배포 전에 이미 시작된 v2 세션을 깨뜨리지 않기 위한 호환 코드에만 남아
있으며, 현재 콘텐츠 카탈로그에는 사용할 수 없습니다.
`home_teach`에는 비어 있지 않은 `learning_session_id`와 `practice_result_id`가
반드시 필요합니다. `practice_summary`는 시작 요청에 함께 보내거나, 같은
`practice_result_id`로 미리 저장한 결과를 복구할 수 있습니다. 같은 결과 ID를
재전송하면 최초 저장된 반복 결과가 유지됩니다.

궁금해사전은 집 커리큘럼 36개와 현재 카페 4개에 각각 한 장씩 등록되어 있습니다.
각 카드는 도움 문구와 별개인 `concept`, 구체적인 `example`, 그 예시에 근거한 전용
`visual`, `source_refs`, 승인 정보와 콘텐츠 버전을 가집니다. 대화 시작 시 카드 본문과
해시가 세션에 고정되며, 각 `TurnContract.dictionary_ref`는 화면이 사용할 카드의
ID·버전·해시를 알려 줍니다. 프론트가 힌트 문구를 조합해 사전 내용을 만들지 않습니다.
아동 화면에는 `title`·`concept`·`example`·`visual`만 사전 본문으로 표시하고,
`learning_goal`·출처·승인 정보는 운영 및 검수용 메타데이터로 유지합니다.

거스름돈 단계는 현재 FE 계약을 따릅니다. 모르미가 고른 메뉴 하나에 10,000원을
내며, 아이는 `10,000 − 모르미 메뉴 가격`을 계산합니다.

요청의 `response_id`는 멱등키입니다. 같은 응답을 재전송하면 상태를 다시 진행하지 않고 최초 생성된 결과 턴을 반환합니다.

## 입력 계약

```json
{
  "turn_id": "turn_...",
  "response_id": "9cda3c1e-6539-4b35-9ac5-c63f91e203b1",
  "type": "text",
  "text": "왼쪽 줄에 세 명, 오른쪽 줄에 다섯 명 있어",
  "choice_ids": [],
  "values": {},
  "asr_confidence": null,
  "latency_ms": 4200
}
```

`type`은 `text`, `choice`, `fill`, `count`, `equation`, `action`, `no_response`를
지원합니다. Spring 백엔드는 프론트에서 받은 입력을 이전 턴의 `input.kind`에 맞는
형태로 이 서비스에 전달합니다.

## 출력 계약

```json
{
  "conversation_id": "conversation_...",
  "turn": {
    "turn_id": "turn_...",
    "task_id": "cafe_queue",
    "stage_id": "queue",
    "mormi": {
      "text": "왼쪽 줄에는 3명이 있구나. 오른쪽은 몇 명이야?",
      "mood": "listening",
      "max_lines": 2
    },
    "input": {"kind": "text", "target_slots": ["right_count"]},
    "visual": {"type": "cafe_queues", "data": {}},
    "help_card": null,
    "task_anchor": {
      "anchor_id": "cafe_queue:short_reason",
      "title": "지금 모르미에게 알려줄 것",
      "prompt": "나는 왜 그 줄이 더 빠른지 헷갈려... 알려줄 수 있어?",
      "completed_items": [],
      "target_slots": ["reason"]
    },
    "dictionary_ref": {
      "card_id": "dictionary.cafe.cafe-queue",
      "curriculum_session_id": "cafe_queue",
      "schema_version": 1,
      "content_version": 1,
      "content_hash": "<sha256>"
    },
    "note_update": null,
    "status": "active",
    "state_version": 2
  }
}
```

내부 `L/H`, 검증 슬롯, 병목은 기본 응답에서 숨깁니다. 로컬 디버깅에서만 `MORMI_SHOW_INTERNAL_PEDAGOGY=true`로 노출할 수 있습니다.

## 원문 데이터 저장과 접근 보호

사전 동의를 완료한 파일럿 참여자의 모르미 질문과 아이 원문 발화·선택을 기본적으로
DB에서 바로 읽을 수 있는 평문으로 영구 저장합니다. 기존 배포에서 생성한 `fernet:`
레코드는 새 버전의 첫 시작 시 하나의 트랜잭션에서 `plain:` 레코드로 변환합니다.

- 운영 환경에서는 PostgreSQL과 `MORMI_SERVICE_API_KEY`가 필수입니다.
- Spring 백엔드와의 서비스 간 호출은 `X-Mormi-Service-Key` 헤더로 보호합니다.
- Claude API 키는 Mormi-AI에만 두고, Spring과 프론트에는 전달하지 않습니다.
- Mormi-AI 서비스 키는 Spring 서버에만 두고, 브라우저에는 전달하지 않습니다.
- 원문은 발화 이해 요청과 대화 기록에만 사용하고 학습 프로필에는 복사하지 않습니다.
- DB 계정, 운영 서버, 백업 접근 권한으로 평문 원문을 보호해야 합니다.
- 원문 동의가 없으면 아이 원문은 저장하지 않고 구조화 판정만 저장합니다.
- 원문 보존 정책은 `no_raw`, `30_days`, `90_days` 중 하나입니다.
- 실제 아동 대상 운영 전에 보호자·기관의 동의 철회·삭제 요청 절차를 확정해야 합니다.

## 운영 환경변수

AI 서버의 `/etc/mormi-ai/mormi.env`에는 다음 값을 둡니다.

| 변수 | 운영 필수 | 설명 |
|---|---:|---|
| `MORMI_ENVIRONMENT=production` | 예 | 운영 안전 검증 활성화 |
| `MORMI_DATABASE_URL` | 예 | `postgresql+asyncpg://...` 형식의 PostgreSQL 접속 문자열 |
| `MORMI_ANTHROPIC_API_KEY` | 예 | 자유 발화 분류와 모르미 발화 생성용 Claude API 키 |
| `MORMI_RAW_DATA_ENCRYPTION_KEY` | 아니요 | 이전 `fernet:` 레코드를 최초 1회 평문으로 변환할 때만 필요한 기존 키 |
| `MORMI_SERVICE_API_KEY` | 예 | Spring→AI 호출을 보호하는 서비스 간 공유 키 |
| `MORMI_SKIP_STARTUP_MAINTENANCE` | 아니오 | 로컬·개발·테스트 진단 전용. `true`이면 시작 시 스키마 생성·저장소 마이그레이션·만료 원문 정리를 모두 건너뛴다. 운영 환경에서는 사용할 수 없다. |
| `MORMI_OBSERVATION_INGEST_URL` | 관찰 전송 시 예 | Spring의 `/internal/v1/observations/events` 전체 URL |
| `MORMI_OBSERVATION_INGEST_KEY` | 관찰 전송 시 예 | AI→Spring 관찰 이벤트 호출을 보호하는 전용 공유 키 |
| `MORMI_STAR_NOTE_EVENTS_ENABLED` | 아니요 | BE 별노트 수신 계약 배포 후 `true`; 기본 `false`에서는 별노트 이벤트를 pending으로 보존 |
| `MORMI_OUTBOX_POLL_INTERVAL_SECONDS` | 아니요 | 전송할 이벤트가 없을 때 폴링 간격, 기본 2초 |
| `MORMI_OUTBOX_BATCH_SIZE` | 아니요 | 한 폴링 주기에 처리할 최대 이벤트 수, 기본 20개 |
| `MORMI_OUTBOX_REQUEST_TIMEOUT_SECONDS` | 아니요 | Spring 수신 API 제한 시간, 기본 5초 |
| `MORMI_OUTBOX_RETRY_BASE_SECONDS` | 아니요 | 재시도 백오프 시작값, 기본 2초 |
| `MORMI_OUTBOX_RETRY_MAX_SECONDS` | 아니요 | 재시도 백오프 상한, 기본 300초 |
| `MORMI_OUTBOX_LEASE_SECONDS` | 아니요 | 전송 중 worker 장애를 감지하는 처리 임대 시간, 기본 30초 |
| `MORMI_CLASSIFIER_MODEL` | 아니요 | 기본값 `claude-haiku-4-5-20251001` |
| `MORMI_SPEAKER_MODEL` | 아니요 | 기본값 `claude-sonnet-4-6` |
| `MORMI_SPEAKER_TIMEOUT_SECONDS` | 아니요 | 화자 생성 제한 시간, 기본 8초 |
| `MORMI_SPEAKER_VERIFIER_ENABLED` | 아니요 | 위험 턴 조건부 의미 검증, 기본 `true` |
| `MORMI_SPEAKER_VERIFIER_TIMEOUT_SECONDS` | 아니요 | 의미 검증 제한 시간, 기본 1.8초 |
| `MORMI_IDEMPOTENCY_RETENTION_DAYS` | 아니요 | 멱등 응답 보존 기간, 기본 30일 |
| `MORMI_CORS_ORIGINS` | 아니요 | 브라우저 직접 호출을 허용할 오리진 JSON 배열. Spring 경유만 하면 `[]` |
| `MORMI_SHOW_INTERNAL_PEDAGOGY=false` | 아니요 | 운영 응답에서 내부 L/H·판정 근거를 숨김 |

대화 프록시 호출을 위해 Spring 서버와 맞출 값은 다음 두 가지입니다.

- `MORMI_DIALOGUE_BASE_URL`: AI 서버의 내부 주소
- `MORMI_DIALOGUE_SERVICE_KEY`: AI의 `MORMI_SERVICE_API_KEY`와 동일한 값

관찰 이벤트 파이프라인을 켤 때는 AI 서버에 Spring 수신 URL과
`MORMI_OBSERVATION_INGEST_KEY`를 함께 설정하고, Spring 수신 API에도 같은 키를
설정합니다. 둘 중 하나만 있으면 기존 대화 API는 정상 시작하되 전송기는 비활성화되어
설정 누락 로그를 남깁니다. 키 값과 관찰 payload는 전송 로그에 기록하지 않습니다.

별노트 B안은 같은 수신 URL에 `event_type=star_note_created`를 별도 이벤트로 보냅니다.
AI는 별노트가 생성된 턴과 같은 트랜잭션에서 outbox row를 만들지만,
`MORMI_STAR_NOTE_EVENTS_ENABLED=false`인 동안에는 전송 대상으로 가져오지 않습니다.
Spring이 해당 이벤트의 멱등 수집·조회 원장을 배포한 뒤 이 값을 `true`로 바꾸면 기존
pending 이벤트까지 차례로 전달됩니다. AI를 먼저 배포할 때도 별노트가 422로 유실되지
않도록 운영 순서를 보장하기 위한 플래그입니다.

Anthropic 키와 DB 접속 문자열은 FE나 Spring 서버에 전달하지 않습니다. 첫 평문 전환이
완료되기 전에는 기존 원문 암호화 키도 AI 서버에 유지해야 합니다.

## 검증

```bash
ruff check .
mypy src
pytest
```

도움카드는 세 겹으로 출시 전 검수합니다. 코드 검증은 테스트와 콘텐츠 import에서 항상
실행되고, 오프라인 AI 검수는 실제 대화 중이 아니라 콘텐츠 등록 전에만 실행합니다.

```bash
# 질문·화면·H1·H2·H3를 한 블록에 모은 사람 검수표
PYTHONPATH=src .venv/bin/python scripts/audit_help_cards.py \
  --report /tmp/mormi-help-card-review.md

# 선택 실행: 의미 중복, 이해 가능성, 사실 근거와 풀이 강요 여부까지 오프라인 AI 검수
PYTHONPATH=src .venv/bin/python scripts/audit_help_cards.py \
  --ai --report /tmp/mormi-help-card-review.md \
  --json /tmp/mormi-help-card-ai-audit.json
```

새 스테이지가 `SCENARIOS`에 등록되면 사람·AI 검수 목록에도 자동으로 포함됩니다.
H1~H3 누락, 미등록 사실 참조, 학습 기능과 맞지 않는 지원 수단, H3에서 완료할 수 없는
필수 슬롯은 빌드 전에 실패합니다.

궁금해사전도 같은 세 겹의 출시 전 검수를 거칩니다. 결정형 검사는 40개 활성 과제
전수 대응, 승인·출처·버전 해시, 문장 독립성, 산술·시각 사실 일치, 도움카드 문구 복사,
고아·누락 카드와 `3십` 같은 비표준 자릿값 표현을 검사합니다. 순차 수 세기 그림은
`1`부터 목표 개수까지 빠짐없이 증가하지 않으면 카탈로그 로딩이 실패합니다.

```bash
# 개념·예시·시각자료·연결 과제를 함께 보는 사람 검수표
PYTHONPATH=src .venv/bin/python scripts/audit_dictionary_cards.py \
  --report /tmp/mormi-dictionary-review.md

# 선택 실행: 이해 가능성·수학 정확성·풀이 강요·도움카드 중복 의미 검수
PYTHONPATH=src .venv/bin/python scripts/audit_dictionary_cards.py \
  --ai --report /tmp/mormi-dictionary-review.md \
  --json /tmp/mormi-dictionary-ai-audit.json
```

상세 설계와 API 계약은 `docs/`를 참고하세요.

- 사람이 읽는 API 명세: [`docs/API_SPEC.md`](./docs/API_SPEC.md)
- 대화 관찰·리포트 이벤트: [`docs/OBSERVATION_EVENTS.md`](./docs/OBSERVATION_EVENTS.md)
- OpenAPI 원본: [`docs/openapi.json`](./docs/openapi.json)
- 시각자료 계약: [`docs/VISUAL_CONTRACTS.md`](./docs/VISUAL_CONTRACTS.md)
