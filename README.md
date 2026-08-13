# Mormi AI Dialogue Service

경계선지능 아동이 AI 동생 모르미를 가르치며 기초 수학을 복습하고, 카페 같은 생활 장면에 적용하도록 돕는 독립 AI 대화 서비스입니다.

이 저장소는 화면을 직접 렌더링하거나 일반 서비스 백엔드를 대신하지 않습니다.
다음 교육적 결정을 담당하고 Spring 백엔드가 프론트엔드에 전달할 수 있는
`TurnContract`를 반환합니다.

- 집 반복학습 결과를 받아 모르미 가르치기 시작 수준 결정
- 현재 집 커리큘럼 36개 세션의 검수된 가르치기 시나리오 생성
- 카페의 줄 서기, 예산 메뉴 선택, 메뉴값 덧셈, 거스름돈 진행
- 발화사다리 `L4~L0`와 힌트사다리 `H0~H3`를 독립적으로 조절
- 도움 카드 자동 공개
- 발화 이해 LLM, 결정형 오케스트레이터, 모르미 화자 LLM 분리
- 직접 설명과 공동 완성을 구분한 별노트 생성
- 학습자별 안정 발화 단계와 최근 힌트 의존도 저장
- 모르미 질문과 아이 원문 발화·선택 기록을 암호화 저장

## 서비스 책임 경계

이 저장소는 다음 AI·교육 로직을 소유합니다.

- 아이 발화 이해와 사실 슬롯·누락 슬롯·안전 유형 분류
- 결정형 발화사다리·힌트사다리 및 세션 진행
- 도움 카드와 시각자료 계약 생성
- 모르미 대사 생성과 출력 안전 검증
- 별노트 후보 문장·귀속 및 학습 프로필 변경값 계산
- 대화 중 활성 상태와 암호화 턴 기록

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
  → TurnContract
```

- **통제는 코드, 언어는 LLM**: LLM이 정답, 진도, L/H 전환, 힌트, 별노트 귀속을 결정하지 않습니다.
- **두 축을 분리**: 표현이 어렵다면 `L`만 낮추고, 개념이 어렵다면 `H`만 높입니다.
- **힌트의 주체는 도움 카드**: 모르미는 카드를 함께 보자고 요청할 뿐, 스스로 정답을 가르치지 않습니다.
- **부분 성공 보존**: 한 응답에서 맞은 슬롯은 기억하고 빠진 것만 다시 묻습니다.
- **자연스러운 하강**: “내가 한꺼번에 많이 물어봤네”처럼 질문 조정의 책임을 모르미가 집니다.
- **원문 기록 분리**: 원문은 암호화된 대화 기록에만 저장하며 학습 상태에는 검증된 사실만 저장합니다.
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
| GET | `/v1/conversations/{conversation_id}` | 최신 상태와 턴 복구 |
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
시각자료와 공동 별노트 문장을 생성합니다. 현재 FE 커리큘럼 36개 세션 ID를 모두
지원하며, 클라이언트가 정답이나 가르치기 문장을 임의로 보내지는 않습니다.
`home_teach`에는 비어 있지 않은 `learning_session_id`와 `practice_result_id`가
반드시 필요합니다. `practice_summary`는 시작 요청에 함께 보내거나, 같은
`practice_result_id`로 미리 저장한 결과를 복구할 수 있습니다. 같은 결과 ID를
재전송하면 최초 저장된 반복 결과가 유지됩니다.

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
    "note_update": null,
    "status": "active",
    "state_version": 2
  }
}
```

내부 `L/H`, 검증 슬롯, 병목은 기본 응답에서 숨깁니다. 로컬 디버깅에서만 `MORMI_SHOW_INTERNAL_PEDAGOGY=true`로 노출할 수 있습니다.

## 원문 데이터 보호

사용자 요구에 따라 모르미의 질문과 아이 원문 발화·선택을 저장합니다.

- 운영 환경에서는 `MORMI_RAW_DATA_ENCRYPTION_KEY` 없이는 서버가 시작되지 않습니다.
- 운영 환경에서는 PostgreSQL과 `MORMI_SERVICE_API_KEY`가 필수입니다.
- Spring 백엔드와의 서비스 간 호출은 `X-Mormi-Service-Key` 헤더로 보호합니다.
- Claude API 키는 Mormi-AI에만 두고, Spring과 프론트에는 전달하지 않습니다.
- Mormi-AI 서비스 키는 Spring 서버에만 두고, 브라우저에는 전달하지 않습니다.
- 원문은 발화 이해 요청과 암호화 기록에만 사용하고 학습 프로필에는 복사하지 않습니다.
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
| `MORMI_RAW_DATA_ENCRYPTION_KEY` | 예 | 원문 질문·응답 저장 암호화 키. 배포 후 임의 변경 금지 |
| `MORMI_SERVICE_API_KEY` | 예 | Spring→AI 호출을 보호하는 서비스 간 공유 키 |
| `MORMI_CLASSIFIER_MODEL` | 아니요 | 기본값 `claude-haiku-4-5-20251001` |
| `MORMI_SPEAKER_MODEL` | 아니요 | 기본값 `claude-sonnet-4-6` |
| `MORMI_IDEMPOTENCY_RETENTION_DAYS` | 아니요 | 멱등 응답 보존 기간, 기본 30일 |
| `MORMI_CORS_ORIGINS` | 아니요 | 브라우저 직접 호출을 허용할 오리진 JSON 배열. Spring 경유만 하면 `[]` |
| `MORMI_SHOW_INTERNAL_PEDAGOGY=false` | 아니요 | 운영 응답에서 내부 L/H·판정 근거를 숨김 |

Spring 서버에는 AI 비밀 전체가 아니라 다음 두 값만 공유합니다.

- `MORMI_DIALOGUE_BASE_URL`: AI 서버의 내부 주소
- `MORMI_DIALOGUE_SERVICE_KEY`: AI의 `MORMI_SERVICE_API_KEY`와 동일한 값

Anthropic 키, 원문 암호화 키와 DB 접속 문자열은 FE나 Spring 서버에 전달하지 않습니다.

## 검증

```bash
ruff check .
mypy src
pytest
```

상세 설계와 API 계약은 `docs/`를 참고하세요.

- 사람이 읽는 API 명세: [`docs/API_SPEC.md`](./docs/API_SPEC.md)
- OpenAPI 원본: [`docs/openapi.json`](./docs/openapi.json)
- 시각자료 계약: [`docs/VISUAL_CONTRACTS.md`](./docs/VISUAL_CONTRACTS.md)
