# 모르미 AI 대화 API 명세

> 버전: `0.1.0`
>
> 기준 경로: `/v1`
>
> 로컬 주소: `http://localhost:8000`

이 문서는 프론트엔드와 일반 학습 백엔드가 모르미 AI 대화 백엔드를 연동할 때
필요한 계약을 한곳에 정리한 문서입니다. 기계 판독용 전체 명세는
[`openapi.json`](./openapi.json)을 사용합니다.

## 1. 연결 구조

```text
브라우저
  → Next.js BFF
      → FastAPI AI 대화 백엔드
```

브라우저가 FastAPI를 직접 호출하지 않습니다. 서비스 키는 Next.js 서버 또는
일반 학습 백엔드에서만 보관합니다.

### 공통 헤더

```http
Content-Type: application/json
X-Mormi-Service-Key: <service-key>
```

- `GET /health`는 인증 없이 사용할 수 있습니다.
- `MORMI_SERVICE_API_KEY`가 설정된 환경에서는 모든 `/v1/*` 요청에
  `X-Mormi-Service-Key`가 필요합니다.
- 브라우저 번들과 `NEXT_PUBLIC_*` 환경 변수에는 서비스 키를 넣지 않습니다.

## 2. 엔드포인트 요약

| Method | Path | 역할 |
|---|---|---|
| GET | `/health` | 서버·LLM·DB 상태 확인 |
| POST | `/v1/practice-results` | 반복학습 결과 스냅샷 저장 |
| POST | `/v1/conversations` | AI 대화 시작 |
| POST | `/v1/conversations/{conversation_id}/responses` | 아이 응답 제출 및 다음 턴 생성 |
| GET | `/v1/conversations/{conversation_id}` | 최신 턴 복구 |
| GET | `/v1/learners/{learner_id}/skill-profiles` | 학습자별 시작 발화 단계 정보 조회 |
| GET | `/v1/learners/{learner_id}/star-notes` | 별노트 조회 |
| GET | `/v1/conversations/{conversation_id}/transcript` | 보호된 대화 기록 조회 |

## 3. 상태 확인

### `GET /health`

응답 `200 OK`:

```json
{
  "status": "ok",
  "llm_configured": true,
  "database": "postgresql"
}
```

`llm_configured=false`이면 선택·조작 기반 결정형 턴은 처리할 수 있지만 자유 발화
분류에는 Claude API 키 설정이 필요합니다.

## 4. 반복학습 결과 저장

### `POST /v1/practice-results`

일반 학습 백엔드가 반복학습 결과를 미리 저장할 때 사용합니다. 대화 시작 시
`practice_result_id`만 전달하려면 이 API에 먼저 저장되어 있어야 합니다.

요청:

```json
{
  "practice_result_id": "practice_123",
  "learner_id": 1,
  "skill_id": "basic_addition",
  "question_count": 5,
  "first_try_correct_count": 3,
  "wrong_attempt_count": 2,
  "earned_reward": 850,
  "misconception_tags": ["count_all_error"],
  "attempts": [
    {
      "item_id": "add_01",
      "correct": true,
      "latency_ms": 3400
    }
  ]
}
```

응답 `201 Created`: 저장된 객체를 그대로 반환합니다.

`attempts`와 집계 필드는 함께 또는 집계 필드만 전달할 수 있습니다. 아이 이름,
문제 원문, 음성 파일은 전달하지 않습니다.

## 5. 대화 시작

### `POST /v1/conversations`

지원 시나리오:

| `scene` | `scenario_id` | 설명 |
|---|---|---|
| `home_teach` | `home_addition_teach` | 반복한 덧셈을 모르미에게 가르치기 |
| `cafe` | `cafe_queue_demo` | 카페 줄 서기 단일 시나리오 |
| `cafe` | `cafe_outing` | 줄 서기·메뉴값·거스름돈·종합 수행 |

### 반복 결과가 이미 저장된 경우

```json
{
  "learner_id": 1,
  "scene": "home_teach",
  "scenario_id": "home_addition_teach",
  "learning_session_id": "session_123",
  "practice_result_id": "practice_123",
  "conversation_storage_consent": false,
  "retention_policy": "no_raw"
}
```

### MVP에서 요약 스냅샷을 함께 보내는 경우

```json
{
  "learner_id": 1,
  "scene": "home_teach",
  "scenario_id": "home_addition_teach",
  "learning_session_id": "session_123",
  "practice_result_id": "practice_123",
  "practice_summary": {
    "skill_id": "basic_addition",
    "question_count": 5,
    "first_try_correct_count": 3,
    "wrong_attempt_count": 2,
    "earned_reward": 850,
    "misconception_tags": ["count_all_error"]
  },
  "conversation_storage_consent": true,
  "retention_policy": "30_days"
}
```

`practice_summary` 안에는 `learner_id`와 `practice_result_id`를 반복하지 않습니다.
바깥 필드를 단일 출처로 사용합니다.

원문 저장 정책 조합:

| 동의 | `retention_policy` | 결과 |
|---|---|---|
| `false` | `no_raw` | 아이 원문을 저장하지 않음 |
| `true` | `30_days` | 암호화된 아이 원문을 30일 보관 |
| `true` | `90_days` | 암호화된 아이 원문을 90일 보관 |

그 외 조합은 `422`입니다.

응답 `201 Created`:

```json
{
  "conversation_id": "conversation_abc",
  "turn": {
    "turn_id": "turn_001",
    "scene": "home_teach",
    "scenario_id": "home_addition_teach",
    "task_id": "home_teach_3_plus_5",
    "stage_id": "home_teach",
    "task_index": 0,
    "mormi": {
      "text": "모두 얼마일까? 어떻게 계산했는지도 알려줘.",
      "mood": "curious",
      "max_lines": 2
    },
    "input": {
      "kind": "text",
      "choices": [],
      "target_slots": ["sum", "reason"]
    },
    "visual": {
      "type": "vertical_equation",
      "data": {}
    },
    "help_card": null,
    "note_update": null,
    "status": "active",
    "state_version": 1,
    "completion": null,
    "pedagogy": null
  }
}
```

응답에 표시되는 구체적인 질문과 입력 방식은 학습자 프로필과 반복 결과에 따라
달라질 수 있습니다. 프론트는 특정 시작 단계나 문구를 가정하지 않습니다.

## 6. 아이 응답 제출

### `POST /v1/conversations/{conversation_id}/responses`

공통 필드:

```json
{
  "turn_id": "turn_001",
  "response_id": "9cda3c1e-6539-4b35-9ac5-c63f91e203b1",
  "type": "text",
  "text": "3개랑 5개를 합치면 8개야",
  "choice_ids": [],
  "values": {},
  "asr_confidence": null,
  "latency_ms": 4200
}
```

- `turn_id`: 화면에 표시 중인 최신 턴 ID
- `response_id`: 사용자 행동마다 프론트가 생성하는 UUID 멱등키
- `latency_ms`: 턴을 본 뒤 응답하기까지 걸린 시간
- 네트워크 재시도에는 같은 `turn_id`와 `response_id`를 사용

입력 방식별 요청:

| 이전 턴 `input.kind` | 요청 `type` | 필수 데이터 |
|---|---|---|
| `text` | `text` | `text` |
| `choices` | `choice` | `choice_ids` |
| `fill` | `fill` | `choice_ids` |
| `count` | `count` | `values` |
| `equation` | `equation` | `values` |
| `joint` | `action` | `values` |
| `button` | `action` | `values` |
| `none` | 전송하지 않음 | 완료 연출만 표시 |

선택형 예시:

```json
{
  "turn_id": "turn_002",
  "response_id": "d6b10425-77ed-45c4-a930-85de0b9d9f30",
  "type": "choice",
  "choice_ids": ["left"],
  "latency_ms": 1800
}
```

프론트는 화면 문구가 아닌, 직전 턴에서 받은 `choice.id`를 그대로 보냅니다.

응답 `200 OK`: 대화 시작과 동일한 `{ conversation_id, turn }` 구조로 다음 턴을
반환합니다.

## 7. `TurnContract`

```ts
type TurnContract = {
  conversation_id: string;
  turn: {
    turn_id: string;
    scene: "home_teach" | "cafe";
    scenario_id: string;
    task_id: string;
    stage_id: string;
    task_index: number;
    mormi: {
      text: string; // 최대 50자, 최대 두 줄
      mood: "curious" | "listening" | "thinking" | "relieved" | "celebrating";
      max_lines: 1 | 2;
    };
    input: {
      kind: "text" | "choices" | "fill" | "count" | "equation" | "joint" | "button" | "none";
      placeholder?: string;
      choices: Array<{
        id: string;
        label: string;
        image_url?: string;
        disabled?: boolean;
      }>;
      target_slots: string[];
      submit_label?: string;
      config: Record<string, unknown>;
    };
    visual: {
      type: string;
      data: Record<string, unknown>;
    };
    help_card: null | {
      visible: boolean;
      auto_open: boolean;
      level: "H0" | "H1" | "H2" | "H3";
      title: string;
      body: string;
      visual_type?: string;
      visual_data: Record<string, unknown>;
    };
    note_update: null | {
      note_id: string;
      skill_id: string;
      text: string;
      attribution: "child" | "coauthored";
      evidence: "direct_explanation" | "supported_completion";
      attribution_label: string;
    };
    status: "active" | "completed";
    state_version: number;
    completion: null | {
      outcome: "taught" | "supported" | "bright_exit";
      teach_reward_eligible: boolean;
    };
    pedagogy?: unknown;
  };
};
```

프론트 처리 원칙:

- `input.kind`에 해당하는 입력 UI 하나만 활성화합니다.
- `help_card.auto_open=true`이면 도움 카드를 즉시 엽니다.
- `note_update`가 있을 때만 별노트를 추가합니다.
- 정오, 오개념, L/H 전환, 별노트 귀속을 프론트가 다시 계산하지 않습니다.
- `status=completed`이면 입력을 보내지 않고 완료 연출로 이동합니다.
- 보상 여부는 `completion.teach_reward_eligible`만 사용합니다.

시각자료별 필드는 [`VISUAL_CONTRACTS.md`](./VISUAL_CONTRACTS.md)를 참고합니다.

## 8. 완료와 보상

```json
{
  "status": "completed",
  "completion": {
    "outcome": "supported",
    "teach_reward_eligible": true
  }
}
```

| `outcome` | 의미 | 가르치기 보상 |
|---|---|---|
| `taught` | 아이가 독립적인 문장 설명으로 완료 | 가능 |
| `supported` | 선택·조작·도움 카드 지원을 받아 완료 | 가능 |
| `bright_exit` | 안전하게 종료했지만 가르치기 완료는 아님 | 불가 |

실제 보상 지급과 중복 방지는 일반 학습 백엔드가 담당합니다. AI 대화 백엔드는
지갑을 직접 변경하지 않습니다.

## 9. 최신 턴 복구

### `GET /v1/conversations/{conversation_id}`

새로고침, 네트워크 복구, `409` 발생 후 최신 턴을 가져올 때 사용합니다.

응답 `200 OK`: `{ conversation_id, turn }`

프론트는 로컬 대화 기록보다 이 응답을 최신 상태의 단일 출처로 사용합니다.

## 10. 학습자 상태와 별노트

### `GET /v1/learners/{learner_id}/skill-profiles`

```json
{
  "learner_id": 1,
  "skills": [
    {
      "skill_id": "basic_addition",
      "highest_stable_expression_level": "L2",
      "h0_success_streak": 1,
      "recent_max_hint": "H1",
      "concept_mastery": 0.6,
      "expression_independence": 0.5,
      "last_bottleneck": "expression"
    }
  ]
}
```

이 값은 다음 대화의 시작 발화 단계와 힌트 의존도 조정에 사용됩니다. 아동
화면에 직접 노출하지 않습니다.

### `GET /v1/learners/{learner_id}/star-notes`

```json
{
  "learner_id": 1,
  "notes": [
    {
      "note_id": "note_123",
      "skill_id": "basic_addition",
      "text": "3과 5를 더하면 8이 된다.",
      "attribution": "child",
      "evidence": "direct_explanation",
      "attribution_label": "아이가 알려줌"
    }
  ]
}
```

- `child`: 아이가 독립적으로 완결된 사실 문장을 설명
- `coauthored`: 선택·빈칸·조작·도움 카드로 함께 완성
- 프론트는 노트 본문과 귀속을 새로 만들거나 수정하지 않습니다.

## 11. 보호된 대화 기록

### `GET /v1/conversations/{conversation_id}/transcript`

운영자 분석용 보호 엔드포인트입니다. 일반 아동 화면에서 호출하지 않습니다.

- 모르미 질문은 화면 복구를 위해 항상 암호화 저장됩니다.
- 아이 원문은 저장 동의가 있을 때만 암호화 저장됩니다.
- 동의가 없으면 선택 ID, 응답 유형, 분류 결과 등 구조 데이터만 남습니다.
- 음성 파일은 저장하지 않습니다.

## 12. 오류 계약

| HTTP | 의미 | 프론트 처리 |
|---:|---|---|
| `401` | 서비스 키 누락·불일치 | BFF 환경 변수 확인 |
| `404` | 대화 또는 시나리오 없음 | 시작 화면으로 복구하거나 개발 로그 기록 |
| `409` | 이미 답한 턴 또는 오래된 `turn_id` | 최신 턴 GET 후 화면 복구 |
| `422` | 요청 필드·정책 조합 오류 | 입력을 보존하고 개발 로그 기록 |
| `503` | LLM 호출 실패 | 상태는 그대로이므로 같은 `response_id`로 재시도 |

`409 Conflict`:

```json
{
  "detail": {
    "code": "stale_turn",
    "message": "turn_id is stale; use the latest turn",
    "conversation_id": "conversation_abc",
    "turn_id": "turn_latest",
    "state_version": 4
  }
}
```

## 13. 멱등성과 중복 클릭

- 프론트는 사용자 행동마다 UUID `response_id`를 한 번 생성합니다.
- 전송 즉시 해당 입력 버튼을 잠급니다.
- 네트워크 실패 후에는 같은 `response_id`로 재전송합니다.
- 백엔드는 기본 30일 동안 같은 `response_id`에 최초 결과 턴을 반환합니다.
- `note_update`와 상태 진행도 중복 생성되지 않습니다.
- 가르치기 보상 중복 방지는 일반 학습 백엔드에서
  `learning_session_id + conversation_id` 기준으로 처리합니다.

## 14. 관련 문서

- [전체 아키텍처](./ARCHITECTURE.md)
- [시각자료 계약](./VISUAL_CONTRACTS.md)
- [시각자료 JSON Schema](./visual-contract.schema.json)
- [OpenAPI JSON](./openapi.json)
