# 대화 관찰 데이터와 리포트 이벤트 계약

## 목적

이 계층은 교사에게 진단을 내리기 위한 것이 아니라, 실제 대화에서 관찰된 사실과
그 근거를 잃지 않기 위한 저장 계약이다. 한 번의 오답으로 오개념을 확정하지 않고,
리포트 문장은 반드시 원본 턴과 구조화된 관찰값으로 되돌아갈 수 있어야 한다.

## 저장 단위

| 테이블 | 역할 |
|---|---|
| `dialogue_turn_observations` | 한 응답의 판정, L/H 전후, 전환 이유, fallback 및 버전 |
| `dialogue_claims` | 분류기가 찾은 슬롯 주장과 코드 검증 결과 |
| `dialogue_task_outcomes` | 과제 완료 시 지원 수준과 근거 관찰 ID 묶음 |
| `note_evidence_links` | 별노트와 그 내용을 만든 모든 관찰 턴·슬롯 연결 |
| `ai_outbox_events` | Spring BE 전달을 위한 내구성 있는 이벤트 원본 |

아이의 질문·원문 발화는 기존 `turns`에 평문으로 보관한다. 슬롯의 근거 구절도
저장 동의가 있는 경우 `dialogue_claims`에 평문으로 보관한다. 물리 컬럼의
`*_encrypted` 이름은 기존 PostgreSQL 스키마 호환을 위해 유지된다. Outbox 분석
이벤트에는 `grounding_span`, `note_candidate`, 원문 `evidence_span`을 복사하지 않는다.

## 핵심 필드

- 응답: `response_type`, `response_category`, `difficulty_class`, `concept_result`
- 교육 상태: `expression_before/after`, `hint_before/after`, `transition_reason`
- 근거: `claims[].semantic_role`, `validation_status`, `newly_verified`
- 지원: 도움 카드 표시·자동 열림·단계
- 안전성: `safety_category`, `speaker_source`, `verifier_status`, `fallback_reason`
- 재현성: 대화 정책·콘텐츠·사전·분류기·화자 모델 버전

카페·놀이동산의 life V3 관찰은 task 전환 전후를 섞지 않도록 다음 서버 소유 metadata를
`runtime_json`에 추가한다. 아이 원문이나 evidence span은 이 metadata에 넣지 않는다.

- schema discriminator `observation_runtime_schema=life-v3-observation-runtime-v1`
- scenario pack ID/version/source hash
- 응답을 받은 source task의 ID/index/variant/content identity
- 결과 턴의 task ID/index/variant/content identity와 `task_transitioned`
- reasoning-ledger schema, 검증된 fact/relation ID, auxiliary evidence 개수
- 별노트 발행 여부와 source task/귀속/evidence/relation provenance

`versions_json`에는 `dialogue_scenario_pack`, `dialogue_scenario_content_version`,
`dialogue_scenario_source_hash`, `dialogue_task_variant`, `reasoning_ledger_schema`를 남긴다.
같은 `observation_runtime_schema` discriminator도 함께 남긴다.
저장소는 pinned scenario/task identity와 runtime 결과가 다르면 관찰을 추측해 저장하지 않고
트랜잭션을 fail closed한다.

`concept_result=not_assessed`는 도움 요청, 입력 오류, 장난 등 수학 개념을 평가할 수
없는 응답이다. 리포트에서 이를 오답으로 합산하면 안 된다.

## Outbox 이벤트

턴 저장, 관찰 저장, 별노트 근거 연결, Outbox 삽입은 한 DB 트랜잭션에서 처리된다.
따라서 네트워크가 끊겨도 관찰 이벤트 자체는 유실되지 않는다.

외래키 부모는 `conversation → turn`, `observation → claim`, `note → evidence link`
순서로 명시적으로 먼저 저장한다. 별노트가 답과 설명처럼 여러 턴에서 완성되면 마지막
턴 하나만 가리키지 않고, `newly_verified`로 확인된 각 슬롯의 관찰을 모두 연결한다.
과제 전환으로 다음 과제의 슬롯 상태가 초기화되더라도 source task의 검증 결과는
사라지지 않는다. V3 별노트의 `task_id`와 evidence link도 결과 턴의 새 task가 아니라 노트를
실제로 만든 source task를 가리킨다.

`dialogue_claims.validation_status`는 저장소가 분류기 원시 claim을 다시 판정해 만들지
않는다. 상태 머신의 근거 검사·현재 질문 슬롯 제한까지 모두 통과한
`PedagogicalDecision.accepted_claims`만 `verified`로 저장한다. 따라서 화면에서는 거절된
과잉 claim이 분석·Outbox·교사용 리포트에서 다시 사실로 살아날 수 없다.

```json
{
  "schema_version": 1,
  "observation_id": "observation_...",
  "conversation_id": "conversation_...",
  "learner_id": 17,
  "scene": "home_teach",
  "scenario_id": "home_teach",
  "task_id": "home_teaching",
  "response_category": "correct_partial",
  "difficulty_class": "unknown",
  "concept_result": "correct_partial",
  "expression_before": "L4",
  "expression_after": "L4",
  "hint_before": "H0",
  "hint_after": "H0",
  "transition_reason": "acknowledge_partial",
  "claims": [
    {
      "slot_id": "answer",
      "semantic_role": "conclusion",
      "value": "600원",
      "factual": true,
      "validation_status": "verified",
      "newly_verified": true
    }
  ]
}
```

AI 프로세스의 백그라운드 전송기는 대화 응답 경로와 분리되어 `pending` 또는 재시도
시각이 지난 `retry` 이벤트를 Spring으로 전달한다.

```text
POST {MORMI_OBSERVATION_INGEST_URL}
X-Mormi-Service-Key: {MORMI_OBSERVATION_INGEST_KEY}
```

```json
{
  "event_id": "event_...",
  "schema_version": 1,
  "event_type": "dialogue_observation",
  "observation": { "observation_id": "observation_..." }
}
```

### 별노트 독립 이벤트(B안)

별노트는 관찰 이벤트 안의 부수 문장이 아니라 별도 생명주기를 가진 서비스 데이터다.
AI는 문장·귀속·근거를 생성하고, Spring BE는 멱등 수집한 별노트를 서비스용 원장과 조회
API로 제공한다. FE가 AI의 별노트 조회 API를 직접 호출하거나 정적 문구를 조합하지 않는다.

별노트 생성, 근거 링크, 아래 outbox row는 한 DB 트랜잭션에 저장된다. 별노트 표시 문장은
서비스 화면에 필요한 검수 결과이므로 포함하지만, 아이 원문 발화와 raw evidence span은
포함하지 않고 관찰 ID와 슬롯 ID만 전달한다.

```json
{
  "event_id": "event_...",
  "schema_version": 1,
  "event_type": "star_note_created",
  "star_note": {
    "note_id": "note_...",
    "note_version": 1,
    "learner_id": 17,
    "conversation_id": "conversation_...",
    "learning_session_id": "session_...",
    "scene": "home_teach",
    "scenario_id": "home_teach",
    "task_id": "home_teaching",
    "stage": "home_teaching",
    "task_index": 0,
    "skill_id": "number-count",
    "text": "색칠된 칸을 하나씩 세면 모두 3개야.",
    "attribution": "child",
    "attribution_label": "아이가 알려줌",
    "evidence": "direct_explanation",
    "evidence_links": [
      {
        "observation_id": "observation_...",
        "source_slot_ids": ["tracking"]
      }
    ],
    "active": true,
    "created_at": "2026-08-19T00:00:00+00:00"
  }
}
```

`event_id`는 전송 멱등 키, `note_id`는 별노트 원장의 멱등 키다. 이벤트 전달 순서는
보장하지 않으므로 Spring은 별노트 이벤트가 관찰 이벤트보다 먼저 도착해도 수용해 나중에
근거를 연결하거나, `409 unknown_observation`을 반환해 재시도시켜야 한다. 향후 수정·비활성화
이벤트를 추가할 수 있도록 `note_version`을 둔다.

Spring 수신 계약이 먼저 배포되지 않은 환경에서는
`MORMI_STAR_NOTE_EVENTS_ENABLED=false`를 유지한다. AI는 별노트 이벤트를 pending으로
보존하되 claim하지 않는다. Spring 배포 후 값을 `true`로 바꾸면 적체분부터 전송한다.

`event_id`가 멱등 키이므로 응답 유실 뒤 같은 이벤트를 다시 보내도 안전하다. 상태 전이는
다음과 같다.

| 결과 | Outbox 처리 |
|---|---|
| `200` (`duplicate: true` 포함) | `sent`, 전달 시각 기록 |
| `409 unknown_conversation`, `unknown_learner`, `unknown_observation`, `missing_evidence_observation` | 선행 데이터 도착을 기다리며 지수 백오프 뒤 `retry` |
| `429`, `5xx`, timeout·network error | 지수 백오프 뒤 `retry` |
| `422` 및 재시도로 해결되지 않는 `4xx` | `failed`, 자동 재시도 중단 |
| `401`/`403` | 현재 이벤트를 `retry`로 돌리고 worker 중단; 설정 수정 후 재시작 |

여러 AI 인스턴스가 동시에 떠도 `FOR UPDATE SKIP LOCKED`로 이벤트를 한 worker만
가져간다. `processing` 상태의 `available_at`은 lease 만료 시각이며, worker가 전송 중
죽으면 다른 worker가 만료 이벤트를 다시 회수한다. attempt 번호를 세대 토큰으로 검사해
늦게 돌아온 이전 worker가 새 처리 결과를 덮어쓰지 못하게 한다.

전송 로그에는 이벤트 ID, attempt, 상태, 안전한 오류 코드만 남기며 서비스 키, 아이 원문,
전체 payload는 기록하지 않는다. `MORMI_OBSERVATION_INGEST_URL`과
`MORMI_OBSERVATION_INGEST_KEY`가 모두 설정된 경우에만 worker가 시작된다.

## 기존 데이터 보존과 마이그레이션

관찰 스키마는 기존 행을 삭제하지 않는다. 대화 identity는 별도의 expand-contract revision으로
전환하며, `20260826_05`에서는 old+new unique를 함께 두고 `20260826_06`에서 reader capability
확인 뒤 old unique만 제거한다.

```bash
python scripts/migrate_database.py
python scripts/backfill_observations.py
```

운영에서는 먼저 DB 백업을 확인한다. `develop` 배포 workflow가 이전 live의 exact
scenario-aware reader capability를 검사해 첫 배포는 transition revision까지만 적용하고
canary/env를 0으로 고정한다. reader-capable 이미지가 live임을 확인한 다음 배포에서만 final
revision을 적용한다. 운영자가 직접 실행할 때도 같은 두 단계와 gate를 지켜야 한다.

기존 배포의 `Base.metadata.create_all()`이 새 테이블을 먼저 만든 경우에는, 다섯 관찰
테이블이 모두 현재 스키마와 함께 존재할 때만 Alembic `head`를 기록한다. 일부만 존재하는
불완전한 상태는 추측해서 고치지 않고 명시적으로 실패시켜 운영자 검토를 요구한다.

애플리케이션 시작 시에도 다섯 테이블의 필수 열·외래키·고유 제약·인덱스를 검사한다.
`create_all()`은 이미 존재하는 불완전한 테이블을 고치지 못하므로, 계약 불일치를 발견하면
아동의 실시간 응답을 받기 전에 서버 시작을 중단한다. 런타임 저장 오류는 중복 응답으로
위장하지 않고 트랜잭션을 rollback한 뒤 재시도 가능한 `persistence_failed`로 반환한다.

관찰 JSON에는 아이 원문을 복제하지 않는다. `analysis_json`은 닫힌 enum·검수 코드와
confidence만 저장하고, 모델 자유문장·산술 evidence span·reference resolution은 제외한다.
exact claim evidence는 consent-controlled column 하나에만 두며 기간 만료 시 turn raw,
structured response, claim evidence와 legacy state evidence를 같은 transaction으로 정리한다.

과거 턴 백필 원칙:

- 이미 저장된 `response_category`, L/H, 도움 카드 노출은 옮긴다.
- 당시 저장하지 않은 claims, bottleneck, confidence, dialogue act는 추정하지 않는다.
- 누락값은 `not_collected`, 출처는 `historical_backfill`로 표시한다.
- 과거 아이 발화를 LLM으로 재분석하지 않는다.
- 백필 이벤트는 자동으로 Spring에 발행하지 않는다.

## 리포트 사용 원칙

- 단일 오답은 오개념 확정이 아니라 `병목 후보`다.
- 후보마다 `evidence_observation_ids`를 제시한다.
- 실제 생활 전이 과제가 아니라면 `전이 성공` 대신 `새 숫자 적용`으로 표현한다.
- 모든 요약에는 `단일 세션의 수행 관찰이며 진단이 아님`을 표시한다.
