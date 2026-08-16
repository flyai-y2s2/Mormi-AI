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
| `note_evidence_links` | 별노트와 그 내용을 만든 관찰 턴 연결 |
| `ai_outbox_events` | Spring BE 전달을 위한 내구성 있는 이벤트 원본 |

아이의 질문·원문 발화는 기존 `turns`에 암호화해 보관한다. 관찰 테이블에는
`grounding_span`, `note_candidate`, 평문 `evidence_span`을 복사하지 않는다.
슬롯의 근거 구절은 저장 동의가 있는 경우에만 별도로 암호화한다.

## 핵심 필드

- 응답: `response_type`, `response_category`, `difficulty_class`, `concept_result`
- 교육 상태: `expression_before/after`, `hint_before/after`, `transition_reason`
- 근거: `claims[].semantic_role`, `validation_status`, `newly_verified`
- 지원: 도움 카드 표시·자동 열림·단계
- 안전성: `safety_category`, `speaker_source`, `verifier_status`, `fallback_reason`
- 한계: `adult_intervention_status`는 현재 UI가 전달하지 않으므로 `not_collected`
- 재현성: 대화 정책·콘텐츠·사전·분류기·화자 모델 버전

`concept_result=not_assessed`는 도움 요청, 입력 오류, 장난 등 수학 개념을 평가할 수
없는 응답이다. 리포트에서 이를 오답으로 합산하면 안 된다.

## Outbox 이벤트

턴 저장, 관찰 저장, 별노트 근거 연결, Outbox 삽입은 한 DB 트랜잭션에서 처리된다.
따라서 네트워크가 끊겨도 관찰 이벤트 자체는 유실되지 않는다.

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

Spring 수신 API가 확정되기 전까지 이벤트는 `pending`으로 남는다. 전송기는
`event_id`를 멱등 키로 사용해야 하며, 수신 API 작업 없이 임의의 URL로 보내지 않는다.

## 기존 데이터 보존과 마이그레이션

스키마 변경은 기존 테이블을 수정·삭제하지 않고 새 테이블만 추가한다.

```bash
python scripts/migrate_database.py
python scripts/backfill_observations.py
```

운영에서는 먼저 DB 백업을 확인하고 마이그레이션을 1회 실행한 뒤 애플리케이션을
교체한다. 이 PR은 운영 배포 단계에서 자동 실행하지 않는다.

과거 턴 백필 원칙:

- 이미 저장된 `response_category`, L/H, 도움 카드 노출은 옮긴다.
- 당시 저장하지 않은 claims, bottleneck, confidence, dialogue act는 추정하지 않는다.
- 누락값은 `not_collected`, 출처는 `historical_backfill`로 표시한다.
- 과거 아이 발화를 LLM으로 재분석하지 않는다.
- 백필 이벤트는 자동으로 Spring에 발행하지 않는다.

## 리포트 사용 원칙

- 단일 오답은 오개념 확정이 아니라 `병목 후보`다.
- 후보마다 `evidence_observation_ids`를 제시한다.
- `adult_intervention_status=not_collected`를 `개입 없음`으로 해석하지 않는다.
- 실제 생활 전이 과제가 아니라면 `전이 성공` 대신 `새 숫자 적용`으로 표현한다.
- 모든 요약에는 `단일 세션의 수행 관찰이며 진단이 아님`을 표시한다.
