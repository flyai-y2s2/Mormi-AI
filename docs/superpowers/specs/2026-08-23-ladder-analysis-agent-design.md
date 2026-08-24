# 발화 사다리 분석 에이전트 설계

## 1. 목표

아이의 현재 발화 단계, 모르미 질문, 아이 발화와 학습 근거를 이용해 발화별 적정
단계를 예측하고, 한 주의 예측과 정답률을 종합해 교사에게 다음 주 시작 단계를
추천한다. 추천은 자동 적용하지 않으며 교사가 승인한 경우에만 다음 주부터 적용한다.

## 2. 프로토타입 범위

- 활성 발화 단계는 `L4 → L3 → L2 → L0`이다. 과거 `L1`은 `L2`로 정규화한다.
- 16명의 시뮬레이션 학습자를 네 유형으로 나누고 7일간 매일 5~8개의 유효 발화를 만든다.
- 약 560~896개 발화를 익명화 JSONL로 만들고 학습자 단위로 10/3/3명 train/validation/test에 나눈다.
- `klue/roberta-base` 기반 순서형 분류 모델을 파인튜닝한다.
- 모델은 발화 한 건의 적정 단계를 예측한다.
- LangGraph 분석 에이전트는 발화 예측과 BE의 정답·도움 통계를 주차별로 집계한다.
- 교사는 리포트에서 추천 근거를 확인하고 승인 또는 유지 결정을 한다.

실제 아동에게 일반화되는 성능 입증, 완전 자동 승급, 온라인 재학습은 프로토타입
범위에 포함하지 않는다.

## 3. 핵심 개념

### 3.1 주간 승인 단계

`approved_level`은 학습자·skill·주차별 기본 시작 단계다. 같은 주에는 바뀌지 않는다.
아이가 한 문제에서 막히면 해당 문제만 한 단계씩 내려갈 수 있으며, 다음 문제는 다시
그 주의 승인 단계에서 시작한다.

### 3.2 발화 단계와 개념 수행의 분리

- 온전한 문장으로 틀린 계산을 설명한 발화는 표현 `L4`, 개념 `incorrect`다.
- 정답만 짧게 말한 발화는 표현 `L3`, 개념 `correct`일 수 있다.
- 표현이 어려우면 L을 낮추고, 개념이 어려우면 H를 높인다.
- 안전 발화, 인식 실패, 시스템 오류는 단계 학습과 추천에서 제외한다.

### 3.3 모델과 에이전트의 분리

모델은 한 발화의 `recommended_level`과 단계별 점수를 출력한다. 분석 에이전트는
여러 발화의 예측, 현재 단계 정답률, 독립 수행률과 도움 사용률을 종합해
`PROMOTE`, `HOLD`, `REVIEW_DOWN`, `INSUFFICIENT_EVIDENCE` 중 하나를 출력한다.

## 4. 모델 데이터 계약

### 4.1 입력

```json
{
  "sample_id": "turn_123",
  "learner_key": "anon_a81f",
  "skill_id": "money-count",
  "week_start": "2026-08-17",
  "current_level": "L4",
  "hint_level": "H0",
  "response_mode": "free_text",
  "question_intent": "ask_answer_and_method",
  "mormi_question": "왜 1,000원인지 알려줄 수 있어?",
  "child_utterance": "500원짜리 두 개요.",
  "concept_result": "correct",
  "response_category": "correct_partial"
}
```

`learner_type`, 실제 learner ID, 이름, 이메일과 전화번호는 모델 입력에 포함하지 않는다.

### 4.2 사람 라벨

```json
{
  "target_level": "L3",
  "annotation_reason": "답은 직접 생성했지만 이유를 연결해 설명하지 않았다.",
  "annotator_id": "reviewer_01",
  "rubric_version": "ladder-label-v1"
}
```

기존 규칙 엔진의 `expression_after`는 `legacy_expression_after`로만 내보내며 정답
라벨로 사용하지 않는다.

### 4.3 모델 출력

```json
{
  "recommended_level": "L3",
  "level_scores": {"L0": 0.02, "L2": 0.11, "L3": 0.73, "L4": 0.14},
  "confidence": 0.73,
  "model_version": "ladder-klue-v1"
}
```

점수는 합이 1인 추론 점수지만 실제 확률로 과장하지 않는다. 검증 세트에서 별도
calibration을 수행한 경우에만 `probabilities`라는 이름을 사용한다.

## 5. 라벨 기준

| 단계 | 관찰 기준 |
|---|---|
| L4 | 선택지나 문장 틀 없이 답과 이유·방법을 자신의 말로 연결한다. |
| L3 | 정답이나 핵심 방법을 직접 생성하지만 짧거나 요구 슬롯 일부만 표현한다. |
| L2 | 독립 생성은 어렵지만 제공된 선택지에서 답이나 방법을 고른다. |
| L0 | 선택지 지원으로도 진행하기 어려워 도움 카드와 공동 수행이 필요하다. |

`몰라` 한 번만으로 L0를 붙이지 않는다. 현재 질문을 짧게 바꾼 뒤에도 의미 있는
응답이 없거나, 선택지 지원 이후에도 수행이 불가능한 근거가 있어야 한다.

## 6. 모델 설계

- tokenizer/model: `klue/roberta-base`
- 입력 직렬화:
  `[LEVEL=L4] [HINT=H0] [MODE=free_text] [INTENT=...] 질문 [SEP] 아이 발화`
- 순서 인덱스: `L0=0`, `L2=1`, `L3=2`, `L4=3`
- head: 세 개의 누적 경계 `P(y>L0)`, `P(y>L2)`, `P(y>L3)`를 학습하는 ordinal head
- loss: 각 경계의 binary cross entropy 평균
- 비교 기준: 다수 클래스 기준선과 일반 4-class classification
- split: learner 단위 10/3/3, 같은 학습자의 데이터는 한 split에만 존재
- seed와 split manifest를 artifact에 저장

프로토타입 평가는 macro F1, 단계 MAE, quadratic weighted kappa, 두 단계 이상 벗어난
위험 오차율과 confusion matrix를 모두 보고한다. 16명 데이터의 점수는 실제 서비스
일반화 성능으로 표현하지 않는다.

## 7. 분석 에이전트

LangGraph 노드는 다음 순서로 고정한다.

```text
collect_evidence
  → predict_utterances
  → aggregate_week
  → recommend_level
  → validate_recommendation
```

### 7.1 프로토타입 승급 규칙

현재 단계에서 다음 조건을 모두 만족하면 한 단계만 승급 추천한다.

- 유효 수행 10회 이상
- 현재 단계 정답률 90% 이상
- 독립 수행률 70% 이상
- 모델이 다음 단계 이상으로 예측한 비율 70% 이상
- 안전·인식 실패·시스템 오류를 분모에서 제외

조건이 부족하면 `HOLD` 또는 `INSUFFICIENT_EVIDENCE`다. 현재 단계보다 낮은 모델 예측이
60% 이상이고 표현 막힘이 서로 다른 문제에서 2회 이상이면 `REVIEW_DOWN`을 표시하지만
자동 강등하지 않는다. 추천은 항상 한 단계만 움직인다.

### 7.2 추천 출력

```json
{
  "skill_id": "money-count",
  "week_start": "2026-08-17",
  "current_level": "L2",
  "recommended_level": "L3",
  "action": "PROMOTE",
  "eligible_count": 12,
  "accuracy": 0.92,
  "independent_rate": 0.75,
  "next_level_prediction_rate": 0.75,
  "evidence_turn_ids": ["turn_1", "turn_2"],
  "model_version": "ladder-klue-v1",
  "policy_version": "weekly-ladder-v1"
}
```

추천 문장은 숫자와 증거를 BE가 정해진 템플릿으로 렌더링한다. 생성형 LLM이 수치를
새로 만들거나 교사 승인 여부를 결정하지 않는다.

## 8. 서비스 경계와 데이터 흐름

1. AI DB의 `turns`, `conversations`, `dialogue_turn_observations`를 기존 codec과 보존
   정책을 통해 읽고 익명화 JSONL을 만든다.
2. 오프라인 훈련이 versioned artifact를 생성한다.
3. BE가 리포트 생성 시 skill별 정답·도움 통계를 AI 내부 분석 API에 전달한다.
4. AI 분석 에이전트가 같은 기간의 보존 가능한 발화를 읽고 모델 추론·집계 결과를 반환한다.
5. BE가 추천과 모델·정책 버전을 저장하고 리포트 DTO에 포함한다.
6. 교사가 승인하면 BE가 다음 주부터 유효한 skill별 시작 단계를 저장한다.
7. 다음 대화 시작 시 BE가 승인 단계를 서버 간 요청에 넣고 AI가 해당 단계에서 시작한다.

브라우저는 learner ID의 소유권, 서비스 키, 모델 artifact 경로나 승인 단계의 유효 시작일을
직접 결정하지 않는다.

## 9. 실패 처리

- 모델 artifact가 없거나 로딩에 실패하면 AI는 `analysis_unavailable`을 반환한다.
- 유효 발화가 10개 미만이면 `INSUFFICIENT_EVIDENCE`를 반환한다.
- 모델 실패를 기존 규칙 기반 추천으로 위장하지 않는다.
- BE가 AI 분석을 받지 못해도 기존 진단 리포트의 다른 영역은 표시한다.
- 교사 승인 API는 이미 처리된 추천에 대해 멱등적으로 같은 결과를 반환한다.
- 승인되지 않은 추천과 만료된 추천은 대화 시작 단계에 영향을 주지 않는다.

## 10. 개인정보와 운영 안전

- 운영 DB 추출은 읽기 전용 연결과 기간·학습자 allowlist를 요구한다.
- 실제 learner ID는 실행 시 제공한 salt로 HMAC 처리한다.
- 원문이 없는 데이터는 복호화를 우회하지 않고 제외한다.
- 추출 파일과 모델 artifact는 Git에서 제외한다.
- 학습 로그와 평가 보고서에는 원문, 실제 ID와 접속 정보를 출력하지 않는다.
- 실제 아동 데이터는 동의 및 조직의 데이터 이용 절차가 확인된 경우에만 사용한다.

## 11. 완료 기준

- 16명 7일 데이터가 learner 누수 없이 JSONL과 split manifest로 생성된다.
- ordinal 모델을 재현 가능한 명령으로 훈련하고 평가 보고서를 만든다.
- 네 대표 유형을 포함한 고정 fixture에서 기대 단계를 예측한다.
- 주간 조건을 만족한 L2 학습자에게만 L3 추천이 생성된다.
- 교사의 승인 전에는 시작 단계가 변하지 않고, 승인 다음 주부터만 반영된다.
- AI, BE, FE의 기존 테스트와 새 계약 테스트가 모두 통과한다.
