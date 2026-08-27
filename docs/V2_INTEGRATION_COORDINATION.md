# V2 시연 범위 통합 협업 메모

이 문서는 Mormi-AI의 네이티브 V2/V3 구현을 BE·FE와 연결할 때 필요한 변경만 정리한다.
이번 작업은 **Mormi-AI만 수정**하며 Mormi-BE, Mormi-FE, 교사용 분석 페이지와
발화사다리 분석 모델 코드는 수정하지 않는다.

## 현재 AI 구현 범위

- 집 준비: 카페 필수 5개와 놀이동산 준비 4개, 총 9개 단일-pack 세션
- 카페: `cafe_queue`, `cafe_budget_menu`, `cafe_menu_total`, `cafe_change`
  - 4개 시나리오, 총 5개 task
- 놀이동산: `amusement_ticket_multiply`, `amusement_snack_divide`,
  `amusement_pass_compare`
  - 3개 시나리오, 기본·전이 포함 총 6개 task
- 그 밖의 집 27개와 비지원 시나리오는 `legacy-v1`

집의 단순 반복 문제 배열·숫자 변형·정답·선택지는 이 브랜치에서 변경하지 않는다.
놀이동산도 기존의 동적 문제 범위와 세 장면의 뼈대를 유지한다. 콘텐츠 revision 2에서
바뀐 것은 간식 나누기와 자유이용권 비교의 방법 선택지·도움 문구뿐이다. 간식은
`간식 값 전체를 사람 수로 나눠`, 자유이용권은
`자유이용권 값을 1회 이용권 값으로 나눠`를 검수 정답으로 사용한다.

네이티브 선택에는 `MORMI_RUNTIME_CONTRACT_VERSION=verdict-v1`, canary 선택과 비어 있지
않은 `learning_session_id`가 모두 필요하다. 홈은 V2 단일-pack snapshot, 카페·놀이동산은
task-scoped V3 scenario snapshot에 고정된다. `/health`의 배포 하한은
`dialogue-v3-snapshot-reader-v1`이다.

새 이미지 asset은 필요하지 않다. AI가 내리는 카페·놀이동산 visual type과 image key는 기존
FE renderer와 asset을 그대로 사용한다. 이번 FE 작업은 새 그림 제작이 아니라 선택 UI·도움
카드 visibility·버튼 동작·별노트 귀속 표시의 계약 정렬이다.

## Mormi-BE에서 필요한 연동

### 안정적인 대화 identity

- 카페·놀이동산 시작 요청에도 방문/학습 단위를 나타내는 안정적인
  `learning_session_id`를 보낸다.
- 동일 시작 요청의 네트워크 재시도는 같은
  `(learner_id, learning_session_id, scene, scenario_id, conversation_round)`를 재사용한다.
- 사용자가 “새 문제로 다시 시작”을 누르면 해당 시나리오의 `conversation_round`를
  증가시킨다. 새로고침·전송 재시도만으로는 증가시키지 않는다.
- 완료된 놀이동산 방문도 이미 도달한 세 시나리오를 각각 다시 연습할 수 있어야 한다.
  `start_mode=restart`를 `park_visit_completed`로 거절하지 말고, 같은 방문·시나리오의
  마지막 round 다음 값으로 새 AI 대화를 만든다. 완료된 방문의 진행도는 되돌리지 않는다.
- 한 방문의 여러 독립 시나리오는 같은 `learning_session_id`와 같은 round를 사용할 수
  있다. AI migration `20260826_05`는 old+new unique가 공존하는 reader 전환 단계이고,
  `20260826_06`에서 old unique를 제거한 뒤 `scene + scenario_id` 단위 생성이 활성화된다.

### 완료와 별노트

- 카페·놀이동산 진행은 `completion.stage_completion_eligible`만 권위로 사용한다.
  정답이나 아이 원문을 BE에서 재채점하지 않는다.
- 놀이동산은 기본 task를 끝냈더라도 전이 task가 완료되기 전에는 stage 완료가 아니다.
- 완료 시 `verified_facts` key는 다음 계약을 그대로 사용한다.

| 시나리오 | 완료 key |
|---|---|
| `cafe_queue` | `left_count`, `right_count`, `final_choice`, `reason` |
| `cafe_budget_menu` | `child_menu_id` |
| `cafe_menu_total` | `child_menu_id`, `result` |
| `cafe_change` | `result` |
| `amusement_ticket_multiply` | `ticket_price`, `party_count`, `total_price` |
| `amusement_snack_divide` | `snack_total`, `payer_count`, `per_person` |
| `amusement_pass_compare` | `single_ride_price`, `day_pass_price`, `break_even_rides`, `benefit_from_rides` |

- AI outbox의 별노트 `attribution`, `evidence`, `attribution_label`을 손실 없이 보존한다.
  `child`와 `coauthored`를 하나의 “아이가 알려줌” 상태로 합치지 않는다.
- 현재 Spring 운영 경로는 별도 `/v1/practice-results` 호출 없이 대화 생성 요청의 인라인
  `practice_summary`를 사용한다. 이전 2단계 흐름으로 되돌릴 필요가 없다.

## Mormi-FE에서 필요한 연동

- 도움 카드는 로컬 버튼 상태가 아니라 매 턴의 `turn.help_card.visible`을 단일 권위로
  렌더링한다. 자유입력 “모르겠어”, 자동 개념 지원, L0-H3 진입과 GET 복구에서도 보여야 한다.
- `help_card.auto_open=true`이면 즉시 펼치고, `false`이면 표시 상태만 유지한다.
- L2는 서버가 준 opaque choice ID, L0는 `input.config.completion_values`를 그대로
  되돌려 보낸다. FE에서 정답이나 effect를 재구성하지 않는다.
- 놀이동산에서 `turn.input.kind=choices`라면 선택지를 첫 화면부터 렌더링한다.
  자유입력 fallback을 먼저 보이게 하는 `deferChoices`는 사용하지 않는다. AI는 L2 시작 시
  모르미 문구도 같은 L2 copy로 내려 주므로 질문과 UI가 모두 선택형이어야 한다.
- 놀이동산의 `이전으로`는 지도 화면으로 돌아가고, `궁금해 사전`은 현재
  `conversation_id`의 사전 modal을 열어야 한다. 두 버튼을 실제 click 통합 테스트로 고정한다.
- `/ai-test`와 실제 오류 UI는 FastAPI 오류의 중첩된 `detail.code`, `detail.message`를
  표시한다. `409 stale_turn`과 `503 model_output_invalid`를 같은 연결 오류로 뭉개지 않는다.
- 별노트에는 본문 아래 AI의 `attribution`을 기준으로 학습자 이름을 조합해
  `OO이가 알려줌` 또는 `OO이와 함께 공부함`을 표시한다. AI가 보낸 본문을 FE에서 다시
  생성하지 않는다.

## 교사용 분석·발화사다리 분석 담당과의 경계

이번 AI 변경은 교사용 페이지, 분석 모델의 학습·추론 정책 또는 승인 흐름을 수정하지 않는다.
다만 해당 소비자는 다음 새 관측 경계를 나중에 수용해야 한다.

- 생활 V3는 scenario 안에서 `task_id`가 전환되며 task별 pack ID/version/hash와 ledger가
  분리된다.
- 별노트 evidence link는 노트가 발행된 완료 턴이 아니라 실제 relation 근거 observation을
  가리킬 수 있다.
- `child`와 `coauthored`는 학습 완료 여부와 별개의 기여 provenance다. 이를 발화사다리
  정답 라벨로 재해석하거나 교사 평가 점수로 자동 변환하면 안 된다.
- 기존 관측 필드와 ladder enum은 유지된다. 분석 쪽에 꼭 필요한 변경은 이 문서의 소비자
  계약으로만 조율하고 이번 AI 브랜치에서 해당 코드에는 손대지 않는다.
- 기존 DB 컬럼이나 교사용 reporting table을 삭제·이름 변경하지 않는다. copy cache table과
  scenario-scoped conversation unique는 migration으로 추가·전환되며 reporting query의 기존
  입력 컬럼은 유지된다.
- 기존 발화사다리 모델 입력 enum과 feature 계약도 유지되므로 이번 병합만으로 학습 데이터를
  다시 만들거나 모델을 재학습할 필요는 없다. V3 task identity·`child|coauthored` provenance를
  새 분석 feature로 사용하고 싶을 때만 담당자가 별도 버전의 데이터셋을 설계한다.

## 배포 순서

1. 첫 배포는 AI DB를 `20260826_05`까지만 올리고 canary/env를 0으로 고정한다.
2. `conversation-scenario-idempotency-reader-v1` live가 확인된 다음 배포에서만 head
   `20260826_06`으로 old unique를 제거한다.
3. 집 V2 stable copy 45개를 prewarm한다. 카페·놀이동산은
   `reviewed_template_only`이므로 generated-copy prewarm 대상이 아니다.
4. PII 없는 provider smoke로 실제 Sonnet `understand_v2`와 Haiku `speak_v2` 구조화 호출을
   검증한다. 후보 health에서 `environment=production`, effective `verdict-v1`, 적용 canary,
   `dialogue-v3-snapshot-reader-v1`, identity reader/schema phase를 함께 확인한다.
5. BE가 새 별노트 attribution/evidence 계약을 수용한 뒤
   `MORMI_STAR_NOTE_EVENTS_ENABLED=true`로 outbox 전송을 연다. 기본값 `false`에서는 이벤트가
   유실되지 않고 pending으로 남는다.
6. 위 BE/FE 계약을 확인한 뒤 시연 계정 또는 낮은 비율부터 canary를 높인다.

장애가 나면 workflow의 `disable-v2` 동작으로 현재 이미지를 그대로 사용해 신규 배정만 즉시
0%로 내린다. 이 경로는 새 build·migration·prewarm·provider smoke를 수행하지 않으며 운영 env
원장에도 0을 남긴다. 이미 시작된 V2 대화는 snapshot reader가 처리하므로 더 오래된
비호환 이미지로 롤백하지 않는다.

## 프로덕션 활성화 판정

- AI 저장소 자체는 전체 회귀·정적 검사·migration/OpenAPI/deploy 계약을 통과하면 develop
  병합 및 canary 0 reader 배포가 가능하다.
- BE가 life 요청의 안정 identity를 보내기 전에는 카페·놀이동산 요청이 의도적으로
  `legacy-v1`에 고정되므로, AI 설정만 100%로 올려도 생활 V2가 활성화되지 않는다.
- FE가 server-authoritative 도움 카드와 즉시 L2 choice를 수용하기 전에는 엔진은 맞아도 화면이
  다른 단계를 보여 줄 수 있다.
- 따라서 develop 병합과 V2 100% 활성화는 같은 작업이 아니다. 05 reader 배포 → 06 contract
  배포 → BE/FE 계약 smoke → 낮은 canary → 100% 순서를 지킨다.
