# Mormi AI Dialogue Service

경계선지능 아동이 AI 동생 모르미를 가르치며 기초 수학을 복습하고, 카페·놀이동산 같은 생활 장면에 적용하도록 돕는 독립 AI 대화 서비스입니다.

이 저장소는 화면을 직접 렌더링하거나 일반 서비스 백엔드를 대신하지 않습니다.
다음 교육적 결정을 담당하고 Spring 백엔드가 프론트엔드에 전달할 수 있는
`TurnContract`를 반환합니다.

- 집 반복학습 결과로 검수된 시나리오를 고르고 첫 가르치기는 항상 `L4-H0`에서 시작
- 모든 새 집 가르치기는 오개념 낚시 없이 모르미의 진짜 L4 도움 요청으로 시작
- 현재 집 커리큘럼 36개 세션의 검수된 가르치기 시나리오 생성
- 카페 필수 5개·놀이동산 준비 4개, 총 9개 집 세션과 카페 4개 시나리오/4개 과제,
  놀이동산 3개 시나리오/6개 과제에 typed reasoning graph 기반 `verdict-v1`
  네이티브 엔진 연결
- 카페 제품 여정의 줄 서기, 메뉴값 덧셈, 거스름돈 진행
- 놀이동산의 표 값 곱셈, 간식값 똑같이 나누기, 자유이용권 손익분기 비교 진행
- 발화사다리 `L4 → L3 → L2(선택지) → L0`와 힌트사다리 `H0~H3`를 독립적으로 조절
  (단계명은 재번호화하지 않으며, 과거 L1 기록만 L2로 읽음)
- 도움 카드 자동 공개
- 도움 카드와 분리된 검수·버전 고정 궁금해사전 40개 제공
- Sonnet Low 발화 이해, 결정형 오케스트레이터, Haiku 모르미 화자와 Haiku
  비학습 브리지를 역할별 프롬프트로 분리
- 원문에 글자 그대로 있는 근거만 통과시키고, 의미 판정을 재채점하지 않는 V2 신뢰 경계와
  단조 증가 reasoning ledger
- 검증된 대사만 전송하는 SSE 진행·대사 스트리밍
- 자유문장은 LLM, 선택·빈칸·조작은 검수 ID 기반 결정형 판정 후 동일 오케스트레이터 사용
- 아이 근거를 보존한 직접 별노트와 검수된 공동 별노트를 구분해 생성
- 학습자별 안정 발화 단계와 최근 힌트 의존도 저장
- 모르미 질문과 아이 원문 발화·선택 기록을 DB에 평문 저장

집 V2 네이티브 콘텐츠팩은 `dialogue_v2_required_home_catalog.json`에 있으며, FE의 마지막
반복문제를 복제하지 않는 AI 소유 가르치기 예시입니다. 생활 장면 팩은
`dialogue_v2_cafe_content.py`와 `dialogue_v2_amusement_content.py`가 FE에서 받은 검증된
장면 값 또는 AI 카탈로그 값을 대화별 strict scenario/task pack으로 materialize합니다.
두 종류 모두 문제·정답·중간결과·L2/L0 효과를 하나의 검증 계약으로 묶습니다.
H0 화면의 각 공개 사실은 graph fact와 경로로 연결하며, 점처럼 정답이 그림에 지각적으로
인코딩되는 경우도 별도 binding으로 검산합니다. 산술식·단위·정답 표면형·도움 카드 공개
범위와 H0 풀이 선공개까지 시작 시 검사합니다.

| 여정 연결 | `curriculum_session_id` | V2 `pack_id` |
|---|---|---|
| 카페 필수 | `number-count` | `home.number-count.v2` |
| 카페 필수 | `number-compare` | `home.number-compare.v2` |
| 카페 필수 | `money-count` | `home.money-count.v2` |
| 카페 필수 | `money-price` | `home.money-price.v2` |
| 카페 필수 | `money-budget` | `home.money-budget.v2` |
| 놀이동산 준비 | `multiply-groups` | `home.multiply-groups.v2` |
| 놀이동산 준비 | `divide-share` | `home.divide-share.v2` |
| 놀이동산 준비 | `divide-group` | `home.divide-group.v2` |
| 놀이동산 준비 | `multiply-easy-tables` | `home.multiply-easy-tables.v2` |

V2 router와 엔진은 서비스에 연결되어 있습니다. `verdict-v1`과 canary가 활성화됐을 때
**새로 만드는** 대화 중 위 집 9개 팩, 카페 4개 시나리오, 놀이동산 3개 시나리오이면서
안정적인 `learning_session_id`가 있는 요청만 서버 해시 버킷으로 V2를 선택합니다. 그 밖의
집 27개 세션, 비지원 시나리오, 안정 ID가 없는 요청과 canary 비선택 요청은 `legacy-v1`
adapter로 명시적으로 고정됩니다. 홈은 pack·reasoning ledger·stable-copy plan과 선택 copy를,
생활 장면은 scenario 전체·task별 variant/ledger/별노트 provenance와 reviewed 문구를
conversation에 고정합니다. 배포나 설정 변경 뒤에도 진행 중인 대화의 엔진·콘텐츠·문구는
바뀌지 않습니다.

집 9개 팩은 각각 `initial_help` 1개, L2 질문 2개, L0 도입·행동 2개로 총 45개의 stable
copy 슬롯을 갖습니다. PII 없는 생성 계획만 키에 포함해 DB의
`dialogue_generated_copy_cache`에 영속 저장하고, 배포 시 45개 전부를 미리 생성·검증합니다.
첫 `L4-H0` 도움 요청도 `initial_help` 슬롯을 사용합니다. 캐시가 바쁘거나 생성·검증에
실패하면 아동 요청을 실패시키지 않고 팩의 사람 검수 fallback을 사용합니다.

카페·놀이동산은 대화마다 숫자·메뉴·선택 결과가 달라지므로 generated stable-copy 캐시를
사용하지 않습니다. `stable_copy_mode=reviewed_template_only`인 검수 템플릿을 materialize할
때 안전하게 조립하고 V3 snapshot에 고정합니다. 따라서 45-slot prewarm은 집 9개에만
적용되며, 생활 장면의 시작·L2·L0·과제 전환 문구가 아동 요청 중 즉석 생성되는 일은 없습니다.

V2 화자 출력은 학습 판정과 분리된 방화벽을 통과합니다. 미해결 숫자는 아라비아 숫자뿐
아니라 `천이야`, `셋이야` 같은 단위 없는 한글 수도 차단하고, Choice/Text 정답 표면형과
L2 선택지 label도 먼저 말하지 못하게 합니다. Haiku 사회적 브리지에는 아동 원문을 전혀
전달하지 않고 분류된 interaction kind만 사용합니다. 개인정보가 섞인 학습 evidence는
ledger 진전은 보존하되 Haiku 화자 입력에서는 원문 구절을 제외합니다.

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
자유 발화
  → Claude Sonnet Low 의미 판정
  → literal evidence guard(원문 근거·graph ID만 검사, 의미 재판정 없음)
  → 단조 증가 reasoning ledger

L2 선택 / L0 공동 수행 / no_response
  → 서버 계약 기반 결정형 의미 경로(이해 LLM 호출 없음)

두 경로 → 결정형 L/H·다음 목표 결정
  → Haiku 주 화자 / Haiku 안전 브리지 / 홈 Sonnet stable copy·검수 문구
  → 구조·근거 중심 코드 출력 계약
  → TurnContract
```

- **의미 판정과 상태 통제 분리**: V2 자유 발화의 수학적 의미 판정은 Sonnet Low의 단일
  verdict를 따릅니다. 코드는 이를 기대값 비교나 산술 재계산으로 뒤집지 않고, 원문 근거와
  graph 소속을 검사한 뒤 고정 콘텐츠의 canonical fact를 ledger에 기록합니다. 진도, 완료,
  L/H 전환과 다음 입력 계약은 ledger와 콘텐츠 계약만으로 결정합니다.
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
- **별노트 근거와 귀속 분리**: 독립 설명 별노트는 원문 offset으로 검증한 relation 근거를
  Haiku가 검수 맥락 안에서 독립 문장으로 다듬고, 호출·검증 실패 시 검수 문구를 사용합니다.
  공동수행 별노트는 항상 검수 문구를 사용합니다.
  필수 relation을 H0/H1에서 아이가 직접 설명했으면 `child`, 도움·선택·공동수행이나
  H2/H3 검수 카드 뒤 확인으로 완성했으면 `coauthored`로 기록합니다. 짧은 “응”도 완료
  근거가 될 수 있지만 독립 가르침으로 과장하지 않습니다.
- **자연스러운 하강**: 아이가 알려 준 부분은 받아들이고 아직 빠진 내용만 동생다운
  말투로 다시 요청합니다. 사다리 변경이나 시스템 상태를 대사로 설명하지 않습니다.
- **동생다운 화자**: 모르미는 교사처럼 퀴즈를 내거나 상태를 보고하지 않고,
  “아, 세 개구나!”, “나 3이랑 5를 어떻게 비교할지 헷갈려...”처럼 자신이 모르는
  지점을 털어놓고 도움을 청합니다. 아이에게 생각의 근거를 입증시키지 않습니다.
- **역질문도 정상 대화**: 현재 문제·도움 카드·풀이에 대한 아이 질문은
  `task_question`으로 분류하고, `reason_or_method`, `meaning`,
  `confirmation_or_challenge` 세 초점만 구분합니다. reasoning ledger는 바꾸지 않고
  H를 한 단계 올려 더 강한 도움 카드를 보여 줍니다. 모르미는 카드의 식·답·방법을
  읽거나 설명하지 않고, 짧게 반응한 뒤 서버가 붙인 현재 목표를 다시 부탁합니다.
- **학습 의미와 대화 행동 분리**: 답·방법 claim과 `meta_question`, `refusal`,
  `safe_play`, `request_mormi_answer`를 서로 독립적으로 보존합니다. 따라서 “너 AI잖아.
  그래도 16,000원이야”처럼 메타 질문과 실제 답이 섞여도 사회적 반응과 학습 진전을
  둘 다 잃지 않습니다.
- **상황별 부탁 방식 분리**: “네가 해”는 모르미가 대신 풀 수 없다는 한계를 말한 뒤
  도움 카드를 보고 다시 알려 달라고 부탁하고, 거절은 아이의 말을 복창하거나 해석하지
  않은 채 “나 꼭 알고 싶은데...”처럼 모르미의 궁금함만 표현합니다. 모든 동적 화자는
  쉬운 반말만 사용하며 `-요/-습니다/-신 거구나` 같은 존댓말을 섞지 않습니다.
- **안전한 자연스러움**: 코드는 검증된 사실, 빠진 슬롯, 질문 목적과 금지 답을
  정하고 Haiku는 그 범위 안에서 말투를 만듭니다. 화자에게는 아이 원문을 전달하지 않고,
  검증된 fact·relation과 서버 소유 질문 초점만 전달합니다.
- **단일 의미 판정**: 자유 발화는 Sonnet Low가 최근 대화 맥락과 현재 질문을 함께
  보고 구조화합니다. 별도의 재판정 LLM을 호출하지 않습니다. V2 literal evidence guard는
  claim ID가 고정 graph에 있는지와 evidence span이 현재 원문에 exact 또는 Unicode NFC로
  존재하는지만 검사하며, 정답·단위·산술을 다시 판정하거나 verdict를 고치지 않습니다.
- **경량 대화 브리지**: 거절·메타·가벼운 장난처럼 학습 상태를 바꾸지 않는 안전한
  사회적 발화에만 Haiku가 짧은 연결 대사를 생성합니다. 브리지는 슬롯·L/H·별노트를
  변경할 수 없습니다. 브리지와 일반 주 화자 모두 반응·인정 한 문장만 만들고, 완료 전에는
  이전 대사가 보이지 않는 UI에서도 질문이 사라지지 않도록 서버가 검수된 현재 목표
  재질문을 항상 붙입니다. 자유발화가 L2 선택이나 L0 공동수행 단계로 내려가더라도
  먼저 Haiku가 그 발화에 반응하며, 단계가 바뀌었다는 이유로 고정 문구가 이를 대신하지
  않습니다. 도움 카드는 과제 역질문에서만 “어? 도움 카드가 나왔어”처럼 언급할 수 있고,
  거절에는 카드 노출 여부와 무관하게 같은 부탁을 밀어붙이지 않습니다.
- **구조 중심 출력 계약**: 화자 출력은 완결 문장인지, 요청한 대화 행동과 미해결
  슬롯을 따르는지, 허용된 사실·인용만 사용하는지 검사합니다. 폭넓은 금칙어 정규식으로
  자연스러운 문장을 추측해 차단하거나 글자 수를 기준으로 문장을 자르지 않습니다.
- **단계별 지연 관찰**: 분류·브리지·화자 생성의 소요 시간을 턴 관찰 기록의
  `runtime_json`에 저장합니다. 기존 `verifier_latency_ms`는 과거 기록 호환용으로만
  남으며 새 턴에서는 `null`입니다.
- **원문 기록 분리**: 원문은 접근이 통제된 대화 기록에 평문 저장하며 학습 상태에는 검증된 사실만 저장합니다.
- **완료 사실 연동**: `completion.verified_facts`에는 현재 실행 계약이 승인해 고정 콘텐츠의
  canonical 값으로 기록한 사실만 담습니다. V2 완료는 reasoning graph의 required fact·relation
  ID가 ledger에 모두 있는지로 계산하며 원문 발화 자체는 넣지 않습니다.

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

Anthropic 프롬프트 캐시는 선택된 V2 역할의 정적 system prefix에만 breakpoint를 두고,
아이 발화와 현재 대화 상태는 매 호출의 동적 user message로 유지합니다. 캐시 쓰기·적중
토큰은 `llm_call` 로그의 `cache_write_tokens`, `cache_read_tokens`로 확인할 수 있습니다.
공급자 연동 smoke는 다음과 같이 실행합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_prompt_cache.py
```

## 주요 API

| Method | Path | 역할 |
|---|---|---|
| POST | `/v1/practice-results` | 집 반복학습 결과 저장 |
| POST | `/v1/conversations` | 가르치기/카페/놀이동산 대화 시작 |
| POST | `/v1/conversations/{conversation_id}/responses` | 발화·선택·조작 응답 제출 |
| POST | `/v1/conversations/{conversation_id}/responses/stream` | SSE 진행 상태와 검증된 다음 턴 스트리밍 |
| GET | `/v1/conversations/{conversation_id}` | 최신 상태와 턴 복구 |
| GET | `/v1/content/dictionary-cards/{curriculum_session_id}` | 현재 승인된 궁금해사전 카드 조회 |
| GET | `/v1/conversations/{conversation_id}/dictionary-card` | 대화에 고정된 궁금해사전 카드 조회 |
| GET | `/v1/learners/{learner_id}/skill-profiles` | 학습자별 L/H 근거 조회 |
| GET | `/v1/learners/{learner_id}/star-notes` | 별노트 조회 |
| GET | `/v1/conversations/{conversation_id}/transcript` | 보호된 원문 질문·응답 기록 조회 |

현재 제품 여정은 `cafe_queue` → `cafe_menu_total` → `cafe_change`의 세 독립
가르치기 스테이지입니다. 합산 스테이지에 들어가기 전 화면에서 모르미가 메뉴 하나를
정하고, 아이가 서로 다른 메뉴 하나를 직접 고릅니다. 이 선택은 별도 채점 스테이지가
아니며 두 메뉴 ID를 `cafe_context`에 담아 하나의 합산 문제로 고정합니다.
`cafe_budget_menu`는 순차 배포와 시작된 대화 복구를 위한 호환 시나리오로만 유지합니다.
전달된 장면값은 재시도나 복구 때 바뀌지 않습니다.

놀이동산은 `amusement_ticket_multiply`, `amusement_snack_divide`,
`amusement_pass_compare`의 3개 독립 시나리오를 지원합니다. Spring BE는 방문 권한과
`scenario_id`만 전달하고, AI의 검수된 카탈로그가 제목·미션·문제 수·정답·오개념·힌트와
전이 문제를 함께 생성해 한 대화에 고정합니다. 기본 문제를 완료한 뒤 같은 기능을 새
숫자에 적용하며, 완료 응답에는 결정적 엔진이 실제로 검증한 기본 문제 사실만 반환합니다.
구버전 순차 배포를 위해 요청의 `park_context`는 당분간 허용합니다. 이때 검수 범위 안의
주어진 숫자만 기존 화면과 맞추고, 문구·정답·전략·힌트·전이 문제는 모두 AI가 다시 만듭니다.

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
현재 Spring 운영 경로는 별도 `/v1/practice-results` 호출 없이 인라인 요약을 사용합니다.
같은 `(learner_id, learning_session_id, scene, scenario_id, conversation_round)`는 생성
재시도로 복구하고, 명시적 재시작은 해당 시나리오의 회차를 증가시켜 기존 기록을 보존한
새 `conversation_id`를 만듭니다. 한 카페 방문 ID를 여러 독립 시나리오가 함께 사용해도
서로 충돌하지 않습니다.
동일 tuple의 동시 생성도 DB가 한 대화만 커밋하고 모든 요청이 그 대화를 반환합니다.

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
  "no_response_kind": null,
  "asr_confidence": null,
  "latency_ms": 4200
}
```

`type`은 `text`, `choice`, `fill`, `count`, `equation`, `action`, `no_response`를
지원합니다. Spring 백엔드는 프론트에서 받은 입력을 이전 턴의 `input.kind`에 맞는
형태로 이 서비스에 전달합니다.

V2의 `no_response`는 `no_response_kind`로 원인을 구분합니다.

- `explicit_help`: 아이가 도움 버튼을 누른 경우입니다. 기존 클라이언트처럼 subtype 없이
  `type=no_response`만 보내도 이 값으로 해석하며, 최초 `L4-H0`에서는 캐시된
  `initial_help` 문구로 질문 부담을 낮춥니다.
- `silence_timeout`: 제한 시간 동안 응답이 없었던 경우입니다. 표현 지원 신호로 처리합니다.
- `asr_empty`: STT 결과가 비어 입력을 얻지 못한 경우입니다. 학습 근거로 처리하지 않습니다.

세 종류 모두 이해 LLM을 호출하지 않으며 사실·관계를 새로 검증하지 않습니다. L2에서는
`TurnContract.input.choices`에 서버가 보낸 opaque `choice_id` 하나만 `choice_ids`로
되돌려 보내고, 서버가 팩에 고정된 effect를 적용합니다. L0-H3에서는
`input.kind=joint`와 함께 받은 `input.config.completion_values`를 `action.values`로 정확히
되돌려 보내야 합니다. 키 누락·추가나 JSON 타입 변환도 거절되며 FE·BE가 정답을 재구성하거나
재채점하지 않습니다.

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
- 원문 보존 정책은 `no_raw`, `30_days`, `90_days`, `permanent` 중 하나이며,
  사전 동의 파일럿의 현재 기본값은 `permanent`입니다.
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
| `MORMI_CLASSIFIER_MODEL` | 아니요 | 기본값 `claude-sonnet-4-6` |
| `MORMI_CLASSIFIER_EFFORT` | 아니요 | 자유 발화 분류 추론 강도, 기본값 `low` |
| `MORMI_BRIDGE_MODEL` | 아니요 | 안전한 사회적 발화 브리지 모델, 기본값 `claude-haiku-4-5-20251001` |
| `MORMI_SPEAKER_MODEL` | 아니요 | 주요 모르미 화자, 기본값 `claude-haiku-4-5-20251001` |
| `MORMI_SPEAKER_EFFORT` | 아니요 | Sonnet 화자로 재정의할 때의 추론 강도, 기본값 `low`; Haiku에는 전달하지 않음 |
| `MORMI_STAR_NOTE_MODEL` | 아니요 | 직접 별노트 문맥 편집 모델, 기본값 `claude-haiku-4-5-20251001` |
| `MORMI_REPORT_MODEL` | 아니요 | 교사용 요약 전용 모델, 기본값 `claude-sonnet-4-6`; 모르미 화자 변경과 독립 |
| `MORMI_PROMPT_CACHING_ENABLED` | 아니요 | Anthropic V2 역할 프롬프트 캐싱 활성화 여부, 기본 `false` |
| `MORMI_PROMPT_CACHE_TTL` | 아니요 | 캐시 TTL. `5m` 또는 `1h`, 기본 `5m` |
| `MORMI_PROMPT_CACHE_STAGES` | 아니요 | 정적 system prefix를 캐시할 V2 역할의 JSON 배열. 기본 `["understanding_v2"]`; `speaker_v2` 추가 가능 |
| `MORMI_RUNTIME_CONTRACT_VERSION` | 아니요 | 신규 대화의 router 모드. 기본 `legacy-v1`; 네이티브 팩 canary를 허용하려면 `verdict-v1` |
| `MORMI_DIALOGUE_V2_CANARY_PERCENT` | 아니요 | V2 적격 신규 대화 중 `verdict-v1`로 고정할 비율(0~100). 기본 `0`; 0보다 크면 runtime 설정도 `verdict-v1`이어야 함 |
| `MORMI_DIALOGUE_V2_CANARY_SALT` | 아니요 | 학습자·학습 세션·회차 기반 결정형 canary 버킷 salt. 진행 중인 대화에는 재적용하지 않음 |
| `MORMI_STABLE_COPY_MODEL` | 아니요 | 45개 stable copy 사전 생성 모델, 기본 `claude-sonnet-4-6` |
| `MORMI_STABLE_COPY_EFFORT` | 아니요 | stable copy 생성 추론 강도, 기본 `low` |
| `MORMI_STABLE_COPY_TIMEOUT_SECONDS` | 아니요 | stable copy 한 슬롯 생성 제한 시간, 기본 8초 |
| `MORMI_STABLE_COPY_PROMPT_VERSION` | 아니요 | 캐시 키에 포함되는 prompt 버전, 기본 `stable-copy-v1` |
| `MORMI_STABLE_COPY_SCHEMA_VERSION` | 아니요 | 캐시 키에 포함되는 출력 schema 버전, 기본 `stable-copy-output-v1` |
| `MORMI_STABLE_COPY_VALIDATOR_VERSION` | 아니요 | 캐시 키에 포함되는 validator 버전, 기본 `stable-copy-validator-v2` |
| `MORMI_STABLE_COPY_CACHE_LEASE_SECONDS` | 아니요 | 동시 생성 단일화 lease, 기본 30초 |
| `MORMI_STABLE_COPY_CACHE_RETRY_BASE_SECONDS` | 아니요 | 실패 캐시 재시도 backoff 시작값, 기본 2초 |
| `MORMI_STABLE_COPY_CACHE_RETRY_MAX_SECONDS` | 아니요 | 실패 캐시 재시도 backoff 상한, 기본 120초 |
| `MORMI_CLASSIFIER_TIMEOUT_SECONDS` | 아니요 | V2/V3 자유발화 이해 호출 1회의 제한 시간, 기본 15초 |
| `MORMI_SPEAKER_TIMEOUT_SECONDS` | 아니요 | 화자 생성 제한 시간, 기본 10초 |
| `MORMI_BRIDGE_TIMEOUT_SECONDS` | 아니요 | 사회적 발화 브리지 생성 제한 시간, 기본 4초 |
| `MORMI_IDEMPOTENCY_RETENTION_DAYS` | 아니요 | 멱등 응답 보존 기간, 기본 30일 |
| `MORMI_CORS_ORIGINS` | 아니요 | 브라우저 직접 호출을 허용할 오리진 JSON 배열. Spring 경유만 하면 `[]` |
| `MORMI_SHOW_INTERNAL_PEDAGOGY=false` | 아니요 | 운영 응답에서 내부 L/H·판정 근거를 숨김 |

`develop` 배포 워크플로는 후보·운영·롤백 API 컨테이너에 분류기/화자/stable-copy
모델과 effort, `stable-copy-validator-v2`를 `docker run -e`로 명시합니다. 따라서 EC2의 기존
`/etc/mormi-ai/mormi.env`에 남아 있는 이전 설정보다 Sonnet Low 이해, Haiku 주 화자,
Sonnet stable copy 설정이 우선합니다. 안전한 비학습 브리지는 별도
`MORMI_BRIDGE_MODEL`의 Haiku 기본값을 사용합니다.

배포는 API 교체 전에 대화 identity를 expand-contract로 전환합니다. 첫 배포는
`20260826_05`에서 기존 visit-wide unique와 새 scene/scenario unique를 함께 유지하고
canary와 운영 env 원장을 `0`으로 고정합니다. 이전 live가
`conversation-scenario-idempotency-reader-v1`을 광고하는 다음 배포에서만 head
`20260826_06`으로 old unique를 제거합니다. 그 뒤
집 stable copy 45개를 prewarm한 뒤,
후보 `/health`가 `verdict-v1`과 홈·카페·놀이공원의 전체 persisted 경계를 뜻하는
`dialogue-v3-snapshot-reader-v1`, 선택한 identity schema phase를 함께 광고하고 LLM 설정도
유효한지 확인합니다. 이어 PII 없는 synthetic 입력으로 실제 Sonnet
`understand_v2`와 Haiku `speak_v2` strict-output 호출까지 성공해야 후보를 시작합니다. 후보
health의 `environment=production`, 실제 `runtime_contract_version=verdict-v1`, 적용 canary
비율도 컨테이너 설정과 일치해야 합니다. 이
aggregate capability는 기존 홈 V2 구성요소, life scenario/task pack, task별 ledger·별노트
provenance, 영속 `turn-contract-v1`까지 포함합니다. 최초
V2-capable 이미지는 canary 기본값 `0`으로 먼저 배포해야 합니다. 이후 비율을 올릴 때도
롤백 이미지가 같은 pinned snapshot을 읽을 수 없으면 배포가 중단됩니다. 수동 develop
배포의 `disable-v2` 동작은 새 이미지 build, DB migration, prewarm과 provider smoke를 거치지
않고 현재 검증된 이미지로 canary `0` 후보를 먼저 띄운 뒤 API를 재시작합니다. 이 값은 운영
env 원장에도 기록되어 다음 자동 develop 배포에서 이전 비율로 되살아나지 않습니다.
이미 생성된 대화는 저장된 runtime pin을 따르며, V2 엔진·pinned snapshot이 없거나 호환되지
않으면 `legacy-v1`로 조용히 전환하지 않고 fail closed 합니다.

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

V2를 받을 배포에서는 테스트와 별개로 DB migration과 stable-copy prewarm을 이 순서로
실행합니다. prewarm은 45개 모두가 immutable `ready` row로 다시 조회될 때만 성공하며,
하나라도 fallback·누락·검증 실패이면 JSON 실패 목록을 출력하고 종료 코드 1을 반환합니다.

```bash
python scripts/migrate_database.py
python scripts/prewarm_dialogue_v2_copy.py
python scripts/smoke_dialogue_v2_models.py
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
- BE·FE·분석 담당 연동 메모: [`docs/V2_INTEGRATION_COORDINATION.md`](./docs/V2_INTEGRATION_COORDINATION.md)
