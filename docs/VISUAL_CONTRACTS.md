# `visual.type` 및 입력 설정 계약

프론트는 `visual.type`을 판별자로 사용하고 `visual.data`를 해당 컴포넌트의 props로
전달합니다. 모르는 타입을 받으면 대화 말풍선은 유지하고 시각 자료 영역에 안전한
기본 화면을 표시합니다.

기계 판독 가능한 JSON Schema는
[`visual-contract.schema.json`](./visual-contract.schema.json)에 있습니다.

## `cafe_queues`

```json
{
  "type": "cafe_queues",
  "data": {
    "left_people": 3,
    "right_people": 5,
    "show_counts": false
  }
}
```

| 필드 | 타입 | 의미 |
|---|---|---|
| `left_people` | integer | 왼쪽 줄 사람 수 |
| `right_people` | integer | 오른쪽 줄 사람 수 |
| `show_counts` | boolean | 사람 수 라벨 공개 여부 |

`input.kind=count`일 때는 `input.config.left_person_ids`와
`input.config.right_person_ids`를 이용해 각각의 사람을 눌러 세는 UI를 활성화합니다.

두 인원은 매 세션 1~5명 범위에서 서로 다르게 정해집니다.

## `cafe_menu`

```json
{
  "type": "cafe_menu",
  "data": {
    "menu_items": [
      {
        "id": "americano",
        "name": "아메리카노",
        "price": 3000,
        "image_url": "/figma/cafe/americano.png?v=2"
      }
    ],
    "budget": 10000,
    "mormi_pick": {"id": "strawberry-juice", "name": "딸기주스", "price": 4000},
    "child_pick": null,
    "auto_total": true,
    "budget_status": "pending"
  }
}
```

- `auto_total=true`: 2단계에서 장바구니 합계를 자동 표시합니다.
- `menu_items`는 프론트가 대화 시작 시 보낸 스냅샷을 그대로 반환합니다.
- `auto_total=false`: 3단계에서 합계를 숨기고 아이가 직접 계산합니다.
- `budget_status`: `pending`, `within`, `over` 중 하나입니다.
- 예산을 넘으면 `child_pick`, `total`, `budget_status=over`를 함께 반환합니다.
- 모르미가 고른 메뉴는 `choice.disabled=true`이며 아이는 다른 메뉴를 고릅니다.

## `cafe_calculation`

```json
{
  "type": "cafe_calculation",
  "data": {
    "left": 4000,
    "right": 3000,
    "operation": "addition",
    "result_hidden": true,
    "mormi_menu": {"id": "strawberry-juice", "name": "딸기주스", "price": 4000},
    "child_menu": {"id": "americano", "name": "아메리카노", "price": 3000}
  }
}
```

카페 계산은 현재 가로식·선택·빈칸 UI만 사용합니다. 세로식 입력은 후속 범위입니다.
거스름돈은 `operation=subtraction`이며 `left`는 항상 10,000원,
`right`는 `mormi_menu_id`가 가리키는 메뉴 하나의 가격입니다.
FE의 4단계 화면과 동일하게 `10,000 − 모르미 메뉴 가격`을 계산합니다.

## `budget_meter`와 `budget_menu_help`

도움 카드 H2·H3에서 예산, 모르미 메뉴 가격과 메뉴판을 비교할 때 사용합니다.

## `money_calculation`과 `joint_money_calculation`

- `money_calculation`: H2에서 결과를 숨긴 가로식을 표시합니다.
- `joint_money_calculation`: H3 공동 수행에서만 결과와 계산 순서를 표시합니다.

## `vertical_equation`

집 반복학습 후 가르치기에서 사용하는 기존 계약입니다. 카페 3·4단계에서는
현재 사용하지 않습니다.

```json
{
  "type": "vertical_equation",
  "data": {
    "left": 2800,
    "right": 3200,
    "operation": "addition",
    "result_hidden": true
  }
}
```

| 필드 | 타입 | 값 |
|---|---|---|
| `left` | integer | 첫 번째 수 |
| `right` | integer | 두 번째 수 |
| `operation` | string | `addition` 또는 `subtraction` |
| `result_hidden` | boolean | 결과 칸 숨김 여부 |

`input.kind=equation`일 때 `input.config.places` 순서대로 숫자 입력 칸을 렌더링합니다.
화면 문구에서는 `11백` 같은 표현을 만들지 않고 자리 이름과 숫자를 분리합니다.

## `home_teaching`

집 반복학습 뒤 AI가 생성한 가르치기 시나리오의 기본 화면입니다.

```json
{
  "type": "home_teaching",
  "data": {
    "curriculum_session_id": "money-count",
    "subject": "money",
    "unit": "돈 계산",
    "title": "돈을 세어요",
    "problem": {
      "prompt": "모두 얼마일까?",
      "answers": ["600원", "500원", "700원"],
      "visual": {"type": "money", "amounts": [500, 100]}
    }
  }
}
```

`problem.correct`는 프론트에 전달하지 않습니다. 화면은 `problem.visual`을 기존 반복
문제 렌더러에 전달하고, 입력 UI는 항상 턴의 `input` 계약을 따릅니다.

- `home_practice_problem`: H2에서 같은 표상을 다시 보여주는 도움 카드
- `joint_reading_card`: H3에서 검수된 핵심 문장을 함께 읽는 카드

## `number_cards`

도움 카드 H2에서 사용하는 수 비교 표상입니다.

```json
{
  "type": "number_cards",
  "data": {
    "cards": [3, 5],
    "neutral_style": true
  }
}
```

## `place_value_equation`

```json
{
  "type": "place_value_equation",
  "data": {
    "left": 2800,
    "right": 3200,
    "operation": "addition"
  }
}
```

같은 자리끼리 정렬된 세로식을 보여주되 결과는 공개하지 않습니다.

## `joint_steps`

```json
{
  "type": "joint_steps",
  "data": {
    "steps": ["한 명씩 세기", "3과 5 비교하기", "사람이 적은 줄 찾기"]
  }
}
```

H3 공동 수행에서 현재 단계를 순서대로 표시합니다.

## `joint_equation_steps`

```json
{
  "type": "joint_equation_steps",
  "data": {
    "left": 2800,
    "right": 3200,
    "operation": "addition",
    "result": 6000
  }
}
```

H3 공동 수행 전용입니다. `input.kind=joint`와 함께 사용하며 도움 카드 안에서만 결과와
순서를 공개합니다. 모르미 말풍선이 이 내용을 직접 설명하지 않습니다.

## `success`

```json
{
  "type": "success",
  "data": {
    "task": "cafe_queue_3_vs_5"
  }
}
```

대화 완료 연출용입니다. 보상 지급 여부는 이 타입이 아니라
`completion.teach_reward_eligible`을 사용합니다.

## 궁금해사전 전용 시각자료

궁금해사전의 `card.visual`은 현재 턴의 `turn.visual`이나 도움카드
`help_card.visual_*`과 별도 계약입니다. 임의의 첫 반복문제 그림을 재사용하지 않고,
카드의 구체적 예시를 그대로 설명할 수 있는 전용 표상을 사용합니다.

수 세기 카드는 한 장의 완성 그림만 보여 주지 않고, 수가 하나씩 늘어나는 순서를
왼쪽에서 오른쪽으로 보여 줍니다.

```json
{
  "type": "count_sequence",
  "data": {
    "count": 3,
    "sequence_counts": [1, 2, 3],
    "layout": "left_to_right",
    "counted_object": "filled_cell"
  },
  "fact_refs": ["count"],
  "alt_text": "색칠된 칸을 1개, 2개, 3개로 차례로 세는 세 장의 그림"
}
```

`sequence_counts`는 반드시 `1`부터 `count`까지 빠짐없이 증가해야 합니다. FE는
세 값을 각각 독립된 패널로 렌더링하고 `1개`, `2개`, `3개`처럼 개수 라벨을 함께
표시합니다. 특정 달걀 이미지는 디자인 자산일 뿐이며, AI 계약은 점·달걀 등 대상이
바뀌어도 같은 순차 세기 구조를 유지합니다.

```json
{
  "type": "money_sum",
  "data": {
    "first_amount": 500,
    "second_amount": 100,
    "total": 600
  },
  "fact_refs": ["first_amount", "second_amount", "total"],
  "alt_text": "500원과 100원을 더해 600원이 되는 그림"
}
```

- `data`의 모든 `fact_refs` 값은 `card.example.facts`와 이름·값이 같아야 합니다.
- 산술 예시는 피연산자와 결과가 실제 계산과 맞아야 합니다.
- `alt_text`는 그림 없이도 핵심 정보를 알 수 있게 80자 이내로 작성합니다.
- `type`은 표상의 의미를 나타냅니다. 현재는 수 세기·수량 비교·돈 덧셈·십모형·
  자릿값·사칙연산·묶음·규칙·시계·시간선·달력·측정·도형·위치·분류·막대그래프·
  가능성·줄 비교·예산 메뉴·메뉴 합계·거스름돈을 지원합니다.
- 같은 그림 파일을 재사용할 수는 있지만, 카드 예시와 맞지 않는 사실이나 조작을
  문장으로 요구해서는 안 됩니다.

프론트는 `card.visual.type`별 컴포넌트를 선택하고 `data`를 그대로 렌더링합니다.
표시할 수 없는 타입을 조용히 일반 문제 그림으로 바꾸지 말고 명시적인 미지원 상태로
처리해야 합니다.

## 선택지

```json
{
  "id": "left_queue",
  "label": "왼쪽 줄",
  "image_url": "/choices/left-queue.png",
  "disabled": false
}
```

`image_url`과 `disabled`는 선택 필드입니다. 프론트는 문구가 아니라 `id`를 응답으로
돌려보냅니다.
