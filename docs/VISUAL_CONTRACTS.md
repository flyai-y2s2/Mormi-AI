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
거스름돈은 `operation=subtraction`이며 `left`는 아이가 실제로 낸 돈
(`cafe_context.paid_amount`), `right`는 두 메뉴의 합계입니다. 고정값이 아닙니다.

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
