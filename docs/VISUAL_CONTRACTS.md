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

## `vertical_equation`

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
