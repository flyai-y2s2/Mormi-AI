from __future__ import annotations

from .dataset import LADDER_ORDER, LadderExample

LABEL_TO_ID = {level: index for index, level in enumerate(LADDER_ORDER)}
ID_TO_LABEL = {index: level.value for level, index in LABEL_TO_ID.items()}


def serialize_model_input(example: LadderExample) -> str:
    return " ".join(
        (
            f"[현재단계={example.current_level.value}]",
            f"[응답방식={example.response_mode.value}]",
            f"[정답여부={example.concept_result.value}]",
            f"[시도횟수={example.attempt_count}]",
            f"[단원={example.skill_id}]",
            example.response_text,
        )
    )


def as_training_record(example: LadderExample) -> dict[str, object]:
    return {
        **example.model_dump(mode="json"),
        "input_text": serialize_model_input(example),
        "label": LABEL_TO_ID[example.target_level],
    }
