from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .copy_quality import validate_child_facing_math_copy

DictionaryMethodPolicy = Literal["open_methods", "target_method"]
DictionaryVisualType = Literal[
    "count_sequence",
    "object_count",
    "quantity_compare",
    "money_sum",
    "ten_frame",
    "place_value",
    "object_operation",
    "equation",
    "equal_groups",
    "pattern_sequence",
    "clock_face",
    "time_line",
    "calendar",
    "measurement_compare",
    "ruler",
    "shape_model",
    "position_model",
    "category_model",
    "bar_chart",
    "chance_bag",
    "queue_compare",
    "budget_menu",
    "menu_sum",
    "change",
]

_DEICTIC_COPY = re.compile(r"(?:그것|이것|저것|아까|방금|여기|저기|위의|아래의)")
_UNREVIEWED_STRATEGY_COPY = re.compile(r"(?:무조건|오직|반드시|큰\s*값부터|느낌으로)")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DictionaryTextBlock(StrictModel):
    lines: list[str] = Field(min_length=1, max_length=2)

    @field_validator("lines")
    @classmethod
    def validate_lines(cls, lines: list[str]) -> list[str]:
        normalized: set[str] = set()
        for line in lines:
            if not line.strip():
                raise ValueError("dictionary lines must not be blank")
            if len(line) > 50:
                raise ValueError("dictionary lines must fit within 50 characters")
            if _DEICTIC_COPY.search(line):
                raise ValueError("dictionary lines must be understandable without prior context")
            validate_child_facing_math_copy([line])
            compact = re.sub(r"\s+", "", line)
            if compact in normalized:
                raise ValueError("dictionary lines must not repeat the same sentence")
            normalized.add(compact)
        return lines


class DictionaryEquation(StrictModel):
    operator: Literal["add", "subtract", "multiply", "divide"]
    operands: list[int] = Field(min_length=2, max_length=2)
    result: int
    expression: str = Field(min_length=3, max_length=40)
    operand_fact_refs: list[str] = Field(min_length=2, max_length=2)
    result_fact_ref: str

    @model_validator(mode="after")
    def validate_arithmetic(self) -> DictionaryEquation:
        left, right = self.operands
        expected = {
            "add": left + right,
            "subtract": left - right,
            "multiply": left * right,
            "divide": left // right if right and left % right == 0 else None,
        }[self.operator]
        if expected is None or self.result != expected:
            raise ValueError("dictionary equation result does not match its operands")
        compact_expression = re.sub(r"[\s,원개묶음cmgL]", "", self.expression)
        if str(left) not in compact_expression or str(right) not in compact_expression:
            raise ValueError("dictionary equation expression must include both operands")
        if str(self.result) not in compact_expression:
            raise ValueError("dictionary equation expression must include the result")
        return self


class DictionaryExample(DictionaryTextBlock):
    facts: dict[str, Any] = Field(min_length=1)
    equation: DictionaryEquation | None = None
    # A worked example may show its answer, but it must be a different
    # assessment instance from the teaching question currently on screen.
    # Keeping the answer explicit lets startup/CI compare the two without
    # guessing which arbitrary fact key represents the outcome.
    answer: str | int | float | bool | None = None

    @model_validator(mode="after")
    def validate_equation_fact_refs(self) -> DictionaryExample:
        if isinstance(self.answer, str) and not self.answer.strip():
            raise ValueError("dictionary example answer must not be blank")
        if self.equation is None:
            return self
        refs = [*self.equation.operand_fact_refs, self.equation.result_fact_ref]
        missing = set(refs) - set(self.facts)
        if missing:
            raise ValueError(f"dictionary equation references unknown facts: {sorted(missing)}")
        for ref, value in zip(self.equation.operand_fact_refs, self.equation.operands, strict=True):
            if self.facts[ref] != value:
                raise ValueError("dictionary equation operands must match example facts")
        if self.facts[self.equation.result_fact_ref] != self.equation.result:
            raise ValueError("dictionary equation result must match example facts")
        return self


class DictionaryVisual(StrictModel):
    type: DictionaryVisualType
    data: dict[str, Any] = Field(min_length=1)
    fact_refs: list[str] = Field(min_length=1)
    alt_text: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_type_contract(self) -> DictionaryVisual:
        validate_child_facing_math_copy([self.alt_text])
        if self.type == "count_sequence":
            count = self.data.get("count")
            sequence = self.data.get("sequence_counts")
            if not isinstance(count, int) or count < 1:
                raise ValueError("count_sequence visual needs a positive count")
            if sequence != list(range(1, count + 1)):
                raise ValueError("count_sequence visual must show every count from 1 to count")
            if self.data.get("layout") != "left_to_right":
                raise ValueError("count_sequence visual must state its left-to-right order")
        return self


class DictionaryReview(StrictModel):
    status: Literal["approved"]
    approved_by: str = Field(min_length=2, max_length=80)
    approved_at: date


class DictionaryCard(StrictModel):
    card_id: str = Field(pattern=r"^dictionary\.(?:home|cafe)\.[a-z0-9-]+$")
    curriculum_session_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    schema_version: int = Field(ge=1)
    content_version: int = Field(ge=1)
    locale: Literal["ko-KR"] = "ko-KR"
    title: str = Field(min_length=2, max_length=40)
    learning_goal: str = Field(min_length=5, max_length=80)
    concept: DictionaryTextBlock
    example: DictionaryExample
    visual: DictionaryVisual
    method_policy: DictionaryMethodPolicy
    source_refs: list[str] = Field(min_length=1, max_length=6)
    review: DictionaryReview

    @model_validator(mode="after")
    def validate_grounded_standalone_card(self) -> DictionaryCard:
        validate_child_facing_math_copy([self.title, self.learning_goal])
        missing_visual_facts = set(self.visual.fact_refs) - set(self.example.facts)
        if missing_visual_facts:
            raise ValueError(
                f"dictionary visual references unknown facts: {sorted(missing_visual_facts)}"
            )
        for ref in self.visual.fact_refs:
            if ref not in self.visual.data:
                raise ValueError(f"dictionary visual data is missing fact {ref}")
            if self.visual.data[ref] != self.example.facts[ref]:
                raise ValueError(f"dictionary visual fact {ref} does not match the example")
        all_copy = [*self.concept.lines, *self.example.lines]
        if self.method_policy == "open_methods" and any(
            _UNREVIEWED_STRATEGY_COPY.search(line) for line in all_copy
        ):
            raise ValueError("open-method dictionary copy must not force one strategy")
        if not any(char.isdigit() for line in self.example.lines for char in line):
            raise ValueError("dictionary example must include a concrete number")
        return self


class DictionaryCatalog(StrictModel):
    catalog_version: int = Field(ge=1)
    cards: list[DictionaryCard] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_identities(self) -> DictionaryCatalog:
        card_ids = [card.card_id for card in self.cards]
        session_ids = [card.curriculum_session_id for card in self.cards]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("dictionary catalog contains duplicate card_id values")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("dictionary catalog contains duplicate curriculum_session_id values")
        return self


class DictionaryReference(StrictModel):
    card_id: str
    curriculum_session_id: str
    schema_version: int
    content_version: int
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class DictionaryCardEnvelope(StrictModel):
    catalog_version: int
    reference: DictionaryReference
    card: DictionaryCard


def dictionary_content_hash(card: DictionaryCard) -> str:
    dumped = card.model_dump(mode="json")
    # `answer` was added after the first café cards were approved. Omitting an
    # absent value preserves those cards' existing hashes while still hashing
    # the explicit worked-example answer required for home cards.
    if dumped["example"].get("answer") is None:
        dumped["example"].pop("answer", None)
    payload = json.dumps(
        dumped,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dictionary_reference(card: DictionaryCard) -> DictionaryReference:
    return DictionaryReference(
        card_id=card.card_id,
        curriculum_session_id=card.curriculum_session_id,
        schema_version=card.schema_version,
        content_version=card.content_version,
        content_hash=dictionary_content_hash(card),
    )
