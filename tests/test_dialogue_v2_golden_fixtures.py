from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.schemas import (
    ChoiceOption,
    CompletionContract,
    DialogueRuntimeContractVersion,
    ExpressionLevel,
    HelpCardContract,
    InputContract,
    MormiContract,
    NoteUpdate,
    PedagogySnapshot,
    SceneType,
    SessionState,
    TaskAnchorCompletedItem,
    TaskAnchorContract,
    TurnContract,
    VisualContract,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dialogue_v2"
FIXTURE_PATHS = tuple(sorted(FIXTURE_DIR.glob("*.json")))
EXPECTED_FIXTURES = {
    "completed_supported",
    "completed_taught",
    "l0_joint_h3",
    "l2_choices",
    "l4_milestone_11000_remaining",
    "l4_text",
}

FORBIDDEN_SENSITIVE_KEYS = {
    "address",
    "birth_date",
    "child_name",
    "child_utterance",
    "email",
    "learner_id",
    "phone",
    "raw_child_text",
    "raw_text",
    "resident_registration_number",
    "transcript",
}
SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)01[016789][ -]?\d{3,4}[ -]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{6}[ -]?[1-4]\d{6}(?!\d)"),
    re.compile(r"(?i)\b(?:bearer|sk-ant-|api[_-]?key)\b"),
)


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _walk_json(value: object, path: str = "$") -> Iterator[tuple[str, object]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]")


def _assert_exact_fields(payload: dict[str, Any], model: type[Any]) -> None:
    assert set(payload) == set(model.model_fields)


def test_golden_fixture_set_covers_all_v2_wire_states() -> None:
    assert {path.stem for path in FIXTURE_PATHS} == EXPECTED_FIXTURES
    assert len(FIXTURE_PATHS) >= 5


@pytest.mark.asyncio
async def test_initial_runtime_turn_matches_the_l4_golden_fixture() -> None:
    state = SessionState(
        learner_id=7,
        learning_session_id="learning-multiply-easy-tables",
        scene=SceneType.HOME_TEACH,
        scenario_id="home_teach",
        task_ids=["home_teaching"],
        task_start_levels={"home_teaching": ExpressionLevel.L4},
        scenario_data={"curriculum_session_id": "multiply-easy-tables"},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
    )
    engine = DialogueV2Engine(object(), show_internal_pedagogy=True)  # type: ignore[arg-type]

    turn = await engine.initialize_state(
        state,
        curriculum_session_id="multiply-easy-tables",
        selector_reason="golden_fixture",
        canary_bucket=17,
    )
    expected = _load_fixture(FIXTURE_DIR / "l4_text.json")

    assert turn.model_copy(
        update={"turn_id": expected["turn_id"]},
        deep=True,
    ).model_dump(mode="json") == expected


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_each_golden_fixture_is_an_exact_turn_contract(path: Path) -> None:
    payload = _load_fixture(path)
    turn = TurnContract.model_validate(payload)

    assert turn.model_dump(mode="json") == payload
    assert turn.schema_version == "turn-contract-v1"
    assert turn.scenario_id == "home_teach"


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_golden_fixtures_include_required_external_wire_fields(path: Path) -> None:
    payload = _load_fixture(path)
    _assert_exact_fields(payload, TurnContract)
    _assert_exact_fields(payload["mormi"], MormiContract)
    _assert_exact_fields(payload["input"], InputContract)
    _assert_exact_fields(payload["visual"], VisualContract)

    for choice in payload["input"]["choices"]:
        _assert_exact_fields(choice, ChoiceOption)
    if payload["help_card"] is not None:
        _assert_exact_fields(payload["help_card"], HelpCardContract)
    if payload["note_update"] is not None:
        _assert_exact_fields(payload["note_update"], NoteUpdate)
    if payload["completion"] is not None:
        _assert_exact_fields(payload["completion"], CompletionContract)
    if payload["pedagogy"] is not None:
        _assert_exact_fields(payload["pedagogy"], PedagogySnapshot)
    if payload["task_anchor"] is not None:
        _assert_exact_fields(payload["task_anchor"], TaskAnchorContract)
        for item in payload["task_anchor"]["completed_items"]:
            _assert_exact_fields(item, TaskAnchorCompletedItem)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_golden_fixtures_contain_no_child_raw_text_or_pii(path: Path) -> None:
    payload = _load_fixture(path)
    for json_path, value in _walk_json(payload):
        if isinstance(value, dict):
            forbidden = FORBIDDEN_SENSITIVE_KEYS.intersection(
                key.casefold() for key in value
            )
            assert not forbidden, f"{json_path} has sensitive keys: {sorted(forbidden)}"
        if isinstance(value, str):
            for pattern in SENSITIVE_VALUE_PATTERNS:
                assert pattern.search(value) is None, (
                    f"{json_path} resembles sensitive data: {pattern.pattern}"
                )


def test_active_ladder_fixture_semantics_are_explicit() -> None:
    l4 = _load_fixture(FIXTURE_DIR / "l4_text.json")
    milestone = _load_fixture(FIXTURE_DIR / "l4_milestone_11000_remaining.json")
    l2 = _load_fixture(FIXTURE_DIR / "l2_choices.json")
    l0 = _load_fixture(FIXTURE_DIR / "l0_joint_h3.json")

    assert (l4["pedagogy"]["expression_level"], l4["input"]["kind"]) == ("L4", "text")

    # Correct partial progress preserves the existing L/H policy. The ledger
    # regression separately proves that 11,000 is pinned as a milestone; the
    # public turn acknowledges it and asks only the still-required targets.
    assert milestone["pedagogy"]["expression_level"] == "L4"
    assert milestone["input"]["target_slots"] == [
        "fact:shortage",
        "relation:calculate_shortage",
    ]
    assert "11,000원" in milestone["mormi"]["text"]

    assert l2["pedagogy"]["expression_level"] == "L2"
    assert l2["pedagogy"]["hint_level"] == "H2"
    assert l2["input"]["kind"] == "choices"
    assert len(l2["input"]["choices"]) >= 2
    assert all(
        "correct" not in choice and "effect" not in choice
        for choice in l2["input"]["choices"]
    )

    assert (l0["pedagogy"]["expression_level"], l0["pedagogy"]["hint_level"]) == (
        "L0",
        "H3",
    )
    assert l0["input"]["kind"] == "joint"
    assert l0["input"]["config"]["completion_values"] == {
        "fact:shortage": 1_000,
        "relation:calculate_shortage": True,
    }
    assert l0["help_card"]["level"] == "H3"
    assert l0["help_card"]["auto_open"] is True


@pytest.mark.parametrize(
    ("fixture_name", "outcome", "reward_eligible"),
    (
        ("completed_taught.json", "taught", True),
        ("completed_supported.json", "supported", False),
    ),
)
def test_completed_fixtures_have_terminal_wire_contracts(
    fixture_name: str,
    outcome: str,
    reward_eligible: bool,
) -> None:
    payload = _load_fixture(FIXTURE_DIR / fixture_name)

    assert payload["status"] == "completed"
    assert payload["input"]["kind"] == "none"
    assert payload["input"]["target_slots"] == []
    assert payload["task_anchor"] is None
    assert payload["completion"]["outcome"] == outcome
    assert payload["completion"]["teach_reward_eligible"] is reward_eligible
    assert payload["completion"]["stage_completion_eligible"] is True
