from __future__ import annotations

import pytest

from mormi_api.dialogue_v2_router import select_dialogue_runtime
from mormi_api.schemas import DialogueRuntimeContractVersion, SceneType


def _select(**overrides: object):
    values: dict[str, object] = {
        "configured_version": DialogueRuntimeContractVersion.VERDICT_V1,
        "canary_percent": 100,
        "canary_salt": "reviewed-test-salt",
        "scene": SceneType.HOME_TEACH,
        "scenario_id": "home_teach",
        "curriculum_session_id": "money-count",
        "learner_id": 7,
        "learning_session_id": "learning-1",
        "conversation_round": 1,
    }
    values.update(overrides)
    return select_dialogue_runtime(**values)  # type: ignore[arg-type]


def test_native_home_pack_is_selected_and_stable_at_full_canary() -> None:
    first = _select()
    second = _select()

    assert first == second
    assert first.version is DialogueRuntimeContractVersion.VERDICT_V1
    assert first.reason == "native_pack_canary_selected"
    assert first.bucket is not None


def test_disabled_or_zero_percent_stays_legacy() -> None:
    disabled = _select(
        configured_version=DialogueRuntimeContractVersion.LEGACY_V1,
    )
    zero = _select(canary_percent=0)

    assert disabled.version is DialogueRuntimeContractVersion.LEGACY_V1
    assert disabled.reason == "runtime_disabled"
    assert zero.version is DialogueRuntimeContractVersion.LEGACY_V1
    assert zero.reason == "canary_not_selected"


def test_unsupported_home_and_non_native_scene_are_explicit_legacy_routes() -> None:
    unsupported = _select(curriculum_session_id="addition-basic")
    cafe = _select(
        scene=SceneType.CAFE,
        scenario_id="cafe_queue_demo",
        curriculum_session_id=None,
    )

    assert unsupported.version is DialogueRuntimeContractVersion.LEGACY_V1
    assert unsupported.reason == "native_pack_unavailable"
    assert cafe.version is DialogueRuntimeContractVersion.LEGACY_V1
    assert cafe.reason == "scene_not_v2_eligible"


@pytest.mark.parametrize(
    ("scene", "scenario_id"),
    [
        (SceneType.CAFE, "cafe_queue"),
        (SceneType.CAFE, "cafe_budget_menu"),
        (SceneType.CAFE, "cafe_menu_total"),
        (SceneType.CAFE, "cafe_change"),
        (SceneType.AMUSEMENT_PARK, "amusement_ticket_multiply"),
        (SceneType.AMUSEMENT_PARK, "amusement_snack_divide"),
        (SceneType.AMUSEMENT_PARK, "amusement_pass_compare"),
    ],
)
def test_native_life_scenario_is_canary_eligible_with_stable_identity(
    scene: SceneType,
    scenario_id: str,
) -> None:
    selected = _select(
        scene=scene,
        scenario_id=scenario_id,
        curriculum_session_id=None,
    )

    assert selected.version is DialogueRuntimeContractVersion.VERDICT_V1
    assert selected.reason == "native_life_pack_canary_selected"
    assert selected.bucket is not None


@pytest.mark.parametrize("learning_session_id", [None, "", "   "])
def test_native_life_scenario_without_stable_identity_stays_legacy(
    learning_session_id: str | None,
) -> None:
    selected = _select(
        scene=SceneType.CAFE,
        scenario_id="cafe_queue",
        curriculum_session_id=None,
        learning_session_id=learning_session_id,
    )

    assert selected.version is DialogueRuntimeContractVersion.LEGACY_V1
    assert selected.reason == "stable_identity_unavailable"


def test_canary_bucket_changes_only_with_pinned_identity_inputs() -> None:
    base = _select(canary_percent=50)
    same = _select(canary_percent=50)
    next_round = _select(canary_percent=50, conversation_round=2)

    assert base == same
    assert base.bucket is not None
    assert next_round.bucket is not None
    assert base.bucket != next_round.bucket
