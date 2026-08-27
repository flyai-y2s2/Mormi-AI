from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .dialogue_v2_amusement_content import AMUSEMENT_NATIVE_V2_SCENARIO_IDS
from .dialogue_v2_cafe_content import CAFE_NATIVE_V2_SCENARIO_IDS
from .dialogue_v2_content import REQUIRED_HOME_SESSION_IDS
from .schemas import DialogueRuntimeContractVersion, SceneType


@dataclass(frozen=True, slots=True)
class DialogueRuntimeSelection:
    """One deterministic, conversation-pinned runtime selection."""

    version: DialogueRuntimeContractVersion
    reason: str
    bucket: int | None = None


def _canary_bucket(
    *,
    salt: str,
    learner_id: int,
    learning_session_id: str,
    conversation_round: int,
) -> int:
    material = (
        f"dialogue-v2-canary\0{salt}\0{learner_id}\0"
        f"{learning_session_id}\0{conversation_round}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % 100


def select_dialogue_runtime(
    *,
    configured_version: DialogueRuntimeContractVersion,
    canary_percent: int,
    canary_salt: str,
    scene: SceneType,
    scenario_id: str,
    curriculum_session_id: str | None,
    learner_id: int,
    learning_session_id: str | None,
    conversation_round: int,
) -> DialogueRuntimeSelection:
    """Select V2 only for native packs, otherwise pin the legacy adapter.

    The selector is evaluated only while creating a new conversation.  A
    response or resume must use ``SessionState.runtime_contract_version`` and
    never recalculate this result from current settings.
    """

    if not 0 <= canary_percent <= 100:
        raise ValueError("dialogue V2 canary percent must be between 0 and 100")
    if configured_version is DialogueRuntimeContractVersion.LEGACY_V1:
        return DialogueRuntimeSelection(
            DialogueRuntimeContractVersion.LEGACY_V1,
            "runtime_disabled",
        )
    selected_reason: str
    if scene is SceneType.HOME_TEACH and scenario_id == "home_teach":
        if curriculum_session_id not in REQUIRED_HOME_SESSION_IDS:
            # This is the explicit legacy adapter boundary for the other home
            # sessions.  A conversation must never be labelled verdict-v1 and then
            # silently executed by the old expected-value engine.
            return DialogueRuntimeSelection(
                DialogueRuntimeContractVersion.LEGACY_V1,
                "native_pack_unavailable",
            )
        selected_reason = "native_pack_canary_selected"
    elif (
        scene is SceneType.CAFE
        and scenario_id in CAFE_NATIVE_V2_SCENARIO_IDS
    ) or (
        scene is SceneType.AMUSEMENT_PARK
        and scenario_id in AMUSEMENT_NATIVE_V2_SCENARIO_IDS
    ):
        selected_reason = "native_life_pack_canary_selected"
    else:
        return DialogueRuntimeSelection(
            DialogueRuntimeContractVersion.LEGACY_V1,
            "scene_not_v2_eligible",
        )
    if not learning_session_id or not learning_session_id.strip():
        return DialogueRuntimeSelection(
            DialogueRuntimeContractVersion.LEGACY_V1,
            "stable_identity_unavailable",
        )

    bucket = _canary_bucket(
        salt=canary_salt,
        learner_id=learner_id,
        learning_session_id=learning_session_id,
        conversation_round=conversation_round,
    )
    if bucket >= canary_percent:
        return DialogueRuntimeSelection(
            DialogueRuntimeContractVersion.LEGACY_V1,
            "canary_not_selected",
            bucket,
        )
    return DialogueRuntimeSelection(
        DialogueRuntimeContractVersion.VERDICT_V1,
        selected_reason,
        bucket,
    )
