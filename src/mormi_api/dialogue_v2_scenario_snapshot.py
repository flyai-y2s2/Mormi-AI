from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .dialogue_v2_life_content import LifeScenarioPackV2, LifeTaskPackV2
from .schemas import (
    PinnedDialogueScenarioRuntimeV3,
    PinnedDialogueTaskNoteStateV3,
)


class DialogueV3ScenarioSnapshotError(ValueError):
    """Raised when a persisted life-scene snapshot is not self-consistent."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def life_task_pack_hash_v3(pack: LifeTaskPackV2) -> str:
    """Return the task identity used to bind one task-scoped ledger."""

    canonical = _canonical_json(pack.model_dump(mode="json"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def empty_life_task_ledger_v3(pack: LifeTaskPackV2) -> dict[str, Any]:
    """Create the exact raw-free ledger envelope consumed by the V2 engine."""

    from .dialogue_v2_ledger import (
        empty_reasoning_ledger_v2,
        pin_life_task_pack_v2,
    )

    return empty_reasoning_ledger_v2(pin_life_task_pack_v2(pack)).model_dump(
        mode="json"
    )


def _task_stages(pack: LifeScenarioPackV2) -> dict[str, object]:
    return {stage.task_id: stage for stage in pack.task_stages}


def _validate_task_state(
    snapshot: PinnedDialogueScenarioRuntimeV3,
    pack: LifeScenarioPackV2,
) -> None:
    stages = _task_stages(pack)
    expected_task_ids = set(stages)
    if set(snapshot.active_variant_ids) != expected_task_ids:
        raise DialogueV3ScenarioSnapshotError(
            "scenario snapshot task scope does not match its content pack"
        )

    for task_id, variant_id in snapshot.active_variant_ids.items():
        stage = pack.stage_by_task_id(task_id)
        try:
            task_pack = stage.variants[variant_id]
        except KeyError as error:
            raise DialogueV3ScenarioSnapshotError(
                f"active life task variant is unavailable: {task_id}:{variant_id}"
            ) from error

        ledger = snapshot.reasoning_ledgers[task_id]
        expected_binding = {
            "pack_id": task_pack.pack_id,
            "content_version": task_pack.content_version,
            "content_hash": life_task_pack_hash_v3(task_pack),
        }
        if any(ledger.get(key) != value for key, value in expected_binding.items()):
            raise DialogueV3ScenarioSnapshotError(
                f"life reasoning ledger is not bound to active task variant: {task_id}"
            )

        note_state = snapshot.task_note_states[task_id]
        relation_ids = {
            relation.relation_id for relation in task_pack.reasoning_graph.relations
        }
        note_relation_ids = set(task_pack.policies.note_relation_ids)
        tracked_relation_ids = {
            *note_state.independent_relation_evidence,
            *note_state.supported_relation_ids,
        }
        if not tracked_relation_ids.issubset(note_relation_ids):
            raise DialogueV3ScenarioSnapshotError(
                f"task note state contains an unreviewed relation: {task_id}"
            )
        if not note_relation_ids.issubset(relation_ids):  # pragma: no cover
            raise DialogueV3ScenarioSnapshotError(
                f"task note policy contains an unknown relation: {task_id}"
            )
        if note_state.note_emitted and task_pack.policies.note_policy == "none":
            raise DialogueV3ScenarioSnapshotError(
                f"note-disabled life task cannot have an emitted note: {task_id}"
            )


def resolve_life_scenario_runtime_v3(
    snapshot: PinnedDialogueScenarioRuntimeV3,
) -> LifeScenarioPackV2:
    """Revalidate and return the exact scenario pack stored in a V3 snapshot."""

    try:
        pack = LifeScenarioPackV2.model_validate(snapshot.scenario_pack_snapshot)
    except (TypeError, ValueError) as error:
        raise DialogueV3ScenarioSnapshotError(
            "pinned life scenario payload is invalid"
        ) from error

    from .dialogue_v2_life_content import life_scenario_hash_v2

    if (
        pack.pack_id != snapshot.scenario_pack_id
        or pack.content_version != snapshot.scenario_content_version
        or life_scenario_hash_v2(pack) != snapshot.scenario_source_hash
    ):
        raise DialogueV3ScenarioSnapshotError(
            "pinned life scenario identity or hash does not match its payload"
        )
    _validate_task_state(snapshot, pack)
    return pack


def pin_life_scenario_runtime_v3(
    pack: LifeScenarioPackV2,
    *,
    active_variant_ids: Mapping[str, str] | None = None,
    reasoning_ledgers: Mapping[str, Mapping[str, Any]] | None = None,
    task_note_states: Mapping[str, PinnedDialogueTaskNoteStateV3] | None = None,
    selector_reason: str,
    canary_bucket: int | None,
) -> PinnedDialogueScenarioRuntimeV3:
    """Pin a complete reviewed life scenario and all task-scoped runtime state."""

    selected_variants = dict(
        active_variant_ids
        or {
            stage.task_id: stage.default_variant_id
            for stage in pack.task_stages
        }
    )
    stages = _task_stages(pack)
    if set(selected_variants) != set(stages):
        raise DialogueV3ScenarioSnapshotError(
            "active life variants must cover every scenario task"
        )

    ledgers = (
        {task_id: dict(ledger) for task_id, ledger in reasoning_ledgers.items()}
        if reasoning_ledgers is not None
        else {
            task_id: empty_life_task_ledger_v3(
                pack.stage_by_task_id(task_id).variants[variant_id]
            )
            for task_id, variant_id in selected_variants.items()
        }
    )
    notes = (
        dict(task_note_states)
        if task_note_states is not None
        else {
            task_id: PinnedDialogueTaskNoteStateV3()
            for task_id in selected_variants
        }
    )

    from .dialogue_v2_life_content import life_scenario_hash_v2

    snapshot = PinnedDialogueScenarioRuntimeV3(
        scenario_pack_id=pack.pack_id,
        scenario_content_version=pack.content_version,
        scenario_source_hash=life_scenario_hash_v2(pack),
        scenario_pack_snapshot=pack.model_dump(mode="json"),
        active_variant_ids=selected_variants,
        reasoning_ledgers=ledgers,
        task_note_states=notes,
        selector_reason=selector_reason,
        canary_bucket=canary_bucket,
    )
    resolve_life_scenario_runtime_v3(snapshot)
    return snapshot
