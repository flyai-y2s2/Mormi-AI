"""Versioned capabilities for persisted V2 conversation snapshots.

This aggregate reader version covers the whole persisted boundary, not only a
single nested model.  It must be bumped when a process can no longer read any
of the listed component formats from an in-flight conversation.
"""

from __future__ import annotations

from typing import Final

DIALOGUE_V2_SNAPSHOT_READER_CAPABILITY_V2: Final = (
    "dialogue-v2-snapshot-reader-v2"
)

# Kept explicit so tests and release reviews can see what the aggregate reader
# promises to decode.  These values are persisted inside SessionState JSON.
DIALOGUE_V2_SNAPSHOT_COMPONENTS_V2: Final = (
    "pinned-dialogue-runtime-v2",
    "pinned-content-v2",
    "content-pack-v2",
    "reasoning-ledger-v2",
    "stable-copy-plan-set-v1",
    "stable-copy-plan-compiler-v1",
    "stable-copy-resolution-v1",
    "stable-copy-cache-artifact-v1",
    "turn-contract-v1",
)

DIALOGUE_V3_SNAPSHOT_READER_CAPABILITY_V1: Final = (
    "dialogue-v3-snapshot-reader-v1"
)

# V3 is an aggregate superset: the same process must resume old single-pack
# home conversations and the new task-scoped cafe/amusement snapshots.
DIALOGUE_V3_SNAPSHOT_COMPONENTS_V1: Final = (
    *DIALOGUE_V2_SNAPSHOT_COMPONENTS_V2,
    "pinned-dialogue-scenario-runtime-v3",
    "task-note-state-v3",
    "life-scenario-pack-v2",
    "life-materializer-v2",
    "life-task-pack-v2",
)
