"""Independent pre-parent service plus current engines: no live provider calls."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from conftest import FakeGateway

from mormi_api.dialogue_v2_life_runtime import DialogueV2LifeEngine
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import DialogueRuntimeContractVersion
from mormi_api.service import ConversationService

REFERENCE_PATH = Path(__file__).parent / "fixtures/session_parent/baseline_service.py"
name = "mormi_api._session_parent_reference_service"
spec = importlib.util.spec_from_file_location(name, REFERENCE_PATH)
assert spec is not None and spec.loader is not None
REFERENCE = importlib.util.module_from_spec(spec)
sys.modules[name] = REFERENCE
spec.loader.exec_module(REFERENCE)


def service(
    repository: Repository,
    gateway: Any,
    *,
    parent: bool = True,
    reference: bool = False,
    percent: int = 100,
) -> Any:
    cls = REFERENCE.ConversationService if reference else ConversationService
    extra = (
        {}
        if reference
        else {
            "session_parent_graph_enabled": parent,
            "session_parent_graph_canary_percent": percent,
        }
    )
    return cls(
        repository,
        ConversationEngine(FakeGateway()),
        v2_engine=DialogueV2Engine(gateway),
        life_v2_engine=DialogueV2LifeEngine(gateway),
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
        dialogue_v2_canary_percent=100,
        **extra,
    )
