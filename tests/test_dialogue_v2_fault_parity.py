from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_dialogue_v2_runtime import RecordingV2Gateway, _initialize, _response, _run_turn, _state

from mormi_api.dialogue_v2_content import REQUIRED_HOME_SESSION_IDS
from mormi_api.dialogue_v2_runtime import DialogueV2Engine, _primitive
from mormi_api.dialogue_v2_speaker import SpeakerOutputV2
from mormi_api.schemas import InputKind, ResponseType, SessionStatus, UnderstandingResponseV2


def complete_understanding(request: Any) -> UnderstandingResponseV2:
    claims = []
    for target in request.targets:
        relation = target.target_kind == "relation"
        claims.append(
            {
                "claim_kind": target.target_kind,
                "claim_id": "synthetic_" + target.target_id,
                "relation_id" if relation else "fact_id": target.target_id,
                "claim_type": "explanation" if relation else "final_answer",
                "evidence_span": request.child_utterance,
                "verdict": "sufficient" if relation else "correct",
            }
        )
    return UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "contains_learning_evidence": True,
            "answer_status": "complete",
            "reasoning_status": "sufficient",
            "claims": claims,
        }
    )


class FaultGateway(RecordingV2Gateway):
    def __init__(self, role: str, failure: str, recover: bool = False) -> None:
        super().__init__()
        self.role, self.failure, self.recover = role, failure, recover
        self.failed_calls = 0

    async def understand_v2(self, request: Any) -> UnderstandingResponseV2:
        self.understanding_requests.append(request)
        if self.role == "note":
            return complete_understanding(request)
        return UnderstandingResponseV2.model_validate(
            {
                "utterance_class": "non_learning_safe"
                if self.role == "bridge"
                else "learning_response",
                **({"non_learning_kind": "meta"} if self.role == "bridge" else {}),
            }
        )

    async def fault(self) -> bool:
        self.failed_calls += 1
        if self.recover and self.failed_calls > 1:
            return False
        if self.failure == "exception":
            raise RuntimeError("synthetic provider failure")
        if self.failure == "timeout":
            await asyncio.Event().wait()
        return True

    async def speak_v2(self, plan: Any) -> SpeakerOutputV2:
        if self.role == "speaker" and await self.fault():
            return SpeakerOutputV2(text="정답은 999999원이야", mood="curious")
        return await super().speak_v2(plan)

    async def bridge_speak_v2(self, plan: Any) -> SpeakerOutputV2:
        if self.role == "bridge" and await self.fault():
            return SpeakerOutputV2(text="정답은 999999원이야", mood="curious")
        return await super().bridge_speak_v2(plan)

    async def contextualize_note(self, context: Any) -> Any:
        if self.role == "note" and await self.fault():
            raise ValueError("synthetic invalid note")
        return await super().contextualize_note(context)


@pytest.mark.parametrize("role", ["speaker", "bridge", "note"])
@pytest.mark.parametrize("failure", ["reject", "exception", "timeout"])
@pytest.mark.parametrize("recover", [False, True])
async def test_role_failure_call_budget(role: str, failure: str, recover: bool) -> None:
    gateway = FaultGateway(role, failure, recover)
    engine = DialogueV2Engine(gateway, speaker_timeout_seconds=0.005, bridge_timeout_seconds=0.005)
    state = _state("divide-share")
    turn = await _initialize(engine, state, "divide-share")
    result = await _run_turn(
        engine,
        state,
        _response(turn.turn_id, ResponseType.TEXT, text="10500을 3으로 나누면 3500원"),
        turn.mormi.text,
    )
    assert gateway.failed_calls == (2 if role == "speaker" else 1)
    if role == "note":
        assert result.turn.note_update is not None
        assert result.turn.status is SessionStatus.COMPLETED


@pytest.mark.parametrize("session_id", sorted(REQUIRED_HOME_SESSION_IDS))
async def test_each_home_pack_finishes_supported(session_id: str) -> None:
    engine = DialogueV2Engine(RecordingV2Gateway())
    state = _state(session_id)
    turn = await _initialize(engine, state, session_id)
    for _ in range(30):
        if turn.status is SessionStatus.COMPLETED:
            assert turn.note_update is not None
            return
        if turn.input.kind is InputKind.JOINT:
            pack, _, _, _ = engine._resolve_state(state)
            response = _response(
                turn.turn_id,
                ResponseType.ACTION,
                values={
                    f"{item.target_kind}:{item.target_id}": _primitive(item.value)
                    if item.target_kind == "fact"
                    else True
                    for item in pack.l0_joint_plan.completion_values
                },
            )
        else:
            response = _response(turn.turn_id, ResponseType.NO_RESPONSE)
        result = await _run_turn(engine, state, response, turn.mormi.text)
        state, turn = result.state, result.turn
    raise AssertionError("support route did not terminate")


@pytest.mark.parametrize("mode", ["cold", "warm", "busy", "failed"])
async def test_runtime_cache_resolution_and_pinning(mode: str, tmp_path: Any) -> None:
    from test_dialogue_v2_copy import ReviewedCopyGenerator, _fallbacks, _runtime

    from mormi_api.dialogue_v2_content import required_home_content_pack_v2
    from mormi_api.dialogue_v2_copy import build_stable_copy_work_items_v2

    item = next(
        item
        for item in build_stable_copy_work_items_v2(required_home_content_pack_v2("money-budget"))
        if item.plan.purpose == "initial_help"
    )
    generator = ReviewedCopyGenerator(_fallbacks(), raises=mode == "failed")
    database, repository, resolver = await _runtime(tmp_path, generator)
    try:
        if mode == "warm":
            await resolver.resolve(
                item.plan,
                reviewed_fallback=item.reviewed_fallback,
                pack_hash=item.pack_hash,
                output_firewall=item.output_firewall,
            )
        elif mode == "busy":
            await repository.acquire(resolver._cache_key(item.plan, item.pack_hash))
        engine = DialogueV2Engine(RecordingV2Gateway(), copy_resolver=resolver)
        state = _state("money-budget")
        initial = await _initialize(engine, state, "money-budget")
        first = await _run_turn(
            engine, state, _response(initial.turn_id, ResponseType.NO_RESPONSE), initial.mormi.text
        )
        assert first.state.pinned_dialogue_v2.copy_snapshots
        calls = len(generator.calls)
        state.pinned_dialogue_v2 = state.pinned_dialogue_v2.model_copy(
            update={
                "copy_snapshots": first.state.pinned_dialogue_v2.copy_snapshots,
            },
            deep=True,
        )
        await _run_turn(
            engine, state, _response(initial.turn_id, ResponseType.NO_RESPONSE), initial.mormi.text
        )
        assert len(generator.calls) == calls
    finally:
        await database.dispose()
