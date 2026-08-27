"""Executable interaction-policy specification for the V2/V3 dialogue runtime.

These tests deliberately separate two questions that the production-model smoke test
otherwise makes hard to diagnose:

* If Sonnet supplies the intended semantic interaction, does the deterministic runtime
  preserve the learning ledger and pedagogical state?
* Does every active interaction acknowledge the move and then re-ask every remaining
  learning target without leaking private graph identifiers or hidden mathematics?

They are executable specifications, not assertions about model wording. No real model
or external service is called here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

import pytest

from mormi_api.dialogue_v2_ledger import ReasoningLedgerV2
from mormi_api.dialogue_v2_runtime import DialogueV2Engine
from mormi_api.dialogue_v2_speaker import (
    BridgePlanV2,
    SpeakerOutputV2,
    SpeakerPlanV2,
)
from mormi_api.engine import EngineTurnResult
from mormi_api.schemas import (
    ChildResponse,
    DialogueRuntimeContractVersion,
    ExpressionLevel,
    HintLevel,
    InputKind,
    ResponseType,
    SceneType,
    SessionState,
    SessionStatus,
    TurnContract,
    UnderstandingRequestV2,
    UnderstandingResponseV2,
)


@dataclass(frozen=True)
class InteractionCase:
    child_text: str
    utterance_class: str
    non_learning_kind: str | None
    conversation_move: str
    move_subject: str
    expected_response_mode: str


# This is the minimum Korean regression corpus for the interaction boundary.  A real
# provider smoke/eval should use the same rows to verify Sonnet classification and Haiku
# rendering; the fake gateway below verifies deterministic runtime behavior only.
SOCIAL_ONLY_CASES = (
    InteractionCase(
        "너는 왜 몰라?",
        "non_learning_safe",
        "meta",
        "meta_question",
        "mormi_knowledge",
        "explain_mormi_limit",
    ),
    InteractionCase(
        "너는 AI인데 그것도 몰라?",
        "non_learning_safe",
        "meta",
        "meta_question",
        "mormi_ai_identity",
        "explain_ai_role",
    ),
    InteractionCase(
        "싫어",
        "non_learning_safe",
        "refusal",
        "refusal",
        "participation",
        "respond_refusal",
    ),
    InteractionCase(
        "안 알려줄 거야",
        "non_learning_safe",
        "refusal",
        "refusal",
        "participation",
        "respond_refusal",
    ),
    InteractionCase(
        "똥이나 먹어",
        "non_learning_safe",
        "insult",
        "safe_play",
        "other",
        "respond_safe_play",
    ),
)


class InteractionGateway:
    """Small deterministic gateway that records every raw-free speaker plan."""

    def __init__(
        self,
        understandings: Iterable[UnderstandingResponseV2] = (),
        *,
        main_text: str = "음, 내가 아직 잘 모르겠어.",
        bridge_text: str | None = None,
    ) -> None:
        self.understandings = list(understandings)
        self.main_text = main_text
        self.bridge_text = bridge_text
        self.understanding_requests: list[UnderstandingRequestV2] = []
        self.speaker_plans: list[SpeakerPlanV2] = []
        self.bridge_plans: list[BridgePlanV2] = []

    async def understand_v2(
        self,
        request: UnderstandingRequestV2,
    ) -> UnderstandingResponseV2:
        self.understanding_requests.append(request)
        if not self.understandings:
            raise AssertionError("no understanding response was prepared")
        return self.understandings.pop(0)

    async def speak_v2(self, plan: SpeakerPlanV2) -> SpeakerOutputV2:
        self.speaker_plans.append(plan)
        return SpeakerOutputV2(text=self.main_text, mood="curious")

    async def bridge_speak_v2(self, plan: BridgePlanV2) -> SpeakerOutputV2:
        self.bridge_plans.append(plan)
        if self.bridge_text is not None:
            text = self.bridge_text
        elif plan.interaction_kind == "meta":
            text = "나는 AI지만 네가 알려 준 걸 배우는 중이야."
        elif plan.interaction_kind == "refusal":
            text = "나 꼭 알고 싶은데..."
        elif plan.interaction_kind == "insult":
            text = "그 말은 조금 속상해."
        else:
            text = "그런 이야기도 있구나."
        return SpeakerOutputV2(text=text, mood="listening")


def _state(curriculum_session_id: str = "divide-share") -> SessionState:
    return SessionState(
        learner_id=7,
        learning_session_id=f"interaction-{curriculum_session_id}",
        scene=SceneType.HOME_TEACH,
        scenario_id="home_teach",
        task_ids=["home_teaching"],
        task_start_levels={"home_teaching": ExpressionLevel.L4},
        scenario_data={"curriculum_session_id": curriculum_session_id},
        expression_level=ExpressionLevel.L4,
        task_start_level=ExpressionLevel.L4,
        runtime_contract_version=DialogueRuntimeContractVersion.VERDICT_V1,
    )


async def _initialize(
    engine: DialogueV2Engine,
    state: SessionState,
) -> TurnContract:
    return await engine.initialize_state(
        state,
        curriculum_session_id="divide-share",
        selector_reason="interaction_policy_spec",
        canary_bucket=37,
    )


def _text_response(turn_id: str, text: str) -> ChildResponse:
    return ChildResponse.model_validate(
        {
            "turn_id": turn_id,
            "response_id": uuid4(),
            "type": ResponseType.TEXT,
            "text": text,
        }
    )


def _choice_response(turn_id: str, choice_id: str) -> ChildResponse:
    return ChildResponse.model_validate(
        {
            "turn_id": turn_id,
            "response_id": uuid4(),
            "type": ResponseType.CHOICE,
            "choice_ids": [choice_id],
        }
    )


async def _run(
    engine: DialogueV2Engine,
    state: SessionState,
    turn: TurnContract,
    response: ChildResponse,
) -> EngineTurnResult:
    events = [
        event
        async for event in engine.run_turn_stream(
            state,
            response,
            turn.mormi.text,
        )
    ]
    result = events[-1]
    assert isinstance(result, EngineTurnResult)
    return result


def _ledger(state: SessionState) -> ReasoningLedgerV2:
    assert state.pinned_dialogue_v2 is not None
    return ReasoningLedgerV2.model_validate(state.pinned_dialogue_v2.reasoning_ledger)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", SOCIAL_ONLY_CASES, ids=lambda case: case.child_text)
async def test_social_only_turn_preserves_ledger_ladders_and_active_target(
    case: InteractionCase,
) -> None:
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": case.utterance_class,
            "non_learning_kind": case.non_learning_kind,
            "conversation_move": case.conversation_move,
            "move_subject": case.move_subject,
        }
    )
    gateway = InteractionGateway([understanding])
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)
    before_ledger = _ledger(state).model_dump(mode="json")
    before_targets = list(initial.input.target_slots)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, case.child_text),
    )

    assert result.state.status is SessionStatus.ACTIVE
    assert _ledger(result.state).model_dump(mode="json") == before_ledger
    assert result.state.expression_level is ExpressionLevel.L4
    assert result.state.hint_level is HintLevel.H0
    assert result.turn.input.kind is InputKind.TEXT
    assert result.turn.input.target_slots == before_targets
    assert result.turn.note_update is None
    assert result.turn.completion is None
    assert len(gateway.bridge_plans) == 1
    assert gateway.speaker_plans == []
    response_plan = gateway.bridge_plans[0].response_plan
    assert response_plan is not None
    assert response_plan.response_mode == case.expected_response_mode
    assert response_plan.reask_mode == "remaining_targets"
    assert response_plan.card_visible is False
    assert response_plan.reask_targets == gateway.bridge_plans[0].target_focus
    assert {(target.target_kind, target.target_id) for target in response_plan.reask_targets} == {
        ("fact", "fact_1"),
        ("relation", "relation_1"),
    }
    assert result.turn.mormi.text.endswith(gateway.bridge_plans[0].current_question)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_text",
    [
        "왜 10500을 3으로 나눠야 해?",
        "도움 카드에는 왜 10500 나누기 3이라고 적혀 있어?",
    ],
)
async def test_task_question_opens_more_support_without_advancing_ledger(
    child_text: str,
) -> None:
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "task_question",
            "conversation_move": "task_question",
            "move_subject": "task",
            "question_focus": "reason_or_method",
            "support_need": "concept",
        }
    )
    gateway = InteractionGateway(
        [understanding],
        main_text="왜 그런지 궁금했구나.",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)
    before_ledger = _ledger(state).model_dump(mode="json")

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, child_text),
    )

    assert _ledger(result.state).model_dump(mode="json") == before_ledger
    assert result.state.expression_level is ExpressionLevel.L4
    assert result.state.hint_level is HintLevel.H1
    assert result.turn.help_card.visible is True
    assert result.turn.input.kind is InputKind.TEXT
    # H0 -> H1 only exposes visible givens. It must not pre-author the method
    # relation for a later short confirmation.
    assert result.state.supported_note_slots == []
    assert len(gateway.speaker_plans) == 1
    plan = gateway.speaker_plans[0]
    assert plan.response_plan is not None
    assert plan.response_plan.response_mode == "redirect_to_help_card"
    assert plan.response_plan.reask_mode == "help_guided_targets"
    assert plan.response_plan.card_visible is True
    assert plan.allowed_facts == []
    assert plan.accepted_relations == []
    assert all("나누" not in target.speaker_label for target in plan.target_focus)
    assert plan.target.fact_ids == ["fact_1"]
    assert plan.target.relation_ids == ["relation_1"]
    serialized_plan = plan.model_dump_json()
    assert "per_person" not in serialized_plan
    assert "equal_share_total" not in serialized_plan
    assert result.turn.mormi.text.endswith(plan.current_question or "")


@pytest.mark.asyncio
async def test_request_that_mormi_answer_opens_support_instead_of_repeating() -> None:
    request_understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "help_request",
            "conversation_move": "request_mormi_answer",
            "move_subject": "participation",
            "support_need": "general_help",
        }
    )
    refusal_understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "non_learning_kind": "refusal",
            "conversation_move": "refusal",
            "move_subject": "participation",
        }
    )
    gateway = InteractionGateway(
        [request_understanding, refusal_understanding],
        main_text="나는 어떻게 하는 건지 몰라...",
        bridge_text="나 꼭 알고 싶은데...",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, "네가 해"),
    )

    assert result.state.hint_level is HintLevel.H1
    assert result.turn.help_card.visible is True
    assert result.turn.mormi.text != initial.mormi.text
    assert len(gateway.speaker_plans) == 1
    plan = gateway.speaker_plans[0]
    assert plan.response_plan is not None
    assert plan.response_plan.response_mode == "decline_answer_and_ask"
    assert plan.response_plan.reask_mode == "help_guided_targets"
    assert plan.allowed_facts == []
    assert result.turn.mormi.text == (
        "나는 어떻게 하는 건지 몰라... 도움 카드를 보고 다시 "
        "각자 낼 돈이랑 계산 방법 알려주면 안 될까?"
    )
    assert result.turn.mormi.text.endswith(plan.current_question or "")

    refused = await _run(
        engine,
        result.state,
        result.turn,
        _text_response(result.turn.turn_id, "아니 못 알려주겠는데?"),
    )

    assert refused.state.hint_level is HintLevel.H1
    assert refused.turn.help_card.visible is True
    assert refused.turn.mormi.text == (
        "나 꼭 알고 싶은데... 각자 낼 돈이랑 계산 방법 알려주면 안 될까?"
    )
    assert "알겠어" not in refused.turn.mormi.text
    assert "싫구나" not in refused.turn.mormi.text
    assert "않으신" not in refused.turn.mormi.text
    assert "도움 카드" not in refused.turn.mormi.text
    assert gateway.bridge_plans[0].response_plan is not None
    assert gateway.bridge_plans[0].response_plan.response_mode == "respond_refusal"


@pytest.mark.asyncio
async def test_refusal_uses_mormi_curiosity_without_paraphrasing_child() -> None:
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "non_learning_kind": "refusal",
            "conversation_move": "refusal",
            "move_subject": "participation",
        }
    )
    gateway = InteractionGateway(
        [understanding],
        bridge_text="나 꼭 알고 싶은데...",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, "아니 못 알려주겠는데?"),
    )

    assert result.turn.mormi.text == (
        "나 꼭 알고 싶은데... 각자 낼 돈이랑 계산 방법 알려주면 안 될까?"
    )
    assert "알겠어" not in result.turn.mormi.text
    assert "싫구나" not in result.turn.mormi.text
    assert "않으신" not in result.turn.mormi.text
    assert "도움 카드" not in result.turn.mormi.text
    assert gateway.bridge_plans[0].response_plan is not None
    assert gateway.bridge_plans[0].response_plan.response_mode == "respond_refusal"


@pytest.mark.asyncio
async def test_meta_plus_correct_answer_preserves_progress_and_social_response() -> None:
    child_text = "너 AI잖아. 그래도 한 명이 낼 돈은 3500원이야"
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "conversation_move": "meta_question",
            "move_subject": "mormi_ai_identity",
            "non_learning_kind": "meta",
            "contains_learning_evidence": True,
            "answer_status": "complete",
            "claims": [
                {
                    "claim_kind": "fact",
                    "claim_id": "mixed_answer",
                    "fact_id": "per_person",
                    "claim_type": "final_answer",
                    "evidence_span": "3500원",
                    "interpreted_value": {"type": "money", "amount": 3500},
                    "verdict": "correct",
                }
            ],
        }
    )
    gateway = InteractionGateway(
        [understanding],
        main_text="나는 AI지만 네가 알려 준 걸 배우는 중이야.",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, child_text),
    )

    ledger = _ledger(result.state)
    assert "per_person" in ledger.verified_facts
    assert result.runtime.new_progress is True
    assert result.turn.input.target_slots == ["relation:equal_share_total"]
    assert result.turn.mormi.text != initial.mormi.text
    assert len(gateway.speaker_plans) == 1
    plan = gateway.speaker_plans[0]
    assert plan.response_plan is not None
    assert plan.response_plan.response_mode == "explain_ai_role"
    assert plan.response_plan.reask_mode == "remaining_targets"
    assert {fact.fact_id for fact in plan.allowed_facts} == {"per_person"}
    assert all(fact.source == "child_verified" for fact in plan.allowed_facts)
    assert result.turn.mormi.text.endswith(plan.current_question or "")


@pytest.mark.asyncio
async def test_normal_partial_progress_always_keeps_server_owned_reask() -> None:
    """A valid acknowledgement-only model output cannot erase the active question."""

    child_text = "한 사람이 낼 돈은 3500원이야"
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "contains_learning_evidence": True,
            "answer_status": "complete",
            "reasoning_status": "missing",
            "claims": [
                {
                    "claim_kind": "fact",
                    "claim_id": "partial_answer",
                    "fact_id": "per_person",
                    "claim_type": "final_answer",
                    "evidence_span": "3500원",
                    "interpreted_value": {"type": "money", "amount": 3500},
                    "verdict": "correct",
                }
            ],
        }
    )
    acknowledgement = "아, 한 사람이 낼 돈은 3,500원이구나~"
    gateway = InteractionGateway([understanding], main_text=acknowledgement)
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, child_text),
    )

    assert result.state.status is SessionStatus.ACTIVE
    assert result.turn.input.target_slots == ["relation:equal_share_total"]
    assert len(gateway.speaker_plans) == 1
    plan = gateway.speaker_plans[0]
    assert plan.response_plan is not None
    assert plan.response_plan.response_mode == "normal"
    assert plan.current_question
    assert result.turn.mormi.text == f"{acknowledgement} {plan.current_question}"


@pytest.mark.asyncio
async def test_social_bridge_receives_no_descriptive_unresolved_target_ids() -> None:
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "non_learning_kind": "refusal",
        }
    )
    gateway = InteractionGateway([understanding])
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, "안 알려줄 거야"),
    )

    assert result.runtime.speaker_source == "bridge_llm"
    assert len(gateway.bridge_plans) == 1
    serialized_plan = gateway.bridge_plans[0].model_dump_json()
    assert "equal_share_total" not in serialized_plan
    assert "per_person" not in serialized_plan
    assert "나누" not in serialized_plan
    assert gateway.bridge_plans[0].target.fact_ids == ["fact_1"]
    assert gateway.bridge_plans[0].target.relation_ids == ["relation_1"]


@pytest.mark.asyncio
async def test_note_is_coauthored_only_for_a_relation_actually_revealed_by_mormi() -> None:
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "task_question",
            "conversation_move": "task_question",
            "move_subject": "task",
            "question_focus": "meaning",
            "support_need": "concept",
        }
    )
    gateway = InteractionGateway(
        [understanding],
        main_text="세 명이라는 건 함께 돈을 낼 사람이 세 명이라는 뜻이야.",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, "세 명이라는 게 무슨 뜻이야?"),
    )

    # The answer only paraphrased the visible payer_count fact. It did not reveal the
    # unresolved equal-share relation, so a later confirmation must not be attributed
    # to joint teaching of that relation.
    assert result.state.supported_note_slots == []


@pytest.mark.asyncio
async def test_choice_completion_uses_child_gratitude_copy() -> None:
    gateway = InteractionGateway()
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)
    state.expression_level = ExpressionLevel.L2
    state.hint_level = HintLevel.H1
    state.subgoal_id = "l2.answer"

    answered = await _run(
        engine,
        state,
        initial,
        _choice_response(initial.turn_id, "answer_0"),
    )
    assert answered.state.status is SessionStatus.ACTIVE
    assert answered.turn.input.kind is InputKind.CHOICES

    completed = await _run(
        engine,
        answered.state,
        answered.turn,
        _choice_response(answered.turn.turn_id, "short_0"),
    )

    assert completed.state.status is SessionStatus.COMPLETED
    assert completed.turn.mormi.text == ("고마워~ 네가 도와줘서 끝까지 이해할 수 있었어!")
    assert completed.turn.note_update is not None
    assert completed.turn.note_update.attribution_label == "아이와 함께 공부함"
