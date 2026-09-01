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

EXPRESSION_BLOCK_CASES = (
    "뭐라고 설명할지 모르겠어",
    "설명하기 어려워",
    "뭐라고 말해야 할지 모르겠어",
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
    plan = gateway.bridge_plans[0]
    assert plan.reaction_mode == case.expected_response_mode
    assert plan.response_plan is None
    assert plan.target_focus == []
    assert plan.target.fact_ids == []
    assert plan.target.relation_ids == []
    assert plan.current_question is None
    assert result.turn.mormi.text.count("?") == 1


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
    assert plan.response_plan is None
    assert plan.allowed_facts == []
    assert plan.accepted_relations == []
    assert plan.target_focus == []
    assert plan.target.fact_ids == []
    assert plan.target.relation_ids == []
    serialized_plan = plan.model_dump_json()
    assert "per_person" not in serialized_plan
    assert "equal_share_total" not in serialized_plan
    assert plan.current_question is None
    assert result.turn.mormi.text.count("?") == 1


@pytest.mark.asyncio
async def test_visible_help_card_reference_is_grounded_but_speaker_gets_only_ui_signal(
) -> None:
    child_text = "저걸 왜 주의 깊게 봐야 돼?"
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "task_question",
            "conversation_move": "task_question",
            "move_subject": "task",
            "question_focus": "reason_or_method",
            "support_need": "concept",
            "ui_reference": {
                "element_id": "help_card.h1",
                "interaction": "asks_why",
                "reference_basis": "deictic",
                "evidence_span": "저걸",
            },
        }
    )
    gateway = InteractionGateway(
        [understanding],
        main_text="그러게, 왜 그런지는 나도 아직 모르겠어...",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)
    # Simulate the preceding committed turn that exposed H1. The response
    # under test must observe this pre-turn card, not the H2 card opened later.
    state.hint_level = HintLevel.H1
    state.task_max_hint = HintLevel.H1
    before_ledger = _ledger(state).model_dump(mode="json")

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, child_text),
    )

    request = gateway.understanding_requests[0]
    assert len(request.visible_ui_elements) == 1
    visible_card = request.visible_ui_elements[0]
    assert visible_card.kind.value == "help_card"
    assert visible_card.hint_level is HintLevel.H1
    assert result.state.hint_level is HintLevel.H2
    assert _ledger(result.state).model_dump(mode="json") == before_ledger
    assert len(gateway.speaker_plans) == 1
    assert gateway.bridge_plans == []
    plan = gateway.speaker_plans[0]
    assert plan.response_signal.ui_reference is not None
    assert plan.response_signal.ui_reference.interaction.value == "asks_why"
    assert plan.response_signal.ui_reference.card_event == "opened_or_strengthened"
    serialized_plan = plan.model_dump_json()
    assert visible_card.text not in serialized_plan
    assert "visible_ui_elements" not in serialized_plan
    assert plan.allowed_facts == []
    assert plan.accepted_relations == []
    assert "다른 도움 카드" not in result.turn.mormi.text
    assert result.turn.mormi.text.count("?") == 1


@pytest.mark.asyncio
async def test_ui_reference_keeps_a_surface_refusal_on_the_grounded_main_route() -> None:
    child_text = "낸 돈은 10500원이고 사람은 3명인데 뭐 어쩌라고"
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "non_learning_safe",
            "conversation_move": "refusal",
            "move_subject": "participation",
            "non_learning_kind": "refusal",
            "support_need": "concept",
            "ui_reference": {
                "element_id": "help_card.h1",
                "interaction": "asks_what_next",
                "reference_basis": "content_echo",
                "evidence_span": "낸 돈은 10500원이고 사람은 3명인데 뭐 어쩌라고",
            },
        }
    )
    gateway = InteractionGateway(
        [understanding],
        main_text="나도 그다음에 뭘 해야 하는지는 아직 모르겠어...",
    )
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)
    state.hint_level = HintLevel.H1
    state.task_max_hint = HintLevel.H1
    before_ledger = _ledger(state).model_dump(mode="json")

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, child_text),
    )

    assert result.state.hint_level is HintLevel.H2
    assert _ledger(result.state).model_dump(mode="json") == before_ledger
    assert gateway.bridge_plans == []
    assert len(gateway.speaker_plans) == 1
    signal = gateway.speaker_plans[0].response_signal.ui_reference
    assert signal is not None
    assert signal.interaction.value == "asks_what_next"
    assert gateway.speaker_plans[0].accepted_relations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("child_text", EXPRESSION_BLOCK_CASES)
@pytest.mark.parametrize(
    ("start_level", "expected_level", "expected_input"),
    [
        (ExpressionLevel.L4, ExpressionLevel.L3, InputKind.TEXT),
        (ExpressionLevel.L3, ExpressionLevel.L2, InputKind.CHOICES),
        (ExpressionLevel.L2, ExpressionLevel.L0, InputKind.JOINT),
    ],
)
async def test_expression_block_immediately_lowers_visible_expression_contract(
    child_text: str,
    start_level: ExpressionLevel,
    expected_level: ExpressionLevel,
    expected_input: InputKind,
) -> None:
    """Sonnet's expression intent must become visible support in the same turn."""

    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "help_request",
            "conversation_move": "none",
            "move_subject": "other",
            "support_need": "expression",
        }
    )
    gateway = InteractionGateway([understanding])
    engine = DialogueV2Engine(gateway)
    state = _state()
    initial = await _initialize(engine, state)
    state.expression_level = start_level
    before_ledger = _ledger(state).model_dump(mode="json")

    result = await _run(
        engine,
        state,
        initial,
        _text_response(initial.turn_id, child_text),
    )

    assert _ledger(result.state).model_dump(mode="json") == before_ledger
    assert result.state.expression_level is expected_level
    assert result.turn.input.kind is expected_input
    assert result.runtime.dialogue_act == "offer_support"
    assert result.runtime.speaker_source == "reviewed_fallback"
    assert result.runtime.new_progress is False
    assert gateway.understanding_requests[0].child_utterance == child_text
    assert gateway.speaker_plans == []
    assert gateway.bridge_plans == []
    assert result.turn.mormi.text == {
        ExpressionLevel.L3: (
            "나 세 명이 똑같이 내면 한 명이 얼마를 내는지 헷갈려... "
            "한 명이 낼 돈만 알려줄 수 있어?"
        ),
        ExpressionLevel.L2: (
            "나 한 명이 얼마를 내는지 아직 헷갈려... 여기에서 골라서 알려줄 수 있어?"
        ),
        ExpressionLevel.L0: (
            "아직 헷갈려서 그런데, 도움 카드를 보면서 나와 같이 계산해 줄 수 있어?"
        ),
    }[expected_level]
    assert result.turn.mormi.text.count("?") == 1
    if expected_level is ExpressionLevel.L0:
        assert result.state.hint_level is HintLevel.H3
        assert result.turn.help_card.visible is True
    else:
        assert result.state.hint_level is HintLevel.H0
        assert result.turn.help_card is None


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
    assert plan.response_plan is None
    assert plan.allowed_facts == []
    assert result.turn.mormi.text == (
        "나는 어떻게 하는 건지 몰라... 도움 카드를 보고 다시 "
        "각자 낼 돈이랑 계산 방법 알려주면 안 될까?"
    )
    assert plan.current_question is None

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
    assert gateway.bridge_plans[0].reaction_mode == "respond_refusal"


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
    assert gateway.bridge_plans[0].reaction_mode == "respond_refusal"


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
    assert plan.response_plan is None
    assert {fact.fact_id for fact in plan.allowed_facts} == {"fact_1"}
    assert all(fact.source == "child_verified" for fact in plan.allowed_facts)
    assert plan.current_question is None
    assert result.turn.mormi.text.count("?") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("support_need", ["none", "expression"])
async def test_normal_partial_progress_always_keeps_server_owned_reask(
    support_need: str,
) -> None:
    """A valid acknowledgement-only model output cannot erase the active question."""

    child_text = "한 사람이 낼 돈은 3500원이야"
    understanding = UnderstandingResponseV2.model_validate(
        {
            "utterance_class": "learning_response",
            "support_need": support_need,
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
    assert gateway.speaker_plans == []
    assert result.runtime.speaker_source == "reviewed_fallback"
    assert result.turn.mormi.text.startswith("아, 3,500원이구나~")
    assert result.turn.mormi.text.count("?") == 1
    assert "나누" not in result.turn.mormi.text


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
    assert gateway.bridge_plans[0].target.fact_ids == []
    assert gateway.bridge_plans[0].target.relation_ids == []


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
