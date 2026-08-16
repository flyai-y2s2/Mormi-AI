from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeGateway
from pydantic import ValidationError

from mormi_api.db import Database
from mormi_api.dictionary_audit import (
    build_dictionary_review_items,
    render_dictionary_human_review,
)
from mormi_api.dictionary_catalog import (
    DICTIONARY_BY_CARD_ID,
    DICTIONARY_CATALOG,
    DictionaryVersionMismatchError,
    get_dictionary_card,
    validate_dictionary_catalog,
    validate_version_manifest,
)
from mormi_api.dictionary_models import DictionaryCard, dictionary_content_hash
from mormi_api.engine import ConversationEngine
from mormi_api.repository import Repository
from mormi_api.schemas import SessionCreate
from mormi_api.security import TextCipher
from mormi_api.service import ConversationService


def test_catalog_covers_36_home_sessions_and_4_current_cafe_stages() -> None:
    validate_dictionary_catalog()
    cards = DICTIONARY_CATALOG.cards
    assert len(cards) == 40
    assert sum(card.card_id.startswith("dictionary.home.") for card in cards) == 36
    assert {
        card.curriculum_session_id
        for card in cards
        if card.card_id.startswith("dictionary.cafe.")
    } == {"cafe_queue", "cafe_budget_menu", "cafe_menu_total", "cafe_change"}


def test_dictionary_copy_is_not_an_alias_for_help_copy() -> None:
    items = build_dictionary_review_items()
    for item in items:
        dictionary_lines = {
            "".join(character for character in line if not character.isspace())
            for line in (*item.concept_lines, *item.example_lines)
        }
        for task in item.related_tasks:
            help_lines = {
                "".join(character for character in line if not character.isspace())
                for line in task.help_plan.values()
            }
            assert dictionary_lines.isdisjoint(help_lines), item.review_id


def test_every_card_has_grounded_visual_and_a_human_review_block() -> None:
    items = build_dictionary_review_items()
    for item in items:
        assert item.related_tasks, item.review_id
        assert set(item.visual["fact_refs"]) <= set(item.facts), item.review_id
        assert item.source_refs, item.review_id

    report = render_dictionary_human_review(items)
    assert report.count("## dictionary:") == 40
    assert "개념·예시·전용 그림" in report
    assert "도움카드와 역할·문구가 분리" in report


def test_card_rejects_an_equation_or_visual_that_disagrees_with_facts() -> None:
    raw = DICTIONARY_CATALOG.cards[2].model_dump(mode="json")
    raw["visual"]["data"]["total"] = 510
    with pytest.raises(ValidationError, match="does not match the example"):
        DictionaryCard.model_validate(raw)

    raw = DICTIONARY_CATALOG.cards[2].model_dump(mode="json")
    raw["example"]["equation"]["result"] = 510
    with pytest.raises(ValidationError, match="does not match its operands"):
        DictionaryCard.model_validate(raw)


def test_expected_version_mismatch_is_explicit() -> None:
    card = get_dictionary_card("money-count")
    assert get_dictionary_card(
        "money-count", expected_content_version=card.content_version
    ) == card
    with pytest.raises(DictionaryVersionMismatchError) as raised:
        get_dictionary_card(
            "money-count", expected_content_version=card.content_version + 1
        )
    assert raised.value.actual == card.content_version


def test_manifest_detects_unaccepted_copy_change(tmp_path: Path) -> None:
    card = DICTIONARY_CATALOG.cards[0]
    changed = card.model_copy(
        update={
            "concept": card.concept.model_copy(
                update={"lines": ["대상을 하나씩 세면 전체 수를 알 수 있어."]}
            )
        }
    )
    catalog = DICTIONARY_CATALOG.model_copy(
        update={"cards": [changed, *DICTIONARY_CATALOG.cards[1:]]}
    )
    manifest = {
        "catalog_version": DICTIONARY_CATALOG.catalog_version,
        "cards": [
            {
                "card_id": current.card_id,
                "content_version": current.content_version,
                "content_hash": dictionary_content_hash(current),
            }
            for current in DICTIONARY_CATALOG.cards
        ],
    }
    manifest_path = tmp_path / "dictionary_versions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="content changed without accepting"):
        validate_version_manifest(catalog, manifest_path=manifest_path)


@pytest.mark.asyncio
async def test_conversation_pins_card_snapshot_and_read_is_state_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'dictionary.db'}")
    await database.create_schema()
    repository = Repository(database, TextCipher("test-encryption-key"))
    service = ConversationService(
        repository,
        ConversationEngine(FakeGateway()),  # type: ignore[arg-type]
    )
    started = await service.create_conversation(
        SessionCreate(
            learner_id=7,
            scene="cafe",
            scenario_id="cafe_queue",
            queue_context={"left_count": 2, "right_count": 5},
        )
    )
    assert started.turn.dictionary_ref is not None
    assert started.turn.dictionary_ref.card_id == "dictionary.cafe.cafe-queue"

    before = await repository.get_state(started.conversation_id)
    original = before.dictionary_snapshots[before.current_task_id]
    changed = original.model_copy(update={"content_version": original.content_version + 1})
    monkeypatch.setitem(DICTIONARY_BY_CARD_ID, original.card_id, changed)

    envelope = await service.dictionary_card(started.conversation_id)
    after = await repository.get_state(started.conversation_id)
    assert envelope.card.content_version == original.content_version
    assert envelope.reference.content_hash == dictionary_content_hash(original)
    assert after.expression_level == before.expression_level
    assert after.hint_level == before.hint_level
    assert after.verified_slots == before.verified_slots
    assert after.state_version == before.state_version
    await database.dispose()
