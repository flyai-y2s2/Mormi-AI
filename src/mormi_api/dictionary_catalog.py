from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .dictionary_models import (
    DictionaryCard,
    DictionaryCardEnvelope,
    DictionaryCatalog,
    dictionary_content_hash,
    dictionary_reference,
)


class DictionaryCardNotFoundError(KeyError):
    pass


class DictionaryVersionMismatchError(ValueError):
    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"dictionary content version mismatch: expected {expected}, current {actual}"
        )


CATALOG_PATH = Path(__file__).with_name("dictionary_catalog.json")
VERSION_MANIFEST_PATH = Path(__file__).with_name("dictionary_versions.json")


def _load_catalog(path: Path = CATALOG_PATH) -> DictionaryCatalog:
    return DictionaryCatalog.model_validate(json.loads(path.read_text(encoding="utf-8")))


DICTIONARY_CATALOG = _load_catalog()
DICTIONARY_BY_CARD_ID = {card.card_id: card for card in DICTIONARY_CATALOG.cards}
DICTIONARY_BY_SESSION_ID = {
    card.curriculum_session_id: card for card in DICTIONARY_CATALOG.cards
}


def get_dictionary_card(
    curriculum_session_id: str,
    *,
    expected_content_version: int | None = None,
) -> DictionaryCard:
    try:
        card = DICTIONARY_BY_SESSION_ID[curriculum_session_id]
    except KeyError as error:
        raise DictionaryCardNotFoundError(curriculum_session_id) from error
    if (
        expected_content_version is not None
        and expected_content_version != card.content_version
    ):
        raise DictionaryVersionMismatchError(
            expected=expected_content_version,
            actual=card.content_version,
        )
    return card


def get_dictionary_card_by_id(card_id: str) -> DictionaryCard:
    try:
        return DICTIONARY_BY_CARD_ID[card_id]
    except KeyError as error:
        raise DictionaryCardNotFoundError(card_id) from error


def dictionary_card_envelope(
    card: DictionaryCard,
    *,
    catalog_version: int | None = None,
) -> DictionaryCardEnvelope:
    return DictionaryCardEnvelope(
        catalog_version=catalog_version or DICTIONARY_CATALOG.catalog_version,
        reference=dictionary_reference(card),
        card=card,
    )


def _normalize_copy(value: str) -> str:
    return re.sub(r"[\s.!?]+", "", value)


def _load_version_manifest(path: Path = VERSION_MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("cards")
    if not isinstance(entries, list):
        raise ValueError("dictionary version manifest needs a cards list")
    manifest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("card_id"), str):
            raise ValueError("dictionary version manifest contains an invalid entry")
        card_id = str(entry["card_id"])
        if card_id in manifest:
            raise ValueError("dictionary version manifest contains duplicate card_id values")
        manifest[card_id] = dict(entry)
    return manifest


def validate_version_manifest(
    catalog: DictionaryCatalog = DICTIONARY_CATALOG,
    *,
    manifest_path: Path = VERSION_MANIFEST_PATH,
) -> None:
    manifest = _load_version_manifest(manifest_path)
    cards = {card.card_id: card for card in catalog.cards}
    if set(manifest) != set(cards):
        missing = sorted(set(cards) - set(manifest))
        orphaned = sorted(set(manifest) - set(cards))
        raise ValueError(
            f"dictionary version manifest coverage mismatch; missing={missing}, "
            f"orphaned={orphaned}"
        )
    for card_id, card in cards.items():
        entry = manifest[card_id]
        if entry.get("content_version") != card.content_version:
            raise ValueError(f"{card_id}: content_version does not match version manifest")
        if entry.get("content_hash") != dictionary_content_hash(card):
            raise ValueError(
                f"{card_id}: content changed without accepting a new version manifest; "
                "bump content_version before replacing reviewed dictionary content"
            )


def validate_dictionary_coverage() -> None:
    """Fail startup and CI when active tasks lack one reviewed dictionary card."""

    # Imported lazily so the versioned catalog remains independent of the
    # dialogue engine and cannot accidentally be generated from help copy.
    from .content import HOME_TEACHING_CATALOG
    from .help_audit import registered_help_tasks
    from .schemas import HintLevel

    home_session_ids = set(HOME_TEACHING_CATALOG)
    catalog_home_ids = {
        card.curriculum_session_id
        for card in DICTIONARY_CATALOG.cards
        if card.card_id.startswith("dictionary.home.")
    }
    if home_session_ids != catalog_home_ids:
        raise ValueError(
            "every home curriculum session needs exactly one reviewed dictionary card"
        )

    used_card_ids: set[str] = set()
    for spec in HOME_TEACHING_CATALOG.values():
        card = get_dictionary_card_by_id(spec.dictionary_card_id)
        if card.curriculum_session_id != spec.id:
            raise ValueError(f"{spec.id}: dictionary_card_id points to another session")

    for registered in registered_help_tasks():
        task = registered.task
        card = get_dictionary_card_by_id(task.dictionary_card_id)
        used_card_ids.add(card.card_id)
        if card.method_policy != task.help_method_policy:
            raise ValueError(
                f"{registered.review_id}: dictionary and task method policies disagree"
            )
        help_copy = {
            _normalize_copy(task.hints[level].body)
            for level in (HintLevel.H1, HintLevel.H2, HintLevel.H3)
        }
        dictionary_copy = {
            _normalize_copy(line)
            for line in (*card.concept.lines, *card.example.lines)
        }
        duplicated = sorted(help_copy & dictionary_copy)
        if duplicated:
            raise ValueError(
                f"{registered.review_id}: dictionary copy must not be reused from help_plan"
            )

    orphaned = set(DICTIONARY_BY_CARD_ID) - used_card_ids
    if orphaned:
        raise ValueError(f"dictionary catalog contains inactive cards: {sorted(orphaned)}")
    validate_version_manifest()


def validate_dictionary_catalog() -> None:
    validate_dictionary_coverage()
