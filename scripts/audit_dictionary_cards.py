from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic

from mormi_api.dictionary_audit import (
    OFFLINE_DICTIONARY_AUDIT_SYSTEM,
    DictionaryAuditBatch,
    DictionaryAuditDecision,
    DictionaryReviewItem,
    build_dictionary_review_items,
    dictionary_audit_prompt,
    render_dictionary_human_review,
)
from mormi_api.dictionary_catalog import validate_dictionary_catalog
from mormi_api.llm import structured_output_schema
from mormi_api.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit every registered dictionary card")
    parser.add_argument("--report", type=Path, help="write the human-review Markdown")
    parser.add_argument("--ai", action="store_true", help="run the offline semantic AI audit")
    parser.add_argument("--json", type=Path, help="write AI decisions as JSON")
    parser.add_argument("--batch-size", type=int, default=5)
    return parser.parse_args()


async def audit_with_ai(
    items: list[DictionaryReviewItem],
    *,
    settings: Settings,
    batch_size: int,
) -> list[DictionaryAuditDecision]:
    if not settings.anthropic_api_key:
        raise RuntimeError("MORMI_ANTHROPIC_API_KEY is required for --ai")
    if batch_size < 1 or batch_size > 8:
        raise ValueError("--batch-size must be between 1 and 8")
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    schema = structured_output_schema(DictionaryAuditBatch)
    decisions: list[DictionaryAuditDecision] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        message = await client.messages.create(
            model=settings.classifier_model,
            max_tokens=2_800,
            temperature=0,
            system=OFFLINE_DICTIONARY_AUDIT_SYSTEM,
            messages=[{"role": "user", "content": dictionary_audit_prompt(batch)}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        decisions.extend(DictionaryAuditBatch.model_validate_json(text).results)
    expected_ids = {item.review_id for item in items}
    received_ids = [decision.review_id for decision in decisions]
    if set(received_ids) != expected_ids or len(received_ids) != len(expected_ids):
        raise RuntimeError("offline AI audit did not return every review_id exactly once")
    return decisions


def main() -> int:
    args = parse_args()
    validate_dictionary_catalog()
    items = build_dictionary_review_items()
    report = render_dictionary_human_review(items)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n", encoding="utf-8")
        print(f"human review: {args.report}")
    else:
        print(report)
    if not args.ai:
        print(f"deterministic dictionary contracts passed: {len(items)} cards")
        return 0
    decisions = asyncio.run(
        audit_with_ai(items, settings=Settings(), batch_size=args.batch_size)
    )
    payload = [decision.model_dump(mode="json") for decision in decisions]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"AI audit: {args.json}")
    rejected = [decision for decision in decisions if not decision.approved]
    for decision in rejected:
        print(f"REJECT {decision.review_id}: {'; '.join(decision.issues)}")
    print(
        f"offline AI dictionary audit: "
        f"{len(decisions) - len(rejected)} passed, {len(rejected)} failed"
    )
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
