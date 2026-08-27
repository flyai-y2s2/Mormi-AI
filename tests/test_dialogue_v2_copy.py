from __future__ import annotations

from pathlib import Path

import pytest

from mormi_api.copy_cache import (
    CopyCacheAcquireState,
    GeneratedCopyCacheRepository,
)
from mormi_api.db import Database
from mormi_api.dialogue_v2_content import (
    load_required_home_content_catalog_v2,
    required_home_content_pack_v2,
)
from mormi_api.dialogue_v2_copy import (
    StableCopyOutputV2,
    StableCopyPlanV2,
    StableCopyResolverV2,
    StableCopyWorkItemV2,
    build_stable_copy_work_items_v2,
    compile_stable_copy_plan_set_v2,
    prewarm_required_home_copy_v2,
    stable_copy_plan_set_hash_v2,
    validate_pinned_stable_copy_plan_set_v2,
)
from mormi_api.dialogue_v2_speaker import stable_copy_output_violation_v2
from mormi_api.settings import Settings


class ReviewedCopyGenerator:
    def __init__(
        self,
        fallback_by_slot: dict[str, str],
        *,
        invalid_slot: str | None = None,
        answer_leak_slot: str | None = None,
        raises: bool = False,
    ) -> None:
        self.fallback_by_slot = fallback_by_slot
        self.invalid_slot = invalid_slot
        self.answer_leak_slot = answer_leak_slot
        self.raises = raises
        self.calls: list[str] = []

    async def generate_stable_copy_v2(
        self,
        plan: StableCopyPlanV2,
    ) -> StableCopyOutputV2:
        self.calls.append(plan.copy_slot)
        if self.raises:
            raise RuntimeError("provider detail must not be persisted")
        output = StableCopyResolverV2._fallback_output(
            plan,
            self.fallback_by_slot[plan.copy_slot],
        )
        if plan.copy_slot == self.invalid_slot:
            return output.model_copy(update={"dialogue_act": "wrong_dialogue_act"})
        if plan.copy_slot == self.answer_leak_slot:
            return output.model_copy(
                update={"text": "정답은 우측이야. 다음 방법도 알려줄래?"}
            )
        return output


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "stable_copy_model": "stable-sonnet",
        "stable_copy_effort": "low",
        "stable_copy_timeout_seconds": 2,
        "stable_copy_prompt_version": "stable-prompt-v1",
        "stable_copy_schema_version": "stable-output-v1",
        "stable_copy_validator_version": "stable-validator-v1",
        "stable_copy_cache_lease_seconds": 10,
        "stable_copy_cache_retry_base_seconds": 2,
        "stable_copy_cache_retry_max_seconds": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


async def _runtime(
    tmp_path: Path,
    generator: ReviewedCopyGenerator,
    *,
    settings: Settings | None = None,
) -> tuple[
    Database,
    GeneratedCopyCacheRepository,
    StableCopyResolverV2,
]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path}/v2-copy.db")
    await database.create_schema()
    active_settings = settings or _settings()
    repository = GeneratedCopyCacheRepository(
        database,
        lease_seconds=active_settings.stable_copy_cache_lease_seconds,
        retry_base_seconds=active_settings.stable_copy_cache_retry_base_seconds,
        retry_max_seconds=active_settings.stable_copy_cache_retry_max_seconds,
    )
    return (
        database,
        repository,
        StableCopyResolverV2(repository, generator, active_settings),
    )


def _fallbacks() -> dict[str, str]:
    return {
        item.plan.copy_slot: item.reviewed_fallback
        for item in _typed_work_items()
    }


def _typed_work_items() -> list[StableCopyWorkItemV2]:
    catalog = load_required_home_content_catalog_v2()
    return [
        item
        for pack in catalog.packs
        for item in build_stable_copy_work_items_v2(pack)
    ]


def _legacy_l0_plan(
    pack_id: str,
    purpose: str,
) -> tuple[StableCopyPlanV2, StableCopyWorkItemV2]:
    pack = required_home_content_pack_v2(pack_id)
    item = next(
        item
        for item in build_stable_copy_work_items_v2(pack)
        if item.plan.purpose == purpose
    )
    slot = next(slot for slot in pack.copy_slots if slot.purpose == purpose)
    payload = item.plan.model_dump(mode="json")
    payload.update(
        {
            "target": {
                **item.plan.target.model_dump(mode="json"),
                "fact_ids": [
                    target.target_id
                    for target in slot.targets
                    if target.target_kind == "fact"
                ],
                "relation_ids": [
                    target.target_id
                    for target in slot.targets
                    if target.target_kind == "relation"
                ],
            },
            "visible_facts": [
                {
                    "fact_id": fact.fact_id,
                    "value": fact.value.model_dump(mode="json"),
                    "speaker_text": fact.speaker_label,
                    "source": "screen",
                }
                for fact in pack.reasoning_graph.facts
                if fact.initially_visible
                or fact.fact_id in pack.help_plan.H3.revealed_fact_ids
            ],
            "joint_action": pack.help_plan.H3.action,
            "generation_brief": slot.generation_brief,
            "reveal_policy": "revealed",
        }
    )
    return StableCopyPlanV2.model_validate(payload), item


def test_nine_packs_compile_to_45_guarded_pii_free_copy_plans() -> None:
    items = _typed_work_items()

    assert len(items) == 45
    assert len({item.plan.copy_slot for item in items}) == 45
    for item in items:
        StableCopyResolverV2._fallback_output(item.plan, item.reviewed_fallback)
        payload = item.plan.model_dump(mode="json")
        encoded = str(payload).lower()
        assert "learner_id" not in encoded
        assert "conversation_id" not in encoded
        assert "child_utterance" not in encoded
        assert "output_firewall" not in payload
        assert len(item.pack_hash) == 64


def test_l0_mormi_copy_keeps_h3_card_truth_out_of_plan_and_cache() -> None:
    items = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("number-compare")
    )
    initial_help = next(item for item in items if item.plan.purpose == "initial_help")
    l0_intro = next(item for item in items if item.plan.purpose == "l0_intro")

    assert [value.type for value in initial_help.output_firewall.forbidden_values] == [
        "choice"
    ]
    assert "오른쪽" in initial_help.output_firewall.forbidden_surfaces
    assert initial_help.output_firewall.forbidden_values[0].choice_id == "right"  # type: ignore[union-attr]
    assert [value.type for value in l0_intro.output_firewall.forbidden_values] == [
        "choice"
    ]
    assert "오른쪽" in l0_intro.output_firewall.forbidden_surfaces
    assert l0_intro.plan.visible_facts == []
    assert l0_intro.plan.target.fact_ids == ["fact_1"]
    assert l0_intro.plan.target.relation_ids == ["relation_1"]
    assert l0_intro.plan.joint_action == "follow_visible_joint_ui"
    assert l0_intro.plan.reveal_policy == "hidden"
    assert l0_intro.reviewed_fallback == (
        "도움 카드를 보면서 나와 같이 해 줄 수 있어?"
    )

    generation_payload = l0_intro.plan.model_dump(mode="json")
    encoded = str(generation_payload)
    assert generation_payload["pack_id"] == l0_intro.plan.pack_id
    assert generation_payload["copy_slot"] == l0_intro.plan.copy_slot
    assert "right" not in encoded
    assert "compare_quantities" not in encoded
    assert "오른쪽" not in encoded

    leaked = StableCopyOutputV2(
        text="도움 카드를 보니 오른쪽이 답이구나. 이제 같이 해 보자!",
        mood="thinking",
        dialogue_act=l0_intro.plan.dialogue_act,
        asked_fact_ids=list(l0_intro.plan.target.fact_ids),
        asked_relation_ids=list(l0_intro.plan.target.relation_ids),
    )
    assert (
        stable_copy_output_violation_v2(
            leaked,
            l0_intro.plan,
            forbidden_values=l0_intro.output_firewall.forbidden_values,
            forbidden_surfaces=l0_intro.output_firewall.forbidden_surfaces,
        )
        == "unresolved_answer_surface"
    )


def test_legacy_l0_plans_remain_readable_in_a_pinned_plan_set() -> None:
    pack = required_home_content_pack_v2("number-compare")
    compiled = compile_stable_copy_plan_set_v2(pack)
    pack_hash = build_stable_copy_work_items_v2(pack)[0].pack_hash
    plans = {copy_slot: dict(payload) for copy_slot, payload in compiled.plans.items()}
    for purpose in ("l0_intro", "l0_action"):
        legacy, _ = _legacy_l0_plan("number-compare", purpose)
        plans[legacy.copy_slot] = legacy.model_dump(mode="json")
    # These payloads emulate an in-flight plan set pinned before provenance was
    # added to SpeakerAllowedFactV2. Its stored hash must remain verifiable.
    for payload in plans.values():
        visible_facts = payload["visible_facts"]
        assert isinstance(visible_facts, list)
        for fact in visible_facts:
            assert isinstance(fact, dict)
            fact.pop("source", None)

    plan_set_hash = stable_copy_plan_set_hash_v2(
        plans,
        pack_hash=pack_hash,
        schema_version=compiled.schema_version,
        compiler_version=compiled.compiler_version,
    )
    validated = validate_pinned_stable_copy_plan_set_v2(
        pack,
        pack_hash=pack_hash,
        schema_version=compiled.schema_version,
        compiler_version=compiled.compiler_version,
        plan_set_hash=plan_set_hash,
        plan_payloads=plans,
    )

    assert validated["number-compare.l0.intro"].is_legacy_l0_pinned_plan()
    assert validated["number-compare.l0.action"].is_legacy_l0_pinned_plan()


@pytest.mark.asyncio
async def test_legacy_l0_plan_bypasses_cache_and_generator(tmp_path: Path) -> None:
    legacy, item = _legacy_l0_plan("number-compare", "l0_intro")
    generator = ReviewedCopyGenerator(_fallbacks())
    database, repository, resolver = await _runtime(tmp_path, generator)

    result = await resolver.resolve(
        legacy,
        reviewed_fallback="오른쪽이 답이니까 점을 짝지어 보자",
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )

    assert result.status == "reviewed_fallback"
    assert result.fallback_reason == "LEGACY_L0_MODEL_BYPASSED"
    assert result.text == "도움 카드를 보면서 나와 같이 해 줄 수 있어?"
    assert generator.calls == []
    assert await repository.get_ready(result.full_cache_key) is None
    await database.dispose()


@pytest.mark.asyncio
async def test_resolver_generates_once_then_returns_immutable_ready_hit(
    tmp_path: Path,
) -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("money-count")
    )[1]
    generator = ReviewedCopyGenerator(_fallbacks())
    database, repository, resolver = await _runtime(tmp_path, generator)

    generated = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )
    hit = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )

    assert generated.status == "generated"
    assert hit.status == "hit"
    assert hit.text == generated.text
    assert hit.full_cache_key == generated.full_cache_key
    assert hit.key_digest == generated.full_cache_key[:16]
    assert hit.artifact_sha256 == generated.artifact_sha256
    assert hit.artifact_metadata.pack_hash == item.pack_hash
    assert hit.artifact_metadata.model_id == "stable-sonnet"
    assert hit.artifact_metadata.effort == "low"
    assert hit.artifact_metadata.generation_config["max_tokens"] == 220
    assert generator.calls == [item.plan.copy_slot]
    ready = await repository.get_ready(hit.full_cache_key)
    assert ready is not None
    encoded_artifact = str(ready.artifact)
    assert "output_firewall" not in encoded_artifact
    assert "forbidden_values" not in encoded_artifact
    assert "forbidden_surfaces" not in encoded_artifact
    await database.dispose()


@pytest.mark.asyncio
async def test_pinned_snapshot_wins_even_when_current_generation_settings_change(
    tmp_path: Path,
) -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("number-count")
    )[0]
    generator = ReviewedCopyGenerator(_fallbacks())
    database, repository, resolver = await _runtime(tmp_path, generator)
    original = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )

    changed_resolver = StableCopyResolverV2(
        repository,
        generator,
        _settings(
            stable_copy_model="new-stable-model",
            stable_copy_prompt_version="stable-prompt-v2",
        ),
    )
    pinned = await changed_resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
        pinned_snapshot=original.model_dump(mode="json"),
    )

    assert pinned.status == "pinned"
    assert pinned.text == original.text
    assert pinned.full_cache_key == original.full_cache_key
    assert pinned.artifact_metadata.model_id == "stable-sonnet"
    assert generator.calls == [item.plan.copy_slot]
    await database.dispose()


@pytest.mark.asyncio
async def test_pinned_snapshot_rejected_by_new_firewall_uses_reviewed_fallback(
    tmp_path: Path,
) -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("number-count")
    )[0]
    generator = ReviewedCopyGenerator(_fallbacks())
    database, _repository, resolver = await _runtime(tmp_path, generator)
    original = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )
    child_unsafe_snapshot = original.model_copy(
        update={"text": "정답은 세 개야. 그대로 말해 줘."},
        deep=True,
    )

    recovered = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
        pinned_snapshot=child_unsafe_snapshot,
    )

    assert recovered.status == "reviewed_fallback"
    assert recovered.text == item.reviewed_fallback
    assert recovered.fallback_reason == "PINNED_OUTPUT_GUARD_REJECTED"
    assert generator.calls == [item.plan.copy_slot]
    await database.dispose()


def test_cache_key_changes_with_pack_and_every_generation_setting() -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("money-price")
    )[0]
    generator = ReviewedCopyGenerator(_fallbacks())
    database = Database("sqlite+aiosqlite:///:memory:")
    repository = GeneratedCopyCacheRepository(
        database,
        lease_seconds=10,
        retry_base_seconds=2,
        retry_max_seconds=30,
    )
    base = StableCopyResolverV2(repository, generator, _settings())
    base_key = base._cache_key(item.plan, item.pack_hash)
    variants = [
        _settings(stable_copy_prompt_version="prompt-v2"),
        _settings(stable_copy_schema_version="schema-v2"),
        _settings(stable_copy_validator_version="validator-v2"),
        _settings(stable_copy_model="other-model"),
        _settings(stable_copy_effort="medium"),
        _settings(stable_copy_timeout_seconds=3),
    ]

    assert all(
        StableCopyResolverV2(repository, generator, settings)._cache_key(
            item.plan,
            item.pack_hash,
        )
        != base_key
        for settings in variants
    )
    assert base._cache_key(item.plan, "b" * 64) != base_key


@pytest.mark.asyncio
async def test_guard_rejection_records_backoff_and_returns_reviewed_fallback(
    tmp_path: Path,
) -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("number-compare")
    )[0]
    generator = ReviewedCopyGenerator(
        _fallbacks(),
        invalid_slot=item.plan.copy_slot,
    )
    database, repository, resolver = await _runtime(tmp_path, generator)

    result = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )
    retry = await repository.acquire(result.full_cache_key)

    assert result.status == "generation_fallback"
    assert result.text == item.reviewed_fallback
    assert result.fallback_reason == "OUTPUT_GUARD_REJECTED"
    assert result.artifact_sha256 is None
    assert await repository.get_ready(result.full_cache_key) is None
    assert retry.state is CopyCacheAcquireState.BACKOFF
    await database.dispose()


@pytest.mark.asyncio
async def test_l0_guard_rejection_caches_only_reviewed_joint_fallback(
    tmp_path: Path,
) -> None:
    item = next(
        candidate
        for candidate in build_stable_copy_work_items_v2(
            required_home_content_pack_v2("money-budget")
        )
        if candidate.plan.purpose == "l0_intro"
    )
    generator = ReviewedCopyGenerator(
        _fallbacks(),
        invalid_slot=item.plan.copy_slot,
    )
    database, repository, resolver = await _runtime(tmp_path, generator)

    result = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )
    ready = await repository.get_ready(result.full_cache_key)

    assert result.status == "seeded_reviewed_fallback"
    assert result.text == item.reviewed_fallback
    assert result.fallback_reason is None
    assert ready is not None
    assert ready.artifact["metadata"]["origin"] == "reviewed_fallback"
    assert ready.artifact["output"]["text"] == item.reviewed_fallback
    await database.dispose()


@pytest.mark.asyncio
async def test_busy_lease_uses_fallback_without_calling_generator(
    tmp_path: Path,
) -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("money-budget")
    )[0]
    generator = ReviewedCopyGenerator(_fallbacks())
    database, repository, resolver = await _runtime(tmp_path, generator)
    cache_key = resolver._cache_key(item.plan, item.pack_hash)
    acquired = await repository.acquire(cache_key)
    assert acquired.state is CopyCacheAcquireState.LEASED

    result = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )

    assert result.status == "contended_fallback"
    assert result.fallback_reason == "CACHE_BUSY"
    assert result.text == item.reviewed_fallback
    assert generator.calls == []
    await database.dispose()


@pytest.mark.asyncio
async def test_generator_failure_persists_only_safe_code_and_returns_fallback(
    tmp_path: Path,
) -> None:
    item = build_stable_copy_work_items_v2(
        required_home_content_pack_v2("divide-group")
    )[0]
    generator = ReviewedCopyGenerator(_fallbacks(), raises=True)
    database, repository, resolver = await _runtime(tmp_path, generator)

    result = await resolver.resolve(
        item.plan,
        reviewed_fallback=item.reviewed_fallback,
        pack_hash=item.pack_hash,
        output_firewall=item.output_firewall,
    )

    assert result.status == "generation_fallback"
    assert result.fallback_reason == "GENERATION_FAILED"
    retry = await repository.acquire(result.full_cache_key)
    assert retry.state is CopyCacheAcquireState.BACKOFF
    await database.dispose()


@pytest.mark.asyncio
async def test_prewarm_requires_all_45_slots_to_have_ready_rows(tmp_path: Path) -> None:
    generator = ReviewedCopyGenerator(_fallbacks())
    database, repository, resolver = await _runtime(tmp_path, generator)

    first = await prewarm_required_home_copy_v2(resolver, repository)
    second = await prewarm_required_home_copy_v2(resolver, repository)

    assert first.expected == first.ready == 45
    assert first.succeeded is True
    assert first.failures == ()
    assert second.expected == second.ready == 45
    assert second.succeeded is True
    assert len(generator.calls) == 45
    assert "text" not in first.as_json()
    await database.dispose()


@pytest.mark.asyncio
async def test_prewarm_reports_nonready_slot_for_nonzero_cli_decision(
    tmp_path: Path,
) -> None:
    first_item = _typed_work_items()[0]
    generator = ReviewedCopyGenerator(
        _fallbacks(),
        invalid_slot=first_item.plan.copy_slot,
    )
    database, repository, resolver = await _runtime(tmp_path, generator)

    report = await prewarm_required_home_copy_v2(resolver, repository)

    assert report.expected == 45
    assert report.ready == 44
    assert report.succeeded is False
    assert len(report.failures) == 1
    assert report.failures[0].copy_slot == first_item.plan.copy_slot
    assert report.failures[0].status == "generation_fallback"
    await database.dispose()


@pytest.mark.asyncio
async def test_pack_aware_answer_leak_never_becomes_ready_and_fails_prewarm(
    tmp_path: Path,
) -> None:
    leak_item = next(
        item
        for item in _typed_work_items()
        if item.plan.pack_id == "home.number-compare.v2"
        and item.plan.purpose == "initial_help"
    )
    generator = ReviewedCopyGenerator(
        _fallbacks(),
        answer_leak_slot=leak_item.plan.copy_slot,
    )
    database, repository, resolver = await _runtime(tmp_path, generator)

    report = await prewarm_required_home_copy_v2(resolver, repository)
    leak_key = resolver._cache_key(leak_item.plan, leak_item.pack_hash)

    assert report.expected == 45
    assert report.ready == 44
    assert report.succeeded is False
    assert len(report.failures) == 1
    assert report.failures[0].copy_slot == leak_item.plan.copy_slot
    assert report.failures[0].reason == "OUTPUT_GUARD_REJECTED"
    assert await repository.get_ready(leak_key) is None
    retry = await repository.acquire(leak_key)
    assert retry.state is CopyCacheAcquireState.BACKOFF
    await database.dispose()


@pytest.mark.asyncio
async def test_prewarm_rejects_ready_row_with_wrong_artifact_contract(
    tmp_path: Path,
) -> None:
    first_item = _typed_work_items()[0]
    generator = ReviewedCopyGenerator(_fallbacks())
    database, repository, resolver = await _runtime(tmp_path, generator)
    cache_key = resolver._cache_key(first_item.plan, first_item.pack_hash)
    acquired = await repository.acquire(cache_key)
    assert acquired.lease is not None
    completed = await repository.complete(
        acquired.lease,
        artifact={"artifact_schema_version": "not-the-stable-copy-contract"},
    )
    assert completed is not None

    report = await prewarm_required_home_copy_v2(resolver, repository)

    assert report.expected == 45
    assert report.ready == 44
    assert report.succeeded is False
    assert len(report.failures) == 1
    assert report.failures[0].copy_slot == first_item.plan.copy_slot
    assert report.failures[0].status == "reviewed_fallback"
    assert report.failures[0].reason == "CACHE_ARTIFACT_INVALID"
    await database.dispose()
