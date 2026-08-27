from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "deploy.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_v2_copy_is_migrated_and_prewarmed_before_candidate_start() -> None:
    workflow = _workflow()

    migration = workflow.index("python scripts/migrate_database.py")
    prewarm = workflow.index("python scripts/prewarm_dialogue_v2_copy.py")
    provider_smoke = workflow.index("python scripts/smoke_dialogue_v2_models.py")
    candidate = workflow.index("--name mormi-ai-candidate")
    live_removal = workflow.index("docker rm -f mormi-ai 2>/dev/null")

    assert migration < prewarm < provider_smoke < candidate < live_removal
    assert "MORMI_STABLE_COPY_MODEL=claude-sonnet-4-6" in workflow
    assert "MORMI_STABLE_COPY_EFFORT=low" in workflow
    assert "MORMI_STABLE_COPY_TIMEOUT_SECONDS=20" in workflow
    assert "for PREWARM_ATTEMPT in 1 2 3" in workflow
    assert 'if [ "${PREWARM_ATTEMPT}" -lt 3 ]' in workflow
    assert "sleep 12" in workflow
    assert "Stable-copy prewarm did not produce all 45 validated ready artifacts" in workflow
    assert "for MODEL_SMOKE_ATTEMPT in 1 2 3" in workflow
    assert 'if [ "${MODEL_SMOKE_ATTEMPT}" -lt 3 ]' in workflow
    assert "Sonnet understanding and Haiku speaker smoke failed all attempts" in workflow
    # A stale host env file must not revive cache artifacts accepted by the
    # weaker pre-firewall validator in prewarm, candidate, live, rollback, or worker.
    assert workflow.count(
        "MORMI_STABLE_COPY_VALIDATOR_VERSION=stable-copy-validator-v2"
    ) == 7


def test_v2_canary_requires_candidate_and_rollback_reader_capabilities() -> None:
    workflow = _workflow()

    capability_check = workflow.index(
        'candidate does not advertise the verdict-v1 reader/runtime capability'
    )
    rollback_check = workflow.index(
        "V2 canary activation requires a snapshot-compatible verdict-v1 rollback image"
    )
    live_removal = workflow.index("docker rm -f mormi-ai 2>/dev/null")

    assert capability_check < rollback_check < live_removal
    assert "previous_supports_v2" in workflow
    assert '"dialogue-v3-snapshot-reader-v1"' in workflow
    assert '"dialogue-v2-snapshot-reader-v1"' not in workflow
    assert (
        "candidate cannot read the pinned home and life conversation snapshots"
        in workflow
    )
    assert "candidate_canary_percent" in workflow
    assert "MORMI_DIALOGUE_V2_CANARY_PERCENT=0" in workflow


def test_candidate_requires_a_configured_llm_even_when_copy_cache_is_warm() -> None:
    workflow = _workflow()

    llm_check = workflow.index("candidate has no configured LLM provider")
    live_removal = workflow.index("docker rm -f mormi-ai 2>/dev/null")

    assert '"llm_configured"[[:space:]]*:[[:space:]]*true' in workflow
    assert llm_check < live_removal


def test_candidate_health_exposes_and_gates_effective_rollout_configuration() -> None:
    workflow = _workflow()

    environment_gate = workflow.index(
        "candidate is not using the production environment profile"
    )
    runtime_gate = workflow.index("candidate effective runtime is not verdict-v1")
    canary_gate = workflow.index(
        "candidate health does not match its effective V2 canary percent"
    )
    live_removal = workflow.index("docker rm -f mormi-ai 2>/dev/null")

    assert environment_gate < runtime_gate < canary_gate < live_removal
    assert '"environment"[[:space:]]*:[[:space:]]*"production"' in workflow
    assert (
        '"runtime_contract_version"[[:space:]]*:[[:space:]]*"verdict-v1"'
        in workflow
    )
    assert "dialogue_v2_canary_percent" in workflow
    # Host-owned env files may predate the production profile setting. Every
    # database/model/runtime container must receive the explicit production
    # boundary; only the model-file preflight runs without application settings.
    assert workflow.count("MORMI_ENVIRONMENT=production") == 9


def test_emergency_canary_zero_path_skips_build_migration_and_prewarm() -> None:
    workflow = _workflow()
    emergency = workflow[
        workflow.index("emergency-disable-v2:") : workflow.index("  test:")
    ]

    assert "inputs.deployment_action == 'disable-v2'" in emergency
    assert "MORMI_DIALOGUE_V2_CANARY_PERCENT=0" in emergency
    assert "CURRENT_IMAGE" in emergency
    assert 'EMERGENCY_RUNTIME_CONTRACT_VERSION="legacy-v1"' in emergency
    assert 'EMERGENCY_RUNTIME_CONTRACT_VERSION="verdict-v1"' in emergency
    assert emergency.count(
        'MORMI_RUNTIME_CONTRACT_VERSION="${EMERGENCY_RUNTIME_CONTRACT_VERSION}"'
    ) == 2
    assert "docker build" not in emergency
    assert "scripts/migrate_database.py" not in emergency
    assert "scripts/prewarm_dialogue_v2_copy.py" not in emergency
    assert "scripts/smoke_dialogue_v2_models.py" not in emergency
    assert "effective_canary" in emergency
    assert "persist_mormi_env" in emergency
    assert "-v /etc/mormi-ai:/mormi-config" in emergency
    assert (
        "persist_mormi_env \\\n"
        "            \"${CURRENT_IMAGE}\" \\\n"
        "            MORMI_DIALOGUE_V2_CANARY_PERCENT \\\n"
        "            0"
    ) in emergency
    assert '>> "${CANARY_ENV_FILE}"' not in emergency


def test_manual_develop_deploy_can_pin_new_assignments_to_zero_percent() -> None:
    workflow = _workflow()

    assert "dialogue_v2_canary_percent:" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "github.ref == 'refs/heads/develop'" in workflow
    assert "DIALOGUE_V2_CANARY_PERCENT_OVERRIDE" in workflow
    assert 'MORMI_DIALOGUE_V2_CANARY_PERCENT=${DIALOGUE_V2_CANARY_PERCENT_OVERRIDE}' in workflow
    # Candidate, live and automatic rollback must all use the rollout-safe
    # effective value (forced zero during expand, requested value after contract).
    assert workflow.count('"${DEPLOY_CANARY_ARGS[@]}"') == 3
    env_persist = workflow.index(
        'PERSISTED_CANARY_PERCENT="0"'
    )
    candidate_gate = workflow.index(
        "candidate cannot read the pinned home and life conversation snapshots"
    )
    live_removal = workflow.index("docker rm -f mormi-ai 2>/dev/null")
    assert candidate_gate < env_persist < live_removal


def test_scenario_identity_uses_two_phase_migration_and_reader_safe_rollback() -> None:
    workflow = _workflow()

    previous_reader = workflow.index(
        'grep -q \'"conversation-scenario-idempotency-reader-v1"\''
    )
    migration = workflow.index("python scripts/migrate_database.py")
    candidate = workflow.index("--name mormi-ai-candidate")
    phase_check = workflow.index(
        "candidate identity schema phase does not match the selected migration phase"
    )
    live_removal = workflow.index("docker rm -f mormi-ai 2>/dev/null")

    assert previous_reader < migration < candidate < phase_check < live_removal
    assert 'MIGRATION_TARGET="20260826_05"' in workflow
    assert 'MIGRATION_TARGET="head"' in workflow
    assert 'MORMI_DATABASE_MIGRATION_TARGET="${MIGRATION_TARGET}"' in workflow
    assert 'EXPECTED_IDENTITY_SCHEMA_PHASE="transition"' in workflow
    assert 'EXPECTED_IDENTITY_SCHEMA_PHASE="final"' in workflow
    assert "transition schema must keep newly created V2 conversations at canary 0" in workflow
    assert 'PERSISTED_CANARY_PERCENT="0"' in workflow


def test_live_api_persists_verdict_runtime_and_rollback_matches_previous_reader() -> None:
    workflow = _workflow()
    candidate_and_live = workflow[workflow.index("--name mormi-ai-candidate") :]

    # Candidate and live start as verdict readers even while assignment stays
    # at zero. The host source of truth is updated only after candidate gates;
    # an older rollback image is explicitly given the runtime it can read.
    assert candidate_and_live.count(
        "MORMI_RUNTIME_CONTRACT_VERSION=verdict-v1"
    ) >= 2
    assert 'ROLLBACK_RUNTIME_CONTRACT_VERSION="legacy-v1"' in workflow
    assert 'ROLLBACK_RUNTIME_CONTRACT_VERSION="verdict-v1"' in workflow
    assert (
        'MORMI_RUNTIME_CONTRACT_VERSION="${ROLLBACK_RUNTIME_CONTRACT_VERSION}"'
        in workflow
    )
    assert workflow.count("persist_mormi_env()") == 2
    assert workflow.count("-v /etc/mormi-ai:/mormi-config") == 2
    assert (
        "persist_mormi_env \\\n"
        "              \"${IMAGE_URI}\" \\\n"
        "              MORMI_DIALOGUE_V2_CANARY_PERCENT \\\n"
        "              \"${PERSISTED_CANARY_PERCENT}\""
    ) in candidate_and_live
    assert (
        "persist_mormi_env \\\n"
        "            \"${IMAGE_URI}\" \\\n"
        "            MORMI_RUNTIME_CONTRACT_VERSION \\\n"
        "            verdict-v1"
    ) in candidate_and_live
    assert '>> "${RUNTIME_ENV_FILE}"' not in candidate_and_live
