from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_image_installs_inference_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert ".[postgres,inference]" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile


def test_deployment_preflights_model_before_replacing_api() -> None:
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

    preflight = workflow.index("scripts/check_ladder_runtime.py")
    replace_api = workflow.index("docker rm -f mormi-ai")
    assert preflight < replace_api
    assert "/opt/mormi/models:/opt/mormi/models:ro" in workflow


def test_deployment_separates_api_and_ladder_worker() -> None:
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "--name mormi-ai" in workflow
    assert "--name mormi-ladder-worker" in workflow
    assert "MORMI_LADDER_ANALYSIS_WORKER_ENABLED=false" in workflow
    assert "MORMI_LADDER_ANALYSIS_WORKER_ENABLED=true" in workflow
    assert "MORMI_LADDER_ANALYSIS_POLL_INTERVAL_SECONDS=2" in workflow
    assert "MORMI_LADDER_ANALYSIS_BATCH_SIZE=1" in workflow
    assert "MORMI_LADDER_ANALYSIS_LEASE_SECONDS=180" in workflow
    assert "python scripts/run_ladder_worker.py" in workflow


def test_deployment_health_checks_candidate_before_removing_live_api() -> None:
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

    candidate = workflow.index("--name mormi-ai-candidate")
    replace_api = workflow.index("docker rm -f mormi-ai ")
    assert candidate < replace_api
    assert "PREVIOUS_IMAGE" in workflow


def test_pull_requests_build_the_production_image() -> None:
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "Docker production image" in workflow
    assert "github.event_name == 'pull_request'" in workflow


def test_worker_entrypoint_does_not_boot_the_http_application() -> None:
    worker_script = (ROOT / "scripts/run_ladder_worker.py").read_text(encoding="utf-8")

    assert "LadderAnalysisWorker" in worker_script
    assert "OutboxDispatcher" not in worker_script
    assert "uvicorn" not in worker_script
