from pathlib import Path


def test_deploy_requeues_model_failures_after_worker_starts() -> None:
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "deploy.yml").read_text(
        encoding="utf-8"
    )

    worker_start = workflow.index("--name mormi-ladder-worker")
    requeue = workflow.index("scripts/requeue_ladder_analyses.py --confirm")
    assert requeue > worker_start
