# Ladder PyTorch Runtime Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the validated v2 PyTorch ladder classifier in a production background container without disrupting the dialogue API.

**Architecture:** Add a runtime-only dependency extra and model smoke command, retain the existing lazy `LadderModelRuntime`, split API and Worker containers in deployment, mount the EC2 model read-only, and explicitly requeue only model-configuration failures.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Safetensors, FastAPI, SQLAlchemy, Docker, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-08-24-ladder-pytorch-runtime-deployment-design.md`

## Global Constraints

- Use model version `ladder-speech-klue-v2` with L2/L3/L4 labels and L0 policy-only.
- Never log child speech, tokens, DB URLs, encryption keys, or service keys.
- Install inference dependencies but omit training-only `datasets` and `accelerate` from production.
- Validate the model before replacing the current API container.
- Run exactly one production ladder Worker with claim batch 1 and lease 180 seconds.
- Keep model artifacts git-ignored and mount them read-only.

---

### Task 1: Runtime dependency boundary and model smoke command

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/check_ladder_runtime.py`
- Create: `tests/test_ladder_runtime_smoke.py`

**Interfaces:**
- Produces: optional `inference` extra with NumPy, Safetensors, PyTorch, and Transformers
- Produces: `check_model(model_dir: Path) -> RuntimePrediction`

- [ ] Write tests for missing files, checksum mismatch, successful bounded prediction, and output that contains no input speech.
- [ ] Run `uv run --extra dev pytest tests/test_ladder_runtime_smoke.py -q` and confirm failure because the script is absent.
- [ ] Implement manifest/checksum validation and one fixed non-sensitive prediction through `LadderModelRuntime`.
- [ ] Add and lock the `inference` extra; run focused pytest and Ruff.
- [ ] Commit with `feat: add production ladder model smoke check`.

### Task 2: Explicit retry for model-configuration failures

**Files:**
- Modify: `src/mormi_api/ladder_analysis_repository.py`
- Create: `scripts/requeue_ladder_analyses.py`
- Modify: `tests/test_ladder_analysis_worker.py`

**Interfaces:**
- Produces: `LadderAnalysisRepository.requeue_model_failures() -> int`

- [ ] Write a failing test proving only `MODEL_NOT_FOUND`, `MODEL_DEPENDENCY_MISSING`, and `MODEL_LOAD_FAILED` return to `pending` with the same analysis ID.
- [ ] Implement the transactional repository method and a `--confirm` CLI that prints counts only.
- [ ] Run focused pytest, Ruff, and mypy.
- [ ] Commit with `feat: requeue ladder model failures safely`.

### Task 3: Separate API and Worker production containers

**Files:**
- Modify: `Dockerfile`
- Modify: `.github/workflows/deploy.yml`
- Create: `tests/test_ladder_deployment_contract.py`

**Interfaces:**
- Produces: `mormi-ai` with Worker disabled and port 8000
- Produces: `mormi-ladder-worker` with Worker enabled, no published port, and `/opt/mormi/models:/opt/mormi/models:ro`

- [ ] Write a failing text contract test for the inference extra, preflight smoke before removal, two containers, environment overrides, read-only mount, claim batch 1, and lease 180.
- [ ] Update the Dockerfile to install `.[postgres,inference]`.
- [ ] Update deployment to preflight the mounted model, preserve old containers on preflight failure, then start API and Worker separately.
- [ ] Run contract pytest and build the Docker image.
- [ ] Commit with `deploy: run ladder analysis in an isolated worker`.

### Task 4: Full verification and operations documentation

**Files:**
- Modify: `docs/LADDER_ANALYSIS_OPERATIONS.md`
- Modify: `docs/LADDER_MODEL_LOCAL_TRAINING.md`

- [ ] Document the EC2 model layout, six environment values, upload checksum, smoke, retry, status, and rollback commands without secrets.
- [ ] Run `uv run --extra analysis --extra inference --extra dev pytest -q`.
- [ ] Run Ruff and mypy across `src`, `scripts`, and tests.
- [ ] Run a local Docker smoke with the real ignored `run-v2` model mounted read-only.
- [ ] Verify no model artifact is tracked and commit documentation.
- [ ] Review the branch and create a PR; do not merge until the user approves.
