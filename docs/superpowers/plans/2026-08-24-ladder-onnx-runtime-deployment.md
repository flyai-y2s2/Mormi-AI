# Ladder ONNX Runtime Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the trained KLUE/RoBERTa ladder classifier to a validated INT8 ONNX artifact and run production ladder analysis without PyTorch in an isolated background container.

**Architecture:** Training and export remain local under the existing `analysis` extra. Production installs a new lightweight `inference` extra, tokenizes with `tokenizers`, predicts with `onnxruntime`, and processes DB jobs in a container separate from the public API. Deployment validates the mounted model before replacing containers and exposes bounded operational status without child speech or secrets.

**Tech Stack:** Python 3.12, PyTorch/Transformers for local export only, ONNX, ONNX Runtime dynamic INT8 quantization, Hugging Face Tokenizers, FastAPI, SQLAlchemy, Docker, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-08-24-ladder-onnx-runtime-deployment-design.md`

## Global Constraints

- Production inference must not import or install PyTorch or Transformers model classes.
- The ONNX runtime must preserve the existing `RuntimeBatchResult` and L2/L3/L4 recommendation contract; L0 remains policy-only.
- Maximum token length remains 256.
- INT8 test accuracy may decrease by at most 2 percentage points from the PyTorch baseline.
- Per-label recall may decrease by at most 5 percentage points.
- PyTorch-to-ONNX predicted-label agreement must be at least 98%.
- Child speech, tokens, encryption keys, service keys, and DB URLs must never be logged.
- Model weights and generated ONNX artifacts remain git-ignored.
- Initial DB claim batch is 1, inference micro-batch is 2, and lease is 180 seconds.
- A missing or invalid model must not stop the public dialogue API.

---

### Task 1: Split local export and production inference dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/mormi_api/settings.py`
- Test: `tests/test_ladder_analysis.py`

**Interfaces:**
- Produces: `Settings.ladder_inference_batch_size: int`
- Produces: optional dependency group `inference` containing `numpy`, `onnxruntime`, and `tokenizers`
- Produces: `analysis` dependencies capable of exporting and quantizing ONNX

- [ ] **Step 1: Write the failing settings test**

Add a test that constructs `Settings` with `MORMI_LADDER_INFERENCE_BATCH_SIZE=2`, asserts the parsed value is 2, and asserts values outside 1 through 16 are rejected.

```python
def test_ladder_inference_batch_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORMI_LADDER_INFERENCE_BATCH_SIZE", "2")
    assert Settings().ladder_inference_batch_size == 2
    monkeypatch.setenv("MORMI_LADDER_INFERENCE_BATCH_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings()
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `uv run --extra dev pytest tests/test_ladder_analysis.py -k inference_batch_size -q`

Expected: FAIL because `ladder_inference_batch_size` is not defined.

- [ ] **Step 3: Add the dependencies and setting**

Add this field to `Settings`:

```python
ladder_inference_batch_size: int = Field(default=2, ge=1, le=16)
```

Add an `inference` extra with `numpy>=2.2,<3`, `onnxruntime>=1.22,<2`, and `tokenizers>=0.21,<1`. Add `onnx>=1.18,<2` and `onnxruntime>=1.22,<2` to the local `analysis` extra so export and validation use pinned compatible packages. Refresh `uv.lock` with `uv lock`.

- [ ] **Step 4: Run the focused test and static checks**

Run: `uv run --extra dev pytest tests/test_ladder_analysis.py -k inference_batch_size -q`

Run: `uv run --extra dev ruff check src/mormi_api/settings.py tests/test_ladder_analysis.py`

Expected: PASS.

- [ ] **Step 5: Commit the dependency boundary**

```bash
git add pyproject.toml uv.lock src/mormi_api/settings.py tests/test_ladder_analysis.py
git commit -m "build: add lightweight ladder inference dependencies"
```

### Task 2: Export and validate the trained classifier as INT8 ONNX

**Files:**
- Create: `src/mormi_api/ladder_model/onnx_export.py`
- Create: `scripts/export_ladder_onnx.py`
- Create: `tests/test_ladder_onnx_export.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `export_onnx_model(source_dir: Path, output_dir: Path, evaluation_path: Path) -> ExportReport`
- Produces: `evaluate_onnx_parity(source_dir: Path, onnx_dir: Path, examples: Sequence[LadderExample]) -> ParityReport`
- Produces: `model.int8.onnx` plus tokenizer/config files and schema-version-3 `model-manifest.json`

- [ ] **Step 1: Write manifest and threshold tests with fake metrics**

Test that the exporter rejects label order other than `L2/L3/L4`, rejects accuracy loss greater than `0.02`, rejects any recall loss greater than `0.05`, rejects agreement below `0.98`, and writes SHA-256 values for source and ONNX weights when all thresholds pass.

```python
def test_parity_gate_rejects_accuracy_regression() -> None:
    baseline = Evaluation(accuracy=0.7833, recall={"L2": 1.0, "L3": 0.35, "L4": 1.0})
    candidate = Evaluation(accuracy=0.75, recall={"L2": 1.0, "L3": 0.30, "L4": 0.95})
    with pytest.raises(ValueError, match="accuracy_regression"):
        require_parity(baseline, candidate, agreement=0.99)
```

- [ ] **Step 2: Run the export tests and confirm failure**

Run: `uv run --extra analysis --extra dev pytest tests/test_ladder_onnx_export.py -q`

Expected: FAIL because the export module does not exist.

- [ ] **Step 3: Implement deterministic export and quantization**

Load `AutoTokenizer` and `AutoModelForSequenceClassification` from the source directory with `local_files_only=True`. Export logits with opset 17, dynamic axes for batch and sequence, inputs `input_ids` and `attention_mask`, and `dynamo=False`. Quantize only MatMul/Gemm weights using `quantize_dynamic(..., weight_type=QuantType.QInt8)`. Copy only runtime tokenizer/config files, calculate checksums, evaluate the repository's fixed speech test split, and write the manifest only after all parity gates pass.

The CLI must require explicit paths:

```powershell
uv run --extra analysis python scripts/export_ladder_onnx.py `
  --source artifacts/ladder-model/run-v2/model `
  --source-manifest artifacts/ladder-model/run-v2/model-manifest.json `
  --dataset artifacts/ladder-model/run-v2/dataset/test.jsonl `
  --output artifacts/ladder-model/run-v2-onnx
```

If the recorded test split is stored under a different existing filename, the CLI must accept that exact file through `--dataset`; it must never silently evaluate training examples.

- [ ] **Step 4: Test the export helpers without committing a real model**

Run: `uv run --extra analysis --extra dev pytest tests/test_ladder_onnx_export.py -q`

Run: `uv run --extra dev ruff check src/mormi_api/ladder_model/onnx_export.py scripts/export_ladder_onnx.py tests/test_ladder_onnx_export.py`

Expected: PASS and no generated weights tracked by Git.

- [ ] **Step 5: Commit the export pipeline**

```bash
git add .gitignore src/mormi_api/ladder_model/onnx_export.py scripts/export_ladder_onnx.py tests/test_ladder_onnx_export.py
git commit -m "feat: export validated INT8 ONNX ladder models"
```

### Task 3: Replace the PyTorch production runtime with ONNX Runtime

**Files:**
- Modify: `src/mormi_api/ladder_model/runtime.py`
- Create: `tests/test_ladder_onnx_runtime.py`
- Modify: `tests/test_ladder_analysis.py`

**Interfaces:**
- Consumes: `model.int8.onnx`, `tokenizer.json`, `config.json`, `model-manifest.json`
- Produces: `LadderModelRuntime(model_dir: Path | str, *, inference_batch_size: int = 2)`
- Preserves: `predict(texts: list[str]) -> RuntimeBatchResult`

- [ ] **Step 1: Write failing runtime tests using a tiny ONNX fixture**

Tests must cover missing directory, missing dependency, invalid graph, empty input, micro-batch splitting, label mapping, confidence range, maximum length 256, and input filtering based on `InferenceSession.get_inputs()`.

```python
def test_runtime_splits_speech_into_micro_batches(model_fixture: Path) -> None:
    runtime = LadderModelRuntime(model_fixture, inference_batch_size=2)
    result = runtime.predict(["하나", "둘", "셋", "넷", "다섯"])
    assert result.available is True
    assert len(result.predictions) == 5
    assert runtime.inference_call_count == 3
```

Expose `inference_call_count` only through an injected fake session in tests; do not add production telemetry that contains input content.

- [ ] **Step 2: Run runtime tests and confirm failure**

Run: `uv run --extra inference --extra dev pytest tests/test_ladder_onnx_runtime.py tests/test_ladder_analysis.py -q`

Expected: FAIL because the runtime still imports PyTorch and Transformers.

- [ ] **Step 3: Implement the ONNX runtime**

Use `Tokenizer.from_file`, enable truncation at 256, pad each micro-batch, build `numpy.int64` tensors, supply only graph-declared inputs, and call a CPU `InferenceSession`. Implement numerically stable NumPy softmax and map `config.json` `id2label` values to `LadderLevel`. Preserve existing bounded errors: `MODEL_NOT_FOUND`, `MODEL_DEPENDENCY_MISSING`, `MODEL_LOAD_FAILED`, and `MODEL_INFERENCE_FAILED`.

- [ ] **Step 4: Run runtime, worker, and typing tests**

Run: `uv run --extra inference --extra dev pytest tests/test_ladder_onnx_runtime.py tests/test_ladder_analysis.py tests/test_ladder_analysis_worker.py -q`

Run: `uv run --extra inference --extra dev mypy src/mormi_api/ladder_model/runtime.py src/mormi_api/ladder_analysis_worker.py`

Expected: PASS without importing `torch` or `transformers` in `runtime.py`.

- [ ] **Step 5: Commit the runtime replacement**

```bash
git add src/mormi_api/ladder_model/runtime.py tests/test_ladder_onnx_runtime.py tests/test_ladder_analysis.py
git commit -m "feat: run ladder predictions with ONNX Runtime"
```

### Task 4: Add Worker health, retry, and isolated startup behavior

**Files:**
- Modify: `src/mormi_api/main.py`
- Modify: `src/mormi_api/ladder_analysis_repository.py`
- Modify: `src/mormi_api/ladder_analysis_worker.py`
- Create: `scripts/requeue_ladder_analyses.py`
- Create: `scripts/check_ladder_runtime.py`
- Modify: `tests/test_ladder_analysis_api.py`
- Modify: `tests/test_ladder_analysis_worker.py`

**Interfaces:**
- Produces: internal `GET /v1/internal/ladder-analyses/status`
- Produces: `LadderAnalysisRepository.requeue_model_failures() -> int`
- Produces: `python scripts/check_ladder_runtime.py --model-dir PATH`

- [ ] **Step 1: Write failing status and retry tests**

Assert that status reports enabled, loaded, model version, provider, queue counts, and the most recent bounded error without speech. Assert requeue changes only `MODEL_NOT_FOUND`, `MODEL_DEPENDENCY_MISSING`, and `MODEL_LOAD_FAILED` records from `failed` to `pending`, keeps the same analysis ID, and does not requeue `SPEECH_LOAD_FAILED`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run --extra inference --extra dev pytest tests/test_ladder_analysis_api.py tests/test_ladder_analysis_worker.py -q`

Expected: FAIL because status and requeue operations do not exist.

- [ ] **Step 3: Implement bounded operational status and explicit requeue**

Pass `settings.ladder_inference_batch_size` to `LadderModelRuntime`. Add runtime properties for `loaded`, `provider`, and `model_version`, Worker `last_success_at`, and repository aggregate counts. Protect the status endpoint with existing `InternalReportingAuth`. Implement the requeue script with a `--confirm` flag and print counts only, never record payloads.

- [ ] **Step 4: Add a non-sensitive runtime smoke command**

`check_ladder_runtime.py` must verify required files and checksums, load the ONNX session, predict the fixed literal `"나는 수를 세어서 답을 찾았어."`, and print only `MODEL_OK version=<version> provider=<provider>`.

- [ ] **Step 5: Run focused tests and static checks**

Run: `uv run --extra inference --extra dev pytest tests/test_ladder_analysis_api.py tests/test_ladder_analysis_worker.py -q`

Run: `uv run --extra inference --extra dev ruff check src/mormi_api/main.py src/mormi_api/ladder_analysis_repository.py src/mormi_api/ladder_analysis_worker.py scripts/requeue_ladder_analyses.py scripts/check_ladder_runtime.py`

Expected: PASS.

- [ ] **Step 6: Commit Worker operations**

```bash
git add src/mormi_api/main.py src/mormi_api/ladder_analysis_repository.py src/mormi_api/ladder_analysis_worker.py scripts/requeue_ladder_analyses.py scripts/check_ladder_runtime.py tests/test_ladder_analysis_api.py tests/test_ladder_analysis_worker.py
git commit -m "feat: harden ladder analysis worker operations"
```

### Task 5: Run API and ladder Worker as separate production containers

**Files:**
- Modify: `Dockerfile`
- Modify: `.github/workflows/deploy.yml`
- Create: `tests/test_ladder_deployment_contract.py`

**Interfaces:**
- Consumes: host path `/opt/mormi/models/ladder-v2/model`
- Produces: containers `mormi-ai` and `mormi-ladder-worker`
- Produces: API override `MORMI_LADDER_ANALYSIS_WORKER_ENABLED=false`
- Produces: Worker override `MORMI_LADDER_ANALYSIS_WORKER_ENABLED=true`

- [ ] **Step 1: Write a failing deployment contract test**

Read the Dockerfile and workflow as text and assert that the image installs `.[postgres,inference]`, the API disables the Worker, the Worker mounts `/opt/mormi/models:/opt/mormi/models:ro`, no Worker port is published, the model smoke command runs before container replacement, and both containers have restart and log rotation settings.

- [ ] **Step 2: Run the contract test and confirm failure**

Run: `uv run --extra dev pytest tests/test_ladder_deployment_contract.py -q`

Expected: FAIL because the current image installs only `.[postgres]` and launches one container without a model mount.

- [ ] **Step 3: Update the image and deployment workflow**

Install `.[postgres,inference]` in the runtime image. After pulling the new image, validate the mounted model with a temporary container before removing the current API container. Start `mormi-ai` with Worker disabled and port 8000. Start `mormi-ladder-worker` without a published port, with the read-only model mount, Worker enabled, claim batch 1, inference batch 2, and lease 180. Use the same private Docker network and env file without printing it.

- [ ] **Step 4: Add post-start health checks and safe rollback behavior**

Keep the existing API health loop. For the Worker, use `docker exec mormi-ladder-worker python scripts/check_ladder_runtime.py --model-dir /opt/mormi/models/ladder-v2/model`. If preflight fails, leave the old containers untouched. If Worker start fails after the API update, fail the deployment and print only bounded container logs; do not remove the healthy API.

- [ ] **Step 5: Run contract and Docker build tests**

Run: `uv run --extra dev pytest tests/test_ladder_deployment_contract.py -q`

Run: `docker build -t mormi-ai:ladder-onnx-test .`

Expected: PASS and a successful image build without PyTorch in the installed production dependency set.

- [ ] **Step 6: Commit deployment isolation**

```bash
git add Dockerfile .github/workflows/deploy.yml tests/test_ladder_deployment_contract.py
git commit -m "deploy: isolate the ONNX ladder worker"
```

### Task 6: Convert the actual v2 model and measure the constrained runtime

**Files:**
- Generated, not committed: `artifacts/ladder-model/run-v2-onnx/**`
- Create: `docs/LADDER_ONNX_DEPLOYMENT.md`

**Interfaces:**
- Consumes: `artifacts/ladder-model/run-v2/model` and the preserved v2 test split
- Produces: validated upload directory `artifacts/ladder-model/run-v2-onnx`
- Produces: measured accuracy, agreement, file size, peak RSS, cold latency, and warm latency

- [ ] **Step 1: Locate and verify the real source artifact**

Verify the source manifest is schema 2, model version is `ladder-speech-klue-v2`, label order is `L2/L3/L4`, weight size is 442,505,820 bytes, and SHA-256 matches `3c4c373427701ad496ca177f6af805f9b4f26e2086a4370db8c1017f086d8057`.

- [ ] **Step 2: Export and evaluate the real model**

Run the Task 2 CLI against `artifacts/ladder-model/run-v2/model` and the preserved v2 test split. Expected baseline accuracy is `0.7833333333333333`, baseline macro F1 is `0.757745166550198`, and severe error rate is `0.0`. The command must fail if the dataset file is missing; do not substitute synthetic or training data.

- [ ] **Step 3: Measure production-like memory and latency**

Run the built image with a 1GB memory limit and the artifact mounted read-only. Record cold and ten warm predictions plus peak container memory for inference batches 1 and 2. Repeat with a 2GB limit if 1GB fails. Do not include speech strings in the report.

- [ ] **Step 4: Choose the minimum safe instance from evidence**

Add the measurements to `docs/LADDER_ONNX_DEPLOYMENT.md`. Approve `t3.micro` only if API and Worker measured memory plus 25% headroom is below 1GB; approve `t3.small` only if below 2GB; otherwise require `t3.medium`. Require at least 16GB disk and recommend 20GB.

- [ ] **Step 5: Commit documentation, not weights**

```bash
git add docs/LADDER_ONNX_DEPLOYMENT.md
git commit -m "docs: record ONNX ladder deployment measurements"
```

### Task 7: Full regression and handoff

**Files:**
- Modify: `docs/LADDER_ANALYSIS_OPERATIONS.md`
- Modify: `docs/LADDER_MODEL_LOCAL_TRAINING.md`
- Modify: `README.md`

**Interfaces:**
- Produces: exact EC2 directory, upload, environment, smoke, retry, rollback, and status commands

- [ ] **Step 1: Update operator documentation**

Document the six Worker variables, the model directory structure, read-only mount, status endpoint, explicit failed-job requeue, and the rule that API and Worker containers must not both run the Worker. Include commands that never display secrets.

- [ ] **Step 2: Run the complete test matrix**

Run: `uv run --extra analysis --extra inference --extra dev pytest -q`

Run: `uv run --extra analysis --extra inference --extra dev ruff check .`

Run: `uv run --extra analysis --extra inference --extra dev mypy src`

Expected: all tests and checks pass.

- [ ] **Step 3: Verify generated artifacts stay untracked**

Run: `git status --short`

Expected: no `model.safetensors`, `.onnx`, tokenizer artifact, dataset, or child speech file appears in tracked changes.

- [ ] **Step 4: Perform a local container smoke test**

Run the API and Worker containers on an isolated Docker network with a test DB and the generated model mounted read-only. Enqueue one non-sensitive fixture job, wait for completion, and assert the stored result includes model version `ladder-speech-klue-v2-onnx-int8` and an L2/L3/L4 prediction.

- [ ] **Step 5: Commit final operations documentation**

```bash
git add docs/LADDER_ANALYSIS_OPERATIONS.md docs/LADDER_MODEL_LOCAL_TRAINING.md README.md
git commit -m "docs: document ONNX ladder worker operations"
```

- [ ] **Step 6: Request code review before any production deployment**

Review the complete branch diff against the spec, run the verification commands again after review fixes, and create a PR. Do not merge or alter EC2 until the user approves the PR and the validated model artifact is ready to upload.
