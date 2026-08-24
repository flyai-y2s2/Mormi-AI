# 발화 사다리 예측 모델 프로토타입 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 운영 데이터의 보존 정책을 지키는 익명화 데이터셋을 만들고 `klue/roberta-base` 순서형 분류 모델을 파인튜닝해 발화별 적정 L4/L3/L2/L0 단계를 예측한다.

**Architecture:** 데이터 추출·검증, 순서형 모델 학습, artifact 평가와 런타임 추론을 서로 독립된 모듈로 둔다. 훈련 의존성은 API 기본 런타임과 분리하고, 모델 입력에는 실제 learner ID나 유형 이름을 넣지 않는다.

**Tech Stack:** Python 3.12, SQLAlchemy asyncio, Pydantic v2, PyTorch, Hugging Face Transformers/Datasets, scikit-learn, pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-08-23-ladder-analysis-agent-design.md`

## Global Constraints

- 활성 단계는 정확히 `L4`, `L3`, `L2`, `L0`이며 `L1`은 읽을 때만 `L2`로 정규화한다.
- 모델 라벨은 사람이 작성한 `target_level`이고 기존 `expression_after`는 정답으로 사용하지 않는다.
- learner 단위로 10/3/3 split하며 같은 learner는 두 split에 등장하지 않는다.
- 원문 추출은 기존 `TextCodec`과 retention policy를 통과해야 한다.
- 실제 ID·원문·접속 정보는 로그와 Git에 남기지 않는다.
- 프로토타입 점수를 실제 아동 일반화 성능으로 표현하지 않는다.

---

### Task 1: 분석 데이터 계약과 순서형 라벨

**Files:**
- Create: `src/mormi_api/ladder_analysis/__init__.py`
- Create: `src/mormi_api/ladder_analysis/contracts.py`
- Test: `tests/test_ladder_analysis_contracts.py`

**Interfaces:**
- Produces: `LadderLevel`, `LadderTrainingExample`, `AnnotatedLadderExample`, `LadderPrediction`, `canonical_ladder_level(value: str) -> LadderLevel`

- [ ] **Step 1: Write the failing contract tests**

```python
def test_legacy_l1_is_canonical_l2() -> None:
    assert canonical_ladder_level("L1") is LadderLevel.L2


def test_annotation_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        AnnotatedLadderExample(**base_example(), target_level="L3", annotation_reason="")


def test_rank_is_ordered_by_support() -> None:
    assert [level.rank for level in LadderLevel] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `python -m pytest tests/test_ladder_analysis_contracts.py -q`

Expected: FAIL because `mormi_api.ladder_analysis.contracts` does not exist.

- [ ] **Step 3: Implement the contracts**

```python
class LadderLevel(StrEnum):
    L0 = "L0"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"

    @property
    def rank(self) -> int:
        return {"L0": 0, "L2": 1, "L3": 2, "L4": 3}[self.value]


def canonical_ladder_level(value: str) -> LadderLevel:
    return LadderLevel.L2 if value.upper() == "L1" else LadderLevel(value.upper())
```

Define `LadderTrainingExample` with the exact input fields from the spec and
`AnnotatedLadderExample` with non-empty `annotation_reason`, `annotator_id`, and
`rubric_version="ladder-label-v1"`.

- [ ] **Step 4: Run focused tests and static checks**

Run: `python -m pytest tests/test_ladder_analysis_contracts.py -q`

Expected: PASS.

Run: `python -m ruff check src/mormi_api/ladder_analysis tests/test_ladder_analysis_contracts.py`

Expected: no violations.

- [ ] **Step 5: Commit**

```bash
git add src/mormi_api/ladder_analysis tests/test_ladder_analysis_contracts.py
git commit -m "feat: define ladder analysis contracts"
```

### Task 2: 읽기 전용 익명화 데이터 추출기

**Files:**
- Create: `src/mormi_api/ladder_analysis/exporter.py`
- Create: `scripts/export_ladder_training_data.py`
- Modify: `.gitignore`
- Test: `tests/test_ladder_dataset_export.py`

**Interfaces:**
- Consumes: `LadderTrainingExample`, existing `Repository.text_codec`
- Produces: `export_ladder_examples(repository, learner_ids, since, until, hmac_salt) -> AsyncIterator[LadderTrainingExample]`

- [ ] **Step 1: Write failing privacy and retention tests**

```python
async def test_export_hashes_learner_and_omits_expired_raw(repository) -> None:
    rows = [row async for row in export_ladder_examples(
        repository,
        learner_ids=[41],
        since=datetime(2026, 8, 17, tzinfo=UTC),
        until=datetime(2026, 8, 24, tzinfo=UTC),
        hmac_salt=b"prototype-test-salt",
    )]
    assert all(row.learner_key != "41" for row in rows)
    assert all("@" not in row.child_utterance for row in rows)
    assert {row.sample_id for row in rows} == {"turn_retained"}
```

Add fixtures for one retained answered turn, one expired raw turn, one safety turn and one system-error turn.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m pytest tests/test_ladder_dataset_export.py -q`

Expected: FAIL because `export_ladder_examples` is missing.

- [ ] **Step 3: Implement read-only extraction**

Join `ConversationRecord`, `TurnRecord`, and `DialogueTurnObservationRecord` by
`conversation_id` and `source_turn_id`. Require an answered turn, a retained raw response,
normal safety, no system error in `runtime_json`, and an allowlisted learner. Decode raw
text only through `repository.text_codec.load()`.

Generate `learner_key` as:

```python
digest = hmac.new(hmac_salt, str(learner_id).encode(), hashlib.sha256).hexdigest()
learner_key = f"anon_{digest[:12]}"
```

Do not select learner names, account rows, email, phone or birth data.

- [ ] **Step 4: Add the CLI and ignored output directory**

The CLI must require all of:

```text
--learner-ids 1,2,3
--since 2026-08-17
--until 2026-08-24
--output artifacts/ladder/raw/examples.jsonl
MORMI_LADDER_EXPORT_SALT in the environment
```

Exit non-zero if the allowlist is empty, the salt is absent, the output exists without
`--overwrite`, or the DB URL is SQLite without `--allow-local-sqlite`.

Add `artifacts/ladder/` to `.gitignore`.

- [ ] **Step 5: Verify the extractor**

Run: `python -m pytest tests/test_ladder_dataset_export.py -q`

Expected: PASS with no raw text in captured logs.

Run: `python -m ruff check src/mormi_api/ladder_analysis/exporter.py scripts/export_ladder_training_data.py tests/test_ladder_dataset_export.py`

Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/mormi_api/ladder_analysis/exporter.py scripts/export_ladder_training_data.py tests/test_ladder_dataset_export.py
git commit -m "feat: export anonymized ladder training data"
```

### Task 3: 라벨 검증과 learner 단위 split

**Files:**
- Create: `src/mormi_api/ladder_analysis/dataset.py`
- Create: `scripts/validate_ladder_annotations.py`
- Create: `scripts/split_ladder_dataset.py`
- Test: `tests/test_ladder_dataset.py`

**Interfaces:**
- Produces: `validate_annotations(rows) -> AnnotationReport`, `split_by_learner(rows, seed=20260823) -> DatasetSplit`

- [ ] **Step 1: Write failing leakage and label tests**

```python
def test_split_never_leaks_a_learner() -> None:
    split = split_by_learner(rows_for_sixteen_learners(), seed=20260823)
    assert split.train.learners.isdisjoint(split.validation.learners)
    assert split.train.learners.isdisjoint(split.test.learners)
    assert split.validation.learners.isdisjoint(split.test.learners)
    assert tuple(map(len, (split.train.learners, split.validation.learners, split.test.learners))) == (10, 3, 3)


def test_validator_rejects_legacy_engine_as_target() -> None:
    report = validate_annotations([row_with_missing_human_annotation()])
    assert report.valid is False
    assert report.issues[0].code == "missing_human_target"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_ladder_dataset.py -q`

Expected: FAIL because dataset functions are absent.

- [ ] **Step 3: Implement validation and split manifest**

Validate unique `sample_id`, four canonical labels, non-empty annotation fields, no actual
learner ID field, at least two skills per learner, and per-label counts. Split shuffled
learner keys into exactly 10/3/3 and write `split-manifest.json` containing seed, learner
keys, source SHA-256 and rubric version.

- [ ] **Step 4: Run tests and validate a fixture dataset**

Run: `python -m pytest tests/test_ladder_dataset.py -q`

Expected: PASS.

Run: `python scripts/validate_ladder_annotations.py --input tests/fixtures/ladder/annotated-valid.jsonl`

Expected: exit 0 and counts only; no utterance text.

- [ ] **Step 5: Commit**

```bash
git add src/mormi_api/ladder_analysis/dataset.py scripts/validate_ladder_annotations.py scripts/split_ladder_dataset.py tests/test_ladder_dataset.py tests/fixtures/ladder
git commit -m "feat: validate and split ladder labels"
```

### Task 4: 순서형 RoBERTa 모델과 학습 명령

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mormi_api/ladder_analysis/modeling.py`
- Create: `src/mormi_api/ladder_analysis/training.py`
- Create: `scripts/train_ladder_model.py`
- Test: `tests/test_ladder_modeling.py`

**Interfaces:**
- Produces: `OrdinalLadderClassifier`, `ordinal_targets(labels)`, `decode_ordinal_logits(logits)`, `train_model(config) -> ArtifactManifest`

- [ ] **Step 1: Add failing ordinal boundary tests**

```python
def test_ordinal_targets_use_three_cumulative_boundaries() -> None:
    assert ordinal_targets(torch.tensor([0, 1, 2, 3])).tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]


def test_decoder_returns_four_level_scores() -> None:
    prediction = decode_ordinal_logits(torch.tensor([[8.0, 6.0, -7.0]]))
    assert prediction.level is LadderLevel.L3
    assert set(prediction.level_scores) == set(LadderLevel)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_ladder_modeling.py -q`

Expected: FAIL because modeling functions are absent.

- [ ] **Step 3: Add isolated ML dependencies**

Add an `analysis` optional dependency group containing `torch>=2.7,<3`,
`transformers>=4.55,<5`, `datasets>=4,<5`, `scikit-learn>=1.7,<2`, and
`safetensors>=0.6,<1`. Do not add these to the default API dependencies. Resolve and commit
the environment lock file used by this repository after the focused tests pass.

- [ ] **Step 4: Implement the ordinal head and training loop**

Use the base encoder pooled CLS representation and `nn.Linear(hidden_size, 3)`. Train the
three ordered boundaries with `binary_cross_entropy_with_logits`; decode the rank by the
number of sigmoid boundary scores greater than `0.5`. Save model weights as safetensors,
the tokenizer, label map, base-model revision, split-manifest hash, rubric version, seed,
and training arguments under one versioned artifact directory.

- [ ] **Step 5: Run focused tests and a one-batch smoke train**

Run: `python -m pytest tests/test_ladder_modeling.py -q`

Expected: PASS.

Run: `python scripts/train_ladder_model.py --train tests/fixtures/ladder/train.jsonl --validation tests/fixtures/ladder/validation.jsonl --base-model tests/fixtures/tiny-roberta --epochs 1 --output artifacts/ladder/test-model --seed 20260823`

Expected: exit 0 and `manifest.json` plus safetensors; logs contain counts and losses only.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/mormi_api/ladder_analysis/modeling.py src/mormi_api/ladder_analysis/training.py scripts/train_ladder_model.py tests/test_ladder_modeling.py tests/fixtures/tiny-roberta
git commit -m "feat: train ordinal ladder classifier"
```

### Task 5: 평가 보고서와 런타임 predictor

**Files:**
- Create: `src/mormi_api/ladder_analysis/evaluation.py`
- Create: `src/mormi_api/ladder_analysis/predictor.py`
- Create: `scripts/evaluate_ladder_model.py`
- Test: `tests/test_ladder_evaluation.py`
- Test: `tests/test_ladder_predictor.py`

**Interfaces:**
- Consumes: versioned artifact from Task 4
- Produces: `evaluate_predictions(expected, predicted) -> LadderMetrics`, `LadderPredictor.predict(example) -> LadderPrediction`

- [ ] **Step 1: Write failing metric and artifact-validation tests**

```python
def test_severe_error_counts_rank_distance_two_or_more() -> None:
    metrics = evaluate_predictions(
        [LadderLevel.L4, LadderLevel.L2],
        [LadderLevel.L0, LadderLevel.L3],
    )
    assert metrics.severe_error_count == 1
    assert metrics.mean_absolute_rank_error == 2.0


def test_predictor_rejects_unknown_rubric(tmp_path) -> None:
    write_manifest(tmp_path, rubric_version="ladder-label-v9")
    with pytest.raises(ModelArtifactError, match="rubric"):
        LadderPredictor.load(tmp_path, expected_rubric="ladder-label-v1")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_ladder_evaluation.py tests/test_ladder_predictor.py -q`

Expected: FAIL because evaluation and predictor modules are absent.

- [ ] **Step 3: Implement metrics and fail-closed loading**

Calculate macro F1, confusion matrix, rank MAE, quadratic weighted kappa and severe error
rate. `LadderPredictor.load` must verify manifest schema, rubric, label order, base revision,
weight hash and tokenizer presence before loading; it must never download a model implicitly
in production mode.

- [ ] **Step 4: Run the full model prototype checks**

Run: `python -m pytest tests/test_ladder_analysis_contracts.py tests/test_ladder_dataset_export.py tests/test_ladder_dataset.py tests/test_ladder_modeling.py tests/test_ladder_evaluation.py tests/test_ladder_predictor.py -q`

Expected: PASS.

Run: `python -m ruff check src/mormi_api/ladder_analysis scripts tests`

Expected: no new violations.

Run: `python -m mypy src/mormi_api/ladder_analysis`

Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/mormi_api/ladder_analysis/evaluation.py src/mormi_api/ladder_analysis/predictor.py scripts/evaluate_ladder_model.py tests/test_ladder_evaluation.py tests/test_ladder_predictor.py
git commit -m "feat: evaluate and load ladder model artifacts"
```

### Task 6: 실제 16명 프로토타입 훈련 체크포인트

**Files:**
- Create locally only: `artifacts/ladder/raw/examples.jsonl`
- Create locally only: `artifacts/ladder/annotated/examples.jsonl`
- Create locally only: `artifacts/ladder/splits/split-manifest.json`
- Create locally only: `artifacts/ladder/models/ladder-klue-v1/`
- Create locally only: `artifacts/ladder/reports/ladder-klue-v1.json`

**Interfaces:**
- Produces: reviewed model artifact consumed by the weekly analysis-agent plan

- [ ] **Step 1: Export the approved 16-learner, seven-day window**

Run the exporter with an explicit learner allowlist and date window. Verify the output count
is between 560 and 896; stop and review collection coverage if it is outside this range.

- [ ] **Step 2: Apply and validate human labels**

Label every row using `ladder-label-v1`; double-review all disagreements and at least 20% of
the remaining rows. Run the validator and record label counts without copying utterances to
the report.

- [ ] **Step 3: Generate the learner split**

Run: `python scripts/split_ladder_dataset.py --input artifacts/ladder/annotated/examples.jsonl --output artifacts/ladder/splits --seed 20260823`

Expected: exactly 10 train, 3 validation and 3 test learner keys with no overlap.

- [ ] **Step 4: Train and evaluate**

Run the ordinal model and the four-class baseline with the same split. Keep the ordinal model
only if it beats the majority baseline in macro F1 and does not increase severe error rate;
otherwise preserve both reports and mark the artifact `prototype_rejected`.

- [ ] **Step 5: Record the checkpoint without committing sensitive artifacts**

Commit only a redacted model card containing counts, split method, label rubric, metrics,
limitations and artifact hash. Do not commit JSONL, weights, learner keys or raw examples.
