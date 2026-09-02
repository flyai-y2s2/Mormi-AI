# 발화사다리 예측 모델 v2 로컬 학습

v1의 4분류 결과는 `현재 단계 + 응답 방식`만으로 정답을 100% 복원할 수 있어 모델
성능으로 사용할 수 없습니다. v2는 상호작용 정책과 발화 분석 모델을 분리합니다.

- `잘 모르겠어요`: 현재 단계에서 한 단계 하강하는 규칙
- `choice`: L2 규칙
- `solve_together`: L0 규칙
- 실제 텍스트 응답: 발화 내용만 사용해 L2/L3/L4를 예측하는 모델

## 1. v2 데이터 생성

Mormi-AI 폴더에서 실행합니다. salt는 로컬에서 같은 데이터 분할을 재현할 때 동일한
값을 사용하고 Git이나 메신저에 남기지 않습니다.

```powershell
$env:MORMI_LADDER_HMAC_SALT = "로컬에서만-사용할-16자이상-문자열"
uv run python scripts/prepare_ladder_speech_data.py
```

생성 위치는 `artifacts/ladder-model/dataset-v2/`입니다. 생성기는 다음 조건을 만족하지
않으면 중단합니다.

- train/validation/test 학습자 중복 0
- 문장 템플릿 그룹 중복 0
- 완전히 같은 모델 입력 중복 0
- L0 및 비텍스트 응답 제외

현재 기본 구성은 train 180건, validation 60건, test 60건이며 각 세트에서 L2/L3/L4가
균형을 이룹니다.

## 2. 학습

`analysis` extra를 실행 명령에 직접 지정해야 numpy, torch, transformers가 유지됩니다.

```powershell
uv run --extra analysis python scripts/train_ladder_speech_model.py --epochs 8
```

NVIDIA GPU를 사용할 때:

```powershell
uv run --extra analysis python scripts/train_ladder_speech_model.py --epochs 8 --fp16
```

validation macro F1이 2 epoch 동안 개선되지 않으면 8 epoch 전에 자동 종료합니다.

## 3. 결과

결과는 `artifacts/ladder-model/run-v2/`에 생성됩니다.

- `test-metrics.json`: accuracy, macro F1, 단계 MAE, 위험 오차율
- `confusion-matrix.csv`: L2/L3/L4 혼동행렬
- `model/`: 모델과 tokenizer
- `model-manifest.json`: 라벨·rubric·재현 설정과 모델 해시

`artifacts/ladder-model/run-v1/`은 누수 원인 분석 기록으로만 보존하며 서비스나 발표
성능으로 사용하지 않습니다. v2 역시 합성 발화가 많은 프로토타입이므로 실제 아동에 대한
일반화 성능을 주장할 수 없습니다.

## 4. 분석 작업자에서 사용

학습 결과 중 `run-v2/model/`과 같은 상위 폴더의 `model-manifest.json`을 운영 AI
서버의 영구 디스크에 별도로 배치합니다. 모델 가중치와 학습 데이터는 Git에 커밋하지
않습니다. AI 서버에는 모델 폴더의 절대 경로를 지정합니다.

```powershell
$env:MORMI_LADDER_MODEL_DIR = "D:\mormi-models\ladder-v2\model"
$env:MORMI_LADDER_ANALYSIS_WORKER_ENABLED = "true"
$env:MORMI_LADDER_ANALYSIS_POLL_INTERVAL_SECONDS = "2"
$env:MORMI_LADDER_ANALYSIS_BATCH_SIZE = "1"
$env:MORMI_LADDER_ANALYSIS_LEASE_SECONDS = "180"
```

작업자는 모델을 첫 분석 때 지연 로드합니다. `MORMI_LADDER_MODEL_DIR`이 없거나 모델을
읽지 못하면 해당 작업은 제한된 오류 코드로 실패하고 발화 원문을 오류 메시지나 신규
분석 테이블에 복사하지 않습니다.
