# 발화 사다리 ONNX Runtime 운영 배포 설계

## 목표

발화 사다리 분류 모델의 학습 환경과 운영 추론 환경을 분리한다. 학습과 ONNX 변환은 로컬에서 PyTorch와 Transformers로 수행하고, 운영 EC2에서는 PyTorch 없이 ONNX Runtime으로만 예측한다. 기존 약 80% 테스트 정확도를 허용 범위 안에서 유지하면서 모델 파일, 컨테이너 이미지, 메모리 사용량을 줄이고 대화 API와 백그라운드 분석을 격리한다.

운영 배포는 모델이나 실행 의존성이 빠진 상태를 성공으로 간주하지 않는다. 모델 로드와 실제 예측 검증이 통과한 뒤에만 분석 Worker를 정상 상태로 표시한다.

## 범위

이번 변경은 다음을 포함한다.

- 저장된 KLUE/RoBERTa 발화 사다리 모델을 ONNX로 내보내는 로컬 스크립트
- 동적 INT8 양자화와 PyTorch 대비 정확도·예측 일치율 검증
- `onnxruntime`, `tokenizers`, `numpy`만 사용하는 운영 추론기
- AI API와 분석 Worker를 분리한 Docker 실행 구성
- EC2 모델 디렉터리의 읽기 전용 마운트와 배포 전후 검증
- 기존 실패 분석 작업의 제한적 재처리
- 모델 상태와 Worker 상태를 노출하는 내부 운영 상태 점검

주간 비교 방식, 리포트 스냅샷 생성, 단원별 AI 문장 개선은 별도 리포트 설계에서 다룬다. 이번 변경은 예측 모델이 운영에서 실제로 실행되고 결과가 저장되는 경로를 확실하게 만드는 데 집중한다.

## 모델 산출물

로컬 변환 결과는 다음 구조를 갖는다.

```text
ladder-v2/
├─ model-manifest.json
└─ model/
   ├─ model.int8.onnx
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   └─ config.json
```

`model-manifest.json`에는 최소한 다음을 기록한다.

- 모델 버전과 생성 시각
- 원본 PyTorch 모델 체크섬
- ONNX 모델 체크섬
- opset과 양자화 방식
- 최대 입력 길이 256
- 라벨 순서 L2/L3/L4
- 기준 테스트 정확도와 ONNX 테스트 정확도
- PyTorch 대비 라벨 일치율

가중치와 실제 모델 산출물은 Git에 넣지 않는다. Git에는 변환·검증 스크립트와 배포 계약만 저장한다.

## 변환과 정확도 기준

변환은 학습 결과 폴더를 입력으로 받아 먼저 FP32 ONNX를 만들고, 그 결과를 동적 INT8로 양자화한다. 운영 후보는 동일한 고정 테스트 세트로 평가한다.

배포 가능한 모델의 기준은 다음과 같다.

- ONNX 입력 토큰이 기존 토크나이저 결과와 동일하다.
- 전체 테스트 정확도 하락은 PyTorch 기준 2%p 이내다.
- 클래스별 재현율 하락은 각 라벨에서 5%p 이내다.
- PyTorch와 ONNX의 최종 라벨 일치율은 98% 이상이다.
- 빈 문자열, 매우 긴 문장, 한글·숫자 혼합 문장에 대해 예측기가 실패하지 않는다.

INT8 모델이 기준을 통과하지 못하면 FP32 ONNX를 운영 후보로 평가한다. FP32도 기준을 통과하지 못하면 기존 PyTorch 모델을 유지하는 것이 아니라 배포를 중단하고 변환 문제를 수정한다.

## 운영 추론기

운영 추론기는 Transformers의 모델 클래스와 PyTorch를 사용하지 않는다.

- `tokenizers`가 `tokenizer.json`을 읽어 학습 때와 동일하게 truncation, padding, attention mask를 만든다.
- ONNX 그래프가 선언한 입력 이름만 전달해 모델별 불필요한 입력 차이를 제거한다.
- `onnxruntime.InferenceSession`은 Worker 프로세스에서 한 번 지연 생성하고 재사용한다.
- 한 분석 작업의 발화가 많아도 추론 micro-batch를 최대 2개 문장으로 나누어 최대 메모리를 제한한다.
- 각 micro-batch의 logits를 softmax로 변환하고 기존 L2/L3/L4 결과 계약을 유지한다.
- 발화 원문과 토큰은 로그나 분석 결과 테이블에 저장하지 않는다.

환경변수 `MORMI_LADDER_ANALYSIS_BATCH_SIZE`는 DB에서 한 번에 임대하는 작업 수다. 모델 추론 micro-batch와 의미를 분리해 혼동하지 않도록 새 설정 `MORMI_LADDER_INFERENCE_BATCH_SIZE`를 추가하고 기본값을 2로 둔다.

## 컨테이너와 EC2 배포

하나의 소스 이미지에서 API용과 Worker용 실행 명령을 나누되, 운영 이미지에는 ONNX 추론에 필요한 런타임 의존성을 포함한다. API 컨테이너에서는 분석 Worker를 강제로 비활성화하고, Worker 컨테이너에서만 활성화한다.

```text
mormi-ai                 실시간 대화 API, 포트 8000
mormi-ladder-worker      발화 사다리 DB 작업 처리, 외부 포트 없음
/opt/mormi/models        두 컨테이너 중 Worker에만 읽기 전용 마운트
운영 PostgreSQL          작업 큐와 분석 결과 공유
```

Worker는 `--restart unless-stopped`로 실행하며 CPU와 메모리 제한을 둔다. 최초 운영 설정은 작업 임대 배치 1, 추론 micro-batch 2, lease 180초다. 측정 결과에 따라 작업 임대 배치만 2까지 올릴 수 있다.

현재 AI EC2에는 `/opt/mormi/models/ladder-v2`가 없으므로 배포 전에 모델을 별도로 업로드해야 한다. 배포 워크플로는 다음을 검사한다.

1. 모델 디렉터리와 필수 파일이 존재한다.
2. 파일 체크섬이 manifest와 일치한다.
3. Worker 이미지에서 ONNX 세션을 생성할 수 있다.
4. 고정된 비민감 smoke 입력 한 건을 예측할 수 있다.
5. Worker 컨테이너가 실행 중이며 DB에 연결할 수 있다.

검사 실패 시 기존 API 컨테이너는 유지하고 Worker만 배포 실패로 처리한다. 모델 문제 때문에 실시간 학습·대화 API를 중단하지 않는다.

## 상태 점검과 실패 복구

운영 상태는 원문이나 비밀값 없이 다음만 반환한다.

- Worker 활성화 여부
- 모델 로드 여부와 모델 버전
- ONNX provider
- 최근 성공 시각
- pending/running/failed 작업 수
- 최근 제한된 오류 코드

`MODEL_NOT_FOUND`, `MODEL_DEPENDENCY_MISSING`, `MODEL_LOAD_FAILED`로 실패한 기존 작업은 운영자가 명시적으로 재처리 명령을 실행할 때만 `pending`으로 되돌린다. 동일 분석 ID를 재사용하며 새 중복 작업을 만들지 않는다. 발화 원문 보존 기간이 끝난 작업은 재처리하지 않고 근거 부족 상태로 남긴다.

Worker가 중단되어 lease가 만료된 `running` 작업은 기존 임대 로직으로 다시 처리한다. 한 작업의 실패가 다음 작업이나 API 컨테이너에 영향을 주지 않는다.

## 인프라 기준

ONNX INT8 전환 후 실제 Linux 컨테이너에서 다음을 측정한다.

- Worker 유휴 및 최초 로드 후 메모리
- 1개와 2개 문장 micro-batch의 최대 RSS
- 최초 예측과 재사용 예측의 지연 시간
- API 컨테이너와 Worker 동시 실행 시 EC2 가용 메모리

현재 `t3.micro`는 가용 메모리가 약 313MB이고 남은 디스크가 약 2.2GB이므로 그대로 배포하지 않는다. 먼저 디스크를 최소 16GB, 권장 20GB로 확장한다. 메모리 제한 환경 테스트 결과 최대 사용량과 API 여유를 합쳐 25% 이상의 안전 여유가 확인될 때만 `t3.micro` 또는 `t3.small`을 사용한다. 기준을 만족하지 못하면 `t3.medium`으로 운영한다.

Swap은 일시적인 로딩 급증의 안전장치일 뿐 정상 메모리 용량을 대신하는 근거로 사용하지 않는다.

## 설정

운영 Worker의 초기 설정은 다음과 같다.

```env
MORMI_LADDER_MODEL_DIR=/opt/mormi/models/ladder-v2/model
MORMI_LADDER_ANALYSIS_WORKER_ENABLED=true
MORMI_LADDER_ANALYSIS_POLL_INTERVAL_SECONDS=2
MORMI_LADDER_ANALYSIS_BATCH_SIZE=1
MORMI_LADDER_ANALYSIS_LEASE_SECONDS=180
MORMI_LADDER_INFERENCE_BATCH_SIZE=2
```

API 컨테이너는 같은 환경 파일을 사용하더라도 실행 인자에서 `MORMI_LADDER_ANALYSIS_WORKER_ENABLED=false`를 덮어써 중복 Worker 실행을 막는다. 비밀번호, DB URL, 암호화 키, 서비스 키는 Git이나 배포 로그에 출력하지 않는다.

## 검증

- 변환 단위 테스트: 필수 산출물, manifest, 체크섬, 라벨 순서
- 토크나이저 동등성 테스트: PyTorch 학습 입력과 운영 입력 토큰 일치
- 예측 동등성 테스트: PyTorch·FP32 ONNX·INT8 ONNX 라벨과 확률 비교
- 데이터셋 평가: 정확도, 클래스별 precision/recall/F1, 혼동행렬
- Runtime 테스트: 빈 입력, 긴 입력, micro-batch 분할, 잘못된 모델 파일
- Worker 통합 테스트: pending 작업 완료, lease 복구, 원문 비로그
- 배포 테스트: 모델 마운트 누락 시 Worker만 실패하고 API는 유지
- 운영 smoke 테스트: 모델 버전 확인, 예측 한 건, 리포트 근거에 완료 결과 노출

모든 기준을 통과한 뒤에만 기존 운영 Worker 환경변수를 활성화한다.
