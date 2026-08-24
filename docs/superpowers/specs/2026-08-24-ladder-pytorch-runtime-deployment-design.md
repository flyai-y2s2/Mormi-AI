# 발화 사다리 PyTorch 운영 배포 설계

## 목표

검증된 `ladder-speech-klue-v2` 모델을 재변환하지 않고 원본 PyTorch 가중치 그대로 AI EC2에서 실행한다. 실시간 대화 API와 분석 Worker를 컨테이너로 분리하고, 모델 파일·체크섬·실제 예측이 확인되기 전에는 새 배포가 기존 API를 중단하지 않게 한다.

## 운영 구조

- `mormi-ai`: 실시간 대화 API. 분석 Worker를 강제로 비활성화한다.
- `mormi-ladder-worker`: 같은 이미지와 DB를 사용하되 분석 Worker만 활성화하고 외부 포트를 공개하지 않는다.
- `/opt/mormi/models/ladder-v2`: EC2 호스트 모델 경로. Worker에만 읽기 전용으로 연결한다.
- 운영 이미지는 `torch`, `transformers`, `safetensors`를 포함하는 별도 `inference` extra를 설치한다. 학습용 `datasets`, `accelerate`는 설치하지 않는다.

초기 설정은 작업 임대 배치 1, 폴링 2초, lease 180초다. 현재 `t3.medium`의 4GB 메모리와 29GB 루트 파일시스템을 기준으로 운영한다.

## 배포 안전장치

새 이미지를 받은 뒤 기존 컨테이너를 제거하기 전에 임시 컨테이너에서 다음을 검사한다.

1. 모델 디렉터리와 manifest가 존재한다.
2. `model.safetensors` SHA-256이 manifest와 일치한다.
3. 토크나이저와 분류 모델을 로컬 파일만으로 로드한다.
4. 고정된 비민감 문장 한 건을 L2/L3/L4 중 하나로 예측한다.

검사 실패 시 기존 API와 기존 Worker를 유지하고 배포를 실패 처리한다. 성공 후 API를 교체하고 Worker를 별도로 시작한다. Worker 시작 실패는 API 컨테이너를 제거하지 않는다.

## 실패 복구와 보안

`MODEL_NOT_FOUND`, `MODEL_DEPENDENCY_MISSING`, `MODEL_LOAD_FAILED`로 실패한 작업만 운영자가 명시적으로 재처리할 수 있다. 분석 ID는 유지하며 발화 원문은 출력하거나 새 테이블에 복제하지 않는다. smoke 검사와 운영 로그는 모델 버전, 예측 단계, 제한된 오류 코드만 출력한다.

## 검증

- 운영 이미지에 추론 의존성이 있고 학습 전용 패키지는 없는지 검사한다.
- 모델 smoke 검사 성공·체크섬 불일치·파일 누락을 테스트한다.
- 실패 작업 재처리 범위를 테스트한다.
- 배포 워크플로의 API/Worker 분리, 모델 마운트, 사전검증 순서를 테스트한다.
- 전체 pytest, Ruff, mypy와 Docker build를 통과한다.
