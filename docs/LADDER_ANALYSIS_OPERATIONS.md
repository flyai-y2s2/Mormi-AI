# 소단원 발화 사다리 분석 운영 가이드

## 목적과 실행 조건

같은 학습자가 같은 소단원 반복학습을 최근 두 번 연속 완료하면 Spring BE가 완료
트랜잭션 뒤 Mormi-AI에 분석 작업을 등록합니다. 등록은 학습 완료 응답을 막지 않으며,
아동 화면에 알림을 띄우지 않습니다. AI 작업자는 실제 발화를 필요한 순간에만 읽어
모델 추론하고 다음 메타데이터만 분석 작업에 보관합니다.

- 학습자·소단원·두 완료 세션 ID
- 단계별 정답 수와 시도 수
- 최근 예측 단계와 신뢰도
- 추천 단계, 판정 코드, 모델 버전, 승인 상태

발화 원문, 음성, 서비스 키는 `ladder_analysis_jobs`와 애플리케이션 로그에 저장하지
않습니다.

## 운영 환경 변수

### Mormi-AI 서버

```dotenv
MORMI_SERVICE_API_KEY=<BE와 공유하는 긴 임의 문자열>
MORMI_LADDER_MODEL_DIR=/opt/mormi/models/ladder-v2/model
MORMI_LADDER_ANALYSIS_WORKER_ENABLED=true
MORMI_LADDER_ANALYSIS_POLL_INTERVAL_SECONDS=2
MORMI_LADDER_ANALYSIS_BATCH_SIZE=1
MORMI_LADDER_ANALYSIS_LEASE_SECONDS=180
```

`MORMI_LADDER_MODEL_DIR`에는 tokenizer와 분류 모델 파일이 있어야 하며, 바로 위 폴더에
`model-manifest.json`을 함께 둡니다. 모델 파일은 이미지나 Git 저장소에 포함하지 않고
배포 단계에서 영구 디스크로 주입합니다.

운영 배포는 같은 이미지로 두 컨테이너를 실행합니다. `mormi-ai`는 HTTP 요청만 처리하고
작업자를 끄며, `mormi-ladder-worker`만 위 모델 폴더를 읽기 전용으로 연결해 분석을
수행합니다. 새 이미지는 기존 컨테이너를 교체하기 전에 다음 검사로 실제 모델의 해시와
1회 추론을 확인합니다.

```bash
python scripts/check_ladder_runtime.py \
  --model-dir /opt/mormi/models/ladder-v2/model
```

`MODEL_OK`가 출력되지 않으면 배포를 중단하고 기존 컨테이너를 유지합니다. 이미지에는
CPU 전용 PyTorch를 사용하므로 GPU·CUDA 패키지는 필요하지 않습니다.

## 최초 모델 배치

모델 아카이브는 저장소나 Docker 이미지에 넣지 않고 AI EC2에 한 번만 업로드합니다.
현재 검증된 아카이브와 가중치의 SHA-256은 다음과 같습니다.

```text
ladder-v2-pytorch.tar.gz  3e36c36790dbfa37fd5199fd3a85409657a38cf1de52ca5d70e993070df9eb50
model.safetensors         3c4c373427701ad496ca177f6af805f9b4f26e2086a4370db8c1017f086d8057
```

EC2에서 업로드한 파일을 검사하고 영구 경로에 풉니다.

```bash
sha256sum /tmp/ladder-v2-pytorch.tar.gz
sudo install -d -m 755 /opt/mormi/models/ladder-v2
sudo tar -xzf /tmp/ladder-v2-pytorch.tar.gz \
  -C /opt/mormi/models/ladder-v2
sudo sha256sum /opt/mormi/models/ladder-v2/model/model.safetensors
```

두 해시가 위 값과 정확히 같을 때만 배포합니다. 배치가 끝난 뒤에도 컨테이너에는
`/opt/mormi/models`를 읽기 전용으로만 연결합니다.

### Spring BE 서버

```dotenv
MORMI_DIALOGUE_BASE_URL=https://<mormi-ai-host>
MORMI_DIALOGUE_SERVICE_KEY=<AI의 MORMI_SERVICE_API_KEY와 같은 값>
```

분석 등록과 승인도 기존 대화 연동과 같은 서버 간 키를 사용합니다. 이 키는 FE, Vercel
공개 환경 변수, 브라우저 응답에 넣지 않습니다.

## 서버 간 경로

| 호출 | 경로 | 역할 |
|---|---|---|
| BE → AI | `POST /v1/internal/ladder-analyses` | 멱등 분석 작업 등록 |
| BE → AI | `GET /v1/internal/learners/{learner_id}/report-evidence` | 최신 추천 조회 |
| BE → AI | `POST /v1/internal/ladder-analyses/{analysis_id}/approve` | 교사가 확인한 한 단계 변경 적용 |

세 경로 모두 `X-Mormi-Service-Key`가 필요합니다. 추천은 자동 생성되지만
`highest_stable_expression_level`은 승인 전까지 변경하지 않습니다. 승급과 하향은
`L0 < L2 < L3 < L4` 순서에서 한 단계만 적용하며 과거 L1은 L2로 읽습니다.

## 확인 절차

1. AI `/health/authenticated`를 BE와 같은 서비스 키로 호출해 연결과 인증을 확인합니다.
2. 같은 학습자·소단원 반복학습 두 건을 순서대로 완료합니다.
3. AI DB에서 해당 멱등 키의 작업이 `pending → running → completed`로 바뀌는지 봅니다.
4. 교사용 주간 리포트에서 같은 소단원을 선택해 최근 발화, 예측 단계, 현재 단계
   정답률과 추천 상태를 확인합니다.
5. 승급 또는 하향 추천에서 `이 단계로 적용`을 누르고 `적용 완료`를 확인합니다.
6. 다음 해당 소단원 대화가 승인된 시작 단계에서 시작하는지 확인합니다.

컨테이너 상태는 다음처럼 확인합니다. API만 8000 포트를 공개하고 Worker는 포트를
공개하지 않아야 합니다.

```bash
docker ps --filter name=mormi-ai --filter name=mormi-ladder-worker
docker stats --no-stream mormi-ai mormi-ladder-worker
docker logs --tail=100 mormi-ladder-worker
```

## 장애 대응

- `MODEL_NOT_FOUND`: 모델 경로와 볼륨 마운트를 확인합니다.
- `MODEL_DEPENDENCY_MISSING`: AI 배포에 `inference` 의존성이 포함됐는지 확인합니다.
- `MODEL_LOAD_FAILED`: tokenizer·가중치·manifest가 같은 학습 결과인지 확인합니다.
- `MODEL_INFERENCE_FAILED`: 모델 파일 무결성과 실행 메모리를 확인한 뒤 새 분석 작업을
  등록합니다. 원문을 로그에 붙여 넣지 않습니다.
- 등록 API 401: BE의 `MORMI_DIALOGUE_SERVICE_KEY`와 AI의
  `MORMI_SERVICE_API_KEY`가 같은지 확인합니다.
- 등록 API 503: AI에 서비스 키가 설정됐는지 확인합니다.

실행 중 종료된 `running` 작업은 lease 만료 뒤 다시 선점됩니다. 같은 두 세션의 완료
요청을 재전송해도 멱등 키로 한 작업만 유지됩니다.

모델 미설치 시기에 실패한 작업은 모델 배포 후 아래 명령으로 원래 분석 ID를 유지한 채
`pending`으로 되돌립니다. 발화 로드 실패 등 다른 오류는 자동 재처리하지 않습니다.

```bash
python scripts/requeue_ladder_analyses.py --confirm
```

새 Worker가 반복 종료되면 API는 유지한 채 Worker만 중지하고 직전 정상 이미지로 다시
실행합니다. 모델 smoke가 실패한 이미지는 API까지 교체하지 않습니다. GitHub Actions의
직전 성공 배포에 기록된 ECR 이미지 URI를 사용해 재배포한 뒤 `/health`와 Worker 상태를
다시 확인합니다. 롤백 중에도 모델 폴더를 수정하거나 쓰기 가능으로 연결하지 않습니다.
