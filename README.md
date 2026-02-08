# AI Research Agent - Experiment Queue Manager

실행 중인 K8s Pod에서 ML 모델 학습 실험을 순차적으로 실행하고, MLflow에 결과를 기록하는 MCP 서버입니다.

## Features

- **실험 큐 관리**: 여러 실험을 순차적으로 실행
- **K8s Pod 활용**: 이미 실행 중인 Pod에서 학습 명령어 실행
- **MLflow 연동**: 실험 메트릭 자동 수집 및 태그 기록
- **자동 평가**: AUC, LogLoss, Calibration Error 기반 모델 평가

## Quick Start

### 1. 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2. 환경 설정

```bash
cp .env.example .env
# MLFLOW_TRACKING_URI, K8S_NAMESPACE 설정
```

### 3. MCP 서버 실행

```bash
uv run python -m src.mcp.server
```

## Workflow

1. 사용자가 실행 중인 K8s pod name 제공 (namespace: tf-box)
2. 실험 목록 제공 (각 실험의 명령어 + description)
3. 시스템이 순차적으로 각 실험 실행
4. 각 학습 완료 후 MLflow에서 메트릭 수집 → 평가 → description 기록
5. 전체 결과 요약 제공

## MCP Tools

| Tool | Description |
|------|-------------|
| `submit_experiments` | 실험 배치 제출 (pod_name + experiments[]) |
| `get_queue_status` | 큐 상태 확인 |
| `get_experiment_results` | 결과 조회 |
| `stop_batch` | 진행 중인 배치 중단 |

### 사용 예시

```python
# 실험 제출
await submit_experiments(
    pod_name="my-training-pod",
    experiments=[
        {
            "description": "Baseline model",
            "training_command": "cd /app && python train.py"
        },
        {
            "description": "Higher learning rate",
            "training_command": "cd /app && python train.py --lr 0.01"
        },
    ],
    mlflow_experiment_name="my-experiment"
)

# 상태 확인
await get_queue_status()

# 결과 조회
await get_experiment_results()
```

## Project Structure

```
ai-research-agent/
├── src/
│   ├── core/               # 설정 및 데이터 모델
│   │   ├── config.py
│   │   └── models.py
│   ├── experiment/          # 실험 큐 및 러너
│   │   ├── queue.py
│   │   └── runner.py
│   ├── evaluation/          # 모델 평가
│   │   └── evaluator.py
│   ├── integrations/        # MLflow 클라이언트
│   │   └── mlflow_client.py
│   ├── k8s/                 # K8s Pod 실행
│   │   └── pod_executor.py
│   └── mcp/                 # MCP 서버
│       └── server.py
└── tests/
    ├── unit/
    └── integration/
```

## Evaluation Criteria

| Metric | Threshold | Condition |
|--------|-----------|-----------|
| AUC | 0.85 | > threshold |
| LogLoss | 0.35 | < threshold |
| Calibration Error | 0.02 | < threshold |

## Development

```bash
# 테스트
uv run pytest

# 린트
uv run ruff check .

# 포맷
uv run ruff format .
```

## Links

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
