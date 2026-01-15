# AI Research Automation Agent

Tech Spec 문서를 입력받아 코드 생성, 실험 실행, PR 관리까지 자동화하는 AI Agent입니다.

## Features

- **코드 자동 생성**: Tech spec 기반 패턴 매칭 코드 생성
- **Draft PR 워크플로우**: PR 생성 후 사용자 승인을 받고 실험 실행
- **실험 자동화**: K8s GPU/CPU Pod 자동 할당 및 실험 실행
- **결과 평가**: 실험 결과 자동 평가 및 리포팅
- **PR 관리**: GitHub PR 생성, 결과에 따른 자동 merge/delete
- **Slack 연동**: Slack을 통한 간편한 인터페이스

## Quick Start

### 1. 설치

```bash
# UV 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

### 2. 환경 설정

```bash
cp .env.example .env
```

필수 환경변수:
- `ANTHROPIC_API_KEY`: Claude API key
- `GITHUB_TOKEN`: GitHub personal access token

### 3. 실행

```bash
# MCP 서버 실행
uv run python -m src.main
```

## Workflow (Model Training)

```
1. analyze_tech_spec     - Notion spec 분석
2. generate_implementation - 코드 생성
3. create_draft_pr       - Draft PR 생성
         ↓
   [사용자 PR 리뷰]
         ↓
4. approve_and_run_experiment - 승인 후 실험 실행
         ↓
   K8s GPU Pod에서 실행:
   python -c "from {module}.train import train; train('{utc_time}')"
         ↓
5. evaluate_experiment   - 결과 평가
6. finalize_pr           - PR 머지/종료
```

## Usage

### MCP Tools

```python
# 1. Draft PR 생성 (실험 대기)
result = await run_full_workflow(request)
# -> experiment_id, pr_url 반환

# 2. 사용자가 PR 리뷰 후 승인
result = await approve_and_run_experiment(experiment_id="abc123")
# -> 실험 실행 및 결과 반환
```

### Slack 명령어

```
@ai_research_auto_agent [Notion URL] 실험해줘

# 옵션 지정
@ai_research_auto_agent
- spec: [Notion URL]
- repo: ai-craft
- gpu: true
```

### 지원 명령어

| 명령어 | 설명 |
|--------|------|
| `help` | 도움말 |
| `status <id>` | 실험 상태 확인 |
| `list` | 대기 중인 실험 목록 |
| `approve <id>` | 실험 승인 및 실행 |
| `cancel <id>` | 실험 취소 |

## Project Structure

```
ai-research-agent/
├── src/
│   ├── mcp/                 # FastMCP 서버
│   │   ├── server.py
│   │   └── tools/
│   │       ├── code_generator.py
│   │       ├── github.py
│   │       └── evaluator.py
│   ├── core/                # 핵심 로직
│   │   ├── config.py
│   │   └── github_code_reference.py
│   └── k8s/                 # K8s 관리
├── k8s/                     # K8s 배포 설정
├── .claude/                 # Claude 참조 문서
│   └── COMMIT_STRATEGY.md
└── pyproject.toml
```

## Development

```bash
# 테스트
uv run pytest

# 린트
uv run ruff check .

# 포맷
uv run ruff format .
```

## Evaluation Criteria

### Model Training
- AUC > 0.85
- LogLoss < 0.35
- Calibration Error < 0.02

### Feature Engineering
- Null Ratio < 10%
- Importance Score > 0.05
- Latency Impact < 10ms

## Links

- [설계 문서 (Notion)](https://www.notion.so/2e85bbc0e5c280cea91aed1898c5f53c)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Contact

- **Team**: AI Team
- **Slack**: #ai-team
