"""FastMCP Server for AI Research Agent - Experiment Queue Management."""

import asyncio
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context
from pydantic import Field

from src.core.config import settings
from src.core.models import ExperimentConfig, ExperimentBatch, ExperimentStatus
from src.experiment.runner import ExperimentRunner
from src.k8s.pod_executor import PodExecutor

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

mcp = FastMCP("AI Research Agent - Experiment Queue")

# Intent patterns for code search (keyword-based, structure-independent)
INTENT_PATTERNS = {
    "optimizer": r"optimizer|Optimizer|OPTIMIZER|optim\.",
    "lr_scheduler": r"scheduler|lr_schedule|learning_rate|LearningRate|LR_|warmup",
    "feature": r"feature|FEATURE|feature_list|FEATURE_LIST|feat_",
    "hyperparameter": r"hidden|dropout|batch_size|HIDDEN|DROPOUT|BATCH|embed_dim|num_layers|num_heads",
    "preprocessing": r"preprocess|transform|normalize|Preprocess|_prep|preprocessing",
    "model_structure": r"class.*Model|class.*Network|def forward|nn\.Module|self\.layers|backbone",
    "loss": r"loss|Loss|criterion|LOSS|bce|cross_entropy|mse",
    "regularization": r"weight_decay|l1_reg|l2_reg|regulariz|WEIGHT_DECAY",
}

# Global state for tracking active batches
_active_runner: Optional[ExperimentRunner] = None
_active_batch: Optional[ExperimentBatch] = None
_batch_task: Optional[asyncio.Task] = None


@mcp.tool()
async def submit_experiments(
    ctx: Context,
    pod_name: Optional[str] = Field(
        None,
        description="Name of the running K8s pod (in tf-box namespace). "
        "REQUIRED: Must ask user if not provided.",
    ),
    utc_ymdh: Optional[str] = Field(
        None,
        description="UTC 기준 학습 데이터 시점. 형식: 'yyyy-mm-dd-hh' (예: '2026-02-06-00'). "
        "학습 시 train(utc_ymdh=...) 인자로 전달됨. "
        "REQUIRED: 사용자에게 반드시 확인 후 사용할 것.",
    ),
    experiments: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of experiments. Each must have 'description' and 'training_command'. "
        "Optional: 'setup_commands' (코드 수정 셸 명령어, 학습 전 실행됨), 'parameters' dict. "
        "매 실험 시작 시 자동으로 git checkout . 후 setup_commands 실행 → git diff main 기록 → 학습. "
        "REQUIRED: Must ask user if not provided.",
    ),
    repo_path: str = Field(
        default="/home/dable/ai-craft",
        description="Path to the git repository on the pod",
    ),
    mlflow_experiment_name: Optional[str] = Field(
        None, description="MLflow experiment name for metrics tracking",
    ),
    stop_on_failure: bool = Field(
        False, description="Stop batch if an experiment fails",
    ),
    poll_interval: int = Field(
        600, description="실험 완료 확인 간격 (초). 기본 600초 (10분).",
    ),
) -> Dict[str, Any]:
    """Submit a batch of experiments to run sequentially on an existing K8s pod.

    IMPORTANT: utc_ymdh는 학습 데이터의 기준 시점으로, 반드시 사용자에게 확인 후 설정해야 합니다.
    형식: 'yyyy-mm-dd-hh' (예: '2026-02-06-00')

    Each experiment flow:
        1. git checkout . (자동 reset)
        2. setup_commands 실행 (코드 수정 - sed, patch 등)
        3. git diff main 캡처 (변경 내역 기록)
        4. training_command 백그라운드 실행 (학습)
        5. poll_interval 간격으로 완료 확인 (기본 10분)
        6. MLflow metrics 수집 & 평가

    Example:
        experiments = [
            {
                "description": "Baseline (all features)",
                "training_command": "cd /home/dable/ai-craft && python -c \\"from ...train import train; train(utc_ymdh='{utc_ymdh}')\\""
            },
            {
                "description": "Ablation: remove content_landing features",
                "setup_commands": "cd /home/dable/ai-craft && sed -i '/content_landing/s/^/            # /' src/dable_ai_craft/.../conf.py",
                "training_command": "cd /home/dable/ai-craft && python -c \\"from ...train import train; train(utc_ymdh='{utc_ymdh}')\\""
            },
        ]
    """
    global _active_runner, _active_batch, _batch_task

    # Check for required parameters and provide clear guidance
    missing_params = []
    if pod_name is None:
        missing_params.append("pod_name")
    if utc_ymdh is None:
        missing_params.append("utc_ymdh")
    if experiments is None or len(experiments) == 0:
        missing_params.append("experiments")

    if missing_params:
        error_messages = []

        if "pod_name" in missing_params:
            error_messages.append(
                "❌ **pod_name** is required.\n"
                "   → Ask user: 'Which pod should I use for training?' (예: 'ai-craft-train-pod')"
            )

        if "utc_ymdh" in missing_params:
            error_messages.append(
                "❌ **utc_ymdh** is required.\n"
                "   → Ask user: 'What training data timestamp should I use?' (형식: 'yyyy-mm-dd-hh', 예: '2026-02-06-00')\n"
                "   → This is the UTC timestamp of the training data. YOU MUST ASK THE USER."
            )

        if "experiments" in missing_params:
            error_messages.append(
                "❌ **experiments** is required.\n"
                "   → Ask user: 'What experiments would you like to run?' (예: baseline, ablation tests, etc.)"
            )

        return {
            "status": "error",
            "error": "Missing required parameters. Please ask the user for the following information:\n\n" + "\n\n".join(error_messages),
            "missing_parameters": missing_params,
        }

    # Validate utc_ymdh format: yyyy-mm-dd-hh
    utc_ymdh_pattern = r'^\d{4}-\d{2}-\d{2}-\d{2}$'
    if not re.match(utc_ymdh_pattern, utc_ymdh):
        return {
            "status": "error",
            "error": f"Invalid utc_ymdh format: '{utc_ymdh}'. "
            "Expected format: 'yyyy-mm-dd-hh' (예: '2026-02-06-00')",
        }

    if _active_runner is not None:
        return {
            "status": "error",
            "error": "A batch is already running. Use stop_batch() first or wait for it to complete.",
        }

    # Create experiment configs
    configs = []
    for i, exp in enumerate(experiments):
        # Replace {utc_ymdh} placeholder in commands
        training_command = exp["training_command"].replace("{utc_ymdh}", utc_ymdh)
        setup_commands = exp.get("setup_commands")
        if setup_commands:
            setup_commands = setup_commands.replace("{utc_ymdh}", utc_ymdh)
        config = ExperimentConfig(
            experiment_id=f"exp-{uuid.uuid4().hex[:8]}",
            description=exp["description"],
            training_command=training_command,
            setup_commands=setup_commands,
            repo_path=repo_path,
            parameters={**exp.get("parameters", {}), "utc_ymdh": utc_ymdh},
        )
        configs.append(config)

    # Create batch
    batch_id = f"batch-{uuid.uuid4().hex[:8]}"
    _active_batch = ExperimentBatch(
        batch_id=batch_id,
        pod_name=pod_name,
        namespace=settings.k8s_namespace,
        experiments=configs,
    )

    # Create runner
    _active_runner = ExperimentRunner(
        pod_name=pod_name,
        namespace=settings.k8s_namespace,
        mlflow_experiment_name=mlflow_experiment_name,
        poll_interval=poll_interval,
    )

    # Verify pod first
    try:
        await _active_runner.verify_pod()
    except RuntimeError as e:
        _active_runner = None
        _active_batch = None
        return {
            "status": "error",
            "error": str(e),
        }

    # Run experiments in background using ensure_future to keep task alive
    async def _run_batch():
        global _active_runner, _active_batch
        try:
            logger.info(f"[BATCH] Starting batch execution: {len(configs)} experiments")
            results = await _active_runner.run_all(configs, stop_on_failure=stop_on_failure)
            if _active_batch:
                _active_batch.results = results
            logger.info(f"[BATCH] Batch execution completed: {len(results)} results")
        except Exception as e:
            logger.error(f"Batch execution error: {e}", exc_info=True)
        finally:
            # Keep results available but mark runner as done
            logger.info("[BATCH] Batch task finished")

    # Use ensure_future instead of create_task to keep strong reference
    _batch_task = asyncio.ensure_future(_run_batch())
    # Wait a bit to ensure task actually starts
    await asyncio.sleep(1.0)
    logger.info(f"[BATCH] Task created and started: {_batch_task}")

    return {
        "status": "submitted",
        "batch_id": batch_id,
        "pod_name": pod_name,
        "utc_ymdh": utc_ymdh,
        "poll_interval": poll_interval,
        "experiment_count": len(configs),
        "experiments": [
            {"experiment_id": c.experiment_id, "description": c.description}
            for c in configs
        ],
        "message": f"Submitted {len(configs)} experiments on pod '{pod_name}' (utc_ymdh={utc_ymdh}, poll_interval={poll_interval}s)",
    }


@mcp.tool()
async def get_queue_status(ctx: Context) -> Dict[str, Any]:
    """Get the current status of the experiment queue.

    Returns information about pending, running, and completed experiments.
    """
    if _active_runner is None or _active_batch is None:
        return {
            "status": "idle",
            "message": "No active batch. Use submit_experiments() to start.",
        }

    queue = _active_runner.queue
    current = queue.current

    return {
        "batch_id": _active_batch.batch_id,
        "pod_name": _active_batch.pod_name,
        "total": queue.total_count,
        "completed": queue.completed_count,
        "pending": queue.pending_count,
        "current_experiment": {
            "experiment_id": current.experiment_id,
            "description": current.description,
        } if current else None,
        "is_running": _batch_task is not None and not _batch_task.done(),
    }


@mcp.tool()
async def get_experiment_results(
    ctx: Context,
    batch_id: Optional[str] = Field(None, description="Batch ID (optional, uses current batch if not specified)"),
) -> Dict[str, Any]:
    """Get results of completed experiments.

    Returns metrics, evaluation results, and a summary for all completed experiments.
    """
    if _active_runner is None:
        return {
            "status": "error",
            "error": "No batch has been run yet.",
        }

    summary = _active_runner.get_summary()

    # Generate markdown report if we have results and batch info
    markdown_report = None
    if _active_batch and summary.get("results"):
        # Extract utc_ymdh from the first experiment's parameters
        utc_ymdh = ""
        if summary["results"] and len(summary["results"]) > 0:
            first_result = summary["results"][0]
            # Try to get utc_ymdh from stored parameters
            for exp in _active_batch.experiments:
                if exp.experiment_id == first_result.get("experiment_id"):
                    utc_ymdh = exp.parameters.get("utc_ymdh", "")
                    break

        # Extract model name from description if possible
        model_name = "model"
        if summary["results"]:
            # Try to infer model name from descriptions
            desc = summary["results"][0].get("description", "")
            if "zigzag" in desc.lower():
                model_name = "zigzag_conv_mtl12"
            elif "model" in desc.lower():
                # Extract model name if mentioned
                model_name = desc.split()[0] if desc else "model"

        markdown_report = _active_runner.get_markdown_report(
            batch_id=_active_batch.batch_id,
            utc_ymdh=utc_ymdh,
            model_name=model_name,
        )

    result = {
        "batch_id": _active_batch.batch_id if _active_batch else "unknown",
        "is_running": _batch_task is not None and not _batch_task.done(),
        **summary,
    }

    if markdown_report:
        result["markdown_report"] = markdown_report

    return result


@mcp.tool()
async def stop_batch(ctx: Context) -> Dict[str, Any]:
    """Stop the currently running batch.

    Cancels any pending experiments. Already completed experiments retain their results.
    """
    global _active_runner, _active_batch, _batch_task

    if _batch_task is None or _batch_task.done():
        # Clean up state
        results = _active_runner.get_summary() if _active_runner else {}
        _active_runner = None
        _active_batch = None
        _batch_task = None
        return {
            "status": "stopped",
            "message": "No active batch to stop (or batch already completed).",
            "results": results,
        }

    _batch_task.cancel()
    try:
        await _batch_task
    except asyncio.CancelledError:
        pass

    results = _active_runner.get_summary() if _active_runner else {}

    _active_runner = None
    _active_batch = None
    _batch_task = None

    return {
        "status": "stopped",
        "message": "Batch stopped. Completed experiments retain their results.",
        "results": results,
    }


def _parse_grep_output(output: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse grep output into structured format.

    Args:
        output: Raw grep output with file:line:content format

    Returns:
        Dictionary mapping filenames to list of matches
    """
    matches: Dict[str, List[Dict[str, Any]]] = {}
    current_file = None
    current_matches = []

    for line in output.split("\n"):
        if not line.strip():
            continue

        # Skip section headers
        if line.startswith("==="):
            continue

        # Parse grep output: filename:line_number:content or filename-line_number-content
        match = re.match(r"^([^:]+):(\d+)[:\-](.*)$", line)
        if match:
            filename = match.group(1).split("/")[-1]  # Just the filename
            line_num = int(match.group(2))
            content = match.group(3)

            if filename not in matches:
                matches[filename] = []

            matches[filename].append({
                "line": line_num,
                "content": content.strip(),
            })

    return matches


def _parse_paths_from_output(output: str) -> Dict[str, str]:
    """Extract paths from script output.

    Args:
        output: Script output containing path information

    Returns:
        Dictionary with conf_path, nn_path, model_dir
    """
    paths = {}
    for line in output.split("\n"):
        if line.startswith("conf_path:"):
            paths["conf_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("nn_path:"):
            paths["nn_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("model_dir:"):
            paths["model_dir"] = line.split(":", 1)[1].strip()
    return paths


@mcp.tool()
async def get_modification_guide(
    ctx: Context,
    pod_name: str = Field(..., description="Name of the running K8s pod"),
    model_name: str = Field(..., description="모델 이름 (예: zigzag_conv_mtl12, xandr_mtl105, korea_mtl86)"),
    intent: str = Field(
        ...,
        description="수정 의도: optimizer, lr_scheduler, feature, hyperparameter, "
        "preprocessing, model_structure, loss, regularization",
    ),
    repo_path: str = Field(
        default="/home/dable/ai-craft",
        description="Path to the git repository on the pod",
    ),
    context_lines: int = Field(default=3, description="매칭 라인 주변 컨텍스트 줄 수"),
) -> Dict[str, Any]:
    """특정 모델에서 수정 의도에 맞는 코드 위치를 동적으로 검색.

    코드베이스 전체를 탐색하지 않고, 모델 관련 파일만 검색하여 토큰 절약.

    구조:
    - whisky_v1/{model}/conf.py + common/nn/{model}/*.py
    - vodka_v3/internal/per_country/{model}/ 또는 external/per_ssp/{model}/
    - vodka_v3/common/nn_v2/{mtl_version}/

    Args:
        pod_name: K8s pod 이름
        model_name: 모델 이름
        intent: 수정 의도 (optimizer, feature, hyperparameter 등)
        repo_path: git 저장소 경로
        context_lines: grep 결과 주변 컨텍스트 줄 수

    Returns:
        모델 경로, 검색된 코드 위치 및 내용
    """
    if intent not in INTENT_PATTERNS:
        return {
            "status": "error",
            "error": f"Unknown intent: {intent}. Valid intents: {list(INTENT_PATTERNS.keys())}",
        }

    pattern = INTENT_PATTERNS[intent]
    base = f"{repo_path}/src/dable_ai_craft/dsp_models"

    # Script to find model paths and search for pattern
    script = f"""
echo "=== PATHS ==="
# Find conf.py for this model
CONF_PATH=$(find {base} -type f -name 'conf.py' -path '*{model_name}*' 2>/dev/null | head -1)

# Find common nn directory for this model
NN_PATH=$(find {base} -type d \\( -path '*common/nn/*' -o -path '*common/nn_v2/*' \\) -name '*{model_name}*' 2>/dev/null | head -1)

# If NN_PATH is empty, try to find by extracting mtl version (for vodka models like mtl86, mtl105)
if [ -z "$NN_PATH" ]; then
    MTL_VERSION=$(echo "{model_name}" | grep -oE 'mtl[0-9]+')
    if [ -n "$MTL_VERSION" ]; then
        NN_PATH=$(find {base} -type d -path '*common/nn_v2/*' -name "*$MTL_VERSION*" 2>/dev/null | head -1)
    fi
fi

MODEL_DIR=$(dirname "$CONF_PATH" 2>/dev/null)

echo "conf_path: $CONF_PATH"
echo "nn_path: $NN_PATH"
echo "model_dir: $MODEL_DIR"

echo ""
echo "=== CONFIG (conf.py) ==="
if [ -f "$CONF_PATH" ]; then
    grep -n -E -i "{pattern}" "$CONF_PATH" -C {context_lines} 2>/dev/null || echo "No matches in conf.py"
else
    echo "conf.py not found"
fi

echo ""
echo "=== NN CODE ==="
if [ -d "$NN_PATH" ] && [ -n "$NN_PATH" ]; then
    grep -rn -E -i "{pattern}" "$NN_PATH"/*.py -C {context_lines} 2>/dev/null | head -100 || echo "No matches in nn path"
elif [ -d "$MODEL_DIR" ] && [ -n "$MODEL_DIR" ]; then
    # vodka external: all code in model directory
    grep -rn -E -i "{pattern}" "$MODEL_DIR"/*.py -C {context_lines} 2>/dev/null | head -100 || echo "No matches in model dir"
else
    echo "No nn path found"
fi
"""

    try:
        pod_executor = PodExecutor()
        await pod_executor.verify_pod_running(pod_name)

        result = await pod_executor.execute_on_pod(
            pod_name=pod_name,
            script=script,
            timeout=60,
        )

        if not result["success"]:
            return {
                "status": "error",
                "error": result["stderr"] or "Script execution failed",
            }

        output = result["stdout"]

        # Parse paths and matches
        paths = _parse_paths_from_output(output)
        matches = _parse_grep_output(output)

        return {
            "status": "success",
            "model": model_name,
            "intent": intent,
            "pattern": pattern,
            "paths": paths,
            "matches": matches,
            "hint": _get_modification_hint(intent),
        }

    except RuntimeError as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in get_modification_guide: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def _get_modification_hint(intent: str) -> str:
    """Get helpful hint for the given intent.

    Args:
        intent: The modification intent

    Returns:
        Helpful hint string
    """
    hints = {
        "optimizer": "conf.py에서 OPTIMIZER, LEARNING_RATE 등을 수정. trainer.py에서 실제 optimizer 생성 로직 확인.",
        "lr_scheduler": "conf.py에서 LR_SCHEDULER, WARMUP_STEPS 등 설정. trainer.py에서 scheduler 생성 로직 확인.",
        "feature": "conf.py에서 FEATURE_LIST 수정. datasetter.py에서 feature 처리 로직 확인.",
        "hyperparameter": "conf.py에서 HIDDEN_UNITS, DROPOUT, BATCH_SIZE 등 수정.",
        "preprocessing": "network.py의 forward() 또는 _preprocess() 메서드 확인.",
        "model_structure": "network.py에서 모델 클래스 정의 확인. layer 추가/제거 시 forward() 수정 필요.",
        "loss": "conf.py에서 LOSS_TYPE 설정. loss.py 또는 trainer.py에서 loss 계산 로직 확인.",
        "regularization": "conf.py에서 WEIGHT_DECAY 등 설정. trainer.py에서 regularization 적용 확인.",
    }
    return hints.get(intent, "")


@mcp.tool()
async def list_models(
    ctx: Context,
    pod_name: str = Field(..., description="Name of the running K8s pod"),
    product: str = Field(
        default="all",
        description="Product filter: whisky_v1, vodka_v3, or all",
    ),
    repo_path: str = Field(
        default="/home/dable/ai-craft",
        description="Path to the git repository on the pod",
    ),
) -> Dict[str, Any]:
    """List available models in the codebase.

    Args:
        pod_name: K8s pod 이름
        product: 제품 필터 (whisky_v1, vodka_v3, all)
        repo_path: git 저장소 경로

    Returns:
        제품별 모델 목록
    """
    base = f"{repo_path}/src/dable_ai_craft/dsp_models"

    script = f"""
echo "=== WHISKY_V1 MODELS ==="
find {base}/whisky_v1 -maxdepth 1 -type d -name '*_*' 2>/dev/null | xargs -I {{}} basename {{}} | sort

echo ""
echo "=== VODKA_V3 INTERNAL MODELS ==="
find {base}/vodka_v3/internal -type d -name '*_*' 2>/dev/null | xargs -I {{}} basename {{}} | sort | uniq

echo ""
echo "=== VODKA_V3 EXTERNAL MODELS ==="
find {base}/vodka_v3/external -type d -name '*_*' 2>/dev/null | xargs -I {{}} basename {{}} | sort | uniq
"""

    try:
        pod_executor = PodExecutor()
        await pod_executor.verify_pod_running(pod_name)

        result = await pod_executor.execute_on_pod(
            pod_name=pod_name,
            script=script,
            timeout=30,
        )

        if not result["success"]:
            return {
                "status": "error",
                "error": result["stderr"] or "Script execution failed",
            }

        output = result["stdout"]
        models: Dict[str, List[str]] = {
            "whisky_v1": [],
            "vodka_v3_internal": [],
            "vodka_v3_external": [],
        }

        current_section = None
        for line in output.split("\n"):
            line = line.strip()
            if "WHISKY_V1" in line:
                current_section = "whisky_v1"
            elif "VODKA_V3 INTERNAL" in line:
                current_section = "vodka_v3_internal"
            elif "VODKA_V3 EXTERNAL" in line:
                current_section = "vodka_v3_external"
            elif line and not line.startswith("===") and current_section:
                models[current_section].append(line)

        # Filter by product if specified
        if product != "all":
            if product == "whisky_v1":
                models = {"whisky_v1": models["whisky_v1"]}
            elif product == "vodka_v3":
                models = {
                    "vodka_v3_internal": models["vodka_v3_internal"],
                    "vodka_v3_external": models["vodka_v3_external"],
                }

        return {
            "status": "success",
            "models": models,
        }

    except RuntimeError as e:
        return {
            "status": "error",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in list_models: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    mcp.run()
