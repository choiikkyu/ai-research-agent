"""FastMCP Server for AI Research Agent - Experiment Queue Management."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context
from pydantic import Field

from src.core.config import settings
from src.core.models import ExperimentConfig, ExperimentBatch, ExperimentStatus
from src.experiment.runner import ExperimentRunner

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

mcp = FastMCP("AI Research Agent - Experiment Queue")

# Global state for tracking active batches
_active_runner: Optional[ExperimentRunner] = None
_active_batch: Optional[ExperimentBatch] = None
_batch_task: Optional[asyncio.Task] = None


@mcp.tool()
async def submit_experiments(
    ctx: Context,
    pod_name: str = Field(..., description="Name of the running K8s pod (in tf-box namespace)"),
    utc_ymdh: str = Field(
        ...,
        description="UTC 기준 학습 데이터 시점. 형식: 'YYYYMMDDHH' (예: '2026020600'). "
        "학습 시 train(utc_ymdh=...) 인자로 전달됨. 사용자에게 반드시 확인 후 사용할 것.",
    ),
    experiments: List[Dict[str, Any]] = Field(
        ...,
        description="List of experiments. Each must have 'description' and 'training_command'. "
        "Optional: 'setup_commands' (코드 수정 셸 명령어, 학습 전 실행됨), 'parameters' dict. "
        "매 실험 시작 시 자동으로 git checkout . 후 setup_commands 실행 → git diff main 기록 → 학습.",
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

    # Run experiments in background
    async def _run_batch():
        global _active_runner, _active_batch
        try:
            results = await _active_runner.run_all(configs, stop_on_failure=stop_on_failure)
            if _active_batch:
                _active_batch.results = results
        except Exception as e:
            logger.error(f"Batch execution error: {e}")
        finally:
            # Keep results available but mark runner as done
            pass

    _batch_task = asyncio.create_task(_run_batch())

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


if __name__ == "__main__":
    mcp.run()
