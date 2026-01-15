"""Experiment runner using Kubernetes and dable-k8s-runner.

Workflow:
1. Draft PR created -> Wait for user approval
2. User approves -> Launch K8s pod (GPU for model training)
3. Clone repo, checkout branch
4. Run training command: python -c "from {module_path}.train import train; train('{utc_time}')"
5. Collect metrics and report results
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import uuid

from src.core.config import settings
from src.core.models import TechSpec, ExperimentResult
from src.k8s.pod_launcher import K8sPodLauncher

logger = logging.getLogger(__name__)


def get_utc_time_hours_ago(hours: int = 4) -> str:
    """Get UTC time string from N hours ago.

    Args:
        hours: Number of hours ago (default: 4)

    Returns:
        UTC time string in format 'YYYY-MM-DD HH:00:00'
    """
    utc_now = datetime.now(timezone.utc)
    time_ago = utc_now - timedelta(hours=hours)
    # Round down to hour
    time_ago = time_ago.replace(minute=0, second=0, microsecond=0)
    return time_ago.strftime("%Y-%m-%d %H:%M:%S")


def convert_path_to_module(implementation_path: str) -> str:
    """Convert file path to Python module path.

    Args:
        implementation_path: e.g., "src/dable_ai_craft/dsp_models/vodka_v3/external/per_ssp/xandr_mtltest"

    Returns:
        Module path: e.g., "dable_ai_craft.dsp_models.vodka_v3.external.per_ssp.xandr_mtltest"
    """
    # Remove leading src/ if present
    path = implementation_path
    if path.startswith("src/"):
        path = path[4:]

    # Remove trailing slash
    path = path.rstrip("/")

    # Convert slashes to dots
    module_path = path.replace("/", ".")

    return module_path


async def run_experiment(
    implementation: Dict[str, Any],
    spec: TechSpec,
    gpu_enabled: bool = False
) -> ExperimentResult:
    """
    Run experiment in Kubernetes environment.

    Args:
        implementation: Generated code implementation
        spec: Technical specification
        gpu_enabled: Whether to use GPU

    Returns:
        ExperimentResult with metrics and status
    """
    experiment_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting experiment {experiment_id} for {spec.title}")

    # Initialize pod launcher
    launcher = K8sPodLauncher()

    # Determine resource requirements
    if spec.task_type == "MODEL_TRAINING" or gpu_enabled:
        pod_type = "gpu"
        instance_type = settings.default_gpu_instance
    else:
        pod_type = "cpu"
        instance_type = settings.default_cpu_instance

    pod_name = f"ai-tf-box-exp-{experiment_id}"

    try:
        # Launch pod
        logger.info(f"Launching {pod_type} pod: {pod_name}")
        pod_info = await launcher.launch_pod(
            pod_name=pod_name,
            pod_type=pod_type,
            instance_type=instance_type
        )

        # Prepare experiment script
        experiment_script = prepare_experiment_script(implementation, spec)

        # Execute experiment
        logger.info(f"Executing experiment on pod {pod_name}")
        execution_result = await launcher.execute_on_pod(
            pod_name=pod_name,
            script=experiment_script
        )

        # Collect metrics
        metrics = await collect_metrics(experiment_id, spec.task_type)

        # Create result
        result = ExperimentResult(
            experiment_id=experiment_id,
            status="SUCCESS" if execution_result.get("success") else "FAILURE",
            metrics=metrics,
            pr_url=None,  # Will be set by PR manager
            pod_name=pod_name,
            recommendations=generate_recommendations(metrics, spec.task_type)
        )

        logger.info(f"Experiment {experiment_id} completed with status: {result.status}")
        return result

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {str(e)}")

        # Return failure result
        return ExperimentResult(
            experiment_id=experiment_id,
            status="FAILURE",
            metrics={},
            pr_url=None,
            pod_name=pod_name,
            recommendations=[f"Experiment failed: {str(e)}"]
        )

    finally:
        # Cleanup pod if in development mode
        if settings.is_development:
            logger.info(f"Cleaning up pod {pod_name}")
            await launcher.cleanup_pod(pod_name)


def prepare_experiment_script(
    implementation: Dict[str, Any],
    spec: TechSpec
) -> str:
    """
    Prepare experiment execution script.

    For MODEL_TRAINING:
    1. Clone ai-craft repo and checkout the PR branch
    2. Run training command: python -c "from {module}.train import train; train('{utc_time}')"

    Args:
        implementation: Generated code with branch_name, repository, implementation_path
        spec: Technical specification

    Returns:
        Bash script to execute experiment
    """
    branch_name = implementation.get("branch_name", "main")
    repository = implementation.get("repository", "ai-craft")
    implementation_path = implementation.get("implementation_path", "")

    script_lines = [
        "#!/bin/bash",
        "set -e",  # Exit on error
        "",
        f"echo '=== Starting experiment for {spec.title} ==='",
        f"echo 'Branch: {branch_name}'",
        f"echo 'Repository: {repository}'",
        "",
    ]

    if spec.task_type == "MODEL_TRAINING":
        # Get UTC time 4 hours ago for training
        utc_time = get_utc_time_hours_ago(4)

        # Convert path to module
        module_path = convert_path_to_module(implementation_path)

        # Get repo URL
        repo_url = f"git@github.com:{settings.github_org}/{repository}.git"

        script_lines.extend([
            "# Clone repository and checkout branch",
            f"REPO_DIR=/tmp/{repository}",
            "rm -rf $REPO_DIR",
            f"git clone {repo_url} $REPO_DIR",
            "cd $REPO_DIR",
            f"git checkout {branch_name}",
            "",
            "# Install dependencies",
            "pip install -e . || pip install -r requirements.txt || true",
            "",
            "# Run model training",
            f"echo 'Running training for module: {module_path}'",
            f"echo 'UTC time parameter: {utc_time}'",
            "",
            f'python -c "from {module_path}.train import train; train(\'{utc_time}\')"',
            "",
            "echo '=== Training completed ==='",
        ])

    elif spec.task_type == "FEATURE_ENGINEERING":
        # Feature engineering uses different execution pattern
        script_lines.extend([
            "# Clone repository and checkout branch",
            f"REPO_DIR=/tmp/{repository}",
            "rm -rf $REPO_DIR",
            f"git clone git@github.com:{settings.github_org}/{repository}.git $REPO_DIR",
            "cd $REPO_DIR",
            f"git checkout {branch_name}",
            "",
            "# Install dependencies",
            "pip install -e . || pip install -r requirements.txt || true",
            "",
            "# Run feature pipeline",
            "python -m feature_pipeline",
            "",
            "echo '=== Feature engineering completed ==='",
        ])

    return "\n".join(script_lines)


def generate_training_command(module_path: str, utc_time: str) -> str:
    """Generate the model training command.

    Args:
        module_path: Python module path (e.g., dable_ai_craft.dsp_models.vodka_v3....)
        utc_time: UTC time string for training

    Returns:
        Training command string
    """
    return f'python -c "from {module_path}.train import train; train(\'{utc_time}\')"'


async def collect_metrics(
    experiment_id: str,
    task_type: str
) -> Dict[str, float]:
    """
    Collect metrics from MLflow or other sources.

    Args:
        experiment_id: Experiment identifier
        task_type: Type of task

    Returns:
        Dictionary of metrics
    """
    # TODO: Implement actual MLflow integration
    # For now, return mock metrics

    if task_type == "MODEL_TRAINING":
        return {
            "auc": 0.87,
            "logloss": 0.32,
            "calibration_error": 0.015,
            "training_time_minutes": 45.2,
            "num_parameters": 1500000,
        }
    elif task_type == "FEATURE_ENGINEERING":
        return {
            "null_ratio": 0.05,
            "importance_score": 0.08,
            "latency_ms": 8.5,
            "coverage": 0.95,
            "num_features": 25,
        }
    else:
        return {}


def generate_recommendations(
    metrics: Dict[str, float],
    task_type: str
) -> list[str]:
    """
    Generate recommendations based on metrics.

    Args:
        metrics: Experiment metrics
        task_type: Type of task

    Returns:
        List of recommendations
    """
    recommendations = []

    if task_type == "MODEL_TRAINING":
        auc = metrics.get("auc", 0)
        logloss = metrics.get("logloss", 1)

        if auc < settings.model_auc_threshold:
            recommendations.append(
                f"AUC ({auc:.3f}) is below threshold ({settings.model_auc_threshold}). "
                "Consider: increasing model complexity, adding features, or tuning hyperparameters."
            )

        if logloss > settings.model_logloss_threshold:
            recommendations.append(
                f"LogLoss ({logloss:.3f}) exceeds threshold ({settings.model_logloss_threshold}). "
                "Consider: regularization, dropout, or reducing model complexity."
            )

        if auc > 0.9 and logloss < 0.3:
            recommendations.append(
                "Excellent model performance! Consider deploying to production."
            )

    elif task_type == "FEATURE_ENGINEERING":
        null_ratio = metrics.get("null_ratio", 1)
        importance = metrics.get("importance_score", 0)

        if null_ratio > settings.feature_null_ratio_threshold:
            recommendations.append(
                f"High null ratio ({null_ratio:.2%}) detected. "
                "Consider: data imputation or removing the feature."
            )

        if importance < settings.feature_importance_threshold:
            recommendations.append(
                f"Low feature importance ({importance:.3f}). "
                "Consider: feature transformation or engineering new features."
            )

        if null_ratio < 0.01 and importance > 0.1:
            recommendations.append(
                "High-quality feature! Ready for production use."
            )

    return recommendations if recommendations else ["No specific recommendations."]