"""Experiment runner using Kubernetes and dable-k8s-runner."""

import asyncio
import logging
from typing import Any, Dict, Optional
import uuid

from src.core.config import settings
from src.core.models import TechSpec, ExperimentResult
from src.k8s.pod_launcher import K8sPodLauncher

logger = logging.getLogger(__name__)


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

    Args:
        implementation: Generated code
        spec: Technical specification

    Returns:
        Bash script to execute experiment
    """
    files = implementation.get("files", {})

    script_lines = [
        "#!/bin/bash",
        "set -e",  # Exit on error
        "",
        "# Create experiment directory",
        f"EXPERIMENT_DIR=/tmp/experiment_{implementation.get('branch_name', 'default')}",
        "mkdir -p $EXPERIMENT_DIR",
        "cd $EXPERIMENT_DIR",
        "",
        "# Write generated files",
    ]

    # Write each generated file
    for filename, content in files.items():
        # Escape special characters for bash
        escaped_content = content.replace("'", "'\"'\"'")
        script_lines.extend([
            f"cat > {filename} << 'EOF'",
            content,
            "EOF",
            "",
        ])

    # Add execution commands based on task type
    if spec.task_type == "MODEL_TRAINING":
        script_lines.extend([
            "# Install dependencies",
            "pip install -r requirements.txt || true",
            "",
            "# Run training",
            "python train.py",
        ])
    elif spec.task_type == "FEATURE_ENGINEERING":
        script_lines.extend([
            "# Install dependencies",
            "pip install -r requirements.txt || true",
            "",
            "# Run feature pipeline",
            "python feature_pipeline.py",
        ])

    return "\n".join(script_lines)


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