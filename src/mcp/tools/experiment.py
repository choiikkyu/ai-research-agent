"""Experiment runner using Kubernetes and dable-k8s-runner.

Training workflows are dynamically parsed from ai-craft's Airflow DAG files:
  - Location: ai-craft/workflow/airflow_dags/

This ensures the agent always uses the same workflow as production.

Workflow:
1. Draft PR created -> Wait for user approval
2. User approves -> Launch K8s pod (GPU for model training)
3. Clone repo, checkout branch
4. Parse DAG file to determine training workflow
5. Execute training workflow
6. Collect metrics and report results
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import uuid

from src.core.config import settings
from src.core.models import TechSpec, ExperimentResult, ExperimentSession, SessionStatus
from src.k8s.pod_launcher import K8sPodLauncher
from src.mcp.tools.workflow_parser import (
    WorkflowParser,
    WorkflowStep,
    TrainingWorkflow,
    generate_training_script_from_workflow,
)

logger = logging.getLogger(__name__)


# ===== Session Management (In-Memory) =====

_active_sessions: Dict[str, ExperimentSession] = {}


def get_utc_time_hours_ago(hours: int = 4) -> str:
    """Get UTC time string from N hours ago.

    Args:
        hours: Number of hours ago (default: 4)

    Returns:
        UTC time string in format 'YYYY-MM-DD-HH' (e.g., '2026-01-15-02')
    """
    utc_now = datetime.now(timezone.utc)
    time_ago = utc_now - timedelta(hours=hours)
    # Round down to hour
    time_ago = time_ago.replace(minute=0, second=0, microsecond=0)
    return time_ago.strftime("%Y-%m-%d-%H")


def get_utc_time_iso(hours_ago: int = 4) -> str:
    """Get UTC time in ISO format (for dataset creation).

    Args:
        hours_ago: Number of hours ago (default: 4)

    Returns:
        UTC time string in ISO format (e.g., '2026-01-15T02:00:00')
    """
    utc_now = datetime.now(timezone.utc)
    time_ago = utc_now - timedelta(hours=hours_ago)
    time_ago = time_ago.replace(minute=0, second=0, microsecond=0)
    return time_ago.strftime("%Y-%m-%dT%H:%M:%S")


def get_utc_time_iso_days_ago(days: int = 90) -> str:
    """Get UTC time in ISO format from N days ago (for dataset range).

    Args:
        days: Number of days ago (default: 90)

    Returns:
        UTC time string in ISO format
    """
    utc_now = datetime.now(timezone.utc)
    time_ago = utc_now - timedelta(days=days)
    time_ago = time_ago.replace(hour=0, minute=0, second=0, microsecond=0)
    return time_ago.strftime("%Y-%m-%dT%H:%M:%S")


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
    gpu_enabled: bool = False,
    selected_steps: Optional[list[str]] = None,
) -> ExperimentResult:
    """
    Run experiment in Kubernetes environment.

    Args:
        implementation: Generated code implementation
        spec: Technical specification
        gpu_enabled: Whether to use GPU
        selected_steps: Workflow steps to execute. Default is ["train"] only.

    Returns:
        ExperimentResult with metrics and status
    """
    experiment_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting experiment {experiment_id} for {spec.title}")
    logger.info(f"Selected steps: {selected_steps or ['train']}")

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

        # Prepare experiment script with selected steps
        experiment_script = prepare_experiment_script(
            implementation, spec, selected_steps=selected_steps
        )

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
    spec: TechSpec,
    ai_craft_path: Optional[str] = None,
    selected_steps: Optional[list[str]] = None,
) -> str:
    """
    Prepare experiment execution script by parsing DAG files.

    The workflow is dynamically determined from ai-craft's Airflow DAG files,
    ensuring the agent uses the same workflow as production.

    Args:
        implementation: Generated code with branch_name, repository, implementation_path
        spec: Technical specification
        ai_craft_path: Path to ai-craft repo (for parsing DAGs)
        selected_steps: Steps to execute. Default is ["train"] only.
                       Options: ["dataset", "train", "calibrate_m3", "calibrate_m1", "mark_success", "validation"]

    Returns:
        Bash script to execute experiment
    """
    # Convert string step names to WorkflowStep enum
    workflow_steps = None
    if selected_steps:
        workflow_steps = []
        for step in selected_steps:
            try:
                workflow_steps.append(WorkflowStep(step))
            except ValueError:
                logger.warning(f"Unknown step: {step}, skipping")
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
        # Get UTC times
        utc_time = get_utc_time_hours_ago(4)
        utc_time_iso = get_utc_time_iso(4)
        utc_time_90d_iso = get_utc_time_iso_days_ago(90)

        # Convert path to module
        module_path = convert_path_to_module(implementation_path)

        # Parse workflow from DAG files
        if ai_craft_path:
            parser = WorkflowParser(ai_craft_path)
            workflow = parser.find_workflow_for_model(module_path)
        else:
            # Try common paths
            for path in ["/Users/choieq/Projects/dable/ai-craft", "/home/dable/ai-craft"]:
                try:
                    parser = WorkflowParser(path)
                    workflow = parser.find_workflow_for_model(module_path)
                    if workflow:
                        break
                except Exception:
                    continue
            else:
                workflow = None

        # Get repo URL (using HTTPS with token for pod access)
        repo_url = f"https://${{GITHUB_TOKEN}}@github.com/{settings.github_org}/{repository}.git"

        script_lines.extend([
            f"echo 'Module: {module_path}'",
            "",
            "# Clone repository and checkout branch",
            f"REPO_DIR=/tmp/{repository}",
            "rm -rf $REPO_DIR",
            f"git clone {repo_url} $REPO_DIR",
            "cd $REPO_DIR",
            f"git checkout {branch_name}",
            "",
        ])

        if workflow:
            # Filter to selected steps (default: train only)
            if workflow_steps:
                filtered_workflow = workflow.filter_steps(workflow_steps)
            else:
                filtered_workflow = workflow.filter_steps([WorkflowStep.TRAIN])

            logger.info(f"Full workflow: {workflow.steps}")
            logger.info(f"Selected steps: {filtered_workflow.steps}")

            script_lines.append(f"echo 'Selected steps: {' -> '.join(s.value for s in filtered_workflow.steps)}'")
            script_lines.append("")
            script_lines.extend(generate_training_script_from_workflow(
                filtered_workflow, utc_time, utc_time_iso, utc_time_90d_iso,
                selected_steps=workflow_steps
            ))
        else:
            # Fallback: simple train command
            logger.warning(f"No workflow found for {module_path}, using simple train")
            script_lines.extend([
                "echo 'Workflow: fallback (simple train)'",
                "",
                f"echo 'Running training for: {module_path}'",
                f'python -c "from {module_path}.train import train; train(\'{utc_time}\')"',
                "",
                "echo '=== Training completed ==='",
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
    task_type: str = "MODEL_TRAINING",
    run_id: str = None
) -> Dict[str, float]:
    """
    Collect metrics from MLflow or other sources.

    Args:
        experiment_id: Experiment identifier (used as MLflow experiment name)
        task_type: Type of task (always MODEL_TRAINING)
        run_id: Optional MLflow run ID for direct lookup

    Returns:
        Dictionary of metrics from MLflow, or default values if unavailable
    """
    from src.integrations.mlflow_client import MLflowClient

    logger.info(f"Collecting metrics for experiment_id={experiment_id}, run_id={run_id}")

    # Initialize MLflow client
    mlflow_client = MLflowClient(tracking_uri=settings.mlflow_tracking_uri)

    # Try to get metrics from MLflow
    metrics = {}

    if run_id:
        # Direct run lookup
        logger.info(f"Fetching metrics directly from run_id: {run_id}")
        metrics = await mlflow_client.get_run_metrics(run_id)
    else:
        # Lookup by experiment name (latest run)
        logger.info(f"Fetching latest run from experiment: {experiment_id}")
        metrics = await mlflow_client.get_latest_run_metrics(experiment_id)

    # Check if we got required metrics
    required_metrics = ["auc", "logloss"]
    has_required = any(m in metrics for m in required_metrics)

    if has_required and metrics:
        logger.info(f"Successfully retrieved {len(metrics)} metrics from MLflow")

        # Log available metrics for debugging
        available_keys = set(metrics.keys())
        logger.debug(f"Available metrics: {sorted(available_keys)}")

        # Return all metrics from MLflow (maximizing flexibility)
        return metrics
    else:
        # Fallback to default values if MLflow doesn't have required metrics
        logger.warning(
            f"Required metrics not found in MLflow. "
            f"Available: {list(metrics.keys()) if metrics else 'None'}. "
            f"Using default values."
        )

        return {
            "auc": 0.0,
            "logloss": 1.0,
            "calibration_error": 1.0,
            "training_time_minutes": 0.0,
            "num_parameters": 0,
        }


def generate_recommendations(
    metrics: Dict[str, float],
    task_type: str = "MODEL_TRAINING"
) -> list[str]:
    """
    Generate recommendations based on metrics.

    Args:
        metrics: Experiment metrics
        task_type: Type of task (always MODEL_TRAINING)

    Returns:
        List of recommendations
    """
    recommendations = []

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

    return recommendations if recommendations else ["No specific recommendations."]


# ===== Session Management Functions =====


async def create_session(
    spec: TechSpec,
    gpu_enabled: bool = False
) -> ExperimentSession:
    """
    Create an experiment session with a persistent pod.

    The pod remains active for running multiple experiments sequentially.

    Args:
        spec: Technical specification
        gpu_enabled: Whether to use GPU

    Returns:
        Created ExperimentSession
    """
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"Creating session {session_id} for {spec.title}")

    launcher = K8sPodLauncher()

    # Determine resource requirements
    if spec.task_type == "MODEL_TRAINING" or gpu_enabled:
        pod_type = "gpu"
        instance_type = settings.default_gpu_instance
    else:
        pod_type = "cpu"
        instance_type = settings.default_cpu_instance

    pod_name = f"ai-tf-box-session-{session_id}"

    try:
        # Launch pod
        logger.info(f"Launching {pod_type} pod for session: {pod_name}")
        pod_info = await launcher.launch_pod(
            pod_name=pod_name,
            pod_type=pod_type,
            instance_type=instance_type
        )

        session = ExperimentSession(
            session_id=session_id,
            pod_name=pod_name,
            status=SessionStatus.ACTIVE,
            pod_type=pod_type,
            instance_type=instance_type
        )

        _active_sessions[session_id] = session
        logger.info(f"Session {session_id} created with pod {pod_name}")

        return session

    except Exception as e:
        logger.error(f"Failed to create session {session_id}: {str(e)}")
        raise


async def run_experiment_in_session(
    session_id: str,
    implementation: Dict[str, Any],
    spec: TechSpec,
    selected_steps: Optional[list[str]] = None
) -> ExperimentResult:
    """
    Run an experiment in an existing session.

    Uses the already-running pod from the session.

    Args:
        session_id: Session ID
        implementation: Generated code implementation
        spec: Technical specification
        selected_steps: Workflow steps to execute

    Returns:
        ExperimentResult with metrics and status
    """
    if session_id not in _active_sessions:
        raise ValueError(f"Session not found: {session_id}")

    session = _active_sessions[session_id]

    if session.status != SessionStatus.ACTIVE:
        raise ValueError(f"Session {session_id} is not active (status: {session.status})")

    experiment_id = str(uuid.uuid4())[:8]
    logger.info(f"Running experiment {experiment_id} in session {session_id}")

    launcher = K8sPodLauncher()

    try:
        # Prepare experiment script
        experiment_script = prepare_experiment_script(
            implementation, spec, selected_steps=selected_steps
        )

        # Execute on existing pod
        logger.info(f"Executing experiment on session pod {session.pod_name}")
        execution_result = await launcher.execute_on_pod(
            pod_name=session.pod_name,
            script=experiment_script
        )

        # Collect metrics
        metrics = await collect_metrics(experiment_id, spec.task_type)

        # Create result
        result = ExperimentResult(
            experiment_id=experiment_id,
            status="SUCCESS" if execution_result.get("success") else "FAILURE",
            metrics=metrics,
            pr_url=None,
            pod_name=session.pod_name,
            recommendations=generate_recommendations(metrics, spec.task_type)
        )

        # Track experiment in session
        session.experiment_ids.append(experiment_id)

        logger.info(f"Experiment {experiment_id} completed in session {session_id}")
        return result

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed in session {session_id}: {str(e)}")

        # Return failure result
        return ExperimentResult(
            experiment_id=experiment_id,
            status="FAILURE",
            metrics={},
            pr_url=None,
            pod_name=session.pod_name,
            recommendations=[f"Experiment failed: {str(e)}"]
        )


async def terminate_session(session_id: str) -> Dict[str, str]:
    """
    Terminate a session and cleanup its pod.

    Args:
        session_id: Session ID to terminate

    Returns:
        Status message
    """
    if session_id not in _active_sessions:
        logger.warning(f"Session not found: {session_id}")
        return {"status": "error", "message": f"Session not found: {session_id}"}

    session = _active_sessions[session_id]
    launcher = K8sPodLauncher()

    try:
        # Cleanup pod
        logger.info(f"Terminating session {session_id}, cleaning up pod {session.pod_name}")
        await launcher.cleanup_pod(session.pod_name)

        # Update session status
        session.status = SessionStatus.TERMINATED

        # Remove from active sessions
        del _active_sessions[session_id]

        logger.info(f"Session {session_id} terminated successfully")
        return {
            "status": "success",
            "message": f"Session {session_id} terminated",
            "experiments_run": len(session.experiment_ids)
        }

    except Exception as e:
        logger.error(f"Failed to terminate session {session_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to terminate session: {str(e)}"
        }


async def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a session.

    Args:
        session_id: Session ID

    Returns:
        Session information dict or None if not found
    """
    if session_id not in _active_sessions:
        return None

    session = _active_sessions[session_id]

    return {
        "session_id": session.session_id,
        "pod_name": session.pod_name,
        "status": session.status.value,
        "created_at": session.created_at.isoformat(),
        "experiments_run": len(session.experiment_ids),
        "experiment_ids": session.experiment_ids,
        "pod_type": session.pod_type,
        "instance_type": session.instance_type
    }


async def list_active_sessions() -> list[Dict[str, Any]]:
    """
    List all active sessions.

    Returns:
        List of session information dicts
    """
    sessions = []

    for session_id, session in _active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "pod_name": session.pod_name,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "experiments_run": len(session.experiment_ids),
            "pod_type": session.pod_type,
            "instance_type": session.instance_type
        })

    return sessions