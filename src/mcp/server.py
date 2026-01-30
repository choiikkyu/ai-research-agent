"""FastMCP Server for AI Research Automation Agent.

Workflow (Model Training):
1. analyze_tech_spec - Parse Notion spec
2. generate_implementation - Generate code
3. create_draft_pr - Create Draft PR for review
4. [User reviews PR and approves]
5. approve_and_run_experiment - Run experiment on K8s
6. evaluate_experiment - Check results
7. finalize_pr - Merge or close based on results
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context
from pydantic import Field

from src.core.config import settings
from src.core.models import (
    TechSpec,
    ExperimentRequest,
    ExperimentResult,
)
from src.mcp.tools import (
    analyze_spec,
    generate_code,
    run_experiment,
    evaluate_results,
    manage_pr,
    cleanup_resources,
)
from src.mcp.tools.experiment import (
    convert_path_to_module,
    get_utc_time_hours_ago,
    generate_training_command,
    create_session,
    run_experiment_in_session,
    terminate_session,
    get_session_info,
    list_active_sessions,
)
from src.mcp.tools.github import create_model_pr_with_2commit_strategy

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("AI Research Automation Agent")


@mcp.tool()
async def analyze_tech_spec(
    ctx: Context,
    spec_url: str = Field(..., description="URL to the technical specification (Notion, etc.)"),
) -> TechSpec:
    """
    Analyze a technical specification document from Notion or other sources.

    This tool extracts and analyzes the specification to determine:
    - Task type (MODEL_TRAINING or FEATURE_ENGINEERING)
    - Requirements and constraints
    - Expected outputs
    """
    logger.info(f"Analyzing tech spec from: {spec_url}")
    return await analyze_spec(spec_url)


@mcp.tool()
async def generate_implementation(
    ctx: Context,
    spec: TechSpec = Field(..., description="Analyzed technical specification"),
    target_repo: str = Field(default="ai-craft", description="Target repository"),
) -> Dict[str, Any]:
    """
    Generate code implementation based on the technical specification.

    Uses Claude API to generate:
    - Model training code for pCTR/pCVR models
    - Feature engineering pipelines
    - Required configuration files
    """
    logger.info(f"Generating implementation for: {spec.title}")
    return await generate_code(spec, target_repo)


@mcp.tool()
async def launch_experiment(
    ctx: Context,
    implementation: Dict[str, Any] = Field(..., description="Generated implementation"),
    spec: TechSpec = Field(..., description="Technical specification"),
    gpu_enabled: bool = Field(default=False, description="Whether to use GPU"),
) -> ExperimentResult:
    """
    Launch an experiment in Kubernetes using dable-k8s-runner.

    This tool:
    - Creates appropriate K8s pod (GPU/CPU based on task)
    - Runs the experiment
    - Monitors progress
    - Collects metrics
    """
    logger.info(f"Launching experiment for: {spec.title}")
    return await run_experiment(implementation, spec, gpu_enabled)


@mcp.tool()
async def evaluate_experiment(
    ctx: Context,
    result: ExperimentResult = Field(..., description="Experiment result"),
    spec: TechSpec = Field(..., description="Technical specification"),
) -> Dict[str, Any]:
    """
    Evaluate experiment results against success criteria.

    Evaluation criteria:
    - Model Training: AUC > 0.85, LogLoss < 0.35, Calibration Error < 0.02
    - Feature Engineering: Null Ratio < 10%, Importance > 0.05, Latency Impact < 10ms
    """
    logger.info(f"Evaluating experiment: {result.experiment_id}")
    return await evaluate_results(result, spec)


@mcp.tool()
async def manage_pull_request(
    ctx: Context,
    implementation: Dict[str, Any] = Field(..., description="Implementation details"),
    evaluation: Dict[str, Any] = Field(..., description="Evaluation results"),
    auto_merge: bool = Field(default=False, description="Auto-merge if successful"),
) -> Dict[str, Any]:
    """
    Manage GitHub pull request lifecycle.

    Actions:
    - Create PR with experiment results
    - Update PR status based on evaluation
    - Auto-merge if successful and enabled
    - Close PR if experiment failed
    """
    logger.info(f"Managing PR for implementation: {implementation.get('branch_name')}")
    return await manage_pr(implementation, evaluation, auto_merge)


@mcp.tool()
async def cleanup_experiment_resources(
    ctx: Context,
    experiment_id: str = Field(..., description="Experiment identifier"),
    cleanup_pr: bool = Field(default=False, description="Also cleanup PR and branch"),
) -> Dict[str, Any]:
    """
    Cleanup experiment resources.

    Cleans up:
    - Kubernetes pods
    - Temporary files
    - GitHub branches and PRs (if requested)
    """
    logger.info(f"Cleaning up resources for experiment: {experiment_id}")
    return await cleanup_resources(experiment_id, cleanup_pr)


# ==============================================================================
# SIMPLIFIED WORKFLOW: Direct Experiment Execution
# ==============================================================================


@mcp.tool()
async def run_full_workflow(
    ctx: Context,
    request: ExperimentRequest = Field(..., description="Complete experiment request"),
) -> Dict[str, Any]:
    """
    Run simplified workflow: analyze, generate, and execute experiment.

    Steps:
    1. Analyze technical specification
    2. Generate code implementation
    3. Launch experiment immediately (no PR yet)
    4. Return results

    After experiment completes, user can optionally create PR using create_pull_request.
    """
    logger.info(f"Starting workflow for: {request.spec_url}")

    try:
        # Step 1: Analyze spec
        spec = await analyze_spec(request.spec_url)

        # Step 2: Generate code
        implementation = await generate_code(spec, request.repo)

        # Step 3: Launch experiment directly
        result = await run_experiment(implementation, spec, request.gpu_enabled)

        # Step 4: Evaluate results
        evaluation = await evaluate_results(result, spec)

        return {
            "experiment_id": result.experiment_id,
            "status": result.status,
            "spec": spec.model_dump(),
            "evaluation": evaluation,
            "metrics": result.metrics,
            "pod_name": result.pod_name,
            "recommendations": result.recommendations,
            "message": (
                f"Experiment {result.experiment_id} completed.\n"
                f"Status: {result.status}\n"
                f"Use create_pull_request(experiment_id='{result.experiment_id}') to create a PR."
            ),
            "next_step": "create_pull_request (optional)",
        }

    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e),
        }


@mcp.tool()
async def run_full_workflow_auto(
    ctx: Context,
    request: ExperimentRequest = Field(..., description="Complete experiment request"),
) -> Dict[str, Any]:
    """
    Run complete end-to-end workflow WITHOUT waiting for approval.

    Use this only when you trust the auto-generated code completely.

    Steps:
    1. Analyze technical specification
    2. Generate code implementation
    3. Create PR
    4. Launch experiment immediately
    5. Evaluate results
    6. Update PR (merge/close based on results)
    """
    logger.info(f"Starting auto workflow for: {request.spec_url}")

    try:
        # Step 1: Analyze spec
        spec = await analyze_spec(request.spec_url)

        # Step 2: Generate code
        implementation = await generate_code(spec, request.repo)

        # Step 3: Launch experiment
        result = await run_experiment(implementation, spec, request.gpu_enabled)

        # Step 4: Evaluate results
        evaluation = await evaluate_results(result, spec)

        # Step 5: Manage PR
        pr_result = await manage_pr(implementation, evaluation, request.auto_merge)

        # Step 6: Cleanup if failed
        if evaluation.get("passed") == False and request.cleanup_on_failure:
            await cleanup_resources(result.experiment_id, cleanup_pr=True)

        return {
            "experiment_id": result.experiment_id,
            "status": "SUCCESS" if evaluation.get("passed") else "FAILURE",
            "spec": spec.model_dump(),
            "evaluation": evaluation,
            "pr_result": pr_result,
            "metrics": result.metrics,
        }

    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e),
        }


@mcp.resource("experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> str:
    """Get details of a specific experiment."""
    # TODO: Implement fetching from Redis or database
    return f"Experiment {experiment_id} details not yet available"


@mcp.resource("metrics/{metric_type}")
async def get_metrics(metric_type: str) -> str:
    """Get specific system metrics."""
    # TODO: Implement metrics collection
    return f"Metrics for {metric_type} not yet implemented"


# ==============================================================================
# SESSION MANAGEMENT: Multi-Experiment Pod Reuse
# ==============================================================================


@mcp.tool()
async def start_experiment_session(
    ctx: Context,
    task_description: str = Field(..., description="Description of the task (e.g., 'Experiment with whisky mtl models')"),
    gpu_enabled: bool = Field(default=True, description="Whether to use GPU (default: True)"),
) -> Dict[str, Any]:
    """
    Start an experiment session with a persistent pod.

    The pod remains active across multiple experiments until explicitly terminated.
    Use this for running multiple experiments sequentially without recreating the pod.

    Returns:
        session_id to use for subsequent experiments
    """
    from src.mcp.tools import analyze_spec

    logger.info(f"Starting experiment session for: {task_description}")

    try:
        # Analyze spec to determine requirements
        spec = await analyze_spec(task_description)

        # Create session with pod
        session = await create_session(spec, gpu_enabled)

        return {
            "session_id": session.session_id,
            "pod_name": session.pod_name,
            "pod_type": session.pod_type,
            "instance_type": session.instance_type,
            "status": session.status.value,
            "message": (
                f"Session started!\n"
                f"Session ID: {session.session_id}\n"
                f"Pod: {session.pod_name}\n"
                f"Type: {session.pod_type} ({session.instance_type})\n\n"
                f"Use run_in_session(session_id='{session.session_id}', ...) to run experiments."
            ),
        }

    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def run_in_session(
    ctx: Context,
    session_id: str = Field(..., description="Session ID from start_experiment_session"),
    experiment_description: str = Field(..., description="Description of the experiment to run"),
) -> Dict[str, Any]:
    """
    Run an experiment in an existing session.

    Uses the already-running pod from the session.
    No new pod is created, allowing for faster iteration.

    Returns:
        Experiment result with metrics
    """
    from src.mcp.tools import analyze_spec, generate_code

    logger.info(f"Running experiment in session {session_id}: {experiment_description}")

    try:
        # Analyze spec
        spec = await analyze_spec(experiment_description)

        # Generate code
        implementation = await generate_code(spec)

        # Run experiment in session
        result = await run_experiment_in_session(session_id, implementation, spec)

        return {
            "experiment_id": result.experiment_id,
            "session_id": session_id,
            "status": result.status,
            "metrics": result.metrics,
            "pod_name": result.pod_name,
            "recommendations": result.recommendations,
        }

    except Exception as e:
        logger.error(f"Failed to run experiment in session {session_id}: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
async def stop_session(
    ctx: Context,
    session_id: str = Field(..., description="Session ID to terminate"),
) -> Dict[str, Any]:
    """
    Terminate a session and cleanup its pod.

    After calling this, the session cannot be reused.
    """
    logger.info(f"Terminating session {session_id}")

    result = await terminate_session(session_id)

    return result


@mcp.tool()
async def get_session_status(
    ctx: Context,
    session_id: str = Field(..., description="Session ID"),
) -> Dict[str, Any]:
    """
    Get information about a session.

    Returns:
        Session details including pod info, experiments run, etc.
    """
    info = await get_session_info(session_id)

    if info is None:
        return {
            "error": f"Session not found: {session_id}",
        }

    return info


@mcp.tool()
async def list_sessions(
    ctx: Context,
) -> List[Dict[str, Any]]:
    """
    List all active experiment sessions.

    Returns:
        List of active sessions with their details
    """
    sessions = await list_active_sessions()

    return sessions


@mcp.tool()
async def run_batch_experiments(
    ctx: Context,
    base_task_description: str = Field(
        ...,
        description="Base task description (e.g., 'Create mtl02 model based on mtl01 with learning rate {lr}')"
    ),
    hyperparameter_grid: Dict[str, List[str]] = Field(
        ...,
        description="Hyperparameter combinations to test. Example: {'lr': ['0.1', '0.01', '0.001'], 'layers': ['2', '3']}"
    ),
    use_single_pod: bool = Field(default=True, description="Use single pod for all experiments (sequential execution)"),
    gpu_enabled: bool = Field(default=True, description="Whether to use GPU"),
) -> Dict[str, Any]:
    """
    Run batch experiments with different hyperparameters.

    Generates all combinations of hyperparameters and runs experiments sequentially.
    If use_single_pod=True, creates one session and runs all experiments on the same pod.

    Example:
        base_task_description: "Test whisky mtl model with lr={lr} and layers={layers}"
        hyperparameter_grid: {"lr": ["0.1", "0.01"], "layers": ["2", "3"]}

        This will run 4 experiments:
        1. lr=0.1, layers=2
        2. lr=0.1, layers=3
        3. lr=0.01, layers=2
        4. lr=0.01, layers=3

    Returns:
        Batch results with all experiments and best performing configuration
    """
    from itertools import product
    from src.mcp.tools import analyze_spec, generate_code

    logger.info(f"Starting batch experiments for: {base_task_description}")
    logger.info(f"Hyperparameter grid: {hyperparameter_grid}")

    # Generate all combinations
    keys = list(hyperparameter_grid.keys())
    values = list(hyperparameter_grid.values())
    combinations = list(product(*values))

    logger.info(f"Total combinations to test: {len(combinations)}")

    results = []
    session_id = None

    try:
        # Create session if using single pod
        if use_single_pod:
            base_spec = await analyze_spec(base_task_description)
            session = await create_session(base_spec, gpu_enabled)
            session_id = session.session_id
            logger.info(f"Created session {session_id} for batch experiments")

        # Run each combination
        for i, combination in enumerate(combinations, 1):
            # Create parameter dict
            param_dict = dict(zip(keys, combination))

            # Format task description with parameters
            task_description = base_task_description
            for key, value in param_dict.items():
                task_description = task_description.replace(f"{{{key}}}", str(value))

            logger.info(f"Running experiment {i}/{len(combinations)}: {param_dict}")

            try:
                # Analyze spec
                spec = await analyze_spec(task_description)

                # Generate code
                implementation = await generate_code(spec)

                # Run experiment
                if use_single_pod and session_id:
                    result = await run_experiment_in_session(session_id, implementation, spec)
                else:
                    result = await run_experiment(implementation, spec, gpu_enabled)

                results.append({
                    "parameters": param_dict,
                    "experiment_id": result.experiment_id,
                    "status": result.status,
                    "metrics": result.metrics,
                    "recommendations": result.recommendations,
                })

                logger.info(
                    f"Experiment {i} completed: "
                    f"AUC={result.metrics.get('auc', 'N/A')}, "
                    f"LogLoss={result.metrics.get('logloss', 'N/A')}"
                )

            except Exception as e:
                logger.error(f"Experiment {i} failed: {e}")
                results.append({
                    "parameters": param_dict,
                    "status": "error",
                    "error": str(e),
                })

        # Cleanup session if created
        if session_id:
            await terminate_session(session_id)

        # Find best result
        successful_results = [r for r in results if r.get("status") == "SUCCESS"]
        if successful_results:
            best_result = max(
                successful_results,
                key=lambda r: r["metrics"].get("auc", 0)
            )
        else:
            best_result = None

        return {
            "total_experiments": len(combinations),
            "successful": len(successful_results),
            "failed": len(combinations) - len(successful_results),
            "results": results,
            "best_result": best_result,
            "session_id": session_id,
            "message": (
                f"Batch experiments completed: {len(successful_results)}/{len(combinations)} successful.\n"
                f"Best config: {best_result['parameters'] if best_result else 'None'}"
            )
        }

    except Exception as e:
        logger.error(f"Batch experiment failed: {e}")

        # Cleanup session on error
        if session_id:
            try:
                await terminate_session(session_id)
            except:
                pass

        return {
            "status": "error",
            "error": str(e),
            "completed": len(results),
            "results": results,
        }


# ==============================================================================
# PULL REQUEST CREATION (After Experiment)
# ==============================================================================


@mcp.tool()
async def create_pull_request(
    ctx: Context,
    experiment_id: str = Field(..., description="Experiment ID to create PR for"),
    auto_merge: bool = Field(default=False, description="Auto-merge if evaluation passed"),
) -> Dict[str, Any]:
    """
    Create a pull request for a completed experiment.

    Call this after running an experiment (via run_experiment, run_in_session, etc.)
    to create a PR with the experiment code and results.

    Args:
        experiment_id: ID of the completed experiment
        auto_merge: Automatically merge if experiment passed evaluation

    Returns:
        PR information (URL, number, etc.)
    """
    logger.info(f"Creating PR for experiment {experiment_id}")

    try:
        # For now, return a placeholder
        # TODO: Implement actual PR creation using manage_pr
        # This requires tracking experiment implementations

        return {
            "experiment_id": experiment_id,
            "status": "pending",
            "message": (
                f"PR creation for experiment {experiment_id} is not yet implemented.\n"
                "Currently, experiments run without creating PRs automatically.\n"
                "You can manually create a PR from the experiment branch if needed."
            ),
            "note": "This feature will be implemented in a future update."
        }

    except Exception as e:
        logger.error(f"Failed to create PR for experiment {experiment_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    import uvloop

    # Use uvloop for better async performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Start the MCP server
    mcp.run()