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
    ExperimentState,
    PendingExperiment,
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
)
from src.mcp.tools.github import create_model_pr_with_2commit_strategy

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("AI Research Automation Agent")

# In-memory storage for pending experiments (TODO: Move to Redis for production)
pending_experiments: Dict[str, PendingExperiment] = {}


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
# NEW WORKFLOW: Draft PR -> Approval -> Experiment
# ==============================================================================


@mcp.tool()
async def create_draft_pr_for_review(
    ctx: Context,
    spec: TechSpec = Field(..., description="Analyzed technical specification"),
    implementation: Dict[str, Any] = Field(..., description="Generated implementation"),
) -> Dict[str, Any]:
    """
    Create a Draft PR for user review before running experiment.

    This is Step 3 of the workflow. After this:
    1. User reviews the Draft PR
    2. User calls approve_and_run_experiment to start the experiment

    Returns experiment_id to track this pending experiment.
    """
    experiment_id = str(uuid.uuid4())[:8]
    logger.info(f"Creating Draft PR for experiment {experiment_id}: {spec.title}")

    try:
        # Determine PR strategy and create PR
        pr_strategy = implementation.get("pr_strategy", "new_implementation")
        repository = implementation.get("repository", "ai-craft")

        if pr_strategy == "model_modification" and implementation.get("reference_path"):
            # Use 2-commit strategy for model modifications
            reference_name = implementation.get("reference_name", "")
            new_model_name = implementation.get("implementation_path", "").rstrip("/").split("/")[-1]

            # Extract modifications from generated content
            modifications = _extract_modifications(implementation, reference_name, new_model_name)

            pr_result = await create_model_pr_with_2commit_strategy(
                repo_name=repository,
                reference_path=implementation.get("reference_path"),
                destination_path=implementation.get("implementation_path"),
                reference_name=reference_name,
                new_name=new_model_name,
                modifications=modifications,
                pr_title=f"[AI Agent] {spec.title}",
                pr_description=spec.content[:500],
                draft=True
            )
        else:
            # Standard PR creation
            evaluation = {"passed": None, "task_type": spec.task_type}  # No evaluation yet
            pr_result = await manage_pr(implementation, evaluation, auto_merge=False)

        # Generate training command for display
        module_path = convert_path_to_module(implementation.get("implementation_path", ""))
        utc_time = get_utc_time_hours_ago(4)
        training_cmd = generate_training_command(module_path, utc_time) if spec.task_type == "MODEL_TRAINING" else None

        # Store pending experiment
        pending = PendingExperiment(
            experiment_id=experiment_id,
            state=ExperimentState.DRAFT_PR_CREATED,
            spec=spec,
            implementation=implementation,
            pr_url=pr_result.get("pr_url", ""),
            pr_number=pr_result.get("pr_number", 0),
            branch_name=pr_result.get("branch_name", implementation.get("branch_name", "")),
            training_command=training_cmd,
            module_path=module_path,
        )
        pending_experiments[experiment_id] = pending

        logger.info(f"Draft PR created: {pr_result.get('pr_url')}")

        return {
            "experiment_id": experiment_id,
            "state": ExperimentState.DRAFT_PR_CREATED.value,
            "pr_url": pr_result.get("pr_url"),
            "pr_number": pr_result.get("pr_number"),
            "branch_name": pending.branch_name,
            "training_command": training_cmd,
            "message": "Draft PR created. Review the PR and call approve_and_run_experiment to start.",
        }

    except Exception as e:
        logger.error(f"Failed to create Draft PR: {e}")
        return {
            "experiment_id": experiment_id,
            "state": "error",
            "error": str(e),
        }


@mcp.tool()
async def list_pending_experiments(
    ctx: Context,
) -> List[Dict[str, Any]]:
    """
    List all experiments waiting for user approval.

    Returns list of pending experiments with their PR URLs and states.
    """
    result = []
    for exp_id, exp in pending_experiments.items():
        result.append({
            "experiment_id": exp_id,
            "state": exp.state,
            "title": exp.spec.title,
            "pr_url": exp.pr_url,
            "pr_number": exp.pr_number,
            "branch_name": exp.branch_name,
            "task_type": exp.spec.task_type,
            "training_command": exp.training_command,
            "created_at": exp.created_at.isoformat(),
        })
    return result


@mcp.tool()
async def approve_and_run_experiment(
    ctx: Context,
    experiment_id: str = Field(..., description="Experiment ID from create_draft_pr_for_review"),
    gpu_enabled: bool = Field(default=True, description="Whether to use GPU (default: True for model training)"),
    selected_steps: list[str] = Field(
        default=["train"],
        description="Workflow steps to execute. Default is ['train'] only. "
                   "Options: 'dataset', 'train', 'calibrate_m3', 'calibrate_m1', 'mark_success', 'validation'. "
                   "Example: ['train', 'calibrate_m3'] to run training and calibration."
    ),
) -> Dict[str, Any]:
    """
    Approve a pending experiment and run it on K8s.

    Call this after reviewing the Draft PR.
    This will:
    1. Launch a K8s pod (GPU for model training)
    2. Clone the repo and checkout the PR branch
    3. Run the selected workflow steps (default: train only)
    4. Collect metrics and return results

    Available steps:
    - dataset: Create dataset (required for whisky/wheres models)
    - train: Run model training
    - calibrate_m3: Run calibration (cal_m3)
    - calibrate_m1: Run calibration (cal_m1)
    - mark_success: Mark model as successful
    - validation: Run model validation

    By default, only 'train' is executed. Add more steps as needed.
    """
    if experiment_id not in pending_experiments:
        return {
            "error": f"Experiment {experiment_id} not found",
            "available_experiments": list(pending_experiments.keys()),
        }

    pending = pending_experiments[experiment_id]

    if pending.state != ExperimentState.DRAFT_PR_CREATED:
        return {
            "error": f"Experiment {experiment_id} is not in DRAFT_PR_CREATED state",
            "current_state": pending.state,
        }

    logger.info(f"Approving and running experiment {experiment_id}: {pending.spec.title}")

    # Update state
    pending.state = ExperimentState.APPROVED
    pending_experiments[experiment_id] = pending

    try:
        # Determine GPU requirement
        use_gpu = gpu_enabled or pending.spec.task_type == "MODEL_TRAINING"

        # Update state to running
        pending.state = ExperimentState.RUNNING
        pending_experiments[experiment_id] = pending

        # Run the experiment with selected steps
        result = await run_experiment(
            implementation=pending.implementation,
            spec=pending.spec,
            gpu_enabled=use_gpu,
            selected_steps=selected_steps,
        )

        # Update result with PR info
        result.pr_url = pending.pr_url

        # Update state based on result
        if result.status == "SUCCESS":
            pending.state = ExperimentState.COMPLETED
        else:
            pending.state = ExperimentState.FAILED
        pending_experiments[experiment_id] = pending

        logger.info(f"Experiment {experiment_id} completed with status: {result.status}")

        return {
            "experiment_id": experiment_id,
            "state": pending.state.value if hasattr(pending.state, 'value') else pending.state,
            "status": result.status,
            "metrics": result.metrics,
            "pr_url": pending.pr_url,
            "pod_name": result.pod_name,
            "recommendations": result.recommendations,
            "training_command": pending.training_command,
        }

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        pending.state = ExperimentState.FAILED
        pending_experiments[experiment_id] = pending

        return {
            "experiment_id": experiment_id,
            "state": ExperimentState.FAILED.value,
            "error": str(e),
        }


@mcp.tool()
async def cancel_pending_experiment(
    ctx: Context,
    experiment_id: str = Field(..., description="Experiment ID to cancel"),
    cleanup_pr: bool = Field(default=False, description="Also close PR and delete branch"),
) -> Dict[str, Any]:
    """
    Cancel a pending experiment.

    Optionally cleans up the Draft PR and branch.
    """
    if experiment_id not in pending_experiments:
        return {"error": f"Experiment {experiment_id} not found"}

    pending = pending_experiments[experiment_id]

    # Update state
    pending.state = ExperimentState.CANCELLED
    pending_experiments[experiment_id] = pending

    result = {
        "experiment_id": experiment_id,
        "state": ExperimentState.CANCELLED.value,
        "message": "Experiment cancelled",
    }

    if cleanup_pr:
        cleanup_result = await cleanup_resources(experiment_id, cleanup_pr=True)
        result["cleanup"] = cleanup_result

    logger.info(f"Experiment {experiment_id} cancelled")
    return result


def _extract_modifications(
    implementation: Dict[str, Any],
    reference_name: str,
    new_name: str
) -> Dict[str, Dict[str, str]]:
    """Extract modifications from implementation for 2-commit strategy.

    This is a placeholder - in practice, this should compare the generated
    code with the reference to find actual differences.
    """
    # TODO: Implement actual diff extraction from generated content
    # For now, return empty modifications (Commit 2 will have no changes)
    return {}


@mcp.tool()
async def run_full_workflow(
    ctx: Context,
    request: ExperimentRequest = Field(..., description="Complete experiment request"),
) -> Dict[str, Any]:
    """
    Run workflow up to Draft PR creation (stops for user approval).

    Steps:
    1. Analyze technical specification
    2. Generate code implementation
    3. Create Draft PR for review

    After this, user should:
    4. Review the Draft PR
    5. Call approve_and_run_experiment(experiment_id) to run the experiment

    For auto-run without approval, use run_full_workflow_auto.
    """
    logger.info(f"Starting workflow for: {request.spec_url}")

    try:
        # Step 1: Analyze spec
        spec = await analyze_spec(request.spec_url)

        # Step 2: Generate code
        implementation = await generate_code(spec, request.repo)

        # Step 3: Create Draft PR (stops here for user approval)
        draft_result = await create_draft_pr_for_review(ctx, spec, implementation)

        return {
            "experiment_id": draft_result.get("experiment_id"),
            "state": draft_result.get("state"),
            "pr_url": draft_result.get("pr_url"),
            "pr_number": draft_result.get("pr_number"),
            "branch_name": draft_result.get("branch_name"),
            "training_command": draft_result.get("training_command"),
            "spec": spec.model_dump(),
            "message": (
                "Draft PR created. Review the PR and run:\n"
                f"  approve_and_run_experiment(experiment_id='{draft_result.get('experiment_id')}')\n"
                "to start the experiment."
            ),
            "next_step": "approve_and_run_experiment",
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


if __name__ == "__main__":
    import uvloop

    # Use uvloop for better async performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Start the MCP server
    mcp.run()