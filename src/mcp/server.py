"""FastMCP Server for AI Research Automation Agent."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context
from pydantic import Field

from src.core.config import settings
from src.core.models import TechSpec, ExperimentRequest, ExperimentResult
from src.mcp.tools import (
    analyze_spec,
    generate_code,
    run_experiment,
    evaluate_results,
    manage_pr,
    cleanup_resources,
)

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


@mcp.tool()
async def run_full_workflow(
    ctx: Context,
    request: ExperimentRequest = Field(..., description="Complete experiment request"),
) -> Dict[str, Any]:
    """
    Run the complete end-to-end workflow.

    Steps:
    1. Analyze technical specification
    2. Generate code implementation
    3. Launch experiment
    4. Evaluate results
    5. Manage PR
    6. Cleanup resources (if needed)
    """
    logger.info(f"Starting full workflow for: {request.spec_url}")

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