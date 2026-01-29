"""Data models for AI Research Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperimentState(str, Enum):
    """Experiment lifecycle states.

    Flow:
    DRAFT_PR_CREATED -> (user approval) -> APPROVED -> RUNNING -> COMPLETED/FAILED
    """
    DRAFT_PR_CREATED = "draft_pr_created"  # Draft PR created, waiting for approval
    APPROVED = "approved"                   # User approved, ready to run
    RUNNING = "running"                     # Experiment in progress
    COMPLETED = "completed"                 # Experiment finished successfully
    FAILED = "failed"                       # Experiment failed
    CANCELLED = "cancelled"                 # User cancelled


class TechSpec(BaseModel):
    """Technical specification from Notion or other sources."""

    title: str = Field(..., description="Title of the specification")
    content: str = Field(..., description="Specification content")
    task_type: str = Field(default="MODEL_TRAINING", description="Task type: MODEL_TRAINING only")
    repository: str = Field(default="ai-craft", description="Target repository (ai-craft only)")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Additional requirements")


class ExperimentRequest(BaseModel):
    """Request to run an experiment."""

    spec_url: str = Field(..., description="URL to the technical specification")
    repo: str = Field(default="ai-craft", description="Target repository")
    gpu_enabled: bool = Field(default=False, description="Whether to use GPU")
    auto_merge: bool = Field(default=False, description="Auto-merge if successful")
    cleanup_on_failure: bool = Field(default=True, description="Cleanup resources on failure")


class ExperimentResult(BaseModel):
    """Result of an experiment."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    status: str = Field(..., description="Status: SUCCESS, FAILURE, or IN_PROGRESS")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Experiment metrics")
    pr_url: Optional[str] = Field(None, description="Pull request URL")
    pod_name: Optional[str] = Field(None, description="Kubernetes pod name")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class PendingExperiment(BaseModel):
    """Experiment waiting for user approval.

    Created after Draft PR is made, before experiment runs.
    """

    experiment_id: str = Field(..., description="Unique experiment identifier")
    state: ExperimentState = Field(default=ExperimentState.DRAFT_PR_CREATED)
    spec: TechSpec = Field(..., description="Technical specification")
    implementation: Dict[str, Any] = Field(..., description="Generated implementation")
    pr_url: str = Field(..., description="Draft PR URL")
    pr_number: int = Field(..., description="PR number")
    branch_name: str = Field(..., description="Git branch name")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Training specific info
    training_command: Optional[str] = Field(None, description="Training command to run")
    module_path: Optional[str] = Field(None, description="Python module path for training")

    class Config:
        use_enum_values = True