"""Data models for AI Research Agent."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TechSpec(BaseModel):
    """Technical specification from Notion or other sources."""

    title: str = Field(..., description="Title of the specification")
    content: str = Field(..., description="Specification content")
    task_type: str = Field(..., description="Task type: MODEL_TRAINING or FEATURE_ENGINEERING")
    repository: str = Field(default="ai-craft", description="Target repository")
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