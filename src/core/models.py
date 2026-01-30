"""Data models for AI Research Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field




class SessionStatus(str, Enum):
    """Experiment session status.

    Flow:
    ACTIVE -> TERMINATED
    """
    ACTIVE = "active"           # Session is running with a pod
    TERMINATED = "terminated"   # Session and pod terminated


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


class ExperimentSession(BaseModel):
    """Experiment session for running multiple experiments on a single pod.

    The pod remains active across multiple experiments until explicitly terminated.
    """

    session_id: str = Field(..., description="Unique session identifier")
    pod_name: str = Field(..., description="Kubernetes pod name")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    experiment_ids: List[str] = Field(default_factory=list, description="List of experiment IDs in this session")
    pod_type: str = Field(..., description="Pod type: gpu or cpu")
    instance_type: str = Field(..., description="Instance type (e.g., g4dn.xlarge)")

    class Config:
        use_enum_values = True