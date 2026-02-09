"""Data models for AI Research Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    description: str = Field(..., description="Human-readable experiment description")
    training_command: str = Field(..., description="Command to execute for training")
    setup_commands: Optional[str] = Field(None, description="Shell commands to run before training (e.g., code modifications)")
    repo_path: str = Field(default="/home/dable/ai-craft", description="Path to git repo on pod")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class ExperimentResult(BaseModel):
    """Result of a single experiment."""

    config: ExperimentConfig = Field(..., description="Experiment configuration")
    status: ExperimentStatus = Field(..., description="Experiment status")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Experiment metrics")
    duration_seconds: Optional[float] = Field(None, description="Execution duration in seconds")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    git_diff: str = Field(default="", description="git diff main output before training")
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Evaluation results")
    log_file: Optional[str] = Field(None, description="Path to log file on pod")
    diff_file: Optional[str] = Field(None, description="Path to diff file on pod")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")

    model_config = ConfigDict(use_enum_values=True)


class ExperimentBatch(BaseModel):
    """A batch of experiments to run sequentially."""

    batch_id: str = Field(..., description="Unique batch identifier")
    pod_name: str = Field(..., description="Kubernetes pod name")
    namespace: str = Field(default="tf-box", description="Kubernetes namespace")
    experiments: List[ExperimentConfig] = Field(default_factory=list, description="List of experiments")
    results: List[ExperimentResult] = Field(default_factory=list, description="List of results")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Batch creation time")

    model_config = ConfigDict(use_enum_values=True)
