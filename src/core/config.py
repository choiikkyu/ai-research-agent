"""Configuration management for AI Research Agent."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic API
    anthropic_api_key: SecretStr = Field(
        ..., description="Anthropic API key for Claude"
    )

    # Slack Configuration
    slack_bot_token: Optional[SecretStr] = Field(
        None, description="Slack Bot User OAuth Token"
    )
    slack_app_token: Optional[SecretStr] = Field(
        None, description="Slack App-Level Token"
    )
    slack_signing_secret: Optional[SecretStr] = Field(
        None, description="Slack Signing Secret"
    )

    # GitHub Configuration
    github_token: SecretStr = Field(
        ..., description="GitHub Personal Access Token"
    )
    github_org: str = Field(default="teamdable", description="GitHub Organization")
    github_repo_ai_craft: str = Field(
        default="ai-craft", description="AI Craft Repository"
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://mlflow.ai.svc.cluster.local:5000",
        description="MLflow Tracking Server URI",
    )
    mlflow_s3_endpoint_url: Optional[str] = Field(
        None, description="S3 Endpoint URL for MLflow artifacts"
    )

    # Redis Configuration
    redis_host: str = Field(
        default="redis.ai.svc.cluster.local", description="Redis host"
    )
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")

    # Kubernetes Configuration
    k8s_namespace: str = Field(
        default="tf-box", description="Kubernetes namespace for pods"
    )
    k8s_service_account: str = Field(
        default="ai-research-agent", description="Kubernetes service account"
    )

    # AWS Configuration
    aws_default_region: str = Field(
        default="ap-northeast-2", description="AWS default region"
    )
    aws_access_key_id: Optional[str] = Field(None, description="AWS Access Key ID")
    aws_secret_access_key: Optional[SecretStr] = Field(
        None, description="AWS Secret Access Key"
    )

    # Environment Configuration
    environment: str = Field(
        default="development", description="Application environment"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    # Experiment Configuration
    default_gpu_instance: str = Field(
        default="g4dn.4xlarge",
        description="Default GPU instance type for experiments",
    )
    default_cpu_instance: str = Field(
        default="r6i.2xlarge",
        description="Default CPU instance type for experiments",
    )

    # Evaluation Thresholds (Model Training only)
    model_auc_threshold: float = Field(
        default=0.85, description="Minimum AUC threshold for model training"
    )
    model_logloss_threshold: float = Field(
        default=0.35, description="Maximum log loss threshold for model training"
    )
    model_calibration_error_threshold: float = Field(
        default=0.02, description="Maximum calibration error threshold"
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()