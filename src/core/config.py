"""Configuration management for AI Research Agent."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://mlflow.ai.svc.cluster.local:5000",
        description="MLflow Tracking Server URI",
    )

    # Kubernetes Configuration
    k8s_namespace: str = Field(
        default="tf-box", description="Kubernetes namespace for pods"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Evaluation Thresholds
    model_auc_threshold: float = Field(
        default=0.85, description="Minimum AUC threshold for model training"
    )
    model_logloss_threshold: float = Field(
        default=0.35, description="Maximum log loss threshold for model training"
    )
    model_calibration_error_threshold: float = Field(
        default=0.02, description="Maximum calibration error threshold"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
