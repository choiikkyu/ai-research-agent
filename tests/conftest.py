"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import ExperimentConfig, ExperimentResult, ExperimentStatus


@pytest.fixture
def sample_experiment_config():
    """Sample ExperimentConfig for testing."""
    return ExperimentConfig(
        experiment_id="test-exp-001",
        description="Test baseline model training",
        training_command="cd /app && python train.py --model baseline",
        parameters={"lr": 0.001, "epochs": 10},
    )


@pytest.fixture
def sample_experiment_configs():
    """Multiple ExperimentConfigs for batch testing."""
    return [
        ExperimentConfig(
            experiment_id="test-exp-001",
            description="Baseline model",
            training_command="python train.py",
        ),
        ExperimentConfig(
            experiment_id="test-exp-002",
            description="Higher learning rate",
            training_command="python train.py --lr 0.01",
        ),
        ExperimentConfig(
            experiment_id="test-exp-003",
            description="More layers",
            training_command="python train.py --layers 4",
        ),
    ]


@pytest.fixture
def passing_metrics():
    """Metrics that pass all thresholds."""
    return {
        "auc": 0.9,
        "logloss": 0.3,
        "calibration_error": 0.01,
    }


@pytest.fixture
def failing_metrics():
    """Metrics that fail all thresholds."""
    return {
        "auc": 0.7,
        "logloss": 0.5,
        "calibration_error": 0.05,
    }


@pytest.fixture
def mixed_metrics():
    """Metrics with mixed pass/fail."""
    return {
        "auc": 0.9,
        "logloss": 0.5,
        "calibration_error": 0.01,
    }


@pytest.fixture
def mock_pod_executor():
    """Mock PodExecutor."""
    with patch("src.k8s.pod_executor.PodExecutor") as mock:
        mock_executor = AsyncMock()
        mock_executor.verify_pod_running.return_value = True
        mock_executor.execute_on_pod.return_value = {
            "success": True,
            "stdout": "Training completed successfully",
            "stderr": "",
            "returncode": 0,
        }
        mock_executor.get_pod_status.return_value = "Running"
        mock.return_value = mock_executor
        yield mock_executor


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflowClient."""
    with patch("src.integrations.mlflow_client.MLflowClient") as mock:
        mock_client = AsyncMock()
        mock_client.is_available = True
        mock_client.get_latest_run_metrics.return_value = {
            "auc": 0.9,
            "logloss": 0.3,
            "calibration_error": 0.01,
        }
        mock_client.search_runs.return_value = [
            {"run_id": "run-123", "status": "COMPLETED", "metrics": {"auc": 0.9}}
        ]
        mock_client.set_run_tags.return_value = True
        mock.return_value = mock_client
        yield mock_client


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (with mocks)")
