"""Mock tests for MLflow client."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "mlflow_client",
    Path(__file__).parent.parent.parent / "src" / "integrations" / "mlflow_client.py",
)
mlflow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlflow_module)

MLflowClient = mlflow_module.MLflowClient


class TestMLflowClient:
    """Test MLflowClient with mocked MLflow API."""

    @pytest.mark.asyncio
    async def test_get_latest_run_metrics_success(self):
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp_123"

        mock_run = Mock()
        mock_run.info.run_id = "run_456"
        mock_run.data.metrics = {
            "auc": 0.85,
            "logloss": 0.3,
            "calibration_error": 0.01,
        }

        mock_client_instance = Mock()
        mock_client_instance.get_experiment_by_name.return_value = mock_experiment
        mock_client_instance.search_runs.return_value = [mock_run]

        with patch.object(mlflow_module, "_MlflowClient", return_value=mock_client_instance):
            client = MLflowClient()
            result = await client.get_latest_run_metrics("test_experiment")

            assert result["auc"] == 0.85
            assert result["logloss"] == 0.3
            assert result["calibration_error"] == 0.01

    @pytest.mark.asyncio
    async def test_get_latest_run_metrics_no_runs(self):
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp_123"

        mock_client_instance = Mock()
        mock_client_instance.get_experiment_by_name.return_value = mock_experiment
        mock_client_instance.search_runs.return_value = []

        with patch.object(mlflow_module, "_MlflowClient", return_value=mock_client_instance):
            client = MLflowClient()
            result = await client.get_latest_run_metrics("test_experiment")
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_latest_run_metrics_experiment_not_found(self):
        mock_client_instance = Mock()
        mock_client_instance.get_experiment_by_name.return_value = None

        with patch.object(mlflow_module, "_MlflowClient", return_value=mock_client_instance):
            client = MLflowClient()
            result = await client.get_latest_run_metrics("nonexistent")
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_run_metrics_success(self):
        mock_run = Mock()
        mock_run.data.metrics = {"auc": 0.9, "logloss": 0.25}

        mock_mlflow = Mock()
        mock_mlflow.get_run.return_value = mock_run

        with patch.object(mlflow_module, "mlflow", mock_mlflow):
            client = MLflowClient()
            result = await client.get_run_metrics("run_789")

            assert result["auc"] == 0.9
            assert result["logloss"] == 0.25

    @pytest.mark.asyncio
    async def test_get_run_metrics_error(self):
        mock_mlflow = Mock()
        mock_mlflow.get_run.side_effect = Exception("Run not found")

        with patch.object(mlflow_module, "mlflow", mock_mlflow):
            client = MLflowClient()
            result = await client.get_run_metrics("invalid_run")
            assert result == {}

    @pytest.mark.asyncio
    async def test_search_runs_success(self):
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp_123"

        mock_run1 = Mock()
        mock_run1.info.run_id = "run_1"
        mock_run1.info.start_time = 1234567890
        mock_run1.info.status = "COMPLETED"
        mock_run1.data.metrics = {"auc": 0.85}
        mock_run1.data.params = {"lr": "0.01"}

        mock_client_instance = Mock()
        mock_client_instance.get_experiment_by_name.return_value = mock_experiment
        mock_client_instance.search_runs.return_value = [mock_run1]

        with patch.object(mlflow_module, "_MlflowClient", return_value=mock_client_instance):
            client = MLflowClient()
            results = await client.search_runs("test_experiment", max_results=10)

            assert len(results) == 1
            assert results[0]["run_id"] == "run_1"

    @pytest.mark.asyncio
    async def test_set_run_tags_success(self):
        mock_client_instance = Mock()

        with patch.object(mlflow_module, "_MlflowClient", return_value=mock_client_instance):
            client = MLflowClient()
            result = await client.set_run_tags("run_123", {"tag1": "value1"})

            assert result is True
            mock_client_instance.set_tag.assert_called_once_with("run_123", "tag1", "value1")

    @pytest.mark.asyncio
    async def test_set_run_tags_unavailable(self):
        with patch.object(mlflow_module, "MLFLOW_AVAILABLE", False):
            client = MLflowClient()
            result = await client.set_run_tags("run_123", {"tag1": "value1"})
            assert result is False

    def test_is_available_property(self):
        with patch.object(mlflow_module, "MLFLOW_AVAILABLE", True):
            mock_client_instance = Mock()
            with patch.object(mlflow_module, "_MlflowClient", return_value=mock_client_instance):
                client = MLflowClient()
                assert client.is_available is True

        with patch.object(mlflow_module, "MLFLOW_AVAILABLE", False):
            client = MLflowClient()
            assert client.is_available is False
