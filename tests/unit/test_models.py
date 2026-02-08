"""Unit tests for data models."""

import pytest
from datetime import datetime
from src.core.models import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentBatch,
    ExperimentStatus,
)


class TestExperimentStatus:
    def test_enum_values(self):
        assert ExperimentStatus.PENDING == "pending"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.FAILED == "failed"


class TestExperimentConfig:
    def test_create_config(self):
        config = ExperimentConfig(
            experiment_id="exp-001",
            description="Test experiment",
            training_command="python train.py",
        )
        assert config.experiment_id == "exp-001"
        assert config.description == "Test experiment"
        assert config.parameters == {}

    def test_create_config_with_parameters(self):
        config = ExperimentConfig(
            experiment_id="exp-002",
            description="Test with params",
            training_command="python train.py --lr 0.01",
            parameters={"lr": 0.01, "epochs": 10},
        )
        assert config.parameters["lr"] == 0.01
        assert config.parameters["epochs"] == 10


class TestExperimentResult:
    def test_create_result(self):
        config = ExperimentConfig(
            experiment_id="exp-001",
            description="Test",
            training_command="python train.py",
        )
        result = ExperimentResult(
            config=config,
            status=ExperimentStatus.COMPLETED,
            metrics={"auc": 0.9},
        )
        assert result.status == "completed"
        assert result.metrics["auc"] == 0.9
        assert result.stdout == ""
        assert result.stderr == ""

    def test_create_failed_result(self):
        config = ExperimentConfig(
            experiment_id="exp-002",
            description="Failing test",
            training_command="python fail.py",
        )
        result = ExperimentResult(
            config=config,
            status=ExperimentStatus.FAILED,
            stderr="Error: something went wrong",
        )
        assert result.status == "failed"
        assert "Error" in result.stderr


class TestExperimentBatch:
    def test_create_batch(self):
        configs = [
            ExperimentConfig(
                experiment_id=f"exp-{i}",
                description=f"Experiment {i}",
                training_command=f"python train.py --run {i}",
            )
            for i in range(3)
        ]
        batch = ExperimentBatch(
            batch_id="batch-001",
            pod_name="test-pod",
            experiments=configs,
        )
        assert batch.batch_id == "batch-001"
        assert len(batch.experiments) == 3
        assert batch.namespace == "tf-box"
        assert len(batch.results) == 0

    def test_batch_default_namespace(self):
        batch = ExperimentBatch(
            batch_id="batch-002",
            pod_name="test-pod",
        )
        assert batch.namespace == "tf-box"
