"""Unit tests for ExperimentQueue."""

import pytest
import asyncio
from src.core.models import ExperimentConfig, ExperimentResult, ExperimentStatus


class TestExperimentQueue:
    """Test ExperimentQueue class."""

    def _make_config(self, id: str = "exp-001", desc: str = "Test") -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id=id,
            description=desc,
            training_command="python train.py",
        )

    def _make_result(self, config: ExperimentConfig, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> ExperimentResult:
        return ExperimentResult(
            config=config,
            status=status,
            metrics={"auc": 0.9, "logloss": 0.3, "calibration_error": 0.01},
        )

    def test_add_batch(self):
        from src.experiment.queue import ExperimentQueue

        queue = ExperimentQueue()
        configs = [self._make_config(f"exp-{i}") for i in range(3)]
        count = queue.add_batch(configs)

        assert count == 3
        assert queue.pending_count == 3
        assert queue.total_count == 3
        assert queue.completed_count == 0

    @pytest.mark.asyncio
    async def test_next(self):
        from src.experiment.queue import ExperimentQueue

        queue = ExperimentQueue()
        configs = [self._make_config(f"exp-{i}", f"Test {i}") for i in range(2)]
        queue.add_batch(configs)

        first = await queue.next()
        assert first is not None
        assert first.experiment_id == "exp-0"
        assert queue.current == first

        second = await queue.next()
        assert second is not None
        assert second.experiment_id == "exp-1"

        third = await queue.next()
        assert third is None
        assert queue.current is None

    def test_record_result(self):
        from src.experiment.queue import ExperimentQueue

        queue = ExperimentQueue()
        config = self._make_config()
        queue.add_batch([config])

        result = self._make_result(config)
        queue.record_result(result)

        assert queue.completed_count == 1
        assert len(queue.results) == 1

    def test_get_summary_empty(self):
        from src.experiment.queue import ExperimentQueue

        queue = ExperimentQueue()
        summary = queue.get_summary()
        assert summary["total"] == 0
        assert summary["completed"] == 0
        assert summary["failed"] == 0

    def test_get_summary_with_results(self):
        from src.experiment.queue import ExperimentQueue

        queue = ExperimentQueue()
        configs = [self._make_config(f"exp-{i}") for i in range(3)]
        queue.add_batch(configs)

        # Record 2 successes, 1 failure
        queue.record_result(self._make_result(configs[0], ExperimentStatus.COMPLETED))
        queue.record_result(self._make_result(configs[1], ExperimentStatus.FAILED))
        queue.record_result(self._make_result(configs[2], ExperimentStatus.COMPLETED))

        summary = queue.get_summary()
        assert summary["total"] == 3
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert "best_experiment" in summary
