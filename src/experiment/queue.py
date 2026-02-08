"""Experiment queue for managing sequential experiment execution."""

import asyncio
import logging
from typing import List, Optional

from src.core.models import ExperimentConfig, ExperimentResult, ExperimentStatus
from src.experiment.report_generator import generate_markdown_report

logger = logging.getLogger(__name__)


class ExperimentQueue:
    """Queue for managing experiment execution order."""

    def __init__(self):
        self._queue: asyncio.Queue[ExperimentConfig] = asyncio.Queue()
        self._results: List[ExperimentResult] = []
        self._current: Optional[ExperimentConfig] = None
        self._total: int = 0

    def add_batch(self, configs: List[ExperimentConfig]) -> int:
        """Add a batch of experiments to the queue.

        Args:
            configs: List of experiment configurations

        Returns:
            Number of experiments added
        """
        for config in configs:
            self._queue.put_nowait(config)
        self._total += len(configs)
        logger.info(f"Added {len(configs)} experiments to queue (total: {self._total})")
        return len(configs)

    async def next(self) -> Optional[ExperimentConfig]:
        """Get the next experiment from the queue.

        Returns:
            Next experiment config, or None if queue is empty
        """
        if self._queue.empty():
            self._current = None
            return None
        self._current = self._queue.get_nowait()
        return self._current

    @property
    def current(self) -> Optional[ExperimentConfig]:
        """Get the currently running experiment."""
        return self._current

    def record_result(self, result: ExperimentResult) -> None:
        """Record a completed experiment result.

        Args:
            result: The experiment result to record
        """
        self._results.append(result)
        self._current = None
        logger.info(
            f"Recorded result for {result.config.experiment_id}: {result.status}"
        )

    @property
    def pending_count(self) -> int:
        """Number of pending experiments."""
        return self._queue.qsize()

    @property
    def completed_count(self) -> int:
        """Number of completed experiments."""
        return len(self._results)

    @property
    def total_count(self) -> int:
        """Total number of experiments."""
        return self._total

    @property
    def results(self) -> List[ExperimentResult]:
        """Get all results."""
        return list(self._results)

    def get_summary(self) -> dict:
        """Get a summary of all experiment results.

        Returns:
            Summary dictionary with counts, results, and best experiment
        """
        succeeded = [r for r in self._results if r.status == ExperimentStatus.COMPLETED]
        failed = [r for r in self._results if r.status == ExperimentStatus.FAILED]

        summary = {
            "total": self._total,
            "completed": len(succeeded),
            "failed": len(failed),
            "pending": self.pending_count,
            "results": [],
        }

        for result in self._results:
            entry = {
                "experiment_id": result.config.experiment_id,
                "description": result.config.description,
                "status": result.status,
                "duration_seconds": result.duration_seconds,
                "metrics": result.metrics,
                "git_diff_lines": len(result.git_diff.splitlines()) if result.git_diff else 0,
                "log_file": result.log_file,
            }
            if result.evaluation:
                entry["passed"] = result.evaluation.get("passed", False)
                entry["score"] = result.evaluation.get("score", 0)
            summary["results"].append(entry)

        # Find best result by AUC
        if succeeded:
            best = max(succeeded, key=lambda r: r.metrics.get("auc", 0))
            summary["best_experiment"] = {
                "experiment_id": best.config.experiment_id,
                "description": best.config.description,
                "metrics": best.metrics,
                "score": best.evaluation.get("score", 0) if best.evaluation else 0,
            }

        return summary

    def generate_markdown_report(
        self,
        batch_id: str,
        pod_name: str,
        utc_ymdh: str,
        model_name: str = "model",
    ) -> str:
        """Generate a markdown report of experiment results with MLflow metrics comparison.

        Args:
            batch_id: Batch identifier
            pod_name: K8s pod name used for experiments
            utc_ymdh: UTC timestamp of training data
            model_name: Name of the model being experimented on

        Returns:
            Markdown formatted report string
        """
        # Get the summary data
        summary = self.get_summary()

        # Convert results to the format expected by report generator
        results = []
        for result in self._results:
            result_dict = {
                "description": result.config.description,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                "duration_seconds": result.duration_seconds,
                "metrics": result.metrics,
                "git_diff_lines": len(result.git_diff.splitlines()) if result.git_diff else 0,
                "config": {
                    "parameters": result.config.parameters
                }
            }
            results.append(result_dict)

        # Generate the markdown report
        return generate_markdown_report(
            batch_id=batch_id,
            pod_name=pod_name,
            utc_ymdh=utc_ymdh,
            results=results,
            model_name=model_name,
            best_experiment=summary.get("best_experiment"),
        )
