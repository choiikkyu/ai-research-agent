"""MLflow client wrapper for experiment tracking."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient as _MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    _MlflowClient = None

from src.core.config import settings

logger = logging.getLogger(__name__)


class MLflowClient:
    """MLflow client wrapper with error handling and fallbacks."""

    def __init__(self, tracking_uri: str = None):
        """Initialize MLflow client."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, using mock client")
            self._available = False
            self.client = None
            return

        self._available = True

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI set to: {tracking_uri}")

            self.client = _MlflowClient()
            logger.info("MLflow client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            self._available = False
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if MLflow is available."""
        return self._available

    async def get_latest_run_metrics(
        self, experiment_name: str
    ) -> Dict[str, float]:
        """Get metrics from the latest run of an experiment."""
        if not self.is_available:
            logger.warning("MLflow not available, returning empty metrics")
            return {}

        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.warning(f"Experiment not found: {experiment_name}")
                return {}

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs and len(runs) > 0:
                run = runs[0]
                metrics = run.data.metrics
                metrics_dict = {k: float(v) for k, v in metrics.items()}

                logger.info(
                    f"Retrieved {len(metrics_dict)} metrics from {experiment_name} "
                    f"(run_id: {run.info.run_id})"
                )

                key_metrics = ["auc", "test_auc", "logloss", "test_logloss", "calibration_error"]
                for km in key_metrics:
                    if km in metrics_dict:
                        logger.info(f"  {km}: {metrics_dict[km]:.4f}")

                return metrics_dict
            else:
                logger.warning(f"No runs found for experiment: {experiment_name}")
                return {}

        except Exception as e:
            logger.error(f"Failed to fetch latest run metrics: {e}")
            return {}

    async def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics from a specific run."""
        if not self.is_available:
            logger.warning("MLflow not available, returning empty metrics")
            return {}

        try:
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            metrics_dict = {k: float(v) for k, v in metrics.items()}
            logger.info(f"Retrieved {len(metrics_dict)} metrics from run {run_id}")
            return metrics_dict

        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return {}

    async def search_runs(
        self,
        experiment_name: str,
        max_results: int = 10,
        order_by: list[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for runs in an experiment."""
        if not self.is_available:
            return []

        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.warning(f"Experiment not found: {experiment_name}")
                return []

            if order_by is None:
                order_by = ["start_time DESC"]

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=order_by,
            )

            runs_info = []
            for run in runs:
                runs_info.append({
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time,
                    "status": run.info.status,
                    "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                    "params": run.data.params,
                })

            logger.info(f"Found {len(runs_info)} runs for {experiment_name}")
            return runs_info

        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []

    async def log_metrics(
        self, run_id: str, metrics: Dict[str, float]
    ) -> bool:
        """Log metrics to an existing run."""
        if not self.is_available:
            logger.warning("MLflow not available, cannot log metrics")
            return False

        try:
            with mlflow.start_run(run_id=run_id, nested=True):
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

            logger.info(f"Logged {len(metrics)} metrics to run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log metrics to run {run_id}: {e}")
            return False

    async def set_run_tags(
        self, run_id: str, tags: Dict[str, str]
    ) -> bool:
        """Set tags on an MLflow run.

        Args:
            run_id: MLflow run ID
            tags: Dictionary of tag names to values

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            logger.warning("MLflow not available, cannot set tags")
            return False

        try:
            for key, value in tags.items():
                self.client.set_tag(run_id, key, value)

            logger.info(f"Set {len(tags)} tags on run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set tags on run {run_id}: {e}")
            return False

    async def create_experiment_run(
        self,
        experiment_name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Create a new run in an experiment.

        Args:
            experiment_name: Name of the MLflow experiment
            description: Description for the run
            tags: Optional tags to set on the run

        Returns:
            Run ID if successful, None otherwise
        """
        if not self.is_available:
            logger.warning("MLflow not available, cannot create run")
            return None

        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id

            run = self.client.create_run(
                experiment_id=experiment_id,
                tags=tags or {},
            )

            if description:
                self.client.set_tag(run.info.run_id, "mlflow.note.content", description)

            logger.info(
                f"Created run {run.info.run_id} in experiment {experiment_name}"
            )
            return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to create run in {experiment_name}: {e}")
            return None
