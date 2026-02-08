"""Experiment runner - orchestrates sequential experiment execution."""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Optional

from src.core.models import ExperimentConfig, ExperimentResult, ExperimentStatus
from src.evaluation.evaluator import evaluate_model_training
from src.experiment.queue import ExperimentQueue
from src.integrations.mlflow_client import MLflowClient
from src.k8s.pod_executor import PodExecutor

logger = logging.getLogger(__name__)

# 로그 디렉토리 상수
EXP_LOG_DIR = "exp-log"


class ExperimentRunner:
    """Orchestrates sequential experiment execution on a K8s pod."""

    DEFAULT_POLL_INTERVAL = 600  # 10분

    def __init__(
        self,
        pod_name: str,
        namespace: str = "tf-box",
        mlflow_experiment_name: Optional[str] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        self.pod_name = pod_name
        self.namespace = namespace
        self.mlflow_experiment_name = mlflow_experiment_name
        self.poll_interval = poll_interval
        self.pod_executor = PodExecutor()
        self.mlflow_client = MLflowClient()
        self.queue = ExperimentQueue()
        self._exp_number = 0  # 실험 번호 추적

    async def verify_pod(self) -> bool:
        """Verify that the target pod is running.

        Returns:
            True if pod is running

        Raises:
            RuntimeError: If pod is not found or not running
        """
        return await self.pod_executor.verify_pod_running(self.pod_name)

    async def run_all(
        self,
        configs: List[ExperimentConfig],
        stop_on_failure: bool = False,
    ) -> List[ExperimentResult]:
        """Run all experiments sequentially.

        Args:
            configs: List of experiment configurations
            stop_on_failure: Stop execution if an experiment fails

        Returns:
            List of experiment results
        """
        # Verify pod is running
        await self.verify_pod()

        # Add experiments to queue
        self.queue.add_batch(configs)
        total = len(configs)

        logger.info(f"Starting {total} experiments on pod '{self.pod_name}'")

        # Process queue
        position = 0
        while True:
            config = await self.queue.next()
            if config is None:
                break

            position += 1
            logger.info(f"[{position}/{total}] Running: {config.description}")

            result = await self._run_single(config, position, total)
            self.queue.record_result(result)

            if result.status == ExperimentStatus.FAILED and stop_on_failure:
                logger.warning(
                    f"Stopping batch due to failure: {config.experiment_id}"
                )
                break

        logger.info(
            f"Batch complete: {self.queue.completed_count}/{total} experiments finished"
        )
        return self.queue.results

    async def _reset_repo(self, repo_path: str) -> None:
        """Reset the git repo on the pod to clean state."""
        reset_script = f"cd {repo_path} && git checkout . && git clean -fd"
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=reset_script,
            timeout=60,
        )
        if not result["success"]:
            logger.warning(f"git reset warning: {result['stderr']}")
        else:
            logger.info(f"Reset repo: {repo_path}")

    async def _run_setup(self, config: ExperimentConfig) -> bool:
        """Run setup commands (code modifications) before training.

        Returns:
            True if setup succeeded
        """
        if not config.setup_commands:
            return True

        logger.info(f"Running setup commands for: {config.description}")
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=config.setup_commands,
            timeout=120,
        )
        if not result["success"]:
            logger.error(f"Setup failed: {result['stderr']}")
            return False
        return True

    async def _capture_git_diff(self, repo_path: str) -> str:
        """Capture git diff main on the pod."""
        diff_script = f"cd {repo_path} && git diff main"
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=diff_script,
            timeout=30,
        )
        if result["success"]:
            return result["stdout"]
        return ""

    async def _ensure_log_dir(self, repo_path: str) -> str:
        """Ensure log directory exists on the pod.

        Returns:
            Path to log directory
        """
        log_dir = f"{repo_path}/{EXP_LOG_DIR}"
        mkdir_script = f"mkdir -p {log_dir}"
        await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=mkdir_script,
            timeout=30,
        )
        return log_dir

    def _get_log_file_path(self, repo_path: str, exp_number: int) -> str:
        """Get log file path for an experiment.

        Args:
            repo_path: Path to the repo on pod
            exp_number: Experiment number (1-indexed)

        Returns:
            Full path to log file
        """
        return f"{repo_path}/{EXP_LOG_DIR}/exp{exp_number}.log"

    async def _fetch_log_file(self, log_path: str) -> str:
        """Fetch log file content from pod.

        Args:
            log_path: Path to log file on pod

        Returns:
            Log file content
        """
        cat_script = f"cat {log_path} 2>/dev/null || echo ''"
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=cat_script,
            timeout=60,
        )
        if result["success"]:
            return result["stdout"]
        return ""

    async def _check_marker(self, marker_path: str) -> bool:
        """Check if a marker file exists on the pod.

        Args:
            marker_path: Path to marker file

        Returns:
            True if marker exists
        """
        check_script = f"test -f {marker_path} && echo 'EXISTS' || echo 'NOT_EXISTS'"
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=check_script,
            timeout=30,
        )
        return result["success"] and "EXISTS" in result["stdout"]

    async def _wait_for_completion(
        self,
        log_file: str,
        position: int,
        total: int,
    ) -> str:
        """Poll for experiment completion.

        Args:
            log_file: Path to log file (used for marker file paths)
            position: Current experiment position
            total: Total experiments

        Returns:
            "completed" or "failed"
        """
        done_marker = f"{log_file}.done"
        failed_marker = f"{log_file}.failed"

        while True:
            done = await self._check_marker(done_marker)
            if done:
                return "completed"

            failed = await self._check_marker(failed_marker)
            if failed:
                return "failed"

            logger.info(
                f"[{position}/{total}] Experiment running... "
                f"next check in {self.poll_interval}s"
            )
            await asyncio.sleep(self.poll_interval)

    async def _run_single(
        self,
        config: ExperimentConfig,
        position: int,
        total: int,
    ) -> ExperimentResult:
        """Run a single experiment.

        Flow:
            1. git checkout . (reset to clean state)
            2. Run setup_commands (code modifications)
            3. git diff main (capture changes)
            4. Run training command
            5. Fetch MLflow metrics & evaluate

        Args:
            config: Experiment configuration
            position: Current position in batch (1-indexed)
            total: Total number of experiments

        Returns:
            Experiment result
        """
        started_at = datetime.utcnow()
        start_time = time.time()
        git_diff = ""

        try:
            # Step 1: Reset repo to clean state
            await self._reset_repo(config.repo_path)

            # Step 2: Run setup commands (code modifications)
            if config.setup_commands:
                setup_ok = await self._run_setup(config)
                if not setup_ok:
                    duration = time.time() - start_time
                    return ExperimentResult(
                        config=config,
                        status=ExperimentStatus.FAILED,
                        duration_seconds=round(duration, 1),
                        stderr="Setup commands failed",
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                    )

            # Step 3: Capture git diff
            git_diff = await self._capture_git_diff(config.repo_path)
            if git_diff:
                logger.info(
                    f"[{position}/{total}] Code changes ({len(git_diff.splitlines())} lines diff)"
                )
            else:
                logger.info(f"[{position}/{total}] No code changes (baseline)")

            # Step 4: Ensure log directory exists and get log file path
            self._exp_number += 1
            await self._ensure_log_dir(config.repo_path)
            log_file = self._get_log_file_path(config.repo_path, self._exp_number)
            logger.info(f"[{position}/{total}] Log file: {log_file}")

            # Step 5: Execute training command in background with nohup
            # 백그라운드로 실행하고 완료 시 마커 파일 생성
            bg_script = f"""
nohup bash -c '
  {config.training_command}
  if [ $? -eq 0 ]; then
    touch {log_file}.done
  else
    touch {log_file}.failed
  fi
' > {log_file} 2>&1 &
echo $!
"""
            exec_result = await self.pod_executor.execute_on_pod(
                pod_name=self.pod_name,
                script=bg_script,
                timeout=60,  # 백그라운드 시작은 빠르게 완료
            )

            if not exec_result["success"]:
                duration = time.time() - start_time
                return ExperimentResult(
                    config=config,
                    status=ExperimentStatus.FAILED,
                    duration_seconds=round(duration, 1),
                    stderr=f"Failed to start experiment: {exec_result['stderr']}",
                    git_diff=git_diff,
                    log_file=log_file,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

            pid = exec_result["stdout"].strip()
            logger.info(f"[{position}/{total}] Started background process (PID: {pid})")

            # Step 6: Poll for completion
            completion_status = await self._wait_for_completion(
                log_file, position, total
            )

            duration = time.time() - start_time
            completed_at = datetime.utcnow()

            # Fetch log file content
            log_content = await self._fetch_log_file(log_file)

            if completion_status == "failed":
                logger.error(
                    f"[{position}/{total}] FAILED: {config.description}"
                )
                logger.error(f"[{position}/{total}] Log saved to: {log_file}")
                return ExperimentResult(
                    config=config,
                    status=ExperimentStatus.FAILED,
                    duration_seconds=round(duration, 1),
                    stdout=log_content,  # 로그 파일 내용 사용
                    stderr=exec_result["stderr"],
                    git_diff=git_diff,
                    log_file=log_file,
                    started_at=started_at,
                    completed_at=completed_at,
                )

            # Step 7: Training succeeded - fetch metrics from MLflow
            metrics = {}
            if self.mlflow_experiment_name and self.mlflow_client.is_available:
                metrics = await self.mlflow_client.get_latest_run_metrics(
                    self.mlflow_experiment_name
                )

                # Set description tag on the latest run
                if metrics:
                    runs = await self.mlflow_client.search_runs(
                        self.mlflow_experiment_name, max_results=1
                    )
                    if runs:
                        await self.mlflow_client.set_run_tags(
                            runs[0]["run_id"],
                            {
                                "description": config.description,
                                "experiment_id": config.experiment_id,
                                "batch_position": str(position),
                            },
                        )

            # Evaluate metrics
            evaluation = None
            if metrics:
                evaluation = evaluate_model_training(metrics)

            status = ExperimentStatus.COMPLETED
            logger.info(
                f"[{position}/{total}] COMPLETED: {config.description} "
                f"(duration: {duration:.0f}s)"
            )
            logger.info(f"[{position}/{total}] Log saved to: {log_file}")

            return ExperimentResult(
                config=config,
                status=status,
                metrics=metrics,
                duration_seconds=round(duration, 1),
                stdout=log_content,  # 로그 파일 내용 사용
                stderr=exec_result["stderr"],
                git_diff=git_diff,
                log_file=log_file,
                evaluation=evaluation,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{position}/{total}] ERROR: {config.description}: {str(e)}"
            )
            return ExperimentResult(
                config=config,
                status=ExperimentStatus.FAILED,
                duration_seconds=round(duration, 1),
                stderr=str(e),
                git_diff=git_diff,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    def get_summary(self) -> dict:
        """Get summary of all experiment results."""
        return self.queue.get_summary()

    def get_markdown_report(
        self,
        batch_id: str,
        utc_ymdh: str,
        model_name: str = "model",
    ) -> str:
        """Get markdown formatted report of experiment results.

        Args:
            batch_id: Batch identifier
            utc_ymdh: UTC timestamp of training data
            model_name: Name of the model being experimented on

        Returns:
            Markdown formatted report string
        """
        return self.queue.generate_markdown_report(
            batch_id=batch_id,
            pod_name=self.pod_name,
            utc_ymdh=utc_ymdh,
            model_name=model_name,
        )
