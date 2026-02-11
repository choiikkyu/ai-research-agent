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

        # Validate pod environment with first experiment config
        if configs:
            await self._validate_pod_environment(configs[0])

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

    async def _cleanup_ray_session(self) -> None:
        """Clean up any existing Ray sessions on the pod."""
        cleanup_script = """
# Kill any existing Ray processes
pkill -9 -f ray 2>/dev/null || true

# Remove Ray temp directories
rm -rf /tmp/ray 2>/dev/null || true

# Wait for cleanup
sleep 2

echo "Ray cleanup completed"
"""
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=cleanup_script,
            timeout=30,
        )
        if result["success"]:
            logger.info("Ray session cleaned up successfully")
        else:
            logger.warning(f"Ray cleanup warning: {result['stderr']}")

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

    async def _save_diff_to_disk(
        self,
        diff_content: str,
        diff_file_path: str
    ) -> bool:
        """Save git diff content to a file on the pod.

        Args:
            diff_content: Git diff output to save
            diff_file_path: Path where diff should be saved

        Returns:
            True if save succeeded, False otherwise
        """
        # Handle empty diff case
        content_to_save = diff_content if diff_content else "No code changes (baseline experiment)\n"

        # Use heredoc to safely write diff content (handles special characters)
        save_script = f"""cat > {diff_file_path} << 'DIFF_EOF'
{content_to_save}
DIFF_EOF
"""

        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=save_script,
            timeout=30,
        )

        if not result["success"]:
            logger.warning(f"Failed to save diff file: {diff_file_path} - {result['stderr']}")
            return False

        logger.info(f"Saved diff to: {diff_file_path}")
        return True

    async def _execute_kubectl_direct(
        self,
        command: str,
        timeout: int = 300,
    ) -> dict:
        """Execute command directly via kubectl without intermediate script file.

        This bypasses PodExecutor to avoid quote escaping issues with complex commands.

        Args:
            command: Shell command to execute on pod
            timeout: Command timeout in seconds

        Returns:
            Dict with success, stdout, stderr, returncode
        """
        kubectl_cmd = [
            "kubectl", "exec", "-n", self.namespace, self.pod_name,
            "--", "bash", "-c", command
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *kubectl_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8") if stdout else "",
                "stderr": stderr.decode("utf-8") if stderr else "",
                "returncode": process.returncode,
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
            }
        except Exception as e:
            logger.error(f"kubectl exec failed: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    async def _ensure_log_dir(self, repo_path: str) -> str:
        """Ensure log directory exists on the pod.

        Returns:
            Path to log directory

        Raises:
            RuntimeError: If directory creation fails
        """
        log_dir = f"{repo_path}/{EXP_LOG_DIR}"
        mkdir_command = f"mkdir -p {log_dir} && echo 'SUCCESS'"

        result = await self._execute_kubectl_direct(
            command=mkdir_command,
            timeout=30,
        )

        if not result["success"] or "SUCCESS" not in result["stdout"]:
            raise RuntimeError(
                f"Failed to create log directory '{log_dir}': {result.get('stderr', 'Unknown error')}"
            )

        logger.info(f"Log directory ready: {log_dir}")
        return log_dir

    async def _validate_pod_environment(self, config: ExperimentConfig) -> None:
        """Validate that the pod environment is ready for experiments.

        Args:
            config: Experiment configuration to validate against

        Raises:
            RuntimeError: If environment is not suitable for experiments
        """
        validation_script = f"""
# Check if repo path exists
if [ ! -d "{config.repo_path}" ]; then
    echo "ERROR: repo_path does not exist: {config.repo_path}"
    exit 1
fi

# Check if we can write to repo path
if [ ! -w "{config.repo_path}" ]; then
    echo "ERROR: repo_path is not writable: {config.repo_path}"
    exit 1
fi

# Check if Python is available
if ! which python > /dev/null 2>&1; then
    echo "ERROR: Python not found in PATH"
    exit 1
fi

# Check if git is available (for setup commands)
if ! which git > /dev/null 2>&1; then
    echo "WARNING: git not found in PATH"
fi

echo "VALIDATION_SUCCESS"
"""

        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=validation_script,
            timeout=30,
        )

        if not result["success"] or "VALIDATION_SUCCESS" not in result["stdout"]:
            raise RuntimeError(
                f"Pod environment validation failed:\n{result.get('stdout', '')}"
            )

        logger.info(f"Pod environment validated successfully")

    def _get_log_file_path(self, repo_path: str, exp_number: int) -> str:
        """Get log file path for an experiment.

        Args:
            repo_path: Path to the repo on pod
            exp_number: Experiment number (1-indexed)

        Returns:
            Full path to log file
        """
        return f"{repo_path}/{EXP_LOG_DIR}/exp{exp_number}.log"

    def _get_diff_file_path(self, repo_path: str, exp_number: int) -> str:
        """Get diff file path for an experiment.

        Args:
            repo_path: Path to the repo on pod
            exp_number: Experiment number (1-indexed)

        Returns:
            Full path to diff file
        """
        return f"{repo_path}/{EXP_LOG_DIR}/exp{exp_number}.diff"

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

    async def _get_log_size(self, log_file: str) -> int:
        """Get size of log file in bytes.

        Args:
            log_file: Path to log file

        Returns:
            Size in bytes, or 0 if file doesn't exist
        """
        # Use wc -c which is more portable than stat
        size_script = f"wc -c < {log_file} 2>/dev/null || echo 0"
        result = await self.pod_executor.execute_on_pod(
            pod_name=self.pod_name,
            script=size_script,
            timeout=10,
        )
        if result["success"]:
            try:
                return int(result["stdout"].strip())
            except ValueError:
                return 0
        return 0

    async def _wait_for_completion(
        self,
        log_file: str,
        position: int,
        total: int,
    ) -> str:
        """Poll for experiment completion with fast failure detection.

        Args:
            log_file: Path to log file (used for marker file paths)
            position: Current experiment position
            total: Total experiments

        Returns:
            "completed" or "failed"
        """
        done_marker = f"{log_file}.done"
        failed_marker = f"{log_file}.failed"

        # Fast polling for first 5 minutes (check every 30s)
        fast_poll_duration = 300  # 5 minutes
        fast_poll_interval = 30  # 30 seconds
        elapsed_time = 0
        last_log_size = 0
        no_progress_count = 0

        while True:
            # Check markers
            done = await self._check_marker(done_marker)
            if done:
                return "completed"

            failed = await self._check_marker(failed_marker)
            if failed:
                return "failed"

            # Check log size to detect progress
            current_log_size = await self._get_log_size(log_file)

            # Detect stalled training (no log growth for 3 consecutive checks)
            if current_log_size > 0 and current_log_size == last_log_size:
                no_progress_count += 1
                if no_progress_count >= 3 and elapsed_time > 180:  # No progress for 3 checks after 3 minutes
                    logger.error(
                        f"[{position}/{total}] Training appears stalled (no log growth). "
                        f"Log size: {current_log_size} bytes"
                    )
                    # Check if process is still running
                    return "failed"
            else:
                no_progress_count = 0

            last_log_size = current_log_size

            # Use fast polling initially, then switch to normal polling
            if elapsed_time < fast_poll_duration:
                poll_interval = fast_poll_interval
                logger.info(
                    f"[{position}/{total}] Training in progress... "
                    f"(log: {current_log_size} bytes, next check in {poll_interval}s)"
                )
            else:
                poll_interval = self.poll_interval
                logger.info(
                    f"[{position}/{total}] Training in progress... "
                    f"(log: {current_log_size} bytes, next check in {poll_interval}s)"
                )

            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval

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
            # Step 0: Clean up old marker files and log files to prevent false completion detection
            self._exp_number += 1
            await self._ensure_log_dir(config.repo_path)
            log_file = self._get_log_file_path(config.repo_path, self._exp_number)
            diff_file = self._get_diff_file_path(config.repo_path, self._exp_number)
            cleanup_script = f"rm -f {log_file} {log_file}.done {log_file}.failed {diff_file}"
            await self.pod_executor.execute_on_pod(
                pod_name=self.pod_name,
                script=cleanup_script,
                timeout=30,
            )
            logger.info(f"[{position}/{total}] Cleaned up old markers for: {log_file}")

            # Step 0.5: Clean up Ray sessions to prevent GCS connection issues
            logger.info(f"[{position}/{total}] Cleaning up Ray sessions...")
            await self._cleanup_ray_session()

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
                        diff_file=diff_file,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                    )

            # Step 3: Capture git diff and save to disk
            git_diff = await self._capture_git_diff(config.repo_path)

            # Save diff to disk (non-blocking - log warning on failure)
            await self._save_diff_to_disk(git_diff, diff_file)

            if git_diff:
                logger.info(
                    f"[{position}/{total}] Code changes ({len(git_diff.splitlines())} lines diff)"
                )
            else:
                logger.info(f"[{position}/{total}] No code changes (baseline)")

            # Step 4: Log file path (already set in Step 0)
            logger.info(f"[{position}/{total}] Log file: {log_file}")

            # Step 5: Execute training command directly in background with nohup
            # Use direct kubectl exec to avoid quote escaping issues
            # This approach matches the successful manual execution
            bg_command = f'''
nohup bash -c "
  set -e
  {config.training_command}
  exit_code=\\$?
  if [ \\$exit_code -eq 0 ]; then
    touch {log_file}.done
  else
    touch {log_file}.failed
  fi
  exit \\$exit_code
" > {log_file} 2>&1 &
echo $!
'''

            exec_result = await self._execute_kubectl_direct(
                command=bg_command,
                timeout=60,
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
                    diff_file=diff_file,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

            pid = exec_result["stdout"].strip()
            logger.info(f"[{position}/{total}] Started background process (PID: {pid})")

            # Step 5.5: Verify process started successfully
            # Wait 5 seconds, then check if process is still running
            await asyncio.sleep(5)

            check_pid_command = f"ps -p {pid} > /dev/null 2>&1 && echo 'RUNNING' || echo 'NOT_RUNNING'"
            pid_check_result = await self._execute_kubectl_direct(
                command=check_pid_command,
                timeout=10,
            )

            if "NOT_RUNNING" in pid_check_result.get("stdout", ""):
                # Process died immediately - fetch log for error details
                log_content = await self._fetch_log_file(log_file)
                duration = time.time() - start_time

                logger.error(f"[{position}/{total}] Process died within 5 seconds")
                logger.error(f"[{position}/{total}] Log content:\n{log_content[-1000:] if log_content else 'No log available'}")

                return ExperimentResult(
                    config=config,
                    status=ExperimentStatus.FAILED,
                    duration_seconds=round(duration, 1),
                    stderr=f"Training process died within 5 seconds. Check log for details.",
                    stdout=log_content,
                    git_diff=git_diff,
                    log_file=log_file,
                    diff_file=diff_file,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

            logger.info(f"[{position}/{total}] Process verified running (PID: {pid})")

            # Step 6: Poll for completion
            completion_status = await self._wait_for_completion(
                log_file, position, total
            )

            duration = time.time() - start_time
            completed_at = datetime.utcnow()

            # Fetch log file content
            log_content = await self._fetch_log_file(log_file)

            if completion_status == "failed":
                # Extract error information from log
                error_summary = "Training failed"
                if log_content:
                    # Get last 50 lines to find error
                    log_lines = log_content.split("\n")
                    last_lines = log_lines[-50:] if len(log_lines) > 50 else log_lines

                    # Look for common error patterns
                    for line in reversed(last_lines):
                        if any(pattern in line for pattern in ["Error", "ERROR", "Exception", "FAILED", "Failed to connect"]):
                            error_summary = line.strip()
                            break

                logger.error(
                    f"[{position}/{total}] FAILED: {config.description}"
                )
                logger.error(f"[{position}/{total}] Error: {error_summary}")
                logger.error(f"[{position}/{total}] Full log: {log_file}")

                return ExperimentResult(
                    config=config,
                    status=ExperimentStatus.FAILED,
                    duration_seconds=round(duration, 1),
                    stdout=log_content,  # 로그 파일 내용 사용
                    stderr=f"{error_summary}\n\nFull stderr: {exec_result['stderr']}",
                    git_diff=git_diff,
                    log_file=log_file,
                    diff_file=diff_file,
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
                diff_file=diff_file,
                evaluation=evaluation,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{position}/{total}] ERROR: {config.description}: {str(e)}"
            )
            # diff_file may not be defined if exception occurred very early
            diff_file_path = self._get_diff_file_path(config.repo_path, self._exp_number) if self._exp_number > 0 else None
            return ExperimentResult(
                config=config,
                status=ExperimentStatus.FAILED,
                duration_seconds=round(duration, 1),
                stderr=str(e),
                git_diff=git_diff,
                diff_file=diff_file_path,
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
