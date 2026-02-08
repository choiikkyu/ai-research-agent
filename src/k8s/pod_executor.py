"""Kubernetes pod executor for running commands on existing pods."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from src.core.config import settings

logger = logging.getLogger(__name__)


class PodExecutor:
    """Execute commands on existing Kubernetes pods."""

    def __init__(self):
        """Initialize Kubernetes client."""
        try:
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except Exception:
            config.load_kube_config()
            logger.info("Using local Kubernetes configuration")

        self.v1 = client.CoreV1Api()
        self.namespace = settings.k8s_namespace

    async def verify_pod_running(self, pod_name: str) -> bool:
        """Verify that a pod exists and is in Running state.

        Args:
            pod_name: Name of the pod to verify

        Returns:
            True if pod is running, False otherwise

        Raises:
            RuntimeError: If pod does not exist or is not running
        """
        status = await self.get_pod_status(pod_name)
        if status is None:
            raise RuntimeError(f"Pod '{pod_name}' not found in namespace '{self.namespace}'")
        if status != "Running":
            raise RuntimeError(f"Pod '{pod_name}' is not running (status: {status})")
        logger.info(f"Pod '{pod_name}' is running")
        return True

    async def execute_on_pod(
        self,
        pod_name: str,
        script: str,
        timeout: int = 3600,
    ) -> Dict[str, Any]:
        """Execute a script on a running pod.

        Args:
            pod_name: Name of the pod
            script: Script content to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution result dictionary
        """
        logger.info(f"Executing script on pod: {pod_name}")

        try:
            script_path = "/tmp/experiment_script.sh"
            write_cmd = f"cat > {script_path} << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF"

            kubectl_write = [
                "kubectl", "exec", "-n", self.namespace, pod_name,
                "--", "bash", "-c", write_cmd,
            ]

            await self._run_command(kubectl_write)

            kubectl_run = [
                "kubectl", "exec", "-n", self.namespace, pod_name,
                "--", "bash", "-c", f"chmod +x {script_path} && {script_path}",
            ]

            result = await self._run_command(kubectl_run, timeout=timeout)

            return {
                "success": result["returncode"] == 0,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"],
            }

        except asyncio.TimeoutError:
            logger.error(f"Script execution timed out on pod {pod_name}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1,
            }
        except Exception as e:
            logger.error(f"Failed to execute script on pod {pod_name}: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    async def get_pod_logs(self, pod_name: str, tail_lines: int = 100) -> str:
        """Get logs from a pod.

        Args:
            pod_name: Name of the pod
            tail_lines: Number of lines to tail

        Returns:
            Pod logs as string
        """
        try:
            logs = self.v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=tail_lines,
            )
            return logs
        except ApiException as e:
            logger.error(f"Failed to get logs for pod {pod_name}: {str(e)}")
            return ""

    async def get_pod_status(self, pod_name: str) -> Optional[str]:
        """Get current status of a pod.

        Args:
            pod_name: Name of the pod

        Returns:
            Pod phase (Pending, Running, Succeeded, Failed, Unknown) or None
        """
        try:
            pod = self.v1.read_namespaced_pod_status(
                name=pod_name,
                namespace=self.namespace,
            )
            return pod.status.phase
        except ApiException:
            return None

    async def _run_command(
        self,
        cmd: list[str],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Run a shell command asynchronously.

        Args:
            cmd: Command to run as list of strings
            timeout: Command timeout in seconds

        Returns:
            Dictionary with returncode, stdout, and stderr
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8") if stdout else "",
                "stderr": stderr.decode("utf-8") if stderr else "",
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            raise
