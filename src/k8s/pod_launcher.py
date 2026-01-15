"""Kubernetes pod launcher using ak-launch-tf-box."""

import asyncio
import logging
import subprocess
from typing import Any, Dict, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from src.core.config import settings

logger = logging.getLogger(__name__)


class K8sPodLauncher:
    """Manage Kubernetes pods for experiments using ak-launch-tf-box."""

    def __init__(self):
        """Initialize Kubernetes client."""
        try:
            # Try in-cluster config first (when running in K8s)
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except:
            # Fall back to local kubeconfig
            config.load_kube_config()
            logger.info("Using local Kubernetes configuration")

        self.v1 = client.CoreV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.namespace = settings.k8s_namespace

    async def launch_pod(
        self,
        pod_name: str,
        pod_type: str = "cpu",
        instance_type: str = None,
        memory: str = None,
        cpu: str = None
    ) -> Dict[str, Any]:
        """
        Launch a pod using ak-launch-tf-box.

        Args:
            pod_name: Name for the pod
            pod_type: "cpu" or "gpu"
            instance_type: AWS instance type (e.g., "g4dn.4xlarge:ondemand")
            memory: Memory request/limit (e.g., "100Gi")
            cpu: CPU request/limit (e.g., "14000m")

        Returns:
            Pod information dictionary
        """
        if instance_type is None:
            instance_type = (
                settings.default_gpu_instance if pod_type == "gpu"
                else settings.default_cpu_instance
            )

        # Ensure instance type has capacity type
        if ":" not in instance_type:
            instance_type = f"{instance_type}:ondemand"

        # Construct ak-launch-tf-box command
        cmd = [
            "ak-launch-tf-box",
            "-t", pod_type,
            "-n", pod_name,
            "-i", instance_type
        ]

        # Add optional parameters
        if memory:
            cmd.extend(["-m", memory])
        if cpu:
            cmd.extend(["-c", cpu])

        logger.info(f"Launching pod with command: {' '.join(cmd)}")

        try:
            # Execute ak-launch-tf-box command
            result = await self._run_command(cmd)

            if result["returncode"] == 0:
                logger.info(f"Successfully launched pod: {pod_name}")

                # Wait for pod to be ready
                await self._wait_for_pod_ready(pod_name)

                return {
                    "pod_name": pod_name,
                    "pod_type": pod_type,
                    "instance_type": instance_type,
                    "status": "Running",
                    "namespace": self.namespace
                }
            else:
                raise RuntimeError(f"Failed to launch pod: {result['stderr']}")

        except Exception as e:
            logger.error(f"Failed to launch pod {pod_name}: {str(e)}")
            raise

    async def execute_on_pod(
        self,
        pod_name: str,
        script: str,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute a script on a running pod.

        Args:
            pod_name: Name of the pod
            script: Script content to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution result dictionary
        """
        logger.info(f"Executing script on pod: {pod_name}")

        try:
            # Write script to temporary file on pod
            script_path = "/tmp/experiment_script.sh"
            write_cmd = f"cat > {script_path} << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF"

            # Execute via kubectl exec
            kubectl_write = [
                "kubectl", "exec", "-n", self.namespace, pod_name,
                "--", "bash", "-c", write_cmd
            ]

            await self._run_command(kubectl_write)

            # Make script executable and run it
            kubectl_run = [
                "kubectl", "exec", "-n", self.namespace, pod_name,
                "--", "bash", "-c", f"chmod +x {script_path} && {script_path}"
            ]

            result = await self._run_command(kubectl_run, timeout=timeout)

            return {
                "success": result["returncode"] == 0,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"]
            }

        except asyncio.TimeoutError:
            logger.error(f"Script execution timed out on pod {pod_name}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1
            }
        except Exception as e:
            logger.error(f"Failed to execute script on pod {pod_name}: {str(e)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    async def get_pod_logs(
        self,
        pod_name: str,
        tail_lines: int = 100
    ) -> str:
        """
        Get logs from a pod.

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
                tail_lines=tail_lines
            )
            return logs
        except ApiException as e:
            logger.error(f"Failed to get logs for pod {pod_name}: {str(e)}")
            return ""

    async def cleanup_pod(self, pod_name: str) -> bool:
        """
        Cleanup/delete a pod.

        Args:
            pod_name: Name of the pod to delete

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Cleaning up pod: {pod_name}")

        try:
            # Use helm uninstall to properly cleanup
            helm_cmd = [
                "helm", "uninstall",
                f"chart-{pod_name}",
                "-n", self.namespace
            ]

            result = await self._run_command(helm_cmd)

            if result["returncode"] == 0:
                logger.info(f"Successfully cleaned up pod: {pod_name}")
                return True
            else:
                # Try direct pod deletion as fallback
                try:
                    self.v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions()
                    )
                    logger.info(f"Deleted pod directly: {pod_name}")
                    return True
                except:
                    logger.warning(f"Could not delete pod {pod_name}")
                    return False

        except Exception as e:
            logger.error(f"Failed to cleanup pod {pod_name}: {str(e)}")
            return False

    async def _run_command(
        self,
        cmd: list[str],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run a shell command asynchronously.

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
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8") if stdout else "",
                "stderr": stderr.decode("utf-8") if stderr else ""
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            raise

    async def _wait_for_pod_ready(
        self,
        pod_name: str,
        timeout: int = 300
    ) -> bool:
        """
        Wait for a pod to be ready.

        Args:
            pod_name: Name of the pod
            timeout: Maximum time to wait in seconds

        Returns:
            True if pod is ready, False otherwise
        """
        logger.info(f"Waiting for pod {pod_name} to be ready...")

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                pod = self.v1.read_namespaced_pod_status(
                    name=pod_name,
                    namespace=self.namespace
                )

                if pod.status.phase == "Running":
                    # Check if all containers are ready
                    if pod.status.container_statuses:
                        all_ready = all(
                            container.ready
                            for container in pod.status.container_statuses
                        )
                        if all_ready:
                            logger.info(f"Pod {pod_name} is ready")
                            return True

            except ApiException as e:
                if e.status != 404:  # Ignore not found errors
                    logger.warning(f"Error checking pod status: {str(e)}")

            await asyncio.sleep(5)

        logger.warning(f"Pod {pod_name} did not become ready within {timeout} seconds")
        return False

    async def get_pod_status(self, pod_name: str) -> Optional[str]:
        """
        Get current status of a pod.

        Args:
            pod_name: Name of the pod

        Returns:
            Pod phase (Pending, Running, Succeeded, Failed, Unknown) or None
        """
        try:
            pod = self.v1.read_namespaced_pod_status(
                name=pod_name,
                namespace=self.namespace
            )
            return pod.status.phase
        except ApiException:
            return None