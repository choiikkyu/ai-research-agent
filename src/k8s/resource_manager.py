"""Resource manager for tracking and cleaning up Kubernetes resources."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import redis.asyncio as aioredis

from src.core.config import settings
from src.k8s.pod_launcher import K8sPodLauncher

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manage and track Kubernetes resources for experiments."""

    def __init__(self):
        """Initialize resource manager."""
        self.launcher = K8sPodLauncher()
        self.redis_client: Optional[aioredis.Redis] = None

    async def connect(self):
        """Connect to Redis for resource tracking."""
        try:
            self.redis_client = await aioredis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for resource tracking")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {str(e)}")
            self.redis_client = None

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()

    async def register_resource(
        self,
        experiment_id: str,
        resource_type: str,
        resource_name: str,
        metadata: Dict = None
    ):
        """
        Register a resource for tracking.

        Args:
            experiment_id: Experiment identifier
            resource_type: Type of resource (pod, branch, pr)
            resource_name: Name/identifier of the resource
            metadata: Additional metadata
        """
        if not self.redis_client:
            logger.warning("Redis not connected, skipping resource registration")
            return

        key = f"experiment:{experiment_id}:resources"
        resource = {
            "type": resource_type,
            "name": resource_name,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        try:
            await self.redis_client.hset(
                key,
                f"{resource_type}:{resource_name}",
                str(resource)
            )

            # Set TTL for auto-cleanup (7 days)
            await self.redis_client.expire(key, 7 * 24 * 3600)

            logger.info(
                f"Registered {resource_type} resource {resource_name} "
                f"for experiment {experiment_id}"
            )
        except Exception as e:
            logger.error(f"Failed to register resource: {str(e)}")

    async def get_experiment_resources(
        self,
        experiment_id: str
    ) -> List[Dict]:
        """
        Get all resources for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            List of resource dictionaries
        """
        if not self.redis_client:
            logger.warning("Redis not connected")
            return []

        key = f"experiment:{experiment_id}:resources"

        try:
            resources_raw = await self.redis_client.hgetall(key)
            resources = []

            for resource_key, resource_str in resources_raw.items():
                try:
                    resource = eval(resource_str)  # Convert string back to dict
                    resources.append(resource)
                except:
                    logger.warning(f"Could not parse resource: {resource_key}")

            return resources

        except Exception as e:
            logger.error(f"Failed to get experiment resources: {str(e)}")
            return []

    async def cleanup_experiment(
        self,
        experiment_id: str,
        force: bool = False
    ) -> Dict[str, List[str]]:
        """
        Cleanup all resources for an experiment.

        Args:
            experiment_id: Experiment identifier
            force: Force cleanup even if experiment is recent

        Returns:
            Dictionary of cleaned resources by type
        """
        logger.info(f"Cleaning up resources for experiment {experiment_id}")

        cleaned = {
            "pods": [],
            "branches": [],
            "prs": []
        }

        # Get all resources
        resources = await self.get_experiment_resources(experiment_id)

        for resource in resources:
            resource_type = resource.get("type")
            resource_name = resource.get("name")

            try:
                if resource_type == "pod":
                    if await self.launcher.cleanup_pod(resource_name):
                        cleaned["pods"].append(resource_name)
                        logger.info(f"Cleaned up pod: {resource_name}")

                elif resource_type == "branch":
                    # TODO: Implement branch cleanup
                    logger.info(f"Branch cleanup not yet implemented: {resource_name}")

                elif resource_type == "pr":
                    # TODO: Implement PR cleanup
                    logger.info(f"PR cleanup not yet implemented: {resource_name}")

            except Exception as e:
                logger.error(
                    f"Failed to cleanup {resource_type} {resource_name}: {str(e)}"
                )

        # Clear resource tracking in Redis
        if self.redis_client:
            key = f"experiment:{experiment_id}:resources"
            await self.redis_client.delete(key)

        return cleaned

    async def cleanup_stale_resources(
        self,
        max_age_hours: int = 24
    ) -> Dict[str, List[str]]:
        """
        Cleanup stale resources older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Dictionary of cleaned resources
        """
        logger.info(f"Cleaning up stale resources older than {max_age_hours} hours")

        if not self.redis_client:
            logger.warning("Redis not connected, cannot cleanup stale resources")
            return {}

        cleaned = {
            "pods": [],
            "experiments": []
        }

        try:
            # Find all experiment keys
            cursor = 0
            pattern = "experiment:*:resources"

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    # Extract experiment ID from key
                    parts = key.split(":")
                    if len(parts) >= 2:
                        experiment_id = parts[1]

                        # Check age of resources
                        resources = await self.get_experiment_resources(experiment_id)

                        for resource in resources:
                            created_at_str = resource.get("created_at")
                            if created_at_str:
                                try:
                                    created_at = datetime.fromisoformat(created_at_str)
                                    age = datetime.now() - created_at

                                    if age > timedelta(hours=max_age_hours):
                                        # Cleanup this experiment
                                        result = await self.cleanup_experiment(
                                            experiment_id,
                                            force=True
                                        )
                                        cleaned["pods"].extend(result.get("pods", []))
                                        cleaned["experiments"].append(experiment_id)
                                        break  # Move to next experiment

                                except Exception as e:
                                    logger.warning(
                                        f"Could not parse timestamp for resource: {str(e)}"
                                    )

                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Failed to cleanup stale resources: {str(e)}")

        logger.info(
            f"Cleaned up {len(cleaned['experiments'])} stale experiments "
            f"and {len(cleaned['pods'])} pods"
        )

        return cleaned

    async def get_resource_stats(self) -> Dict[str, int]:
        """
        Get statistics about tracked resources.

        Returns:
            Dictionary with resource counts
        """
        stats = {
            "experiments": 0,
            "pods": 0,
            "branches": 0,
            "prs": 0
        }

        if not self.redis_client:
            return stats

        try:
            # Count experiments
            cursor = 0
            pattern = "experiment:*:resources"

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )

                stats["experiments"] += len(keys)

                for key in keys:
                    # Count resources per experiment
                    resources = await self.redis_client.hgetall(key)
                    for resource_key in resources.keys():
                        if resource_key.startswith("pod:"):
                            stats["pods"] += 1
                        elif resource_key.startswith("branch:"):
                            stats["branches"] += 1
                        elif resource_key.startswith("pr:"):
                            stats["prs"] += 1

                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Failed to get resource stats: {str(e)}")

        return stats