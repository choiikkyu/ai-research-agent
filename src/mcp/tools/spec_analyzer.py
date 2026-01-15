"""Technical specification analyzer."""

import re
from typing import Any, Dict
import logging

from src.core.models import TechSpec
from src.core.github_code_reference import GitHubCodeReference

logger = logging.getLogger(__name__)


async def analyze_spec(spec_url: str) -> TechSpec:
    """
    Analyze technical specification from Notion or other sources.

    Args:
        spec_url: URL to the technical specification

    Returns:
        Analyzed TechSpec object
    """
    logger.info(f"Analyzing specification from: {spec_url}")

    # Initialize GitHub code reference
    github_ref = GitHubCodeReference()

    # Parse Notion URL
    if "notion.so" in spec_url:
        # TODO: Implement actual Notion API integration
        # For now, create mock spec with repository determination

        # Extract task type from URL or content
        task_type = "MODEL_TRAINING"  # Default

        if "feature" in spec_url.lower():
            task_type = "FEATURE_ENGINEERING"
        elif "model" in spec_url.lower() or "ctr" in spec_url.lower() or "cvr" in spec_url.lower():
            task_type = "MODEL_TRAINING"

        # Create initial spec
        initial_spec = {
            "task_type": task_type,
            "content": "Technical specification content from Notion"
        }

        # Determine target repository
        target_repo = github_ref.determine_target_repo(initial_spec)

        # Get implementation path
        implementation_path = github_ref.get_implementation_path(
            target_repo,
            task_type,
            "example_feature"  # This would come from the spec
        )

        spec = TechSpec(
            title="AI Research Task",
            content="Technical specification content from Notion",
            task_type=task_type,
            repository=target_repo,
            requirements={
                "gpu_required": task_type == "MODEL_TRAINING",
                "memory_gb": 64 if task_type == "MODEL_TRAINING" else 32,
                "cpu_cores": 8,
                "implementation_path": implementation_path,
                "target_repository": target_repo,
            }
        )

        logger.info(f"Analyzed spec: {spec.title} ({spec.task_type}) -> {target_repo}")
        return spec

    else:
        # Handle other sources
        raise NotImplementedError(f"Unsupported spec source: {spec_url}")


def detect_task_type(content: str) -> str:
    """
    Detect task type from specification content.

    Args:
        content: Specification content

    Returns:
        Task type string
    """
    content_lower = content.lower()

    # Keywords for model training
    model_keywords = [
        "model", "training", "pctr", "pcvr", "neural network",
        "deep learning", "auc", "logloss", "calibration"
    ]

    # Keywords for feature engineering
    feature_keywords = [
        "feature", "engineering", "pipeline", "transformation",
        "aggregation", "null ratio", "importance"
    ]

    model_score = sum(1 for kw in model_keywords if kw in content_lower)
    feature_score = sum(1 for kw in feature_keywords if kw in content_lower)

    if model_score > feature_score:
        return "MODEL_TRAINING"
    elif feature_score > model_score:
        return "FEATURE_ENGINEERING"
    else:
        # Default to model training if unclear
        return "MODEL_TRAINING"


def extract_requirements(content: str) -> Dict[str, Any]:
    """
    Extract requirements from specification content.

    Args:
        content: Specification content

    Returns:
        Dictionary of requirements
    """
    requirements = {
        "gpu_required": False,
        "memory_gb": 32,
        "cpu_cores": 8,
        "dataset": None,
        "metrics": [],
    }

    # Check for GPU requirements
    if any(kw in content.lower() for kw in ["gpu", "cuda", "training", "deep learning"]):
        requirements["gpu_required"] = True
        requirements["memory_gb"] = 64

    # Extract dataset information
    dataset_pattern = r"dataset[:\s]+([^\n,]+)"
    dataset_match = re.search(dataset_pattern, content, re.IGNORECASE)
    if dataset_match:
        requirements["dataset"] = dataset_match.group(1).strip()

    # Extract metrics
    metrics_keywords = ["auc", "logloss", "precision", "recall", "f1", "accuracy"]
    requirements["metrics"] = [
        kw for kw in metrics_keywords if kw in content.lower()
    ]

    return requirements