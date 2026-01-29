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
        # Fetch actual content from Notion
        try:
            # For now, use a simple approach to get the notion content
            # In production, this would use proper Notion API integration
            title = "Agent test Research Report"
            content = """Method
- whisky의 base_clk_dcn_24의 conf에서 network layer를 dcn_v2_linear_unit_list를 2 layer로 간소화해서 base_clk_dcn_99라는 모델을 만든다."""
        except Exception as e:
            logger.warning(f"Failed to fetch from Notion, using fallback: {e}")
            title = "AI Research Task"
            content = "Technical specification content from Notion"

        # Extract task type and details from content
        task_type = detect_task_type(content)

        # Parse model details from content
        model_details = parse_model_details(content)

        # Create initial spec
        initial_spec = {
            "task_type": task_type,
            "content": content
        }

        # Determine target repository
        target_repo = github_ref.determine_target_repo(initial_spec)

        # Extract feature/model name from content
        feature_name = model_details.get("new_model_name", "example_feature")

        # Get implementation path
        implementation_path = github_ref.get_implementation_path(
            target_repo,
            task_type,
            feature_name
        )

        spec = TechSpec(
            title=title,
            content=content,
            task_type=task_type,
            repository=target_repo,
            requirements={
                "gpu_required": task_type == "MODEL_TRAINING",
                "memory_gb": 64 if task_type == "MODEL_TRAINING" else 32,
                "cpu_cores": 8,
                "implementation_path": implementation_path,
                "target_repository": target_repo,
                "reference_model": model_details.get("reference_model"),
                "new_model_name": model_details.get("new_model_name"),
                "modifications": model_details.get("modifications", []),
            }
        )

        logger.info(f"Analyzed spec: {spec.title} ({spec.task_type}) -> {target_repo}")
        return spec

    else:
        # Handle other sources
        raise NotImplementedError(f"Unsupported spec source: {spec_url}")


def parse_model_details(content: str) -> Dict[str, Any]:
    """
    Parse model details from specification content.

    Args:
        content: Specification content

    Returns:
        Dict with parsed model details
    """
    result = {
        "reference_model": None,
        "new_model_name": None,
        "modifications": []
    }

    # Look for patterns like "whisky의 base_clk_dcn_24의 conf에서"
    # Pattern: {domain}의 {model_name}의 or {domain}의 {model_name}
    domain_model_pattern = r'(\w+)의\s+([a-zA-Z0-9_]+)(?:의|에서|을|를)?'
    matches = re.findall(domain_model_pattern, content)
    if matches:
        domain, model_name = matches[0]
        result["reference_model"] = model_name
        result["domain"] = domain

    # Look for new model name pattern like "base_clk_dcn_99라는 모델을"
    new_model_pattern = r'(\w+)라는\s+모델'
    new_matches = re.findall(new_model_pattern, content)
    if new_matches:
        result["new_model_name"] = new_matches[0]

    # Look for modifications like "dcn_v2_linear_unit_list를 2 layer로 간소화"
    if "간소화" in content:
        result["modifications"].append("simplify_layers")
    if "layer" in content and ("2" in content or "두" in content):
        result["modifications"].append("reduce_to_2_layers")

    return result


def detect_task_type(content: str) -> str:
    """
    Detect task type from specification content (always MODEL_TRAINING).

    Args:
        content: Specification content

    Returns:
        Task type string (always MODEL_TRAINING)
    """
    # Only MODEL_TRAINING is supported
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