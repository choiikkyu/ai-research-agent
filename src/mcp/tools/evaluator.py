"""Experiment evaluator."""

import logging
from typing import Any, Dict

from src.core.config import settings
from src.core.models import ExperimentResult, TechSpec

logger = logging.getLogger(__name__)


async def evaluate_results(
    result: ExperimentResult,
    spec: TechSpec
) -> Dict[str, Any]:
    """
    Evaluate experiment results against success criteria.

    Args:
        result: Experiment result
        spec: Technical specification

    Returns:
        Evaluation dictionary with pass/fail and details
    """
    logger.info(f"Evaluating experiment {result.experiment_id}")

    if spec.task_type == "MODEL_TRAINING":
        evaluation = evaluate_model_training(result.metrics)
    elif spec.task_type == "FEATURE_ENGINEERING":
        evaluation = evaluate_feature_engineering(result.metrics)
    else:
        evaluation = {
            "passed": False,
            "reason": f"Unknown task type: {spec.task_type}",
            "details": {}
        }

    evaluation["experiment_id"] = result.experiment_id
    evaluation["task_type"] = spec.task_type
    evaluation["metrics"] = result.metrics
    evaluation["recommendations"] = result.recommendations

    logger.info(
        f"Evaluation result for {result.experiment_id}: "
        f"{'PASSED' if evaluation['passed'] else 'FAILED'}"
    )

    return evaluation


def evaluate_model_training(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate model training metrics.

    Success criteria:
    - AUC > 0.85
    - LogLoss < 0.35
    - Calibration Error < 0.02

    Args:
        metrics: Model training metrics

    Returns:
        Evaluation result
    """
    auc = metrics.get("auc", 0)
    logloss = metrics.get("logloss", float("inf"))
    calibration_error = metrics.get("calibration_error", float("inf"))

    # Check thresholds
    auc_pass = auc > settings.model_auc_threshold
    logloss_pass = logloss < settings.model_logloss_threshold
    calibration_pass = calibration_error < settings.model_calibration_error_threshold

    # Overall pass/fail
    passed = auc_pass and logloss_pass and calibration_pass

    # Detailed evaluation
    details = {
        "auc": {
            "value": auc,
            "threshold": settings.model_auc_threshold,
            "passed": auc_pass,
            "message": f"AUC: {auc:.3f} {'✓' if auc_pass else '✗'} (threshold: {settings.model_auc_threshold})"
        },
        "logloss": {
            "value": logloss,
            "threshold": settings.model_logloss_threshold,
            "passed": logloss_pass,
            "message": f"LogLoss: {logloss:.3f} {'✓' if logloss_pass else '✗'} (threshold: {settings.model_logloss_threshold})"
        },
        "calibration_error": {
            "value": calibration_error,
            "threshold": settings.model_calibration_error_threshold,
            "passed": calibration_pass,
            "message": f"Calibration Error: {calibration_error:.3f} {'✓' if calibration_pass else '✗'} (threshold: {settings.model_calibration_error_threshold})"
        }
    }

    # Generate failure reason if not passed
    if not passed:
        failed_metrics = [
            name for name, detail in details.items()
            if not detail["passed"]
        ]
        reason = f"Failed metrics: {', '.join(failed_metrics)}"
    else:
        reason = "All metrics passed thresholds"

    return {
        "passed": passed,
        "reason": reason,
        "details": details,
        "score": calculate_model_score(metrics)
    }


def evaluate_feature_engineering(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluate feature engineering metrics.

    Success criteria:
    - Null Ratio < 10%
    - Importance Score > 0.05
    - Latency Impact < 10ms

    Args:
        metrics: Feature engineering metrics

    Returns:
        Evaluation result
    """
    null_ratio = metrics.get("null_ratio", 1.0)
    importance_score = metrics.get("importance_score", 0)
    latency_ms = metrics.get("latency_ms", float("inf"))

    # Check thresholds
    null_ratio_pass = null_ratio < settings.feature_null_ratio_threshold
    importance_pass = importance_score > settings.feature_importance_threshold
    latency_pass = latency_ms < settings.serving_latency_increase_threshold

    # Overall pass/fail
    passed = null_ratio_pass and importance_pass and latency_pass

    # Detailed evaluation
    details = {
        "null_ratio": {
            "value": null_ratio,
            "threshold": settings.feature_null_ratio_threshold,
            "passed": null_ratio_pass,
            "message": f"Null Ratio: {null_ratio:.2%} {'✓' if null_ratio_pass else '✗'} (threshold: {settings.feature_null_ratio_threshold:.1%})"
        },
        "importance_score": {
            "value": importance_score,
            "threshold": settings.feature_importance_threshold,
            "passed": importance_pass,
            "message": f"Importance: {importance_score:.3f} {'✓' if importance_pass else '✗'} (threshold: {settings.feature_importance_threshold})"
        },
        "latency_ms": {
            "value": latency_ms,
            "threshold": settings.serving_latency_increase_threshold,
            "passed": latency_pass,
            "message": f"Latency: {latency_ms:.1f}ms {'✓' if latency_pass else '✗'} (threshold: {settings.serving_latency_increase_threshold}ms)"
        }
    }

    # Generate failure reason if not passed
    if not passed:
        failed_metrics = [
            name for name, detail in details.items()
            if not detail["passed"]
        ]
        reason = f"Failed metrics: {', '.join(failed_metrics)}"
    else:
        reason = "All metrics passed thresholds"

    # Additional quality checks
    coverage = metrics.get("coverage", 0)
    if coverage < 0.8:
        details["coverage_warning"] = {
            "value": coverage,
            "message": f"Low feature coverage: {coverage:.1%}. Consider improving data availability."
        }

    return {
        "passed": passed,
        "reason": reason,
        "details": details,
        "score": calculate_feature_score(metrics)
    }


def calculate_model_score(metrics: Dict[str, float]) -> float:
    """
    Calculate overall model score (0-100).

    Args:
        metrics: Model metrics

    Returns:
        Score between 0 and 100
    """
    auc = metrics.get("auc", 0)
    logloss = metrics.get("logloss", 1)
    calibration_error = metrics.get("calibration_error", 1)

    # Normalize metrics to 0-1 range
    auc_score = min(max((auc - 0.5) * 2, 0), 1)  # Map 0.5-1.0 to 0-1
    logloss_score = min(max(1 - logloss, 0), 1)  # Lower is better
    calibration_score = min(max(1 - calibration_error * 10, 0), 1)  # Lower is better

    # Weighted average
    weights = {"auc": 0.5, "logloss": 0.3, "calibration": 0.2}
    score = (
        auc_score * weights["auc"] +
        logloss_score * weights["logloss"] +
        calibration_score * weights["calibration"]
    ) * 100

    return round(score, 1)


def calculate_feature_score(metrics: Dict[str, float]) -> float:
    """
    Calculate overall feature score (0-100).

    Args:
        metrics: Feature metrics

    Returns:
        Score between 0 and 100
    """
    null_ratio = metrics.get("null_ratio", 1)
    importance_score = metrics.get("importance_score", 0)
    latency_ms = metrics.get("latency_ms", 100)
    coverage = metrics.get("coverage", 0)

    # Normalize metrics to 0-1 range
    null_score = min(max(1 - null_ratio, 0), 1)  # Lower is better
    importance_norm = min(max(importance_score * 10, 0), 1)  # Scale 0-0.1 to 0-1
    latency_score = min(max(1 - latency_ms / 100, 0), 1)  # Lower is better
    coverage_score = min(max(coverage, 0), 1)

    # Weighted average
    weights = {
        "null": 0.2,
        "importance": 0.4,
        "latency": 0.2,
        "coverage": 0.2
    }

    score = (
        null_score * weights["null"] +
        importance_norm * weights["importance"] +
        latency_score * weights["latency"] +
        coverage_score * weights["coverage"]
    ) * 100

    return round(score, 1)