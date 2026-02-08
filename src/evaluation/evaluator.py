"""Model training evaluator."""

import logging
from typing import Any, Dict

from src.core.config import settings

logger = logging.getLogger(__name__)


def evaluate_model_training(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Evaluate model training metrics.

    Success criteria:
    - AUC > 0.85
    - LogLoss < 0.35
    - Calibration Error < 0.02

    Args:
        metrics: Model training metrics

    Returns:
        Evaluation result with pass/fail details
    """
    # Extract primary metrics with flexible key names
    auc = (
        metrics.get("auc")
        or metrics.get("test_auc")
        or metrics.get("validation_auc")
        or 0.0
    )
    logloss = (
        metrics.get("logloss")
        or metrics.get("test_logloss")
        or metrics.get("validation_logloss")
        or float("inf")
    )
    calibration_error = (
        metrics.get("calibration_error")
        or metrics.get("calib_error")
        or metrics.get("m3_calibration_error")
        or metrics.get("m3_calib_error")
        or float("inf")
    )

    # Check thresholds
    auc_pass = auc > settings.model_auc_threshold
    logloss_pass = logloss < settings.model_logloss_threshold
    calibration_pass = calibration_error < settings.model_calibration_error_threshold

    passed = auc_pass and logloss_pass and calibration_pass

    details = {
        "auc": {
            "value": auc,
            "threshold": settings.model_auc_threshold,
            "passed": auc_pass,
            "message": f"AUC: {auc:.3f} {'✓' if auc_pass else '✗'} (threshold: {settings.model_auc_threshold})",
        },
        "logloss": {
            "value": logloss,
            "threshold": settings.model_logloss_threshold,
            "passed": logloss_pass,
            "message": f"LogLoss: {logloss:.3f} {'✓' if logloss_pass else '✗'} (threshold: {settings.model_logloss_threshold})",
        },
        "calibration_error": {
            "value": calibration_error,
            "threshold": settings.model_calibration_error_threshold,
            "passed": calibration_pass,
            "message": f"Calibration Error: {calibration_error:.3f} {'✓' if calibration_pass else '✗'} (threshold: {settings.model_calibration_error_threshold})",
        },
    }

    if not passed:
        failed_metrics = [
            name for name, detail in details.items() if not detail["passed"]
        ]
        reason = f"Failed metrics: {', '.join(failed_metrics)}"
    else:
        reason = "All metrics passed thresholds"

    all_metrics_summary = {}
    primary_keys = ["auc", "test_auc", "logloss", "test_logloss", "calibration_error"]
    for key in primary_keys:
        if key in metrics:
            all_metrics_summary[key] = metrics[key]

    return {
        "passed": passed,
        "reason": reason,
        "details": details,
        "score": calculate_model_score(metrics),
        "all_metrics": all_metrics_summary,
        "raw_metrics": metrics,
    }


def calculate_model_score(metrics: Dict[str, float]) -> float:
    """Calculate overall model score (0-100).

    Args:
        metrics: Model metrics

    Returns:
        Score between 0 and 100
    """
    auc = metrics.get("auc", 0)
    logloss = metrics.get("logloss", 1)
    calibration_error = metrics.get("calibration_error", 1)

    auc_score = min(max((auc - 0.5) * 2, 0), 1)
    logloss_score = min(max(1 - logloss, 0), 1)
    calibration_score = min(max(1 - calibration_error * 10, 0), 1)

    weights = {"auc": 0.5, "logloss": 0.3, "calibration": 0.2}
    score = (
        auc_score * weights["auc"]
        + logloss_score * weights["logloss"]
        + calibration_score * weights["calibration"]
    ) * 100

    return round(score, 1)
