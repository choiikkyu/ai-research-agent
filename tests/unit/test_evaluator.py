"""Unit tests for evaluator module."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "evaluator",
    Path(__file__).parent.parent.parent / "src" / "evaluation" / "evaluator.py",
)
evaluator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator_module)

evaluate_model_training = evaluator_module.evaluate_model_training
calculate_model_score = evaluator_module.calculate_model_score


class TestEvaluateModelTraining:
    """Test evaluate_model_training function."""

    def test_evaluate_passing_metrics(self):
        metrics = {"auc": 0.9, "logloss": 0.3, "calibration_error": 0.01}
        result = evaluate_model_training(metrics)
        assert result["passed"] is True
        assert result["score"] > 75
        assert result["details"]["auc"]["passed"] is True
        assert result["details"]["logloss"]["passed"] is True
        assert result["details"]["calibration_error"]["passed"] is True

    def test_evaluate_failing_metrics(self):
        metrics = {"auc": 0.7, "logloss": 0.5, "calibration_error": 0.05}
        result = evaluate_model_training(metrics)
        assert result["passed"] is False
        assert result["details"]["auc"]["passed"] is False
        assert result["details"]["logloss"]["passed"] is False
        assert result["details"]["calibration_error"]["passed"] is False
        assert "Failed metrics" in result["reason"]

    def test_evaluate_mixed_metrics(self):
        metrics = {"auc": 0.9, "logloss": 0.5, "calibration_error": 0.01}
        result = evaluate_model_training(metrics)
        assert result["passed"] is False
        assert result["details"]["auc"]["passed"] is True
        assert result["details"]["logloss"]["passed"] is False

    def test_flexible_metric_keys_auc(self):
        for key in ["auc", "test_auc", "validation_auc"]:
            metrics = {key: 0.9, "logloss": 0.3, "calibration_error": 0.01}
            result = evaluate_model_training(metrics)
            assert result["details"]["auc"]["passed"] is True

    def test_flexible_metric_keys_logloss(self):
        for key in ["logloss", "test_logloss", "validation_logloss"]:
            metrics = {"auc": 0.9, key: 0.3, "calibration_error": 0.01}
            result = evaluate_model_training(metrics)
            assert result["details"]["logloss"]["passed"] is True

    def test_flexible_metric_keys_calibration(self):
        for key in ["calibration_error", "calib_error", "m3_calibration_error", "m3_calib_error"]:
            metrics = {"auc": 0.9, "logloss": 0.3, key: 0.01}
            result = evaluate_model_training(metrics)
            assert result["details"]["calibration_error"]["passed"] is True

    def test_all_metrics_preserved(self):
        metrics = {
            "auc": 0.9,
            "logloss": 0.3,
            "calibration_error": 0.01,
            "custom_metric": 100,
            "test_auc": 0.88,
        }
        result = evaluate_model_training(metrics)
        assert result["raw_metrics"] == metrics
        assert "custom_metric" not in result["all_metrics"]

    def test_edge_case_threshold_values(self):
        metrics = {"auc": 0.85, "logloss": 0.3, "calibration_error": 0.01}
        result = evaluate_model_training(metrics)
        assert result["details"]["auc"]["passed"] is False

        metrics = {"auc": 0.9, "logloss": 0.35, "calibration_error": 0.01}
        result = evaluate_model_training(metrics)
        assert result["details"]["logloss"]["passed"] is False


class TestCalculateModelScore:
    """Test calculate_model_score function."""

    def test_perfect_score(self):
        metrics = {"auc": 1.0, "logloss": 0.0, "calibration_error": 0.0}
        score = calculate_model_score(metrics)
        assert score == 100.0

    def test_good_score(self):
        metrics = {"auc": 0.9, "logloss": 0.3, "calibration_error": 0.01}
        score = calculate_model_score(metrics)
        assert score > 75

    def test_poor_score(self):
        metrics = {"auc": 0.6, "logloss": 0.8, "calibration_error": 0.1}
        score = calculate_model_score(metrics)
        assert score < 50

    def test_missing_metrics(self):
        metrics = {}
        score = calculate_model_score(metrics)
        assert 0 <= score <= 100

    def test_score_weights(self):
        metrics = {"auc": 0.85, "logloss": 0.35, "calibration_error": 0.02}
        score = calculate_model_score(metrics)
        assert isinstance(score, float)
        assert 0 <= score <= 100
