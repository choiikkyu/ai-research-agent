"""Test the enhanced markdown report generator with sample data."""

from src.experiment.report_generator import generate_markdown_report


def test_report_with_metrics():
    """Test report generation with MLflow metrics."""
    sample_results = [
        {
            "description": "Baseline (all features)",
            "status": "COMPLETED",
            "duration_seconds": 1362.6,
            "metrics": {
                "auc": 0.8234,
                "test_auc": 0.8156,
                "logloss": 0.4521,
                "test_logloss": 0.4678,
                "calibration_error": 0.0234,
            },
            "git_diff_lines": 22,
            "config": {
                "parameters": {
                    "ctr_linear_unit_list": "[16, 8]",
                    "cvr_linear_unit_list": "[16, 8]",
                }
            },
        },
        {
            "description": "Deeper Architecture",
            "status": "COMPLETED",
            "duration_seconds": 921.1,
            "metrics": {
                "auc": 0.8456,
                "test_auc": 0.8321,
                "logloss": 0.4123,
                "test_logloss": 0.4234,
                "calibration_error": 0.0189,
            },
            "git_diff_lines": 22,
            "config": {
                "parameters": {
                    "ctr_linear_unit_list": "[32, 16, 8]",
                    "cvr_linear_unit_list": "[32, 16, 8]",
                }
            },
        },
        {
            "description": "Wide Architecture",
            "status": "FAILED",
            "duration_seconds": 450.2,
            "metrics": {},
            "git_diff_lines": 18,
            "config": {
                "parameters": {
                    "ctr_linear_unit_list": "[64, 32]",
                    "cvr_linear_unit_list": "[64, 32]",
                }
            },
        },
    ]

    best_experiment = {
        "experiment_id": "exp-2",
        "description": "Deeper Architecture",
        "metrics": sample_results[1]["metrics"],
        "score": 0.85,
    }

    report = generate_markdown_report(
        batch_id="batch-test123",
        pod_name="ai-tf-box-test",
        utc_ymdh="2026020600",
        results=sample_results,
        model_name="zigzag_conv_mtl12",
        best_experiment=best_experiment,
    )

    # Verify key sections are present
    assert "# Zigzag_Conv_Mtl12 Experiment Report" in report
    assert "## Experiment Overview" in report
    assert "## Experiments Conducted" in report
    assert "## Metrics Comparison" in report
    assert "## Key Findings" in report
    assert "### Performance Analysis" in report
    assert "## Recommendations" in report
    assert "## Summary" in report

    # Verify metrics table
    assert "| Experiment | AUC | Test AUC | LogLoss | Test LogLoss | Calibration Error |" in report
    assert "0.8156" in report  # baseline test_auc
    assert "0.8321" in report  # deeper test_auc
    assert "↑" in report  # improvement indicator

    # Verify performance analysis
    assert "Best Test AUC" in report
    assert "Overfitting Warning" in report  # Should detect overfitting in exp 2
    assert "Best Calibration" in report

    print("✅ Report generation test passed!")
    print("\n" + "=" * 80)
    print("Generated Report:")
    print("=" * 80)
    print(report)


def test_report_without_metrics():
    """Test report generation without MLflow metrics."""
    sample_results = [
        {
            "description": "Configuration A",
            "status": "COMPLETED",
            "duration_seconds": 600.0,
            "metrics": {},
            "git_diff_lines": 10,
            "config": {"parameters": {}},
        },
        {
            "description": "Configuration B",
            "status": "COMPLETED",
            "duration_seconds": 750.0,
            "metrics": {},
            "git_diff_lines": 15,
            "config": {"parameters": {}},
        },
    ]

    report = generate_markdown_report(
        batch_id="batch-test456",
        pod_name="test-pod",
        utc_ymdh="2026020700",
        results=sample_results,
        model_name="test_model",
    )

    # Verify report still generates without metrics
    assert "# Test_Model Experiment Report" in report
    assert "## Experiment Overview" in report
    assert "## Experiments Conducted" in report

    # Metrics comparison should not be present
    assert "## Metrics Comparison" not in report

    print("\n✅ Report without metrics test passed!")


if __name__ == "__main__":
    test_report_with_metrics()
    test_report_without_metrics()