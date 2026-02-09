"""Generate enhanced markdown reports for experiment results with MLflow metrics comparison."""

from datetime import datetime
from typing import Dict, List, Any, Optional


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"

    minutes = int(seconds / 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes} minutes {remaining_seconds} seconds ({seconds:.1f}s)"

    hours = int(minutes / 60)
    remaining_minutes = int(minutes % 60)
    return f"{hours} hours {remaining_minutes} minutes ({seconds:.1f}s)"


def format_metric_value(value: Optional[float], metric_type: str = "default") -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"

    if metric_type in ["auc", "test_auc"]:
        return f"{value:.4f}"
    elif metric_type in ["logloss", "test_logloss"]:
        return f"{value:.4f}"
    elif metric_type in ["calibration_error"]:
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"


def compare_metrics(current: Optional[float], baseline: Optional[float]) -> str:
    """Compare current metric with baseline and return indicator."""
    if current is None or baseline is None:
        return ""

    diff = current - baseline
    if abs(diff) < 0.0001:  # Consider as equal
        return ""

    # For AUC, higher is better
    # For LogLoss and Calibration Error, lower is better
    return "↑" if diff > 0 else "↓"


def generate_markdown_report(
    batch_id: str,
    pod_name: str,
    utc_ymdh: str,
    results: List[Dict[str, Any]],
    model_name: str = "model",
    best_experiment: Optional[Dict[str, Any]] = None,
    full_results: Optional[List[Any]] = None,
) -> str:
    """Generate enhanced markdown report with MLflow metrics comparison.

    Args:
        batch_id: Batch identifier
        pod_name: K8s pod name used for experiments
        utc_ymdh: UTC timestamp of training data
        results: List of experiment results with metrics
        model_name: Name of the model being experimented on
        best_experiment: Optional best experiment info

    Returns:
        Markdown formatted report string
    """
    report = []

    # Header
    report.append(f"# {model_name.title()} Experiment Report")
    report.append("")

    # Overview
    report.append("## Experiment Overview")
    report.append(f"- **Model**: {model_name}")
    report.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}")

    # Format UTC timestamp
    if utc_ymdh and len(utc_ymdh) == 10:  # Format: YYYYMMDDHH
        formatted_date = f"{utc_ymdh[:4]}-{utc_ymdh[4:6]}-{utc_ymdh[6:8]}-{utc_ymdh[8:10]}"
        report.append(f"- **Training Data**: UTC {formatted_date}")
    else:
        report.append(f"- **Training Data**: {utc_ymdh}")

    report.append(f"- **Pod**: {pod_name}")
    report.append(f"- **Batch ID**: {batch_id}")
    report.append("")

    # Experiments Conducted
    report.append("## Experiments Conducted")
    report.append("")

    for i, result in enumerate(results, 1):
        report.append(f"### Experiment {i}: {result.get('description', 'Unnamed')}")

        # Extract parameters if available
        if 'parameters' in result.get('config', {}):
            params = result['config']['parameters']
            # Look for common parameter patterns
            if any(key for key in params if 'layer' in key.lower() or 'unit' in key.lower()):
                for key, value in params.items():
                    if 'layer' in key.lower() or 'unit' in key.lower():
                        report.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        status_icon = "✅" if result.get('status') == 'COMPLETED' else "❌"
        report.append(f"- **Status**: {status_icon} {result.get('status', 'Unknown')}")

        duration = result.get('duration_seconds', 0)
        report.append(f"- **Duration**: {format_duration(duration)}")

        if result.get('git_diff_lines'):
            report.append(f"- **Code Changes**: {result['git_diff_lines']} lines modified")

        report.append("")

    # MLflow Metrics Comparison Table
    if any(result.get('metrics') for result in results):
        report.append("## Metrics Comparison")
        report.append("")
        report.append("| Experiment | AUC | Test AUC | LogLoss | Test LogLoss | Calibration Error |")
        report.append("|------------|-----|----------|---------|--------------|-------------------|")

        # Find baseline (first experiment) metrics for comparison
        baseline_metrics = results[0].get('metrics', {}) if results else {}

        for i, result in enumerate(results, 1):
            metrics = result.get('metrics', {})
            exp_name = f"Exp {i}"

            # Get metric values
            auc = metrics.get('auc')
            test_auc = metrics.get('test_auc')
            logloss = metrics.get('logloss')
            test_logloss = metrics.get('test_logloss')
            cal_error = metrics.get('calibration_error')

            # Format with comparison indicators (compare to baseline)
            if i == 1:
                # Baseline - no comparison
                auc_str = format_metric_value(auc, 'auc')
                test_auc_str = format_metric_value(test_auc, 'test_auc')
                logloss_str = format_metric_value(logloss, 'logloss')
                test_logloss_str = format_metric_value(test_logloss, 'test_logloss')
                cal_error_str = format_metric_value(cal_error, 'calibration_error')
            else:
                # Compare with baseline
                auc_str = f"{format_metric_value(auc, 'auc')} {compare_metrics(auc, baseline_metrics.get('auc'))}"
                test_auc_str = f"{format_metric_value(test_auc, 'test_auc')} {compare_metrics(test_auc, baseline_metrics.get('test_auc'))}"

                # For loss metrics, lower is better, so reverse the arrow
                logloss_indicator = compare_metrics(baseline_metrics.get('logloss'), logloss)
                test_logloss_indicator = compare_metrics(baseline_metrics.get('test_logloss'), test_logloss)
                cal_error_indicator = compare_metrics(baseline_metrics.get('calibration_error'), cal_error)

                logloss_str = f"{format_metric_value(logloss, 'logloss')} {logloss_indicator}"
                test_logloss_str = f"{format_metric_value(test_logloss, 'test_logloss')} {test_logloss_indicator}"
                cal_error_str = f"{format_metric_value(cal_error, 'calibration_error')} {cal_error_indicator}"

            report.append(f"| {exp_name} | {auc_str} | {test_auc_str} | {logloss_str} | {test_logloss_str} | {cal_error_str} |")

        report.append("")
        report.append("_Note: ↑ indicates improvement, ↓ indicates degradation compared to baseline (Exp 1)_")
        report.append("")

    # Key Findings
    report.append("## Key Findings")
    report.append("")

    # Training Time Analysis
    report.append("### Training Time Analysis")
    if len(results) >= 2:
        durations = [(i, r.get('duration_seconds', 0), r.get('description', f'Exp {i}'))
                     for i, r in enumerate(results, 1)]
        fastest = min(durations, key=lambda x: x[1])
        slowest = max(durations, key=lambda x: x[1])

        if fastest[1] > 0 and slowest[1] > 0:
            time_diff_pct = ((slowest[1] - fastest[1]) / fastest[1]) * 100
            report.append(f"- Fastest: {fastest[2]} ({format_duration(fastest[1])})")
            report.append(f"- Slowest: {slowest[2]} ({format_duration(slowest[1])}) - **{time_diff_pct:.0f}% slower**")
    else:
        duration = results[0].get('duration_seconds', 0) if results else 0
        report.append(f"- Single experiment completed in {format_duration(duration)}")
    report.append("")

    # Performance Analysis (if metrics available)
    if any(result.get('metrics') for result in results):
        report.append("### Performance Analysis")

        # Find best AUC
        best_auc_result = max(
            (r for r in results if r.get('metrics', {}).get('test_auc')),
            key=lambda x: x['metrics'].get('test_auc', 0),
            default=None
        )

        if best_auc_result:
            exp_idx = results.index(best_auc_result) + 1
            test_auc = best_auc_result['metrics'].get('test_auc', 0)
            report.append(f"- **Best Test AUC**: Experiment {exp_idx} ({test_auc:.4f})")

        # Analyze overfitting
        for i, result in enumerate(results, 1):
            metrics = result.get('metrics', {})
            if metrics.get('auc') and metrics.get('test_auc'):
                overfit_gap = metrics['auc'] - metrics['test_auc']
                if overfit_gap > 0.01:
                    report.append(f"- **Overfitting Warning**: Experiment {i} shows {overfit_gap:.4f} gap between train/test AUC")

        # Calibration analysis
        best_cal_result = min(
            (r for r in results if r.get('metrics', {}).get('calibration_error') is not None),
            key=lambda x: x['metrics'].get('calibration_error', float('inf')),
            default=None
        )

        if best_cal_result:
            exp_idx = results.index(best_cal_result) + 1
            cal_error = best_cal_result['metrics'].get('calibration_error', 0)
            report.append(f"- **Best Calibration**: Experiment {exp_idx} (error: {cal_error:.4f})")

        report.append("")

    # Configuration Changes
    if any(result.get('git_diff_lines') for result in results):
        report.append("### Configuration Changes")

        # Identify common changes
        total_changes = sum(r.get('git_diff_lines', 0) for r in results)
        report.append(f"- Total code changes across all experiments: {total_changes} lines")

        # List unique configurations if descriptions are meaningful
        unique_configs = set()
        for result in results:
            desc = result.get('description', '')
            if desc and desc != 'Unnamed':
                unique_configs.add(desc)

        if unique_configs:
            report.append("- Configurations tested:")
            for config in sorted(unique_configs):
                report.append(f"  - {config}")

        report.append("")

    # Code Changes Details
    if full_results and any(r.git_diff for r in full_results):
        report.append("### Code Changes Details")
        report.append("")

        for i, result in enumerate(full_results, 1):
            if result.git_diff:
                report.append(f"#### Experiment {i}: {result.config.description}")
                report.append("")
                report.append("<details>")
                report.append(f"<summary>View diff ({len(result.git_diff.splitlines())} lines)</summary>")
                report.append("")
                report.append("```diff")
                report.append(result.git_diff)
                report.append("```")
                report.append("")
                report.append("</details>")
                report.append("")

        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    # Generate recommendations based on results
    recommendations = []

    if best_experiment:
        recommendations.append(
            f"1. **Best Overall**: {best_experiment.get('description', 'Unknown')} "
            f"(AUC: {best_experiment.get('metrics', {}).get('test_auc', 0):.4f})"
        )

    # Check for significant differences in metrics
    if len(results) >= 2:
        metrics_vary = False
        if all(r.get('metrics', {}).get('test_auc') for r in results):
            aucs = [r['metrics']['test_auc'] for r in results]
            if max(aucs) - min(aucs) > 0.01:
                metrics_vary = True
                recommendations.append(
                    f"2. **Significant Performance Variation**: Test AUC ranges from {min(aucs):.4f} to {max(aucs):.4f}"
                )

        if not metrics_vary:
            recommendations.append("2. **Similar Performance**: All configurations show comparable metrics")

    recommendations.append("3. **Next Steps**:")
    recommendations.append("   - Validate the best performing configuration on a holdout dataset")
    recommendations.append("   - Consider ensemble methods combining top configurations")
    recommendations.append("   - Run additional experiments with intermediate parameter values")

    for rec in recommendations:
        report.append(rec)

    report.append("")

    # Summary
    report.append("## Summary")

    completed = sum(1 for r in results if r.get('status') == 'COMPLETED')
    failed = sum(1 for r in results if r.get('status') == 'FAILED')

    summary_text = f"{completed} out of {len(results)} experiments completed successfully."

    if failed > 0:
        summary_text += f" {failed} experiment(s) failed."

    if best_experiment:
        summary_text += f" The best performing model achieved a test AUC of {best_experiment.get('metrics', {}).get('test_auc', 0):.4f}."

    if any(r.get('metrics') for r in results):
        summary_text += " Detailed metrics comparison shows clear performance differences between configurations."

    report.append(summary_text)

    return "\n".join(report)