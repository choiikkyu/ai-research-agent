#!/usr/bin/env python
"""Local testing script for AI Research Agent."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp.tools import (
    analyze_spec,
    generate_code,
    evaluate_results,
)
from src.core.models import ExperimentResult, TechSpec
from src.core.repo_manager import RepositoryManager


async def test_spec_analysis(spec_url: str):
    """Test specification analysis."""
    print(f"\n📝 Analyzing spec: {spec_url}")
    spec = await analyze_spec(spec_url)

    print(f"  Title: {spec.title}")
    print(f"  Task Type: {spec.task_type}")
    print(f"  Target Repository: {spec.repository}")
    print(f"  Implementation Path: {spec.requirements.get('implementation_path')}")

    return spec


async def test_code_generation(spec: TechSpec):
    """Test code generation."""
    print(f"\n🤖 Generating code for: {spec.title}")

    implementation = await generate_code(spec, spec.repository)

    print(f"  Branch: {implementation.get('branch_name')}")
    print(f"  Files generated: {len(implementation.get('files', {}))}")

    for filename in implementation.get('files', {}).keys():
        print(f"    - {filename}")

    return implementation


async def test_evaluation():
    """Test evaluation with mock data."""
    print("\n📊 Testing evaluation...")

    # Mock model training result
    model_result = ExperimentResult(
        experiment_id="test-001",
        status="SUCCESS",
        metrics={
            "auc": 0.87,
            "logloss": 0.32,
            "calibration_error": 0.015
        },
        pr_url=None,
        pod_name="test-pod",
        recommendations=[]
    )

    model_spec = TechSpec(
        title="Test Model",
        content="Test",
        task_type="MODEL_TRAINING",
        repository="ai-craft"
    )

    model_eval = await evaluate_results(model_result, model_spec)
    print(f"\nModel Training Evaluation:")
    print(f"  Passed: {model_eval.get('passed')}")
    print(f"  Score: {model_eval.get('score')}/100")
    print(f"  Reason: {model_eval.get('reason')}")

    # Mock feature engineering result
    feature_result = ExperimentResult(
        experiment_id="test-002",
        status="SUCCESS",
        metrics={
            "null_ratio": 0.05,
            "importance_score": 0.08,
            "latency_ms": 8.5
        },
        pr_url=None,
        pod_name="test-pod",
        recommendations=[]
    )

    feature_spec = TechSpec(
        title="Test Feature",
        content="Test",
        task_type="FEATURE_ENGINEERING",
        repository="ai-feature-store"
    )

    feature_eval = await evaluate_results(feature_result, feature_spec)
    print(f"\nFeature Engineering Evaluation:")
    print(f"  Passed: {feature_eval.get('passed')}")
    print(f"  Score: {feature_eval.get('score')}/100")
    print(f"  Reason: {feature_eval.get('reason')}")


async def test_repo_manager():
    """Test repository manager."""
    print("\n🔍 Testing Repository Manager...")

    manager = RepositoryManager()

    # Test repository determination
    test_specs = [
        {"task_type": "MODEL_TRAINING", "content": "Train a pCTR model"},
        {"task_type": "FEATURE_ENGINEERING", "content": "Create user features"},
        {"content": "Implement feature pipeline for user behavior"},
        {"content": "Train neural network model for CTR prediction"},
    ]

    for spec in test_specs:
        repo = manager.determine_target_repo(spec)
        path = manager.get_implementation_path(
            repo,
            spec.get("task_type", "MODEL_TRAINING"),
            "test_feature"
        )
        print(f"  Spec: {spec.get('content', spec.get('task_type'))[:50]}...")
        print(f"    -> Repository: {repo}")
        print(f"    -> Path: {path}")


async def main():
    """Main test function."""
    print("🚀 AI Research Agent - Local Test Suite")
    print("=" * 50)

    # Test 1: Repository Manager
    await test_repo_manager()

    # Test 2: Spec Analysis
    test_urls = [
        "https://notion.so/test/model-training-spec",
        "https://notion.so/test/feature-engineering-spec",
    ]

    specs = []
    for url in test_urls:
        spec = await test_spec_analysis(url)
        specs.append(spec)

    # Test 3: Code Generation (optional - requires API key)
    if "--with-claude" in sys.argv:
        for spec in specs[:1]:  # Test with first spec only
            await test_code_generation(spec)
    else:
        print("\n⚠️  Skipping code generation (requires --with-claude flag)")

    # Test 4: Evaluation
    await test_evaluation()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())