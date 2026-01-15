#!/usr/bin/env python
"""End-to-end test script for AI Research Agent.

This script tests:
1. Code generation with Claude API
2. GitHub branch and draft PR creation
3. K8s pod creation (optional)
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Verify required environment variables
REQUIRED_VARS = ["ANTHROPIC_API_KEY", "GITHUB_TOKEN"]
missing = [v for v in REQUIRED_VARS if not os.getenv(v) or os.getenv(v) == "your-api-key-here"]
if missing:
    print(f"❌ Missing required environment variables: {missing}")
    print("Please set them in .env file")
    sys.exit(1)

from src.mcp.tools.code_generator import generate_code
from src.mcp.tools.github import manage_pr
from src.core.models import TechSpec
from src.core.config import settings

# Test configuration
TEST_REPO = "ai-craft"  # or "ai-feature-store"
TEST_TASK_TYPE = "MODEL_TRAINING"
TEST_SKIP_K8S = True  # Set to False to test K8s pod creation


async def test_code_generation():
    """Test code generation with Claude API."""
    print("\n" + "="*60)
    print("🤖 STEP 1: Testing Code Generation with Claude API")
    print("="*60)

    # Create test spec
    spec = TechSpec(
        title="Test pCTR Model - Whisky Campaign",
        content="""
        Implement a pCTR (predicted Click-Through Rate) model for the Whisky campaign.

        Requirements:
        - Use TensorFlow/Keras for model implementation
        - Input features: user_id, ad_id, context features
        - Output: CTR probability (0-1)
        - Include model training script with MLflow tracking
        - Add basic data preprocessing

        Target metrics:
        - AUC > 0.85
        - LogLoss < 0.35
        """,
        task_type=TEST_TASK_TYPE,
        repository=TEST_REPO,
        requirements={
            "gpu_required": True,
            "memory_gb": 64,
            "implementation_path": "models/whisky_ctr/"
        }
    )

    print(f"📋 Spec: {spec.title}")
    print(f"📁 Target repo: {spec.repository}")
    print(f"🎯 Task type: {spec.task_type}")

    try:
        implementation = await generate_code(spec, spec.repository)

        print(f"\n✅ Code generation successful!")
        print(f"📂 Branch: {implementation.get('branch_name')}")
        print(f"📝 Files generated: {len(implementation.get('files', {}))}")

        for filename, content in implementation.get('files', {}).items():
            lines = len(content.split('\n'))
            print(f"   - {filename} ({lines} lines)")

        return implementation, spec

    except Exception as e:
        print(f"\n❌ Code generation failed: {str(e)}")
        raise


async def test_github_pr(implementation: dict, spec: TechSpec):
    """Test GitHub branch and draft PR creation."""
    print("\n" + "="*60)
    print("🐙 STEP 2: Testing GitHub Draft PR Creation")
    print("="*60)

    # Create mock evaluation for draft PR
    mock_evaluation = {
        "passed": False,  # Draft PR - not yet evaluated
        "score": 0,
        "task_type": spec.task_type,
        "experiment_id": "e2e-test-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "reason": "Draft PR - pending experiment",
        "details": {},
        "metrics": {},
        "recommendations": ["Run experiment to evaluate"]
    }

    print(f"📋 Creating draft PR for branch: {implementation.get('branch_name')}")
    print(f"📁 Repository: {settings.github_org}/{implementation.get('repository')}")

    try:
        pr_result = await manage_pr(
            implementation=implementation,
            evaluation=mock_evaluation,
            auto_merge=False  # Never auto-merge in test
        )

        if pr_result.get("error"):
            print(f"\n⚠️ PR creation had issues: {pr_result.get('error')}")
        else:
            print(f"\n✅ PR created successfully!")
            print(f"🔗 PR URL: {pr_result.get('pr_url')}")
            print(f"📌 PR Number: #{pr_result.get('pr_number')}")
            print(f"🏷️ Status: {pr_result.get('status')}")

        return pr_result

    except Exception as e:
        print(f"\n❌ PR creation failed: {str(e)}")
        raise


async def test_k8s_pod():
    """Test K8s pod creation (optional)."""
    if TEST_SKIP_K8S:
        print("\n" + "="*60)
        print("⏭️ STEP 3: Skipping K8s Pod Creation (TEST_SKIP_K8S=True)")
        print("="*60)
        return None

    print("\n" + "="*60)
    print("☸️ STEP 3: Testing K8s Pod Creation")
    print("="*60)

    from src.k8s.pod_launcher import K8sPodLauncher

    launcher = K8sPodLauncher()
    pod_name = f"ai-tf-box-e2e-test-{datetime.now().strftime('%H%M%S')}"

    print(f"📋 Creating test pod: {pod_name}")
    print(f"🏷️ Type: cpu")
    print(f"📦 Instance: t3.large:ondemand (minimal for test)")

    try:
        pod_info = await launcher.launch_pod(
            pod_name=pod_name,
            pod_type="cpu",
            instance_type="t3.large:ondemand"  # Minimal instance for test
        )

        print(f"\n✅ Pod created successfully!")
        print(f"📌 Pod name: {pod_info.get('pod_name')}")
        print(f"🏷️ Status: {pod_info.get('status')}")

        # Cleanup
        print(f"\n🧹 Cleaning up test pod...")
        await launcher.cleanup_pod(pod_name)
        print(f"✅ Pod cleaned up")

        return pod_info

    except Exception as e:
        print(f"\n❌ Pod creation failed: {str(e)}")
        print("This might be expected if K8s cluster is not accessible")
        return None


async def main():
    """Run end-to-end tests."""
    print("\n" + "🚀"*30)
    print("AI Research Agent - End-to-End Test")
    print("🚀"*30)

    print(f"\n📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Environment: {settings.environment}")
    print(f"📁 Target repository: {TEST_REPO}")

    results = {
        "code_generation": False,
        "github_pr": False,
        "k8s_pod": None
    }

    try:
        # Step 1: Code Generation
        implementation, spec = await test_code_generation()
        results["code_generation"] = True

        # Step 2: GitHub PR
        pr_result = await test_github_pr(implementation, spec)
        results["github_pr"] = not pr_result.get("error")

        # Step 3: K8s Pod (optional)
        if not TEST_SKIP_K8S:
            pod_result = await test_k8s_pod()
            results["k8s_pod"] = pod_result is not None
        else:
            results["k8s_pod"] = "skipped"

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        if result == "skipped":
            status = "⏭️ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {test_name}: {status}")

    print("\n" + "="*60)

    # Return exit code based on critical tests
    if results["code_generation"] and results["github_pr"]:
        print("🎉 All critical tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)