#!/usr/bin/env python
"""Test K8s Pod creation using ak-launch-tf-box.

This script tests:
1. Pod creation with ak-launch-tf-box
2. Pod status monitoring
3. Pod cleanup
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

from src.k8s.pod_launcher import K8sPodLauncher


# Test configuration
POD_TYPE = "cpu"  # Use CPU for cheaper test
INSTANCE_TYPE = "t3.large"  # Smallest instance for test
SKIP_CLEANUP = False  # Set to True to keep pod running


async def test_pod_creation():
    """Test K8s pod creation."""
    print("\n" + "="*60)
    print("☸️ Testing K8s Pod Creation")
    print("="*60)

    launcher = K8sPodLauncher()

    # Generate pod name
    timestamp = datetime.now().strftime("%H%M%S")
    pod_name = f"ai-tf-box-test-{timestamp}"

    print(f"\n📋 Pod Configuration:")
    print(f"   Name: {pod_name}")
    print(f"   Type: {POD_TYPE}")
    print(f"   Instance: {INSTANCE_TYPE}")
    print(f"   Namespace: {launcher.namespace}")

    # Create pod
    print(f"\n🚀 Creating pod...")

    try:
        pod_info = await launcher.launch_pod(
            pod_name=pod_name,
            pod_type=POD_TYPE,
            instance_type=f"{INSTANCE_TYPE}:ondemand"
        )

        print(f"\n✅ Pod created successfully!")
        print(f"   Pod name: {pod_info.get('pod_name')}")
        print(f"   Status: {pod_info.get('status')}")
        print(f"   Instance: {pod_info.get('instance_type')}")

        # Get pod status
        print(f"\n📊 Checking pod status...")
        status = await launcher.get_pod_status(pod_name)
        print(f"   Current status: {status}")

        # Execute simple command
        print(f"\n🔧 Executing test command on pod...")
        result = await launcher.execute_on_pod(
            pod_name=pod_name,
            script="echo 'Hello from AI Research Agent!' && python --version && nvidia-smi 2>/dev/null || echo 'No GPU available'",
            timeout=60
        )

        if result.get("success"):
            print(f"✅ Command executed successfully!")
            print(f"Output:\n{result.get('stdout', '')[:500]}")
        else:
            print(f"⚠️ Command execution had issues:")
            print(f"   stderr: {result.get('stderr', '')[:200]}")

        # Cleanup
        if not SKIP_CLEANUP:
            print(f"\n🧹 Cleaning up pod...")
            cleanup_success = await launcher.cleanup_pod(pod_name)
            if cleanup_success:
                print(f"✅ Pod cleaned up successfully!")
            else:
                print(f"⚠️ Pod cleanup may have had issues")
        else:
            print(f"\n⚠️ Skipping cleanup (SKIP_CLEANUP=True)")
            print(f"   To clean up manually: helm uninstall chart-{pod_name} -n tf-box")

        return {
            "success": True,
            "pod_name": pod_name,
            "status": status
        }

    except Exception as e:
        print(f"\n❌ Pod creation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


async def main():
    """Run the test."""
    print("\n" + "🚀"*30)
    print("AI Research Agent - K8s Pod Test")
    print("🚀"*30)

    print(f"\n📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check kubectl context
    print(f"\n🔍 Checking kubectl...")
    import subprocess
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True
        )
        print(f"   Current context: {result.stdout.strip()}")
    except:
        print("   ⚠️ kubectl not available or not configured")

    # Check ak-launch-tf-box
    print(f"\n🔍 Checking ak-launch-tf-box...")
    try:
        result = subprocess.run(
            ["which", "ak-launch-tf-box"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"   Found: {result.stdout.strip()}")
        else:
            print("   ❌ ak-launch-tf-box not found!")
            return 1
    except:
        print("   ⚠️ Could not check ak-launch-tf-box")

    # Run test
    result = await test_pod_creation()

    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    if result and result.get("success"):
        print("✅ K8s Pod creation: PASSED")
        return 0
    else:
        print("❌ K8s Pod creation: FAILED")
        if result:
            print(f"Error: {result.get('error')}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)