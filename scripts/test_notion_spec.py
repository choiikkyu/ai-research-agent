#!/usr/bin/env python3
"""Test script for AI Research Agent with Notion spec input.

Tests the workflow:
1. Parse Notion spec -> TechSpec
2. Generate code implementation
3. Create Draft PR (simulation)
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta

from src.core.models import TechSpec
from src.mcp.tools.code_generator import PatternBasedCodeGenerator, PRStrategy
from src.mcp.tools.experiment import (
    convert_path_to_module,
    get_utc_time_hours_ago,
    get_utc_time_iso,
    get_utc_time_iso_days_ago,
    generate_training_command,
    prepare_experiment_script,
)
from src.mcp.tools.workflow_parser import (
    WorkflowParser,
    WorkflowStep,
    TrainingWorkflow,
    generate_training_script_from_workflow,
)


async def test_workflow():
    """Test the full workflow with the Notion spec."""

    print("=" * 60)
    print("AI Research Agent - Workflow Test")
    print("=" * 60)

    # =========================================================================
    # Step 1: Parse Notion Spec (Manual since Notion API integration is TODO)
    # =========================================================================
    print("\n[Step 1] Parsing Notion Spec...")

    notion_content = """
    Method:
    - whisky의 base_clk_dcn24의 conf에서 network layer를
      dcn_v2_linear_unit_list를 2 layer로 간소화해서
      base_clk_dcn99라는 모델을 만든다.
    """

    spec = TechSpec(
        title="Create base_clk_dcn99 model with 2-layer DCN",
        content=notion_content,
        task_type="MODEL_TRAINING",
        repository="ai-craft",
        requirements={
            "reference_model": "base_clk_dcn24",
            "new_model": "base_clk_dcn99",
            "modification": "dcn_v2_linear_unit_list: [1024, 512, 256] -> [1024, 512]",
            "location": "whisky_v1"
        }
    )

    print(f"  Title: {spec.title}")
    print(f"  Task Type: {spec.task_type}")
    print(f"  Repository: {spec.repository}")
    print(f"  Reference Model: {spec.requirements.get('reference_model')}")
    print(f"  New Model: {spec.requirements.get('new_model')}")

    # =========================================================================
    # Step 2: Test Code Generator Context Collection
    # =========================================================================
    print("\n[Step 2] Testing Code Generator Context Collection...")

    generator = PatternBasedCodeGenerator()

    # Get context for generation
    reference_path = "src/dable_ai_craft/dsp_models/whisky_v1/base_clk_dcn24"
    context = await generator._github_ref.get_context_for_generation(
        repo_name="ai-craft",
        task_type="MODEL_TRAINING",
        reference_path=reference_path,
        keywords=["base_clk_dcn24", "whisky"]
    )

    print(f"  Conventions (CLAUDE.md): {len(context.get('conventions', ''))} chars")
    print(f"  Reference Path: {context.get('reference_path')}")
    print(f"  Reference Files: {list(context.get('reference_files', {}).keys())}")

    # =========================================================================
    # Step 3: Test PR Strategy Detection
    # =========================================================================
    print("\n[Step 3] Testing PR Strategy Detection...")

    pr_strategy = generator._determine_pr_strategy(spec, context)
    print(f"  PR Strategy: {pr_strategy.value}")
    print(f"  Expected: model_modification (2-commit strategy)")

    if pr_strategy == PRStrategy.MODEL_MODIFICATION:
        print("  ✅ Correct! Will use 2-commit strategy")
    else:
        print("  ⚠️ Using standard strategy (reference files may be < 3)")

    # =========================================================================
    # Step 4: Test Path to Module Conversion
    # =========================================================================
    print("\n[Step 4] Testing Path to Module Conversion...")

    new_model_path = "src/dable_ai_craft/dsp_models/whisky_v1/base_clk_dcn99"
    module_path = convert_path_to_module(new_model_path)

    print(f"  File Path: {new_model_path}")
    print(f"  Module Path: {module_path}")

    # =========================================================================
    # Step 5: Test Training Command Generation
    # =========================================================================
    print("\n[Step 5] Testing Training Command Generation...")

    utc_time = get_utc_time_hours_ago(4)  # Format: YYYY-MM-DD-HH (e.g., 2026-01-15-02)
    training_cmd = generate_training_command(module_path, utc_time)

    print(f"  UTC Time (4h ago): {utc_time}")  # e.g., 2026-01-15-02
    print(f"  Training Command:")
    print(f"    {training_cmd}")

    # =========================================================================
    # Step 6: Test Workflow Parser (Dynamic DAG Parsing)
    # =========================================================================
    print("\n[Step 6] Testing Workflow Parser...")

    # Try to parse workflow from ai-craft DAG files
    ai_craft_path = "/Users/choieq/Projects/dable/ai-craft"
    try:
        parser = WorkflowParser(ai_craft_path)
        workflow = parser.find_workflow_for_model(module_path)

        if workflow:
            print(f"  ✅ Found workflow for {workflow.model_name}")
            print(f"  Module: {workflow.module_path}")
            print(f"  Dataset: {workflow.dataset_name or 'N/A'}")
            print(f"  Local Training: {workflow.local_training}")
            print(f"  Full Steps: {' -> '.join(s.value for s in workflow.steps)}")

            # Show default (train only)
            default_workflow = workflow.filter_steps([WorkflowStep.TRAIN])
            print(f"\n  Default (train only): {' -> '.join(s.value for s in default_workflow.steps)}")

            # Show with calibration
            with_cal = workflow.filter_steps([WorkflowStep.TRAIN, WorkflowStep.CALIBRATE_M3])
            print(f"  With calibration: {' -> '.join(s.value for s in with_cal.steps)}")
        else:
            print(f"  ⚠️ No workflow found for {module_path}")
            print("  Will use fallback workflow")
    except Exception as e:
        print(f"  ⚠️ Could not parse workflow: {e}")
        workflow = None

    # =========================================================================
    # Step 7: Test Experiment Script Generation
    # =========================================================================
    print("\n[Step 7] Testing Experiment Script Generation...")

    mock_implementation = {
        "branch_name": f"ai-agent-base_clk_dcn99-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "repository": "ai-craft",
        "implementation_path": new_model_path,
        "files": {},
    }

    # Test with default steps (train only)
    print("  [Default: train only]")
    script_default = prepare_experiment_script(
        mock_implementation, spec, ai_craft_path=ai_craft_path, selected_steps=["train"]
    )

    print("  Generated Script (key lines):")
    for i, line in enumerate(script_default.split("\n")):
        if any(kw in line for kw in ["echo '===", "Selected steps", "python", "train", "calibrate"]):
            print(f"    {i+1:2}: {line}")

    # Test with train + calibration
    print("\n  [With calibration: train, calibrate_m3]")
    script_with_cal = prepare_experiment_script(
        mock_implementation, spec, ai_craft_path=ai_craft_path,
        selected_steps=["train", "calibrate_m3"]
    )

    print("  Generated Script (key lines):")
    for i, line in enumerate(script_with_cal.split("\n")):
        if any(kw in line for kw in ["echo '===", "Selected steps", "python", "train", "calibrate"]):
            print(f"    {i+1:2}: {line}")

    # =========================================================================
    # Step 8: Summary - What would happen in real execution
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY - Real Execution Would:")
    print("=" * 60)

    print(f"""
1. Create Draft PR in ai-craft repo:
   - Branch: {mock_implementation['branch_name']}
   - Copy base_clk_dcn24 -> base_clk_dcn99 (Commit 1)
   - Modify dcn_v2_linear_unit_list (Commit 2)

2. Wait for user approval

3. On approval, run on K8s GPU pod (tf-box namespace):
   {training_cmd}

4. Collect metrics and evaluate results

5. Update PR based on results
""")

    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return {
        "spec": spec.model_dump(),
        "reference_path": reference_path,
        "new_model_path": new_model_path,
        "module_path": module_path,
        "training_command": training_cmd,
        "pr_strategy": pr_strategy.value,
    }


if __name__ == "__main__":
    result = asyncio.run(test_workflow())
    print("\n\nTest Result JSON:")
    print(json.dumps(result, indent=2, default=str))
