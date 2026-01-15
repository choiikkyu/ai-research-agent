"""Parse Airflow DAG files to extract training workflows.

This module reads DAG files from ai-craft repo to dynamically determine
how to run experiments, rather than hardcoding workflow logic.

DAG files location: ai-craft/workflow/airflow_dags/
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowStep(str, Enum):
    """Types of workflow steps."""
    DATASET = "dataset"          # dataset_operator
    TRAIN = "train"              # train()
    CALIBRATE_M1 = "calibrate_m1"  # calibrate_m1()
    CALIBRATE_M3 = "calibrate_m3"  # calibrate_m3()
    MARK_SUCCESS = "mark_success"  # mark_success()
    VALIDATION = "validation"      # model_validation_operator


@dataclass
class TrainingWorkflow:
    """Represents a model's training workflow extracted from DAG."""
    model_name: str
    module_path: str
    steps: List[WorkflowStep]
    dataset_name: Optional[str] = None
    dataset_days: int = 90
    local_training: bool = False
    conf_id_gpu: str = ""
    conf_id_cpu: str = ""

    def needs_dataset(self) -> bool:
        return WorkflowStep.DATASET in self.steps

    def needs_mark_success(self) -> bool:
        return WorkflowStep.MARK_SUCCESS in self.steps

    def filter_steps(self, selected_steps: List[WorkflowStep]) -> "TrainingWorkflow":
        """Create a new workflow with only selected steps.

        Args:
            selected_steps: Steps to include in the workflow

        Returns:
            New TrainingWorkflow with filtered steps
        """
        filtered = [s for s in self.steps if s in selected_steps]
        return TrainingWorkflow(
            model_name=self.model_name,
            module_path=self.module_path,
            steps=filtered,
            dataset_name=self.dataset_name if WorkflowStep.DATASET in filtered else None,
            dataset_days=self.dataset_days,
            local_training=self.local_training,
            conf_id_gpu=self.conf_id_gpu,
            conf_id_cpu=self.conf_id_cpu,
        )

    @staticmethod
    def default_experiment_steps() -> List[WorkflowStep]:
        """Default steps for experiment (train only)."""
        return [WorkflowStep.TRAIN]

    @staticmethod
    def all_steps() -> List[WorkflowStep]:
        """All available steps."""
        return list(WorkflowStep)


@dataclass
class DAGInfo:
    """Information about a DAG file."""
    file_path: Path
    dag_id: str
    models: Dict[str, TrainingWorkflow] = field(default_factory=dict)


class WorkflowParser:
    """Parse Airflow DAG files to extract training workflows."""

    def __init__(self, ai_craft_path: str):
        """Initialize parser with ai-craft repo path.

        Args:
            ai_craft_path: Path to ai-craft repository
        """
        self.ai_craft_path = Path(ai_craft_path)
        self.dag_dir = self.ai_craft_path / "workflow" / "airflow_dags"
        self._cache: Dict[str, TrainingWorkflow] = {}

    def find_workflow_for_model(self, module_path: str) -> Optional[TrainingWorkflow]:
        """Find training workflow for a given model module path.

        Args:
            module_path: e.g., "dable_ai_craft.dsp_models.vodka_v3.internal.per_country.korea_mtl86"

        Returns:
            TrainingWorkflow if found, None otherwise
        """
        # Check cache first
        if module_path in self._cache:
            return self._cache[module_path]

        # Extract model info from path
        model_info = self._parse_module_path(module_path)
        if not model_info:
            logger.warning(f"Could not parse module path: {module_path}")
            return None

        # Find matching DAG file
        dag_file = self._find_dag_file(model_info)
        if not dag_file:
            logger.warning(f"No DAG file found for: {module_path}")
            return self._create_default_workflow(module_path, model_info)

        # Parse DAG file to extract workflow
        workflow = self._parse_dag_file(dag_file, module_path, model_info)
        if workflow:
            self._cache[module_path] = workflow

        return workflow

    def _parse_module_path(self, module_path: str) -> Optional[Dict[str, str]]:
        """Parse module path to extract model info.

        Returns dict with: domain (vodka/whisky), version, model_name
        """
        # Pattern: dable_ai_craft.dsp_models.{domain}_{version}.{...}.{model_name}
        parts = module_path.split(".")

        if len(parts) < 4 or parts[1] != "dsp_models":
            return None

        domain_version = parts[2]  # e.g., "vodka_v3", "whisky_v1"
        model_name = parts[-1]     # e.g., "korea_mtl86"

        # Parse domain and version
        match = re.match(r"(vodka|whisky|wheres)_v(\d+)", domain_version)
        if not match:
            return None

        return {
            "domain": match.group(1),
            "version": match.group(2),
            "domain_version": domain_version,
            "model_name": model_name,
            "full_path": module_path,
        }

    def _find_dag_file(self, model_info: Dict[str, str]) -> Optional[Path]:
        """Find the DAG file containing this model."""
        domain = model_info["domain"]
        version = model_info["version"]

        # DAG file patterns to try
        patterns = [
            f"ai_dsp_model_{domain}_v{version}_8h.py",
            f"ai_dsp_model_{domain}_v{version}_1d.py",
            f"ai_dsp_model_{domain}_v{version}_1h.py",
            f"ai_dsp_model_{domain}_v{version}_6h.py",
        ]

        # Also try wheres for whisky
        if domain == "whisky":
            patterns.extend([
                f"ai_dsp_model_wheres_v{version}_8h.py",
                f"ai_dsp_model_wheres_v{version}_1d.py",
                f"ai_dsp_model_wheres_v{version}_6h.py",
            ])

        for pattern in patterns:
            dag_path = self.dag_dir / pattern
            if dag_path.exists():
                # Check if model is in this file
                if self._model_in_dag_file(dag_path, model_info["model_name"]):
                    return dag_path

        # Fallback: search all DAG files
        for dag_path in self.dag_dir.glob("ai_dsp_model_*.py"):
            if self._model_in_dag_file(dag_path, model_info["model_name"]):
                return dag_path

        return None

    def _model_in_dag_file(self, dag_path: Path, model_name: str) -> bool:
        """Check if model is defined in DAG file."""
        try:
            content = dag_path.read_text()
            # Look for TaskGroup with model name
            return f"group_id='{model_name}'" in content or f".{model_name}." in content
        except Exception:
            return False

    def _parse_dag_file(
        self,
        dag_path: Path,
        module_path: str,
        model_info: Dict[str, str]
    ) -> Optional[TrainingWorkflow]:
        """Parse DAG file to extract workflow for specific model."""
        try:
            content = dag_path.read_text()
            model_name = model_info["model_name"]

            # Find TaskGroup block for this model
            task_group_content = self._extract_task_group(content, model_name)
            if not task_group_content:
                # Model might be directly in another TaskGroup
                task_group_content = self._find_model_context(content, model_name)

            if not task_group_content:
                logger.warning(f"Could not find TaskGroup for {model_name} in {dag_path}")
                return self._create_default_workflow(module_path, model_info)

            # Parse workflow steps
            steps = self._extract_workflow_steps(task_group_content)

            # Extract dataset info if present
            dataset_name = None
            if WorkflowStep.DATASET in steps:
                dataset_name = self._extract_dataset_name(task_group_content)

            # Check for local_training parameter
            local_training = "local_training=True" in task_group_content

            # Extract conf_ids
            conf_id_gpu = self._extract_conf_id(task_group_content, "gpu")
            conf_id_cpu = self._extract_conf_id(task_group_content, "cpu")

            return TrainingWorkflow(
                model_name=model_name,
                module_path=module_path,
                steps=steps,
                dataset_name=dataset_name,
                local_training=local_training,
                conf_id_gpu=conf_id_gpu or f"/ai/dsp_models/{model_info['domain_version']}_train/gpu",
                conf_id_cpu=conf_id_cpu or f"/ai/dsp_models/{model_info['domain_version']}_train/cpu",
            )

        except Exception as e:
            logger.error(f"Error parsing DAG file {dag_path}: {e}")
            return self._create_default_workflow(module_path, model_info)

    def _extract_task_group(self, content: str, model_name: str) -> Optional[str]:
        """Extract TaskGroup block for a model."""
        # Pattern for TaskGroup with model name
        pattern = rf"with TaskGroup\(group_id='{model_name}'.*?\) as .*?:\s*\n(.*?)(?=\nwith TaskGroup|\n[a-zA-Z_]+ = |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def _find_model_context(self, content: str, model_name: str) -> Optional[str]:
        """Find context containing the model (might be nested TaskGroup)."""
        # Look for model import pattern
        import_pattern = rf"\.{model_name}\.train import"

        # Find the section containing this import
        lines = content.split('\n')
        start_idx = None
        brace_count = 0

        for i, line in enumerate(lines):
            if re.search(import_pattern, line):
                # Found the model, now find the enclosing TaskGroup
                for j in range(i, -1, -1):
                    if "with TaskGroup" in lines[j]:
                        start_idx = j
                        break
                break

        if start_idx is None:
            return None

        # Extract from TaskGroup to end of block
        result_lines = []
        indent_level = None

        for i in range(start_idx, len(lines)):
            line = lines[i]
            if indent_level is None and "with TaskGroup" in line:
                indent_level = len(line) - len(line.lstrip())

            if indent_level is not None:
                current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 1
                if line.strip() and current_indent <= indent_level and i > start_idx:
                    break
                result_lines.append(line)

        return '\n'.join(result_lines)

    def _extract_workflow_steps(self, content: str) -> List[WorkflowStep]:
        """Extract workflow steps from TaskGroup content."""
        steps = []

        # Check for dataset_operator
        if "dataset_operator(" in content:
            steps.append(WorkflowStep.DATASET)

        # Check for train
        if "import train" in content or ".train import train" in content:
            steps.append(WorkflowStep.TRAIN)

        # Check for calibration
        if "calibrate_m3" in content:
            steps.append(WorkflowStep.CALIBRATE_M3)
        elif "calibrate_m1" in content:
            steps.append(WorkflowStep.CALIBRATE_M1)

        # Check for mark_success
        if "mark_success" in content:
            steps.append(WorkflowStep.MARK_SUCCESS)

        # Check for model_validation_operator
        if "model_validation_operator" in content:
            steps.append(WorkflowStep.VALIDATION)

        return steps

    def _extract_dataset_name(self, content: str) -> Optional[str]:
        """Extract dataset name from dataset_operator call."""
        match = re.search(r"dataset_name=['\"]([^'\"]+)['\"]", content)
        return match.group(1) if match else None

    def _extract_conf_id(self, content: str, resource_type: str) -> Optional[str]:
        """Extract conf_id for gpu or cpu."""
        pattern = rf"conf_id=['\"]([^'\"]*{resource_type}[^'\"]*)['\"]"
        match = re.search(pattern, content)
        return match.group(1) if match else None

    def _create_default_workflow(
        self,
        module_path: str,
        model_info: Dict[str, str]
    ) -> TrainingWorkflow:
        """Create default workflow when DAG file is not found."""
        domain = model_info["domain"]

        if domain in ("whisky", "wheres"):
            # Wheres workflow
            return TrainingWorkflow(
                model_name=model_info["model_name"],
                module_path=module_path,
                steps=[
                    WorkflowStep.DATASET,
                    WorkflowStep.TRAIN,
                    WorkflowStep.CALIBRATE_M3,
                    WorkflowStep.MARK_SUCCESS,
                ],
                dataset_name="wheres_dataset__click_view__v2",
                local_training=True,
                conf_id_gpu=f"/ai/dsp_models/{model_info['domain_version']}_train/gpu",
                conf_id_cpu=f"/ai/dsp_models/{model_info['domain_version']}_train/cpu",
            )
        else:
            # DNA (vodka) workflow
            return TrainingWorkflow(
                model_name=model_info["model_name"],
                module_path=module_path,
                steps=[
                    WorkflowStep.TRAIN,
                    WorkflowStep.CALIBRATE_M3,
                ],
                local_training=False,
                conf_id_gpu=f"/ai/dsp_models/{model_info['domain_version']}_train/gpu",
                conf_id_cpu=f"/ai/dsp_models/{model_info['domain_version']}_train/cpu",
            )


def generate_training_script_from_workflow(
    workflow: TrainingWorkflow,
    utc_time: str,
    utc_time_iso: str = "",
    utc_time_90d_iso: str = "",
    selected_steps: Optional[List[WorkflowStep]] = None,
) -> List[str]:
    """Generate bash script lines from workflow definition.

    Args:
        workflow: TrainingWorkflow extracted from DAG
        utc_time: Training time in YYYY-MM-DD-HH format
        utc_time_iso: ISO format time for dataset end
        utc_time_90d_iso: ISO format time for dataset start (90 days ago)
        selected_steps: Steps to execute. If None, only TRAIN is executed.

    Returns:
        List of bash script lines
    """
    # Default: only train step for experiments
    if selected_steps is None:
        selected_steps = [WorkflowStep.TRAIN]

    # Filter workflow to selected steps
    workflow = workflow.filter_steps(selected_steps)
    lines = [
        "# ===========================================",
        f"# Training Workflow for {workflow.model_name}",
        f"# Steps: {' -> '.join(s.value for s in workflow.steps)}",
        "# ===========================================",
        "",
    ]

    module_path = workflow.module_path

    # Step 1: Dataset (if needed)
    if workflow.needs_dataset():
        lines.extend([
            "# Step: Create dataset",
            f"echo 'Dataset: {workflow.dataset_name}'",
            f"echo 'Range: {utc_time_90d_iso} to {utc_time_iso}'",
            "# Note: Dataset creation handled by Airflow in production",
            "# For experiment, assuming dataset exists",
            "",
        ])

    # Step 2: Train
    if WorkflowStep.TRAIN in workflow.steps:
        local_training_param = ", local_training=True" if workflow.local_training else ""
        lines.extend([
            "# Step: Training",
            f"echo 'Training module: {module_path}'",
            f"echo 'UTC time: {utc_time}'",
            "",
        ])

        if workflow.local_training:
            # Wheres style - capture result
            lines.extend([
                f'TRAIN_RESULT=$(/home/dable/.venv/bin/python -c "',
                f"import json",
                f"from {module_path}.train import train",
                f"result = train('{utc_time}'{local_training_param})",
                f"print(json.dumps(result) if result else '{{}}')",
                f'")',
                "",
                'echo "Train result: $TRAIN_RESULT"',
                "",
                "# Extract model info",
                "MODEL_NAME=$(echo $TRAIN_RESULT | python -c \"import sys,json; d=json.load(sys.stdin); print(d.get('model_name',''))\")",
                "RUN_ID=$(echo $TRAIN_RESULT | python -c \"import sys,json; d=json.load(sys.stdin); print(d.get('run_info',{}).get('run_id',''))\")",
                "",
            ])
        else:
            # DNA style - simple call
            lines.extend([
                f'python -c "from {module_path}.train import train; train(\'{utc_time}\')"',
                "",
            ])

    # Step 3: Calibration
    if WorkflowStep.CALIBRATE_M3 in workflow.steps:
        lines.append("# Step: Calibration (cal_m3)")
        if workflow.local_training:
            lines.extend([
                'if [ -n "$MODEL_NAME" ] && [ -n "$RUN_ID" ]; then',
                '    echo "Running calibration..."',
                f'    CAL_RESULT=$(/home/dable/.venv/bin/python -c "',
                f"import json",
                f"from {module_path}.train import calibrate_m3",
                f"result = calibrate_m3(",
                f"    utc_ymdh='{utc_time}',",
                f"    model_name='$MODEL_NAME',",
                f"    run_id='$RUN_ID',",
                f"    local_training=True",
                f")",
                f"print(json.dumps(result) if result else '{{}}')",
                f'")',
                "",
                '    echo "Calibration result: $CAL_RESULT"',
                "",
                "    REGISTERED_MODEL_NAME=$(echo $CAL_RESULT | python -c \"import sys,json; d=json.load(sys.stdin); print(d.get('registered_model_name',''))\")",
                "    VERSION=$(echo $CAL_RESULT | python -c \"import sys,json; d=json.load(sys.stdin); print(d.get('version',''))\")",
            ])
        else:
            lines.extend([
                'echo "Running calibration..."',
                f'python -c "from {module_path}.train import calibrate_m3; calibrate_m3(\'{utc_time}\')"',
                "",
            ])
    elif WorkflowStep.CALIBRATE_M1 in workflow.steps:
        lines.extend([
            "# Step: Calibration (cal_m1)",
            'echo "Running calibration (m1)..."',
            f'python -c "from {module_path}.train import calibrate_m1; calibrate_m1(\'{utc_time}\')"',
            "",
        ])

    # Step 4: Mark success (if needed)
    if workflow.needs_mark_success():
        lines.extend([
            "",
            "    # Step: Mark success",
            '    if [ -n "$REGISTERED_MODEL_NAME" ]; then',
            '        echo "Marking success..."',
            f'        /home/dable/.venv/bin/python -c "',
            f"from {module_path}.train import mark_success",
            f"mark_success(",
            f"    model_name='$MODEL_NAME',",
            f"    registered_model_name='$REGISTERED_MODEL_NAME',",
            f"    run_id='$RUN_ID',",
            f"    version='$VERSION'",
            f")",
            f'"',
            '        echo "Success marked!"',
            "    fi",
            "fi",
        ])

    lines.append("")
    lines.append("echo '=== Training completed ==='")

    return lines
