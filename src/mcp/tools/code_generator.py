"""Code generator using Claude API with pattern-based generation.

This module generates code by:
1. Loading CLAUDE.md conventions from target repo
2. Finding and reading similar existing implementations
3. Using Claude API to generate new code following the same patterns

## PR Strategy Selection
- MODEL_MODIFICATION: Use 2-commit strategy (copy first, then modify)
- NEW_IMPLEMENTATION: Use standard single-commit PR
See .claude/COMMIT_STRATEGY.md for details.
"""

import logging
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from src.core.config import settings
from src.core.models import TechSpec
from src.core.github_code_reference import GitHubCodeReference

logger = logging.getLogger(__name__)


class PRStrategy(Enum):
    """PR creation strategy based on task type.

    MODEL_MODIFICATION: When creating a new model based on existing one.
        - Uses 2-commit strategy for clear diff review
        - Commit 1: Copy with name changes only
        - Commit 2: Apply actual modifications

    NEW_IMPLEMENTATION: Standard new code implementation.
        - Uses single commit with all generated files
    """
    MODEL_MODIFICATION = "model_modification"  # 2-commit strategy
    NEW_IMPLEMENTATION = "new_implementation"  # Standard single commit


class PatternBasedCodeGenerator:
    """Pattern-based code generator using Claude API.

    Generates code that follows existing repository patterns by:
    1. Reading CLAUDE.md for conventions
    2. Finding similar implementations as reference
    3. Generating new code with the same structure
    """

    def __init__(self):
        """Initialize code generator."""
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self._github_ref = GitHubCodeReference()

    async def generate(
        self,
        spec: TechSpec,
        reference_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate code implementation based on spec and reference.

        Args:
            spec: Technical specification
            reference_path: Optional explicit reference implementation path

        Returns:
            Implementation dict with files and metadata
        """
        logger.info(f"Generating code for: {spec.title}")

        # Determine target repository
        target_repo = self._github_ref.determine_target_repo({
            "task_type": spec.task_type,
            "content": spec.content,
            "repository": spec.repository
        })

        # Get reference model path from spec requirements
        ref_model_name = None
        if hasattr(spec, 'requirements') and spec.requirements:
            ref_model_name = spec.requirements.get('reference_model')

        # Find reference path based on model name
        resolved_reference_path = reference_path
        if ref_model_name and not reference_path:
            resolved_reference_path = await self._find_model_path(target_repo, ref_model_name)

        # Collect context (CLAUDE.md + reference files)
        context = await self._github_ref.get_context_for_generation(
            repo_name=target_repo,
            task_type=spec.task_type,
            reference_path=resolved_reference_path,
            keywords=self._extract_keywords(spec)
        )

        # Build prompt with context
        prompt = self._build_prompt(spec, context)

        # Generate code using Claude
        try:
            response = await self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                temperature=0.2,
                system=self._get_system_prompt(context),
                messages=[{"role": "user", "content": prompt}]
            )

            generated_content = response.content[0].text if response.content else ""
            code_blocks = self._extract_code_blocks(generated_content)

            # Determine implementation path
            impl_path = self._github_ref.get_implementation_path(
                target_repo,
                spec.task_type,
                self._extract_model_name(spec)
            )

            # Determine PR strategy
            pr_strategy = self._determine_pr_strategy(spec, context)

            implementation = {
                "branch_name": f"ai-agent-{spec.task_type.lower()}-{self._generate_branch_suffix()}",
                "files": code_blocks,
                "task_type": spec.task_type,
                "repository": target_repo,
                "implementation_path": impl_path,
                "reference_path": context.get("reference_path"),
                "reference_name": self._extract_reference_name(context.get("reference_path")),
                "spec_title": spec.title,
                "generated_content": generated_content,
                "pr_strategy": pr_strategy.value,  # "model_modification" or "new_implementation"
            }

            logger.info(
                f"Generated {len(code_blocks)} files for {target_repo}/{impl_path}"
            )
            return implementation

        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            raise

    def _get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with conventions."""
        base_prompt = (
            "You are an expert AI engineer specializing in pCTR/pCVR models "
            "and feature engineering. Generate production-ready code that "
            "follows the existing repository patterns exactly."
        )

        if context.get("conventions"):
            base_prompt += f"\n\n## Repository Conventions (CLAUDE.md)\n\n{context['conventions']}"

        return base_prompt

    def _build_prompt(self, spec: TechSpec, context: Dict[str, Any]) -> str:
        """Build generation prompt with spec and reference code."""
        prompt_parts = []

        # Task description
        prompt_parts.append(f"## Task: {spec.title}\n")
        prompt_parts.append(f"Task Type: {spec.task_type}\n")
        prompt_parts.append(f"Repository: {context['repo_name']}\n")
        prompt_parts.append(f"\n## Requirements\n{spec.content}\n")

        # Reference implementation
        if context.get("reference_path") and context.get("reference_files"):
            prompt_parts.append(f"\n## Reference Implementation: {context['reference_path']}\n")
            prompt_parts.append(
                "Use this existing implementation as a template. "
                "Follow the same file structure, class patterns, and coding style.\n"
            )

            # Directory structure
            if context.get("directory_structure"):
                prompt_parts.append("\n### Directory Structure\n```\n")
                for item in context["directory_structure"]:
                    icon = "📁" if item["type"] == "dir" else "📄"
                    prompt_parts.append(f"{icon} {item['name']}\n")
                prompt_parts.append("```\n")

            # Reference files
            prompt_parts.append("\n### Reference Files\n")
            for filename, content in context["reference_files"].items():
                # Truncate very long files
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                prompt_parts.append(f"\n#### {filename}\n```python\n{content}\n```\n")

        # Generation instructions
        prompt_parts.append("\n## Instructions\n")
        prompt_parts.append(
            "1. Create a new implementation following the same structure as the reference\n"
            "2. Maintain the same file organization and naming conventions\n"
            "3. Use the same base classes and imports where applicable\n"
            "4. Apply the modifications described in the requirements\n"
            "5. Ensure all code is production-ready with proper error handling\n"
        )

        prompt_parts.append(
            "\n## Output Format\n"
            "Generate each file with clear boundaries:\n"
            "```python:filename.py\n"
            "code content\n"
            "```\n"
        )

        return "".join(prompt_parts)

    async def _find_model_path(self, repo_name: str, model_name: str) -> Optional[str]:
        """Find the path to a model by its name.

        Args:
            repo_name: Repository name
            model_name: Model name to find (e.g., base_clk_dcn_24)

        Returns:
            Full path to the model directory if found
        """
        # Common model directories in ai-craft (prioritize whisky_v1 first)
        search_paths = [
            "src/dable_ai_craft/dsp_models/whisky_v1/",
            "src/dable_ai_craft/dsp_models/vodka_v3/external/per_ssp/",
            "src/dable_ai_craft/dsp_models/vodka_v3/internal/",
        ]

        logger.info(f"Searching for model {model_name} in {repo_name}")

        # Also try common variations
        model_variations = [
            model_name,  # exact
            model_name.replace("_", ""),  # no underscores: base_clk_dcn_24 -> baseclkdcn24
            model_name.replace("-", ""),  # no dashes
        ]

        # Special case for numbered models like base_clk_dcn_24 -> base_clk_dcn24
        import re
        numbered_pattern = r'(.+)_(\d+)$'
        match = re.match(numbered_pattern, model_name)
        if match:
            base_part, number = match.groups()
            numbered_variant = f"{base_part}{number}"
            model_variations.append(numbered_variant)

        for base_path in search_paths:
            try:
                logger.debug(f"Searching in {base_path}")
                # List directories in the base path
                contents = await self._github_ref.list_directory(repo_name, base_path)
                for item in contents:
                    if item["type"] == "dir":
                        # Try all variations
                        for variation in model_variations:
                            if item["name"] == variation:
                                logger.info(f"Found model: {item['name']} for {model_name} in {base_path}")
                                return f"{base_path}{item['name']}"
            except Exception as e:
                logger.debug(f"Failed to search in {base_path}: {e}")
                continue

        logger.warning(f"Model {model_name} not found in {repo_name}")
        return None

    def _extract_keywords(self, spec: TechSpec) -> List[str]:
        """Extract keywords from spec for finding similar implementations."""
        keywords = []

        # Extract from title
        title_words = spec.title.lower().split()
        keywords.extend([w for w in title_words if len(w) > 3])

        # Extract model/feature names from content
        content_lower = spec.content.lower()

        # Common patterns
        patterns = [
            r'model[:\s]+(\w+)',
            r'based on[:\s]+(\w+)',
            r'reference[:\s]+(\w+)',
            r'similar to[:\s]+(\w+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content_lower)
            keywords.extend(matches)

        return list(set(keywords))[:5]  # Limit to 5 keywords

    def _extract_model_name(self, spec: TechSpec) -> Optional[str]:
        """Extract model/feature name from spec."""
        # Try to find explicit name in requirements
        if hasattr(spec, 'requirements') and spec.requirements:
            # First try new_model_name (from parsed content)
            if 'new_model_name' in spec.requirements:
                return spec.requirements['new_model_name']
            if 'name' in spec.requirements:
                return spec.requirements['name']
            if 'model_name' in spec.requirements:
                return spec.requirements['model_name']

        # Extract from title
        title = spec.title.lower()
        # Remove common prefixes/suffixes
        for remove in ['new', 'add', 'create', 'implement', 'model', 'feature']:
            title = title.replace(remove, '')

        words = [w.strip() for w in title.split() if w.strip()]
        if words:
            return '_'.join(words[:2])

        return None

    def _extract_code_blocks(self, content: str) -> Dict[str, str]:
        """Extract code blocks from Claude's response."""
        code_blocks = {}

        # Pattern to match code blocks with filename
        pattern = r"```(?:python|yaml|txt|json):([^\n]+)\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        for filename, code in matches:
            filename = filename.strip()
            code = code.strip()
            code_blocks[filename] = code

        # If no labeled blocks found, try generic code blocks
        if not code_blocks:
            generic_pattern = r"```(?:python|yaml)?\n(.*?)```"
            generic_matches = re.findall(generic_pattern, content, re.DOTALL)

            for i, code in enumerate(generic_matches):
                filename = self._infer_filename(code, i)
                code_blocks[filename] = code.strip()

        return code_blocks

    def _infer_filename(self, code: str, index: int) -> str:
        """Infer filename from code content."""
        code_lower = code.lower()

        if "class" in code and "model" in code_lower:
            return "model.py"
        elif "def train" in code_lower or "training" in code_lower:
            return "train.py"
        elif "config" in code_lower and ("yaml" in code_lower or ":" in code):
            return "config.yaml"
        elif "feature" in code_lower:
            return "feature_pipeline.py"
        elif "test" in code_lower:
            return "test.py"

        return f"generated_{index}.py"

    def _generate_branch_suffix(self) -> str:
        """Generate unique branch suffix."""
        return str(uuid.uuid4())[:8]

    def _determine_pr_strategy(self, spec: TechSpec, context: Dict[str, Any]) -> PRStrategy:
        """Determine which PR strategy to use.

        Use MODEL_MODIFICATION (2-commit strategy) when:
        1. Task type is MODEL_TRAINING
        2. There's a reference model being used as base
        3. The reference has multiple files (>=3)

        Otherwise use NEW_IMPLEMENTATION (standard single commit).

        Args:
            spec: Technical specification
            context: Generation context with reference info

        Returns:
            PRStrategy enum value
        """
        # Check if this is a model training task
        if spec.task_type != "MODEL_TRAINING":
            logger.debug(f"PR strategy: NEW_IMPLEMENTATION (task_type={spec.task_type})")
            return PRStrategy.NEW_IMPLEMENTATION

        # Check if there's a reference model
        reference_path = context.get("reference_path")
        if not reference_path:
            logger.debug("PR strategy: NEW_IMPLEMENTATION (no reference model)")
            return PRStrategy.NEW_IMPLEMENTATION

        # Check if reference model path is in model directories
        model_indicators = ["dsp_models", "models", "vodka", "whisky"]
        is_model_path = any(ind in reference_path.lower() for ind in model_indicators)
        if not is_model_path:
            logger.debug(f"PR strategy: NEW_IMPLEMENTATION (reference not a model: {reference_path})")
            return PRStrategy.NEW_IMPLEMENTATION

        # Check if reference has multiple files
        reference_files = context.get("reference_files", {})
        if len(reference_files) < 3:
            logger.debug(f"PR strategy: NEW_IMPLEMENTATION (only {len(reference_files)} reference files)")
            return PRStrategy.NEW_IMPLEMENTATION

        # All conditions met - use 2-commit strategy
        logger.info(
            f"PR strategy: MODEL_MODIFICATION (2-commit strategy) "
            f"- reference: {reference_path}, files: {len(reference_files)}"
        )
        return PRStrategy.MODEL_MODIFICATION

    def _extract_reference_name(self, reference_path: Optional[str]) -> Optional[str]:
        """Extract model/feature name from reference path.

        Args:
            reference_path: Path like "src/.../dsp_models/vodka_v3/external/per_ssp/xandr_mtl93"

        Returns:
            Name like "xandr_mtl93" or None
        """
        if not reference_path:
            return None

        # Get the last directory name
        path = reference_path.rstrip("/")
        return path.split("/")[-1]


# Backwards compatibility - expose as function
async def generate_code(spec: TechSpec, target_repo: str = None) -> Dict[str, Any]:
    """Generate code implementation (backwards compatible).

    Args:
        spec: Technical specification
        target_repo: Target repository name (optional, auto-detected if not provided)

    Returns:
        Implementation dict with files and metadata
    """
    generator = PatternBasedCodeGenerator()
    return await generator.generate(spec, reference_path=None)


async def generate_code_with_reference(
    spec: TechSpec,
    reference_path: str
) -> Dict[str, Any]:
    """Generate code using explicit reference implementation.

    Args:
        spec: Technical specification
        reference_path: Path to reference implementation (e.g., "dsp_models/xandr_mtl93")

    Returns:
        Implementation dict with files and metadata
    """
    generator = PatternBasedCodeGenerator()
    return await generator.generate(spec, reference_path=reference_path)
