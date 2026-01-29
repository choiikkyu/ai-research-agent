"""GitHub API based code reference manager.

This module provides access to target repositories via GitHub API
without requiring local clones. It fetches code on-demand for:
- CLAUDE.md conventions
- Similar implementation references
- Directory structure analysis
"""

import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache
import base64

from github import Github
from github.GithubException import GithubException

from src.core.config import settings

logger = logging.getLogger(__name__)


class GitHubCodeReference:
    """GitHub API based code reference (no clone required).

    Provides efficient access to repository code for:
    1. Reading CLAUDE.md conventions
    2. Finding similar implementations
    3. Reading reference files for pattern-based generation
    """

    # Supported repositories (Model Training only)
    REPOS = {
        "ai-craft": {
            "full_name": "teamdable/ai-craft",
            "model_dirs": [
                "src/dable_ai_craft/dsp_models/vodka_v3/external/per_ssp/",
                "src/dable_ai_craft/dsp_models/vodka_v3/internal/",
                "src/dable_ai_craft/dsp_models/whisky_v1/",
            ],
            "default_task": "MODEL_TRAINING"
        }
    }

    def __init__(self):
        """Initialize GitHub client."""
        self._github = Github(settings.github_token.get_secret_value())
        self._claude_md_cache: Dict[str, str] = {}

    def _get_repo(self, repo_name: str):
        """Get PyGithub repository object."""
        if repo_name not in self.REPOS:
            raise ValueError(f"Unknown repository: {repo_name}. Supported: {list(self.REPOS.keys())}")
        return self._github.get_repo(self.REPOS[repo_name]["full_name"])

    # ===== Level 1: CLAUDE.md (Conventions) =====

    async def get_claude_md(self, repo_name: str) -> str:
        """Get CLAUDE.md content (cached).

        Args:
            repo_name: Repository name (ai-craft or ai-feature-store)

        Returns:
            CLAUDE.md content as string
        """
        # Check cache
        if repo_name in self._claude_md_cache:
            logger.debug(f"Using cached CLAUDE.md for {repo_name}")
            return self._claude_md_cache[repo_name]

        try:
            content = await self.get_file_content(repo_name, "CLAUDE.md")
            self._claude_md_cache[repo_name] = content
            logger.info(f"Loaded CLAUDE.md for {repo_name} ({len(content)} chars)")
            return content
        except GithubException as e:
            logger.warning(f"CLAUDE.md not found in {repo_name}: {e}")
            return ""

    # ===== Level 2: Reference Files =====

    async def get_file_content(
        self,
        repo_name: str,
        path: str,
        ref: str = "main"
    ) -> str:
        """Get file content from GitHub.

        Args:
            repo_name: Repository name
            path: File path within repository
            ref: Git reference (branch, tag, or commit)

        Returns:
            File content as string
        """
        repo = self._get_repo(repo_name)
        try:
            content = repo.get_contents(path, ref=ref)
            if isinstance(content, list):
                raise ValueError(f"Path is a directory, not a file: {path}")

            # Decode base64 content
            decoded = base64.b64decode(content.content).decode('utf-8')
            logger.debug(f"Loaded {path} from {repo_name} ({len(decoded)} chars)")
            return decoded
        except GithubException as e:
            logger.error(f"Failed to get {path} from {repo_name}: {e}")
            raise

    async def list_directory(
        self,
        repo_name: str,
        path: str,
        ref: str = "main"
    ) -> List[Dict[str, Any]]:
        """List directory contents.

        Args:
            repo_name: Repository name
            path: Directory path
            ref: Git reference

        Returns:
            List of {name, type, path, size} dicts
        """
        repo = self._get_repo(repo_name)
        try:
            contents = repo.get_contents(path, ref=ref)
            if not isinstance(contents, list):
                contents = [contents]

            return [
                {
                    "name": c.name,
                    "type": c.type,  # "file" or "dir"
                    "path": c.path,
                    "size": c.size if c.type == "file" else 0
                }
                for c in contents
            ]
        except GithubException as e:
            logger.error(f"Failed to list {path} in {repo_name}: {e}")
            return []

    async def get_directory_files(
        self,
        repo_name: str,
        path: str,
        extensions: List[str] = None,
        max_files: int = 10
    ) -> Dict[str, str]:
        """Get all files in a directory.

        Args:
            repo_name: Repository name
            path: Directory path
            extensions: Filter by file extensions (e.g., [".py", ".yaml"])
            max_files: Maximum number of files to read

        Returns:
            Dict mapping filename to content
        """
        if extensions is None:
            extensions = [".py", ".yaml", ".yml", ".json"]

        contents = await self.list_directory(repo_name, path)
        files = {}

        for item in contents:
            if item["type"] != "file":
                continue

            # Check extension
            if not any(item["name"].endswith(ext) for ext in extensions):
                continue

            if len(files) >= max_files:
                break

            try:
                content = await self.get_file_content(repo_name, item["path"])
                files[item["name"]] = content
            except Exception as e:
                logger.warning(f"Failed to read {item['path']}: {e}")
                continue

        logger.info(f"Loaded {len(files)} files from {repo_name}/{path}")
        return files

    # ===== Similar Implementation Search =====

    async def find_similar_implementation(
        self,
        repo_name: str,
        task_type: str = "MODEL_TRAINING",
        keywords: List[str] = None
    ) -> Optional[str]:
        """Find a similar implementation directory for reference.

        Args:
            repo_name: Repository name (ai-craft only)
            task_type: Task type (always MODEL_TRAINING)
            keywords: Optional keywords to match

        Returns:
            Path to similar implementation directory, or None
        """
        repo_info = self.REPOS.get(repo_name, {})
        search_dirs = repo_info.get("model_dirs", [])

        for base_dir in search_dirs:
            try:
                subdirs = await self.list_directory(repo_name, base_dir.rstrip('/'))

                # Filter to directories only
                dirs = [d for d in subdirs if d["type"] == "dir"]

                if not dirs:
                    continue

                # If keywords provided, try to match
                if keywords:
                    for d in dirs:
                        if any(kw.lower() in d["name"].lower() for kw in keywords):
                            return f"{base_dir}{d['name']}"

                # Return first directory as default
                if dirs:
                    return f"{base_dir}{dirs[0]['name']}"

            except Exception as e:
                logger.debug(f"Could not search {base_dir}: {e}")
                continue

        return None

    async def find_implementation_by_name(
        self,
        repo_name: str,
        name: str
    ) -> Optional[str]:
        """Find implementation directory by exact or partial name.

        Args:
            repo_name: Repository name
            name: Implementation name to find (e.g., "xandr_mtl93")

        Returns:
            Full path to implementation directory, or None
        """
        repo_info = self.REPOS.get(repo_name, {})
        search_dirs = repo_info.get("model_dirs", [])

        for base_dir in search_dirs:
            try:
                subdirs = await self.list_directory(repo_name, base_dir.rstrip('/'))

                for d in subdirs:
                    if d["type"] == "dir" and name.lower() in d["name"].lower():
                        return f"{base_dir}{d['name']}"

            except Exception as e:
                logger.debug(f"Could not search {base_dir}: {e}")
                continue

        return None

    # ===== Context Collection for Code Generation =====

    async def get_context_for_generation(
        self,
        repo_name: str,
        task_type: str = "MODEL_TRAINING",
        reference_path: str = None,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Collect context for code generation.

        This is the main method used by code generator to get:
        1. CLAUDE.md conventions
        2. Reference implementation files

        Args:
            repo_name: Target repository (ai-craft only)
            task_type: Task type (always MODEL_TRAINING)
            reference_path: Optional explicit reference path
            keywords: Optional keywords for finding similar implementation

        Returns:
            Context dict with conventions and reference files
        """
        context = {
            "repo_name": repo_name,
            "task_type": "MODEL_TRAINING",
            "conventions": "",
            "reference_path": None,
            "reference_files": {},
            "directory_structure": []
        }

        # Level 1: CLAUDE.md (always load)
        context["conventions"] = await self.get_claude_md(repo_name)

        # Level 2: Find reference implementation
        if reference_path:
            ref_path = reference_path
        else:
            ref_path = await self.find_similar_implementation(
                repo_name, "MODEL_TRAINING", keywords
            )

        if ref_path:
            context["reference_path"] = ref_path

            # Get directory structure
            context["directory_structure"] = await self.list_directory(
                repo_name, ref_path
            )

            # Load reference files
            context["reference_files"] = await self.get_directory_files(
                repo_name, ref_path, max_files=5
            )

        logger.info(
            f"Collected context for {repo_name}/{task_type}: "
            f"conventions={len(context['conventions'])} chars, "
            f"reference_files={len(context['reference_files'])} files"
        )

        return context

    # ===== Repository Selection =====

    def determine_target_repo(self, spec: Dict[str, Any]) -> str:
        """Determine target repository (always ai-craft for model training).

        Args:
            spec: Technical specification dict

        Returns:
            Repository name (always ai-craft)
        """
        # Always return ai-craft for model training
        return "ai-craft"

    def get_implementation_path(
        self,
        repo_name: str,
        task_type: str,
        feature_name: Optional[str] = None
    ) -> str:
        """Get suggested implementation path within repository.

        Args:
            repo_name: Target repository (ai-craft only)
            task_type: Task type (MODEL_TRAINING only)
            feature_name: Optional model name

        Returns:
            Suggested path for new implementation
        """
        base = "src/dable_ai_craft/dsp_models/"

        if feature_name:
            safe_name = feature_name.lower().replace(" ", "_").replace("-", "_")
            return f"{base}{safe_name}/"

        return base
