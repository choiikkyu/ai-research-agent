"""GitHub integration for PR management.

Includes the 2-Commit Strategy for model modifications.
See .claude/COMMIT_STRATEGY.md for detailed documentation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from github import Github
from github.GithubException import GithubException

from src.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 2-COMMIT STRATEGY FOR MODEL MODIFICATIONS
# =============================================================================
# When creating a new model based on an existing one, use 2 commits:
#   Commit 1: Copy all files with name changes only (no logic changes)
#   Commit 2: Apply actual modifications (layer structure, hyperparameters, etc.)
#
# This makes PR diffs clear - reviewers can see exactly what changed in Commit 2.
# See .claude/COMMIT_STRATEGY.md for full documentation.
# =============================================================================


async def create_model_pr_with_2commit_strategy(
    repo_name: str,
    reference_path: str,
    destination_path: str,
    reference_name: str,
    new_name: str,
    modifications: Dict[str, Dict[str, Any]],
    pr_title: str,
    pr_description: str = "",
    branch_name: Optional[str] = None,
    draft: bool = True
) -> Dict[str, Any]:
    """
    Create a PR using the 2-commit strategy for model modifications.

    This strategy ensures clear PR diffs by:
    1. First copying all files with name changes only
    2. Then applying actual logic modifications in a separate commit

    Args:
        repo_name: Repository name (e.g., "ai-craft")
        reference_path: Path to reference model (e.g., "src/.../xandr_mtl93")
        destination_path: Path for new model (e.g., "src/.../xandr_mtltest")
        reference_name: Reference model name for replacement (e.g., "xandr_mtl93")
        new_name: New model name (e.g., "xandr_mtltest")
        modifications: Dict of {filename: {old_content: new_content}} for Commit 2
        pr_title: PR title
        pr_description: PR description (optional)
        branch_name: Branch name (auto-generated if not provided)
        draft: Create as draft PR (default: True)

    Returns:
        Dict with pr_number, pr_url, branch_name, commits info
    """
    logger.info(f"Creating PR with 2-commit strategy: {reference_name} -> {new_name}")

    g = Github(settings.github_token.get_secret_value())
    full_repo_name = f"{settings.github_org}/{repo_name}"
    repo = g.get_repo(full_repo_name)

    # Generate branch name if not provided
    if not branch_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"ai-agent-{new_name}-{timestamp}"

    # Create branch from main
    base_branch = repo.get_branch("main")
    try:
        repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=base_branch.commit.sha
        )
        logger.info(f"Created branch: {branch_name}")
    except GithubException as e:
        if e.status != 422:  # Not "already exists"
            raise

    # Get reference files
    reference_files = _get_directory_contents(repo, reference_path)

    # =========================================================================
    # COMMIT 1: Copy with name changes only
    # =========================================================================
    commit1_message = f"[AI Agent] Copy {reference_name} -> {new_name} (name changes only)"
    commit1_files = {}

    for file_path, content in reference_files.items():
        # Replace reference name with new name (case-sensitive variants)
        new_content = _replace_model_name(content, reference_name, new_name)

        # Calculate new file path
        new_file_path = file_path.replace(reference_path, destination_path)
        commit1_files[new_file_path] = new_content

    # Create files for Commit 1
    for file_path, content in commit1_files.items():
        repo.create_file(
            path=file_path,
            message=commit1_message,
            content=content,
            branch=branch_name
        )
    logger.info(f"Commit 1: Created {len(commit1_files)} files with name changes")

    # =========================================================================
    # COMMIT 2: Apply actual modifications
    # =========================================================================
    commit2_message = _generate_commit2_message(new_name, modifications)
    commit2_count = 0

    for filename, changes in modifications.items():
        file_path = f"{destination_path}/{filename}"
        try:
            file_obj = repo.get_contents(file_path, ref=branch_name)
            current_content = file_obj.decoded_content.decode('utf-8')

            # Apply modifications
            new_content = current_content
            for old_str, new_str in changes.items():
                new_content = new_content.replace(old_str, new_str)

            if new_content != current_content:
                repo.update_file(
                    path=file_path,
                    message=commit2_message,
                    content=new_content,
                    sha=file_obj.sha,
                    branch=branch_name
                )
                commit2_count += 1
                logger.info(f"Commit 2: Modified {filename}")

        except Exception as e:
            logger.error(f"Failed to modify {filename}: {e}")

    logger.info(f"Commit 2: Modified {commit2_count} files")

    # =========================================================================
    # CREATE PR
    # =========================================================================
    pr_body = _generate_2commit_pr_description(
        reference_name, new_name, commit1_files, modifications, pr_description
    )

    pr = repo.create_pull(
        title=pr_title,
        body=pr_body,
        head=branch_name,
        base="main",
        draft=draft
    )

    logger.info(f"Created PR #{pr.number}: {pr.html_url}")

    return {
        "pr_number": pr.number,
        "pr_url": pr.html_url,
        "branch_name": branch_name,
        "commit1": {
            "message": commit1_message,
            "files_count": len(commit1_files)
        },
        "commit2": {
            "message": commit2_message,
            "files_modified": commit2_count
        },
        "status": "created"
    }


def _get_directory_contents(repo, path: str) -> Dict[str, str]:
    """Get all file contents from a directory."""
    files = {}
    try:
        contents = repo.get_contents(path)
        if not isinstance(contents, list):
            contents = [contents]

        for item in contents:
            if item.type == "file":
                files[item.path] = item.decoded_content.decode('utf-8')
            elif item.type == "dir":
                files.update(_get_directory_contents(repo, item.path))
    except Exception as e:
        logger.error(f"Failed to get contents of {path}: {e}")

    return files


def _replace_model_name(content: str, old_name: str, new_name: str) -> str:
    """Replace model name with various case conventions."""
    result = content

    # Exact match (e.g., xandr_mtl93 -> xandr_mtltest)
    result = result.replace(old_name, new_name)

    # PascalCase (e.g., XandrMtl93 -> XandrMtltest)
    old_pascal = _to_pascal_case(old_name)
    new_pascal = _to_pascal_case(new_name)
    result = result.replace(old_pascal, new_pascal)

    # UPPER_CASE (e.g., XANDR_MTL93 -> XANDR_MTLTEST)
    result = result.replace(old_name.upper(), new_name.upper())

    return result


def _to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def _generate_commit2_message(model_name: str, modifications: Dict) -> str:
    """Generate commit message for Commit 2."""
    changes = []
    for filename, mods in modifications.items():
        for old_val, new_val in mods.items():
            # Truncate long values for commit message
            old_short = old_val[:30] + "..." if len(old_val) > 30 else old_val
            new_short = new_val[:30] + "..." if len(new_val) > 30 else new_val
            changes.append(f"  - {filename}: {old_short} -> {new_short}")

    return f"[AI Agent] Apply modifications to {model_name}\n\n" + "\n".join(changes[:5])


def _generate_2commit_pr_description(
    reference_name: str,
    new_name: str,
    commit1_files: Dict[str, str],
    modifications: Dict[str, Dict[str, Any]],
    additional_description: str = ""
) -> str:
    """Generate PR description for 2-commit strategy PR."""
    description = f"""## 2-Commit Strategy PR

This PR uses the **2-commit strategy** for clear diff review.

### Reference Model
`{reference_name}`

### New Model
`{new_name}`

---

## Commit 1: Copy with Name Changes

Copied {len(commit1_files)} files from reference model with name replacements only.

**Files:**
"""
    for file_path in sorted(commit1_files.keys()):
        filename = file_path.split('/')[-1]
        description += f"- `{filename}`\n"

    description += f"""
---

## Commit 2: Actual Modifications

**To see the actual changes, review Commit 2 diff only.**

**Modified Files:**
"""
    for filename, changes in modifications.items():
        description += f"\n### `{filename}`\n"
        for old_val, new_val in changes.items():
            description += f"```diff\n- {old_val}\n+ {new_val}\n```\n"

    if additional_description:
        description += f"\n---\n\n## Additional Notes\n\n{additional_description}\n"

    description += """
---

*This PR was automatically generated by the AI Research Automation Agent using the 2-commit strategy.*
*See `.claude/COMMIT_STRATEGY.md` for documentation on this approach.*
"""
    return description


async def manage_pr(
    implementation: Dict[str, Any],
    evaluation: Dict[str, Any],
    auto_merge: bool = False
) -> Dict[str, Any]:
    """
    Manage GitHub pull request lifecycle.

    Args:
        implementation: Code implementation details
        evaluation: Evaluation results
        auto_merge: Whether to auto-merge if successful

    Returns:
        PR management result
    """
    logger.info(f"Managing PR for branch: {implementation.get('branch_name')}")

    # Initialize GitHub client
    g = Github(settings.github_token.get_secret_value())

    try:
        # Get repository
        repo_name = f"{settings.github_org}/{implementation.get('repository', settings.github_repo_ai_craft)}"
        repo = g.get_repo(repo_name)

        # Create branch
        branch_name = implementation.get("branch_name")
        base_branch = repo.get_branch("main")

        # Create new branch from main
        try:
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_branch.commit.sha
            )
            logger.info(f"Created branch: {branch_name}")
        except GithubException as e:
            if e.status == 422:  # Branch already exists
                logger.info(f"Branch already exists: {branch_name}")
            else:
                raise

        # Create or update files
        for filename, content in implementation.get("files", {}).items():
            try:
                # Check if file exists
                try:
                    file = repo.get_contents(filename, ref=branch_name)
                    # Update existing file
                    repo.update_file(
                        path=filename,
                        message=f"Update {filename} via AI Research Agent",
                        content=content,
                        sha=file.sha,
                        branch=branch_name
                    )
                    logger.info(f"Updated file: {filename}")
                except:
                    # Create new file
                    repo.create_file(
                        path=filename,
                        message=f"Create {filename} via AI Research Agent",
                        content=content,
                        branch=branch_name
                    )
                    logger.info(f"Created file: {filename}")

            except Exception as e:
                logger.error(f"Failed to create/update file {filename}: {str(e)}")

        # Create pull request
        pr_title = f"[AI Agent] {implementation.get('spec_title', 'Automated Implementation')}"
        pr_body = generate_pr_description(implementation, evaluation)

        try:
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base="main"
            )
            logger.info(f"Created PR #{pr.number}: {pr.html_url}")

            # Add labels
            labels = determine_pr_labels(implementation, evaluation)
            pr.add_to_labels(*labels)

            # Handle auto-merge
            if auto_merge and evaluation.get("passed", False):
                logger.info(f"Auto-merging PR #{pr.number}")
                pr.merge(
                    commit_title=pr_title,
                    commit_message=f"Auto-merged by AI Research Agent\n\nScore: {evaluation.get('score', 0)}/100",
                    merge_method="squash"
                )
                logger.info(f"Successfully merged PR #{pr.number}")

                # Delete branch after merge
                repo.get_git_ref(f"heads/{branch_name}").delete()
                logger.info(f"Deleted branch: {branch_name}")

            result = {
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "branch_name": branch_name,
                "status": "merged" if (auto_merge and evaluation.get("passed")) else "open",
                "labels": labels
            }

        except GithubException as e:
            if e.status == 422:  # PR already exists
                # Find existing PR
                prs = repo.get_pulls(head=f"{settings.github_org}:{branch_name}")
                if prs.totalCount > 0:
                    pr = prs[0]
                    logger.info(f"Found existing PR #{pr.number}")

                    # Update PR description
                    pr.edit(body=pr_body)

                    result = {
                        "pr_number": pr.number,
                        "pr_url": pr.html_url,
                        "branch_name": branch_name,
                        "status": "updated",
                        "labels": []
                    }
                else:
                    raise
            else:
                raise

        return result

    except Exception as e:
        logger.error(f"PR management failed: {str(e)}")
        return {
            "error": str(e),
            "branch_name": implementation.get("branch_name"),
            "status": "failed"
        }


async def cleanup_resources(
    experiment_id: str,
    cleanup_pr: bool = False
) -> Dict[str, Any]:
    """
    Cleanup experiment resources including GitHub branches and PRs.

    Args:
        experiment_id: Experiment identifier
        cleanup_pr: Whether to cleanup PR and branch

    Returns:
        Cleanup result
    """
    logger.info(f"Cleaning up resources for experiment: {experiment_id}")

    cleanup_result = {
        "experiment_id": experiment_id,
        "cleaned_resources": []
    }

    if cleanup_pr:
        # TODO: Implement PR/branch cleanup logic
        # This would need to track the branch name associated with the experiment
        logger.info("PR cleanup requested but not yet implemented")

    # Cleanup Kubernetes resources
    from src.k8s.pod_launcher import K8sPodLauncher
    launcher = K8sPodLauncher()

    pod_name = f"ai-tf-box-exp-{experiment_id}"
    try:
        await launcher.cleanup_pod(pod_name)
        cleanup_result["cleaned_resources"].append(f"Pod: {pod_name}")
        logger.info(f"Cleaned up pod: {pod_name}")
    except Exception as e:
        logger.error(f"Failed to cleanup pod {pod_name}: {str(e)}")

    return cleanup_result


def generate_pr_description(
    implementation: Dict[str, Any],
    evaluation: Dict[str, Any]
) -> str:
    """
    Generate PR description with experiment results.

    Args:
        implementation: Implementation details
        evaluation: Evaluation results

    Returns:
        Formatted PR description
    """
    passed = evaluation.get("passed", False)
    score = evaluation.get("score", 0)
    task_type = evaluation.get("task_type", "Unknown")

    # Status emoji
    status_emoji = "✅" if passed else "❌"

    description = f"""
## {status_emoji} AI Research Agent Implementation

**Task Type**: {task_type}
**Specification**: {implementation.get('spec_title', 'N/A')}
**Experiment ID**: {evaluation.get('experiment_id', 'N/A')}
**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Evaluation Results

**Status**: {'PASSED' if passed else 'FAILED'}
**Score**: {score}/100
**Reason**: {evaluation.get('reason', 'N/A')}

### Detailed Metrics
"""

    # Add detailed metrics
    details = evaluation.get("details", {})
    for metric_name, metric_detail in details.items():
        if isinstance(metric_detail, dict) and "message" in metric_detail:
            description += f"- {metric_detail['message']}\n"

    # Add metrics table
    metrics = evaluation.get("metrics", {})
    if metrics:
        description += "\n### Raw Metrics\n\n| Metric | Value |\n|--------|-------|\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                description += f"| {key} | {value:.4f} |\n"
            else:
                description += f"| {key} | {value} |\n"

    # Add recommendations
    recommendations = evaluation.get("recommendations", [])
    if recommendations:
        description += "\n## Recommendations\n\n"
        for rec in recommendations:
            description += f"- {rec}\n"

    # Add file list
    files = implementation.get("files", {})
    if files:
        description += f"\n## Generated Files ({len(files)})\n\n"
        for filename in files.keys():
            description += f"- `{filename}`\n"

    description += """
---

*This PR was automatically generated by the AI Research Automation Agent.*
*Please review the implementation and experiment results before merging.*
"""

    return description


def determine_pr_labels(
    implementation: Dict[str, Any],
    evaluation: Dict[str, Any]
) -> list[str]:
    """
    Determine appropriate labels for the PR.

    Args:
        implementation: Implementation details
        evaluation: Evaluation results

    Returns:
        List of label names
    """
    labels = ["ai-generated"]

    # Task type label
    task_type = evaluation.get("task_type", "").lower()
    if "model" in task_type:
        labels.append("model-training")
    elif "feature" in task_type:
        labels.append("feature-engineering")

    # Status label
    if evaluation.get("passed", False):
        labels.append("passed-evaluation")
        score = evaluation.get("score", 0)
        if score >= 90:
            labels.append("high-quality")
    else:
        labels.append("needs-improvement")

    # Repository-specific labels
    repo = implementation.get("repository", "")
    if "craft" in repo:
        labels.append("ai-craft")
    elif "feature" in repo:
        labels.append("ai-feature-store")

    return labels