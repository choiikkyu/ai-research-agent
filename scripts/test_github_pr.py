#!/usr/bin/env python
"""Test GitHub PR creation only (without Claude API).

This script tests:
1. GitHub branch creation
2. File creation on branch
3. Draft PR creation
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
if not os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") == "ghp_your-token-here":
    print("❌ Missing GITHUB_TOKEN in .env file")
    sys.exit(1)

from github import Github
from github.GithubException import GithubException
from src.core.config import settings


# Test configuration
TEST_REPO = "ai-craft"
TEST_BRANCH_PREFIX = "ai-agent-test"


async def test_github_pr():
    """Test GitHub branch and PR creation."""
    print("\n" + "="*60)
    print("🐙 Testing GitHub Draft PR Creation")
    print("="*60)

    # Initialize GitHub client
    g = Github(settings.github_token.get_secret_value())

    # Get repository
    repo_name = f"{settings.github_org}/{TEST_REPO}"
    print(f"📁 Repository: {repo_name}")

    try:
        repo = g.get_repo(repo_name)
        print(f"✅ Repository found: {repo.full_name}")
    except GithubException as e:
        print(f"❌ Failed to access repository: {e}")
        return False

    # Create branch name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"{TEST_BRANCH_PREFIX}-{timestamp}"
    print(f"\n📌 Creating branch: {branch_name}")

    # Get base branch
    try:
        base_branch = repo.get_branch("main")
        base_sha = base_branch.commit.sha
        print(f"✅ Base branch (main) SHA: {base_sha[:8]}...")
    except GithubException as e:
        print(f"❌ Failed to get base branch: {e}")
        return False

    # Create new branch
    try:
        repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=base_sha
        )
        print(f"✅ Branch created: {branch_name}")
    except GithubException as e:
        if e.status == 422:
            print(f"⚠️ Branch already exists: {branch_name}")
        else:
            print(f"❌ Failed to create branch: {e}")
            return False

    # Create test file
    test_file_path = f"experiments/ai-agent-test/{timestamp}/model.py"
    test_file_content = f'''"""
Test model file created by AI Research Agent.
Generated at: {datetime.now().isoformat()}
Branch: {branch_name}
"""

import tensorflow as tf
from tensorflow import keras


class WhiskyCTRModel(keras.Model):
    """Simple pCTR model for Whisky campaign."""

    def __init__(self, embedding_dim: int = 64, hidden_units: list = [128, 64]):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Embedding layers
        self.user_embedding = keras.layers.Embedding(100000, embedding_dim)
        self.ad_embedding = keras.layers.Embedding(10000, embedding_dim)

        # Dense layers
        self.dense_layers = []
        for units in hidden_units:
            self.dense_layers.append(keras.layers.Dense(units, activation='relu'))
            self.dense_layers.append(keras.layers.Dropout(0.2))

        # Output layer
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        user_id, ad_id = inputs['user_id'], inputs['ad_id']

        # Get embeddings
        user_emb = self.user_embedding(user_id)
        ad_emb = self.ad_embedding(ad_id)

        # Concatenate
        x = tf.concat([user_emb, ad_emb], axis=-1)

        # Dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)

        return self.output_layer(x)


if __name__ == "__main__":
    model = WhiskyCTRModel()
    print(f"Model created: {{model}}")
'''

    print(f"\n📝 Creating test file: {test_file_path}")

    try:
        repo.create_file(
            path=test_file_path,
            message=f"[AI Agent Test] Add test model file\n\nBranch: {branch_name}",
            content=test_file_content,
            branch=branch_name
        )
        print(f"✅ File created: {test_file_path}")
    except GithubException as e:
        print(f"❌ Failed to create file: {e}")
        return False

    # Create config file
    config_file_path = f"experiments/ai-agent-test/{timestamp}/config.yaml"
    config_content = f'''# AI Agent Test Configuration
# Generated at: {datetime.now().isoformat()}

experiment:
  name: whisky-ctr-test
  branch: {branch_name}
  type: MODEL_TRAINING

model:
  embedding_dim: 64
  hidden_units: [128, 64]
  dropout: 0.2

training:
  batch_size: 1024
  epochs: 10
  learning_rate: 0.001

evaluation:
  metrics:
    - auc
    - logloss
    - calibration_error
  thresholds:
    auc: 0.85
    logloss: 0.35
'''

    try:
        repo.create_file(
            path=config_file_path,
            message=f"[AI Agent Test] Add config file\n\nBranch: {branch_name}",
            content=config_content,
            branch=branch_name
        )
        print(f"✅ Config file created: {config_file_path}")
    except GithubException as e:
        print(f"⚠️ Failed to create config file: {e}")

    # Create Pull Request
    print(f"\n🔀 Creating Draft PR...")

    pr_title = f"[AI Agent Test] Test PR - {timestamp}"
    pr_body = f'''## 🤖 AI Research Agent Test PR

**This is a test PR created by the AI Research Automation Agent.**

### Details
- **Branch**: `{branch_name}`
- **Created at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Task Type**: MODEL_TRAINING
- **Target**: Whisky Campaign pCTR Model

### Generated Files
- `{test_file_path}` - Test model implementation
- `{config_file_path}` - Configuration file

### Status
⏳ **Draft** - Pending experiment execution

---
*This PR was automatically generated by the AI Research Automation Agent for testing purposes.*
*Please delete this PR after verification.*
'''

    try:
        pr = repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base="main",
            draft=True  # Create as draft PR
        )
        print(f"\n✅ Draft PR created successfully!")
        print(f"🔗 PR URL: {pr.html_url}")
        print(f"📌 PR Number: #{pr.number}")
        print(f"🏷️ Status: Draft")

        # Add labels
        try:
            pr.add_to_labels("ai-generated", "test")
            print(f"🏷️ Labels added: ai-generated, test")
        except:
            print(f"⚠️ Could not add labels (labels might not exist)")

        return {
            "success": True,
            "pr_url": pr.html_url,
            "pr_number": pr.number,
            "branch_name": branch_name
        }

    except GithubException as e:
        print(f"❌ Failed to create PR: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Run the test."""
    print("\n" + "🚀"*30)
    print("AI Research Agent - GitHub PR Test")
    print("🚀"*30)

    print(f"\n📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    result = await test_github_pr()

    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    if result and result.get("success"):
        print("✅ GitHub PR creation: PASSED")
        print(f"\n🔗 PR URL: {result.get('pr_url')}")
        print(f"\n⚠️ Please delete this test PR after verification!")
        return 0
    else:
        print("❌ GitHub PR creation: FAILED")
        if result:
            print(f"Error: {result.get('error')}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)