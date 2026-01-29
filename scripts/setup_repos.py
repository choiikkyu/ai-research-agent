#!/usr/bin/env python
"""Setup script to initialize or update target repositories."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.repo_manager import RepositoryManager


async def main():
    """Setup repositories."""
    print("🚀 Setting up target repositories...")

    manager = RepositoryManager()

    try:
        # Setup repos (clone or update)
        await manager.setup_repos()
        print("✅ Repositories setup complete!")

        # Analyze repositories
        print("\n📊 Analyzing repositories...")

        for repo_name in ["ai-craft"]:
            print(f"\n--- {repo_name} ---")
            analysis = await manager.analyze_codebase(repo_name)

            print(f"Structure:")
            for key, value in analysis["structure"].items():
                if isinstance(value, dict):
                    print(f"  {key}: {value.get('file_count', 0)} files")

            if "config_files" in analysis["conventions"]:
                print(f"Config files: {', '.join(analysis['conventions']['config_files'])}")

            deps = analysis["dependencies"].get("python", [])
            if deps:
                print(f"Dependencies: {len(deps)} packages")

        print("\n✅ Repository analysis complete!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())