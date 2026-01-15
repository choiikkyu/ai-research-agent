"""Claude API client wrapper."""

import logging
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import Message

from src.core.config import settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Wrapper for Claude API interactions."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude client.

        Args:
            api_key: Optional API key override
        """
        self.api_key = api_key or settings.anthropic_api_key.get_secret_value()
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate_code(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.2,
        model: str = "claude-3-opus-20240229"
    ) -> str:
        """
        Generate code using Claude.

        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            model: Model to use

        Returns:
            Generated text
        """
        logger.info(f"Generating code with Claude ({model})")

        try:
            messages = [{"role": "user", "content": prompt}]

            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or self._get_default_system_prompt(),
                messages=messages
            )

            if response.content:
                return response.content[0].text
            else:
                return ""

        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            raise

    async def analyze_code(
        self,
        code: str,
        analysis_type: str = "review"
    ) -> Dict[str, Any]:
        """
        Analyze code for quality, bugs, or improvements.

        Args:
            code: Code to analyze
            analysis_type: Type of analysis (review, bugs, optimize)

        Returns:
            Analysis results
        """
        prompts = {
            "review": "Review this code for quality, best practices, and potential improvements:",
            "bugs": "Identify any bugs or potential issues in this code:",
            "optimize": "Suggest optimizations for this code:"
        }

        prompt = f"{prompts.get(analysis_type, prompts['review'])}\n\n```python\n{code}\n```"

        response = await self.generate_code(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2000
        )

        # Parse response into structured format
        return self._parse_analysis_response(response, analysis_type)

    async def explain_code(
        self,
        code: str,
        detail_level: str = "medium"
    ) -> str:
        """
        Explain what code does.

        Args:
            code: Code to explain
            detail_level: Level of detail (brief, medium, detailed)

        Returns:
            Explanation text
        """
        detail_prompts = {
            "brief": "Briefly explain what this code does in 2-3 sentences:",
            "medium": "Explain what this code does, including key components:",
            "detailed": "Provide a detailed explanation of this code, including all functions and logic:"
        }

        prompt = f"{detail_prompts.get(detail_level, detail_prompts['medium'])}\n\n```python\n{code}\n```"

        return await self.generate_code(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500
        )

    async def suggest_improvements(
        self,
        spec: str,
        current_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Suggest improvements based on specification and metrics.

        Args:
            spec: Original specification
            current_metrics: Current performance metrics

        Returns:
            List of improvement suggestions
        """
        metrics_text = "\n".join([
            f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}"
            for key, value in current_metrics.items()
        ])

        prompt = f"""
Based on this specification and current metrics, suggest concrete improvements:

Specification:
{spec}

Current Metrics:
{metrics_text}

Provide 3-5 specific, actionable improvements that could help achieve better results.
Format as a numbered list.
"""

        response = await self.generate_code(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1000
        )

        # Extract suggestions from response
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering or bullets
                suggestion = re.sub(r'^[\d\.\-\*]+\s*', '', line)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions[:5]  # Return top 5 suggestions

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for code generation."""
        return """You are an expert AI engineer specializing in machine learning models and feature engineering.
You have deep knowledge of:
- pCTR/pCVR models and recommendation systems
- TensorFlow, PyTorch, and scikit-learn
- Feature engineering and data preprocessing
- Production ML systems and best practices
- Python, SQL, and data engineering

Generate clean, production-ready code following these principles:
1. Write modular, reusable code
2. Include proper error handling
3. Add type hints and docstrings
4. Follow PEP 8 style guidelines
5. Optimize for performance and scalability
6. Include necessary imports and dependencies

When generating code for Dable's ai-craft repository:
- Follow existing code patterns and conventions
- Use MLflow for experiment tracking
- Include configuration files (YAML/JSON)
- Add unit tests where appropriate"""

    def _parse_analysis_response(
        self,
        response: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Parse analysis response into structured format.

        Args:
            response: Raw response text
            analysis_type: Type of analysis performed

        Returns:
            Structured analysis results
        """
        import re

        result = {
            "type": analysis_type,
            "summary": "",
            "findings": [],
            "suggestions": []
        }

        # Extract summary (first paragraph)
        paragraphs = response.split('\n\n')
        if paragraphs:
            result["summary"] = paragraphs[0].strip()

        # Extract findings and suggestions
        current_section = None
        for line in response.split('\n'):
            line = line.strip()

            # Detect section headers
            if any(keyword in line.lower() for keyword in ["issue", "problem", "bug", "finding"]):
                current_section = "findings"
            elif any(keyword in line.lower() for keyword in ["suggest", "improve", "recommend"]):
                current_section = "suggestions"

            # Extract items
            elif line and (line[0].isdigit() or line.startswith('-')):
                item = re.sub(r'^[\d\.\-\*]+\s*', '', line)
                if item:
                    if current_section == "findings":
                        result["findings"].append(item)
                    elif current_section == "suggestions":
                        result["suggestions"].append(item)

        return result