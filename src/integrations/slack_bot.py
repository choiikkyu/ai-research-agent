"""Slack Bot interface for AI Research Agent."""

import asyncio
import logging
import re
from typing import Any, Dict, Optional

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest

from src.core.config import settings
from src.core.models import ExperimentRequest

logger = logging.getLogger(__name__)


class SlackBot:
    """Slack Bot for AI Research Agent."""

    def __init__(self):
        """Initialize Slack Bot."""
        # Initialize Slack clients
        self.web_client = AsyncWebClient(
            token=settings.slack_bot_token.get_secret_value() if settings.slack_bot_token else ""
        )

        self.socket_client = None
        if settings.slack_app_token:
            self.socket_client = SocketModeClient(
                app_token=settings.slack_app_token.get_secret_value(),
                web_client=self.web_client
            )

        # Command patterns
        self.command_patterns = {
            "help": re.compile(r"^help$", re.IGNORECASE),
            "status": re.compile(r"^status\s+(\S+)", re.IGNORECASE),
            "list": re.compile(r"^list$", re.IGNORECASE),
            "cancel": re.compile(r"^cancel\s+(\S+)", re.IGNORECASE),
            "cleanup": re.compile(r"^cleanup\s+(\S+)", re.IGNORECASE),
            "experiment": re.compile(r"(.*)\s+실험해줘", re.IGNORECASE),
        }

    async def start(self):
        """Start the Slack bot."""
        if not self.socket_client:
            logger.warning("Slack bot not configured, skipping startup")
            return

        logger.info("Starting Slack bot...")

        # Register event handlers
        self.socket_client.socket_mode_request_listeners.append(
            self.handle_socket_mode_request
        )

        # Start socket mode client
        await self.socket_client.connect()
        logger.info("Slack bot connected and listening")

    async def stop(self):
        """Stop the Slack bot."""
        if self.socket_client:
            await self.socket_client.disconnect()
            logger.info("Slack bot disconnected")

    async def handle_socket_mode_request(self, client: SocketModeClient, req: SocketModeRequest):
        """Handle incoming Socket Mode requests."""
        # Acknowledge the request
        response = SocketModeResponse(envelope_id=req.envelope_id)
        await client.send_socket_mode_response(response)

        # Handle different event types
        if req.type == "events_api":
            await self.handle_event(req.payload)
        elif req.type == "interactive":
            await self.handle_interactive(req.payload)
        elif req.type == "slash_commands":
            await self.handle_slash_command(req.payload)

    async def handle_event(self, payload: Dict[str, Any]):
        """Handle Events API events."""
        event = payload.get("event", {})
        event_type = event.get("type")

        if event_type == "app_mention":
            await self.handle_app_mention(event)
        elif event_type == "message":
            # Handle direct messages
            if event.get("channel_type") == "im":
                await self.handle_direct_message(event)

    async def handle_app_mention(self, event: Dict[str, Any]):
        """Handle app mention events."""
        channel = event.get("channel")
        user = event.get("user")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")

        # Remove bot mention from text
        bot_id = await self.get_bot_id()
        text = re.sub(f"<@{bot_id}>", "", text).strip()

        logger.info(f"App mention from {user} in {channel}: {text}")

        # Parse command
        response = await self.process_command(text, user)

        # Send response
        await self.web_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=response.get("text", ""),
            blocks=response.get("blocks"),
            attachments=response.get("attachments")
        )

    async def handle_direct_message(self, event: Dict[str, Any]):
        """Handle direct messages."""
        channel = event.get("channel")
        user = event.get("user")
        text = event.get("text", "")

        # Ignore bot's own messages
        if event.get("bot_id"):
            return

        logger.info(f"Direct message from {user}: {text}")

        # Parse command
        response = await self.process_command(text, user)

        # Send response
        await self.web_client.chat_postMessage(
            channel=channel,
            text=response.get("text", ""),
            blocks=response.get("blocks"),
            attachments=response.get("attachments")
        )

    async def handle_interactive(self, payload: Dict[str, Any]):
        """Handle interactive components (buttons, select menus, etc.)."""
        action_type = payload.get("type")

        if action_type == "block_actions":
            await self.handle_block_actions(payload)

    async def handle_block_actions(self, payload: Dict[str, Any]):
        """Handle block action interactions."""
        user = payload.get("user", {}).get("id")
        channel = payload.get("channel", {}).get("id")
        actions = payload.get("actions", [])

        for action in actions:
            action_id = action.get("action_id")
            value = action.get("value")

            logger.info(f"Block action {action_id} from {user}: {value}")

            # Handle different action types
            if action_id == "retry_experiment":
                await self.retry_experiment(value, channel, user)
            elif action_id == "cleanup_experiment":
                await self.cleanup_experiment(value, channel, user)
            elif action_id == "merge_pr":
                await self.merge_pr(value, channel, user)

    async def process_command(self, text: str, user: str) -> Dict[str, Any]:
        """
        Process a command from a user.

        Args:
            text: Command text
            user: User ID

        Returns:
            Response dictionary with text/blocks
        """
        # Check for help command
        if self.command_patterns["help"].match(text):
            return self.get_help_message()

        # Check for status command
        match = self.command_patterns["status"].match(text)
        if match:
            experiment_id = match.group(1)
            return await self.get_experiment_status(experiment_id)

        # Check for list command
        if self.command_patterns["list"].match(text):
            return await self.list_experiments()

        # Check for cancel command
        match = self.command_patterns["cancel"].match(text)
        if match:
            experiment_id = match.group(1)
            return await self.cancel_experiment(experiment_id)

        # Check for cleanup command
        match = self.command_patterns["cleanup"].match(text)
        if match:
            pr_url = match.group(1)
            return await self.cleanup_pr(pr_url)

        # Check for experiment request
        match = self.command_patterns["experiment"].match(text)
        if match:
            spec_url = match.group(1).strip()
            return await self.start_experiment(spec_url, user)

        # Unknown command
        return {
            "text": "명령어를 이해하지 못했습니다. `help`를 입력하여 사용 가능한 명령어를 확인하세요."
        }

    def get_help_message(self) -> Dict[str, Any]:
        """Get help message."""
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*AI Research Automation Agent - 도움말*"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*기본 사용법:*\n"
                                "`@ai_research_auto_agent [노션 URL] 실험해줘`"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*지원 명령어:*\n"
                                "• `help` - 도움말 표시\n"
                                "• `status <experiment_id>` - 실험 상태 확인\n"
                                "• `list` - 진행 중인 작업 목록\n"
                                "• `cancel <experiment_id>` - 실험 취소\n"
                                "• `cleanup <pr_url>` - PR 및 branch 삭제"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*고급 옵션:*\n"
                                "```\n"
                                "@ai_research_auto_agent\n"
                                "- spec: [노션 URL]\n"
                                "- repo: ai-craft\n"
                                "- gpu: true\n"
                                "- auto_merge: true\n"
                                "```"
                    }
                }
            ]
        }

    async def start_experiment(self, spec_url: str, user: str) -> Dict[str, Any]:
        """Start an experiment from a specification URL."""
        # Extract options from text if present
        # For now, use defaults
        request = ExperimentRequest(
            spec_url=spec_url,
            repo="ai-craft",
            gpu_enabled=True if "gpu" in spec_url.lower() else False,
            auto_merge=False,
            cleanup_on_failure=True
        )

        # TODO: Call MCP server to run workflow
        # For now, return mock response
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"🚀 *실험 시작*\n"
                                f"• 사양: {spec_url}\n"
                                f"• 실험 ID: exp-12345\n"
                                f"• GPU: {'활성화' if request.gpu_enabled else '비활성화'}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"요청자: <@{user}> | 시작 시간: <!date^{int(asyncio.get_event_loop().time())}^{{date_short_pretty}} {{time}}|now>"
                        }
                    ]
                }
            ]
        }

    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment status."""
        # TODO: Implement actual status check
        return {
            "text": f"실험 {experiment_id} 상태: 진행 중 (모의 데이터)"
        }

    async def list_experiments(self) -> Dict[str, Any]:
        """List running experiments."""
        # TODO: Implement actual listing
        return {
            "text": "현재 진행 중인 실험이 없습니다."
        }

    async def cancel_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Cancel an experiment."""
        # TODO: Implement actual cancellation
        return {
            "text": f"실험 {experiment_id}를 취소했습니다."
        }

    async def cleanup_pr(self, pr_url: str) -> Dict[str, Any]:
        """Cleanup a PR and branch."""
        # TODO: Implement actual cleanup
        return {
            "text": f"PR {pr_url}를 정리했습니다."
        }

    async def retry_experiment(self, experiment_id: str, channel: str, user: str):
        """Retry a failed experiment."""
        await self.web_client.chat_postMessage(
            channel=channel,
            text=f"실험 {experiment_id}를 재시도합니다..."
        )

    async def cleanup_experiment(self, experiment_id: str, channel: str, user: str):
        """Cleanup experiment resources."""
        await self.web_client.chat_postMessage(
            channel=channel,
            text=f"실험 {experiment_id} 리소스를 정리합니다..."
        )

    async def merge_pr(self, pr_url: str, channel: str, user: str):
        """Merge a pull request."""
        await self.web_client.chat_postMessage(
            channel=channel,
            text=f"PR {pr_url}를 병합합니다..."
        )

    async def get_bot_id(self) -> str:
        """Get the bot's user ID."""
        try:
            response = await self.web_client.auth_test()
            return response.get("user_id", "")
        except:
            return ""

    def format_experiment_result(
        self,
        experiment_id: str,
        status: str,
        metrics: Dict[str, Any],
        pr_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format experiment result for Slack message.

        Args:
            experiment_id: Experiment ID
            status: Status (SUCCESS/FAILURE)
            metrics: Experiment metrics
            pr_url: Pull request URL if created

        Returns:
            Formatted message blocks
        """
        # Status emoji
        status_emoji = "✅" if status == "SUCCESS" else "❌"

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{status_emoji} *실험 완료: {experiment_id}*"
                }
            }
        ]

        # Add metrics
        if metrics:
            metrics_text = "\n".join([
                f"• {key}: {value:.3f}" if isinstance(value, float) else f"• {key}: {value}"
                for key, value in metrics.items()
            ])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*메트릭:*\n{metrics_text}"
                }
            })

        # Add PR link
        if pr_url:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*PR:* <{pr_url}|View on GitHub>"
                }
            })

        # Add action buttons
        if status == "FAILURE":
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "재시도"
                        },
                        "style": "primary",
                        "action_id": "retry_experiment",
                        "value": experiment_id
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "리소스 정리"
                        },
                        "style": "danger",
                        "action_id": "cleanup_experiment",
                        "value": experiment_id
                    }
                ]
            })
        elif status == "SUCCESS" and pr_url:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "PR 병합"
                        },
                        "style": "primary",
                        "action_id": "merge_pr",
                        "value": pr_url
                    }
                ]
            })

        return {"blocks": blocks}