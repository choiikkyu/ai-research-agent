"""Main entry point for AI Research Agent."""

import asyncio
import logging
import signal
import sys
from typing import Optional

import uvloop

from src.core.config import settings
from src.mcp.server import mcp
from src.integrations.slack_bot import SlackBot
from src.k8s.resource_manager import ResourceManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIResearchAgent:
    """Main application class for AI Research Agent."""

    def __init__(self):
        """Initialize the agent."""
        self.slack_bot: Optional[SlackBot] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.shutdown_event = asyncio.Event()

    async def start(self):
        """Start all services."""
        logger.info("Starting AI Research Agent...")

        # Initialize resource manager
        self.resource_manager = ResourceManager()
        await self.resource_manager.connect()
        logger.info("Resource manager initialized")

        # Initialize and start Slack bot
        if settings.slack_bot_token:
            self.slack_bot = SlackBot()
            asyncio.create_task(self.slack_bot.start())
            logger.info("Slack bot started")
        else:
            logger.warning("Slack bot token not configured, running without Slack integration")

        # Start MCP server
        logger.info("Starting MCP server...")
        asyncio.create_task(self.run_mcp_server())

        # Start periodic cleanup task
        asyncio.create_task(self.periodic_cleanup())

        logger.info("AI Research Agent started successfully")

    async def run_mcp_server(self):
        """Run the MCP server."""
        try:
            # MCP server will handle its own async loop
            await asyncio.to_thread(mcp.run)
        except Exception as e:
            logger.error(f"MCP server error: {str(e)}")
            self.shutdown_event.set()

    async def periodic_cleanup(self):
        """Periodically cleanup stale resources."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for 1 hour
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=3600
                )
            except asyncio.TimeoutError:
                # Perform cleanup
                if self.resource_manager:
                    logger.info("Running periodic resource cleanup...")
                    try:
                        result = await self.resource_manager.cleanup_stale_resources(
                            max_age_hours=24
                        )
                        logger.info(f"Cleanup result: {result}")
                    except Exception as e:
                        logger.error(f"Cleanup failed: {str(e)}")

    async def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("Shutting down AI Research Agent...")

        # Set shutdown event
        self.shutdown_event.set()

        # Stop Slack bot
        if self.slack_bot:
            await self.slack_bot.stop()
            logger.info("Slack bot stopped")

        # Disconnect resource manager
        if self.resource_manager:
            await self.resource_manager.disconnect()
            logger.info("Resource manager disconnected")

        logger.info("AI Research Agent shutdown complete")

    async def run(self):
        """Run the agent."""
        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler(sig):
            logger.info(f"Received signal {sig}")
            asyncio.create_task(self.shutdown())

        for sig in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

        # Start services
        await self.start()

        # Wait for shutdown
        await self.shutdown_event.wait()


async def health_check_server():
    """Simple health check HTTP server."""
    from aiohttp import web

    async def health(request):
        return web.json_response({"status": "healthy"})

    async def ready(request):
        # TODO: Check if all services are ready
        return web.json_response({"status": "ready"})

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8000)
    await site.start()
    logger.info("Health check server started on port 8000")


async def main():
    """Main entry point."""
    # Use uvloop for better async performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Start health check server
    asyncio.create_task(health_check_server())

    # Create and run agent
    agent = AIResearchAgent()
    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())