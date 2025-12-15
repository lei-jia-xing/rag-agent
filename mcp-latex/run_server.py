#!/usr/bin/env python3
"""
Wrapper script to keep the container running for MCP LaTeX server.
The actual MCP server uses stdio, not HTTP.
"""

import asyncio
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    logger.info("Received shutdown signal")
    sys.exit(0)

async def main():
    """Keep the container running for stdio connections."""
    logger.info("MCP LaTeX Server container is running")
    logger.info("Connect your MCP client to this container's stdio")
    logger.info("Container will stay alive until stopped")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep the container running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())