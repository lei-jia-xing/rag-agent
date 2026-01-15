#!/usr/bin/env python3
"""
Wrapper script to keep the container running for MCP LaTeX server.
The actual MCP server uses stdio, not HTTP.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import signal
import sys

# 配置日志 - 只使用文件日志，不输出到控制台
logs_dir = Path('/workspace/logs')
logs_dir.mkdir(exist_ok=True)

log_file = logs_dir / f"run_server_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')
    ]
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
