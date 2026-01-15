#!/usr/bin/env python3
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

logs_dir = Path("/workspace/logs")
logs_dir.mkdir(exist_ok=True)

log_file = logs_dir / f"run_server_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8")],
)
logger = logging.getLogger(__name__)

_shutdown_requested = False


def signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True


async def main():
    logger.info("MCP LaTeX Server container is running")
    logger.info("Connect your MCP client to this container's stdio")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not _shutdown_requested:
            await asyncio.sleep(1)
        logger.info("Received shutdown signal, exiting...")
    except KeyboardInterrupt:
        pass

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
