"""RAG Agent - 设备问答智能体"""

__version__ = "0.1.0"

from rag_agent.cli import app


def main() -> None:
    """主入口函数 - 使用 Typer CLI"""
    app()
