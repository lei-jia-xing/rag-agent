"""报告生成应用

专注于技术报告生成的 RAG 应用。
"""

from typing import Any

from langchain_core.documents import Document
from rich.console import Console

from rag_agent.apps.base import AppConfig, BaseApp
from rag_agent.rag_engine import RAGEngine

console = Console()


class ReportApp(BaseApp):
    """
    报告生成应用
    """

    def __init__(self) -> None:
        """初始化报告应用"""
        super().__init__()
        self.engine = RAGEngine()
        self._config = AppConfig(
            name="report",
            description="报告生成 - 自动生成结构化技术报告",
        )

    @property
    def config(self) -> AppConfig:
        """获取应用配置"""
        return self._config

    def initialize(self) -> None:
        """初始化报告引擎"""
        if self._initialized:
            return

        console.print("[cyan]📝 初始化报告应用...[/cyan]")
        self.engine.initialize(load_only=True)
        self._initialized = True
        console.print("[green]✓ 报告应用就绪[/green]")

    def run(self, query: str, **kwargs: Any) -> str:
        """生成报告

        Args:
            query: 报告主题
            **kwargs: 额外参数
                - k: 检索文档数量（默认 5，报告需要更多上下文）
                - verbose: 是否显示检索结果

        Returns:
            Markdown 格式的报告
        """
        if not self._initialized:
            self.initialize()

        # 报告需要更多上下文
        k = kwargs.get("k", 5)
        verbose = kwargs.get("verbose", False)

        # 检索相关文档
        documents = self.engine.retrieve(query, k=k)

        if verbose:
            console.print(f"[dim]检索到 {len(documents)} 个相关文档[/dim]")

        # 生成报告
        report = self.engine.generate_report(query, documents)
        return report

    def get_context(self, query: str, k: int = 5) -> list[Document]:
        """获取相关上下文

        Args:
            query: 查询
            k: 返回文档数（报告默认更多）

        Returns:
            相关文档列表
        """
        if not self._initialized:
            self.initialize()

        return self.engine.retrieve(query, k=k)
