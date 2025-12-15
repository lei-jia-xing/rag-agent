"""问答应用

专注于设备知识问答的 RAG 应用。
"""

from typing import Any

from langchain_core.documents import Document
from rich.console import Console

from rag_agent.apps.base import AppConfig, BaseApp
from rag_agent.rag_engine import RAGEngine

console = Console()


class QAApp(BaseApp):
    """
    问答应用
    """

    def __init__(self, engine: RAGEngine | None = None) -> None:
        """初始化问答应用"""
        super().__init__()
        self.engine = engine if engine is not None else RAGEngine()
        self._config = AppConfig(
            name="qa",
            description="设备知识问答 - 基于电气工程知识库的智能问答",
        )

    @property
    def config(self) -> AppConfig:
        """获取应用配置"""
        return self._config

    def initialize(self) -> None:
        """初始化问答引擎"""
        if self._initialized:
            return

        console.print("[cyan]初始化问答应用...[/cyan]")
        self.engine.initialize(load_only=True)
        self._initialized = True
        console.print("[green]问答应用就绪[/green]")

    def run(self, query: str, **kwargs: Any) -> str:
        """执行问答

        Args:
            query: 用户问题
            **kwargs: 额外参数
                - k: 检索文档数量（默认 3）
                - verbose: 是否显示检索结果

        Returns:
            回答内容
        """
        if not self._initialized:
            self.initialize()

        k = kwargs.get("k", 3)
        verbose = kwargs.get("verbose", False)

        # 检索相关文档
        documents = self.engine.retrieve(query, k=k)

        if verbose:
            console.print(f"[dim]检索到 {len(documents)} 个相关文档[/dim]")

        # 生成回答
        answer = self.engine.generate_answer(query, documents)
        return answer

    def get_context(self, query: str, k: int = 3) -> list[Document]:
        """获取相关上下文

        Args:
            query: 查询
            k: 返回文档数

        Returns:
            相关文档列表
        """
        if not self._initialized:
            self.initialize()

        return self.engine.retrieve(query, k=k)
