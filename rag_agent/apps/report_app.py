"""报告生成应用

专注于技术报告生成的 RAG 应用。
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rich.console import Console

from rag_agent.apps.base import AppConfig, BaseApp
from rag_agent.pdf_generator import generate_report_pdf
from rag_agent.rag_engine import RAGEngine

console = Console()


class ReportApp(BaseApp):
    """
    报告生成应用
    """

    def __init__(self, engine: RAGEngine | None = None) -> None:
        """初始化报告应用"""
        super().__init__()
        self.engine = engine if engine is not None else RAGEngine()
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
                - output_format: 输出格式，支持 "markdown" 或 "pdf"
                - output_path: PDF 输出路径（当 output_format="pdf" 时使用）

        Returns:
            Markdown 格式的报告或 PDF 文件路径
        """
        if not self._initialized:
            self.initialize()

        # 报告需要更多上下文
        k = kwargs.get("k", 5)
        verbose = kwargs.get("verbose", False)
        output_format = kwargs.get("output_format", "markdown").lower()
        output_path = kwargs.get("output_path", None)

        # 检索相关文档
        documents = self.engine.retrieve(query, k=k)

        if verbose:
            console.print(f"[dim]检索到 {len(documents)} 个相关文档[/dim]")

        # 生成报告
        report = self.engine.generate_report(query, documents)

        # 处理不同的输出格式
        if output_format == "pdf":
            return self._generate_pdf_report(query, report, documents, output_path)
        else:
            return report

    def _generate_pdf_report(
        self,
        query: str,
        report_content: str,
        documents: list[Document],
        output_path: str | Path | None = None,
    ) -> str:
        """生成 PDF 报告

        Args:
            query: 报告主题
            report_content: Markdown 格式的报告内容
            documents: 参考文档列表
            output_path: 输出路径，如果为 None 则自动生成

        Returns:
            生成的 PDF 文件路径
        """
        console.print("[cyan]正在生成 PDF 报告...[/cyan]")

        # 生成输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = re.sub(r"[^\w\s-]", "", query)[:20].strip()
            safe_title = re.sub(r"[-\s]+", "_", safe_title)
            output_path = Path(f"report_{safe_title}_{timestamp}.pdf")
        else:
            output_path = Path(output_path)

        # 不添加元数据
        metadata = None

        try:
            # 生成 PDF
            pdf_path = generate_report_pdf(
                content=report_content,
                output_path=output_path,
                title=f"技术报告: {query}",
                metadata=metadata,
            )

            console.print(f"[green]✓ PDF 报告已生成: {pdf_path}[/green]")
            return str(pdf_path)

        except Exception as e:
            console.print(f"[red]生成 PDF 失败: {e}[/red]")
            console.print("[yellow]返回 Markdown 格式报告[/yellow]")
            return report_content

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
