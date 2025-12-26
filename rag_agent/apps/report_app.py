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
                - output_format: 输出格式，支持 "markdown", "pdf", "latex" 或 "diagnosis"
                - output_path: PDF 输出路径（当 output_format="pdf" 时使用）

        Returns:
            Markdown 格式的报告或 PDF 文件路径
        """
        import time

        if not self._initialized:
            self.initialize()

        # 报告需要更多上下文
        k = kwargs.get("k", 5)
        verbose = kwargs.get("verbose", False)
        output_format = kwargs.get("output_format", "markdown").lower()
        output_path = kwargs.get("output_path", None)

        # Stage 1: 检索相关文档
        console.print("[cyan][1/4] 检索相关文档...[/cyan]")
        start_time = time.time()
        documents = self.engine.retrieve(query, k=k)
        elapsed = time.time() - start_time
        console.print(f"[green]  ✓ 检索完成，找到 {len(documents)} 个相关文档 ({elapsed:.1f}s)[/green]")

        if verbose:
            console.print(f"[dim]检索到 {len(documents)} 个相关文档[/dim]")

        # 生成报告
        report = self.engine.generate_report(query, documents)

        # 处理不同的输出格式
        if output_format == "pdf":
            return self._generate_pdf_report(query, report, documents, output_path)
        elif output_format == "latex":
            return self._generate_latex_report(query, documents, output_path)
        elif output_format == "diagnosis":
            return self._generate_diagnosis_report(query, documents, output_path)
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

    def _generate_latex_report(
        self,
        query: str,
        documents: list[Document],
        output_path: str | Path | None = None,
    ) -> str:
        """生成 LaTeX 报告并使用 MCP 服务编译

        Args:
            query: 报告主题
            documents: 参考文档列表
            output_path: 输出路径，如果为 None 则自动生成

        Returns:
            生成的 PDF 文件路径或 LaTeX 内容
        """
        import time

        # 使用旧的 LaTeX 生成方式（保留向后兼容）
        console.print("[cyan]正在生成 LaTeX 报告...[/cyan]")

        start_time = time.time()
        try:
            # 调用 LLM 生成 LaTeX
            full_latex = self.engine.generate_latex_content(query, documents)
            elapsed = time.time() - start_time
            console.print(f"[green]  ✓ LaTeX 内容生成完成 ({elapsed:.1f}s)[/green]")

            # 生成输出路径
            if output_path is None:
                reports_dir = Path("reports")
                reports_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_title = re.sub(r"[^\w\s-]", "", query)[:20].strip()
                safe_title = re.sub(r"[-\s]+", "_", safe_title)
                output_path = reports_dir / f"latex_{safe_title}_{timestamp}.pdf"
            else:
                output_path = Path(output_path)

            # 编译 LaTeX
            console.print("[cyan]正在编译 LaTeX 文档...[/cyan]")
            start_time = time.time()
            from rag_agent.mcp.latex_client import compile_latex
            result = compile_latex(content=full_latex, format="pdf", template="custom")
            elapsed = time.time() - start_time

            if result.get("success"):
                console.print(f"[green]  ✓ LaTeX编译成功 ({elapsed:.1f}s)[/green]")
                console.print("[cyan]正在保存 PDF 文件...[/cyan]")
                start_time = time.time()
                import shutil

                source_path = Path(result["output_path"])
                shutil.copy(source_path, output_path)
                elapsed = time.time() - start_time
                console.print(f"[green]  ✓ PDF文件已保存 ({elapsed:.2f}s)[/green]")
                return str(output_path)
            else:
                error_msg = result.get("error", "未知错误")
                console.print(f"[red]LaTeX 编译失败: {error_msg}[/red]")
                console.print("[yellow]返回 LaTeX 格式报告[/yellow]")
                return full_latex

        except Exception as e:
            console.print(f"[red]生成 LaTeX 报告失败: {e}[/red]")
            return f"生成 LaTeX 报告失败: {e}"

    def _generate_diagnosis_report(
        self,
        device_name: str,
        documents: list[Document],
        output_path: str | Path | None = None,
    ) -> str:
        """生成设备健康诊断报告（使用 LaTeX MCP 内置模板）

        Args:
            device_name: 设备名称
            documents: 参考文档列表
            output_path: 输出路径，如果为 None 则自动生成

        Returns:
            生成的 PDF 文件路径或 LaTeX 内容（如果编译失败）
        """
        import re
        import time
        from datetime import datetime

        try:
            # Stage 2: 生成诊断字段数据
            console.print("\n[cyan][2/3] 生成诊断字段数据...[/cyan]")
            start_time = time.time()
            diagnosis_data = self.engine.generate_diagnosis_fields(device_name, documents)
            elapsed = time.time() - start_time
            console.print(f"[green]  ✓ 字段数据生成完成 ({elapsed:.1f}s)[/green]")

            # Stage 3: 使用 LaTeX MCP 生成报告
            console.print("\n[cyan][3/3] 使用 LaTeX MCP 生成报告...[/cyan]")
            start_time = time.time()

            from rag_agent.mcp.latex_client import generate_diagnosis_report

            result = generate_diagnosis_report(
                data=diagnosis_data,
                template_id="device_diagnosis",
            )
            elapsed = time.time() - start_time

            if not result.get("success"):
                error_msg = result.get("error", "未知错误")
                console.print(f"[red]报告生成失败: {error_msg}[/red]")
                return f"生成失败: {error_msg}"

            console.print(f"[green]  ✓ 报告生成成功 ({elapsed:.1f}s)[/green]")

            # 复制 PDF 到指定路径
            if output_path is None:
                reports_dir = Path("reports")
                reports_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r"[^\w\s-]", "", device_name)[:20].strip()
                safe_name = re.sub(r"[-\s]+", "_", safe_name)
                output_path = reports_dir / f"diagnosis_{safe_name}_{timestamp}.pdf"
            else:
                output_path = Path(output_path)

            import shutil
            source = Path(result["output_path"]) if result.get("output_path") else None
            if source and source.exists():
                shutil.copy(source, output_path)
                console.print(f"[green]  ✓ PDF 文件已保存: {output_path}[/green]")
                return str(output_path)
            else:
                console.print("[yellow]PDF 文件不存在[/yellow]")
                return "生成失败"

        except Exception as e:
            console.print(f"[red]生成诊断报告失败: {e}[/red]")
            return f"生成失败: {e}"



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
