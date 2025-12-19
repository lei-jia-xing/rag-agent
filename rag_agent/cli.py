"""RAG Agent CLI

使用 Typer 构建的专业命令行界面。
"""

from pathlib import Path
from typing import Annotated

import pyfiglet
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from rag_agent.apps import QAApp, ReportApp
from rag_agent.apps.base import BaseApp
from rag_agent.config import config
from rag_agent.data_loader import DatasetLoader
from rag_agent.pdf_generator import generate_report_pdf
from rag_agent.rag_engine import RAGEngine

# Typer 应用
app = typer.Typer(
    name="rag-agent",
    help="RAG Agent - 智能问答与报告生成系统",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()

# 交互模式命令定义
SLASH_COMMANDS: dict[str, tuple[str, str]] = {
    "/qa": ("切换问答模式", "switch"),
    "/report": ("切换报告模式", "switch"),
    "/pdf": ("生成PDF报告", "action"),
    "/clear": ("清屏", "action"),
    "/help": ("显示帮助", "action"),
    "/exit": ("退出", "action"),
}


class SlashCommandCompleter(Completer):
    """斜杠命令补全器

    类似 Claude Code 的 / 命令补全。
    只在输入 / 开头时触发，显示可用命令和描述。
    """

    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor

        # 只在输入 / 开头时触发补全
        if not text.startswith("/"):
            return

        # 匹配命令
        for cmd, (desc, _) in SLASH_COMMANDS.items():
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


def display_banner() -> None:
    """显示启动 banner"""
    banner = pyfiglet.figlet_format("RAG Agent", font="slant")
    console.print(f"[cyan]{banner}[/cyan]", end="")
    console.print("[dim]v0.1.0[/dim]\n")


class InteractiveSession:
    """交互式会话"""

    def __init__(self, mode: str = "qa") -> None:
        self.mode = mode
        # 共享的 RAG 引擎，避免重复加载
        self.shared_engine = RAGEngine()
        self.app: BaseApp = QAApp(self.shared_engine) if mode == "qa" else ReportApp(self.shared_engine)
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(".rag_history"),
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
        )
        self._interrupt_count = 0  # 连续中断计数
        self._last_result = ""  # 最后一次的结果，用于PDF生成
        self._engine_initialized = False  # 引擎是否已初始化

    def get_prompt(self) -> str:
        """生成提示符"""
        icon = "💬" if self.mode == "qa" else "📝"
        name = "QA" if self.mode == "qa" else "Report"
        return f"{icon} {name} › "

    def switch_mode(self, mode: str) -> None:
        """切换模式"""
        if mode == self.mode:
            console.print(f"[dim]已在 {mode} 模式[/dim]")
            return

        self.mode = mode
        # 复用共享的 RAG 引擎，避免重新加载
        self.app = QAApp(self.shared_engine) if mode == "qa" else ReportApp(self.shared_engine)
        console.print(f"[green]✓ 切换到 {'问答' if mode == 'qa' else '报告'} 模式[/green]")

        # 如果引擎还未初始化，则初始化
        if not self._engine_initialized:
            self._init_app()
        else:
            # 如果引擎已初始化，只需要初始化应用状态
            self.app._initialized = True  # 复用引擎的初始化状态
            console.print(f"[green]✓ {'问答' if mode == 'qa' else '报告'} 应用就绪[/green]")

    def _init_app(self) -> bool:
        """初始化应用"""
        try:
            with console.status("[dim]加载中...[/dim]"):
                self.app.initialize()
            self._engine_initialized = True  # 标记引擎已初始化
            console.print("[green]✓ 就绪[/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ 初始化失败: {e}[/red]")
            return False

    def execute(self, query: str) -> None:
        """执行查询"""
        try:
            with console.status("[cyan]思考中...[/cyan]", spinner="dots"):
                result = self.app.run(query)

            # 存储最后一次结果（用于PDF生成）
            self._last_result = result

            if self.mode == "qa":
                console.print(Panel(result, border_style="cyan", padding=(0, 1)))
            else:
                console.print(Panel(Markdown(result), border_style="green", padding=(0, 1)))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")

    def generate_pdf_report(self, topic: str | None = None) -> None:
        """直接生成PDF报告"""
        if topic is None:
            console.print("[yellow]请指定报告主题: /pdf <主题>[/yellow]")
            return

        try:
            # 确保在报告模式或可以切换到报告模式
            if self.mode != "report":
                self.original_mode = self.mode  # 保存原始模式
                console.print("[dim]切换到报告模式生成PDF...[/dim]")
                self.switch_mode("report")

            # 生成PDF文件名
            import re
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = re.sub(r"[^\w\s-]", "", topic)[:20].strip()
            safe_topic = re.sub(r"[-\s]+", "_", safe_topic)
            output_path = Path(f"report_latex_{safe_topic}_{timestamp}.pdf")

            # 使用LaTeX生成PDF（进度由report_app显示）
            pdf_path = self.app.run(
                topic,
                output_format="latex",
                output_path=output_path,
            )

            # 如果返回的是路径，则使用该路径；否则可能是错误信息
            if isinstance(pdf_path, str) and pdf_path.endswith(".pdf"):
                # 存储最后结果（可能是LaTeX内容，但我们存储路径）
                self._last_result = pdf_path
                console.print(f"\n[green]✓ LaTeX PDF已生成: {pdf_path}[/green]")
            else:
                # 可能是错误信息或LaTeX内容
                console.print(f"\n[yellow]PDF生成可能有问题: {pdf_path[:100]}...[/yellow]")
                # 仍然存储
                self._last_result = pdf_path

            # 如果切换了模式，询问是否切换回去
            if hasattr(self, "original_mode") and self.original_mode != "report":
                console.print(f"[dim]提示：您仍在报告模式，使用 /{self.original_mode} 可切换回去[/dim]")

        except Exception as e:
            console.print(f"[red]PDF生成失败: {e}[/red]")

    def generate_pdf_from_last_result(self, topic: str | None = None) -> None:
        """从最后一次结果生成PDF（保留原功能）"""
        # 如果提供了topic参数，应该重新生成报告而不是使用缓存
        if topic is not None:
            console.print("[yellow]检测到主题参数，正在生成新的报告...[/yellow]")
            self.generate_pdf_report(topic)
            return

        if not hasattr(self, "_last_result") or not self._last_result:
            # 如果没有上次结果，提示用户指定主题
            console.print("[yellow]没有上次的查询结果，请使用 /pdf <主题> 来生成新的PDF报告[/yellow]")
            return

        # 没有topic参数时，使用_last_result生成PDF
        try:
            with console.status("[cyan]正在生成PDF...[/cyan]", spinner="dots"):
                # 生成PDF文件名
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_topic = "cached_result"  # 标记这是缓存的结果
                output_path = Path(f"report_{safe_topic}_{timestamp}.pdf")

                # 生成PDF
                pdf_path = generate_report_pdf(
                    content=self._last_result,
                    output_path=output_path,
                    title=f"技术报告: {topic}",
                )

            console.print(f"[green]✓ PDF已生成: {pdf_path}[/green]")
        except Exception as e:
            console.print(f"[red]PDF生成失败: {e}[/red]")

    def run(self) -> None:
        """主循环"""
        display_banner()

        try:
            config.validate()
        except ValueError as e:
            console.print(f"[red]配置错误: {e}[/red]")
            raise typer.Exit(1) from None

        if not self._init_app():
            raise typer.Exit(1)

        console.print("[dim]输入问题开始，/qa /report 切换模式，/exit 退出[/dim]\n")

        while True:
            try:
                user_input = self.session.prompt(self.get_prompt()).strip()

                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in ("/exit", "/quit", "/q"):
                    break
                elif cmd == "/qa":
                    self.switch_mode("qa")
                elif cmd == "/report":
                    self.switch_mode("report")
                elif cmd == "/clear":
                    console.clear()
                    display_banner()
                elif cmd == "/help":
                    console.print("\n[bold]命令:[/bold]")
                    console.print("  [cyan]/qa[/cyan]      切换问答模式")
                    console.print("  [cyan]/report[/cyan]  切换报告模式")
                    console.print("  [cyan]/pdf[/cyan]      生成PDF报告 [/pdf <主题>]")
                    console.print("  [cyan]/clear[/cyan]   清屏")
                    console.print("  [cyan]/exit[/cyan]    退出\n")
                elif user_input.startswith("/pdf"):
                    # 解析命令参数：/pdf [topic]
                    parts = user_input.split(maxsplit=1)
                    topic = parts[1] if len(parts) > 1 else None
                    self.generate_pdf_from_last_result(topic)
                elif user_input.startswith("/"):
                    console.print(f"[yellow]未知命令: {cmd}[/yellow]")
                else:
                    self.execute(user_input)

                # 正常输入，重置中断计数
                self._interrupt_count = 0

            except KeyboardInterrupt:
                self._interrupt_count += 1
                if self._interrupt_count >= 2:
                    console.print("\n[dim]再见！[/dim]")
                    break
                console.print("\n[dim]再按一次 Ctrl+C 退出[/dim]")
            except EOFError:
                break


# === Typer 命令 ===


@app.command()
def qa(
    query: Annotated[str | None, typer.Argument(help="问题（留空进入交互模式）")] = None,
) -> None:
    """问答模式 - 基于知识库的智能问答"""
    if query:
        display_banner()
        qa_app = QAApp()  # 单次命令使用独立实例
        with console.status("[dim]初始化...[/dim]"):
            qa_app.initialize()
        with console.status("[cyan]思考中...[/cyan]"):
            result = qa_app.run(query)
        console.print(Panel(result, border_style="cyan", padding=(0, 1)))
    else:
        session = InteractiveSession(mode="qa")
        session.run()


@app.command()
def report(
    topic: Annotated[str | None, typer.Argument(help="报告主题（留空进入交互模式）")] = None,
    output_format: Annotated[str, typer.Option("--format", "-f", help="输出格式: markdown 或 pdf")] = "markdown",
    output_path: Annotated[str | None, typer.Option("--output", "-o", help="输出文件路径（PDF 格式时使用）")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="显示详细信息")] = False,
) -> None:
    """报告模式 - 自动生成技术报告"""
    if topic:
        display_banner()
        report_app = ReportApp()  # 单次命令使用独立实例
        with console.status("[dim]初始化...[/dim]"):
            report_app.initialize()

        with console.status("[cyan]生成报告...[/cyan]"):
            result = report_app.run(topic, output_format=output_format, output_path=output_path, verbose=verbose)

        if output_format.lower() == "pdf" and result.endswith(".pdf"):
            console.print(
                Panel(f"PDF 报告已生成:\n[result_path]{result}[/result_path]", border_style="green", padding=(0, 1))
            )
        else:
            console.print(Panel(Markdown(result), border_style="green", padding=(0, 1)))
    else:
        session = InteractiveSession(mode="report")
        session.run()


@app.command()
def build(
    force: Annotated[bool, typer.Option("--force", "-f", help="强制重新构建")] = False,
) -> None:
    """🔨 构建向量数据库（首次使用需要）"""
    console.print("[bold cyan]构建向量数据库[/bold cyan]\n")

    try:
        loader = DatasetLoader(config.DATASET_NAME, load_all=True)
        documents = loader.load()

        engine = RAGEngine()
        engine.build_vectorstore(documents, force=force)

        console.print("\n[green]✓ 构建完成[/green]")
        console.print("[dim]运行 'rag-agent qa' 开始使用[/dim]")
    except Exception as e:
        console.print(f"[red]✗ 构建失败: {e}[/red]")
        raise typer.Exit(1) from None


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", "-v", help="显示版本")] = False,
) -> None:
    """RAG Agent - 智能问答与报告生成系统"""
    if version:
        console.print("RAG Agent v0.1.0")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        session = InteractiveSession(mode="qa")
        session.run()


if __name__ == "__main__":
    app()
