"""RAG Agent CLI

使用 Typer 构建的专业命令行界面。
"""

from typing import Annotated, Optional

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
        self.app: BaseApp = QAApp() if mode == "qa" else ReportApp()
        self.session: PromptSession[str] = PromptSession(
            history=FileHistory(".rag_history"),
            completer=SlashCommandCompleter(),
            complete_while_typing=True,
        )
        self._interrupt_count = 0  # 连续中断计数

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
        self.app = QAApp() if mode == "qa" else ReportApp()
        console.print(f"[green]✓ 切换到 {'问答' if mode == 'qa' else '报告'} 模式[/green]")
        self._init_app()

    def _init_app(self) -> bool:
        """初始化应用"""
        try:
            with console.status("[dim]加载中...[/dim]"):
                self.app.initialize()
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

            if self.mode == "qa":
                console.print(Panel(result, border_style="cyan", padding=(0, 1)))
            else:
                console.print(Panel(Markdown(result), border_style="green", padding=(0, 1)))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")

    def run(self) -> None:
        """主循环"""
        display_banner()

        try:
            config.validate()
        except ValueError as e:
            console.print(f"[red]配置错误: {e}[/red]")
            raise typer.Exit(1)

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
                    console.print("  [cyan]/clear[/cyan]   清屏")
                    console.print("  [cyan]/exit[/cyan]    退出\n")
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
    query: Annotated[Optional[str], typer.Argument(help="问题（留空进入交互模式）")] = None,
) -> None:
    """问答模式 - 基于知识库的智能问答"""
    if query:
        display_banner()
        qa_app = QAApp()
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
    topic: Annotated[Optional[str], typer.Argument(help="报告主题（留空进入交互模式）")] = None,
) -> None:
    """报告模式 - 自动生成技术报告"""
    if topic:
        display_banner()
        report_app = ReportApp()
        with console.status("[dim]初始化...[/dim]"):
            report_app.initialize()
        with console.status("[cyan]生成报告...[/cyan]"):
            result = report_app.run(topic)
        console.print(Panel(Markdown(result), border_style="green", padding=(0, 1)))
    else:
        session = InteractiveSession(mode="report")
        session.run()


@app.command()
def build(
    force: Annotated[bool, typer.Option("--force", "-f", help="强制重新构建")] = False,
) -> None:
    """🔨 构建向量数据库（首次使用需要）"""
    console.print("[bold cyan]🔨 构建向量数据库[/bold cyan]\n")

    try:
        loader = DatasetLoader(config.DATASET_NAME, load_all=True)
        documents = loader.load()

        engine = RAGEngine()
        engine.build_vectorstore(documents, force=force)

        console.print("\n[green]✓ 构建完成[/green]")
        console.print("[dim]运行 'rag-agent qa' 开始使用[/dim]")
    except Exception as e:
        console.print(f"[red]✗ 构建失败: {e}[/red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", "-v", help="显示版本")] = False,
) -> None:
    """RAG Agent - 智能问答与报告生成系统"""
    if version:
        console.print("RAG Agent v0.1.0")
        raise typer.Exit()

    # 没有子命令时，默认进入 QA 交互模式
    if ctx.invoked_subcommand is None:
        session = InteractiveSession(mode="qa")
        session.run()


if __name__ == "__main__":
    app()
