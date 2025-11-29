"""CLI 交互界面"""

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from rag_agent.config import config
from rag_agent.data_loader import DatasetLoader
from rag_agent.rag_engine import RAGEngine

console = Console()


class CLI:
    """命令行界面"""

    def __init__(self):
        """初始化 CLI"""
        self.rag_engine = RAGEngine()
        self.running = False

        # 创建 prompt_toolkit session，支持历史记录
        self.session = PromptSession(history=InMemoryHistory())

        # 定义提示符样式
        self.prompt_style = Style.from_dict(
            {
                "prompt": "#00aa00 bold",
            }
        )

    def display_banner(self):
        """显示欢迎横幅"""
        banner = """
╔═══════════════════════════════════════════════════╗
║     设备问答智能体 RAG Agent                      ║
║     基于电气工程知识库                            ║
╚═══════════════════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold cyan"))

    def initialize_system(self):
        """初始化系统"""
        try:
            # 验证配置
            config.validate()

            # 加载数据集
            console.print(
                f"\n[cyan]正在加载数据集: {config.DATASET_NAME} "
                f"(采样 {config.DATASET_SAMPLE_SIZE} 条)[/cyan]"
            )

            loader = DatasetLoader(
                dataset_name=config.DATASET_NAME,
                split=config.DATASET_SPLIT,
                sample_size=config.DATASET_SAMPLE_SIZE,
            )
            documents = loader.load()

            # 初始化 RAG 引擎
            self.rag_engine.initialize(documents)

            console.print("\n[green]✓ 系统初始化完成！[/green]")
            console.print(
                "[dim]提示: 输入问题开始对话，使用 Ctrl+C 或输入 'exit'/'quit' 退出[/dim]\n"
            )

        except Exception as e:
            console.print(f"[red]✗ 初始化失败: {e}[/red]")
            raise

    def display_answer(self, result: dict):
        """
        显示答案

        Args:
            result: 查询结果
        """
        # 显示答案
        console.print("\n[bold green]回答:[/bold green]")
        console.print(Panel(result["answer"], style="green", padding=(1, 2)))

        # 显示来源文档
        if result["source_documents"]:
            console.print("\n[bold blue]参考来源:[/bold blue]")
            for i, doc in enumerate(result["source_documents"][:3], 1):
                content = (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                )
                console.print(f"[dim blue]{i}. {content}[/dim blue]\n")

    def run(self):
        """运行 CLI"""
        self.display_banner()

        try:
            self.initialize_system()
            self.running = True

            while self.running:
                try:
                    # 使用 prompt_toolkit 获取用户输入
                    question = self.session.prompt(
                        [
                            ("class:prompt", "❓ 问题"),
                            ("", " ❯ "),
                        ],
                        style=self.prompt_style,
                    )

                    # 检查退出命令
                    if question.lower() in ["exit", "quit", "q", "退出"]:
                        console.print("\n[yellow]再见！[/yellow]")
                        break

                    # 检查空输入
                    if not question.strip():
                        continue

                    # 查询问题
                    result = self.rag_engine.query(question)
                    self.display_answer(result)

                except KeyboardInterrupt:
                    # Ctrl+C 继续
                    console.print("\n[dim]按 Ctrl+C 再次或输入 'exit' 退出[/dim]")
                    continue
                except EOFError:
                    # Ctrl+D 退出
                    console.print("\n[yellow]再见！[/yellow]")
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]程序已中断[/yellow]")
        except Exception as e:
            console.print(f"\n[red]发生错误: {e}[/red]")
            raise


def start_cli():
    """启动 CLI"""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    start_cli()
