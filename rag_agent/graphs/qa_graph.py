"""QA Graph - 问答智能体流程图

使用 LangGraph 构建智能问答的完整流程：
检索 → 答案合成
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraphType

from rag_agent.agents.qa_agent import qa_retrieval_node, qa_synthesis_node
from rag_agent.schemas.state import QAState


def build_qa_graph() -> CompiledGraphType:
    """构建问答智能体图

    创建一个完整的问答流程，包含以下节点：
    1. retrieval - 检索相关知识
    2. synthesis - 合成答案

    Returns:
        编译后的 CompiledStateGraph，可用于执行问答流程

    Examples:
        >>> graph = build_qa_graph()
        >>> result = await graph.ainvoke({
        ...     "query": "变压器的正常温度范围是多少？",
        ...     "documents": [],
        ...     "context": "",
        ...     "answer": "",
        ...     "confidence": 0.0,
        ...     "sources": [],
        ...     "messages": []
        ... })
        >>> print(result["answer"])
    """

    workflow = StateGraph(QAState)

    # 添加节点
    workflow.add_node("retrieval", qa_retrieval_node)
    workflow.add_node("synthesis", qa_synthesis_node)

    # 设置入口点
    workflow.set_entry_point("retrieval")

    # 添加边（定义执行流程）
    workflow.add_edge("retrieval", "synthesis")
    workflow.add_edge("synthesis", END)

    # 编译图
    app = workflow.compile()

    return app


def visualize_qa_graph() -> str:
    """生成问答图的文本表示

    Returns:
        图的文本表示
    """
    return """
┌─────────────────────────────────────────────────────┐
│               QA Agent Flow                          │
└─────────────────────────────────────────────────────┘

    Start
      ↓
    ┌──────────────┐
    │  Retrieval   │  检索相关知识
    │  Node        │  - k=5 个文档
    └──────┬───────┘  - 格式化上下文
           ↓
    ┌──────────────┐
    │  Synthesis   │  合成答案
    │  Node        │  - LLM 生成
    └──────┬───────┘  - 计算置信度
           ↓           - 提取来源
         END
"""


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_qa_graph():
        """测试问答图"""
        from rich.console import Console

        console = Console()

        console.print("[bold cyan]测试 QA Graph[/bold cyan]\n")

        # 显示图结构
        console.print(visualize_qa_graph())

        # 构建图
        console.print("[yellow]构建问答图...[/yellow]")
        graph = build_qa_graph()
        console.print("[green]✓ 问答图构建成功[/green]\n")

        # 准备测试输入
        console.print("[yellow]执行问答流程...[/yellow]\n")
        initial_state: QAState = {
            "query": "变压器的正常温度范围是多少？",
            "documents": [],
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "sources": [],
            "messages": [],
        }

        try:
            # 执行图
            result = await graph.ainvoke(initial_state)

            # 显示结果
            console.print("\n[bold cyan]执行结果[/bold cyan]")
            console.print(f"问题: {result['query']}")
            console.print(f"检索文档数: {len(result['documents'])}")
            console.print(f"上下文长度: {len(result['context'])} 字符")
            console.print(f"答案长度: {len(result['answer'])} 字符")
            console.print(f"置信度: {result['confidence']:.2f}")
            console.print(f"来源数: {len(result['sources'])}")

            if result["sources"]:
                console.print(f"来源: {', '.join(result['sources'][:3])}")

            console.print(f"\n[bold]答案:[/bold]\n{result['answer']}")

            console.print("\n[bold green]✓ 测试完成[/bold green]")

        except Exception as e:
            console.print(f"\n[red]✗ 测试失败: {e}[/red]")
            import traceback

            traceback.print_exc()

    # 运行测试
    asyncio.run(test_qa_graph())
