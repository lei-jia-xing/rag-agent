"""Diagnosis Graph - 诊断智能体流程图

使用 LangGraph 构建设备诊断的完整流程：
检索 → 分析 → 字段生成 → 报告生成
"""


from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraphType

from rag_agent.agents.diagnosis_agent import (
    diagnosis_analysis_node,
    diagnosis_fields_node,
    diagnosis_report_node,
    diagnosis_retrieval_node,
)
from rag_agent.schemas.state import DiagnosisState


def build_diagnosis_graph() -> CompiledGraphType:
    """构建设备诊断智能体图

    创建一个完整的诊断流程，包含以下节点：
    1. retrieval - 检索相关文档
    2. analysis - 分析设备状态
    3. fields - 生成诊断字段
    4. report - 生成 PDF 报告

    Returns:
        编译后的 CompiledStateGraph，可用于执行诊断流程

    Examples:
        >>> graph = build_diagnosis_graph()
        >>> result = await graph.ainvoke({
        ...     "query": "变压器异常振动",
        ...     "device_name": "",
        ...     "documents": [],
        ...     "diagnosis_data": {},
        ...     "report_path": "",
        ...     "analysis_result": "",
        ...     "messages": []
        ... })
        >>> print(result["report_path"])
    """

    workflow = StateGraph(DiagnosisState)

    # 添加节点
    workflow.add_node("retrieval", diagnosis_retrieval_node)
    workflow.add_node("analysis", diagnosis_analysis_node)
    workflow.add_node("fields", diagnosis_fields_node)
    workflow.add_node("report", diagnosis_report_node)

    # 设置入口点
    workflow.set_entry_point("retrieval")

    # 添加边（定义执行流程）
    workflow.add_edge("retrieval", "analysis")
    workflow.add_edge("analysis", "fields")
    workflow.add_edge("fields", "report")
    workflow.add_edge("report", END)

    # 编译图
    app = workflow.compile()

    return app


# 可视化图的执行流程
def visualize_diagnosis_graph() -> str:
    """生成诊断图的文本表示

    Returns:
        图的文本表示
    """
    return """
┌─────────────────────────────────────────────────────┐
│            Diagnosis Agent Flow                     │
└─────────────────────────────────────────────────────┘

    Start
      ↓
    ┌──────────────┐
    │  Retrieval   │  检索设备文档
    │  Node        │  - k=10 获取更多上下文
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │  Analysis    │  分析设备状态
    │  Node        │  - LLM 分析
    └──────┬───────┘  - 识别问题
           ↓           - 评估风险
    ┌──────────────┐
    │  Fields      │  生成诊断字段
    │  Node        │  - 32个字段
    └──────┬───────┘  - 用于报告
           ↓
    ┌──────────────┐
    │  Report      │  生成PDF报告
    │  Node        │  - LaTeX MCP
    └──────┬───────┘  - 专业模板
           ↓
         END
"""


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_diagnosis_graph():
        """测试诊断图"""
        from rich.console import Console

        console = Console()

        console.print("[bold cyan]测试 Diagnosis Graph[/bold cyan]\n")

        # 显示图结构
        console.print(visualize_diagnosis_graph())

        # 构建图
        console.print("[yellow]构建诊断图...[/yellow]")
        graph = build_diagnosis_graph()
        console.print("[green]✓ 诊断图构建成功[/green]\n")

        # 准备测试输入
        console.print("[yellow]执行诊断流程...[/yellow]\n")
        initial_state: DiagnosisState = {
            "query": "变压器异常振动",
            "device_name": "",
            "documents": [],
            "diagnosis_data": {},
            "report_path": "",
            "analysis_result": "",
            "messages": [],
        }

        try:
            # 执行图
            result = await graph.ainvoke(initial_state)

            # 显示结果
            console.print("\n[bold cyan]执行结果[/bold cyan]")
            console.print(f"设备名称: {result['device_name']}")
            console.print(f"检索文档数: {len(result['documents'])}")
            console.print(f"分析结果长度: {len(result['analysis_result'])} 字符")
            console.print(f"诊断字段数: {len(result['diagnosis_data'])}")
            console.print(f"报告路径: {result['report_path']}")

            console.print("\n[bold green]✓ 测试完成[/bold green]")

        except Exception as e:
            console.print(f"\n[red]✗ 测试失败: {e}[/red]")
            import traceback

            traceback.print_exc()

    # 运行测试
    asyncio.run(test_diagnosis_graph())
