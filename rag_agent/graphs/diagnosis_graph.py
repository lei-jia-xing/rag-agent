"""Diagnosis Graph - 设备诊断流程图

优化后的10节点诊断流程（并行化）：

    retrieval → core_assessment ─┬─→ fault_analysis ──┐
                                 ├─→ risk_analysis ───┤
                                 ├─→ device_info ─────┼─→ maintenance → validation → merge → report
                                 └─→ monitoring ──────┘

并行分支：
- fault_analysis, risk_analysis, device_info, monitoring 可并行执行
- maintenance 依赖 fault_analysis 和 risk_analysis 的结果
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_agent.agents.diagnosis_agent import (
    core_assessment_node,
    maintenance_node,
    merge_fields_node,
    parallel_analysis_node,
    report_node,
    retrieval_node,
    validation_node,
)
from rag_agent.schemas.state import DiagnosisState


def build_diagnosis_graph() -> CompiledStateGraph:
    """构建诊断流程图（10节点，优化并行）

    优化点：
    1. fault_analysis, risk_analysis, device_info, monitoring 并行执行
    2. 减少约 50% 的 LLM 调用等待时间
    """

    workflow = StateGraph(DiagnosisState)

    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("core_assessment", core_assessment_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("maintenance", maintenance_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("merge", merge_fields_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("retrieval")

    workflow.add_edge("retrieval", "core_assessment")
    workflow.add_edge("core_assessment", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "maintenance")
    workflow.add_edge("maintenance", "validation")
    workflow.add_edge("validation", "merge")
    workflow.add_edge("merge", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def visualize_diagnosis_graph() -> str:
    return """
┌─────────────────────────────────────────────────────────────┐
│         Diagnosis Agent Flow (Optimized - 7 Steps)          │
└─────────────────────────────────────────────────────────────┘

    Start
      ↓
    ┌──────────────────┐
    │  1. Retrieval    │  检索设备文档 (k=10)
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  2. Core         │  核心健康评估
    │     Assessment   │  - health_score (0-100)
    └────────┬─────────┘  - health_status, risk_level
             ↓
    ┌──────────────────────────────────────────┐
    │  3-6. Parallel Analysis (asyncio.gather) │
    │  ┌────────────┐ ┌────────────┐          │
    │  │ Fault      │ │ Risk       │          │
    │  │ Analysis   │ │ Analysis   │          │
    │  └────────────┘ └────────────┘          │
    │  ┌────────────┐ ┌────────────┐          │
    │  │ Device     │ │ Monitoring │          │
    │  │ Info       │ │ Analysis   │          │
    │  └────────────┘ └────────────┘          │
    └────────────────────┬─────────────────────┘
                         ↓
    ┌──────────────────┐
    │  7. Maintenance  │  维护建议字段组
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  8. Validation   │  一致性校验
    └────────┬─────────┘  - 检查矛盾、自动修正
             ↓
    ┌──────────────────┐
    │  9. Merge        │  合并所有字段
    └────────┬─────────┘  - Pydantic 验证
             ↓
    ┌──────────────────┐
    │  10. Report      │  生成 PDF 报告
    └────────┬─────────┘  - LaTeX MCP
             ↓
           END
"""


if __name__ == "__main__":
    import asyncio

    from rich.console import Console

    console = Console()

    async def test():
        console.print("[bold cyan]测试诊断流程[/bold cyan]\n")
        console.print(visualize_diagnosis_graph())

        graph = build_diagnosis_graph()
        console.print("[green]✓ 图构建成功[/green]\n")

        initial_state: DiagnosisState = {
            "query": "变压器",
            "device_name": "",
            "documents": [],
            "messages": [],
        }

        result = await graph.ainvoke(initial_state)
        console.print(f"\n报告路径: {result.get('report_path', 'N/A')}")

    asyncio.run(test())
