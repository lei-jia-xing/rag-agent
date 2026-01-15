"""Main Graph - 主智能体图

集成路由、诊断和问答功能的主智能体。
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraphType

from rag_agent.agents.router import route_condition, router_node
from rag_agent.schemas.state import AgentState


def build_main_graph() -> CompiledGraphType:
    """构建主智能体图（带路由）

    整合所有子智能体，根据用户意图自动路由：
    - diagnosis → Diagnosis Agent
    - qa → QA Agent
    - reasoning → Reasoning Agent (待实现)

    Returns:
        编译后的 CompiledStateGraph

    Examples:
        >>> graph = build_main_graph()
        >>> result = await graph.ainvoke({
        ...     "query": "生成变压器诊断报告",
        ...     "intent": "",
        ...     "context": "",
        ...     "documents": [],
        ...     "reasoning_steps": [],
        ...     "tools_used": [],
        ...     "answer": "",
        ...     "diagnosis_data": {},
        ...     "report_path": "",
        ...     "messages": [],
        ...     "confidence": 0.0,
        ...     "need_clarification": False
        ... })
    """

    workflow = StateGraph(AgentState)

    # 添加路由节点
    workflow.add_node("router", router_node)

    # 添加子图（使用 send/await 模式）
    # 注意：LangGraph 中子图的集成需要特殊处理
    # 这里我们先添加节点，然后在条件边中调用子图

    # 为了简化，我们直接将子图的入口节点作为主图的节点
    # 实际应用中可能需要更复杂的集成方式
    from rag_agent.agents.diagnosis_agent import (
        core_assessment_node as diagnosis_analysis_node,
    )
    from rag_agent.agents.diagnosis_agent import (
        merge_fields_node as diagnosis_fields_node,
    )
    from rag_agent.agents.diagnosis_agent import (
        report_node as diagnosis_report_node,
    )
    from rag_agent.agents.diagnosis_agent import (
        retrieval_node as diagnosis_retrieval_node,
    )
    from rag_agent.agents.qa_agent import qa_retrieval_node, qa_synthesis_node

    # 添加诊断节点
    workflow.add_node("diagnosis_retrieval", diagnosis_retrieval_node)
    workflow.add_node("diagnosis_analysis", diagnosis_analysis_node)
    workflow.add_node("diagnosis_fields", diagnosis_fields_node)
    workflow.add_node("diagnosis_report", diagnosis_report_node)

    # 添加问答节点
    workflow.add_node("qa_retrieval", qa_retrieval_node)
    workflow.add_node("qa_synthesis", qa_synthesis_node)

    # 设置入口点
    workflow.set_entry_point("router")

    # 添加条件边（路由到不同的子流程）
    workflow.add_conditional_edges(
        "router",
        route_condition,
        {
            "diagnosis": "diagnosis_retrieval",
            "qa": "qa_retrieval",
            "reasoning": "qa_retrieval",  # 暂时使用 QA 流程
        },
    )

    # 添加诊断流程边
    workflow.add_edge("diagnosis_retrieval", "diagnosis_analysis")
    workflow.add_edge("diagnosis_analysis", "diagnosis_fields")
    workflow.add_edge("diagnosis_fields", "diagnosis_report")
    workflow.add_edge("diagnosis_report", END)

    # 添加问答流程边
    workflow.add_edge("qa_retrieval", "qa_synthesis")
    workflow.add_edge("qa_synthesis", END)

    # 编译图
    app = workflow.compile()

    return app


def visualize_main_graph() -> str:
    """生成主图的文本表示

    Returns:
        图的文本表示
    """
    return """
┌─────────────────────────────────────────────────────┐
│             Main Agent Flow (Router)                 │
└─────────────────────────────────────────────────────┘

    Start
      ↓
    ┌─────────┐
    │ Router  │  意图分类
    └────┬────┘
         │
    ┌────┴────────┬────────────┐
    ↓             ↓             ↓
┌───────┐    ┌───────┐    ┌───────┐
│Diag.  │    │  QA   │    |Reason |
│ Flow  │    │ Flow  │    │ Flow  │
└───┬───┘    └───┬───┘    └───┬───┘
    │            │            │
    ↓            ↓            ↓
[Retrieval]  [Retrieval]  [Retrieval]
    ↓            ↓            ↓
[Analysis]   [Synthesis]  [Synthesis]
    ↓            ↓            ↓
[Fields]          └────────────┤
    ↓                        ↓
[Report]                  END
    ↓
   END
"""


async def main_agent_invoke(query: str, session_id: str = "default") -> dict:
    """主智能体调用接口

    这是对外提供的主要接口，处理用户查询。

    Args:
        query: 用户查询
        session_id: 会话 ID（用于记忆）

    Returns:
        执行结果

    Examples:
        >>> result = await main_agent_invoke(
        ...     "生成变压器诊断报告",
        ...     session_id="user-123"
        ... )
        >>> print(result["answer"])
    """
    graph = build_main_graph()

    # 初始状态
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "context": "",
        "documents": [],
        "reasoning_steps": [],
        "tools_used": [],
        "answer": "",
        "diagnosis_data": {},
        "report_path": "",
        "messages": [],
        "confidence": 0.0,
        "need_clarification": False,
    }

    # 执行图
    result = await graph.ainvoke(initial_state)

    return result


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_main_graph():
        """测试主图"""
        from rich.console import Console

        console = Console()

        console.print("[bold cyan]测试 Main Graph[/bold cyan]\n")

        # 显示图结构
        console.print(visualize_main_graph())
        console.print()

        # 测试用例
        test_cases = [
            ("生成变压器诊断报告", "diagnosis"),
            ("变压器的正常温度范围是多少？", "qa"),
        ]

        for query, expected_intent in test_cases:
            console.print(f"[yellow]测试查询: {query}[/yellow]")
            console.print(f"[dim]预期意图: {expected_intent}[/dim]")

            try:
                result = await main_agent_invoke(query)
                console.print(f"✓ 识别意图: {result['intent']}")

                if result["report_path"]:
                    console.print(f"✓ 报告路径: {result['report_path']}")
                elif result["answer"]:
                    console.print(f"✓ 答案: {result['answer'][:100]}...")

                console.print()

            except Exception as e:
                console.print(f"✗ 失败: {e}\n")
                import traceback

                traceback.print_exc()

        console.print("[bold green]✓ 测试完成[/bold green]")

    asyncio.run(test_main_graph())
