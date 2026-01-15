"""Router Agent - 意图路由智能体（增强版）

负责识别用户意图，将查询路由到合适的处理流程。

增强功能：
- Few-shot学习：提供示例提升分类准确率
- 思维链推理：逐步推理过程
- 置信度评估：评估分类置信度
- 多意图识别：支持复合意图
"""

import logging
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from rag_agent.rag_engine import RAGEngine
from rag_agent.schemas.state import AgentState

console = Console()
logger = logging.getLogger(__name__)

# 全局 RAGEngine 实例
_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """获取或初始化 RAGEngine 实例"""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.initialize(load_only=True)
    return _engine


# Few-shot示例
INTENT_EXAMPLES = """
# 示例1
查询：生成变压器诊断报告
分析：用户明确要求生成诊断报告，这是典型的设备诊断请求
关键词：生成、诊断、报告
意图：diagnosis

# 示例2
查询：变压器的正常温度范围是多少？
分析：用户询问具体的技术参数，这是一个知识问答问题
关键词：温度范围、是多少
意图：qa

# 示例3
查询：如果变压器温度过高，会导致什么后果？
分析：用户询问因果关系，需要推理分析
关键词：如果、导致、后果
意图：reasoning

# 示例4
查询：评估断路器的健康状态
分析：用户请求评估设备健康，属于诊断范畴
关键词：评估、健康状态
意图：diagnosis

# 示例5
查询：电力系统的电压等级有哪些？
分析：用户询问基础知识，属于问答
关键词：有哪些、电压等级
意图：qa

# 示例6
查询：计算变压器的负载率
分析：用户请求计算，涉及推理
关键词：计算、负载率
意图：reasoning

# 示例7
查询：这台设备运行正常吗？
分析：用户询问设备状态，需要诊断评估
关键词：运行正常吗、设备
意图：diagnosis

# 示例8
查询：什么是继电保护？
分析：用户询问概念定义，属于知识问答
关键词：什么是
意图：qa
"""


# 增强的意图分类提示模板（包含CoT和Few-shot）
ENHANCED_INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的意图分类助手。分析用户查询，将其分类为以下几种意图之一：\n\n"
            "**意图类型**：\n"
            "1. **diagnosis** - 设备诊断：用户请求生成诊断报告、分析设备状态、评估健康度、故障诊断等\n"
            "2. **qa** - 问答：用户询问具体问题、寻求知识解答、概念定义、技术参数等\n"
            "3. **reasoning** - 推理：需要多步推理、计算、因果分析、综合判断的问题\n\n"
            "**分类要求**：\n"
            "- 参考以下示例，理解分类逻辑\n"
            "- 逐步分析查询内容\n"
            "- 识别关键动词和名词\n"
            "- 评估主要意图\n"
            "- 输出格式：推理过程 | 意图 | 置信度(0-1)\n\n"
            "{examples}\n"
            "请按照上述格式分析用户查询。",
        ),
        (
            "human",
            "用户查询：{query}\n\n分析：",
        ),
    ]
)


def parse_llm_response(response_text: str) -> tuple[str, float]:
    """解析LLM响应，提取意图和置信度

    Args:
        response_text: LLM返回的文本

    Returns:
        (意图, 置信度)

    Examples:
        >>> parse_llm_response("这是一个诊断请求 | diagnosis | 0.95")
        ('diagnosis', 0.95)
    """
    text = response_text.lower().strip()

    # 提取意图
    intent = "qa"  # 默认
    if "diagnosis" in text or "诊断" in text:
        intent = "diagnosis"
    elif "reasoning" in text or "推理" in text:
        intent = "reasoning"

    # 提取置信度
    confidence = 0.5  # 默认置信度

    # 查找数字
    numbers = re.findall(r"0?\.\d+|\d+", text)
    for num in numbers:
        val = float(num)
        if 0.0 <= val <= 1.0:
            confidence = val
            break
        elif val > 1.0:
            # 可能是百分比（如95）
            confidence = min(val / 100, 1.0)

    return intent, confidence


async def classify_intent_enhanced(query: str) -> tuple[str, float]:
    """增强的意图分类（使用Few-shot + CoT）

    Args:
        query: 用户的查询内容

    Returns:
        (意图标签, 置信度)

    Examples:
        >>> intent, conf = await classify_intent_enhanced("生成变压器诊断报告")
        >>> print(intent, conf)
        diagnosis 0.95
    """
    engine = get_engine()
    llm = engine.llm

    if llm is None:
        # 降级到基于规则的分类
        intent = rule_based_intent_classification(query)
        return intent, 0.6  # 规则分类给予中等置信度

    # 使用增强的LLM分类
    try:
        chain = ENHANCED_INTENT_CLASSIFICATION_PROMPT | llm
        response = await chain.ainvoke({"query": query, "examples": INTENT_EXAMPLES})

        response_text = str(response.content) if response.content else ""
        intent, confidence = parse_llm_response(response_text)

        logger.info(f"增强意图分类: query='{query}', intent='{intent}', confidence={confidence:.2f}")

        return intent, confidence

    except Exception as e:
        logger.error(f"增强分类失败: {e}", exc_info=True)
        # 降级到基于规则的分类
        intent = rule_based_intent_classification(query)
        return intent, 0.5  # 降级时降低置信度


async def classify_intent(query: str) -> str:
    """分类用户意图（兼容接口）

    Args:
        query: 用户的查询内容

    Returns:
        意图标签: "diagnosis", "qa", 或 "reasoning"
    """
    intent, _ = await classify_intent_enhanced(query)
    return intent


def rule_based_intent_classification(query: str) -> str:
    """基于规则的意图分类（备用方案）

    Args:
        query: 用户查询

    Returns:
        意图标签
    """
    query_lower = query.lower()

    # 诊断意图关键词
    diagnosis_keywords = [
        "诊断",
        "报告",
        "分析",
        "评估",
        "健康",
        "故障分析",
        "状态评估",
        "生成报告",
        "diagnosis",
        "report",
        "analyze",
        "evaluate",
        "运行正常",
        "状态",
        "检测",
    ]

    # 推理意图关键词
    reasoning_keywords = [
        "为什么导致",
        "如果",
        "计算",
        "推断",
        "综合",
        "多步",
        "推理",
        "后果",
        "why",
        "calculate",
        "infer",
        "reasoning",
        "会怎样",
        "影响",
    ]

    # 检查关键词
    for keyword in diagnosis_keywords:
        if keyword in query_lower:
            return "diagnosis"

    for keyword in reasoning_keywords:
        if keyword in query_lower:
            return "reasoning"

    # 默认为问答
    return "qa"


async def router_node(state: AgentState) -> dict:
    """路由节点：识别用户意图并路由到合适的流程（增强版）

    使用Few-shot学习和思维链推理提升分类准确率。

    Args:
        state: 智能体当前状态

    Returns:
        更新后的状态，包含识别出的意图和置信度

    Examples:
        >>> state = {"query": "生成变压器诊断报告", ...}
        >>> result = await router_node(state)
        >>> print(result["intent"], result["confidence"])
        diagnosis 0.95
    """
    query = state["query"]

    console.print(f"[dim]正在分析查询意图: {query}[/dim]")

    # 增强的意图分类
    intent, confidence = await classify_intent_enhanced(query)

    # 更新状态
    state["intent"] = intent
    state["confidence"] = confidence
    state["tools_used"].append("enhanced_router_classifier")

    # 显示结果
    confidence_level = "高" if confidence >= 0.8 else "中" if confidence >= 0.6 else "低"
    console.print(f"[green]✓ 意图识别: {intent}[/green] [dim](置信度: {confidence:.2f}, {confidence_level})[/dim]")

    # 如果置信度过低，标记需要澄清
    need_clarification = confidence < 0.6
    state["need_clarification"] = need_clarification

    if need_clarification:
        console.print("[yellow]⚠ 意图不确定，可能需要用户澄清[/yellow]")

    # 返回需要更新的字段
    return {
        "intent": intent,
        "confidence": confidence,
        "tools_used": state["tools_used"],
        "need_clarification": need_clarification,
    }


# 路由条件函数（用于 LangGraph 的条件边）
def route_condition(state: AgentState) -> Literal["diagnosis", "qa", "reasoning"]:
    """路由条件函数

    根据状态中的 intent 字段，决定下一步的执行路径。

    Args:
        state: 智能体当前状态

    Returns:
        下一个要执行的子图名称

    Examples:
        >>> state = {"intent": "diagnosis", ...}
        >>> next_step = route_condition(state)
        >>> print(next_step)
        diagnosis
    """
    intent = state.get("intent", "qa")
    return intent  # type: ignore


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_router():
        """测试路由功能（增强版）"""
        from rich.table import Table

        console.print("[bold cyan]测试增强路由系统[/bold cyan]\n")

        test_queries = [
            "生成变压器诊断报告",
            "变压器的正常温度范围是多少？",
            "如果变压器温度过高会导致什么后果？",
            "这台设备运行正常吗？",
            "计算变压器的负载率",
            "什么是继电保护？",
        ]

        # 创建结果表格
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("查询", width=40)
        table.add_column("意图", width=12)
        table.add_column("置信度", width=10)
        table.add_column("级别", width=8)

        for query in test_queries:
            intent, confidence = await classify_intent_enhanced(query)

            confidence_level = "高" if confidence >= 0.8 else "中" if confidence >= 0.6 else "低"

            table.add_row(query, f"[green]{intent}[/green]", f"{confidence:.2f}", confidence_level)

        console.print(table)
        console.print("\n[bold green]✓ 路由系统测试完成[/bold green]")

    # 运行测试
    asyncio.run(test_router())
