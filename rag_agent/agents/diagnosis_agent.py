"""Diagnosis Agent - 设备诊断智能体

负责设备健康诊断的完整流程：
1. 检索相关文档（使用增强检索）
2. 分析设备状态
3. 生成诊断字段
4. 生成PDF报告
"""

import asyncio
import json
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from rag_agent.rag_engine import RAGEngine
from rag_agent.retrieval import EnhancedRetriever
from rag_agent.schemas.diagnosis import DiagnosisFields
from rag_agent.schemas.state import DiagnosisState

console = Console()
logger = logging.getLogger(__name__)

# 全局实例
_engine: RAGEngine | None = None
_enhanced_retriever: EnhancedRetriever | None = None


def get_engine() -> RAGEngine:
    """获取或初始化 RAGEngine 实例"""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.initialize(load_only=True)
    return _engine


def get_enhanced_retriever() -> EnhancedRetriever:
    """获取或初始化增强检索器"""
    global _enhanced_retriever
    if _enhanced_retriever is None:
        engine = get_engine()
        _enhanced_retriever = EnhancedRetriever(
            engine=engine,
            enable_query_expansion=True,
            enable_hybrid_search=False,  # 需要文档构建BM25索引
            enable_reranking=False,  # 可选：启用重排序
        )
        logger.info("增强检索器初始化完成")
    return _enhanced_retriever


async def diagnosis_retrieval_node(state: DiagnosisState) -> dict:
    """诊断流程 - 检索节点（使用增强检索）

    根据设备名称或问题描述，使用增强检索技术检索相关的技术文档、
    故障案例、维护记录等信息。

    增强功能：
    - 查询重写：优化设备名称描述
    - 多查询生成：从不同角度检索设备信息
    - 更多上下文：k=10获取更全面的信息

    Args:
        state: 诊断状态，包含查询信息

    Returns:
        更新后的状态，包含检索到的文档

    Examples:
        >>> state = {"query": "变压器异常振动", "device_name": "", ...}
        >>> result = await diagnosis_retrieval_node(state)
        >>> print(f"检索到 {len(result['documents'])} 个文档")
    """
    query = state["query"]
    device_name = state.get("device_name", query)

    console.print(f"[dim]正在使用增强检索设备文档: {device_name}[/dim]")

    try:
        retriever = get_enhanced_retriever()

        # 使用增强检索（获取更多文档用于诊断）
        documents = retriever.retrieve(
            query=device_name,
            top_k=10,  # 诊断需要更多上下文
            enable_query_expansion=True,
            enable_multi_query=True,  # 多角度检索
            enable_hyde=False,  # HyDE（可选启用）
        )

        # 更新设备名称（如果之前为空）
        if not state.get("device_name"):
            state["device_name"] = device_name

        console.print(f"[green]✓ 检索到 {len(documents)} 个相关文档（增强检索）[/green]")

        return {
            "documents": documents,
            "device_name": device_name,
        }

    except Exception as e:
        console.print(f"[red]✗ 增强检索失败: {e}[/red]")
        logger.error(f"增强检索失败: {e}", exc_info=True)

        # 降级到基础检索
        console.print("[yellow]降级到基础检索...[/yellow]")
        try:
            engine = get_engine()
            documents = engine.retrieve(device_name, k=10)

            console.print(f"[yellow]✓ 基础检索返回 {len(documents)} 个文档[/yellow]")

            return {
                "documents": documents,
                "device_name": device_name,
            }
        except Exception as e2:
            console.print(f"[red]✗ 基础检索也失败: {e2}[/red]")
            return {
                "documents": [],
                "device_name": device_name,
            }


async def diagnosis_analysis_node(state: DiagnosisState) -> dict:
    """诊断流程 - 分析节点

    基于检索到的文档，使用 LLM 分析设备的当前状态、
    识别潜在问题、评估健康程度。

    Args:
        state: 诊断状态，包含检索到的文档

    Returns:
        更新后的状态，包含分析结果

    Examples:
        >>> state = {"device_name": "变压器", "documents": [...], ...}
        >>> result = await diagnosis_analysis_node(state)
        >>> print(result["analysis_result"])
    """
    device_name = state["device_name"]
    documents = state["documents"]

    if not documents:
        error_msg = f"未找到 {device_name} 的相关文档，无法进行分析"
        console.print(f"[red]✗ {error_msg}[/red]")
        return {"analysis_result": error_msg}

    console.print("[dim]正在分析设备状态...[/dim]")

    try:
        engine = get_engine()
        llm = engine.llm

        if llm is None:
            raise RuntimeError("LLM 未初始化")

        # 准备上下文
        context = "\n\n".join([doc.page_content for doc in documents])

        # 构建分析提示
        analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的设备健康诊断专家。"
                    "基于提供的文档信息，分析设备的当前状态。\n\n"
                    "请提供以下方面的分析：\n"
                    "1. 设备整体状态评估\n"
                    "2. 发现的问题和异常\n"
                    "3. 潜在的风险\n"
                    "4. 紧急程度评估\n"
                    "5. 建议的处理措施\n\n"
                    "请以结构化的方式输出分析结果。",
                ),
                (
                    "human",
                    "设备名称：{device_name}\n\n相关文档信息：\n{context}\n\n请基于以上信息，提供详细的诊断分析。",
                ),
            ]
        )

        # 执行分析
        chain = analysis_prompt | llm
        response = await chain.ainvoke({"device_name": device_name, "context": context})

        analysis_result = response.content

        console.print("[green]✓ 设备状态分析完成[/green]")

        return {"analysis_result": analysis_result}

    except Exception as e:
        error_msg = f"分析失败: {e}"
        console.print(f"[red]✗ {error_msg}[/red]")
        logger.error(f"分析失败: {e}", exc_info=True)
        return {"analysis_result": error_msg}


DIAGNOSIS_FIELDS_PROMPT = """你是一个专业的设备健康诊断专家。基于提供的上下文信息和分析结果，生成设备健康诊断报告的各个字段内容。

**重要**：你必须返回一个纯JSON格式的响应，不要包含任何其他文字。

返回的JSON必须包含以下字段：
- title: 报告标题
- report_id: 报告编号（格式：DX-YYYYMMDD-001）
- device_name: 设备名称
- device_model: 设备型号
- location: 安装位置
- diagnosis_date: 诊断日期
- data_range: 数据采集范围
- health_score: 整体健康评分（0-100）
- health_status: 健康状态（正常/警告/异常/严重）
- risk_level: 风险等级（低/中/高）
- issue_count: 主要问题数
- abstract: 诊断摘要
- device_basic_info: 设备基本信息
- operating_environment: 运行环境
- maintenance_history: 历史维护记录
- monitoring_data_summary: 监测数据汇总
- key_metrics_analysis: 关键指标分析
- trend_analysis: 趋势分析
- anomaly_detection: 异常检测
- fault_description: 故障现象描述
- fault_cause_analysis: 故障原因分析
- fault_location: 故障定位
- urgent_measures: 紧急处理措施
- maintenance_plan: 维护计划
- spare_parts_suggestion: 备件建议
- current_risks: 当前风险
- potential_risks: 潜在风险
- risk_control: 风险控制建议
- conclusion_and_recommendations: 结论与建议
- technical_parameters: 技术参数
- related_standards: 相关标准
- diagnosis_method: 诊断方法说明"""


async def generate_diagnosis_fields_enhanced(
    engine: RAGEngine,
    device_name: str,
    documents: list[Document],
    analysis_result: str = "",
) -> dict[str, Any]:
    """生成诊断字段，整合分析结果作为额外上下文"""
    if engine.llm is None:
        raise RuntimeError("LLM 未初始化")

    context = "\n\n".join([doc.page_content for doc in documents])

    human_message = f"设备名称：{device_name}\n\n参考文档：\n{context}"
    if analysis_result:
        human_message += f"\n\n已有分析结果：\n{analysis_result}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DIAGNOSIS_FIELDS_PROMPT),
            ("human", human_message),
        ]
    )

    chain = prompt | engine.llm
    response = await chain.ainvoke({})
    response_text = str(response.content)

    try:
        raw_data = json.loads(response_text)
    except json.JSONDecodeError:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        raw_data = json.loads(response_text)

    validated = DiagnosisFields.from_llm_response(raw_data)
    return validated.model_dump()


async def diagnosis_fields_node(state: DiagnosisState) -> dict:
    """诊断流程 - 字段生成节点

    基于设备信息、检索文档和分析结果，生成报告所需的32个字段。
    利用 analysis_result 作为额外上下文，使字段生成更加精准。

    Args:
        state: 诊断状态，包含文档和分析结果

    Returns:
        更新后的状态，包含诊断数据
    """
    device_name = state["device_name"]
    documents = state["documents"]
    analysis_result = state.get("analysis_result", "")

    console.print("[dim]正在生成诊断字段数据...[/dim]")

    try:
        engine = get_engine()
        diagnosis_data = await generate_diagnosis_fields_enhanced(
            engine=engine,
            device_name=device_name,
            documents=documents,
            analysis_result=analysis_result,
        )

        console.print(f"[green]✓ 生成了 {len(diagnosis_data)} 个诊断字段[/green]")
        return {"diagnosis_data": diagnosis_data}

    except Exception as e:
        console.print(f"[red]✗ 字段生成失败: {e}[/red]")
        logger.error(f"字段生成失败: {e}", exc_info=True)
        return {"diagnosis_data": {}}


async def diagnosis_report_node(state: DiagnosisState) -> dict:
    """诊断流程 - 报告生成节点

    使用诊断字段数据，通过 LaTeX MCP 生成专业的 PDF 报告。

    Args:
        state: 诊断状态，包含诊断数据

    Returns:
        更新后的状态，包含报告路径

    Examples:
        >>> state = {"device_name": "变压器", "diagnosis_data": {...}, ...}
        >>> result = await diagnosis_report_node(state)
        >>> print(f"报告已生成: {result['report_path']}")
    """
    diagnosis_data = state["diagnosis_data"]

    if not diagnosis_data:
        error_msg = "诊断数据为空，无法生成报告"
        console.print(f"[red]✗ {error_msg}[/red]")
        return {"report_path": error_msg}

    console.print("[dim]正在生成 PDF 报告...[/dim]")

    try:
        # 调用 LaTeX MCP 生成报告
        result = await generate_diagnosis_report_async(diagnosis_data)

        if not result.get("success"):
            error_msg = result.get("error", "报告生成失败")
            console.print(f"[red]✗ {error_msg}[/red]")
            return {"report_path": error_msg}

        report_path = result.get("output_path", "")
        console.print(f"[green]✓ 报告生成成功: {report_path}[/green]")

        return {"report_path": report_path}

    except Exception as e:
        error_msg = f"报告生成失败: {e}"
        console.print(f"[red]✗ {error_msg}[/red]")
        logger.error(f"报告生成失败: {e}", exc_info=True)
        return {"report_path": error_msg}


async def generate_diagnosis_report_async(data: dict[str, Any]) -> dict[str, Any]:
    """异步生成诊断报告（在线程池中运行同步 MCP 调用）"""
    try:
        import concurrent.futures

        from rag_agent.mcp.latex_client import generate_diagnosis_report as gen_report

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, lambda: gen_report(data, template_id="device_diagnosis"))
            return result

    except Exception as e:
        logger.error(f"异步报告生成失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_diagnosis_flow():
        """测试诊断流程"""
        console.print("[bold cyan]测试 Diagnosis Agent[/bold cyan]\n")

        # 创建测试状态
        state: DiagnosisState = {
            "query": "变压器异常振动",
            "device_name": "",
            "documents": [],
            "diagnosis_data": {},
            "report_path": "",
            "analysis_result": "",
            "messages": [],
        }

        # 测试各个节点
        console.print("[yellow]步骤 1: 检索节点[/yellow]")
        result = await diagnosis_retrieval_node(state)
        state.update(result)
        console.print(f"文档数: {len(state['documents'])}\n")

        if state["documents"]:
            console.print("[yellow]步骤 2: 分析节点[/yellow]")
            result = await diagnosis_analysis_node(state)
            state.update(result)
            console.print(f"分析结果长度: {len(state['analysis_result'])}\n")

            console.print("[yellow]步骤 3: 字段生成节点[/yellow]")
            result = await diagnosis_fields_node(state)
            state.update(result)
            console.print(f"字段数: {len(state['diagnosis_data'])}\n")

            if state["diagnosis_data"]:
                console.print("[yellow]步骤 4: 报告生成节点[/yellow]")
                result = await diagnosis_report_node(state)
                state.update(result)
                console.print(f"报告路径: {state['report_path']}\n")

        console.print("[bold green]✓ 测试完成[/bold green]")

    # 运行测试
    asyncio.run(test_diagnosis_flow())
