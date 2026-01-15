"""Diagnosis Agent - 设备诊断智能体

诊断流程（10节点，优化并行）：
1. 检索 - 检索获取设备文档
2. 核心评估 - 健康评分、风险等级
3-6. 并行分析 - 故障分析、风险分析、设备信息、监测分析（asyncio.gather）
7. 维护建议生成 - 维护计划字段组
8. 一致性校验 - 校验并修正矛盾
9. 合并输出 - 整合所有字段
10. 报告生成 - LaTeX PDF
"""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from rich.console import Console

from rag_agent.config import config
from rag_agent.mcp.latex_client import generate_diagnosis_report_async
from rag_agent.prompts.diagnosis import (
    CORE_ASSESSMENT_PROMPT,
    DEVICE_INFO_PROMPT,
    FAULT_ANALYSIS_PROMPT,
    FEW_SHOT_EXAMPLE,
    MAINTENANCE_PROMPT,
    MONITORING_PROMPT,
    RISK_ANALYSIS_PROMPT,
    VALIDATION_PROMPT,
)
from rag_agent.rag_engine import RAGEngine
from rag_agent.retrieval import EnhancedRetriever
from rag_agent.schemas.diagnosis import DiagnosisFields
from rag_agent.schemas.state import DiagnosisState

console = Console()
logger = logging.getLogger(__name__)

_engine: RAGEngine | None = None
_enhanced_retriever: EnhancedRetriever | None = None
_json_parser: JsonOutputParser | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.initialize(load_only=True)
    return _engine


def get_enhanced_retriever() -> EnhancedRetriever:
    global _enhanced_retriever
    if _enhanced_retriever is None:
        engine = get_engine()
        _enhanced_retriever = EnhancedRetriever(
            engine=engine,
            enable_query_expansion=True,
            enable_hybrid_search=False,
            enable_reranking=False,
        )
    return _enhanced_retriever


def get_json_parser() -> JsonOutputParser:
    global _json_parser
    if _json_parser is None:
        _json_parser = JsonOutputParser()
    return _json_parser


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text


def _clean_control_chars(text: str) -> str:
    """Remove control characters that break JSON parsing."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)


def _parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse JSON from LLM response with multiple fallback strategies."""
    parser = get_json_parser()

    cleaned_text = _clean_control_chars(response_text)

    try:
        return parser.parse(cleaned_text)
    except Exception:
        pass

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    try:
        extracted = _extract_json_from_markdown(cleaned_text)
        extracted = _clean_control_chars(extracted)
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    try:
        return parser.parse(extracted)
    except Exception as e:
        logger.error(f"All JSON parsing strategies failed: {e}")
        raise ValueError(f"Failed to parse JSON response: {response_text[:200]}...") from e


async def _call_llm(engine: RAGEngine, system_prompt: str, human_message: str) -> dict[str, Any]:
    if engine.llm is None:
        raise RuntimeError("LLM 未初始化")

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]
    response = await engine.llm.ainvoke(messages)
    return _parse_json_response(str(response.content))


async def retrieval_node(state: DiagnosisState) -> dict:
    query = state["query"]
    device_name = state.get("device_name") or query

    console.print(f"[dim][1/10] 检索设备文档: {device_name}[/dim]")

    try:
        retriever = get_enhanced_retriever()
        documents = retriever.retrieve(
            query=device_name,
            top_k=config.RETRIEVAL_TOP_K,
            enable_query_expansion=True,
            enable_multi_query=True,
        )
        console.print(f"[green]✓ 检索到 {len(documents)} 个文档[/green]")
        return {"documents": documents, "device_name": device_name}

    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        engine = get_engine()
        documents = engine.retrieve(device_name, k=config.RETRIEVAL_TOP_K)
        return {"documents": documents, "device_name": device_name}


async def core_assessment_node(state: DiagnosisState) -> dict:
    device_name = state["device_name"]
    documents = state["documents"]

    if not documents:
        return {"core_assessment": {"health_score": 0, "health_status": "异常", "risk_level": "高", "issue_count": 0}}

    console.print("[dim][2/10] 核心健康评估...[/dim]")

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"设备名称：{device_name}\n\n设备文档：\n{context}\n\n{FEW_SHOT_EXAMPLE}"

    try:
        engine = get_engine()
        result = await _call_llm(engine, CORE_ASSESSMENT_PROMPT, human_msg)
        console.print(f"[green]✓ 健康评分: {result.get('health_score', 'N/A')}[/green]")
        return {"core_assessment": result}
    except Exception as e:
        logger.error(f"核心评估失败: {e}", exc_info=True)
        return {"core_assessment": {"health_score": 50, "health_status": "警告", "risk_level": "中", "issue_count": 0}}


async def fault_analysis_node(state: DiagnosisState) -> dict:
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})

    console.print("[dim][3/10] 故障分析...[/dim]")

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}

核心评估结果：
- 健康评分：{core.get("health_score", "N/A")}
- 健康状态：{core.get("health_status", "N/A")}
- 风险等级：{core.get("risk_level", "N/A")}
- 评估依据：{core.get("assessment_reasoning", "N/A")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        result = await _call_llm(engine, FAULT_ANALYSIS_PROMPT, human_msg)
        console.print("[green]✓ 故障分析完成[/green]")
        return {"fault_analysis": result}
    except Exception as e:
        logger.error(f"故障分析失败: {e}", exc_info=True)
        return {"fault_analysis": {}}


async def risk_analysis_node(state: DiagnosisState) -> dict:
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})
    fault = state.get("fault_analysis", {})

    console.print("[dim][4/10] 风险分析...[/dim]")

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}

核心评估：健康评分 {core.get("health_score", "N/A")}，{core.get("health_status", "N/A")}

故障分析：
{fault.get("fault_description", "无")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        result = await _call_llm(engine, RISK_ANALYSIS_PROMPT, human_msg)
        console.print("[green]✓ 风险分析完成[/green]")
        return {"risk_analysis": result}
    except Exception as e:
        logger.error(f"风险分析失败: {e}", exc_info=True)
        return {"risk_analysis": {}}


async def _fault_analysis_task(state: DiagnosisState) -> dict:
    """Internal task for parallel execution - fault analysis."""
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}

核心评估结果：
- 健康评分：{core.get("health_score", "N/A")}
- 健康状态：{core.get("health_status", "N/A")}
- 风险等级：{core.get("risk_level", "N/A")}
- 评估依据：{core.get("assessment_reasoning", "N/A")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        return await _call_llm(engine, FAULT_ANALYSIS_PROMPT, human_msg)
    except Exception as e:
        logger.error(f"故障分析失败: {e}", exc_info=True)
        return {}


async def _risk_analysis_task(state: DiagnosisState) -> dict:
    """Internal task for parallel execution - risk analysis."""
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}

核心评估：健康评分 {core.get("health_score", "N/A")}，{core.get("health_status", "N/A")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        return await _call_llm(engine, RISK_ANALYSIS_PROMPT, human_msg)
    except Exception as e:
        logger.error(f"风险分析失败: {e}", exc_info=True)
        return {}


async def _device_info_task(state: DiagnosisState) -> dict:
    """Internal task for parallel execution - device info."""
    device_name = state["device_name"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"设备名称：{device_name}\n\n设备文档：\n{context}"

    try:
        engine = get_engine()
        return await _call_llm(engine, DEVICE_INFO_PROMPT, human_msg)
    except Exception as e:
        logger.error(f"设备信息生成失败: {e}", exc_info=True)
        return {}


async def _monitoring_task(state: DiagnosisState) -> dict:
    """Internal task for parallel execution - monitoring analysis."""
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}
健康评分：{core.get("health_score", "N/A")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        return await _call_llm(engine, MONITORING_PROMPT, human_msg)
    except Exception as e:
        logger.error(f"监测分析失败: {e}", exc_info=True)
        return {}


async def parallel_analysis_node(state: DiagnosisState) -> dict:
    """并行执行 4 个分析任务：故障分析、风险分析、设备信息、监测分析"""
    console.print("[dim][3-6/10] 并行执行分析任务...[/dim]")

    fault_task = asyncio.create_task(_fault_analysis_task(state))
    risk_task = asyncio.create_task(_risk_analysis_task(state))
    device_info_task = asyncio.create_task(_device_info_task(state))
    monitoring_task = asyncio.create_task(_monitoring_task(state))

    fault_result, risk_result, device_info_result, monitoring_result = await asyncio.gather(
        fault_task, risk_task, device_info_task, monitoring_task
    )

    console.print("[green]✓ 并行分析完成[/green]")

    return {
        "fault_analysis": fault_result,
        "risk_analysis": risk_result,
        "device_info_fields": device_info_result,
        "monitoring_fields": monitoring_result,
    }


async def device_info_node(state: DiagnosisState) -> dict:
    device_name = state["device_name"]
    documents = state["documents"]

    console.print("[dim][5/10] 生成设备信息...[/dim]")

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"设备名称：{device_name}\n\n设备文档：\n{context}"

    try:
        engine = get_engine()
        result = await _call_llm(engine, DEVICE_INFO_PROMPT, human_msg)
        console.print("[green]✓ 设备信息生成完成[/green]")
        return {"device_info_fields": result}
    except Exception as e:
        logger.error(f"设备信息生成失败: {e}", exc_info=True)
        return {"device_info_fields": {}}


async def monitoring_node(state: DiagnosisState) -> dict:
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})

    console.print("[dim][6/10] 生成监测分析...[/dim]")

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}
健康评分：{core.get("health_score", "N/A")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        result = await _call_llm(engine, MONITORING_PROMPT, human_msg)
        console.print("[green]✓ 监测分析完成[/green]")
        return {"monitoring_fields": result}
    except Exception as e:
        logger.error(f"监测分析失败: {e}", exc_info=True)
        return {"monitoring_fields": {}}


async def maintenance_node(state: DiagnosisState) -> dict:
    device_name = state["device_name"]
    documents = state["documents"]
    core = state.get("core_assessment", {})
    fault = state.get("fault_analysis", {})
    risk = state.get("risk_analysis", {})

    console.print("[dim][7/10] 生成维护建议...[/dim]")

    context = "\n\n".join([doc.page_content for doc in documents])
    human_msg = f"""设备名称：{device_name}

核心评估：
- 健康评分：{core.get("health_score", "N/A")}
- 健康状态：{core.get("health_status", "N/A")}
- 风险等级：{core.get("risk_level", "N/A")}

故障分析：
- 故障描述：{fault.get("fault_description", "无")}
- 故障原因：{fault.get("fault_cause_analysis", "无")}

风险分析：
- 当前风险：{risk.get("current_risks", "无")}
- 潜在风险：{risk.get("potential_risks", "无")}

设备文档：
{context}"""

    try:
        engine = get_engine()
        result = await _call_llm(engine, MAINTENANCE_PROMPT, human_msg)
        console.print("[green]✓ 维护建议生成完成[/green]")
        return {"maintenance_fields": result}
    except Exception as e:
        logger.error(f"维护建议生成失败: {e}", exc_info=True)
        return {"maintenance_fields": {}}


async def validation_node(state: DiagnosisState) -> dict:
    console.print("[dim][8/10] 一致性校验...[/dim]")

    core = state.get("core_assessment", {})
    fault = state.get("fault_analysis", {})
    risk = state.get("risk_analysis", {})
    device_info = state.get("device_info_fields", {})
    monitoring = state.get("monitoring_fields", {})
    maintenance = state.get("maintenance_fields", {})

    all_data = {**core, **fault, **risk, **device_info, **monitoring, **maintenance}

    human_msg = f"请检查以下诊断数据的一致性：\n\n{json.dumps(all_data, ensure_ascii=False, indent=2)}"

    try:
        engine = get_engine()
        result = await _call_llm(engine, VALIDATION_PROMPT, human_msg)

        issues = result.get("issues", [])
        corrections = result.get("corrections", {})

        if corrections:
            all_data.update(corrections)
            console.print(f"[yellow]修正了 {len(corrections)} 个字段[/yellow]")

        console.print(f"[green]✓ 校验完成，发现 {len(issues)} 个问题[/green]")
        return {"validation_issues": [i.get("problem", "") for i in issues]}

    except Exception as e:
        logger.error(f"校验失败: {e}", exc_info=True)
        return {"validation_issues": []}


async def merge_fields_node(state: DiagnosisState) -> dict:
    console.print("[dim][9/10] 合并诊断字段...[/dim]")

    device_name = state.get("device_name", "未知设备")
    core = state.get("core_assessment", {})
    fault = state.get("fault_analysis", {})
    risk = state.get("risk_analysis", {})
    device_info = state.get("device_info_fields", {})
    monitoring = state.get("monitoring_fields", {})
    maintenance = state.get("maintenance_fields", {})

    merged = {
        "device_name": device_name,  # Ensure device_name is always present
        **device_info,
        **core,
        **monitoring,
        **fault,
        **risk,
        **maintenance,
    }

    # Ensure device_name is not overwritten by empty value from device_info
    if not merged.get("device_name") or merged.get("device_name") == "":
        merged["device_name"] = device_name

    merged.pop("assessment_reasoning", None)

    try:
        validated = DiagnosisFields.from_llm_response(merged)
        final_data = validated.model_dump()
        console.print(f"[green]✓ 合并完成，共 {len(final_data)} 个字段[/green]")
        return {"diagnosis_data": final_data}
    except Exception as e:
        logger.error(f"字段验证失败: {e}", exc_info=True)
        return {"diagnosis_data": merged}


async def report_node(state: DiagnosisState) -> dict:
    diagnosis_data = state.get("diagnosis_data", {})

    if not diagnosis_data:
        return {"report_path": "诊断数据为空，无法生成报告"}

    console.print("[dim][10/10] 生成 PDF 报告...[/dim]")

    try:
        result = await generate_diagnosis_report_async(diagnosis_data)

        if not result.get("success"):
            return {"report_path": f"报告生成失败: {result.get('error', '未知错误')}"}

        report_path = result.get("output_path", "")
        console.print(f"[green]✓ 报告生成成功: {report_path}[/green]")
        return {"report_path": report_path}

    except Exception as e:
        logger.error(f"报告生成失败: {e}", exc_info=True)
        return {"report_path": f"报告生成失败: {e}"}
