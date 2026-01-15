"""Agent State Schema for LangGraph

定义智能体的状态结构，用于在节点之间传递数据。
"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """智能体状态定义

    包含查询、上下文、推理过程和输出等所有状态信息。
    用于在 LangGraph 的节点之间传递数据。
    """

    # ========== 用户输入 ==========
    query: str  # 用户查询内容
    intent: str  # 识别的意图: "diagnosis", "qa", "reasoning"

    # ========== 上下文信息 ==========
    context: str  # 检索到的上下文（格式化后的文本）
    documents: list[Document]  # 相关文档列表

    # ========== 推理过程 ==========
    reasoning_steps: list[str]  # 推理步骤记录
    tools_used: list[str]  # 使用的工具列表

    # ========== 输出结果 ==========
    answer: str  # 生成的答案
    diagnosis_data: dict  # 诊断数据（用于报告生成）
    report_path: str  # 生成的报告路径

    # ========== 对话历史 ==========
    messages: Annotated[list[BaseMessage], add_messages]  # 消息历史（自动累加）

    # ========== 元数据 ==========
    confidence: float  # 置信度 (0-1)
    need_clarification: bool  # 是否需要澄清用户意图


class DiagnosisState(TypedDict):
    """诊断智能体状态

    专门用于设备诊断流程的状态。
    """

    query: str  # 设备名称或问题描述
    device_name: str  # 设备名称
    documents: list[Document]  # 检索到的文档
    diagnosis_data: dict  # 诊断字段数据
    report_path: str  # 生成的报告路径
    analysis_result: str  # 分析结果
    messages: Annotated[list[BaseMessage], add_messages]


class QAState(TypedDict):
    """问答智能体状态

    专门用于问答流程的状态。
    """

    query: str  # 用户问题
    documents: list[Document]  # 检索到的文档
    context: str  # 上下文
    answer: str  # 生成的答案
    confidence: float  # 置信度
    sources: list[str]  # 答案来源
    messages: Annotated[list[BaseMessage], add_messages]
