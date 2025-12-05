"""应用模块

提供不同类型的 RAG 应用：
- QAApp: 设备知识问答
- ReportApp: 报告生成
"""

from rag_agent.apps.base import BaseApp
from rag_agent.apps.qa_app import QAApp
from rag_agent.apps.report_app import ReportApp

__all__ = ["BaseApp", "QAApp", "ReportApp"]
