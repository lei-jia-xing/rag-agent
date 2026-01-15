"""检索工具

提供基于向量数据库的文档检索功能。
"""


from langchain_core.tools import tool

from rag_agent.rag_engine import RAGEngine

# 全局 RAGEngine 实例（延迟初始化）
_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """获取或初始化 RAGEngine 实例"""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.initialize(load_only=True)
    return _engine


@tool
async def retrieve_device_info(query: str, k: int = 5) -> str:
    """检索设备相关文档和信息

    从向量数据库中检索与设备相关的技术文档、故障案例、
    维护记录等信息。

    Args:
        query: 查询内容，可以是设备名称、故障现象或技术问题
        k: 返回的文档数量，默认为 5

    Returns:
        检索到的文档内容，格式化为文本字符串

    Examples:
        >>> result = await retrieve_device_info.ainvoke("变压器温度过高", k=3)
        >>> print(result)
        根据技术规范，油浸式变压器的正常顶层油温...
    """
    engine = get_engine()

    # 执行检索
    documents = engine.retrieve(query, k=k)

    if not documents:
        return f"未找到与 '{query}' 相关的文档。"

    # 格式化文档内容
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        source = metadata.get("source", "未知来源")
        content = doc.page_content

        formatted_doc = f"[文档 {i}] 来源: {source}\n{content}"
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


@tool
async def retrieve_fault_cases(device_name: str, k: int = 3) -> str:
    """检索设备故障案例

    专门检索特定设备的故障案例和历史记录。

    Args:
        device_name: 设备名称，如 "变压器"、"断路器"
        k: 返回的案例数量，默认为 3

    Returns:
        相关故障案例的详细信息

    Examples:
        >>> result = await retrieve_fault_cases.ainvoke("变压器", k=5)
        >>> print(result)
        案例1: 变压器油温异常升高...
    """
    query = f"{device_name} 故障案例 异常 处理"
    return await retrieve_device_info.ainvoke(query, k)


@tool
async def retrieve_technical_standards(keyword: str, k: int = 3) -> str:
    """检索技术标准和规范

    检索相关的国家标准、行业规范和技术标准。

    Args:
        keyword: 关键词，如 "绝缘"、"接地"、"温度"
        k: 返回的标准数量，默认为 3

    Returns:
        相关技术标准的详细内容

    Examples:
        >>> result = await retrieve_technical_standards.ainvoke("变压器绝缘", k=3)
    """
    query = f"{keyword} 标准 规范 GB DL"
    return await retrieve_device_info.ainvoke(query, k)


@tool
def calculate_relevance_score(query: str, document: str) -> float:
    """计算文档与查询的相关性分数

    Args:
        query: 查询内容
        document: 文档内容

    Returns:
        相关性分数 (0-1)

    Examples:
        >>> score = calculate_relevance_score.invoke({
        ...     "query": "变压器温度",
        ...     "document": "变压器油温应控制在85℃以下"
        ... })
    """
    # 简单的关键词匹配计分
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())

    if not query_words:
        return 0.0

    intersection = query_words.intersection(doc_words)
    score = len(intersection) / len(query_words)

    return round(score, 3)
