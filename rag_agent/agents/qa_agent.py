"""QA Agent - 问答智能体

负责基于知识库的智能问答：
1. 检索相关知识（使用增强检索）
2. 合成答案
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from rag_agent.rag_engine import RAGEngine
from rag_agent.retrieval import EnhancedRetriever
from rag_agent.schemas.state import QAState

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


async def qa_retrieval_node(state: QAState) -> dict:
    """问答流程 - 检索节点（使用增强检索）

    根据用户问题，使用增强检索技术检索相关的知识文档。

    增强功能：
    - 查询重写：优化查询表述
    - 多查询生成：从不同角度检索
    - HyDE：生成假设性答案辅助检索

    Args:
        state: QA 状态，包含用户问题

    Returns:
        更新后的状态，包含检索到的文档和上下文

    Examples:
        >>> state = {"query": "变压器的正常温度范围是多少？", ...}
        >>> result = await qa_retrieval_node(state)
        >>> print(f"检索到 {len(result['documents'])} 个文档")
    """
    query = state["query"]

    console.print(f"[dim]正在使用增强检索: {query}[/dim]")

    try:
        retriever = get_enhanced_retriever()

        # 使用增强检索
        documents = retriever.retrieve(
            query=query,
            top_k=5,
            enable_query_expansion=True,
            enable_multi_query=True,  # 多查询生成
            enable_hyde=False,  # HyDE（可选启用）
        )

        # 格式化上下文
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get("source", "未知来源")

            # 添加重排序分数（如果有）
            score_info = ""
            if "rerank_score" in metadata:
                score_info = f" [相关度: {metadata['rerank_score']:.2f}]"

            context_parts.append(f"[来源 {i}: {source}]{score_info}\n{content}")

        context = "\n\n".join(context_parts)

        console.print(f"[green]✓ 检索到 {len(documents)} 个相关文档（增强检索）[/green]")

        return {
            "documents": documents,
            "context": context,
        }

    except Exception as e:
        console.print(f"[red]✗ 增强检索失败: {e}[/red]")
        logger.error(f"增强检索失败: {e}", exc_info=True)

        # 降级到基础检索
        console.print("[yellow]降级到基础检索...[/yellow]")
        try:
            engine = get_engine()
            documents = engine.retrieve(query, k=5)

            context_parts = []
            for i, doc in enumerate(documents, 1):
                content = doc.page_content
                metadata = doc.metadata
                source = metadata.get("source", "未知来源")
                context_parts.append(f"[来源 {i}: {source}]\n{content}")

            context = "\n\n".join(context_parts)

            console.print(f"[yellow]✓ 基础检索返回 {len(documents)} 个文档[/yellow]")

            return {
                "documents": documents,
                "context": context,
            }
        except Exception as e2:
            console.print(f"[red]✗ 基础检索也失败: {e2}[/red]")
            return {
                "documents": [],
                "context": "",
            }


async def qa_synthesis_node(state: QAState) -> dict:
    """问答流程 - 答案合成节点

    基于检索到的上下文，使用 LLM 生成准确、全面的答案。

    Args:
        state: QA 状态，包含问题和上下文

    Returns:
        更新后的状态，包含生成的答案和置信度

    Examples:
        >>> state = {"query": "...", "context": "...", ...}
        >>> result = await qa_synthesis_node(state)
        >>> print(result["answer"])
    """
    query = state["query"]
    context = state["context"]

    if not context:
        error_msg = "未找到相关上下文，无法生成答案"
        console.print(f"[red]✗ {error_msg}[/red]")
        return {
            "answer": error_msg,
            "confidence": 0.0,
            "sources": [],
        }

    console.print("[dim]正在生成答案...[/dim]")

    try:
        engine = get_engine()
        llm = engine.llm

        if llm is None:
            raise RuntimeError("LLM 未初始化")

        # 构建问答提示
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的电气工程设备知识助手。"
                    "基于提供的上下文信息，准确回答用户的问题。\n\n"
                    "要求：\n"
                    "1. 只使用提供的上下文信息，不要编造答案\n"
                    "2. 如果上下文中没有相关信息，明确说明\n"
                    "3. 答案要准确、清晰、专业\n"
                    "4. 必要时引用具体的来源\n"
                    "5. 如果涉及数值、标准等，要确保准确",
                ),
                (
                    "human",
                    "问题：{query}\n\n参考信息：\n{context}\n\n请基于以上信息回答问题。",
                ),
            ]
        )

        # 执行问答
        chain = qa_prompt | llm
        response = await chain.ainvoke({"query": query, "context": context})

        answer = str(response.content) if response.content else ""

        confidence = calculate_confidence(query, answer, context)

        # 提取来源
        sources = extract_sources(context)

        console.print("[green]✓ 答案生成完成[/green]")

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        }

    except Exception as e:
        error_msg = f"答案生成失败: {e}"
        console.print(f"[red]✗ {error_msg}[/red]")
        logger.error(f"答案生成失败: {e}", exc_info=True)
        return {
            "answer": error_msg,
            "confidence": 0.0,
            "sources": [],
        }


def calculate_confidence(query: str, answer: str, context: str) -> float:
    """计算答案的置信度

    基于简单的启发式方法：
    - 答案长度是否合理
    - 是否包含明确的答案
    - 是否包含不确定性标记

    Args:
        query: 用户问题
        answer: 生成的答案
        context: 上下文

    Returns:
        置信度分数 (0-1)
    """
    confidence = 0.5  # 基础置信度

    # 答案长度合理（50-500字）
    if 50 <= len(answer) <= 500:
        confidence += 0.1
    elif len(answer) < 20:
        confidence -= 0.2

    # 包含明确的答案指示词
    positive_indicators = ["根据", "按照", "标准规定", "技术规范", "应该"]
    negative_indicators = ["不确定", "无法判断", "信息不足", "没有提供"]

    for indicator in positive_indicators:
        if indicator in answer:
            confidence += 0.05

    for indicator in negative_indicators:
        if indicator in answer:
            confidence -= 0.1

    # 答案是否相关（包含问题关键词）
    query_words = set(query.split())
    answer_words = set(answer.split())
    overlap = len(query_words.intersection(answer_words))

    if overlap > 0:
        confidence += min(0.2, overlap * 0.05)

    # 确保在合理范围内
    return max(0.0, min(1.0, confidence))


def extract_sources(context: str) -> list[str]:
    """从上下文中提取来源

    Args:
        context: 格式化的上下文文本

    Returns:
        来源列表
    """
    sources = []
    lines = context.split("\n")

    for line in lines:
        if line.startswith("[来源 ") and ":" in line:
            # 提取来源名称
            try:
                source = line.split(":")[1].split("]")[0].strip()
                if source and source not in sources:
                    sources.append(source)
            except (IndexError, ValueError):
                continue

    return sources


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_qa_flow():
        """测试问答流程"""
        console.print("[bold cyan]测试 QA Agent[/bold cyan]\n")

        # 创建测试状态
        state: QAState = {
            "query": "变压器的正常温度范围是多少？",
            "documents": [],
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "sources": [],
            "messages": [],
        }

        # 测试各个节点
        console.print("[yellow]步骤 1: 检索节点[/yellow]")
        result = await qa_retrieval_node(state)
        state.update(result)
        console.print(f"文档数: {len(state['documents'])}")
        console.print(f"上下文长度: {len(state['context'])} 字符\n")

        if state["documents"]:
            console.print("[yellow]步骤 2: 答案合成节点[/yellow]")
            result = await qa_synthesis_node(state)
            state.update(result)

            console.print(f"答案长度: {len(state['answer'])} 字符")
            console.print(f"置信度: {state['confidence']:.2f}")
            console.print(f"来源数: {len(state['sources'])}")
            console.print(f"\n答案预览:\n{state['answer'][:200]}...\n")

        console.print("[bold green]✓ 测试完成[/bold green]")

    # 运行测试
    asyncio.run(test_qa_flow())
