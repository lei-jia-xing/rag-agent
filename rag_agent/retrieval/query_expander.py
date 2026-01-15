"""Query Expander - 查询扩展和改写

增强查询质量的技术：
1. 查询重写（Query Rewriting）- 优化查询表述
2. 多查询生成（Multi-Query Generation）- 生成多个不同角度的查询
3. HyDE（Hypothetical Document Embeddings）- 假设性文档嵌入
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from rag_agent.rag_engine import RAGEngine

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


class QueryExpander:
    """查询扩展器

    提供多种查询增强技术：
    - 查询重写：改善查询表述
    - 多查询生成：从不同角度生成查询
    - HyDE：生成假设性答案用于检索
    """

    def __init__(self, engine: RAGEngine | None = None):
        """初始化查询扩展器

        Args:
            engine: RAG引擎实例，如果为None则使用默认实例
        """
        self.engine = engine or get_engine()
        self.llm = self.engine.llm

    async def rewrite_query(self, query: str) -> str:
        """重写查询，优化表述

        将用户的原始查询重写为更清晰、更具体的表述。

        Args:
            query: 原始查询

        Returns:
            重写后的查询

        Examples:
            >>> expander = QueryExpander()
            >>> rewritten = await expander.rewrite_query("变压器温度咋样")
            >>> print(rewritten)
            变压器的正常工作温度范围是多少
        """
        if self.llm is None:
            # 如果没有LLM，返回原查询
            return query

        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的查询优化助手。"
                    "将用户的查询重写为更清晰、更具体、更适合检索的表述。\n\n"
                    "要求：\n"
                    "1. 保持原意不变\n"
                    "2. 使用专业术语\n"
                    "3. 补充省略的上下文\n"
                    "4. 使查询更完整\n"
                    "5. 只返回重写后的查询，不要解释",
                ),
                (
                    "human",
                    "原始查询：{query}\n\n"
                    "重写后的查询：",
                ),
            ]
        )

        try:
            chain = rewrite_prompt | self.llm
            response = await chain.ainvoke({"query": query})
            rewritten = response.content.strip()

            logger.info(f"查询重写: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"查询重写失败: {e}", exc_info=True)
            return query

    async def generate_multiple_queries(
        self,
        query: str,
        num_queries: int = 3,
    ) -> list[str]:
        """生成多个不同角度的查询

        从不同角度、使用不同的表述方式生成多个查询，
        用于提升检索召回率。

        Args:
            query: 原始查询
            num_queries: 生成的查询数量

        Returns:
            查询列表（包括原始查询）

        Examples:
            >>> expander = QueryExpander()
            >>> queries = await expander.generate_multiple_queries(
            ...     "变压器温度过高",
            ...     num_queries=3
            ... )
            >>> print(queries)
            ['变压器温度过高的原因',
             '变压器运行温度异常',
             '变压器过热故障分析']
        """
        if self.llm is None:
            return [query]

        multi_query_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的查询扩展助手。"
                    "基于用户的原始查询，生成{num_queries}个不同角度的查询变体。\n\n"
                    "要求：\n"
                    "1. 每个查询从不同角度阐述问题\n"
                    "2. 使用同义词和相关概念\n"
                    "3. 包含不同的表述方式（问题描述、原因分析、解决方案等）\n"
                    "4. 每行一个查询\n"
                    "5. 只返回查询列表，不要编号",
                ),
                (
                    "human",
                    "原始查询：{query}\n\n"
                    "生成{num_queries}个查询变体：",
                ),
            ]
        )

        try:
            chain = multi_query_prompt | self.llm
            response = await chain.ainvoke(
                {"query": query, "num_queries": num_queries}
            )

            # 解析生成的查询
            queries_text = response.content.strip()
            generated_queries = [
                line.strip() for line in queries_text.split("\n") if line.strip()
            ]

            # 确保至少返回原始查询
            all_queries = [query] + generated_queries
            unique_queries = list(dict.fromkeys(all_queries))  # 去重并保持顺序

            logger.info(
                f"多查询生成: 原始='{query}', 生成{len(unique_queries)}个查询"
            )
            return unique_queries[:num_queries]

        except Exception as e:
            logger.error(f"多查询生成失败: {e}", exc_info=True)
            return [query]

    async def generate_hypothetical_document(
        self,
        query: str,
    ) -> str:
        """生成假设性文档（HyDE）

        生成一个假设性的答案文档，然后使用这个文档的嵌入向量进行检索。
        这样可以找到与"理想答案"相似的文档，而不是与查询相似的文档。

        Args:
            query: 用户查询

        Returns:
            假设性的答案文档

        Examples:
            >>> expander = QueryExpander()
            >>> hypothetical = await expander.generate_hypothetical_document(
            ...     "变压器的正常温度范围"
            ... )
            >>> print(hypothetical[:100])
            变压器的正常运行温度一般在60-85℃之间。根据国家标准GB/T...
        """
        if self.llm is None:
            return query

        hyde_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的电气工程知识专家。"
                    "基于用户查询，生成一个详细的、准确的假设性答案。\n\n"
                    "要求：\n"
                    "1. 假设你找到了相关的技术文档\n"
                    "2. 基于专业知识生成详细的答案\n"
                    "3. 包含具体的数据、标准、参数\n"
                    "4. 使用专业术语\n"
                    "5. 长度在200-400字之间\n"
                    "6. 不要编造不确定的信息\n"
                    "7. 直接生成答案，不要说明这是假设",
                ),
                (
                    "human",
                    "问题：{query}\n\n"
                    "请生成一个详细的答案：",
                ),
            ]
        )

        try:
            chain = hyde_prompt | self.llm
            response = await chain.ainvoke({"query": query})
            hypothetical_doc = response.content.strip()

            logger.info(
                f"HyDE生成: 查询='{query}', 文档长度={len(hypothetical_doc)}"
            )
            return hypothetical_doc

        except Exception as e:
            logger.error(f"HyDE生成失败: {e}", exc_info=True)
            return query

    async def expand_query_comprehensive(
        self,
        query: str,
        enable_rewrite: bool = True,
        enable_multi_query: bool = True,
        enable_hyde: bool = False,
        num_queries: int = 3,
    ) -> dict[str, Any]:
        """综合查询扩展

        应用多种查询增强技术。

        Args:
            query: 原始查询
            enable_rewrite: 是否启用查询重写
            enable_multi_query: 是否启用多查询生成
            enable_hyde: 是否启用HyDE
            num_queries: 多查询生成数量

        Returns:
            扩展结果字典

        Examples:
            >>> expander = QueryExpander()
            >>> result = await expander.expand_query_comprehensive(
            ...     "变压器温度高",
            ...     enable_multi_query=True
            ... )
            >>> print(result["rewritten_query"])
            >>> print(result["multiple_queries"])
        """
        result = {
            "original_query": query,
            "rewritten_query": query,
            "multiple_queries": [query],
            "hypothetical_document": "",
        }

        # 1. 查询重写
        if enable_rewrite:
            result["rewritten_query"] = await self.rewrite_query(query)

        # 2. 多查询生成（基于重写后的查询）
        if enable_multi_query:
            result["multiple_queries"] = await self.generate_multiple_queries(
                result["rewritten_query"],
                num_queries=num_queries,
            )

        # 3. HyDE生成
        if enable_hyde:
            result["hypothetical_document"] = await self.generate_hypothetical_document(
                query
            )

        return result


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_query_expander():
        """测试查询扩展器"""
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        console.print("[bold cyan]测试 Query Expander[/bold cyan]\n")

        expander = QueryExpander()

        # 测试查询
        test_query = "变压器温度过高怎么办"

        console.print(f"[yellow]原始查询:[/yellow] {test_query}\n")

        # 1. 测试查询重写
        console.print("[yellow]1. 查询重写[/yellow]")
        rewritten = await expander.rewrite_query(test_query)
        console.print(f"重写后: {rewritten}\n")

        # 2. 测试多查询生成
        console.print("[yellow]2. 多查询生成[/yellow]")
        multi_queries = await expander.generate_multiple_queries(test_query, num_queries=3)
        for i, q in enumerate(multi_queries, 1):
            console.print(f"  {i}. {q}")
        console.print()

        # 3. 测试HyDE
        console.print("[yellow]3. HyDE生成[/yellow]")
        hypothetical = await expander.generate_hypothetical_document(test_query)
        console.print(Panel(hypothetical, title="假设性文档", height=10))
        console.print()

        # 4. 综合扩展
        console.print("[yellow]4. 综合查询扩展[/yellow]")
        result = await expander.expand_query_comprehensive(
            test_query,
            enable_rewrite=True,
            enable_multi_query=True,
            enable_hyde=True,
            num_queries=3,
        )

        console.print(f"原始查询: {result['original_query']}")
        console.print(f"重写查询: {result['rewritten_query']}")
        console.print(f"查询数量: {len(result['multiple_queries'])}")
        console.print(f"HyDE文档长度: {len(result['hypothetical_document'])} 字符")

        console.print("\n[bold green]✓ 测试完成[/bold green]")

    # 运行测试
    asyncio.run(test_query_expander())
