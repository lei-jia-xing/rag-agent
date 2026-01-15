"""Hybrid Retriever - 混合检索器

结合向量检索（稠密）和BM25检索（稀疏）的优势，
使用RRF（Reciprocal Rank Fusion）融合结果。
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_agent.rag_engine import RAGEngine
from rag_agent.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器

    结合向量检索和BM25检索，使用RRF算法融合结果。

    优势：
    - 向量检索：擅长语义相似度匹配
    - BM25检索：擅长关键词精确匹配
    - RRF融合：平衡两种检索结果

    RRF公式:
    score(d) = Σ k / (k + rank(d))

    其中:
    - d: 文档
    - rank(d): 文档在某个检索器中的排名
    - k: 平滑参数（默认60）
    """

    engine: RAGEngine
    bm25_retriever: BM25Retriever
    alpha: float = 0.5  # 向量检索权重（0-1）
    top_k: int = 5
    rrf_k: int = 60  # RRF参数

    class Config:
        """Pydantic配置"""
        arbitrary_types_allowed = True

    def __init__(
        self,
        engine: RAGEngine,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        """初始化混合检索器

        Args:
            engine: RAG引擎（包含向量检索器）
            bm25_retriever: BM25检索器
            alpha: 向量检索权重（0=仅BM25，1=仅向量，0.5=均衡）
            top_k: 返回的top-k文档数量
            rrf_k: RRF平滑参数

        Examples:
            >>> engine = RAGEngine()
            >>> engine.initialize(load_only=True)
            >>> bm25 = BM25Retriever(documents)
            >>> hybrid = HybridRetriever(engine, bm25, alpha=0.5)
            >>> results = hybrid.invoke("变压器温度")
        """
        super().__init__(
            engine=engine,
            bm25_retriever=bm25_retriever,
            alpha=alpha,
            top_k=top_k,
            rrf_k=rrf_k,
        )

        if engine.vectorstore is None:
            raise ValueError("RAGEngine的vectorstore未初始化")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """混合检索

        Args:
            query: 查询文本
            run_manager: 运行管理器（可选）

        Returns:
            融合后的文档列表
        """
        # 1. 向量检索
        if self.engine.vectorstore is None:
            vector_docs = []
            vector_ranks = {}
        else:
            vector_docs = self.engine.vectorstore.similarity_search(query, k=self.top_k * 2)
            # 创建文档到排名的映射
            vector_ranks = {
                self._get_doc_key(doc): i
                for i, doc in enumerate(vector_docs)
            }

        # 2. BM25检索
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_ranks = {
            self._get_doc_key(doc): i
            for i, doc in enumerate(bm25_docs)
        }

        # 3. 融合结果（RRF）
        fused_scores = self._reciprocal_rank_fusion(
            vector_ranks,
            bm25_ranks,
            alpha=self.alpha,
            k=self.rrf_k,
        )

        # 4. 排序并返回top-k
        sorted_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = [
            doc for doc, score in sorted_docs[: self.top_k]
        ]

        logger.info(
            f"混合检索: query='{query}', "
            f"向量={len(vector_docs)}篇, "
            f"BM25={len(bm25_docs)}篇, "
            f"融合后={len(top_docs)}篇"
        )

        return top_docs

    def _get_doc_key(self, doc: Document) -> str:
        """生成文档唯一标识

        Args:
            doc: 文档

        Returns:
            文档唯一标识（使用内容和元数据哈希）
        """
        import hashlib

        content = doc.page_content
        metadata_str = str(sorted(doc.metadata.items()))
        combined = f"{content}|{metadata_str}"

        return hashlib.md5(combined.encode()).hexdigest()

    def _reciprocal_rank_fusion(
        self,
        ranks1: dict[str, int],
        ranks2: dict[str, int],
        alpha: float = 0.5,
        k: int = 60,
    ) -> dict[str, float]:
        """倒数排名融合（Reciprocal Rank Fusion）

        结合多个检索器的排名结果。

        Args:
            ranks1: 第一个检索器的排名（文档→排名）
            ranks2: 第二个检索器的排名
            alpha: 第一个检索器的权重
            k: 平滑参数

        Returns:
            文档→融合分数的映射
        """
        # 收集所有文档
        all_docs = set(ranks1.keys()) | set(ranks2.keys())

        fused_scores = {}

        for doc in all_docs:
            score = 0.0

            # 第一个检索器的贡献
            if doc in ranks1:
                score += alpha * (1 / (k + ranks1[doc]))

            # 第二个检索器的贡献
            if doc in ranks2:
                score += (1 - alpha) * (1 / (k + ranks2[doc]))

            fused_scores[doc] = score

        return fused_scores


# 测试代码
if __name__ == "__main__":
    def test_hybrid_retriever():
        """测试混合检索器"""
        from rich.console import Console

        console = Console()

        console.print("[bold cyan]测试 Hybrid Retriever[/bold cyan]\n")

        # 注意：这需要预先构建的向量数据库
        console.print("[yellow]初始化RAG引擎...[/yellow]")
        try:
            engine = RAGEngine()
            engine.initialize(load_only=True)
            console.print("[green]✓ RAG引擎初始化成功[/green]\n")

            # 获取文档用于BM25
            if engine.vectorstore is not None:
                # 从向量库中获取一些文档
                dummy_query = "变压器"
                docs = engine.vectorstore.similarity_search(dummy_query, k=10)

                console.print(f"[yellow]创建BM25检索器（{len(docs)}个文档）...[/yellow]")
                bm25_retriever = BM25Retriever(docs, top_k=5)
                console.print("[green]✓ BM25检索器创建成功[/green]\n")

                # 创建混合检索器
                console.print("[yellow]创建混合检索器...[/yellow]")
                hybrid_retriever = HybridRetriever(
                    engine=engine,
                    bm25_retriever=bm25_retriever,
                    alpha=0.5,
                    top_k=5,
                )
                console.print("[green]✓ 混合检索器创建成功\n[/green]")

                # 测试查询
                test_queries = [
                    "变压器温度范围",
                    "设备故障诊断",
                ]

                for query in test_queries:
                    console.print(f"[yellow]查询:[/yellow] {query}")

                    # 纯向量检索
                    console.print("[dim]  向量检索:[/dim]")
                    vector_results = engine.vectorstore.similarity_search(query, k=3)
                    for i, doc in enumerate(vector_results, 1):
                        preview = doc.page_content[:40] + "..."
                        console.print(f"    {i}. {preview}")

                    # 纯BM25检索
                    console.print("[dim]  BM25检索:[/dim]")
                    bm25_results = bm25_retriever.invoke(query)[:3]
                    for i, doc in enumerate(bm25_results, 1):
                        preview = doc.page_content[:40] + "..."
                        console.print(f"    {i}. {preview}")

                    # 混合检索
                    console.print("[dim]  混合检索:[/dim]")
                    hybrid_results = hybrid_retriever.invoke(query)
                    for i, doc in enumerate(hybrid_results, 1):
                        preview = doc.page_content[:40] + "..."
                        console.print(f"    {i}. {preview}")

                    console.print()

                console.print("[bold green]✓ 测试完成[/bold green]")

            else:
                console.print("[red]✗ 向量数据库未初始化[/red]")

        except Exception as e:
            console.print(f"[red]✗ 测试失败: {e}[/red]")
            import traceback

            traceback.print_exc()

    # 运行测试
    test_hybrid_retriever()
