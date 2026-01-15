"""Base Enhanced Retriever - 基础增强检索器

整合所有检索增强功能：
- 查询扩展（重写、多查询、HyDE）
- 混合检索（向量 + BM25）
- 重排序
- 元数据过滤
"""

import logging

from langchain_core.documents import Document

from rag_agent.rag_engine import RAGEngine
from rag_agent.retrieval.bm25_retriever import BM25Retriever
from rag_agent.retrieval.hybrid_retriever import HybridRetriever
from rag_agent.retrieval.query_expander import QueryExpander
from rag_agent.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """增强检索器

    整合所有检索增强技术，提供一站式检索解决方案。

    功能：
    1. 查询扩展
    2. 混合检索（向量 + BM25）
    3. 重排序
    4. 元数据过滤

    Examples:
        >>> retriever = EnhancedRetriever(engine)
        >>> results = retriever.retrieve(
        ...     "变压器温度",
        ...     enable_query_expansion=True,
        ...     enable_reranking=True,
        ...     top_k=5
        ... )
    """

    def __init__(
        self,
        engine: RAGEngine,
        enable_query_expansion: bool = True,
        enable_hybrid_search: bool = True,
        enable_reranking: bool = False,
        reranker_type: str = "bge",
        reranker_config: dict | None = None,
    ):
        """初始化增强检索器

        Args:
            engine: RAG引擎
            enable_query_expansion: 是否启用查询扩展
            enable_hybrid_search: 是否启用混合检索
            enable_reranking: 是否启用重排序
            reranker_type: 重排序器类型（"cohere"或"bge"）
            reranker_config: 重排序器配置
        """
        self.engine = engine
        self.enable_query_expansion = enable_query_expansion
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking

        # 初始化组件
        self.query_expander = QueryExpander(engine) if enable_query_expansion else None

        self.bm25_retriever = None
        self.hybrid_retriever = None

        if enable_hybrid_search:
            # 需要文档来构建BM25索引
            # 这里先设置为None，在首次使用时构建
            self._bm25_built = False

        self.reranker = None
        if enable_reranking:
            reranker_config = reranker_config or {}
            self.reranker = Reranker(type=reranker_type, **reranker_config)

    def _ensure_bm25_built(self, documents: list[Document]) -> None:
        """确保BM25索引已构建"""
        if self._bm25_built:
            return

        logger.info("构建BM25索引...")
        self.bm25_retriever = BM25Retriever(documents)
        self.hybrid_retriever = HybridRetriever(
            self.engine,
            self.bm25_retriever,
            alpha=0.5,
            top_k=10,
        )
        self._bm25_built = True
        logger.info("BM25索引构建完成")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_query_expansion: bool | None = None,
        enable_hybrid_search: bool | None = None,
        enable_reranking: bool | None = None,
        enable_multi_query: bool = True,
        enable_hyde: bool = False,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        """增强检索

        Args:
            query: 查询文本
            top_k: 返回的文档数量
            enable_query_expansion: 是否启用查询扩展（默认使用初始化时的设置）
            enable_hybrid_search: 是否启用混合检索
            enable_reranking: 是否启用重排序
            enable_multi_query: 是否启用多查询生成
            enable_hyde: 是否启用HyDE
            metadata_filter: 元数据过滤条件

        Returns:
            检索结果列表
        """
        # 使用默认设置
        enable_query_expansion = (
            enable_query_expansion
            if enable_query_expansion is not None
            else self.enable_query_expansion
        )
        enable_hybrid_search = (
            enable_hybrid_search
            if enable_hybrid_search is not None
            else self.enable_hybrid_search
        )
        enable_reranking = (
            enable_reranking
            if enable_reranking is not None
            else self.enable_reranking
        )

        # 1. 查询扩展
        queries_to_search = [query]

        if enable_query_expansion and self.query_expander:
            # 禁用查询扩展以避免异步问题
            # TODO: 实现真正的异步支持
            logger.info("查询扩展暂时禁用（异步兼容性问题）")
            pass

        # 2. 检索
        all_results = []

        if enable_hybrid_search and self.hybrid_retriever:
            # 混合检索
            for q in queries_to_search:
                results = self.hybrid_retriever.invoke(q)
                all_results.extend(results)

        else:
            # 纯向量检索
            for q in queries_to_search:
                if self.engine.vectorstore:
                    results = self.engine.vectorstore.similarity_search(q, k=top_k)
                    all_results.extend(results)

        # 3. 去重（基于文档内容）
        unique_docs = self._deduplicate_documents(all_results)

        # 4. 元数据过滤
        if metadata_filter:
            unique_docs = self._filter_by_metadata(unique_docs, metadata_filter)

        # 5. 重排序
        if enable_reranking and self.reranker:
            unique_docs = self.reranker.rerank(query, unique_docs, top_k=top_k)

        # 6. 截断到top_k
        final_results = unique_docs[:top_k]

        logger.info(
            f"增强检索: query='{query}', "
            f"查询数={len(queries_to_search)}, "
            f"检索结果={len(all_results)}, "
            f"去重后={len(unique_docs)}, "
            f"最终返回={len(final_results)}"
        )

        return final_results

    def _deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        """去重文档

        Args:
            documents: 文档列表

        Returns:
            去重后的文档列表
        """
        seen = set()
        unique_docs = []

        for doc in documents:
            # 使用内容和元数据哈希去重
            doc_key = self._get_document_key(doc)
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)

        return unique_docs

    def _get_document_key(self, doc: Document) -> str:
        """生成文档唯一标识"""
        import hashlib

        content = doc.page_content
        metadata_str = str(sorted(doc.metadata.items()))
        combined = f"{content}|{metadata_str}"

        return hashlib.md5(combined.encode()).hexdigest()

    def _filter_by_metadata(
        self,
        documents: list[Document],
        filter_dict: dict,
    ) -> list[Document]:
        """根据元数据过滤文档

        Args:
            documents: 文档列表
            filter_dict: 过滤条件

        Returns:
            过滤后的文档列表
        """
        filtered_docs = []

        for doc in documents:
            match = True
            for key, value in filter_dict.items():
                if key not in doc.metadata:
                    match = False
                    break

                doc_value = doc.metadata[key]

                # 支持不同的匹配方式
                if isinstance(value, list):
                    # 值在列表中
                    if doc_value not in value:
                        match = False
                        break
                elif callable(value):
                    # 自定义匹配函数
                    if not value(doc_value):
                        match = False
                        break
                else:
                    # 精确匹配
                    if doc_value != value:
                        match = False
                        break

            if match:
                filtered_docs.append(doc)

        return filtered_docs


# 测试代码
if __name__ == "__main__":
    def test_enhanced_retriever():
        """测试增强检索器"""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        console.print("[bold cyan]测试 Enhanced Retriever[/bold cyan]\n")

        try:
            # 初始化引擎
            console.print("[yellow]初始化RAG引擎...[/yellow]")
            engine = RAGEngine()
            engine.initialize(load_only=True)
            console.print("[green]✓ RAG引擎初始化成功\n[/green]")

            # 创建增强检索器
            console.print("[yellow]创建增强检索器...[/yellow]")
            retriever = EnhancedRetriever(
                engine=engine,
                enable_query_expansion=True,
                enable_hybrid_search=False,  # 需要文档构建BM25
                enable_reranking=False,  # 需要模型
            )
            console.print("[green]✓ 增强检索器创建成功\n[/green]")

            # 测试查询
            query = "变压器温度过高怎么办"
            console.print(f"[yellow]查询:[/yellow] {query}\n")

            # 基础检索
            console.print("[yellow]1. 基础检索[/yellow]")
            results = retriever.retrieve(
                query,
                top_k=3,
                enable_query_expansion=False,
                enable_hybrid_search=False,
                enable_reranking=False,
            )

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("排名", width=6)
            table.add_column("内容预览", width=80)

            for i, doc in enumerate(results, 1):
                preview = doc.page_content[:80] + "..."
                table.add_row(str(i), preview)

            console.print(table)
            console.print()

            # 查询扩展检索
            console.print("[yellow]2. 查询扩展检索[/yellow]")
            results = retriever.retrieve(
                query,
                top_k=3,
                enable_query_expansion=True,
                enable_multi_query=True,
                enable_hybrid_search=False,
                enable_reranking=False,
            )

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("排名", width=6)
            table.add_column("内容预览", width=80)

            for i, doc in enumerate(results, 1):
                preview = doc.page_content[:80] + "..."
                table.add_row(str(i), preview)

            console.print(table)

            console.print("\n[bold green]✓ 测试完成[/bold green]")

        except Exception as e:
            console.print(f"[red]✗ 测试失败: {e}[/red]")
            import traceback

            traceback.print_exc()

    # 运行测试
    test_enhanced_retriever()
