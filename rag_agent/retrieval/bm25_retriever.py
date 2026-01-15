"""BM25 Retriever - BM25稀疏检索器

实现BM25算法进行稀疏检索，与向量检索互补。
BM25对关键词匹配、专业术语检索效果更好。
"""

import logging
import math
from collections import defaultdict
from typing import Any

import jieba
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25检索器

    使用BM25算法进行稀疏检索，擅长：
    - 精确关键词匹配
    - 专业术语检索
    - 短语查询
    - 补充向量检索的不足

    BM25公式:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) /
                          (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))

    其中:
    - qi: 查询中的词项
    - f(qi,D): 词项qi在文档D中的频率
    - |D|: 文档D的长度
    - avgdl: 平均文档长度
    - k1: 调节词频饱和度的参数（默认1.5）
    - b: 调节长度归一化的参数（默认0.75）
    """

    documents: list[Document]
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 5

    # 内部状态
    _corpus: list[list[str]] = []
    _doc_ids: list[int] = []
    _idf: dict[str, float] = {}
    _avgdl: float = 0.0

    class Config:
        """Pydantic配置"""

        arbitrary_types_allowed = True

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 5,
    ):
        """初始化BM25检索器

        Args:
            documents: 文档列表
            k1: BM25参数k1，控制词频影响（默认1.5）
            b: BM25参数b，控制长度归一化（默认0.75）
            top_k: 返回的top-k文档数量

        Examples:
            >>> documents = [Document(page_content="..."), ...]
            >>> retriever = BM25Retriever(documents, k1=1.5, b=0.75)
            >>> results = retriever.invoke("变压器温度")
        """
        super().__init__()
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.top_k = top_k
        self._build_index()

    def _build_index(self) -> None:
        """构建BM25索引

        处理流程：
        1. 分词（使用jieba）
        2. 计算IDF
        3. 计算平均文档长度
        """
        logger.info(f"构建BM25索引，文档数: {len(self.documents)}")

        # 1. 分词
        self._corpus = []
        self._doc_ids = []

        for idx, doc in enumerate(self.documents):
            # 使用jieba分词
            tokens = list(jieba.cut(doc.page_content))
            # 过滤停用词和短词
            tokens = [t for t in tokens if len(t) > 1]
            self._corpus.append(tokens)
            self._doc_ids.append(idx)

        # 2. 计算IDF
        self._idf = self._calculate_idf(self._corpus)

        # 3. 计算平均文档长度
        doc_lengths = [len(doc) for doc in self._corpus]
        self._avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        logger.info(f"BM25索引构建完成: 词汇量={len(self._idf)}, 平均文档长度={self._avgdl:.1f}")

    def _calculate_idf(self, corpus: list[list[str]]) -> dict[str, float]:
        """计算IDF（逆文档频率）

        Args:
            corpus: 分词后的语料库

        Returns:
            词项到IDF分数的映射
        """
        idf = {}
        N = len(corpus)

        # 计算每个词项的文档频率
        df = defaultdict(int)
        for doc in corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                df[token] += 1

        # 计算IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for token, freq in df.items():
            idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

        return idf

    def _get_scores(self, query: str) -> list[float]:
        """计算查询对所有文档的BM25分数

        Args:
            query: 查询文本

        Returns:
            每个文档的BM25分数列表
        """
        # 分词
        query_tokens = list(jieba.cut(query))
        query_tokens = [t for t in query_tokens if len(t) > 1]

        scores = []

        for doc_tokens in self._corpus:
            score = 0.0
            doc_len = len(doc_tokens)

            for token in query_tokens:
                # 词频
                f = doc_tokens.count(token)

                if f == 0:
                    continue

                # IDF
                idf = self._idf.get(token, 0)

                # BM25分数
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                score += idf * (numerator / denominator)

            scores.append(score)

        return scores

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        """检索相关文档

        Args:
            query: 查询文本
            run_manager: 运行管理器（可选）

        Returns:
            按相关性排序的文档列表
        """
        # 计算BM25分数
        scores = self._get_scores(query)

        # 排序并获取top-k
        scored_docs = list(zip(scores, self._doc_ids))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 返回top-k文档
        top_docs = scored_docs[: self.top_k]
        results = [self.documents[doc_id] for score, doc_id in top_docs if score > 0]

        max_score = top_docs[0][0] if top_docs else 0.0
        logger.info(f"BM25检索: query='{query}', 返回{len(results)}个文档, 最高分={max_score:.2f}")

        return results


# 测试代码
if __name__ == "__main__":

    def test_bm25_retriever():
        """测试BM25检索器"""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        console.print("[bold cyan]测试 BM25 Retriever[/bold cyan]\n")

        # 创建测试文档
        test_documents = [
            Document(
                page_content="变压器是电力系统中的重要设备，主要用于电压变换。",
                metadata={"source": "doc1", "topic": "变压器基础"},
            ),
            Document(
                page_content="变压器的正常运行温度一般在60-85摄氏度之间。",
                metadata={"source": "doc2", "topic": "变压器温度"},
            ),
            Document(
                page_content="当变压器温度超过90度时，需要立即检查冷却系统。",
                metadata={"source": "doc3", "topic": "变压器故障"},
            ),
            Document(
                page_content="电力电容器用于无功补偿，提高功率因数。",
                metadata={"source": "doc4", "topic": "电容器"},
            ),
            Document(
                page_content="断路器是电力系统中的重要保护设备。",
                metadata={"source": "doc5", "topic": "断路器"},
            ),
        ]

        # 创建BM25检索器
        console.print("[yellow]创建BM25检索器...[/yellow]")
        retriever = BM25Retriever(
            documents=test_documents,
            k1=1.5,
            b=0.75,
            top_k=3,
        )
        console.print("[green]✓ BM25检索器创建成功\n[/green]")

        # 测试检索
        test_queries = [
            "变压器温度",
            "电力设备",
            "无功补偿",
        ]

        for query in test_queries:
            console.print(f"[yellow]查询:[/yellow] {query}")

            results = retriever.invoke(query)

            # 创建结果表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("排名", width=6)
            table.add_column("分数", width=10)
            table.add_column("来源", width=10)
            table.add_column("内容预览", width=50)

            for i, doc in enumerate(results, 1):
                # 重新计算分数用于显示
                scores = retriever._get_scores(query)
                doc_id = retriever._doc_ids[retriever.documents.index(doc)]
                score = scores[doc_id]

                content_preview = doc.page_content[:50] + "..."
                table.add_row(
                    str(i),
                    f"{score:.3f}",
                    doc.metadata.get("source", "未知"),
                    content_preview,
                )

            console.print(table)
            console.print()

        console.print("[bold green]✓ 测试完成[/bold green]")

    # 运行测试
    test_bm25_retriever()
