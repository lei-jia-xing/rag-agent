"""Reranker - 检索结果重排序

对初步检索结果进行重新排序，提升Top-K准确率。
支持多种重排序模型：
- Cohere Rerank API
- bge-reranker开源模型
- SentenceTransformers交叉编码器
"""

import logging

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseReranker:
    """重排序器基类"""

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """重排序文档

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的top-k文档数量，None表示返回全部

        Returns:
            重排序后的文档列表
        """
        raise NotImplementedError


class CohereReranker(BaseReranker):
    """Cohere Rerank API

    使用Cohere的Rerank API进行重排序。
    需要API key: https://dashboard.cohere.com/api-keys

    优点：
    - 效果好
    - 速度快
    - 支持多语言

    缺点：
    - 需要付费（有免费额度）
    - 需要网络连接
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-multilingual-v3.0",
        top_n: int | None = None,
    ):
        """初始化Cohere重排序器

        Args:
            api_key: Cohere API密钥
            model: 模型名称（默认rerank-multilingual-v3.0）
            top_n: 返回的top-n文档数量
        """
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self.api_url = "https://api.cohere.ai/v1/rerank"

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """使用Cohere API重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的top-k文档数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        top_n = top_k or self.top_n or len(documents)

        # 准备文档文本
        docs_text = [doc.page_content for doc in documents]

        # 调用Cohere API
        try:
            response = httpx.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": docs_text,
                    "top_n": top_n,
                    "return_documents": True,
                },
                timeout=30.0,
            )

            response.raise_for_status()
            results = response.json()

            # 解析结果
            reranked_docs = []
            for result in results.get("results", []):
                idx = result["index"]
                doc = documents[idx]
                # 可以在元数据中添加重排序分数
                doc.metadata["rerank_score"] = result.get("relevance_score", 0.0)
                reranked_docs.append(doc)

            logger.info(f"Cohere重排序: query='{query}', 输入{len(documents)}篇, 返回{len(reranked_docs)}篇")

            return reranked_docs

        except Exception as e:
            logger.error(f"Cohere重排序失败: {e}", exc_info=True)
            # 失败时返回原始文档
            return documents[:top_k]


class BGEReranker(BaseReranker):
    """BGE Reranker（开源）

    使用北京智源研究院的BGE重排序模型。
    模型列表：
    - BAAI/bge-reranker-v2-m3（多语言，轻量）
    - BAAI/bge-reranker-large（中文，效果好）

    优点：
    - 完全免费
    - 可本地部署
    - 中文效果好

    缺点：
    - 需要下载模型
    - 需要GPU加速（CPU较慢）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        """初始化BGE重排序器

        Args:
            model_name: 模型名称
            device: 运行设备（cpu或cuda）
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """延迟加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"加载BGE重排序模型: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
            logger.info("BGE重排序模型加载完成")

        except ImportError:
            logger.error("未安装sentence_transformers，请运行: uv pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载BGE模型失败: {e}", exc_info=True)
            raise

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """使用BGE模型重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的top-k文档数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 加载模型
        if self._model is None:
            self._load_model()

        top_k = top_k or len(documents)

        pairs = [[query, doc.page_content] for doc in documents]

        try:
            if self._model is None:
                logger.warning("BGE模型未加载，返回原始顺序")
                return documents[:top_k]

            scores = self._model.predict(pairs)

            # 排序
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 返回top-k
            reranked_docs = []
            for doc, score in scored_docs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)

            logger.info(f"BGE重排序: query='{query}', 输入{len(documents)}篇, 返回{len(reranked_docs)}篇")

            return reranked_docs

        except Exception as e:
            logger.error(f"BGE重排序失败: {e}", exc_info=True)
            return documents[:top_k]


class Reranker:
    """通用重排序器接口

    根据配置选择合适的重排序器。

    Examples:
        >>> 使用Cohere
        >>> reranker = Reranker(type="cohere", api_key="xxx")
        >>> results = reranker.rerank(query, documents)

        >>> 使用BGE
        >>> reranker = Reranker(type="bge", model_name="BAAI/bge-reranker-v2-m3")
        >>> results = reranker.rerank(query, documents)
    """

    def __init__(
        self,
        type: str = "cohere",  # "cohere" or "bge"
        **kwargs,
    ):
        """初始化重排序器

        Args:
            type: 重排序器类型（"cohere"或"bge"）
            **kwargs: 传递给具体重排序器的参数
        """
        self.type = type

        if type == "cohere":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("Cohere重排序器需要api_key参数")
            self.reranker = CohereReranker(
                api_key=api_key,
                model=kwargs.get("model", "rerank-multilingual-v3.0"),
                top_n=kwargs.get("top_n"),
            )

        elif type == "bge":
            self.reranker = BGEReranker(
                model_name=kwargs.get("model_name", "BAAI/bge-reranker-v2-m3"),
                device=kwargs.get("device", "cpu"),
            )

        else:
            raise ValueError(f"不支持的重排序器类型: {type}")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """重排序文档

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的top-k文档数量

        Returns:
            重排序后的文档列表
        """
        return self.reranker.rerank(query, documents, top_k)


# 测试代码
if __name__ == "__main__":

    def test_reranker():
        """测试重排序器"""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        console.print("[bold cyan]测试 Reranker[/bold cyan]\n")

        # 创建测试文档
        test_documents = [
            Document(
                page_content="变压器的正常运行温度一般在60-85摄氏度之间。",
                metadata={"source": "doc1"},
            ),
            Document(
                page_content="电力系统中的电压等级有110kV、220kV、500kV等。",
                metadata={"source": "doc2"},
            ),
            Document(
                page_content="当变压器温度超过90度时需要检查冷却系统。",
                metadata={"source": "doc3"},
            ),
            Document(
                page_content="断路器是电力系统中的重要保护设备。",
                metadata={"source": "doc4"},
            ),
            Document(
                page_content="变压器的温度监测应包括油温、绕组温度等。",
                metadata={"source": "doc5"},
            ),
        ]

        query = "变压器温度"
        console.print(f"[yellow]查询:[/yellow] {query}")
        console.print(f"[yellow]文档数:[/yellow] {len(test_documents)}\n")

        # 测试BGE重排序器
        console.print("[yellow]测试 BGE Reranker...[/yellow]")
        try:
            reranker = Reranker(
                type="bge",
                model_name="BAAI/bge-reranker-v2-m3",
                device="cpu",
            )

            reranked = reranker.rerank(query, test_documents, top_k=3)

            # 创建结果表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("排名", width=6)
            table.add_column("分数", width=10)
            table.add_column("来源", width=10)
            table.add_column("内容预览", width=50)

            for i, doc in enumerate(reranked, 1):
                score = doc.metadata.get("rerank_score", 0.0)
                content_preview = doc.page_content[:50] + "..."
                table.add_row(
                    str(i),
                    f"{score:.4f}",
                    doc.metadata.get("source", "未知"),
                    content_preview,
                )

            console.print(table)
            console.print()

        except Exception as e:
            console.print(f"[red]BGE测试失败: {e}[/red]\n")

        # 测试Cohere重排序器
        console.print("[yellow]测试 Cohere Reranker（需要API key）...[/yellow]")
        console.print("[dim]跳过（需要配置COHERE_API_KEY环境变量）[/dim]\n")

        console.print("[bold green]✓ 测试完成[/bold green]")

    # 运行测试
    test_reranker()
