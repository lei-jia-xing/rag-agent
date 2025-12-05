"""RAG 问答引擎"""

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from rag_agent.config import config

console = Console()


class RAGEngine:
    """RAG 问答引擎

    提供基于向量检索的问答和报告生成功能。
    """

    def __init__(self) -> None:
        """初始化 RAG 引擎"""
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vectorstore: FAISS | None = None
        self.retriever: BaseRetriever | None = None
        self.llm: ChatOpenAI | None = None

    def initialize(self, documents: list[dict[str, str]] | None = None, load_only: bool = False) -> None:
        """
        初始化 RAG 引擎（支持持久化缓存）

        Args:
            documents: 文档列表，每个文档包含 content 和可选的 metadata
            load_only: 仅加载模式（如果向量库不存在则报错，不自动构建）

        Raises:
            ValueError: 当 API key 未设置或 load_only 模式下向量库不存在时
        """
        console.print("[cyan]正在初始化 RAG 引擎...[/cyan]")

        # 1. 初始化嵌入模型
        console.print(f"[cyan]加载嵌入模型: {config.EMBEDDING_MODEL}...[/cyan]")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # 2. 尝试从磁盘加载已有的向量数据库
        vectorstore_path = Path(config.VECTORSTORE_PATH)
        if vectorstore_path.exists():
            try:
                console.print("[yellow]发现已保存的向量数据库，正在加载...[/yellow]")
                self.vectorstore = FAISS.load_local(
                    str(vectorstore_path), self.embeddings, allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                console.print("[green]已加载向量数据库（跳过文档处理）[/green]")
            except Exception as e:
                console.print(f"[yellow]加载失败: {e}，将重新构建[/yellow]")
                self.vectorstore = None

        # 3. 如果没有缓存，则创建新的向量数据库
        if not self.vectorstore:
            # 仅加载模式下，向量库不存在则报错
            if load_only:
                raise ValueError(
                    f"向量数据库不存在: {vectorstore_path}\n请先运行 'uv run rag-agent build' 构建向量数据库"
                )

            # 需要文档来构建向量库
            if not documents:
                raise ValueError("未提供文档数据，无法构建向量数据库")
            console.print("[cyan]正在分割文档...[/cyan]")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )

            # 转换为 LangChain Document 对象
            langchain_docs = [
                Document(page_content=doc["content"], metadata=doc.get("metadata", {})) for doc in documents
            ]

            # 分割文档
            split_docs = text_splitter.split_documents(langchain_docs)
            console.print(f"[green]文档已分割为 {len(split_docs)} 个块[/green]")

            # 创建向量数据库
            console.print("[cyan]正在创建向量数据库...[/cyan]")
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            console.print("[green]向量数据库创建完成[/green]")

            # 保存到磁盘
            try:
                vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
                self.vectorstore.save_local(str(vectorstore_path))
                console.print(f"[green]✓ 向量数据库已保存到: {vectorstore_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]保存失败: {e}[/yellow]")

        # 4. 初始化 LLM
        console.print(f"[cyan]正在连接 LLM: {config.MODEL_NAME}...[/cyan]")

        # 验证 API key
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置")

        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,  # type: ignore[arg-type]
            base_url=config.OPENAI_API_BASE,
            temperature=0.7,
        )

        console.print("[green]RAG 引擎初始化完成！[/green]")

    def build_vectorstore(self, documents: list[dict[str, str]], force: bool = False) -> None:
        """
        构建向量数据库（预处理命令专用）

        Args:
            documents: 文档列表
            force: 强制重新构建（即使已存在）

        Raises:
            ValueError: 当未提供文档时
        """
        if not documents:
            raise ValueError("未提供文档数据")

        vectorstore_path = Path(config.VECTORSTORE_PATH)

        # 检查是否已存在
        if vectorstore_path.exists() and not force:
            console.print(f"[yellow]⚠ 向量数据库已存在: {vectorstore_path}[/yellow]")
            console.print("[yellow]使用 --force 参数强制重新构建[/yellow]")
            return

        console.print("[cyan]正在构建向量数据库（全量数据）...[/cyan]")

        # 初始化嵌入模型
        if not self.embeddings:
            console.print(f"[cyan]加载嵌入模型: {config.EMBEDDING_MODEL}...[/cyan]")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
            )

        # 分割文档
        console.print("[cyan]正在分割文档...[/cyan]")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        langchain_docs = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) for doc in documents]
        split_docs = text_splitter.split_documents(langchain_docs)
        console.print(f"[green]文档已分割为 {len(split_docs)} 个块[/green]")

        # 创建向量数据库
        console.print("[cyan]正在创建向量数据库（这可能需要几分钟）...[/cyan]")
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        console.print("[green]向量数据库创建完成[/green]")

        # 保存到磁盘
        vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(vectorstore_path))
        console.print(f"[green]✓ 向量数据库已保存到: {vectorstore_path}[/green]")
        console.print(f"[green]✓ 共处理 {len(documents)} 个文档[/green]")

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """
        检索相关文档（独立能力函数，供 Graph 调用）

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        if not self.vectorstore or not self.retriever:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"[cyan]正在检索: {query}[/cyan]")
        try:
            # 动态调整 k 值
            current_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            documents = current_retriever.invoke(query)
            console.print(f"[green]检索到 {len(documents)} 个相关文档[/green]")
            return documents
        except Exception as e:
            console.print(f"[red]检索失败: {e}[/red]")
            return []

    def generate_answer(self, question: str, documents: list[Document]) -> str:
        """
        基于检索到的文档生成答案（独立能力函数，供 Graph 调用）

        Args:
            question: 用户问题
            documents: 检索到的相关文档

        Returns:
            生成的答案文本
        """
        if not self.llm:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"[cyan]正在生成答案: {question}[/cyan]")

        try:
            context = "\n\n".join([doc.page_content for doc in documents])

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一个问答助手。基于以下上下文信息回答用户的问题。"
                        "如果上下文中没有相关信息，请说明无法从给定信息中找到答案。\n\n"
                        "上下文:\n{context}",
                    ),
                    ("human", "{question}"),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({"context": context, "question": question})
            return str(response.content)

        except Exception as e:
            console.print(f"[red]生成答案失败: {e}[/red]")
            return f"抱歉，生成答案时发生错误: {e}"

    def generate_report(self, topic: str, documents: list[Document]) -> str:
        """
        基于文档生成技术报告

        Args:
            topic: 报告主题
            documents: 参考文档列表

        Returns:
            Markdown 格式的报告
        """
        if not self.llm:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"[cyan]正在生成报告: {topic}[/cyan]")

        try:
            context = "\n\n".join([doc.page_content for doc in documents])

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一个技术写作助手。基于提供的上下文，撰写一份简短的技术报告，"
                        "包含：摘要、关键要点和结论。报告使用 Markdown 格式。",
                    ),
                    (
                        "human",
                        "主题: {topic}\n\n上下文:\n{context}\n\n"
                        "请生成一份清晰、结构化的 Markdown 报告，字数控制在 300-800 字之间。",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({"topic": topic, "context": context})
            return str(response.content)

        except Exception as e:
            console.print(f"[red]生成报告失败: {e}[/red]")
            return f"生成报告失败: {e}"

    def query(self, question: str) -> dict[str, Any]:
        """
        查询问题（保留向后兼容的高层接口）

        Args:
            question: 用户问题

        Returns:
            包含答案和来源文档的字典
        """
        if not self.retriever or not self.llm:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"\n[cyan]正在查询: {question}[/cyan]")

        try:
            # 使用新的能力函数组合
            source_documents = self.retrieve(question)
            answer = self.generate_answer(question, source_documents)

            return {
                "answer": answer,
                "source_documents": source_documents,
            }

        except Exception as e:
            console.print(f"[red]查询失败: {e}[/red]")
            return {"answer": f"抱歉，查询时发生错误: {e}", "source_documents": []}
