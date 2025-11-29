"""RAG 问答引擎"""

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from rag_agent.config import config

console = Console()


class RAGEngine:
    """RAG 问答引擎"""

    def __init__(self):
        """初始化 RAG 引擎"""
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None

    def initialize(self, documents: list[dict[str, str]]):
        """
        初始化 RAG 引擎

        Args:
            documents: 文档列表
        """
        console.print("[cyan]正在初始化 RAG 引擎...[/cyan]")

        # 1. 初始化嵌入模型
        console.print(f"[cyan]加载嵌入模型: {config.EMBEDDING_MODEL}...[/cyan]")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # 2. 文本分块
        console.print("[cyan]正在分割文档...[/cyan]")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        # 转换为 LangChain Document 对象
        langchain_docs = [
            Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]

        # 分割文档
        split_docs = text_splitter.split_documents(langchain_docs)
        console.print(f"[green]文档已分割为 {len(split_docs)} 个块[/green]")

        # 3. 创建向量数据库
        console.print("[cyan]正在创建向量数据库...[/cyan]")
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        console.print("[green]向量数据库创建完成[/green]")

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

    def query(self, question: str) -> dict[str, Any]:
        """
        查询问题

        Args:
            question: 用户问题

        Returns:
            包含答案和来源文档的字典
        """
        if not self.retriever or not self.llm:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"\n[cyan]正在查询: {question}[/cyan]")

        try:
            # 1. 检索相关文档
            source_documents = self.retriever.invoke(question)

            # 2. 构建提示词
            context = "\n\n".join([doc.page_content for doc in source_documents])

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

            # 3. 生成回答
            chain = prompt | self.llm
            response = chain.invoke({"context": context, "question": question})

            return {
                "answer": response.content,
                "source_documents": source_documents,
            }

        except Exception as e:
            console.print(f"[red]查询失败: {e}[/red]")
            return {"answer": f"抱歉，查询时发生错误: {e}", "source_documents": []}
