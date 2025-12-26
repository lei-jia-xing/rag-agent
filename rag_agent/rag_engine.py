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
                        "你是一个专业的技术报告写作专家。请基于提供的上下文，撰写一份详细、专业的技术报告。"
                        "报告必须使用Markdown格式，包含以下结构：\n\n"
                        "# 报告标题\n\n"
                        "## 摘要\n"
                        "简要概述报告内容，约200字\n\n"
                        "## 一、技术背景与发展趋势\n"
                        "详细描述该技术的发展历程和当前趋势，约300字\n\n"
                        "## 二、主要技术特点与原理\n"
                        "深入分析核心技术特点和工作原理，约400字\n\n"
                        "## 三、应用领域与案例分析\n"
                        "介绍主要应用场景和具体案例，约300字\n\n"
                        "## 四、技术对比分析\n"
                        "提供不同技术方案的对比，包括优缺点，使用表格形式展示\n\n"
                        "## 五、挑战与未来展望\n"
                        "分析当前面临的挑战和未来发展方向，约200字\n\n"
                        "## 六、结论\n"
                        "总结全文，提出观点和建议，约150字\n\n"
                        "写作要求：\n"
                        "1. 使用专业的学术语言和技术术语\n"
                        "2. 包含具体的技术细节和实例\n"
                        "3. 结构清晰，逻辑严谨\n"
                        "4. 每个部分都要有充分的内容\n"
                        "5. 总字数控制在1500-2000字之间\n"
                        "6. 重要概念用**粗体**标记\n"
                        "7. 仅使用基础Markdown语法：标题、段落、列表、表格、引用\n"
                        "8. 在技术对比部分必须包含一个表格\n"
                        "9. 适当使用有序列表、无序列表和引用\n"
                        "10. 不要使用图表、Mermaid、LaTeX等复杂格式\n"
                        "11. 不要使用代码块（```）\n"
                        "12. 确保所有Markdown语法正确，可以被标准解析器处理\n",
                    ),
                    (
                        "human",
                        "报告主题：{topic}\n\n参考文档：\n{context}\n\n"
                        "请基于以上文档，按照要求的结构撰写一份详细、专业的技术报告。确保内容详实、结构完整、分析深入，使用标准Markdown格式。",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({"topic": topic, "context": context})
            return str(response.content)

        except Exception as e:
            console.print(f"[red]生成报告失败: {e}[/red]")
            return f"生成报告失败: {e}"

    def generate_latex_content(self, topic: str, documents: list[Document]) -> str:
        """
        生成LaTeX格式的报告内容

        Args:
            topic: 报告主题
            documents: 参考文档列表

        Returns:
            LaTeX格式的报告内容（包括摘要和章节，但不包括文档包装）
        """
        if not self.llm:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"[cyan]正在生成LaTeX内容: {topic}[/cyan]")

        try:
            context = "\n\n".join([doc.page_content for doc in documents])

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一个专业的技术报告写作专家，精通LaTeX排版。请基于提供的上下文，撰写一份详细、专业的技术报告。"
                        "报告必须使用纯LaTeX格式（不含文档类声明和导言区），包含以下结构：\n\n"
                        "\\begin{{abstract}}\n"
                        "简要概述报告内容，约200字\n"
                        "\\end{{abstract}}\n\n"
                        "\\section{{技术背景与发展趋势}}\n"
                        "详细描述该技术的发展历程和当前趋势，约300字\n\n"
                        "\\section{{主要技术特点与原理}}\n"
                        "深入分析核心技术特点和工作原理，约400字\n\n"
                        "\\section{{应用领域与案例分析}}\n"
                        "介绍主要应用场景和具体案例，约300字\n\n"
                        "\\section{{技术对比分析}}\n"
                        "提供不同技术方案的对比，包括优缺点，使用表格形式展示\n\n"
                        "\\section{{挑战与未来展望}}\n"
                        "分析当前面临的挑战和未来发展方向，约200字\n\n"
                        "\\section{{结论}}\n"
                        "总结全文，提出观点和建议，约150字\n\n"
                        "写作要求：\n"
                        "1. 使用专业的学术语言和技术术语\n"
                        "2. 包含具体的技术细节和实例\n"
                        "3. 结构清晰，逻辑严谨\n"
                        "4. 每个部分都要有充分的内容\n"
                        "5. 总字数控制在1500-2000字之间\n"
                        "6. 重要概念用\\textbf{{粗体}}标记\n"
                        "7. 使用LaTeX环境：itemize、enumerate、table、tabular等\n"
                        "8. 在技术对比部分必须包含一个tabular表格\n"
                        "9. 适当使用有序列表、无序列表和引用\n"
                        "10. 可以包含简单的数学公式（使用$...$或\\[...\\]）\n"
                        "11. 不要使用图表、tikz等复杂图形\n"
                        "12. 确保所有LaTeX语法正确，可以直接编译\n"
                        "13. 输出必须是纯LaTeX代码，不要包含任何解释性文字\n",
                    ),
                    (
                        "human",
                        "报告主题：{topic}\n\n参考文档：\n{context}\n\n"
                        "请基于以上文档，按照要求的结构撰写一份详细、专业的LaTeX格式技术报告。"
                        "只输出LaTeX代码，不要包含其他任何内容。",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({"topic": topic, "context": context})
            return str(response.content)

        except Exception as e:
            console.print(f"[red]生成LaTeX内容失败: {e}[/red]")
            return f"生成LaTeX内容失败: {e}"

    def generate_diagnosis_fields(self, device_name: str, documents: list[Document]) -> dict[str, str]:
        """
        生成设备健康诊断报告的字段数据（JSON格式）

        Args:
            device_name: 设备名称
            documents: 参考文档列表

        Returns:
            包含所有字段的字典，用于填充LaTeX模板
        """
        if not self.llm:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize()")

        console.print(f"[cyan]正在生成诊断字段: {device_name}[/cyan]")

        try:
            context = "\n\n".join([doc.page_content for doc in documents])

            import json

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一个专业的设备健康诊断专家。基于提供的上下文信息，生成设备健康诊断报告的各个字段内容。你需要返回一个纯JSON格式的响应。\n\n返回的JSON必须包含以下字段：\n- title: 报告标题\n- report_id: 报告编号（格式：DX-YYYYMMDD-001）\n- device_name: 设备名称\n- device_model: 设备型号\n- location: 安装位置\n- diagnosis_date: 诊断日期\n- data_range: 数据采集范围\n- health_score: 整体健康评分（0-100）\n- health_status: 健康状态（正常/警告/异常/严重）\n- risk_level: 风险等级（低/中/高）\n- issue_count: 主要问题数\n- abstract: 诊断摘要\n- device_basic_info: 设备基本信息\n- operating_environment: 运行环境\n- maintenance_history: 历史维护记录\n- monitoring_data_summary: 监测数据汇总\n- key_metrics_analysis: 关键指标分析\n- trend_analysis: 趋势分析\n- anomaly_detection: 异常检测\n- fault_description: 故障现象描述\n- fault_cause_analysis: 故障原因分析\n- fault_location: 故障定位\n- urgent_measures: 紧急处理措施\n- maintenance_plan: 维护计划\n- spare_parts_suggestion: 备件建议\n- current_risks: 当前风险\n- potential_risks: 潜在风险\n- risk_control: 风险控制建议\n- conclusion_and_recommendations: 结论与建议\n- technical_parameters: 技术参数\n- related_standards: 相关标准\n- diagnosis_method: 诊断方法说明",
                    ),
                    ("human", "设备名称：{device_name}\n\n参考文档：\n{context}"),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({"device_name": device_name, "context": context})
            response_text = str(response.content)

            try:
                data = json.loads(response_text)
                return data
            except json.JSONDecodeError:
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                try:
                    data = json.loads(response_text)
                    return data
                except json.JSONDecodeError as e:
                    console.print(f"[red]JSON解析失败: {e}[/red]")
                    console.print(f"[yellow]原始响应: {response_text[:500]}[/yellow]")
                    raise ValueError("无法解析LLM返回的JSON数据") from e

        except Exception as e:
            console.print(f"[red]生成诊断字段失败: {e}[/red]")
            raise

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
