"""Long-term Memory - 长期知识记忆

基于向量存储的长期记忆，支持语义检索。
"""

import logging
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from rich.console import Console

from rag_agent.rag_engine import RAGEngine

logger = logging.getLogger(__name__)
console = Console()


class LongTermMemory:
    """长期记忆管理

    基于向量数据库存储和检索长期知识。
    """

    def __init__(self, engine: RAGEngine | None = None):
        """初始化长期记忆

        Args:
            engine: RAGEngine 实例，如果为 None 则自动创建
        """
        self.engine = engine or RAGEngine()
        if not self.engine.vectorstore:
            self.engine.initialize(load_only=True)

    async def store_memory(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """存储记忆到向量数据库

        Args:
            content: 记忆内容
            metadata: 元数据（如类型、时间戳等）

        Returns:
            存储的记忆 ID

        Examples:
            >>> memory = LongTermMemory()
            >>> memory_id = await memory.store_memory(
            ...     "变压器正常运行温度应控制在85℃以下",
            ...     {"type": "knowledge", "device": "变压器"}
            ... )
        """
        try:
            # 准备元数据
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "memory_type": metadata.get("type", "general"),
                }
            )

            # 创建文档
            doc = Document(page_content=content, metadata=metadata)

            # 添加到向量存储
            if self.engine.vectorstore:
                self.engine.vectorstore.add_documents([doc])
                logger.info(f"存储记忆: {content[:50]}...")
                return f"mem-{datetime.now().timestamp()}"
            else:
                raise RuntimeError("向量存储未初始化")

        except Exception as e:
            logger.error(f"存储记忆失败: {e}", exc_info=True)
            raise

    async def retrieve_memories(
        self, query: str, k: int = 3, filter_metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        """检索相关记忆

        Args:
            query: 查询内容
            k: 返回的记忆数量
            filter_metadata: 元数据过滤条件

        Returns:
            相关记忆列表

        Examples:
            >>> memory = LongTermMemory()
            >>> memories = await memory.retrieve_memories(
            ...     "变压器的温度限制",
            ...     k=3,
            ...     filter_metadata={"device": "变压器"}
            ... )
        """
        try:
            if not self.engine.vectorstore:
                logger.warning("向量存储未初始化，返回空列表")
                return []

            # 执行相似度搜索
            if filter_metadata:
                # 如果有过滤条件，使用向量数据库的过滤功能
                memories = self.engine.vectorstore.similarity_search(query, k=k, filter=filter_metadata)
            else:
                memories = self.engine.vectorstore.similarity_search(query, k=k)

            logger.info(f"检索到 {len(memories)} 条相关记忆")
            return memories

        except Exception as e:
            logger.error(f"检索记忆失败: {e}", exc_info=True)
            return []

    async def search_by_metadata(self, metadata_filter: dict[str, Any], k: int = 10) -> list[Document]:
        """根据元数据搜索记忆

        Args:
            metadata_filter: 元数据过滤条件
            k: 返回的数量

        Returns:
            匹配的记忆列表

        Examples:
            >>> memory = LongTermMemory()
            >>> memories = await memory.search_by_metadata(
            ...     {"device": "变压器", "memory_type": "fault"}
            ... )
        """
        try:
            if not self.engine.vectorstore:
                return []

            # 使用向量数据库的相似度搜索（带空查询）
            memories = self.engine.vectorstore.similarity_search("", k=k, filter=metadata_filter)

            return memories

        except Exception as e:
            logger.error(f"元数据搜索失败: {e}", exc_info=True)
            return []

    async def get_memory_stats(self) -> dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典

        Examples:
            >>> memory = LongTermMemory()
            >>> stats = await memory.get_memory_stats()
            >>> print(stats["total_memories"])
        """
        try:
            if not self.engine.vectorstore:
                return {"total_memories": 0, "indexed": False}

            # 获取向量存储中的文档数量
            # 注意：FAISS 不直接提供 count 方法，这里使用估算
            return {
                "total_memories": "unknown (FAISS)",
                "indexed": True,
                "engine": str(type(self.engine.vectorstore).__name__),
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}", exc_info=True)
            return {"total_memories": 0, "error": str(e)}


# 全局单例
_long_term_memory: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """获取长期记忆单例

    Returns:
        长期记忆实例
    """
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test_long_term_memory():
        """测试长期记忆"""
        console.print("[bold cyan]测试 Long-term Memory[/bold cyan]\n")

        memory = get_long_term_memory()

        # 存储记忆
        console.print("[yellow]存储记忆...[/yellow]")
        await memory.store_memory(
            "变压器油温应控制在85℃以下，绕组温度不超过95℃", {"type": "standard", "device": "变压器"}
        )
        await memory.store_memory(
            "变压器常见故障包括绕组短路、铁芯多点接地等", {"type": "knowledge", "device": "变压器"}
        )
        console.print("✓ 存储了 2 条记忆\n")

        # 检索记忆
        console.print("[yellow]检索记忆...[/yellow]")
        memories = await memory.retrieve_memories("变压器的温度限制", k=2)
        console.print(f"✓ 检索到 {len(memories)} 条记忆")

        for i, mem in enumerate(memories, 1):
            console.print(f"\n{i}. {mem.page_content}")
            console.print(f"   元数据: {mem.metadata}")

        # 统计信息
        console.print("\n[yellow]统计信息:[/yellow]")
        stats = await memory.get_memory_stats()
        console.print(f"✓ {stats}")

        console.print("\n[bold green]✓ 测试完成[/bold green]")

    asyncio.run(test_long_term_memory())
