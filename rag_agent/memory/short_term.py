"""Short-term Memory - 短期对话记忆

管理对话历史，支持多轮对话上下文。
"""

import logging

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class ShortTermMemory:
    """短期记忆管理

    使用 LangChain 的 InMemoryChatMessageHistory 存储对话历史。
    """

    def __init__(self):
        """初始化短期记忆"""
        # 使用字典存储多个会话的历史
        self.histories: dict[str, BaseChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取指定会话的历史记录

        Args:
            session_id: 会话 ID

        Returns:
            该会话的消息历史

        Examples:
            >>> memory = ShortTermMemory()
            >>> history = memory.get_session_history("user-123")
        """
        if session_id not in self.histories:
            from langchain_core.chat_history import InMemoryChatMessageHistory

            self.histories[session_id] = InMemoryChatMessageHistory()
            logger.info(f"创建新会话: {session_id}")

        return self.histories[session_id]

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """添加消息到会话历史

        Args:
            session_id: 会话 ID
            message: 消息内容

        Examples:
            >>> memory = ShortTermMemory()
            >>> memory.add_message("user-123", HumanMessage("你好"))
        """
        history = self.get_session_history(session_id)
        history.add_message(message)

    def add_user_message(self, session_id: str, content: str) -> None:
        """添加用户消息

        Args:
            session_id: 会话 ID
            content: 消息内容
        """
        self.add_message(session_id, HumanMessage(content))

    def add_ai_message(self, session_id: str, content: str) -> None:
        """添加 AI 消息

        Args:
            session_id: 会话 ID
            content: 消息内容
        """
        self.add_message(session_id, AIMessage(content))

    def get_messages(self, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        """获取会话的所有消息

        Args:
            session_id: 会话 ID
            limit: 限制返回的消息数量（最近 N 条）

        Returns:
            消息列表

        Examples:
            >>> memory = ShortTermMemory()
            >>> messages = memory.get_messages("user-123", limit=5)
        """
        history = self.get_session_history(session_id)
        messages = history.messages

        if limit and len(messages) > limit:
            return messages[-limit:]

        return messages

    def clear_session(self, session_id: str) -> None:
        """清除会话历史

        Args:
            session_id: 会话 ID
        """
        if session_id in self.histories:
            del self.histories[session_id]
            logger.info(f"清除会话: {session_id}")

    def get_session_ids(self) -> list[str]:
        """获取所有会话 ID

        Returns:
            会话 ID 列表
        """
        return list(self.histories.keys())


# 全局单例
_short_term_memory: ShortTermMemory | None = None


def get_short_term_memory() -> ShortTermMemory:
    """获取短期记忆单例

    Returns:
        短期记忆实例
    """
    global _short_term_memory
    if _short_term_memory is None:
        _short_term_memory = ShortTermMemory()
    return _short_term_memory


# 测试代码
if __name__ == "__main__":

    def test_short_term_memory():
        """测试短期记忆"""
        console.print("[bold cyan]测试 Short-term Memory[/bold cyan]\n")

        memory = get_short_term_memory()
        session_id = "test-session-1"

        # 添加消息
        console.print("[yellow]添加消息...[/yellow]")
        memory.add_user_message(session_id, "你好，我是用户")
        memory.add_ai_message(session_id, "你好！有什么可以帮助你的？")
        memory.add_user_message(session_id, "我想了解变压器的维护")

        # 获取消息
        console.print("[yellow]获取消息...[/yellow]")
        messages = memory.get_messages(session_id)
        console.print(f"✓ 消息数: {len(messages)}")

        for i, msg in enumerate(messages, 1):
            msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
            console.print(f"  {i}. [{msg_type}] {msg.content}")

        # 测试限制
        console.print("\n[yellow]最近 2 条消息:[/yellow]")
        recent = memory.get_messages(session_id, limit=2)
        for msg in recent:
            console.print(f"  - {msg.content}")

        # 清除会话
        console.print("\n[yellow]清除会话...[/yellow]")
        memory.clear_session(session_id)
        console.print("✓ 会话已清除")

        console.print("\n[bold green]✓ 测试完成[/bold green]")

    test_short_term_memory()
