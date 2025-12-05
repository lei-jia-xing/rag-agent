"""应用基类

定义所有 RAG 应用的统一接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class AppConfig:
    """应用配置"""

    name: str
    description: str
    dataset_name: str | None = None  # 可覆盖默认数据集
    model_name: str | None = None  # 可覆盖默认模型


class BaseApp(ABC):
    """应用基类

    所有 RAG 应用的抽象基类，定义统一接口。
    未来可通过 LangGraph 的意图识别路由到具体应用。
    """

    def __init__(self) -> None:
        """初始化应用"""
        self._initialized: bool = False

    @property
    @abstractmethod
    def config(self) -> AppConfig:
        """获取应用配置"""
        ...

    @property
    def name(self) -> str:
        """应用名称"""
        return self.config.name

    @property
    def description(self) -> str:
        """应用描述"""
        return self.config.description

    @abstractmethod
    def initialize(self) -> None:
        """初始化应用（加载模型、向量库等）"""
        ...

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> str:
        """运行应用

        Args:
            query: 用户输入
            **kwargs: 额外参数

        Returns:
            应用输出
        """
        ...

    def get_context(self, query: str, k: int = 3) -> list[Document]:
        """获取相关上下文（可选实现）

        Args:
            query: 查询
            k: 返回文档数

        Returns:
            相关文档列表
        """
        return []

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized
