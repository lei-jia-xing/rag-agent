"""Hugging Face 数据集加载模块"""

from typing import Any

from datasets import Dataset, load_dataset
from rich.console import Console

console = Console()


class DatasetLoader:
    """数据集加载器

    从 Hugging Face 加载数据集并转换为统一的文档格式。
    """

    def __init__(self, dataset_name: str, split: str = "train", sample_size: int = 100, load_all: bool = False) -> None:
        """
        初始化数据加载器

        Args:
            dataset_name: Hugging Face 数据集名称
            split: 数据集分割（train/validation/test）
            sample_size: 采样数量（load_all=False 时生效）
            load_all: 是否加载全量数据集（用于预构建向量库）
        """
        self.dataset_name = dataset_name
        self.split = split
        self.sample_size = sample_size
        self.load_all = load_all
        self.dataset: Dataset | None = None

    def load(self) -> list[dict[str, Any]]:
        """
        加载数据集

        Returns:
            文档列表，每个文档包含 content 字段和可选的 metadata

        Raises:
            TypeError: 当加载的数据集类型不正确时
            Exception: 当数据集加载失败时
        """
        console.print(f"[cyan]正在加载数据集: {self.dataset_name}...[/cyan]")

        try:
            # 加载数据集
            dataset_raw = load_dataset(self.dataset_name, split=self.split)

            # 类型检查：确保是 Dataset 类型
            if not isinstance(dataset_raw, Dataset):
                raise TypeError(f"期望 Dataset 类型，但获得 {type(dataset_raw)}")

            self.dataset = dataset_raw

            # 采样（仅在非全量模式下）
            if not self.load_all and len(self.dataset) > self.sample_size:
                self.dataset = self.dataset.shuffle(seed=42).select(range(self.sample_size))
                console.print(f"[green]成功加载 {len(self.dataset)} 条数据（采样）[/green]")
            else:
                console.print(f"[green]成功加载 {len(self.dataset)} 条数据（全量）[/green]")

            # 转换为文档格式
            documents = self._convert_to_documents()
            return documents

        except Exception as e:
            console.print(f"[red]加载数据集失败: {e}[/red]")
            raise

    def _convert_to_documents(self) -> list[dict[str, Any]]:
        """
        将数据集转换为文档格式

        Returns:
            文档列表
        """
        documents: list[dict[str, Any]] = []

        if self.dataset is None:
            return documents

        for item in self.dataset:
            # item 的类型是字典
            item_dict = dict(item)  # 确保是字典类型

            # 根据不同数据集格式提取内容
            content: str = ""

            if "input" in item_dict and "output" in item_dict:
                # Instruction-Input-Output 格式（如电气工程数据集）
                question = str(item_dict.get("input", ""))
                answer = str(item_dict.get("output", ""))
                instruction = str(item_dict.get("instruction", ""))
                # 组合为问答对格式
                content = f"问题: {question}\n\n答案: {answer}"
                if instruction and len(instruction) < 200:
                    content = f"说明: {instruction}\n\n{content}"
            elif "context" in item_dict:
                # SQuAD 格式
                content = str(item_dict["context"])
                if "question" in item_dict:
                    content = f"问题: {item_dict['question']}\n上下文: {content}"
            elif "text" in item_dict:
                content = str(item_dict["text"])
            elif "passage" in item_dict:
                content = str(item_dict["passage"])
            else:
                # 尝试获取第一个字符串字段
                values = list(item_dict.values())
                if values:
                    content = str(values[0])

            documents.append({"content": content, "metadata": item_dict})

        console.print(f"[green]转换了 {len(documents)} 个文档[/green]")
        return documents
