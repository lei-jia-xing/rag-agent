"""多数据集加载模块

支持加载和组合多个数据集，包括:
- STEM-AI-mtl/Electrical-engineering (当前使用的英文数据集)
- BAAI/IndustryCorpus2_electric_power_energy (中文工业语料库)
- ETDataset (变压器专业数据集)
"""

import logging
from typing import Any

from datasets import Dataset, load_dataset
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    """多数据集加载器

    从多个来源加载数据集并合并为统一的文档格式。
    支持不同格式和语言的数据集。
    """

    # 支持的数据集配置
    DATASET_CONFIGS = {
        "electrical_engineering": {
            "name": "STEM-AI-mtl/Electrical-engineering",
            "split": "train",
            "format": "qa",
            "language": "en",
            "description": "电气工程问答数据集（英文）",
        },
        "baa_industrial": {
            "name": "BAAI/IndustryCorpus2_electric_power_energy",
            "split": "train",
            "format": "text",
            "language": "zh",
            "description": "电力能源工业语料库（中文）",
        },
    }

    def __init__(
        self,
        datasets: list[str] | None = None,
        sample_size: int = 100,
        load_all: bool = False,
    ) -> None:
        """初始化多数据集加载器

        Args:
            datasets: 数据集列表，如 ["electrical_engineering", "baa_industrial"]
                     如果为 None，则加载所有支持的数据集
            sample_size: 每个数据集的采样数量（load_all=False 时生效）
            load_all: 是否加载全量数据集
        """
        self.datasets = datasets or list(self.DATASET_CONFIGS.keys())
        self.sample_size = sample_size
        self.load_all = load_all
        self.loaded_datasets: dict[str, list[dict[str, Any]]] = {}

    def load_all_datasets(self) -> dict[str, list[dict[str, Any]]]:
        """加载所有配置的数据集

        Returns:
            数据集字典 {dataset_name: documents}
        """
        console.print("[cyan]开始加载多数据集...[/cyan]\n")

        for dataset_key in self.datasets:
            if dataset_key not in self.DATASET_CONFIGS:
                console.print(f"[yellow]警告: 未知数据集 {dataset_key}，跳过[/yellow]")
                continue

            config = self.DATASET_CONFIGS[dataset_key]
            console.print(f"[cyan]加载 {dataset_key}: {config['description']}[/cyan]")

            try:
                documents = self._load_single_dataset(dataset_key, config)
                self.loaded_datasets[dataset_key] = documents
                console.print(
                    f"[green]✓ {dataset_key} 加载完成: {len(documents)} 条数据[/green]\n"
                )
            except Exception as e:
                console.print(f"[red]✗ {dataset_key} 加载失败: {e}[/red]\n")
                logger.error(f"Failed to load dataset {dataset_key}: {e}", exc_info=True)

        self._print_summary()
        return self.loaded_datasets

    def _load_single_dataset(
        self, dataset_key: str, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """加载单个数据集

        Args:
            dataset_key: 数据集键名
            config: 数据集配置

        Returns:
            文档列表
        """
        try:
            # 加载原始数据集
            dataset_raw = load_dataset(config["name"], split=config["split"])

            if not isinstance(dataset_raw, Dataset):
                raise TypeError(f"期望 Dataset 类型，但获得 {type(dataset_raw)}")

            # 采样处理
            if not self.load_all and len(dataset_raw) > self.sample_size:
                dataset_raw = dataset_raw.shuffle(seed=42).select(range(self.sample_size))

            console.print(f"  加载了 {len(dataset_raw)} 条数据")

            # 根据格式转换
            format_type = config["format"]
            if format_type == "qa":
                documents = self._convert_qa_format(dataset_raw, config)
            elif format_type == "text":
                documents = self._convert_text_format(dataset_raw, config)
            else:
                documents = self._convert_generic_format(dataset_raw, config)

            return documents

        except Exception as e:
            console.print(f"[red]加载失败: {e}[/red]")
            raise

    def _convert_qa_format(
        self, dataset: Dataset, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """转换问答格式数据集（instruction-input-output）

        Args:
            dataset: HuggingFace 数据集
            config: 数据集配置

        Returns:
            文档列表
        """
        documents: list[dict[str, Any]] = []

        for item in dataset:
            item_dict = dict(item)

            # 提取问答字段
            question = str(item_dict.get("input", ""))
            answer = str(item_dict.get("output", ""))
            instruction = str(item_dict.get("instruction", ""))

            # 组合内容
            content = f"问题: {question}\n\n答案: {answer}"
            if instruction and len(instruction) < 200:
                content = f"说明: {instruction}\n\n{content}"

            # 添加元数据
            metadata = {
                **item_dict,
                "dataset": config["name"],
                "language": config["language"],
                "format": "qa",
            }

            documents.append({"content": content, "metadata": metadata})

        console.print(f"  转换了 {len(documents)} 个问答文档")
        return documents

    def _convert_text_format(
        self, dataset: Dataset, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """转换纯文本格式数据集（text字段）

        Args:
            dataset: HuggingFace 数据集
            config: 数据集配置

        Returns:
            文档列表
        """
        documents: list[dict[str, Any]] = []

        for item in dataset:
            item_dict = dict(item)

            # 尝试多种文本字段
            content = ""
            if "text" in item_dict:
                content = str(item_dict["text"])
            elif "content" in item_dict:
                content = str(item_dict["content"])
            elif "passage" in item_dict:
                content = str(item_dict["passage"])
            else:
                # 获取第一个字符串字段
                for v in item_dict.values():
                    if isinstance(v, str) and len(v) > 10:
                        content = v
                        break

            if not content:
                continue

            # 添加元数据
            metadata = {
                **item_dict,
                "dataset": config["name"],
                "language": config["language"],
                "format": "text",
            }

            documents.append({"content": content, "metadata": metadata})

        console.print(f"  转换了 {len(documents)} 个文本文档")
        return documents

    def _convert_generic_format(
        self, dataset: Dataset, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """转换通用格式数据集

        Args:
            dataset: HuggingFace 数据集
            config: 数据集配置

        Returns:
            文档列表
        """
        documents: list[dict[str, Any]] = []

        for item in dataset:
            item_dict = dict(item)

            # 获取第一个合适的字段作为内容
            content = ""
            for _key, value in item_dict.items():
                if isinstance(value, str) and len(value) > 10:
                    content = value
                    break

            if not content:
                continue

            metadata = {
                **item_dict,
                "dataset": config["name"],
                "language": config["language"],
                "format": "generic",
            }

            documents.append({"content": content, "metadata": metadata})

        console.print(f"  转换了 {len(documents)} 个通用文档")
        return documents

    def get_combined_documents(self) -> list[dict[str, Any]]:
        """获取所有数据集的合并文档列表

        Returns:
            合并后的文档列表
        """
        combined: list[dict[str, Any]] = []

        for _dataset_name, documents in self.loaded_datasets.items():
            combined.extend(documents)

        console.print(f"\n[green]合并后总计: {len(combined)} 条数据[/green]")
        return combined

    def get_dataset_stats(self) -> dict[str, Any]:
        """获取数据集统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_datasets": len(self.loaded_datasets),
            "total_documents": sum(len(docs) for docs in self.loaded_datasets.values()),
            "datasets": {},
        }

        for dataset_name, documents in self.loaded_datasets.items():
            config = self.DATASET_CONFIGS.get(dataset_name, {})
            stats["datasets"][dataset_name] = {
                "count": len(documents),
                "description": config.get("description", ""),
                "language": config.get("language", ""),
            }

        return stats

    def _print_summary(self) -> None:
        """打印数据集加载摘要"""
        console.print("[bold cyan]数据集加载摘要[/bold cyan]\n")

        table_data = []
        for dataset_name, documents in self.loaded_datasets.items():
            config = self.DATASET_CONFIGS.get(dataset_name, {})
            table_data.append(
                [
                    dataset_name,
                    config.get("description", ""),
                    config.get("language", ""),
                    str(len(documents)),
                ]
            )

        if table_data:
            from rich.table import Table

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("数据集", width=25)
            table.add_column("描述", width=35)
            table.add_column("语言", width=8)
            table.add_column("数量", width=10)

            for row in table_data:
                table.add_row(*row)

            console.print(table)

        total = sum(len(docs) for docs in self.loaded_datasets.values())
        console.print(f"\n[bold green]总计: {total} 条数据[/bold green]\n")
