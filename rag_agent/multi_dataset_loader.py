"""多数据集加载模块

支持加载和组合多个数据集，包括:
- STEM-AI-mtl/Electrical-engineering (当前使用的英文数据集)
- BAAI/IndustryCorpus2_electric_power_energy (中文工业语料库)
- ETDataset (变压器专业数据集)
- IEEE/Mendeley 专业故障诊断数据集
- 本地数据文件（CSV/JSON/JSONL）
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# 数据质量过滤配置
QUALITY_FILTER_CONFIG = {
    "min_content_length": 50,  # 最小内容长度
    "max_content_length": 50000,  # 最大内容长度
    "min_quality_score": 3.5,  # BAAI 数据集的最低质量分
    "max_perplexity": 5000,  # 最大困惑度
    "remove_duplicates": True,  # 去重
}


class MultiDatasetLoader:
    """多数据集加载器：从多个来源加载数据集并合并为统一的文档格式"""

    DATASET_CONFIGS = {
        # === Tier 1: 核心数据集 ===
        "electrical_engineering": {
            "name": "STEM-AI-mtl/Electrical-engineering",
            "split": "train",
            "format": "qa",
            "language": "en",
            "priority": 5,
            "description": "电气工程问答数据集（英文，~1.1k条）",
        },
        "baai_power_energy": {
            "name": "BAAI/IndustryCorpus2_electric_power_energy",
            "split": "train",
            "format": "baai_text",
            "language": "zh",
            "priority": 5,
            "has_quality_score": True,
            "description": "电力能源工业语料库（中文，~1040万条）",
        },
        # === Tier 2: 专业故障诊断数据集 ===
        "transformer_dga": {
            "source": "local",
            "path": "data/transformer_dga.json",
            "format": "dga_fault",
            "language": "en",
            "priority": 4,
            "description": "变压器油色谱故障诊断数据",
        },
        # === Tier 3: 通用电气知识 ===
        "cmmlu_electrical": {
            "name": "haonan-li/cmmlu",
            "subset": "electrical_engineer",
            "split": "test",
            "format": "mcq",
            "language": "zh",
            "priority": 3,
            "description": "中文电气工程师考试题（多选）",
        },
        # === 保留旧的键名兼容性 ===
        "baa_industrial": {
            "alias": "baai_power_energy",
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
            resolved_config = self._resolve_alias(config)
            console.print(f"[cyan]加载 {dataset_key}: {resolved_config.get('description', '')}[/cyan]")

            try:
                documents = self._load_single_dataset(dataset_key, config)
                self.loaded_datasets[dataset_key] = documents
                console.print(f"[green]✓ {dataset_key} 加载完成: {len(documents)} 条数据[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ {dataset_key} 加载失败: {e}[/red]\n")
                logger.error(f"Failed to load dataset {dataset_key}: {e}", exc_info=True)

        self._print_summary()
        return self.loaded_datasets

    def _resolve_alias(self, config: dict[str, Any]) -> dict[str, Any]:
        if "alias" not in config:
            return config
        alias_key = config["alias"]
        if alias_key not in self.DATASET_CONFIGS:
            raise ValueError(f"别名 {alias_key} 指向的数据集不存在")
        console.print(f"  [dim]（别名 → {alias_key}）[/dim]")
        return self.DATASET_CONFIGS[alias_key]

    def _load_huggingface_dataset(self, config: dict[str, Any]) -> Dataset:
        dataset_name = config["name"]
        split = config["split"]
        subset = config.get("subset")

        if subset:
            dataset_raw = load_dataset(dataset_name, subset, split=split)
        else:
            dataset_raw = load_dataset(dataset_name, split=split)

        if not isinstance(dataset_raw, Dataset):
            raise TypeError(f"期望 Dataset 类型，但获得 {type(dataset_raw)}")
        return dataset_raw

    def _apply_sampling(self, dataset: Dataset) -> Dataset:
        if not self.load_all and len(dataset) > self.sample_size:
            return dataset.shuffle(seed=42).select(range(self.sample_size))
        return dataset

    def _convert_by_format(self, dataset: Dataset, config: dict[str, Any]) -> list[dict[str, Any]]:
        format_type = config["format"]
        converters = {
            "qa": self._convert_qa_format,
            "text": self._convert_text_format,
            "baai_text": self._convert_baai_text_format,
            "mcq": self._convert_mcq_format,
        }
        converter = converters.get(format_type, self._convert_generic_format)
        return converter(dataset, config)

    def _load_single_dataset(self, dataset_key: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        config = self._resolve_alias(config)

        try:
            if config.get("source") == "local":
                return self._load_local_dataset(config)

            dataset_raw = self._load_huggingface_dataset(config)
            dataset_raw = self._apply_sampling(dataset_raw)
            console.print(f"  加载了 {len(dataset_raw)} 条数据")

            documents = self._convert_by_format(dataset_raw, config)

            if config.get("has_quality_score"):
                documents = self._filter_by_quality(documents)

            return documents

        except Exception as e:
            console.print(f"[red]加载失败: {e}[/red]")
            raise

    def _convert_qa_format(self, dataset: Dataset, config: dict[str, Any]) -> list[dict[str, Any]]:
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

    def _convert_text_format(self, dataset: Dataset, config: dict[str, Any]) -> list[dict[str, Any]]:
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

    def _convert_generic_format(self, dataset: Dataset, config: dict[str, Any]) -> list[dict[str, Any]]:
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

    def _convert_baai_text_format(self, dataset: Dataset, config: dict[str, Any]) -> list[dict[str, Any]]:
        """BAAI 工业语料库格式（包含 quality_score, perplexity 等字段）"""
        documents: list[dict[str, Any]] = []

        for item in dataset:
            item_dict = dict(item)
            content = str(item_dict.get("text", ""))

            if not content or len(content) < QUALITY_FILTER_CONFIG["min_content_length"]:
                continue

            metadata = {
                "dataset": config["name"],
                "language": config["language"],
                "format": "baai_text",
                "quality_score": item_dict.get("quality_score"),
                "perplexity": item_dict.get("perplexity"),
                "source": item_dict.get("source", ""),
            }
            documents.append({"content": content, "metadata": metadata})

        console.print(f"  转换了 {len(documents)} 个 BAAI 文档")
        return documents

    def _convert_mcq_format(self, dataset: Dataset, config: dict[str, Any]) -> list[dict[str, Any]]:
        """多选题格式（CMMLU 等考试数据集）"""
        documents: list[dict[str, Any]] = []

        for item in dataset:
            item_dict = dict(item)
            question = str(item_dict.get("Question", item_dict.get("question", "")))
            choices = []
            for key in ["A", "B", "C", "D"]:
                if key in item_dict:
                    choices.append(f"{key}. {item_dict[key]}")
            answer = str(item_dict.get("Answer", item_dict.get("answer", "")))

            content = f"问题: {question}\n选项:\n" + "\n".join(choices) + f"\n答案: {answer}"

            metadata = {
                "dataset": config["name"],
                "language": config["language"],
                "format": "mcq",
                "answer": answer,
            }
            documents.append({"content": content, "metadata": metadata})

        console.print(f"  转换了 {len(documents)} 个多选题文档")
        return documents

    def _convert_dga_fault_format(self, data: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        """DGA 油色谱故障诊断数据格式"""
        documents: list[dict[str, Any]] = []

        for item in data:
            gas_values = []
            for gas in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]:
                if gas in item:
                    gas_values.append(f"{gas}: {item[gas]} ppm")

            fault_type = item.get("fault_type", item.get("diagnosis", "未知"))
            content = "油色谱分析数据:\n" + "\n".join(gas_values) + f"\n\n故障诊断: {fault_type}"

            if "description" in item:
                content += f"\n说明: {item['description']}"

            metadata = {
                "dataset": config.get("path", "local"),
                "language": config["language"],
                "format": "dga_fault",
                "fault_type": fault_type,
            }
            documents.append({"content": content, "metadata": metadata})

        console.print(f"  转换了 {len(documents)} 个 DGA 故障文档")
        return documents

    def _load_local_dataset(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """加载本地数据文件（JSON/JSONL/CSV）"""
        file_path = Path(config["path"])
        if not file_path.exists():
            console.print(f"[yellow]本地文件不存在: {file_path}，跳过[/yellow]")
            return []

        console.print(f"  从本地加载: {file_path}")

        if file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        elif file_path.suffix == ".jsonl":
            data = []
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"不支持的本地文件格式: {file_path.suffix}")

        if not self.load_all and len(data) > self.sample_size:
            import random

            random.seed(42)
            data = random.sample(data, self.sample_size)

        console.print(f"  加载了 {len(data)} 条本地数据")

        format_type = config["format"]
        if format_type == "dga_fault":
            return self._convert_dga_fault_format(data, config)

        return self._convert_local_generic(data, config)

    def _convert_local_generic(self, data: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        """通用本地数据转换"""
        documents: list[dict[str, Any]] = []
        for item in data:
            content = str(item.get("content", item.get("text", "")))
            if not content:
                for v in item.values():
                    if isinstance(v, str) and len(v) > 10:
                        content = v
                        break
            if content:
                metadata = {**item, "dataset": config.get("path", "local"), "language": config["language"]}
                documents.append({"content": content, "metadata": metadata})
        console.print(f"  转换了 {len(documents)} 个本地文档")
        return documents

    def _filter_by_quality(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """根据质量分数过滤文档"""
        original_count = len(documents)
        filtered = []

        for doc in documents:
            metadata = doc.get("metadata", {})
            quality_score = metadata.get("quality_score")
            perplexity = metadata.get("perplexity")

            if quality_score is not None and quality_score < QUALITY_FILTER_CONFIG["min_quality_score"]:
                continue
            if perplexity is not None and perplexity > QUALITY_FILTER_CONFIG["max_perplexity"]:
                continue

            content_len = len(doc.get("content", ""))
            if content_len < QUALITY_FILTER_CONFIG["min_content_length"]:
                continue
            if content_len > QUALITY_FILTER_CONFIG["max_content_length"]:
                continue

            filtered.append(doc)

        removed = original_count - len(filtered)
        if removed > 0:
            console.print(f"  [dim]质量过滤: 移除 {removed} 条低质量数据[/dim]")

        return filtered

    def _compute_content_hash(self, content: str) -> str:
        import hashlib

        normalized = " ".join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _deduplicate(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not QUALITY_FILTER_CONFIG["remove_duplicates"]:
            return documents

        seen_hashes: set[str] = set()
        unique_docs: list[dict[str, Any]] = []

        for doc in documents:
            content = doc.get("content", "")
            content_hash = self._compute_content_hash(content)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)

        removed = len(documents) - len(unique_docs)
        if removed > 0:
            console.print(f"[dim]去重: 移除 {removed} 条重复数据[/dim]")

        return unique_docs

    def get_combined_documents(self) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []

        for _dataset_name, documents in self.loaded_datasets.items():
            combined.extend(documents)

        combined = self._deduplicate(combined)

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
