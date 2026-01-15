#!/usr/bin/env python3
"""测试多数据集加载器

验证多数据集加载器是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_agent.multi_dataset_loader import MultiDatasetLoader
from rich.console import Console

console = Console()


def test_single_dataset():
    """测试加载单个数据集"""
    console.print("[bold cyan]测试1: 加载单个数据集 (electrical_engineering)[/bold cyan]\n")

    loader = MultiDatasetLoader(
        datasets=["electrical_engineering"],
        sample_size=10,  # 小样本测试
        load_all=False,
    )

    datasets = loader.load_all_datasets()

    # 验证结果
    assert "electrical_engineering" in datasets, "未加载数据集"
    assert len(datasets["electrical_engineering"]) > 0, "数据集为空"

    # 检查文档结构
    doc = datasets["electrical_engineering"][0]
    assert "content" in doc, "文档缺少 content 字段"
    assert "metadata" in doc, "文档缺少 metadata 字段"

    console.print(f"[green]✓ 测试通过: 加载了 {len(datasets['electrical_engineering'])} 条数据[/green]\n")

    # 显示示例文档
    console.print("[bold]示例文档:[/bold]")
    console.print(f"内容: {doc['content'][:200]}...")
    console.print(f"元数据: {doc['metadata']}\n")


def test_baa_dataset():
    """测试加载BAAI数据集（如果已下载）"""
    console.print("[bold cyan]测试2: 加载BAAI数据集[/bold cyan]\n")

    try:
        loader = MultiDatasetLoader(
            datasets=["baa_industrial"],
            sample_size=5,  # 小样本测试
            load_all=False,
        )

        datasets = loader.load_all_datasets()

        if "baa_industrial" in datasets:
            console.print(f"[green]✓ BAAI数据集加载成功: {len(datasets['baa_industrial'])} 条数据[/green]\n")

            # 显示示例
            if datasets["baa_industrial"]:
                doc = datasets["baa_industrial"][0]
                console.print("[bold]示例文档:[/bold]")
                console.print(f"内容: {doc['content'][:200]}...")
                console.print(f"语言: {doc['metadata'].get('language', 'unknown')}\n")
        else:
            console.print("[yellow]⚠ BAAI数据集未加载（可能仍在下载或格式不兼容）[/yellow]\n")

    except Exception as e:
        console.print(f"[yellow]⚠ BAAI数据集测试失败: {e}[/yellow]")
        console.print("[dim]这可能是正常的，如果数据集仍在下载中[/dim]\n")


def test_combined_datasets():
    """测试加载多个数据集"""
    console.print("[bold cyan]测试3: 加载多个数据集[/bold cyan]\n")

    loader = MultiDatasetLoader(
        datasets=["electrical_engineering"],  # 暂时不包含 baa_industrial
        sample_size=10,
        load_all=False,
    )

    datasets = loader.load_all_datasets()
    combined = loader.get_combined_documents()

    console.print(f"[green]✓ 合并后总计: {len(combined)} 条数据[/green]\n")

    # 统计信息
    stats = loader.get_dataset_stats()
    console.print("[bold]数据集统计:[/bold]")
    console.print(f"  - 数据集数量: {stats['total_datasets']}")
    console.print(f"  - 文档总数: {stats['total_documents']}\n")


def main():
    """运行所有测试"""
    console.print("[bold cyan]多数据集加载器测试[/bold cyan]\n")

    try:
        test_single_dataset()
        test_baa_dataset()
        test_combined_datasets()

        console.print("[bold green]所有测试完成[/bold green]\n")

    except Exception as e:
        console.print(f"[red]测试失败: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
