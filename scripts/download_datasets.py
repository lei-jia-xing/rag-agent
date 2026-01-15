#!/usr/bin/env python3
"""下载和准备数据集的脚本

使用方法:
    python scripts/download_datasets.py --dataset baa_corpus
    python scripts/download_datasets.py --dataset etdataset
    python scripts/download_datasets.py --dataset all
"""

import asyncio
import sys
from pathlib import Path

import typer
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="下载和准备电气工程问答数据集")
console = Console()


@app.command()
def list_datasets():
    """列出所有可用的数据集"""
    console.print("[bold cyan]可用的数据集[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("数据集", width=50)
    table.add_column("语言", width=8)
    table.add_column("大小", width=10)
    table.add_column("优先级", width=8)

    datasets = [
        ("BAAI/IndustryCorpus2_electric_power_energy", "中文", "大", "⭐⭐⭐⭐⭐"),
        ("ETDataset (GitHub)", "中文", "中", "⭐⭐⭐⭐⭐"),
        ("STEM-AI-mtl/Electrical-engineering", "英文", "中", "⭐⭐⭐⭐"),
        ("CMMLU电气工程", "中文", "小", "⭐⭐⭐⭐"),
        ("电力变压器油色谱", "中文", "小", "⭐⭐⭐"),
    ]

    for name, lang, size, priority in datasets:
        table.add_row(name, lang, size, priority)

    console.print(table)
    console.print("\n[yellow]提示:[/yellow] 使用 [green]python scripts/download_datasets.py --dataset <name>[/green] 下载数据集")


@app.command()
def download(
    dataset: str = typer.Argument(..., help="数据集名称 (baa_corpus, etdataset, all)"),
    output_dir: str = typer.Option("data", help="输出目录"),
):
    """下载数据集"""
    console.print(f"[bold cyan]下载数据集: {dataset}[/bold cyan]\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset == "baa_corpus":
        download_baa_corpus(output_path)
    elif dataset == "etdataset":
        download_etdataset(output_path)
    elif dataset == "all":
        download_all(output_path)
    else:
        console.print(f"[red]✗ 未知的数据集: {dataset}[/red]")
        console.print("\n可用的数据集:")
        console.print("  - baa_corpus")
        console.print("  - etdataset")
        console.print("  - all")
        sys.exit(1)


def download_baa_corpus(output_path: Path):
    """下载BAAI工业语料库"""
    console.print("[yellow]正在下载 BAAI/IndustryCorpus2_electric_power_energy...[/yellow]")

    try:
        dataset = load_dataset("BAAI/IndustryCorpus2_electric_power_energy")

        console.print(f"[green]✓ 数据集加载成功[/green]")
        console.print(f"  - Split: {list(dataset.keys())}")

        # 保存到文件
        if "train" in dataset:
            train_path = output_path / "baa_electrical_train.jsonl"
            dataset["train"].to_json(train_path)
            console.print(f"  - 训练集保存到: {train_path}")

        console.print(f"\n[bold green]✓ BAAI工业语料库下载完成[/bold green]")

    except Exception as e:
        console.print(f"[red]✗ 下载失败: {e}[/red]")
        console.print("\n[yellow]提示:[/yellow] 请确保安装了datasets库:")
        console.print("  pip install datasets")


def download_etdataset(output_path: Path):
    """下载ETDataset"""
    console.print("[yellow]正在下载 ETDataset...[/yellow]")

    try:
        import subprocess

        # 克隆GitHub仓库
        repo_url = "https://github.com/zhouhaoyi/ETDataset.git"
        clone_path = output_path / "ETDataset"

        console.print(f"[dim]克隆仓库: {repo_url}[/dim]")

        subprocess.run(
            ["git", "clone", repo_url, str(clone_path)],
            check=True
        )

        console.print(f"[green]✓ ETDataset下载完成: {clone_path}[/green]")

    except Exception as e:
        console.print(f"[red]✗ 下载失败: {e}[/red]")
        console.print("\n[yellow]提示:[/yellow] 请确保安装了git:")
        console.print("  sudo apt-get install git")


def download_all(output_path: Path):
    """下载所有推荐数据集"""
    console.print("[bold cyan]下载所有推荐数据集[/bold cyan]\n")

    # 1. BAAI工业语料库
    console.print("[yellow]1/3 下载 BAAI工业语料库...[/yellow]")
    download_baa_corpus(output_path)
    console.print()

    # 2. ETDataset
    console.print("[yellow]2/3 下载 ETDataset...[/yellow]")
    download_etdataset(output_path)
    console.print()

    # 3. 当前数据集（已在使用）
    console.print("[yellow]3/3 ✓ STEM-AI-mtl/Electrical-engineering (已使用)[/yellow]")

    console.print("\n[bold green]✓ 所有数据集准备完成[/bold green]")
    console.print(f"\n数据保存位置: {output_path.absolute()}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 没有参数时，显示列表
        list_datasets()
    else:
        app()
