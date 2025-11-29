<div align="center">

# RAG Agent

**电气工程设备问答智能体**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/LangChain-1.1.0-orange.svg)](https://www.langchain.com/)

基于 LangChain 和 Hugging Face 构建的 RAG 系统，专注于电气工程和工业设备领域的智能问答

</div>

---

## 系统架构

<div align="center">
  <img src="./Archtecture.png" alt="RAG Agent 系统架构图" width="100%">
</div>

## 快速开始

### 安装依赖

```bash
uv sync
```

### 配置 API

编辑 `.env` 文件：

```env
OPENAI_API_KEY=sk-xxx                              # Silicon Flow API 密钥
OPENAI_API_BASE=https://api.siliconflow.cn/v1     # API 端点
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct                # LLM 模型
DATASET_NAME=STEM-AI-mtl/Electrical-engineering   # 数据集
DATASET_SAMPLE_SIZE=500                            # 采样数量
```

### 启动应用

```bash
uv run rag-agent
```

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| **数据源** | Hugging Face Datasets | 电气工程问答数据集（1131条） |
| **嵌入模型** | Sentence Transformers | paraphrase-multilingual-MiniLM-L12-v2 |
| **向量数据库** | FAISS | Facebook AI 相似度搜索 |
| **LLM** | Qwen2.5-7B-Instruct | 通过 Silicon Flow API 调用 |
| **框架** | LangChain 1.1.0 | RAG 编排与链式调用 |
| **CLI** | prompt_toolkit + Rich | 交互式命令行界面 |
| **包管理** | uv | 快速依赖管理 |
| **代码质量** | Ruff + Pyright | 代码检查与类型检查 |

## 开发指南

### 代码质量检查

```bash
uv run ruff format .

uv run ruff check .

uv run pyright
```

## ⚙️ 配置选项

### 数据集选择

```env
# 电气工程（默认）
DATASET_NAME=STEM-AI-mtl/Electrical-engineering

# 其他英文数据集
DATASET_NAME=squad                    # 斯坦福问答
DATASET_NAME=natural_questions        # Google 自然问题

# 中文数据集
DATASET_NAME=cmrc2018                 # 中文阅读理解
```
