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

### 整体架构

<div align="center">
  <img src="./Archtecture.png" alt="RAG Agent 系统架构图" width="100%">
</div>

### 模块化设计

```
rag_agent/
├── __init__.py          # 主入口
├── cli.py               # CLI 交互层（Typer + prompt_toolkit）
├── config.py            # 配置管理
├── data_loader.py       # 数据集加载
├── rag_engine.py        # RAG 核心引擎（检索 + 生成）
└── apps/                # 应用层
    ├── base.py          # 基础应用抽象类
    ├── qa_app.py        # 问答应用
    └── report_app.py    # 报告生成应用
```


## 快速开始

### 1. 安装依赖

```bash
uv sync
```


### 2. 配置 API

复制示例配置并编辑 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
OPENAI_API_KEY=sk-xxx                              # Silicon Flow API 密钥
OPENAI_API_BASE=https://api.siliconflow.cn/v1     # API 端点
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct                # LLM 模型
DATASET_NAME=STEM-AI-mtl/Electrical-engineering   # 数据集
```

### 3. 构建向量数据库

首次使用需要预先构建向量数据库（全量加载数据集）

```bash
# 构建向量数据库
uv run rag-agent build
```

### 4. 启动应用

向量数据库构建完成后，后续直接运行：

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

## 🔧 开发指南

### 项目结构

```
rag-agent/
├── rag_agent/              # 主包
│   ├── __init__.py         # 入口点
│   ├── cli.py              # CLI（Typer + prompt_toolkit）
│   ├── config.py           # 配置管理
│   ├── data_loader.py      # 数据集加载
│   ├── rag_engine.py       # RAG 核心引擎
│   └── apps/               # 应用层
│       ├── base.py         # 基础应用抽象类
│       ├── qa_app.py       # 问答应用
│       └── report_app.py   # 报告生成应用
├── pyproject.toml          # 项目配置
├── .env.example            # 环境变量模板
└── README.md               # 文档

```

### 代码质量检查

```bash
# 格式化代码
uv run ruff format .

# 检查代码质量
uv run ruff check .

# 类型检查
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

## 📋 TODO

- [ ] 评估当前检索效果，建立基准线
- [ ] 尝试更好的中文 Embedding 模型（bge-large-zh、m3e-large）
- [ ] 实现混合检索（BM25 + 向量）
- [ ] 添加 Reranker 重排序
- [ ] 扩充高质量数据集
