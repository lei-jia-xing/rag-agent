<div align="center">

# RAG Agent

**电气工程设备问答智能体**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/LangChain-1.1.0-orange.svg)](https://www.langchain.com/)

基于 LangChain 和 Hugging Face 构建的 RAG 系统，专注于电气工程和工业设备领域的智能问答

</div>

---

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

---

## 🎨 LaTeX 报告生成

**内置专业模板，支持一键生成设备诊断报告**

本项目集成了 LaTeX MCP 服务，支持专业报告生成。

### 功能特性

- **模板内置**：模板直接集成在 LaTeX MCP 中，无需额外服务
- **简单调用**：单次 MCP 调用完成渲染和编译
- **JSON Schema验证**：确保字段数据符合模板要求
- **Jinja2渲染**：支持复杂的模板逻辑和条件判断
- **专业模板**：内置设备健康诊断报告（32个字段，9大章节）

### 内置模板

#### 设备健康诊断报告 (device_diagnosis)

专业详细技术报告风格的设备诊断模板：

```
封面页
  ├── 报告标题、编号、日期
  ├── 设备信息表格
  └── 健康评分可视化（TikZ圆形仪表盘）

执行摘要
  └── 诊断概要

设备基本信息
  ├── 技术参数
  ├── 运行环境
  └── 历史维护记录

健康状态评估
  └── 健康评分表格（动态颜色）

监测数据分析
  ├── 数据汇总
  ├── 关键指标分析
  ├── 趋势分析
  └── 异常检测

故障诊断详情
  ├── 故障现象
  ├── 原因分析
  └── 故障定位

风险评估
  ├── 当前风险
  ├── 潜在风险
  └── 风险控制建议

维护建议
  ├── 紧急处理措施（高亮框）
  ├── 维护计划
  └── 备件建议

诊断方法与标准
  └── 相关标准引用

结论与建议
```

### 使用示例

```python
from rag_agent.mcp.latex_client import generate_diagnosis_report

# 生成设备诊断报告
result = generate_diagnosis_report(
    data={
        "title": "设备健康诊断报告",
        "device_name": "变压器",
        "health_score": 85,
        "health_status": "正常",
        "risk_level": "低",
        # ... 其他28个字段
    },
    template_id="device_diagnosis"
)

if result["success"]:
    print(f"报告已生成: {result['output_path']}")
```

---

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
| **MCP模板** | Jinja2 + JSON Schema | 模板填充服务 |

---

## 🔧 开发指南

### 项目结构

```
rag_agent/
├── rag_agent/              # 主包
│   ├── __init__.py         # 入口点
│   ├── cli.py              # CLI（Typer + prompt_toolkit）
│   ├── config.py           # 配置管理
│   ├── data_loader.py      # 数据集加载
│   ├── rag_engine.py       # RAG 核心引擎
│   ├── mcp/                # MCP 客户端
│   │   └── latex_client.py # LaTeX MCP 客户端
│   └── apps/               # 应用层
│       ├── base.py         # 基础应用抽象类
│       ├── qa_app.py       # 问答应用
│       └── report_app.py   # 报告生成应用
├── mcp-latex/              # LaTeX MCP 服务（含内置模板）
│   ├── templates/          # 报告模板
│   │   └── device_diagnosis/
│   ├── mcp_latex_tool.py   # MCP 服务器
│   ├── template_manager.py # Jinja2 模板管理器
│   └── Dockerfile          # 容器化部署
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

---

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

---

## 📋 TODO

- [x] 创建 LaTeX MCP 服务（内置模板）
- [x] 实现设备健康诊断报告模板（32字段，9章节）
- [x] 集成到 report_app.py
- [ ] 评估当前检索效果，建立基准线
- [ ] 尝试更好的中文 Embedding 模型（bge-large-zh、m3e-large）
- [ ] 实现混合检索（BM25 + 向量）
- [ ] 添加 Reranker 重排序
- [ ] 扩充高质量数据集

---

## 许可证

MIT License
