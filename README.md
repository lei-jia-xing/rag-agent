<div align="center">

# RAG Agent

**电气工程设备问答智能体系统**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/LangChain-1.1.0-orange.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3.0-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%2B%20pyright-success.svg)](https://github.com/astral-sh/ruff)

基于 **LangChain + LangGraph** 构建的多智能体 RAG 系统，专注于电气工程和工业设备领域的智能问答与设备诊断

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

# 数据集配置（可选）
# 单一数据集模式
DATASET_NAME=STEM-AI-mtl/Electrical-engineering   # 数据集

# 多数据集模式（推荐）- 同时加载多个数据集
MULTI_DATASETS=all                                # all 或指定数据集
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

### 📚 文档

完整的文档请查看 [docs/INDEX.md](docs/INDEX.md)：
- [快速开始指南](docs/QUICKSTART.md) - 开发环境配置
- [架构设计文档](docs/ARCHITECTURE.md) - LangGraph 多智能体架构
- [代码质量报告](docs/CODE_QUALITY_REPORT.md) - Ruff + Pyright 优化
- [实施进度追踪](docs/IMPLEMENTATION_STATUS.md) - 功能完成情况
- [**智能体增强路线图**](docs/AGENT_ENHANCEMENT_ROADMAP.md) - v2.0 增强计划 ⭐
- [**增强 TODO**](docs/ENHANCEMENT_TODO.md) - 快速任务列表 ⭐
- [**优质数据集推荐**](docs/DATASET_QUICKSTART.md) - 数据集快速参考 🆕
- [**数据集详细指南**](docs/DATASET_RECOMMENDATIONS.md) - 数据集使用指南 🆕

### 项目结构

```
rag_agent/
├── rag_agent/              # 主包
│   ├── __init__.py         # 入口点
│   ├── cli.py              # CLI（Typer + prompt_toolkit）
│   ├── config.py           # 配置管理
│   ├── data_loader.py      # 数据集加载
│   ├── rag_engine.py       # RAG 核心引擎
│   │
│   ├── agents/             # LangGraph 智能体节点
│   │   ├── router.py       # 路由智能体（意图分类）
│   │   ├── diagnosis_agent.py  # 诊断智能体（4节点）
│   │   └── qa_agent.py     # 问答智能体（2节点）
│   │
│   ├── graphs/             # LangGraph 工作流图
│   │   ├── main_graph.py   # 主路由图
│   │   ├── diagnosis_graph.py  # 诊断流程图
│   │   └── qa_graph.py     # 问答流程图
│   │
│   ├── schemas/            # 状态定义
│   │   └── state.py        # AgentState, DiagnosisState, QAState
│   │
│   ├── memory/             # 记忆系统
│   │   ├── short_term.py   # 短期记忆（对话历史）
│   │   └── long_term.py    # 长期记忆（向量存储）
│   │
│   ├── tools/              # 工具函数
│   ├── mcp/                # MCP 客户端
│   │   └── latex_client.py # LaTeX MCP 客户端
│   │
│   └── apps/               # 应用层
│       ├── base.py         # 基础应用抽象类
│       ├── qa_app.py       # 问答应用
│       └── report_app.py   # 报告生成应用
│
├── mcp-latex/              # LaTeX MCP 服务（含内置模板）
│   ├── templates/          # 报告模板
│   │   └── device_diagnosis/
│   ├── mcp_latex_tool.py   # MCP 服务器
│   ├── template_manager.py # Jinja2 模板管理器
│   └── Dockerfile          # 容器化部署
│
├── docs/                   # 项目文档
│   ├── INDEX.md            # 文档索引
│   ├── ARCHITECTURE.md     # 架构设计
│   ├── QUICKSTART.md       # 快速开始
│   ├── CODE_QUALITY_REPORT.md  # 代码质量报告
│   └── IMPLEMENTATION_STATUS.md # 实施状态
│
├── pyproject.toml          # 项目配置
├── pyrightconfig.json      # Pyright 配置
├── .env.example            # 环境变量模板
└── README.md               # 文档
```

### 架构概览

本项目使用 **LangGraph** 构建多智能体系统：

```
┌─────────────────────────────────────────┐
│          Main Agent (Router)            │
├─────────────────────────────────────────┤
│  Query → Intent Classification          │
│           ↓                             │
│  ┌─────────┼─────────┐                 │
│  ↓         ↓         ↓                 │
│ Diagnosis  QA    Reasoning             │
│  Agent    Agent   Agent                 │
└─────────────────────────────────────────┘
```

**智能体类型**：
- **Router Agent**: 意图识别，路由到合适的子智能体
- **Diagnosis Agent**: 设备诊断（检索 → 分析 → 字段 → 报告）
- **QA Agent**: 智能问答（检索 → 合成）
- **Reasoning Agent**: 复杂推理（待实现）

详见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)

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

### ✅ 已完成

#### LangGraph 多智能体架构
- [x] 设计并实现 Router Agent（意图分类）
- [x] 实现 Diagnosis Agent（4节点：检索→分析→字段→报告）
- [x] 实现 QA Agent（2节点：检索→合成）
- [x] 构建主路由图（Main Graph）
- [x] 实现记忆系统（短期对话 + 长期向量存储）
- [x] 完成端到端测试

#### 代码质量优化
- [x] 修复 25 个 Ruff 代码质量问题
- [x] 更新 Pyright 类型注解（LangGraph 类型安全）
- [x] 创建 pyrightconfig.json 配置
- [x] 编写代码质量报告文档

#### LaTeX MCP 集成
- [x] 创建 LaTeX MCP 服务（内置模板）
- [x] 实现设备健康诊断报告模板（32字段，9章节）
- [x] 集成到 report_app.py
- [x] 报告输出到 `reports/` 目录
- [x] 模板系统迁移到 LaTeX MCP

#### 多数据集支持 ⭐ NEW
- [x] 实现 MultiDatasetLoader（支持多个数据集同时加载）
- [x] 更新配置系统支持 MULTI_DATASETS
- [x] 集成到 CLI build 命令
- [x] 添加测试脚本和验证
- [x] 创建多数据集集成文档
- [ ] ⏳ BAAI数据集下载（等待网络连接）
- [ ] ⏳ 向量数据库重建（等待网络连接）

### 🔥 高优先级（架构重构）

#### 测试与质量保证
- [ ] 添加单元测试（目标覆盖率 80%+）
  - [ ] `rag_engine.py` 测试
  - [ ] `report_app.py` 测试
  - [ ] `latex_client.py` 测试
- [ ] 添加集成测试
- [ ] 添加端到端测试
- [ ] 添加性能测试（检索、生成）

#### 代码重构
- [ ] 拆分 `RAGEngine`（466 行 → 多个服务）
  - [ ] `RetrievalService` - 检索服务
  - [ ] `GenerationService` - 生成服务
  - [ ] `ReportService` - 报告服务
- [ ] 统一错误处理机制
  - [ ] 定义自定义异常类
  - [ ] 标准化错误响应
  - [ ] 添加重试机制
- [ ] 配置管理重构
  - [ ] 使用 Pydantic Settings
  - [ ] 支持环境变量层次
  - [ ] 添加配置验证

#### Prompt 管理
- [ ] Prompt 模板外部化
  - [ ] 创建 `prompts/` 目录
  - [ ] 支持版本管理
  - [ ] 支持热更新
- [ ] 实现提示词测试框架
- [ ] 添加提示词性能对比

### ⚡ 中优先级（功能增强）

#### 向量存储优化
- [ ] 实现向量存储抽象层
  - [ ] 定义 `VectorStore` 接口
  - [ ] 支持多种数据库（FAISS、Chroma、Qdrant）
- [ ] Embedding 模型缓存
  - [ ] 单例模式管理
  - [ ] 支持 GPU 加速
- [ ] 向量数据库版本控制
- [ ] 增量更新机制（无需重建）

#### LangChain/LangGraph 最佳实践
- [x] 实现 Agent 系统（基于 LangGraph）
  - [x] 定义 Tool 接口
  - [x] 实现 Router Agent（意图分类）
  - [x] 实现 Diagnosis Agent（多节点工作流）
  - [x] 实现 QA Agent（问答流程）
- [x] 添加 Memory 管理
  - [x] 对话历史存储（short_term.py）
  - [x] 长期记忆向量存储（long_term.py）
- [ ] 集成 LangSmith 追踪
- [ ] 实现 Output Parsers
- [ ] 添加 Reasoning Agent（复杂推理）

#### 日志与监控
- [ ] 结构化日志系统
  - [ ] 使用 structlog
  - [ ] 日志级别管理
  - [ ] 日志聚合
- [ ] 添加性能监控
  - [ ] 检索延迟
  - [ ] LLM 调用次数
  - [ ] Token 统计
- [ ] 添加业务指标
  - [ ] 查询成功率
  - [ ] 用户满意度

### 🚀 低优先级（长期规划）

#### 性能优化
- [ ] 异步处理支持
  - [ ] 异步检索
  - [ ] 异步 LLM 调用
- [ ] 批量处理优化
- [ ] 添加缓存层（Redis）
- [ ] 流式输出支持

#### 扩展功能
- [ ] 多模态支持（图片、表格）
- [ ] 多语言支持
- [ ] LangServe 部署
- [ ] Web UI 界面
- [ ] API 服务化

#### 检索增强
- [ ] 评估当前检索效果，建立基准线
- [ ] 尝试更好的中文 Embedding 模型（bge-large-zh、m3e-large）
- [ ] 实现混合检索（BM25 + 向量）
- [ ] 添加 Reranker 重排序
- [ ] 扩充高质量数据集
- [ ] 查询改写和优化

---

## 许可证

MIT License
