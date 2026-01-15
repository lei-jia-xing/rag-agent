# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-15
**Commit:** 7b959f0
**Branch:** master

## OVERVIEW

Multi-agent RAG system for electrical engineering Q&A and device diagnosis. Built with LangChain + LangGraph, uses FAISS vector store, generates LaTeX PDF reports via MCP service.

## STRUCTURE

```
rag-agent/
├── rag_agent/           # Main Python package (entry: cli.py)
│   ├── agents/          # LangGraph node implementations
│   ├── graphs/          # LangGraph workflow definitions
│   ├── retrieval/       # Hybrid retrieval (BM25 + vector + reranker)
│   ├── memory/          # Short-term (chat) + long-term (vector)
│   ├── schemas/         # TypedDict state definitions
│   ├── apps/            # High-level app interfaces (QA, Report)
│   ├── mcp/             # MCP client for LaTeX service
│   └── rag_engine.py    # Core engine (466 lines - needs refactor)
├── mcp-latex/           # Dockerized LaTeX MCP server
├── scripts/             # Dataset download, multi-dataset test
├── article/             # LaTeX paper artifacts (auto-generated)
└── .vectorstore/        # FAISS index (persisted)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new agent type | `rag_agent/agents/` | Implement node, add to router |
| Add new workflow | `rag_agent/graphs/` | Use StateGraph, wire nodes |
| Modify retrieval | `rag_agent/retrieval/` | See hybrid_retriever.py |
| Change state schema | `rag_agent/schemas/state.py` | TypedDict with Annotated messages |
| Add CLI command | `rag_agent/cli.py` | Typer @app.command() |
| Modify LaTeX template | `mcp-latex/templates/` | Jinja2 + JSON Schema |
| Build vector DB | `uv run rag-agent build` | Required before first use |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `RAGEngine` | Class | rag_engine.py | Core orchestrator - retrieval, generation, report |
| `AgentState` | TypedDict | schemas/state.py | Main state for LangGraph flows |
| `router_node` | Function | agents/router.py | Intent classification (Few-shot + CoT) |
| `build_main_graph` | Function | graphs/main_graph.py | Root workflow builder |
| `InteractiveSession` | Class | cli.py | REPL session handler |

## CONVENTIONS

- **Python 3.12+** required
- **uv** for package management (not pip)
- **Line length**: 120 chars (ruff.toml)
- **Imports**: isort via ruff (I rule)
- **Types**: Pyright strict mode, use TypedDict for state
- **LangGraph state**: Always use `Annotated[list[BaseMessage], add_messages]` for message fields
- **Async**: Agent nodes are async, use `await` for LLM calls

## ANTI-PATTERNS (THIS PROJECT)

- **B008 ignored**: Function calls in defaults OK (Typer/LangChain patterns)
- **NEVER** use `allow_dangerous_deserialization=True` with untrusted sources (currently used in rag_engine.py:59 - local-only)
- **NEVER** modify `article/*.bbl`, `article/*.run.xml` - auto-generated
- **TODO at base_enhanced_retriever.py:148**: Async methods are blocking wrappers, not true async

## UNIQUE STYLES

- **Chinese + English**: Codebase mixes Chinese comments/prompts with English code
- **Embedded tests**: `test_*` functions in implementation files, run via `if __name__ == "__main__"`
- **Few-shot prompts**: Intent classification uses hardcoded examples in `INTENT_EXAMPLES`
- **MCP over Docker**: LaTeX compilation isolated in container, not installed locally

## COMMANDS

```bash
# Install dependencies
uv sync

# Build vector database (REQUIRED first time)
uv run rag-agent build

# Run interactive QA
uv run rag-agent

# Generate diagnosis report
uv run rag-agent report --format diagnosis "变压器"

# Code quality
uv run ruff format .
uv run ruff check .
uv run pyright
```

## NOTES

- **Silicon Flow API**: Uses OpenAI-compatible endpoint at api.siliconflow.cn (see .env.example)
- **Embedding model**: paraphrase-multilingual-MiniLM-L12-v2 (local, no API)
- **Vector store path**: `.vectorstore/` - gitignored, must rebuild on fresh clone
- **No formal test suite**: pytest not configured, tests are embedded (see README TODO)
- **RAGEngine refactor planned**: Split into RetrievalService, GenerationService, ReportService
