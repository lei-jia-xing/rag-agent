# rag_agent/ - Main Package

## OVERVIEW

Core RAG application: CLI entry, engine, agents, graphs, retrieval, memory.

## STRUCTURE

```
rag_agent/
├── cli.py              # Typer CLI + prompt_toolkit REPL
├── rag_engine.py       # Core engine (retrieval + generation)
├── config.py           # Pydantic-style config from .env
├── data_loader.py      # Single dataset loader
├── multi_dataset_loader.py  # Multi-dataset support
├── agents/             # LangGraph nodes
├── graphs/             # LangGraph workflows
├── retrieval/          # Retrieval strategies
├── memory/             # Memory systems
├── schemas/            # State definitions
├── apps/               # High-level interfaces
└── mcp/                # MCP clients
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add CLI command | cli.py - @app.command() decorator |
| Add env variable | config.py - add to config object |
| Modify prompt templates | rag_engine.py (inline) - TODO: externalize |
| Add dataset source | multi_dataset_loader.py |

## CONVENTIONS

- **Entry point**: `__init__.py:main()` → `cli.py:app()`
- **Shared engine**: `InteractiveSession` shares one `RAGEngine` across mode switches
- **State passing**: TypedDict from `schemas/state.py`, not classes
- **Error handling**: Try/except with Rich console output

## ANTI-PATTERNS

- **rag_engine.py is 466 lines**: Needs splitting (marked in README TODO)
- **Inline prompts**: All prompt templates hardcoded, should externalize to `prompts/`
- **Global engine in router.py**: `_engine` singleton - acceptable but not ideal
