# rag_agent/memory/ - Memory Systems

## OVERVIEW

Short-term (conversation) and long-term (vector) memory for RAG agents.

## STRUCTURE

```
memory/
├── __init__.py       # Exports memory classes
├── short_term.py     # Conversation history (in-memory)
└── long_term.py      # Vector store memory (FAISS)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Change history window | short_term.py |
| Add memory persistence | short_term.py (currently in-memory only) |
| Modify vector memory | long_term.py |

## CONVENTIONS

- **Short-term**: Session-based, cleared on restart
- **Long-term**: Persisted to `.vectorstore/`

## ANTI-PATTERNS

- **long_term.py warning**: Logs warning if vector store not initialized - silent empty results
