# rag_agent/retrieval/ - Retrieval Subsystem

## OVERVIEW

Hybrid retrieval: BM25 + vector search + reranking + query expansion.

## STRUCTURE

```
retrieval/
├── __init__.py              # Exports all retrievers
├── base_enhanced_retriever.py  # Abstract base with caching
├── bm25_retriever.py        # Sparse retrieval (jieba tokenization)
├── hybrid_retriever.py      # Combines BM25 + vector
├── query_expander.py        # LLM-based query rewriting
└── reranker.py              # Cross-encoder reranking
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add retrieval strategy | Create new file, inherit BaseEnhancedRetriever |
| Modify BM25 tokenization | bm25_retriever.py - uses jieba for Chinese |
| Change reranking model | reranker.py |
| Adjust retrieval weights | hybrid_retriever.py |

## CONVENTIONS

- **Base class**: All retrievers inherit `BaseEnhancedRetriever`
- **Caching**: Built into base class
- **Chinese support**: jieba tokenizer for BM25

## ANTI-PATTERNS

- **TODO at base_enhanced_retriever.py:148**: `# TODO: 实现真正的异步支持` - async methods are sync wrappers
- **Embedded tests**: Each file has `test_*` function at bottom
