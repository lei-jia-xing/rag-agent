# rag_agent/graphs/ - LangGraph Workflows

## OVERVIEW

StateGraph workflow definitions. Wires agent nodes into executable graphs.

## STRUCTURE

```
graphs/
├── __init__.py           # Exports graph builders
├── main_graph.py         # Root router graph
├── diagnosis_graph.py    # 4-node diagnosis workflow
└── qa_graph.py           # 2-node QA workflow
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new workflow | Create new file, export from __init__.py |
| Modify routing | main_graph.py - add_conditional_edges |
| Add diagnosis step | diagnosis_graph.py |
| Add QA step | qa_graph.py |

## WORKFLOW PATTERNS

```python
# Standard graph pattern
from langgraph.graph import StateGraph, START, END

def build_graph():
    workflow = StateGraph(StateType)
    workflow.add_node("node_name", node_function)
    workflow.add_edge(START, "node_name")
    workflow.add_edge("node_name", END)
    return workflow.compile()
```

## CONVENTIONS

- **Builder functions**: `build_*_graph()` returns compiled graph
- **Visualization**: `visualize_*_graph()` saves PNG (requires graphviz)
- **Invoke helpers**: `*_invoke()` for direct graph execution
- **Conditional edges**: Use `add_conditional_edges(node, condition_fn, mapping)`

## KEY FUNCTIONS

| Function | File | Returns |
|----------|------|---------|
| `build_main_graph` | main_graph.py | CompiledGraph |
| `build_diagnosis_graph` | diagnosis_graph.py | CompiledGraph |
| `build_qa_graph` | qa_graph.py | CompiledGraph |
| `main_agent_invoke` | main_graph.py | Final state dict |
