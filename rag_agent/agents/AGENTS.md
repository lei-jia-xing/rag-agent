# rag_agent/agents/ - LangGraph Nodes

## OVERVIEW

Agent node implementations for LangGraph workflows. Each agent is a function that takes state and returns state updates.

## STRUCTURE

```
agents/
├── __init__.py          # Exports agent functions
├── router.py            # Intent classification (Few-shot + CoT)
├── diagnosis_agent.py   # Device diagnosis (4 nodes)
└── qa_agent.py          # Q&A (2 nodes)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add intent type | router.py - INTENT_EXAMPLES, rule_based_intent_classification |
| Add diagnosis step | diagnosis_agent.py - add node function |
| Add QA step | qa_agent.py - add node function |
| Change routing logic | router.py - route_condition() |

## CONVENTIONS

- **Node signature**: `async def node_name(state: StateType) -> dict`
- **Return partial state**: Only return fields that changed
- **Few-shot in prompts**: See INTENT_EXAMPLES in router.py
- **Confidence scoring**: Router returns confidence 0-1, flags low confidence

## KEY FUNCTIONS

| Function | File | Role |
|----------|------|------|
| `router_node` | router.py | Classifies intent → diagnosis/qa/reasoning |
| `route_condition` | router.py | LangGraph conditional edge |
| `classify_intent_enhanced` | router.py | Few-shot + CoT classification |
| `rule_based_intent_classification` | router.py | Fallback keyword matching |

## ANTI-PATTERNS

- **Global _engine**: router.py uses module-level singleton
- **Hardcoded examples**: INTENT_EXAMPLES should be externalized
