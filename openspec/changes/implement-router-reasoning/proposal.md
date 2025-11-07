# Implement Router/Reasoning Layer (P1)

## Why

Enable automatic expert selection based on prompt analysis. Users shouldn't manually specify which experts to load—the router analyzes the task and selects optimal experts automatically.

## What Changes

- Create router module at `/expert/runtime/src/router/`
- Implement heuristic analysis (regex, keywords)
- Implement embedding-based search (MiniLM)
- Integrate with Vectorizer MCP for expert index
- Implement expert scoring and selection
- Add parameter tuning (temperature, max_tokens)
- Cache routing decisions

**Non-breaking**: Adds new functionality, manual expert selection still works

## Impact

- **Affected specs**: router-analysis, expert-selection, parameter-tuning
- **Affected code**: `/expert/runtime/src/router/` (new module)
- **Dependencies**: sentence-transformers (embeddings), Vectorizer MCP
- **Timeline**: P1 milestone (6-8 weeks after P0)
- **Performance target**: Router latency <20ms

