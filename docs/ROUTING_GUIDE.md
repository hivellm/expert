# Expert Routing System Guide

Guide for automatic expert selection in HiveLLM.

## Status: Not Implemented

**Current State**: Manual expert selection via `--experts` flag

**Planned**: v0.3.0+

## Manifest Configuration (Prepared)

All experts have routing metadata in their manifests:

```json
{
  "capabilities": [
    "database:sql",
    "query:sql",
    "task:text2sql"
  ],
  "routing": {
    "keywords": ["sql", "database", "query", "table", "select"],
    "router_hint": "database=sql OR query=sql OR task=text2sql",
    "priority": 0.80
  }
}
```

## Planned Routing Methods

### 1. Keyword-Based Heuristics (v0.3.0)

**Fast**: <1ms latency

**Accuracy**: 85-90%

**How it works**:
1. Extract keywords from user query
2. Match against expert `routing.keywords`
3. Score by keyword overlap
4. Return top-N experts

**Example**:
```
Query: "Show all users from the database"
Keywords: ["database", "users"]
Match: expert-sql (keywords: ["sql", "database", "table"])
Score: 0.85
```

### 2. Embedding Similarity (v0.3.1)

**Medium**: <10ms latency

**Accuracy**: 92-95%

**How it works**:
1. Embed user query (sentence-transformers)
2. Compare with pre-computed expert capability embeddings
3. Cosine similarity scoring
4. Return top-N experts

**Requires**:
- Embedding model (all-MiniLM-L6-v2)
- Pre-computed expert embeddings

### 3. Mini-Policy Network (v0.3.2)

**Slow**: <20ms latency

**Accuracy**: 97-98%

**How it works**:
1. Extract query features (keywords, embeddings, length)
2. Pass through small MLP (2 layers, 256 hidden)
3. Output expert probabilities
4. Return top-N with confidence scores

**Requires**:
- Policy network training data (query → expert pairs)
- Training pipeline for policy network

## Current Workaround

### Manual Selection

```bash
# Specify experts explicitly
expert-cli chat --experts sql,neo4j

# Single expert
expert-cli chat --experts typescript
```

### Auto-Detection (Future)

```bash
# Automatic routing
expert-cli chat --auto-route

# With confidence threshold
expert-cli chat --auto-route --confidence 0.8
```

## Routing Configuration

### Priority Levels

```json
{
  "routing": {
    "priority": 0.90  // High priority (Neo4j - specialized)
  }
}
```

**Ranges**:
- `0.90-1.00`: Specialized experts (Neo4j, SQL)
- `0.70-0.89`: General experts (TypeScript, JSON)
- `0.50-0.69`: Fallback experts

### Router Hints

```json
{
  "routing": {
    "router_hint": "database=sql OR query=sql OR task=text2sql"
  }
}
```

**Format**: Boolean expression with capabilities

### Keywords

```json
{
  "routing": {
    "keywords": [
      "sql",
      "database",
      "query",
      "table",
      "select",
      "join"
    ]
  }
}
```

**Best Practices**:
- 5-10 keywords per expert
- Include variations (SQL, sql, database, db)
- Add task-specific terms (JOIN, SELECT for SQL)

## Routing Strategies

### Strategy 1: Fast First-Match

```
1. Keyword matching (1ms)
2. If confidence >0.9: Return
3. Else: Fallback to embeddings
```

**Best for**: Low latency, single expert selection

### Strategy 2: Multi-Expert Ranking

```
1. Keyword matching (1ms)
2. Embedding similarity (10ms)
3. Combine scores (weighted average)
4. Return top-3 experts
```

**Best for**: Complex queries needing multiple experts

### Strategy 3: Policy Network (Future)

```
1. Extract features (5ms)
2. Policy network inference (15ms)
3. Confidence scoring
4. Return top-N with probabilities
```

**Best for**: Maximum accuracy, production systems

## Implementation Roadmap

### Phase 1: Keyword Routing (v0.3.0)

- [ ] Implement keyword extraction
- [ ] Implement keyword matching
- [ ] Add confidence scoring
- [ ] Test with 100+ queries
- [ ] Document accuracy metrics

### Phase 2: Embedding Router (v0.3.1)

- [ ] Add sentence-transformers dependency
- [ ] Pre-compute expert embeddings
- [ ] Implement cosine similarity
- [ ] Cache embeddings for performance
- [ ] Benchmark latency

### Phase 3: Policy Network (v0.3.2)

- [ ] Collect routing training data
- [ ] Design policy architecture
- [ ] Train policy network
- [ ] Integrate in CLI
- [ ] Validate accuracy

## Expected Performance

### Keyword Router

- Latency: <1ms
- Accuracy: 85-90%
- False positives: <15%

### Embedding Router

- Latency: <10ms
- Accuracy: 92-95%
- False positives: <8%

### Policy Network

- Latency: <20ms
- Accuracy: 97-98%
- False positives: <3%

## Current Status

**Implemented**: Routing metadata in all manifests

**Not Implemented**: Actual routing logic

**Workaround**: Use `--experts` flag to manually select

