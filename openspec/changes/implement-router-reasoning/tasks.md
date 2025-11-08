# Router/Reasoning Implementation Tasks

## 1. Heuristic Analysis

- [ ] 1.1 Implement language detection (langdetect)
- [ ] 1.2 Implement format detection (JSON, XML, YAML)
- [ ] 1.3 Implement technology detection (Neo4j, Python, Rust)
- [ ] 1.4 Implement task classification (parsing, validation, etc.)
- [x] 1.5 Create keyword matching system
- [ ] 1.6 Benchmark heuristic speed (<1ms target)

## 2. Embedding-Based Search

- [ ] 2.1 Integrate SentenceTransformers (MiniLM)
- [ ] 2.2 Embed user prompts (~10ms)
- [ ] 2.3 Build expert index with embeddings
- [ ] 2.4 Implement ANN search (FAISS or Vectorizer MCP)
- [ ] 2.5 Return top-K candidates (K=20-30)
- [ ] 2.6 Set similarity threshold (0.6-0.8)

## 3. Expert Scoring

- [x] 3.1 Implement scoring function
- [ ] 3.2 Weight factors: semantic (40%), heuristic (25%), success rate (20%), VRAM (10%), popularity (5%)
- [ ] 3.3 Filter incompatible experts
- [ ] 3.4 Select top-10 experts
- [ ] 3.5 Order by composition logic (load_order)

## 4. Parameter Tuning

- [ ] 4.1 Map task types to temperature ranges
- [ ] 4.2 Map task types to max_tokens
- [ ] 4.3 Set top-p, top-k defaults
- [x] 4.4 Allow user override
- [ ] 4.5 Document tuning rules

## 5. Decision Caching

- [ ] 5.1 Implement fuzzy prompt hashing
- [ ] 5.2 Cache routing decisions (LRU)
- [ ] 5.3 Invalidate on failure
- [ ] 5.4 Set cache size limit (10k entries)

## 6. Integration

- [x] 6.1 Integrate router with inference runtime
- [ ] 6.2 Add auto-selection API endpoint
- [x] 6.3 Add manual override option
- [x] 6.4 Test end-to-end

## 7. Performance

- [ ] 7.1 Benchmark router latency
- [ ] 7.2 Target <20ms total
- [ ] 7.3 Optimize hot paths
- [ ] 7.4 Profile and eliminate bottlenecks

## 8. Testing

- [x] 8.1 Test accuracy on diverse prompts
- [ ] 8.2 Target >85% correct expert selection
- [ ] 8.3 Test edge cases
- [ ] 8.4 Test cache hit rates

## 9. Documentation

- [ ] 9.1 Update ROUTING_REASONING.md with impl details
- [ ] 9.2 Update STATUS.md
- [ ] 9.3 Mark P1 complete in ROADMAP.md

