# KV Cache Optimization - Implementation Tasks

## Status: NOT STARTED (0%)

## 1. Complete Basic Implementation (BLOCKING - 1 week)

- [ ] 1.1 Implement actual tensor indexing in `write_kv()`
- [ ] 1.2 Implement tensor gathering in `read_kv()`
- [ ] 1.3 Add integration tests for write/read roundtrip
- [ ] 1.4 Benchmark basic operations vs dense cache

## 2. Incremental Block Allocation (1 week)

- [ ] 2.1 Refactor `physical_pages` from pre-allocated to lazy `HashMap<usize, Tensor>`
- [ ] 2.2 Update `allocate_page()` to create blocks on-demand
- [ ] 2.3 Update `write_kv()` and `read_kv()` for sparse pool
- [ ] 2.4 Update `estimate_memory_mb()` to count only allocated blocks
- [ ] 2.5 Test with varying sequence lengths (100, 1k, 10k, 100k tokens)

## 3. Lower Precision KV Storage (2 weeks)

- [ ] 3.1 Research Candle FP8 support and GPU kernel availability
- [ ] 3.2 Add `kv_dtype` config (FP16, BF16, FP8, INT8)
- [ ] 3.3 Implement FP8 quantization utilities
- [ ] 3.4 Implement INT8 quantization utilities  
- [ ] 3.5 Update `write_kv()` to quantize before storing
- [ ] 3.6 Update `read_kv()` to dequantize after loading
- [ ] 3.7 Benchmark quality impact (target: <2% degradation)
- [ ] 3.8 Update memory estimation for new dtypes

## 4. GQA/MQA Optimization (3-5 days)

- [ ] 4.1 Verify Qwen3-0.6B uses GQA (`num_key_value_heads` in config)
- [ ] 4.2 Add `num_kv_heads` to `PagedKVCacheConfig`
- [ ] 4.3 Update tensor shapes to use `num_kv_heads` instead of `num_heads`
- [ ] 4.4 Verify memory savings with GQA models
- [ ] 4.5 Test with Qwen3 inference

## 5. Prefix Caching (2 weeks)

- [ ] 5.1 Design prefix identification (hash first N tokens)
- [ ] 5.2 Implement shared prefix pool with reference counting
- [ ] 5.3 Update `read_kv()` for prefix + local KV concatenation
- [ ] 5.4 Implement prefix eviction policy (LRU with pinning)
- [ ] 5.5 Add prefix cache API (`cache_prefix`, `attach_prefix`, `release_prefix`)
- [ ] 5.6 Test multi-session sharing with same system prompt
- [ ] 5.7 Benchmark hit rate with real prompts

## 6. Chunked Prefill (1-2 weeks)

- [ ] 6.1 Design chunking strategy (512-2048 token chunks)
- [ ] 6.2 Implement chunked forward pass
- [ ] 6.3 Add prefill progress tracking and cancellation support
- [ ] 6.4 Optimize chunk size dynamically based on VRAM headroom
- [ ] 6.5 Test with 200k token prompt (verify no OOM)
- [ ] 6.6 Add prefill benchmarks (1k, 10k, 50k, 100k, 200k)

## 7. Enhanced Eviction Policies (1 week)

- [ ] 7.1 Implement priority-based eviction (prefix > recent > old)
- [ ] 7.2 Implement working set tracking
- [ ] 7.3 Add eviction metrics (rate, hit rate, thrashing detection)
- [ ] 7.4 Implement adaptive eviction policies
- [ ] 7.5 Add configurable policies (LRU, LFU, Priority, Hybrid)

## 8. Advanced Features - OPTIONAL (3-4 weeks)

- [ ] 8.1 Sliding window attention with sink tokens
- [ ] 8.2 Incremental summarization for evicted chunks
- [ ] 8.3 Retrieval mechanism for historical context
- [ ] 8.4 Test with ultra-long contexts (500k, 1M tokens)
- [ ] 8.5 Benchmark quality vs full attention

## 9. Testing & Documentation (1-2 weeks)

- [ ] 9.1 Unit tests for all quantization functions
- [ ] 9.2 Integration tests with Qwen3 end-to-end
- [ ] 9.3 Stress tests (memory pressure, high request rate)
- [ ] 9.4 Quality tests (perplexity comparison)
- [ ] 9.5 Performance benchmarks (latency, throughput, memory)
- [ ] 9.6 Update docs/ARCHITECTURE.md
- [ ] 9.7 Update CHANGELOG.md
- [ ] 9.8 Update STATUS.md

## Summary

**Total**: 0/54 tasks (0% complete)

**Critical Path**: Tasks 1-4, 6, 9 (minimum viable: 7-9 weeks)

**Full Featured**: All tasks (12-16 weeks)

**Success Criteria**:
- 200k tokens in ≤4 GB VRAM ✓
- Incremental allocation (no upfront reservation) ✓
- FP8 KV cache with <2% quality loss ✓
- Chunked prefill handles 200k without OOM ✓
- Complete documentation ✓

