# Optimize KV Cache for 200k Context Support

## Why

Current KV cache implementation pre-allocates memory for maximum sequence length, causing ~3 GB reservation for 200k context even when only using 100 tokens. For a 0.6B model with typical architecture (24 layers, 2048 hidden dim), full 200k context would require ~39 GB VRAM (2 × 24 × 200k × 2048 × 2 bytes), making it impractical for consumer GPUs (8-16GB).

## What Changes

- Implement true incremental paged allocation (blocks allocated on-demand, not pre-allocated)
- Add FP8/INT8 quantization for KV tensors (2× memory reduction)
- Verify and optimize for GQA/MQA support (2-8× reduction if model uses it)
- Implement prefix caching for shared prompt templates (5-20× savings in multi-session)
- Add chunked prefill for long contexts (prevents VRAM spikes during 200k token ingestion)
- Optional: Sliding window with retrieval (constant memory regardless of total context)
- Complete actual tensor read/write operations (currently TODOs in code)

## Impact

- **Affected specs**: runtime-core (inference engine)
- **Affected code**: 
  - `expert/cli/src/inference/paged_kv_cache.rs` (core modifications)
  - `expert/cli/src/inference/qwen.rs` (integration)
  - `expert/cli/src/commands/chat.rs` (usage)
- **Breaking changes**: None - internal optimization
- **Memory savings**: 200k context in ~4 GB instead of ~39 GB
- **Quality impact**: <2% degradation with FP8 quantization
- **Hardware requirements**: FP8 needs Ada/Hopper GPUs (fallback to INT8 or FP16)

