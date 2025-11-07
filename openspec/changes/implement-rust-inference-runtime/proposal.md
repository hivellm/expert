# Implement Rust Inference Runtime (P0)

## Why

After validating architecture with Python prototype, build production-grade Rust inference engine for optimal performance. Target: <10ms expert loading, <20s inference for 1024 tokens, predictable memory management without GC pauses.

## What Changes

- Create Rust inference engine at `/expert/runtime/`
- Load Qwen3-0.6B base model with INT4/INT8 quantization
- Implement hot-swap LoRA/DoRA/IA³ adapter loading
- Implement paged KV cache (vLLM-inspired)
- Support CUDA and ROCm backends
- Expose gRPC/HTTP API
- Add Node.js (NAPI) and Python (PyO3) bindings

**BREAKING**: Replaces Python prototype with production runtime

## Impact

- **Affected specs**: runtime-core, adapter-loading, kv-cache, api
- **Affected code**: New `/expert/runtime/` (Rust workspace)
- **Dependencies**: candle/burn, safetensors, tokio, tonic/axum
- **Timeline**: P0 milestone (4-6 weeks)
- **Performance**: 10-100x faster than Python prototype

