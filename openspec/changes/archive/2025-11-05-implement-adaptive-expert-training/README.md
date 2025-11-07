# Adaptive Expert Training Strategy

**Status:** 📋 Proposed  
**Priority:** ⚡ High  
**Target Release:** v0.3.0  
**Estimated Duration:** 8 weeks

## Quick Overview

Implement intelligent adapter architecture selection based on task complexity to optimize multi-expert deployment on RTX 4090 (24GB VRAM).

### Key Objectives

1. **QLoRA Training Foundation** → Faster training with NF4 + BF16
2. **Multi-Tier Adapters** → Right-sized adapters for each task type
3. **Runtime Optimizations** → Flash Attention, Paged KV-cache, Grammar decoding
4. **Intelligent Routing** → Automatic expert selection <20ms
5. **Multi-Expert Management** → 10+ concurrent experts with hot-swap

## Architecture Tiers

| Tier | Adapter | Use Case | Size | Quality |
|------|---------|----------|------|---------|
| **Ultralight** | IA³ + Soft-Prompt | Format/Style | 1-5 MB | 95% baseline |
| **Medium** | DoRA r=12 | Skill/Domain | 10-20 MB | 105% baseline |
| **Heavy** | DoRA r=24 | Complex Reasoning | 20-50 MB | 110% baseline |

## Success Metrics

### Training
- ⏱️ Training time: **-30%** (via QLoRA)
- 💾 VRAM during training: **<16 GB**
- 🔄 Iteration speed: **<30 min/epoch**

### Inference
- 🎯 Concurrent experts: **≥10 on RTX 4090**
- ⚡ Hot-swap latency: **<50ms**
- 🚀 Throughput: **>50 tokens/s** per expert
- 📦 Total ecosystem: **<200 MB** (10 experts)

### Quality
- ✨ IA³: **≥95%** baseline for format tasks
- 🎨 DoRA r=12: **≥105%** LoRA baseline
- 🧠 DoRA r=24: **≥110%** LoRA baseline
- 🎯 Routing accuracy: **≥98%**

## Implementation Phases

### Phase 1: Training Infrastructure (2 weeks)
- QLoRA support (NF4 + double-quant)
- IA³ adapter implementation
- DoRA adapter implementation
- LoKr adapter implementation (optional)
- Soft-prompt training
- Manifest schema updates

### Phase 2: Expert Migration (2 weeks)
- Expert classification (ultralight/medium/heavy)
- Retrain ultralight experts with IA³
- Retrain medium experts with DoRA r=12
- Retrain heavy experts with DoRA r=24
- Quality benchmarking

### Phase 3: Runtime Optimizations (2 weeks)
- Grammar-based constrained decoding
- Flash Attention 2 integration
- Paged KV-cache implementation
- RoPE scaling for long context
- Multi-expert hot-swap with LRU cache

### Phase 4: Routing System (1 week)
- Keyword-based heuristics
- Embedding similarity router
- Mini-policy network
- Routing confidence scorer
- Expert pre-loading

### Phase 5: Testing & Validation (1 week)
- Multi-expert load tests
- VRAM profiling
- Latency benchmarks
- Quality regression tests
- End-to-end integration tests

## Expert Classification Examples

| Expert | Current | Proposed | Rank | Size | Rationale |
|--------|---------|----------|------|------|-----------|
| JSON Format | LoRA 16 | IA³ + SP | - | ~3 MB | Pure format |
| EN/PT Style | N/A | IA³ + SP | - | ~2 MB | Style only |
| TypeScript | LoRA 16 | DoRA | 12 | ~15 MB | Code gen |
| SQL | LoRA 16 | DoRA | 16 | ~18 MB | Schema + query |
| Neo4j Cypher | LoRA 16 | DoRA | 24 | ~25 MB | Graph reasoning |
| Classification | LoRA 16 | DoRA | 8 | ~12 MB | Categorization |

## Technical Stack

### Required Dependencies
- `transformers >= 4.36.0` (DoRA support)
- `peft >= 0.8.0` (IA³, DoRA, LoKr)
- `bitsandbytes >= 0.42.0` (NF4 quantization)
- `flash-attn >= 2.5.0` (Flash Attention 2)
- `llama-cpp-python` (grammar-based decoding)

### Hardware Requirements
- **Training:** RTX 4090 (24 GB) or better
- **Inference:** RTX 4090 (10+ experts) or RTX 3090 (6-8 experts)
- **CPU Fallback:** Available for ultralight IA³ experts

## Migration Path

### Backward Compatibility
- ✅ Existing LoRA experts continue to work
- ✅ Gradual migration expert-by-expert
- ✅ Version field distinguishes old/new
- ✅ Runtime supports mixed adapter types

### Rollout Strategy
1. Implement infrastructure (Phases 1-3)
2. Migrate lowest-risk experts first (ultralight)
3. Validate quality and performance
4. Gradually migrate remaining experts
5. Deploy routing system last

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| IA³ quality for complex tasks | Use only for proven simple tasks; extensive validation |
| DoRA training instability | Conservative learning rates; fallback to LoRA |
| Runtime complexity | Phased rollout; single-expert optimization first |
| Library support issues | Verify versions; maintain LoRA fallback |

## Future Extensions

- 🔮 Mixture-of-Adapters (dynamic weighted combination)
- 📦 Adapter Distillation (compress post-training)
- ⚡ Quantized Adapters (Int8/Int4 weights)
- 🌳 Hierarchical Routing (multi-stage with clusters)
- 🤖 Auto-Architecture (ML-based adapter selection)

## References

- [DoRA Paper](https://arxiv.org/abs/2402.09353) - Weight-Decomposed Low-Rank Adaptation
- [IA³ Paper](https://arxiv.org/abs/2205.05638) - Few-Shot Parameter-Efficient Fine-Tuning
- [LoKr Paper](https://arxiv.org/abs/2309.14859) - Kronecker Product Low-Rank Adaptation
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient Finetuning of Quantized LLMs
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Fast and Memory-Efficient Attention

## Files

- [`proposal.md`](./proposal.md) - Detailed proposal with technical specifications
- [`tasks.md`](./tasks.md) - Comprehensive task breakdown with acceptance criteria

---

**Created:** 2025-11-03  
**Last Updated:** 2025-11-03  
**Owner:** HiveLLM Core Team

