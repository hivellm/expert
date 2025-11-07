# Adaptive Expert Training Strategy

**Status:** Proposed  
**Created:** 2025-11-03  
**Target Release:** v0.3.0

## Overview

Implement an adaptive training strategy that optimizes expert architecture selection based on task complexity, enabling efficient multi-expert deployment on consumer hardware (RTX 4090) with up to 10 concurrent specialists.

## Problem Statement

Current implementation uses a one-size-fits-all LoRA approach for all experts, which:
- Over-provisions simple format/style tasks (wasting VRAM and inference time)
- Under-provisions complex reasoning tasks (limiting quality ceiling)
- Doesn't leverage modern PEFT methods (DoRA, IA³, LoKr) for optimal quality/size trade-offs
- Lacks runtime optimizations for multi-expert concurrent execution

For Qwen3-0.6B with target of 10+ concurrent experts on 24GB VRAM, we need specialized strategies.

## Proposed Solution

### Training Strategy: QLoRA Foundation

**All experts train with QLoRA** for faster iteration and lower VRAM:
- `bnb_4bit_quant_type: "nf4"`
- `bnb_4bit_use_double_quant: true`
- `compute_dtype: bfloat16`
- `attn_impl: "flash_attention_2"`
- `seq_len: 1536-2048`
- `group_by_length: true`

Benefits:
- Higher batch sizes and sequence lengths during training
- Faster iteration cycles
- Lower training costs
- No perceptible quality loss vs full precision

### Adapter Architecture by Task Type

#### 1. Ultralight (Format/Style) → IA³ + Soft-Prompt

**Use Cases:**
- JSON formatting/repair
- Language switching (EN/PT)
- Tone/style adjustment
- Simple structural tasks

**Configuration:**
```json
{
  "adapter_type": "ia3",
  "target_modules": ["k_proj", "v_proj", "mlp.down_proj"],
  "soft_prompt": {
    "enabled": true,
    "length": 8,
    "init_text": "Task-specific initialization"
  }
}
```

**Benefits:**
- Minimal size (1-5 MB)
- Instant hot-swap
- Multiple concurrent specialists
- Near-zero VRAM overhead

#### 2. Medium (Skill/Domain) → DoRA (r=8-16)

**Use Cases:**
- Classification tasks
- Non-trivial parsing
- Domain-specific instructions
- Code generation (TypeScript, Python)
- SQL query generation

**Configuration:**
```json
{
  "adapter_type": "dora",
  "rank": 12,
  "alpha": 24,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "dropout": 0.05,
  "use_magnitude_layer": true
}
```

**Benefits:**
- Better quality than LoRA at same rank
- Moderate size (10-20 MB)
- Good balance of quality/efficiency

#### 3. Heavy (Complex Reasoning) → DoRA/LoKr (r=16-32)

**Use Cases:**
- Neo4j Cypher generation
- Complex SQL with multiple joins
- Multi-step reasoning
- Schema understanding + generation

**Configuration:**
```json
{
  "adapter_type": "dora",
  "rank": 24,
  "alpha": 48,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.up_proj", "mlp.down_proj"],
  "dropout": 0.05,
  "use_magnitude_layer": true
}
```

**Alternative (LoKr):**
```json
{
  "adapter_type": "lokr",
  "rank": 16,
  "alpha": 32,
  "decompose_both": true,
  "factor": 8
}
```

**Benefits:**
- Maximum quality for challenging tasks
- Use sparingly (1-2 per query)
- Size: 20-50 MB

### Runtime Optimizations

#### 1. Constrained Decoding
- Grammar-based generation (GBNF) for JSON/Schema/SQL
- Temperature: 0.1-0.3
- Top-p: ~0.9
- Repetition penalty: 1.1

#### 2. Attention Optimizations
- Flash Attention 2
- Paged KV-cache
- RoPE scaling for long context
- KV-cache quantization (int8)

#### 3. Multi-Expert Management
- LRU cache for hot experts
- Pre-load top-N by routing confidence
- Stream processing (1 prompt = 1 CUDA stream)
- Adapter hot-swap without base model reload

#### 4. Intelligent Routing
- Heuristic keywords (fast path)
- Embedding similarity (medium path)
- Mini-policy network (slow path, high confidence)
- Select ≤10 experts before generation starts

## Implementation Plan

### Phase 1: Training Infrastructure (Week 1-2)
- [ ] Add QLoRA support to `expert_trainer.py`
- [ ] Implement IA³ adapter type
- [ ] Implement DoRA adapter type
- [ ] Implement LoKr adapter type (optional)
- [ ] Add soft-prompt training support
- [ ] Update manifest schema for adapter type configuration

### Phase 2: Expert Migration (Week 2-3)
- [ ] Classify existing experts by complexity
- [ ] Retrain ultralight experts with IA³
- [ ] Retrain medium experts with DoRA (r=12)
- [ ] Retrain heavy experts with DoRA (r=24)
- [ ] Benchmark quality vs baseline LoRA
- [ ] Update expert packages

### Phase 3: Runtime Optimizations (Week 3-4)
- [ ] Implement grammar-based constrained decoding
- [ ] Add Flash Attention 2 support
- [ ] Implement paged KV-cache
- [ ] Add RoPE scaling configuration
- [ ] Build LRU expert cache manager

### Phase 4: Routing System (Week 4-5)
- [ ] Implement keyword-based heuristics
- [ ] Build embedding similarity router
- [ ] Train mini-policy network
- [ ] Create routing confidence scorer
- [ ] Benchmark routing accuracy

### Phase 5: Testing & Validation (Week 5-6)
- [ ] Multi-expert concurrent load tests
- [ ] VRAM profiling (10+ experts)
- [ ] Latency benchmarks
- [ ] Quality regression tests
- [ ] Hot-swap performance tests

## Expert Classification Matrix

| Expert | Current | Proposed | Rank | Size | Rationale |
|--------|---------|----------|------|------|-----------|
| JSON Format | LoRA r=16 | IA³ + SP | - | ~3 MB | Pure formatting task |
| EN/PT Style | N/A | IA³ + SP | - | ~2 MB | Style transfer only |
| TypeScript | LoRA r=16 | DoRA | 12 | ~15 MB | Code generation |
| SQL | LoRA r=16 | DoRA | 16 | ~18 MB | Schema + query |
| Neo4j Cypher | LoRA r=16 | DoRA | 24 | ~25 MB | Complex graph reasoning |
| Classification | LoRA r=16 | DoRA | 8 | ~12 MB | Simple categorization |

## Success Metrics

### Training Efficiency
- Training time reduction: >30% (via QLoRA higher batch sizes)
- VRAM during training: <16 GB for all experts
- Iteration speed: <30 min per epoch (Qwen3-0.6B)

### Inference Performance
- Concurrent experts: ≥10 on RTX 4090 (24 GB)
- Hot-swap latency: <50ms
- First token latency: <100ms (with grammar)
- Throughput: >50 tokens/s per expert

### Quality Benchmarks
- IA³ experts: ≥95% baseline quality for format tasks
- DoRA (r=12): ≥105% baseline LoRA quality
- DoRA (r=24): ≥110% baseline LoRA quality
- Overall multi-expert accuracy: ≥98% routing correctness

### Package Sizes
- Ultralight: 1-5 MB
- Medium: 10-20 MB
- Heavy: 20-50 MB
- Total ecosystem (10 experts): <200 MB

## Technical Requirements

### Dependencies
- `transformers >= 4.36.0` (DoRA support)
- `peft >= 0.8.0` (IA³, DoRA, LoKr)
- `bitsandbytes >= 0.42.0` (NF4 quantization)
- `flash-attn >= 2.5.0` (Flash Attention 2)
- `llama-cpp-python` (grammar-based decoding)

### Hardware Targets
- **Training:** RTX 4090 (24 GB) or better
- **Inference:** RTX 4090 (10+ experts) or RTX 3090 (6-8 experts)
- **CPU Fallback:** Possible for ultralight IA³ experts

## Migration Path

### Backward Compatibility
- Existing LoRA experts continue to work
- Gradual migration expert-by-expert
- Version field in manifest distinguishes old/new
- Runtime supports mixed adapter types

### User Communication
- Document adapter selection guidelines
- Provide migration scripts
- Publish benchmark comparisons
- Update training guide with recommendations

## Risks & Mitigations

### Risk 1: IA³ Quality for Complex Tasks
**Mitigation:** Use only for proven simple tasks (format/style); extensive validation before deployment

### Risk 2: DoRA Training Instability
**Mitigation:** Conservative learning rates, warmup steps, gradient clipping; fallback to LoRA if needed

### Risk 3: Runtime Complexity
**Mitigation:** Phased rollout; start with single-expert optimization, then multi-expert

### Risk 4: Library Support
**Mitigation:** Verify PEFT library versions; contribute patches if needed; maintain LoRA fallback

## Future Extensions

- **Mixture-of-Adapters:** Dynamic weighted combination during inference
- **Adapter Distillation:** Compress heavy experts post-training
- **Quantized Adapters:** Int8/Int4 adapter weights for extreme efficiency
- **Hierarchical Routing:** Multi-stage routing with expert clusters
- **Auto-Architecture:** ML-based adapter type selection per dataset

## References

- DoRA: [Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- IA³: [Few-Shot Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2205.05638)
- LoKr: [Kronecker Product Low-Rank Adaptation](https://arxiv.org/abs/2309.14859)
- QLoRA: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Flash Attention 2: [Fast and Memory-Efficient Attention](https://arxiv.org/abs/2307.08691)

