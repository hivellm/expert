# Performance Guide

> VRAM budgets, latency targets, long context strategies, and trade-offs

## Overview

The Expert System is designed to run efficiently on consumer GPUs (8-16GB VRAM) while delivering fast, specialized inference. This guide covers resource budgeting, performance targets, optimization strategies, and common trade-offs.

---

## VRAM Budget

### Target Configuration

**Goal**: Fit base model + 10 experts + KV cache in 8-16GB VRAM

```
┌─────────────────────────────────────────────────┐
│         VRAM Allocation (16GB GPU)              │
├─────────────────────────────────────────────────┤
│  Base Model (Qwen3-0.6B INT4)      0.5 GB      │
│  10 Experts (LoRA r=16 avg)        0.7 GB      │
│  KV Cache (128k context)           3.5 GB      │
│  Activations & Buffers             1.0 GB      │
│  System Overhead                   0.3 GB      │
├─────────────────────────────────────────────────┤
│  Total Used                        6.0 GB      │
│  Available for scaling             10.0 GB     │
└─────────────────────────────────────────────────┘
```

### Component Breakdown

#### Base Model

| Quantization | VRAM | Quality | Speed | Use Case |
|--------------|------|---------|-------|----------|
| **INT4** | 0.3-0.4 GB | Good | Fast | Recommended default |
| **INT8** | 0.5-0.6 GB | Better | Moderate | Complex reasoning |
| **FP16** | 1.2 GB | Best | Slower | Rare (high accuracy needs) |

**Formula:**
```
VRAM (GB) = (num_params * bytes_per_param) / (1024^3)

Qwen3-0.6B:
- FP16: 600M * 2 bytes = 1.2 GB
- INT8: 600M * 1 byte = 0.6 GB
- INT4: 600M * 0.5 bytes = 0.3 GB
```

#### Experts

**LoRA VRAM formula:**
```
per_expert_vram = (r * (in_features + out_features) * num_layers * num_modules * 2 bytes) / (1024^3)

Example (r=16, attention only):
- Qwen3-0.6B: 28 layers
- 3 modules (q, v, o): each ~896x896
- per_expert = (16 * (896+896) * 28 * 3 * 2) / 1024^3
             ≈ 0.045 GB ≈ 45 MB
```

**Typical sizes:**

| Adapter Type | Rank | Modules | Size (MB) | VRAM (MB) |
|--------------|------|---------|-----------|-----------|
| LoRA | r=8 | attn (3) | 20-30 | 25 |
| LoRA | r=16 | attn (3) | 40-50 | 50 |
| LoRA | r=16 | attn+mlp (6) | 70-90 | 85 |
| LoRA | r=32 | attn+mlp (6) | 140-180 | 170 |
| DoRA | r=16 | attn+mlp (6) | 80-100 | 95 |
| IA³ | N/A | attn+mlp | 1-5 | 3 |
| Soft-prompt | 64 tokens | N/A | <1 | 0.5 |

**10 experts** (mix):
- 6x LoRA r=16 (attn): 6 * 50 MB = 300 MB
- 3x LoRA r=16 (attn+mlp): 3 * 85 MB = 255 MB
- 1x IA³: 3 MB
- **Total: ~560 MB**

#### KV Cache

**Paged KV cache formula:**
```
kv_cache_size = 2 (k+v) * num_layers * hidden_size * context_length * bytes_per_element

Qwen3-0.6B with 128k context (FP16):
= 2 * 28 * 896 * 131072 * 2 bytes
= 13.1 GB (theoretical max)

With paging and compression:
≈ 3.5 GB (practical)
```

**Context-dependent:**

| Context Length | KV Cache (FP16) | Notes |
|----------------|-----------------|-------|
| 32k | 0.9 GB | Short prompts |
| 64k | 1.8 GB | Medium prompts |
| 128k | 3.5 GB | Long contexts |
| 256k | 7.0 GB | Very long (tight on 8GB GPU) |

**Optimization**: INT8 KV cache halves memory (~1.75 GB for 128k)

---

## Latency Targets

### End-to-End Latency

**Target breakdown (single job, 1024 output tokens):**

| Stage | Target | Typical | GPU | Notes |
|-------|--------|---------|-----|-------|
| Reception | <1ms | 0.5ms | CPU | Parse request |
| Routing | <20ms | 12ms | CPU | Heuristics + embeddings |
| Expert Load (hot) | <10ms | 5ms | SSD→VRAM | Pre-mapped weights |
| Expert Load (cold) | <200ms | 100ms | SSD→VRAM | Decompress + load |
| Inference (prefill) | 50-200ms | 100ms | GPU | Process input prompt |
| Inference (decode) | 5-20s | 10s | GPU | Generate 1024 tokens |
| Post-process | <10ms | 5ms | CPU | Validation |
| **Total (hot)** | **5-20s** | **10s** | | |
| **Total (cold)** | **5-20s** | **10s** | | |

### Token Generation Speed

**Factors:**
- GPU compute capability
- Batch size
- Context length
- Expert count

**Typical speeds (RTX 4090, 128k context, 5 experts):**

| Batch Size | Tokens/sec | Latency (1024 tokens) |
|------------|------------|-----------------------|
| 1 | 50-70 | 15-20s |
| 2 | 90-120 | 8-12s per sequence |
| 4 | 140-180 | 5-7s per sequence |
| 8 | 180-220 | 4-6s per sequence |

**GPU comparison (batch=1, 1024 tokens):**

| GPU | VRAM | Tokens/sec | Latency | Notes |
|-----|------|------------|---------|-------|
| RTX 3060 | 12GB | 25-35 | 30-40s | Tight VRAM |
| RTX 4060 Ti | 16GB | 35-45 | 23-30s | Good balance |
| RTX 4070 | 12GB | 40-55 | 18-25s | Fast but limited VRAM |
| RTX 4080 | 16GB | 55-75 | 14-19s | Excellent |
| RTX 4090 | 24GB | 70-100 | 10-15s | Best consumer |
| A100 40GB | 40GB | 100-150 | 7-10s | Datacenter |

---

## Long Context Strategy (120k-200k tokens)

### RoPE Scaling

**Methods:**

1. **Linear Scaling** (simple, baseline)
```python
scaled_pos = position / scale_factor  # scale_factor = 4 for 4x extension
```

2. **NTK-aware** (better, recommended)
```python
# Adjust frequency instead of position
base_freq = 10000 * (scale_factor ** (dim / (dim - 2)))
```

3. **YaRN** (best, state-of-art)
```python
# Hybrid: scale low frequencies (local), keep high frequencies (long-range)
# See: https://arxiv.org/abs/2309.00071
```

**Configuration:**

```python
rope_config = {
    "method": "yarn",
    "scale": 8.0,  # 8x extension: 16k → 128k
    "alpha": 1.0,  # YaRN attention scaling
    "original_max_position": 16384,
    "target_max_position": 131072
}
```

### Paged Attention

**Benefits:**
- No memory fragmentation
- Dynamic allocation (only use what you need)
- Eviction support (for extremely long contexts)

**Block configuration:**

```python
paged_config = {
    "block_size": 16,  # Tokens per block
    "num_blocks": 8192,  # Max blocks (128k / 16)
    "dtype": "float16",  # or "int8" for 2x compression
}
```

**Memory savings:**

| Context | No Paging | With Paging | Savings |
|---------|-----------|-------------|---------|
| 32k | 3.0 GB | 0.9 GB | 70% |
| 64k | 6.0 GB | 1.8 GB | 70% |
| 128k | 12.0 GB | 3.5 GB | 71% |

### Chunk Routing

For extremely long inputs, pre-filter irrelevant chunks:

```python
def chunk_routing(prompt: str, context: str, max_context: int = 120000):
    """Keep only relevant context chunks"""
    
    # Split context into chunks
    chunks = split_into_chunks(context, chunk_size=4000)
    
    # Embed prompt and chunks
    prompt_emb = embed(prompt)
    chunk_embs = [embed(c) for c in chunks]
    
    # Rank chunks by relevance
    scores = [cosine_sim(prompt_emb, c_emb) for c_emb in chunk_embs]
    ranked_chunks = sorted(zip(scores, chunks), reverse=True)
    
    # Keep top chunks until context limit
    relevant_context = ""
    for score, chunk in ranked_chunks:
        if len(relevant_context) + len(chunk) > max_context:
            break
        relevant_context += chunk
    
    return relevant_context
```

### Long Context Training

**Continual finetuning** for base model:

```python
# Extend base model to 128k
train_data = long_context_dataset(
    min_length=64000,
    max_length=131072,
    num_examples=10000
)

train(
    model=base_model,
    data=train_data,
    epochs=1,  # Short and cheap
    lr=1e-5,   # Low LR to avoid catastrophic forgetting
    rope_scaling="yarn"
)
```

**Position curriculum** for experts:

```python
# Train expert on progressively longer sequences
curriculum = [
    (0, 10000, 16384),    # Easy: short contexts
    (10000, 20000, 32768), # Medium
    (20000, 30000, 65536), # Hard
    (30000, 40000, 131072) # Very hard
]

for start, end, max_len in curriculum:
    train_expert(
        data=dataset[start:end],
        max_length=max_len
    )
```

---

## KV Cache Constraints

### Critical Limitation

**KV cache is NOT portable between different expert sets.**

```python
# This is INVALID:
session = engine.create_session()
engine.attach_experts(session, ["json-parser", "english"])
output_part1 = engine.generate(session, "Parse this JSON")

# Cannot change experts mid-generation!
engine.attach_experts(session, ["neo4j-schema"])  # ❌ Breaks KV cache
output_part2 = engine.generate(session, "...")  # ❌ Incorrect output
```

**Why?**  
Expert adapters modify hidden states, which affect KV cache entries. Changing experts invalidates cached K/V.

**Workarounds:**

1. **Define experts upfront** (recommended):
```python
session = engine.create_session()
engine.attach_experts(session, ["json", "english", "neo4j"])  # All at once
output = engine.generate(session, full_prompt)  # ✓ Works
```

2. **Clear cache between expert changes**:
```python
session = engine.create_session()
engine.attach_experts(session, ["json"])
output1 = engine.generate(session, prompt1)

session.clear_kv_cache()  # Clear cache
engine.attach_experts(session, ["neo4j"])  # Change experts
output2 = engine.generate(session, prompt2)  # ✓ Works (but no context from output1)
```

3. **Multi-turn with same experts**:
```python
# OK: Same experts across turns
session = engine.create_session()
engine.attach_experts(session, ["conversational"])

turn1 = engine.generate(session, "Hello")  # KV cache populated
turn2 = engine.generate(session, "How are you?")  # ✓ Reuses cache
```

---

## Optimization Tips

### 1. Combine LoRA + IA³

```python
# Instead of: 10x LoRA (850 MB)
experts = [lora_r16] * 10

# Use: 7x LoRA + 3x IA³ (400 MB)
experts = [lora_r16] * 7 + [ia3] * 3

# Savings: 450 MB VRAM
# Trade-off: Slightly lower quality for IA³ experts
```

### 2. Soft-Prompts for Style

```python
# Instead of: LoRA for "always output JSON" (50 MB)
json_enforcer_lora = train_lora(json_format_dataset)

# Use: Soft-prompt (0.5 MB)
json_style_prompt = train_soft_prompt(json_format_dataset)

# Savings: 49.5 MB per style expert
```

### 3. Grammar-Guided Decoding

For strict formats (JSON, SQL), use constrained decoding:

```python
from transformers import LogitsProcessorList
from json_schema_enforcer import JSONSchemaLogitsProcessor

schema = {"type": "object", "properties": {"name": {"type": "string"}}}

logits_processor = LogitsProcessorList([
    JSONSchemaLogitsProcessor(schema, tokenizer)
])

output = model.generate(
    ...,
    logits_processor=logits_processor
)

# Guarantees valid JSON without post-processing
# No need for separate "JSON repair" expert
```

### 4. Speculative Decoding

**Speedup: 1.5-2x** if draft acceptance >70%

```python
# Base model as draft (fast, no experts)
draft_tokens = base_model.generate_n(prompt, n=4)

# Base + experts as verifier (slower but accurate)
verified = base_with_experts.verify(draft_tokens)

# Accept matching tokens, restart from divergence
if acceptance_rate > 0.7:
    # Net speedup achieved
```

**When to use:**
- Long outputs (>1000 tokens)
- Experts don't drastically change distribution
- GPU has spare capacity for parallel verification

### 5. INT8 KV Cache

```python
kv_cache_config = {
    "dtype": "int8",  # Instead of float16
    "quantization": "per_channel"
}

# Savings: 2x reduction in KV cache VRAM
# Trade-off: ~1-2% quality loss (usually acceptable)
```

### 6. Flash Attention

```python
# Enable Flash Attention 2 (if supported)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen3-0.6B",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)

# Speedup: 1.5-2x for long contexts
# No quality loss
```

### 7. Expert Prefetching

```python
# While GPU is busy with current job, prefetch next experts
async def overlapped_loading():
    current_job = job_queue.pop()
    await gpu.infer(current_job)  # GPU busy
    
    # Meanwhile, load experts for next job (CPU/SSD)
    next_job = job_queue.peek()
    await prefetch_experts(next_job.experts)  # Parallel I/O
```

---

## Domestic GPU Sweet Spots

### 8GB VRAM (RTX 3060, 4060)

**Configuration:**
- Base: INT4 (0.3 GB)
- Experts: 6-8 experts (prefer IA³, 300 MB)
- KV: 64k context max (1.8 GB)
- Batch: 1-2

**Limitations:**
- Cannot run 128k context reliably
- Limited to simple experts (LoRA r=8 or IA³)

**Recommendation**: Good for prototyping, not production.

### 12GB VRAM (RTX 4070, 3060 Ti)

**Configuration:**
- Base: INT4 (0.3 GB)
- Experts: 8-10 LoRA r=16 (600 MB)
- KV: 128k context (3.5 GB)
- Batch: 1-2

**Sweet spot** for most users. Can handle production workloads.

### 16GB VRAM (RTX 4060 Ti 16GB, 4080)

**Configuration:**
- Base: INT4 (0.3 GB) or INT8 (0.6 GB)
- Experts: 10 mixed LoRA/DoRA (800 MB)
- KV: 128k context (3.5 GB) + INT8 KV cache
- Batch: 2-4

**Excellent** for power users. Room for experimentation.

### 24GB VRAM (RTX 4090)

**Configuration:**
- Base: INT8 (0.6 GB)
- Experts: 15+ experts (1.2 GB)
- KV: 256k context (7 GB) or 128k with batching
- Batch: 4-8

**Best consumer option**. Can exceed spec (>10 experts) comfortably.

---

## Trade-offs

### Quality vs Speed

| Priority | Configuration | Quality | Speed | VRAM |
|----------|---------------|---------|-------|------|
| Max Quality | INT8 base, DoRA r=32, FP16 KV | 100% | 50% | High |
| Balanced | INT4 base, LoRA r=16, FP16 KV | 95% | 100% | Medium |
| Max Speed | INT4 base, IA³, INT8 KV, spec decode | 85% | 150% | Low |

### Context Length vs Throughput

| Context | KV Cache | Batch Size | Throughput |
|---------|----------|------------|------------|
| 32k | 0.9 GB | 8 | High |
| 64k | 1.8 GB | 4 | Medium |
| 128k | 3.5 GB | 2 | Medium-Low |
| 256k | 7.0 GB | 1 | Low |

**Recommendation**: Use 128k for most tasks. Reserve 256k for rare long-document cases.

### Expert Count vs Latency

| Experts | Load Time (hot) | Inference Overhead | Use Case |
|---------|-----------------|--------------------| ---------|
| 1-3 | <5ms | Minimal | Simple tasks |
| 4-7 | 5-10ms | Low (~5%) | Typical |
| 8-10 | 10-15ms | Moderate (~10%) | Complex |
| 11-15 | 15-25ms | Higher (~15%) | Very complex (rare) |

**Recommendation**: Target 5-7 experts for most tasks. Reserve 10 for exceptional cases.

---

## Benchmarking

### Key Metrics

1. **Throughput**: Tokens/second (higher is better)
2. **Latency**: Time to first token + total time (lower is better)
3. **VRAM utilization**: Peak usage (should be <80% of available)
4. **Expert load time**: Cold vs hot (hot should be <10ms)
5. **KV cache hit rate**: Reuse across turns (higher is better)

### Profiling Tools

```python
# PyTorch profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model.generate(...)

prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

```bash
# NVIDIA tools
nvidia-smi dmon -s u  # GPU utilization
nvtop  # Interactive GPU monitor
```

---

## Best Practices

1. **Start with INT4 base + LoRA r=16**: Best default balance
2. **Target 5-7 experts per inference**: Sweet spot for quality/speed
3. **Use 128k context by default**: Sufficient for most tasks
4. **Enable paged attention**: Mandatory for long contexts
5. **Prefer hot cache**: Keep popular experts loaded
6. **Monitor VRAM**: Stay below 80% to avoid OOM
7. **Batch when possible**: 2-4x throughput gain
8. **Profile before optimizing**: Measure, don't guess
9. **Test on target GPU**: Performance varies significantly across GPUs
10. **Use INT8 KV for 256k contexts**: Only way to fit in 16GB

---

## Future Optimizations (Roadmap)

- **Quantized experts**: INT8 LoRA weights (2x smaller)
- **Expert pruning**: Remove low-impact expert layers
- **Dynamic rank**: Adjust LoRA rank per layer
- **Multi-GPU**: Split model across GPUs
- **Expert fusion**: Merge frequently co-loaded experts
- **Sparse experts**: Mixture-of-Experts within each expert
- **Flash Attention 3**: Further speedups

---

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
- See [EXECUTION_PIPELINE.md](EXECUTION_PIPELINE.md) for inference flow
- See [ROADMAP.md](../ROADMAP.md) for performance optimization schedule

