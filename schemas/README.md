# Expert Manifest Schema

JSON Schema for HiveLLM expert package manifests.

## Schema File

`expert-manifest.schema.json` - Complete JSON Schema (Draft 7) defining all valid manifest fields.

## Schema Versions

### v1.0 (Legacy)
- Single base model only
- Adapters at root level
- Simple structure

### v2.0 (Current)
- Multiple base models support
- Adapters nested in base_models
- Enhanced metadata (routing, perf, runtime)

## Field Implementation Status (Updated v0.2.3)

### ✅ Fully Implemented (Used by CLI/Training/Runtime)

| Field | Type | Used By | Description |
|-------|------|---------|-------------|
| `name`, `version`, `description` | String | All | Basic metadata |
| `base_model` (v1.0) | Object | CLI, Training | Single base model |
| `base_models` (v2.0) | Array | CLI, Training | Multiple base models |
| `adapters` | Array | Training | LoRA/DoRA/IA³ configuration |
| `soft_prompts` | Array | Training, Packaging | **NEW**: Trainable prompt embeddings |
| `training.decoding.temperature` | Float | Runtime | **NEW**: Expert-specific temperature |
| `training.decoding.top_p` | Float | Runtime | **NEW**: Expert-specific top-p |
| `training.decoding.top_k` | Integer | Runtime | **NEW**: Expert-specific top-k |
| `capabilities` | Array | Router (future) | Expert capabilities |
| `constraints` | Object | Loader (future) | Loading rules |
| `perf` | Object | Loader (future) | Resource planning |
| `training.config.*` | Object | Training | All hyperparameters |
| `training.dataset.*` | Object | Training | Dataset loading |

### ✅ Fully Implemented (v0.2.3)

| Field | Status | Notes |
|-------|--------|-------|
| `soft_prompts` | ✅ **IMPLEMENTED** | Training, saving, and packaging complete |
| `training.decoding.temperature` | ✅ **IMPLEMENTED** | Loaded from manifest in chat.rs |
| `training.decoding.top_p` | ✅ **IMPLEMENTED** | Loaded from manifest in chat.rs |
| `training.decoding.top_k` | ✅ **IMPLEMENTED** | Loaded from manifest in chat.rs |
| `training.config.use_sdpa` | ✅ **IMPLEMENTED** | SDPA + QLoRA enabled (v0.2.3+) |
| `training.config.packing` | ✅ **AUTO-DETECTED** | Auto-enabled with SFTTrainer |
| `training.config.max_seq_length` | ✅ **IMPLEMENTED** | Passed to SFTTrainer (not TrainingArguments) |
| `rope_scaling` (object) | ✅ **IMPLEMENTED** | Supports string + object formats |
| `runtime.*` | ✅ **METADATA** | Parsed and logged, not enforced yet |

### ⏳ Partially Implemented

| Field | Status | Notes |
|-------|--------|-------|
| `routing` | Future | For automatic expert selection |
| `training.decoding.grammar_file` | Packaged | Included in .expert but not enforced yet |
| `training.decoding.stop_sequences` | Parsed | Not enforced in generation yet |
| `runtime.attention_kernel` | Logged | Hints logged, not enforced yet |

### ❌ Not Implemented (Metadata for Future)

| Field | Purpose | Status |
|-------|---------|--------|
| `training.decoding.use_grammar` | Grammar validation | Parsed but not enforced |
| `training.decoding.validation` | Post-gen validation | Parsed but not enforced |
| `training.dataset.augmentation` | Data augmentation | Not implemented in expert_trainer.py |
| `evaluation` | Test metrics | Future validation feature |
| `runtime.requires_kv_cache_persistence` | KV cache hint | Logged only, not enforced |

## Adapter Types

### Implemented in expert_trainer.py

1. **LoRA** (Low-Rank Adaptation)
   - Requires: `rank`, `alpha`, `target_modules`
   - Size: ~15MB for r=12
   - Best for: General purpose

2. **DoRA** (Weight-Decomposed LoRA)
   - Requires: `rank`, `alpha`, `target_modules`, `scaling: "dora"`
   - Size: ~18MB for r=12
   - Best for: Complex tasks (SQL, code generation)
   - Quality: Better than LoRA at same rank

3. **IA³** (Infused Adapter)
   - Requires: `target_modules` (NO rank/alpha)
   - Size: ~2MB
   - Best for: Simple, repetitive tasks (JSON, formatting)
   - Speed: 10x faster loading than LoRA

### Not Yet Implemented

- **LoKr** (Low-Rank Kronecker)
- **AdaLoRA** (Adaptive LoRA with dynamic rank)

## RoPE Scaling Formats

### String Format (Legacy)
```json
"rope_scaling": "yarn-128k"
```

### Object Format (Recommended for Qwen3)
```json
"rope_scaling": {
  "type": "ntk-by-parts",
  "factor": 8.0,
  "max_position_embeddings": 32768,
  "original_max_position_embeddings": 8192,
  "fine_grained": true
}
```

**Note**: This is metadata. Actual RoPE implementation is in `expert/cli/src/inference/qwen3_model.rs` (hardcoded NTK-by-parts with β=0.25).

## Target Modules (Qwen3-0.6B)

### Safe for Adaptation
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention projections)
- `gate_proj`, `up_proj`, `down_proj` (MLP layers)

### Do NOT Adapt (Causes Instability)
- `q_norm`, `k_norm` (per-head normalization)
- `input_layernorm`, `post_attention_layernorm` (layer norms)
- `embed_tokens`, `lm_head` (embeddings)

## Training Performance Optimizations (v0.2.3)

### SDPA (Flash Attention v2)

**Enable with**: `"use_sdpa": true` in training config

**Requirements**:
- CUDA device
- Works with QLoRA INT4/INT8 (v0.2.3+)

**Impact**: +15-20% throughput, 85-95% GPU utilization

**Note**: Previously blocked with quantization, now fully compatible.

### Sequence Packing (SFTTrainer)

**Automatic**: Enabled when dataset has "text" field

**Requirements**:
- Dataset format: `{"text": "..."}`
- `trl>=0.7.0` installed

**Impact**: +30-40% tokens/s (reduces padding waste from 30-40% to <5%)

**Fallback**: Uses standard Trainer if dataset lacks "text" field

**Config fields**:
```json
{
  "max_seq_length": 2048,
  "packing": true  // metadata only, auto-detected
}
```

### Combined Impact

- Training speed: **2x faster** (4hrs → 2hrs for 3 epochs on SQL dataset)
- GPU utilization: **85-95%** (vs 60-70% before)
- VRAM usage: **unchanged** (~8GB for Qwen3-0.6B INT4 + DoRA r=12)

### Dataloader Optimizations

```json
{
  "dataloader_num_workers": 8,
  "dataloader_pin_memory": true,
  "dataloader_prefetch_factor": 8,
  "dataloader_persistent_workers": true
}
```

**Impact**: Reduces CPU → GPU transfer bottleneck

**Best for**: Large datasets (>10k examples)

---

## Adapter Configuration Guidelines

### IA³ (Simple Tasks - JSON, Formatting)
```json
{
  "type": "ia3",
  "target_modules": ["k_proj", "v_proj", "down_proj"],
  "feedforward_modules": ["down_proj"],
  "scaling": "learned"
}
```
- No rank/alpha needed
- Targets K/V attention + MLP intermediate
- 5+ epochs (tiny model, cheap data)
- Higher LR (0.001 vs 0.0003)

### DoRA (Complex Tasks - SQL, Code, Cypher)
```json
{
  "type": "dora",
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
  "r": 12,
  "alpha": 24,
  "scaling": "dora",
  "dropout": 0.05
}
```
- Rank: 8-16 for SQL/TypeScript, 16-24 for Cypher
- Alpha: 2*r (standard for DoRA)
- Full module coverage (attention + MLP)

## Decoding Parameters (by Task)

| Task | Temperature | Top-p | Top-k | Stop Sequences |
|------|-------------|-------|-------|----------------|
| JSON | 0.1-0.2 | 0.9 | 50 | Grammar-enforced |
| SQL | 0.1 | 0.9 | 50 | [";", "\n\n"] |
| TypeScript | 0.3-0.4 | 0.95 | - | tsc validation |
| Cypher | 0.3-0.35 | 0.9 | 50 | EXPLAIN validation |

**Lower temperature = more deterministic** (critical for SQL/JSON)
**Higher temperature = more creative** (acceptable for natural language)

## Validation

Validate manifest against schema:
```bash
# Using expert-cli
expert-cli validate --expert ./expert-sql

# Using JSON Schema validator (future)
ajv validate -s schemas/expert-manifest.schema.json -d experts/expert-sql/manifest.json
```

## Examples

See `/expert/experts/` for real production examples:
- `expert-json/manifest.json` - IA³ + soft-prompts
- `expert-sql/manifest.json` - DoRA r=12 (optimized)
- `expert-typescript/manifest.json` - DoRA r=12 + soft-prompt
- `expert-neo4j/manifest.json` - DoRA r=20 (high capacity)

## Migration

See `/expert/docs/EXPERT_FORMAT.md` for migration guide from v1.0 to v2.0.

