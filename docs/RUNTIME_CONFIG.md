# Runtime Configuration Guide

Configuration options for HiveLLM expert runtime inference.

## Supported Features (v0.2.3)

### Flash Attention (SDPA)

**Status**: ✅ Implemented in training, metadata in runtime

**Configuration** (in manifest):
```json
{
  "training": {
    "config": {
      "use_sdpa": true,
      "bf16": true,
      "use_tf32": true
    }
  },
  "runtime": {
    "attention_kernel": "flash-v2"
  }
}
```

**Notes**:
- Training: SDPA enabled via PyTorch
- Rust runtime: Metadata logged but not enforced yet

### RoPE Scaling

**Status**: ✅ Implemented in Rust (NTK-by-parts)

**Configuration**:
```json
{
  "base_models": [{
    "rope_scaling": {
      "type": "ntk-by-parts",
      "factor": 8.0,
      "max_position_embeddings": 32768,
      "original_max_position_embeddings": 8192,
      "fine_grained": true
    }
  }]
}
```

**Implementation**: `expert/cli/src/inference/qwen3_model.rs:49-57`

**Notes**:
- Hardcoded β=0.25 for Qwen3
- Extends context to 128k tokens
- Matches HuggingFace Transformers behavior

### Decoding Parameters

**Status**: ✅ Implemented with 3-level priority

**Configuration**:
```json
{
  "training": {
    "decoding": {
      "temperature": 0.1,
      "top_p": 0.9,
      "top_k": 50,
      "stop_sequences": [";", "\n\n"],
      "use_grammar": true,
      "grammar_type": "sql-postgres"
    }
  }
}
```

**Priority System**:
1. CLI flags: `--temperature 0.2` (highest)
2. Manifest defaults: `decoding.temperature`
3. Hardcoded: `0.7` (fallback)

**CLI Usage**:
```bash
expert-cli chat --experts sql --temperature 0.1 --top-p 0.9
```

### KV Cache

**Status**: ✅ Basic implementation, persistence metadata

**Configuration**:
```json
{
  "runtime": {
    "requires_kv_cache_persistence": true
  }
}
```

**Implementation**: 
- Automatic in `qwen.rs` generation loop
- Cleared between generations to prevent context leakage

**Future** (not implemented):
- Paged KV-cache for memory efficiency
- KV cache quantization (int8)
- Cross-generation persistence

### Runtime Metadata

**Status**: ✅ Parsed and logged

**Configuration**:
```json
{
  "runtime": {
    "candle_compatible": true,
    "requires_kv_cache_persistence": true,
    "attention_kernel": "flash-v2"
  }
}
```

**Implementation**: `expert/cli/src/inference/qwen.rs:from_local_with_hints()`

**Current Behavior**:
- Metadata is logged during model loading
- Not enforced (for future optimizations)

## Recommended Configurations

### JSON Expert (Format Task)

```json
{
  "training": {
    "config": {
      "adapter_type": "ia3",
      "use_sdpa": true,
      "bf16": true
    },
    "decoding": {
      "temperature": 0.2,
      "use_grammar": true,
      "grammar_type": "gbnf"
    }
  },
  "runtime": {
    "candle_compatible": true
  }
}
```

### SQL Expert (Complex Queries)

```json
{
  "training": {
    "config": {
      "adapter_type": "dora",
      "rank": 12,
      "use_sdpa": true,
      "batch_size": 32
    },
    "decoding": {
      "temperature": 0.1,
      "top_p": 0.9,
      "top_k": 50,
      "use_grammar": true,
      "grammar_type": "sql-postgres"
    }
  },
  "runtime": {
    "candle_compatible": true,
    "requires_kv_cache_persistence": true,
    "attention_kernel": "flash-v2"
  }
}
```

### Code Expert (TypeScript)

```json
{
  "training": {
    "config": {
      "adapter_type": "dora",
      "rank": 12,
      "use_sdpa": true
    },
    "decoding": {
      "temperature": 0.4,
      "top_p": 0.95
    }
  }
}
```

## Performance Tuning

### GPU Optimization (RTX 4090)

```json
{
  "training": {
    "config": {
      "batch_size": 32,
      "gradient_accumulation_steps": 2,
      "use_sdpa": true,
      "bf16": true,
      "use_tf32": true,
      "optim": "adamw_torch_fused",
      "dataloader_num_workers": 8,
      "dataloader_pin_memory": true,
      "dataloader_prefetch_factor": 8,
      "dataloader_persistent_workers": true
    }
  }
}
```

**Expected**:
- VRAM: 3.75GB / 24GB (16%)
- GPU Util: 85-95%
- Training Speed: 2.5x faster

### Lower VRAM (RTX 3060 8GB)

```json
{
  "training": {
    "config": {
      "batch_size": 8,
      "gradient_accumulation_steps": 8,
      "gradient_checkpointing": true,
      "use_sdpa": true,
      "bf16": true
    }
  }
}
```

**Expected**:
- VRAM: 5-6GB / 8GB
- GPU Util: 70-80%
- Training Speed: 1.5x faster

## Troubleshooting

### High VRAM Usage

1. Reduce `batch_size`
2. Enable `gradient_checkpointing: true`
3. Reduce `max_seq_length` (2048 → 1536)

### Low GPU Utilization

1. Increase `batch_size`
2. Increase `dataloader_num_workers`
3. Enable `dataloader_persistent_workers: true`
4. Verify SDPA is enabled

### Poor Inference Quality

1. Lower `temperature` (0.7 → 0.3)
2. Add `top_k: 50` for more focused sampling
3. Enable `use_grammar: true` for structured outputs

## Future Features (Not Implemented)

- [ ] Paged KV-cache (memory efficient)
- [ ] KV cache quantization (int8)
- [ ] Multi-expert hot-swap with LRU cache
- [ ] Grammar-based constrained decoding (runtime)
- [ ] Expert pre-loading based on routing confidence

