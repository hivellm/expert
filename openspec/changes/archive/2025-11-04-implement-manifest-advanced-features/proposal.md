# Implement Advanced Manifest Features

**Status**: Proposal  
**Priority**: P1 (High - completes manifest feature coverage)  
**Estimated Effort**: 8-12 hours  
**Author**: AI Assistant  
**Date**: 2025-11-04

---

## Problem Statement

The expert manifest schema defines several advanced features that are **parsed but not implemented**:

1. **Soft Prompts**: Accepted in manifest, packaged in `.expert` files, but **not trained** by `expert_trainer.py`
2. **Decoding Parameters**: Defined in manifest (`temperature`, `top_p`, `top_k`, `grammar`, `stop_sequences`) but **hardcoded in Rust runtime** (`chat.rs` uses fixed values)
3. **Runtime Metadata**: `runtime.*` fields exist but are **never read** by the Rust inference engine

This creates confusion:
- Users configure `"temperature": 0.1` in manifest → Runtime ignores it and uses 0.7
- Users add soft prompts → Training skips them silently
- Users set `"attention_kernel": "flash-v2"` → No effect

**Impact**:
- ❌ Manifests lie about behavior (say one thing, do another)
- ❌ Users waste time configuring ignored parameters
- ❌ Soft prompts unusable (10% potential quality improvement lost)
- ❌ Each expert needs custom runtime builds to change decoding params

---

## Proposed Solution

Implement 3 missing feature categories in order of priority:

### 1. Soft Prompt Training (HIGH Priority)

**What**: Train learnable prompt embeddings alongside adapters

**Why**: 
- Soft prompts improve task adherence (JSON formatting, SQL style)
- Tiny VRAM cost (~0.5MB per 48 tokens)
- Already designed in manifest schema

**How**:
- Extend `expert_trainer.py` to use `PromptTuningConfig` from PEFT
- Initialize embeddings from text or random
- Train jointly with LoRA/DoRA/IA³
- Save as `.pt` files in expert package

### 2. Decoding Parameter Loading (MEDIUM Priority)

**What**: Read `training.decoding.*` from manifest at runtime

**Why**:
- SQL needs temp=0.1, TypeScript needs temp=0.4 (currently both get 0.7)
- Grammar validation already configured but unused
- Stop sequences prevent over-generation

**How**:
- Add `DecodingConfig` struct to Rust `manifest.rs`
- Load in `commands/chat.rs` when expert is specified
- Override hardcoded defaults with manifest values
- Implement grammar validation (future work)

### 3. Runtime Metadata Usage (LOW Priority)

**What**: Use `runtime.*` fields for optimization hints

**Why**:
- `attention_kernel` could select best kernel for model
- `requires_kv_cache_persistence` enables cross-turn memory
- Future-proofs for multi-expert loading

**How**:
- Read in `inference/qwen.rs` during engine initialization
- Apply hints to model loading (kernel selection, cache strategy)
- Document which hints are active vs planned

---

## Detailed Changes

### Soft Prompt Training

**Files Modified**:
- `expert/cli/expert_trainer.py`
- `expert/cli/requirements.txt` (ensure PEFT version supports PromptTuning)

**Implementation**:
```python
from peft import PromptTuningConfig, get_peft_model

def configure_soft_prompts(model, config, tokenizer):
    """Configure soft prompts if specified in manifest"""
    if not hasattr(config, 'soft_prompts') or not config.soft_prompts:
        return model
    
    # Currently only support single soft prompt during training
    soft_prompt = config.soft_prompts[0]
    
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=soft_prompt.tokens,
        prompt_tuning_init="TEXT" if soft_prompt.init_text else "RANDOM",
        prompt_tuning_init_text=soft_prompt.init_text if soft_prompt.init_text else None,
        tokenizer_name_or_path=config.base_model_name,
    )
    
    # Apply soft prompt + adapter together
    model = get_peft_model(model, prompt_config)
    
    return model
```

**Manifest Integration**:
```json
"soft_prompts": [
  {
    "name": "json_strict",
    "path": "soft_prompts/json_strict_32.pt",
    "tokens": 32,
    "init_method": "text",
    "init_text": "Generate valid, compact JSON. Follow schema exactly.",
    "purpose": "Enforce JSON formatting conventions"
  }
]
```

### Decoding Parameter Loading

**Files Modified**:
- `expert/cli/src/manifest.rs` (add DecodingConfig struct - already exists but unused)
- `expert/cli/src/commands/chat.rs` (load from manifest)
- `expert/cli/src/inference/qwen.rs` (accept config in generate())

**Implementation**:
```rust
// manifest.rs - struct already exists, just needs to be used

// chat.rs
let decoding_config = if let Some(ref manifest) = expert_manifest {
    manifest.training.decoding.clone()
} else {
    DecodingConfig::default()
};

let gen_config = GenerationConfig {
    max_tokens: 512,
    temperature: decoding_config.temperature.unwrap_or(0.7),
    top_p: decoding_config.top_p,
    top_k: decoding_config.top_k,
    stop_sequences: decoding_config.stop_sequences,
    ..Default::default()
};
```

**Fallback Strategy**:
- If no expert loaded: Use hardcoded defaults (current behavior)
- If expert loaded: Use manifest decoding config
- CLI flags override manifest (e.g., `--temperature 0.5` wins)

### Runtime Metadata Usage

**Files Modified**:
- `expert/cli/src/inference/qwen.rs`
- `expert/cli/src/inference/qwen3_model.rs`

**Implementation**:
```rust
// qwen.rs - QwenEngine::from_local()
if let Some(ref runtime_config) = manifest.runtime {
    if runtime_config.attention_kernel == Some("flash-v2") {
        // Prefer flash attention if available
        model_kwargs.attn_implementation = "flash-v2";
    }
    
    if runtime_config.requires_kv_cache_persistence {
        // Don't clear cache between turns in chat mode
        self.persistent_cache = true;
    }
}
```

---

## Implementation Plan

### Phase 1: Soft Prompt Training (Week 1)

**Tasks**:
1. Add PromptTuningConfig support to expert_trainer.py
2. Handle init_method: "text" | "random" | "vocab_sample"
3. Save trained prompt embeddings as `.pt` files
4. Update packaging to include soft prompt files
5. Test with JSON expert (32 token prompt)

**Success Criteria**:
- `expert-cli train` with soft_prompts in manifest succeeds
- `.expert` package contains `soft_prompts/*.pt`
- Training logs show prompt tuning parameters

### Phase 2: Decoding Parameter Loading (Week 2)

**Tasks**:
1. Add DecodingConfig loading in chat.rs
2. Pass to GenerationConfig construction
3. Implement CLI flag overrides (--temperature, --top-p)
4. Add validation for parameter ranges
5. Test with SQL (temp=0.1) vs TypeScript (temp=0.4)

**Success Criteria**:
- SQL expert generates with temp=0.1 (from manifest)
- CLI `--temperature 0.5` overrides manifest
- Help text documents parameter sources

### Phase 3: Runtime Metadata (Week 3)

**Tasks**:
1. Read runtime.* in QwenEngine::from_local()
2. Apply attention_kernel hint
3. Implement KV cache persistence flag
4. Document active vs planned hints
5. Add to expert validation checks

**Success Criteria**:
- Experts with flash-v2 hint use optimized kernel
- Chat mode respects kv_cache_persistence
- Validation warns if unsupported hints used

---

## Testing Strategy

### Soft Prompts
```bash
# Train JSON expert with soft prompt
cd expert/experts/expert-json
expert-cli train

# Verify soft prompt in package
tar -tzf expert-json-0.0.1.expert | grep soft_prompts

# Test inference (future: load soft prompts in Rust)
expert-cli chat --expert json
```

### Decoding Parameters
```bash
# SQL expert should use temp=0.1 from manifest
expert-cli chat --expert sql --prompt "SELECT * FROM users WHERE"

# Override with CLI
expert-cli chat --expert sql --temperature 0.5 --prompt "..."

# Verify in logs: "Using temperature: 0.1 (from manifest)"
```

### Runtime Metadata
```bash
# Validate expert checks runtime compatibility
expert-cli validate --expert sql

# Should log: "Runtime hints: flash-v2, kv_cache_persistence"
```

---

## Risks and Mitigations

### Risk 1: Soft Prompts Break Existing Training

**Mitigation**: Make soft prompts optional (only if specified in manifest)

### Risk 2: Decoding Params Cause Regressions

**Mitigation**: 
- Keep hardcoded defaults as fallback
- Add `--ignore-manifest-decoding` flag for testing
- Validate parameter ranges (temp 0.0-2.0, top_p 0.0-1.0)

### Risk 3: Performance Impact

**Mitigation**:
- Soft prompts: Minimal (~0.5MB, <1% slowdown)
- Decoding config load: One-time at startup
- Runtime hints: Zero cost (compile-time optimization)

---

## Success Metrics

**Completeness**:
- ✅ All manifest fields either implemented OR documented as planned
- ✅ No silent parameter ignoring

**Quality**:
- JSON expert accuracy +5-10% with soft prompt
- SQL queries use correct temperature (0.1 not 0.7)
- Inference speed unchanged

**User Experience**:
- Manifest behavior matches documentation
- CLI validates all manifest fields
- Clear error messages for unsupported features

---

## Future Work

Beyond this proposal:

1. **Grammar-Based Decoding**: Implement GBNF validation (planned in manifest)
2. **Multi-Soft-Prompt**: Support multiple prompts per expert
3. **Soft Prompt Inference (Rust)**: Load `.pt` files in Candle runtime
4. **Adaptive Decoding**: Adjust temperature based on confidence
5. **Expert-Specific Kernels**: Compile optimized kernels per expert

---

## References

- PEFT PromptTuning: https://github.com/huggingface/peft/tree/main/src/peft/tuners/prompt_tuning
- Soft Prompts Paper: "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)
- Current Manifest Schema: `expert/schemas/expert-manifest.schema.json`
- Implementation Status: `expert/schemas/IMPLEMENTATION_STATUS.md`

