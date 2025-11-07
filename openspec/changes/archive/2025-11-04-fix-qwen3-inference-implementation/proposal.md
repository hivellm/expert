# Fix Qwen3 Inference Implementation (P0)

## Why

Current Qwen3 implementation loads model successfully but generates garbage output. Root causes:
- **Missing LM head** - no final projection layer from hidden states → vocabulary logits
- **Mock forward pass** - `forward_single()` returns hardcoded patterns instead of real inference
- **Basic RoPE** - missing NTK-by-parts scaling required for Qwen3's long context (>32k tokens)
- **Greedy-only sampling** - temperature/top-p/top-k not implemented, limiting generation quality

User reports show model loads correctly (28 layers, 16 heads, 151936 vocab) on CUDA but generates repetitive nonsense like "vecunovecunovecuno..." or random code snippets.

## What Changes

### Critical Fixes
- Add `lm_head` Linear layer to `Qwen3Model` (hidden_size → vocab_size projection)
- Implement real `forward_single()`: embedding → transformer layers → norm → lm_head
- Replace basic RoPE with NTK-by-parts scaling (β=0.25 for contexts >32k)
- Implement proper sampling: temperature scaling + top-p nucleus sampling
- Verify SafeTensors weight names match VarBuilder paths (especially `lm_head.weight`)

### Qwen3-Specific Optimizations
- Validate GQA implementation (16 heads, 2 KV heads, 8 groups) - already correct
- Ensure q_norm/k_norm are applied per-head (critical for Qwen3 stability)
- Handle tied embeddings (`tie_word_embeddings: true` → lm_head shares embed_tokens weights)

### LoRA Preparation
- Add `get_layer_mut()` for adapter injection
- Document LoRA target modules: `{q,k,v,o}_proj` + `{gate,up,down}_proj`
- Exclude norm layers and embeddings from LoRA (Qwen3 best practice)

**BREAKING**: None - fixes broken inference, no API changes

## Impact

- **Affected specs**: runtime-core, expert-composition
- **Affected code**: 
  - `expert/cli/src/inference/qwen3_model.rs` (core model implementation)
  - `expert/cli/src/inference/qwen.rs` (sampling and generation)
  - `expert/cli/Cargo.toml` (add `rand` dependency)
- **Dependencies**: `rand = "0.8"` for probabilistic sampling
- **Timeline**: 1-2 days (P0 blocker - chat doesn't work)
- **Performance**: Enables functional inference + better output quality with sampling
- **Quality**: Proper sampling prevents repetition, improves coherence

## Technical Details

### NTK-by-parts RoPE Scaling
Qwen3 uses non-linear scaling for long contexts:
```rust
let base = if max_seq_len > 32768 {
    let beta = 0.25; // Qwen3-specific parameter
    rope_theta * ((max_seq_len / 32768.0).powf(beta))
} else {
    rope_theta
};
```

### Sampling Implementation
Temperature + top-p nucleus sampling:
1. Apply temperature: `logits /= temperature`
2. Softmax to probabilities
3. Sort by probability, accumulate until cumsum > top_p threshold
4. Zero out low-prob tokens
5. Re-normalize and sample from distribution

### Weight Loading Verification
Check if `lm_head.weight` exists in SafeTensors, or if it's tied to `model.embed_tokens.weight` (common in decoder-only models).

## References

- Qwen3 official repo: https://github.com/QwenLM/Qwen3
- NTK-aware RoPE: https://arxiv.org/abs/2309.00071
- Nucleus sampling: https://arxiv.org/abs/1904.09751
- LoRA best practices: avoid injecting into normalization layers

