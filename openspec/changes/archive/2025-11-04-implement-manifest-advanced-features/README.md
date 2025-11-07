# Implement Manifest Advanced Features

**Change ID**: `implement-manifest-advanced-features`  
**Status**: Proposal  
**Created**: 2025-11-04  

---

## Quick Summary

Currently, the expert manifest schema defines several advanced features that are **accepted but ignored**:

- ✅ **Soft Prompts**: Parsed and packaged, but **not trained**
- ✅ **Decoding Config**: Defined in manifest, but **hardcoded in runtime**
- ✅ **Runtime Hints**: Documented but **never read**

This change implements all three missing feature categories.

---

## Problem

Users configure parameters in `manifest.json`:
```json
{
  "soft_prompts": [{"name": "json_strict", "tokens": 32, ...}],
  "training": {
    "decoding": {
      "temperature": 0.1,
      "top_p": 0.9
    }
  }
}
```

**But**:
- Soft prompts are **skipped** during training
- Runtime uses `temp=0.7` instead of `0.1` from manifest
- No error, no warning, silent failure

**Result**: Confused users, wasted configuration effort, lower quality.

---

## Solution

### Phase 1: Soft Prompt Training (HIGH)
Train learnable prompt embeddings using PEFT PromptTuningConfig.

**Impact**: +5-10% accuracy on structured tasks (JSON, SQL)

### Phase 2: Decoding Config Loading (HIGH)
Read `training.decoding.*` from manifest in Rust runtime.

**Impact**: SQL uses correct temp=0.1, not hardcoded 0.7

### Phase 3: Runtime Hints (MEDIUM)
Use `runtime.*` fields for optimization (attention kernel, cache strategy).

**Impact**: Future-proofs for multi-expert loading

---

## Files

- `proposal.md` - Detailed design document
- `tasks.md` - 45 implementation tasks across 4 phases
- `README.md` - This file

---

## Timeline

**Estimated Effort**: 8-12 hours total

**Phases**:
1. Soft Prompts (3 hours)
2. Decoding Config (2.5 hours)
3. Runtime Hints (2 hours)
4. Documentation (0.5 hours)

---

## Dependencies

**Python**:
- PEFT library (already required, verify PromptTuning support)

**Rust**:
- No new dependencies

---

## Testing

Each phase includes integration tests:
- Phase 1: Train JSON expert with soft prompt, verify .pt file
- Phase 2: Chat with SQL expert, verify temp=0.1 used
- Phase 3: Validate expert, verify runtime hints logged

---

## Success Criteria

- [ ] All manifest parameters functional or documented as "planned"
- [ ] No silent parameter ignoring
- [ ] `expert-cli validate` checks all fields
- [ ] Documentation matches actual behavior

---

## Related Changes

- `optimize-expert-sql-qwen3-config` - Manifest config fixes (completed)
- `fix-qwen3-inference-implementation` - Runtime inference fixes (completed)

This change builds on those foundations by making manifests fully functional.

