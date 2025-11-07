# Proposal: Multi-Model Base Support

## Summary

Enable expert packages to support multiple base models (e.g., Qwen3-0.6B and Qwen3-1.5B) within a single expert repository, with separate packaging per model variant.

## Motivation

### Current Limitation

Currently, each expert is tied to a single base model:
- One expert = one base model = one set of weights
- To support multiple models, we need separate expert repositories
- This creates maintenance overhead and duplicated effort

### Problem Statement

When an expert's logic/capabilities are model-agnostic (e.g., JSON parsing, English grammar), we want to:
1. Train the same expert concept on multiple model sizes
2. Maintain a single repository with unified configuration
3. Share datasets, training configs, and soft prompts
4. Distribute efficiently (one package per user's model)

### Use Cases

**1. Multi-Size Support**
```
expert-json-parser/
├── weights/
│   ├── qwen3-0.6b/adapter.safetensors    # For small devices
│   └── qwen3-1.5b/adapter.safetensors    # For servers
```

**2. Multi-Quantization**
```
expert-medical/
├── weights/
│   ├── qwen3-0.6b-int4/    # Consumer GPUs
│   └── qwen3-0.6b-int8/    # Production servers
```

**3. Cross-Architecture (future)**
```
expert-english/
├── weights/
│   ├── qwen3-0.6b/
│   ├── phi-3-mini/
│   └── gemma-2b/
```

## Proposed Solution

### Schema v2.0 Changes

Introduce `base_models` (array) to replace `base_model` (object):

```json
{
  "schema_version": "2.0",
  "name": "expert-name",
  "base_models": [
    {
      "name": "Qwen3-0.6B",
      "sha256": "...",
      "adapters": [{ "path": "weights/qwen3-0.6b/adapter.safetensors", ... }]
    },
    {
      "name": "Qwen3-1.5B", 
      "sha256": "...",
      "adapters": [{ "path": "weights/qwen3-1.5b/adapter.safetensors", ... }]
    }
  ]
}
```

### Packaging Strategy

Generate **one .expert file per model**:
```bash
expert-cli package --model qwen3-0.6b  # → expert-name-qwen3-0.6b.v1.0.0.expert
expert-cli package --model qwen3-1.5b  # → expert-name-qwen3-1.5b.v1.0.0.expert
```

Each package contains:
- Manifest with only the selected base model
- Model-specific weights
- Shared resources (soft prompts, tokenizer deltas)

### Backward Compatibility

- Schema v1.0 experts (single `base_model`) continue to work
- v2.0 parsers detect `schema_version` field
- Migration is opt-in via version bump

## Benefits

✅ **Single Source of Truth**
- One repository per expert concept
- Unified datasets and training configs

✅ **Efficient Distribution**  
- Users download only their model variant
- No unnecessary weights in package

✅ **Easier Maintenance**
- Update capabilities once, retrain for all models
- Consistent behavior across model sizes

✅ **Future-Proof**
- Foundation for cross-architecture support
- Enables model-agnostic expert marketplace

## Impact Analysis

### Components Affected

| Component | Impact | Changes Required |
|-----------|--------|------------------|
| **Manifest Schema** | Medium | Add `schema_version`, `base_models` field |
| **expert-cli validate** | Medium | Support both v1.0 and v2.0 schemas |
| **expert-cli package** | High | Add `--model` flag, filter weights |
| **expert-cli install** | Medium | Auto-detect compatible variant |
| **Documentation** | Low | Update EXPERT_FORMAT.md |

### Breaking Changes

**None** - fully backward compatible:
- Existing v1.0 manifests parse as before
- New v2.0 manifests use explicit `schema_version: "2.0"`

### Migration Path

For existing experts:
1. Optional - keep using v1.0 schema
2. Or upgrade: bump schema version, restructure manifest

## Alternatives Considered

### Alternative 1: Separate Repositories Per Model
❌ Rejected - too much duplication

### Alternative 2: Runtime Model Detection
❌ Rejected - requires bundling all weights (bloated packages)

### Alternative 3: Package All Models in One .expert
❌ Rejected - wastes bandwidth for users

## Implementation Plan

See `tasks.md` for detailed breakdown.

**Estimated Effort**: 2-3 days  
**Priority**: Medium (improves DX, not blocking current features)

## Success Criteria

1. ✅ Single expert repository supports 2+ base models
2. ✅ Packaging generates separate .expert files per model
3. ✅ Installation auto-selects correct variant
4. ✅ All existing v1.0 experts continue to work
5. ✅ Validation passes for both v1.0 and v2.0 schemas

## References

- **EXPERT_FORMAT.md**: Complete manifest specification
- **CLI.md**: Packaging and installation commands
- **expert-json-parser**: First candidate for migration

