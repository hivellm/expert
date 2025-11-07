# Optimal Binary Expert Format (.expert v2)

**Status**: Proposal  
**Priority**: P4 (Post-MVP, Research/Beta)  
**Estimated Effort**: 12-14 weeks  
**Date**: 2025-11-05

---

## Problem Statement

Current `.expert` format (tar.gz) has limitations:
- **Slow loading**: ~500ms full extraction required
- **Storage inefficient**: Pure LoRA/DoRA suboptimal for storage vs accuracy
- **No streaming**: Can't load layers on-demand from disk
- **No per-section integrity**: Only global SHA256
- **No hot-validation**: Must extract to validate

Recent research ([Alipour & Amiri, Nov 2025](https://arxiv.org/abs/2025.xxxxx)):
- **OSD (Optimal Singular Damage)**: Low-rank + selective sparsity
- Preserves critical singular vectors
- **+5-10% accuracy** at same storage vs pure LoRA
- Ideal for marketplace with diverse adapter types

---

## Proposed Solution

Binary `.expert` v2 format with:

1. **Memory-mapped TOC**: O(1) section lookup without decompression
2. **Multi-adapter support**: LoRA, DoRA, IA³, OSD in same runtime
3. **Streaming**: Load layers on-demand (SSD→RAM→GPU)
4. **Per-section compression**: zstd with 64/256-byte alignment
5. **Enhanced integrity**: SHA256 global + per-section + optional Ed25519
6. **Backward compatible**: Auto-detect tar.gz vs binary

### Binary Layout

```
Offset  Size    Field
0x00    8       MAGIC = "HLLMEXP\0"
0x08    4       VERSION (u16 major, u16 minor)
0x0C    4       HEADER_SIZE (u32)
0x10    4       TOC_COUNT (u32)
0x14    32      FILE_SHA256
0x34    64      SIG_ED25519 (optional, zeros if unused)
0x74    ...     METADATA_JSON (zstd)
...     ...     TOC[n]
...     ...     SECTIONS (compressed blobs)
```

### OSD Adapter Format

```rust
struct OSDModule {
    rank: u16,                // Low-rank dimension
    in_features: u32,
    out_features: u32,
    offset_U: u64,            // SVD factors
    offset_S: u64,
    offset_Vt: u64,
    sparse_format: u32,       // CSR/CSC
    offset_sp_indices: u64,   // Sparse mask
    offset_sp_values: u64,
    sparse_weight: f32,       // Blend [0,1]
}
```

---

## Benefits

1. **2-5x faster loading**: <200ms cold, <50ms hot (vs ~500ms tar.gz)
2. **30-40% smaller**: OSD optimization (30-50 MB vs 50-80 MB)
3. **Better accuracy**: +5-10% with OSD vs LoRA at same storage
4. **Streaming capable**: On-demand layer loading
5. **Future-proof**: Multi-adapter support in same runtime

---

## Implementation Plan

### Phase 1: Format Spec (2 weeks)
- Binary layout design
- Rust structs (#[repr(C)])
- Section types enum
- Metadata JSON schema

### Phase 2: Writer (2 weeks)
- Binary serialization
- zstd compression pipeline
- LoRA/DoRA encoding
- SHA256 calculation

### Phase 3: Reader (2 weeks)
- Memory-mapped reader
- TOC parser
- Section decompression
- Validation logic

### Phase 4: OSD (4 weeks)
- SVD decomposition (scipy)
- Sparsity mask computation
- Importance scoring
- Runtime fusion

### Phase 5: Integration (2 weeks)
- ExpertManager integration
- Auto-format detection
- tar.gz migration tools
- Benchmarks & A/B tests

---

## Success Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Load time (cold) | 500ms | <200ms | **2.5x** |
| Load time (hot) | 150ms | <50ms | **3x** |
| Package size | 60 MB | 35 MB | **40%** |
| Accuracy (OSD) | - | +5-10% | **vs LoRA** |
| VRAM/expert | 80 MB | <50 MB | **37%** |

---

## Backward Compatibility

- Auto-detect format (binary vs tar.gz)
- Keep tar.gz support for 2-3 releases
- Migration tool: `expert-cli convert`
- Deprecation timeline in ROADMAP.md

---

## Testing

### Unit Tests
- Binary serialization/deserialization
- TOC lookup correctness
- Compression round-trip
- SHA256 validation

### Integration Tests
- Load binary .expert in runtime
- Multi-expert with mixed formats
- tar.gz fallback
- Corruption detection

### Performance Tests
- Load time benchmarks
- VRAM profiling
- Streaming bandwidth

### Quality Tests
- A/B: OSD vs LoRA accuracy on SQL
- A/B: Package size vs accuracy trade-off
- Regression: No degradation vs current

---

## Risks

| Risk | Mitigation |
|------|------------|
| OSD too complex | Start with LoRA/DoRA, add OSD in Phase 4 |
| Performance regression | Benchmark early, fallback to tar.gz |
| Breaking changes | Keep tar.gz, migration tool, auto-detect |
| Storage trade-offs | A/B test, per-expert selection |

---

## Open Questions

1. OSD hyperparameters: Auto-tune rank + sparsity per expert?
2. Compression level: zstd 6 vs 10 for different sizes?
3. Signing: Mandatory for marketplace or optional?
4. Migration timeline: How long to support tar.gz?

---

## Files

- `proposal.md` - This file
- `tasks.md` - 126 implementation tasks
- `README.md` - Quick overview
- `specs/binary-format/spec.md` - Detailed binary specification (to be created)

---

## Next Steps

1. Review proposal
2. Run OSD pilot on expert-sql (compare vs current LoRA)
3. Prototype binary writer (Phase 2)
4. Benchmark load times
5. Decision: Go/no-go based on results
