# Optimal Binary Expert Format

**Change ID**: `implement-osd-binary-expert-format`  
**Status**: Proposal  
**Created**: 2025-11-05  

---

## Quick Summary

Binary `.expert` format with OSD adapters for:
- **2-5x faster loading** (<200ms vs ~500ms tar.gz)
- **30-40% smaller packages** (35 MB vs 60 MB)
- **+5-10% better accuracy** (OSD vs LoRA at same storage)
- **Streaming support** (on-demand layer loading)

---

## Problem

Current tar.gz format:
```
expert-sql.expert (60 MB)
├── Full extraction required (~500ms)
├── Pure LoRA (suboptimal storage/accuracy)
└── No streaming support
```

---

## Solution

Binary format with TOC + OSD:
```
expert-sql.expert (35 MB)
├── Header + TOC (instant lookup)
├── OSD adapters (SVD + sparse mask)
├── Streaming-ready sections
└── SHA256 per section + signing
```

**OSD**: `Δ = U·Σ·Vᵀ ⊙ M_sparse` (preserves critical singular vectors)

---

## Implementation

See [tasks.md](./tasks.md) for 126 detailed tasks across 7 phases:
1. Format spec (2 weeks)
2. Binary writer (2 weeks)
3. Binary reader (2 weeks)
4. OSD support (4 weeks)
5. Integration (2 weeks)
6. Signing (1 week, optional)
7. Documentation (1 week)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Load time | <200ms (2.5x faster) |
| Package size | 30-50 MB (40% smaller) |
| Accuracy (OSD) | +5-10% vs LoRA |
| VRAM | <50 MB/expert |

---

## Files

- `proposal.md` - Detailed design
- `tasks.md` - Implementation tasks
- `README.md` - This file

---

## Testing

A/B test OSD vs LoRA on expert-sql:
```bash
# Train LoRA
expert-cli train --adapter-type lora --rank 12

# Train OSD  
expert-cli train --adapter-type osd --rank 16 --sparsity 0.05

# Compare accuracy and size
```

---

## References

- Alipour & Amiri (2025). "Optimal Singular Damage: Efficient LLM Inference in Low Storage Regimes"
- Current packaging: `improve-expert-packaging` change
