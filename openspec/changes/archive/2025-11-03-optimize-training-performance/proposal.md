# Optimize Training Performance for RTX 4090

## Why

Training is severely bottlenecked by CPU/DataLoader overhead, with GPU utilization at only 8% despite 20.4GB VRAM usage. Root causes:

1. Tokenization on-the-fly consuming CPU cycles
2. Datasets potentially on slow storage or cache locations
3. Missing fast kernel optimizations (SDPA, TF32, BF16)
4. Inefficient dataloader configuration (low workers, no prefetch)
5. Small batch sizes with gradient checkpointing overhead
6. FP16 instead of BF16 (RTX 4090 Ada architecture optimized for BF16)

Current performance:
- GPU Utilization: 8%
- VRAM Usage: 20.4/24.0 GB (good, but underutilized)
- Bottleneck: CPU waiting on tokenization and data loading

## What Changes

Training pipeline optimizations across multiple dimensions:

### 1. Dataset Pre-tokenization
- Create scripts to find and consolidate HuggingFace dataset cache
- Pre-tokenize all datasets with multiprocessing (num_proc=8)
- Apply sequence packing (concatenate + chunk to fixed lengths)
- Save as Arrow format for fast mmap loading
- Organize in `datasets_optimized/` directory

### 2. Fast Kernel Activation
- Enable TF32 for matrix operations (free 2x speedup on Ampere/Ada)
- Activate SDPA flash attention (PyTorch 2.x native implementation)
- Switch from FP16 to BF16 (better numerical stability on RTX 4090)
- Update QLoRA config to use NF4 + double quantization with BF16 compute

### 3. Dataloader Optimization
- Increase workers from 4 to 8 (parallel data loading)
- Increase prefetch factor from 2 to 8 (larger queue of ready batches)
- Enable persistent workers (avoid process restart overhead)
- Enable group_by_length (reduce padding waste)

### 4. Training Configuration
- Increase max_seq_length from 1024 to 2048 (more FLOPs per step)
- Optimize batch size and gradient accumulation balance
- Disable gradient checkpointing where VRAM permits (remove recompute overhead)
- Use fused AdamW optimizer

### 5. Manifest Schema Extension
Add new optional fields to manifest training config:
- `bf16`: Boolean for BF16 training
- `use_tf32`: Boolean to enable TF32
- `use_sdpa`: Boolean to activate SDPA attention
- `pretokenized_cache`: Path to pre-tokenized dataset

**Non-breaking**: All changes are backward compatible. Existing manifests work unchanged.

## Impact

### Affected Components
- **Specs**: Manifest schema (backward-compatible additions)
- **Code**: `expert/cli/expert_trainer.py` (kernel activation, cache loading)
- **Manifests**: All 4 expert manifests (json, neo4j, sql, typescript)
- **Infrastructure**: New `datasets_optimized/` directory and scripts

### Dependencies
- PyTorch 2.x (SDPA support)
- BitsAndBytes (NF4 quantization)
- Transformers 4.35+ (SDPA integration)
- HuggingFace Datasets (Arrow format)

### Timeline
- Phase 1 (Dataset prep): 20-30 minutes
- Phase 2 (Trainer updates): 15 minutes
- Phase 3 (Manifest updates): 5 minutes
- Phase 4 (Testing): 10 minutes
- **Total**: 50-60 minutes implementation

### Expected Performance Improvement
**Before**:
- GPU: 8% utilization
- VRAM: 20.4 GB
- Tokens/sec: Baseline
- Training time per epoch: Baseline

**After**:
- GPU: 60-85% utilization (7-10x increase)
- VRAM: 18-22 GB (BF16 savings + no gradient checkpointing)
- Tokens/sec: 5-10x faster
- Training time per epoch: 5-10x faster

**Key improvements**:
- Pre-tokenization eliminates CPU overhead
- SDPA + TF32 + BF16 = 2-3x kernel speedup
- Sequence length 2048 = more work per batch
- Prefetch 8 = GPU never starves
- Group by length = less padding waste

### Constraints Validation
- VRAM: 18-22 GB target (fits RTX 4090 24GB comfortably)
- Context: 2048 tokens (within model capacity)
- Expert count: No change (still max 10)
- Latency: Inference unchanged (training-only optimization)

### Risks
1. **BF16 compatibility**: Some older GPUs don't support BF16
   - Mitigation: Keep FP16 fallback in code
2. **Pre-tokenization storage**: Requires additional disk space
   - Mitigation: ~2-5 GB per expert (manageable)
3. **VRAM with larger batches**: Might OOM on some configs
   - Mitigation: Gradient checkpointing toggle available

## Testing Plan

1. **Phase 1 validation**: Verify pre-tokenized datasets load correctly
2. **Phase 2 validation**: Confirm kernels activate (check logs for TF32/SDPA)
3. **Phase 3 validation**: Test with expert-json (smallest dataset)
4. **Benchmark**: Measure GPU utilization, tokens/sec, epoch time
5. **Regression**: Ensure model quality unchanged (validate on test set)

## Documentation Updates

- Update `docs/TRAINING_GUIDE.md` with optimization section
- Update `docs/CLI.md` with new manifest fields
- Update `EXPERT_FORMAT.md` with new optional config fields
- Add benchmark results to `STATUS.md`

## Success Criteria

- GPU utilization increases to 60-85%
- Training throughput increases by 5-10x
- No regression in model accuracy
- All 4 experts train successfully with new config
- Backward compatibility maintained (old manifests still work)

