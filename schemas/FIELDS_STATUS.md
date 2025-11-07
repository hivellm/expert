# Training Config Fields - Status Check

Generated: 2025-11-05

## Field Mapping Status

| Field Name | Manifest JSON | Rust struct | Python trainer | JSON Schema | Status |
|------------|---------------|-------------|----------------|-------------|--------|
| **Core** |
| method | ✅ | ✅ | ✅ | ✅ | ✅ |
| adapter_type | ✅ | ✅ | ✅ | ✅ | ✅ |
| use_unsloth | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| rank | ✅ | ✅ | ✅ | ✅ | ✅ |
| alpha | ✅ | ✅ | ✅ | ✅ | ✅ |
| target_modules | ✅ | ✅ | ✅ | ✅ | ✅ |
| feedforward_modules | - | ✅ | ✅ | ✅ | ✅ (IA³ only) |
| **Hyperparameters** |
| epochs | ✅ | ✅ | ✅ | ✅ | ✅ |
| learning_rate | ✅ | ✅ | ✅ | ✅ | ✅ |
| batch_size | ✅ | ✅ | ✅ | ✅ | ✅ |
| gradient_accumulation_steps | ✅ | ✅ | ✅ | ✅ | ✅ |
| warmup_steps | ✅ | ✅ | ✅ | ✅ | ✅ |
| warmup_ratio | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| lr_scheduler | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Checkpointing** |
| save_strategy | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| save_steps | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| save_total_limit | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| evaluation_strategy | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| eval_steps | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| load_best_model_at_end | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| metric_for_best_model | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| greater_is_better | ✅ | ✅ | ✅ | ✅ | ✅ NEW |
| logging_steps | ✅ | ✅ | ✅ | ✅ | ✅ |
| **DataLoader** |
| max_seq_length | ✅ | ✅ | ✅ | ✅ | ✅ |
| dataloader_num_workers | ✅ | ✅ | ✅ | ✅ | ✅ |
| dataloader_pin_memory | ✅ | ✅ | ✅ | ✅ | ✅ |
| dataloader_prefetch_factor | ✅ | ✅ | ✅ | ✅ | ✅ |
| dataloader_persistent_workers | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Precision/Optimization** |
| fp16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| bf16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| use_tf32 | ✅ | ✅ | ✅ | ✅ | ✅ |
| use_sdpa | ✅ | ✅ | ✅ | ✅ | ✅ |
| flash_attention_2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| memory_efficient_attention | ✅ | ✅ | ✅ | ✅ | ✅ |
| packing | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Compilation** |
| torch_compile | ✅ | ✅ | ✅ | ✅ | ✅ |
| torch_compile_backend | ✅ | ✅ | ✅ | ✅ | ✅ |
| torch_compile_mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Advanced** |
| optim | ✅ | ✅ | ✅ | ✅ | ✅ |
| group_by_length | ✅ | ✅ | ✅ | ✅ | ✅ |
| gradient_checkpointing | ✅ | ✅ | ✅ | ✅ | ✅ |
| activation_checkpointing | ✅ | ✅ | ✅ | ⚠️ | ⚠️ Missing |
| use_cuda_graphs | ✅ | ✅ | ✅ | ⚠️ | ⚠️ Missing |
| cuda_graph_warmup_steps | ✅ | ✅ | ✅ | ⚠️ | ⚠️ Missing |
| memory_clear_every | ✅ | ✅ | ✅ | ⚠️ | ⚠️ Missing |
| pretokenized_cache | - | ✅ | ✅ | ✅ | ✅ |

## Summary

**Total fields**: 45  
**Fully synced**: 41 ✅  
**Missing in schema**: 4 ⚠️
- activation_checkpointing
- use_cuda_graphs
- cuda_graph_warmup_steps
- memory_clear_every

## Action Required

Build the Rust CLI to apply changes:
```bash
cd expert/cli
cargo build --release
```

Then test training to verify checkpoints save correctly at `weights/qwen3-06b/checkpoint-250`.

