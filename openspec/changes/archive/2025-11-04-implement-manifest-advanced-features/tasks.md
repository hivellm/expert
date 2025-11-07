# Implement Advanced Manifest Features

## 1. Soft Prompt Training

- [x] 1.1 Import PromptTuningConfig from peft
- [x] 1.2 Add configure_soft_prompts() function
- [x] 1.3 Integrate in training pipeline
- [x] 1.4 Save soft prompt embeddings after training
- [x] 1.5 Update packaging for soft prompts

## 2. Decoding Parameter Loading

- [x] 2.1 Add DecodingConfig struct to manifest.rs
- [x] 2.2 Load decoding config in chat.rs from manifest
- [x] 2.3 Add CLI flags (--temperature, --top-p, --top-k, --max-tokens)
- [x] 2.4 Implement 3-level priority: CLI > manifest > default
- [x] 2.5 Log parameter sources

## 3. Runtime Metadata Usage

- [x] 3.1 Add Runtime struct to manifest.rs
- [x] 3.2 Add runtime field to Manifest
- [x] 3.3 Implement from_local_with_hints() in qwen.rs
- [x] 3.4 Log attention_kernel hints
- [x] 3.5 Log kv_cache_persistence flag

## 4. Documentation

- [x] 4.1 Update CHANGELOG.md for v0.2.3
- [x] 4.2 Update IMPLEMENTATION_STATUS.md
- [x] 4.3 Add integration tests (manifest_feature_tests.rs)
