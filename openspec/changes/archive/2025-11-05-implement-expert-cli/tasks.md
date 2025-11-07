# Implementation Tasks

## 1. Project Setup ✅ COMPLETE

- [x] 1.1 Create Rust workspace at `/expert/cli/` ✅
- [x] 1.2 Configure Cargo.toml (edition 2024, dependencies) ✅
- [x] 1.3 Add clap for CLI parsing ✅
- [x] 1.4 Add tokio for async runtime ✅
- [x] 1.5 Setup PyO3 bindings for Python/PyTorch ✅
- [x] 1.6 Add ed25519-dalek for signing ✅
- [x] 1.7 Add serde, serde_json for manifest parsing ✅
- [x] 1.8 Add semver for version checking ✅ (via version field)
- [x] 1.9 Add basic CLI structure with subcommands ✅

## 2. Manifest Schema Support (v1.0 & v2.0) ✅ COMPLETED

- [x] 2.1 Create manifest.rs module
- [x] 2.2 Define Manifest struct with optional base_model/base_models
- [x] 2.3 Define BaseModelV2 struct (base_models: Vec<BaseModelV2>)
- [x] 2.4 Implement schema_version detection (default "1.0")
- [x] 2.5 Add validation: cannot have both base_model and base_models
- [x] 2.6 Implement BaseModel and BaseModelV2 structs
- [x] 2.7 Add manifest deserializer supporting both versions
- [x] 2.8 Add unit tests for v1.0 manifest parsing (18 tests)
- [x] 2.9 Add unit tests for v2.0 manifest parsing (7 tests)
- [x] 2.10 Add validation tests for schema conflicts (all error cases)

**Commits**: bc1ca4e, 765fcbf | **Tests**: 25 unit + 4 integration ✅

## 3. Dataset Generation Command ✅ PARTIALLY COMPLETE

- [x] 3.1 Implement `expert-cli dataset generate --manifest` ✅ (guidance only)
- [x] 3.2 Parse manifest.json training.dataset.generation section ✅
- [x] 3.3 Support both v1.0 and v2.0 manifests ✅
- [ ] 3.4 Integrate Cursor Agent API (future - use Python scripts)
- [ ] 3.5 Integrate DeepSeek API (future - use Python scripts)
- [ ] 3.6 Integrate Claude API (Anthropic) (future - use Python scripts)
- [ ] 3.7 Implement batch generation (100 examples per call) (future)
- [ ] 3.8 Add diversity filtering (embedding-based) (future)
- [ ] 3.9 Add deduplication (exact + fuzzy) (future)
- [ ] 3.10 Save as JSONL format (future)
- [ ] 3.11 Add progress bar and statistics (future)

**Dataset Validate Command:** ✅ **FULLY IMPLEMENTED**
- [x] JSONL format validation ✅
- [x] JSON parsing line-by-line ✅
- [x] Field analysis ✅
- [x] Error reporting ✅

**Status:** Validate complete, generate provides guidance (Python scripts recommended for LLM calls)

## 4. Training Command ✅ COMPLETE

- [x] 4.1 Implement `expert-cli train --manifest` ✅
- [x] 4.2 Parse manifest.json training.config section ✅
- [x] 4.3 Support both v1.0 and v2.0 manifests ✅
- [ ] 4.4 Add `--model <name>` flag for v2.0 multi-model training (future)
- [x] 4.5 Create PyO3 bridge to PyTorch/PEFT ✅ (via subprocess)
- [x] 4.6 Load base model (Qwen3-0.6B or Qwen2.5-0.5B) ✅
- [x] 4.7 Configure LoRA adapter from manifest ✅
- [x] 4.8 Load and tokenize JSONL dataset ✅ (HF datasets + pre-tokenization)
- [x] 4.9 Setup training loop (epochs, batch size, lr) ✅
- [x] 4.10 Save adapter weights to model-specific directory (v2.0) ✅
- [x] 4.11 Update manifest with training metadata ✅
- [x] 4.12 Add training progress reporting ✅

**Implementation:** Fully functional via `expert_trainer.py` called by Rust CLI

## 5. Validation Command ✅ PARTIALLY COMPLETE

- [x] 5.1 Implement `expert-cli validate --expert` ✅
- [x] 5.2 Detect manifest schema version ✅
- [x] 5.3 Validate v1.0 manifest structure ✅
- [x] 5.4 Validate v2.0 manifest structure ✅
- [x] 5.5 Check for schema conflicts (both base_model and base_models) ✅
- [x] 5.6 Validate base_models array is non-empty (v2.0) ✅
- [x] 5.7 Validate weight paths are unique across models (v2.0) ✅
- [x] 5.8 Validate shared resources exist ✅
- [x] 5.9 Load expert weights ✅ (path validation)
- [ ] 5.10 Read tests/test_cases.json (future enhancement)
- [ ] 5.11 Run inference on test cases (future enhancement)
- [ ] 5.12 Compare outputs with expected (future enhancement)
- [ ] 5.13 Calculate accuracy metrics (future enhancement)
- [x] 5.14 Check base model compatibility ✅
- [x] 5.15 Report validation results ✅

**Implementation:** Structural validation complete, inference tests pending

## 6. Packaging Command ✅ COMPLETE

- [x] 6.1 Implement `expert-cli package --manifest` ✅
- [x] 6.2 Add `--model <name>` flag for v2.0 multi-model experts ✅
- [x] 6.3 Read manifest.json and detect schema version ✅
- [x] 6.4 For v1.0: package single model as before ✅
- [x] 6.5 For v2.0 without --model: error (ambiguous) ✅
- [x] 6.6 For v2.0 with --model: filter to selected model ✅
- [x] 6.7 Create filtered manifest (only selected model) ✅
- [x] 6.8 Collect model-specific adapter weights (validation) ✅
- [x] 6.9 Collect shared resources (soft prompts, license) ✅
- [x] 6.10 Generate model-specific filename (v2.0) ✅
  - Format: `expert-name-model.vX.Y.Z.expert`
- [x] 6.11 Compute SHA-256 hashes for included files only ✅
- [x] 6.12 Create tar.gz archive ✅
- [x] 6.13 Save as .expert file ✅
- [x] 6.14 Verify package integrity ✅

**Commits**: d0560b0, e68a96d | **Status**: Fully implemented and tested

## 7. Signing Command ✅ COMPLETE

- [x] 7.1 Implement `expert-cli sign --expert` ✅
- [x] 7.2 Load .expert package ✅
- [x] 7.3 Compute file hashes for all files in package ✅
- [x] 7.4 Load Ed25519 private key ✅
- [x] 7.5 Sign hash manifest ✅
- [x] 7.6 Update manifest with signature ✅
- [x] 7.7 Re-package .expert file with signature ✅
- [x] 7.8 Add `expert-cli keygen` for key generation ✅

**Commits**: 0bf7a55 | **Status:** Fully implemented and tested

## 8. Installation Command ✅ COMPLETE

- [x] 8.1 Implement `expert-cli install <git-url>` ✅
- [x] 8.2 Add `--dev` flag for development mode ✅
- [x] 8.3 Parse Git URL and version tag ✅ (git+https://...#tag)
- [x] 8.4 Clone repository to temp directory ✅
- [x] 8.5 Checkout specific version (if tag provided) ✅
- [x] 8.6 Read manifest.json and detect schema version ✅
- [x] 8.7 Auto-detect installed base model ✅
- [x] 8.8 For v2.0: use first model in base_models ✅
- [x] 8.9 Support file:// URLs for local install ✅
- [x] 8.10 Verify signature (if present) ✅
- [x] 8.11 Check base model compatibility ✅
- [x] 8.12 Resolve dependencies (requires field) ✅
- [x] 8.13 Install dependencies recursively ✅
- [x] 8.14 Copy expert to install directory ✅
- [x] 8.15 Update local registry (~/.expert/expert-registry.json) ✅

**Commits**: 576c1d9, 653f15c, latest | **Status:** 100% COMPLETE with signature verification and dependency resolution

## 9. Management Commands ✅ COMPLETE

- [x] 9.1 Implement `expert-cli list` (installed experts) ✅
- [x] 9.2 Show schema version in list output ✅
- [x] 9.3 Implement verbose mode with full details ✅
- [x] 9.4 Display base model(s) info based on schema version ✅
- [x] 9.5 Implement `expert-cli uninstall <name>` ✅
- [x] 9.6 Implement `expert-cli update <name>` (git pull) ✅
- [x] 9.7 Add local registry management ✅

**Commits**: 653f15c, latest | **Status:** ALL commands complete!

## 10. Multi-Model Examples & Migration ✅ COMPLETE

- [x] 10.1 Create example v2.0 manifest with 2 models ✅
- [x] 10.2 Document migration from v1.0 to v2.0 ✅
- [x] 10.3 Add CLI examples for multi-model workflow ✅
- [x] 10.4 Test packaging multiple variants ✅
- [x] 10.5 Test installation with auto-detection ✅
- [x] 10.6 Migrated all 4 experts to v2.0 ✅ (exceeded expectations)

**Commits**: 3053576, 85e53fb | **Status:** Complete with bonus migrations

## 11. Testing & Documentation ✅ COMPLETE

- [x] 11.1 Write unit tests for manifest parsing (both versions) ✅ (84 tests)
- [x] 11.2 Write integration tests for v1.0 workflow ✅
- [x] 11.3 Write integration tests for v2.0 workflow ✅
- [x] 11.4 Test schema version detection edge cases ✅
- [x] 11.5 Test on Linux (Ubuntu) ✅
- [x] 11.6 Test on Windows (via WSL) ✅
- [x] 11.7 Update CLI.md with v2.0 command examples ✅
- [x] 11.8 Update CLI.md with --model flag documentation ✅
- [x] 11.9 Create usage examples for multi-model experts ✅
- [x] 11.10 Add error handling and helpful messages ✅
- [x] 11.11 Document schema version detection behavior ✅

**Commits**: 765fcbf, 0fcd1cc, b40837f | **Status:** 115+ tests, comprehensive docs


