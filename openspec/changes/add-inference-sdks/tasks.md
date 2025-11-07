# Implementation Tasks: Expert Inference SDKs

**Change ID**: `add-inference-sdks`  
**Status**: Draft

---

## 1. Python SDK - Core (`hivellm-expert`)

### 1.1 Project Setup
- [ ] Create `expert/sdk/python/` directory
- [ ] Initialize Python package (`pyproject.toml`, `setup.py`)
- [ ] Configure dependencies (`transformers`, `peft`, `torch`, `jsonschema`)
- [ ] Setup pytest and testing infrastructure
- [ ] Configure mypy for type checking
- [ ] Create `.gitignore` and `README.md`

### 1.2 Package Loading
- [ ] Implement `ExpertLoader` class
- [ ] Add `.expert` file extraction (tar.gz)
- [ ] Add manifest JSON parsing and validation
- [ ] Implement SHA256 checksum verification
- [ ] Add package caching mechanism (optional)
- [ ] Handle backward compatibility for old package formats

### 1.3 Core Inference API
- [ ] Implement `Expert` class with `load()` class method
- [ ] Implement `generate()` method for single prompt inference
- [ ] Auto-load base model, adapter, tokenizer from extracted package
- [ ] Apply ChatML template automatically
- [ ] Use manifest decoding params (temperature, top_p, top_k)
- [ ] Device auto-detection (CUDA, CPU, MPS)

### 1.4 Multiple Experts Support
- [ ] Implement `ExpertManager` class
- [ ] Add `load_expert(name, path)` method
- [ ] Detect when experts share same base model
- [ ] Implement base model reuse (load once, share across adapters)
- [ ] Add `unload_expert(name)` for memory management
- [ ] Implement `list_loaded_experts()` method
- [ ] Add adapter hot-swapping (without base model reload)

### 1.5 Advanced Features
- [ ] Implement `generate_stream()` for streaming inference
- [ ] Implement `generate_batch()` for multiple prompts
- [ ] Add grammar validation support (llama-cpp-python integration)
- [ ] Implement context length tracking
- [ ] Add memory management helpers
- [ ] Add LRU cache for expert eviction (optional)

### 1.6 Error Handling
- [ ] Define custom exception hierarchy (`ExpertError`, `LoadError`, `InferenceError`)
- [ ] Add validation for invalid `.expert` packages
- [ ] Handle missing dependencies gracefully
- [ ] Add helpful error messages with troubleshooting hints

### 1.7 Testing
- [ ] Unit tests for package loading
- [ ] Unit tests for manifest parsing
- [ ] Integration tests with real `.expert` files
- [ ] Test error conditions (corrupted packages, missing files)
- [ ] Performance benchmarks vs manual approach

### 1.8 Documentation
- [ ] Write API reference (docstrings for all public methods)
- [ ] Create usage guide with examples
- [ ] Write migration guide from manual approach
- [ ] Add Jupyter notebook examples
- [ ] Document configuration options

---

## 2. Rust SDK - Core (`hivellm-expert`)

### 2.1 Project Setup
- [ ] Create `expert/sdk/rust/` directory
- [ ] Initialize Cargo project (`Cargo.toml`)
- [ ] Configure dependencies (`candle-core`, `tokenizers`, `serde`, `tar`, `flate2`)
- [ ] Setup testing infrastructure
- [ ] Add clippy and rustfmt configuration
- [ ] Create `README.md` and examples

### 2.2 Package Loading
- [ ] Implement `ExpertLoader` struct
- [ ] Add `.expert` file extraction (tar + gzip)
- [ ] Reuse `manifest.rs` from CLI for parsing
- [ ] Implement SHA256 checksum verification
- [ ] Add package caching (optional)
- [ ] Handle backward compatibility

### 2.3 Core Inference API
- [ ] Implement `Expert` struct with `load()` method
- [ ] Implement `generate()` method using Candle
- [ ] Reuse `QwenEngine` from CLI (`src/engines/qwen3_engine.rs`)
- [ ] Auto-apply ChatML template
- [ ] Use manifest decoding params
- [ ] Device selection (CUDA, CPU, Metal)

### 2.4 Multiple Experts Support
- [ ] Implement `ExpertManager` struct
- [ ] Add `load_expert(name, path)` method
- [ ] Detect shared base models (compare base_model paths)
- [ ] Implement base model reuse with Arc<Model>
- [ ] Add `unload_expert(name)` method
- [ ] Implement `list_loaded_experts()` method  
- [ ] Add adapter hot-swapping with PEFT
- [ ] Implement LRU eviction policy with `max_loaded_experts` config

### 2.5 Advanced Features
- [ ] Implement `generate_stream()` with async/await
- [ ] Implement `generate_batch()` for batching
- [ ] Add grammar validation (GBNF)
- [ ] Implement context window tracking
- [ ] Add CUDA graph optimization

### 2.6 Error Handling
- [ ] Define error types with `thiserror`
- [ ] Add validation for packages
- [ ] Handle missing dependencies
- [ ] Provide actionable error messages

### 2.7 Testing
- [ ] Unit tests for package loading
- [ ] Unit tests for manifest parsing
- [ ] Integration tests with real experts
- [ ] Property-based tests for edge cases
- [ ] Benchmarks vs Python SDK

### 2.8 Documentation
- [ ] Write rustdoc for all public items
- [ ] Create examples/ directory with CLI, web server
- [ ] Write usage guide in README
- [ ] Document performance optimization tips
- [ ] Add migration guide from manual Candle usage

---

## 3. Integration & Polish

### 3.1 Cross-Platform Testing
- [ ] Test Python SDK on Windows
- [ ] Test Python SDK on Linux
- [ ] Test Python SDK on macOS
- [ ] Test Rust SDK on Windows
- [ ] Test Rust SDK on Linux
- [ ] Test Rust SDK on macOS

### 3.2 Example Applications
- [ ] Create Python REST API example (FastAPI) - single expert
- [ ] Create Python multi-expert REST API (FastAPI) - route by intent
- [ ] Create Rust CLI example
- [ ] Create Jupyter notebook walkthrough
- [ ] Create Rust web server example (Axum) - with ExpertManager
- [ ] Create batch processing script
- [ ] Create multi-expert chat CLI (user selects expert)

### 3.3 Performance Validation
- [ ] Benchmark Python SDK vs manual approach
- [ ] Benchmark Rust SDK vs Python SDK
- [ ] Benchmark memory usage with multiple experts (1 vs 3 vs 5)
- [ ] Measure adapter swap latency (<100ms target)
- [ ] Profile memory usage
- [ ] Optimize hot paths
- [ ] Document performance characteristics
- [ ] Validate base model reuse (memory savings >50%)

### 3.4 Documentation Site
- [ ] Create docs site (mkdocs or mdbook)
- [ ] Add quickstart guide
- [ ] Add API reference
- [ ] Add advanced topics (grammar, streaming, batching)
- [ ] Add troubleshooting guide

---

## 4. Publishing & Release

### 4.1 Python Package
- [ ] Configure PyPI publishing
- [ ] Write CHANGELOG.md
- [ ] Create release workflow (GitHub Actions)
- [ ] Publish to PyPI (test first, then prod)
- [ ] Add to expert documentation

### 4.2 Rust Crate
- [ ] Configure crates.io publishing
- [ ] Write CHANGELOG.md
- [ ] Create release workflow
- [ ] Publish to crates.io
- [ ] Add to expert documentation

### 4.3 Announcement
- [ ] Write blog post announcing SDKs
- [ ] Update main expert README with SDK links
- [ ] Create tutorial videos (optional)
- [ ] Share in community channels

---

## 5. Maintenance & Support

### 5.1 CI/CD
- [ ] Setup GitHub Actions for Python (lint, test, build)
- [ ] Setup GitHub Actions for Rust (clippy, test, build)
- [ ] Add coverage reporting
- [ ] Add dependency security scanning

### 5.2 Compatibility
- [ ] Define semantic versioning policy
- [ ] Document breaking change process
- [ ] Maintain compatibility matrix (SDK vs manifest version)
- [ ] Test against multiple expert versions

### 5.3 Issue Management
- [ ] Create issue templates
- [ ] Setup discussion forum
- [ ] Write contributing guide
- [ ] Define support SLA

---

## Notes

**Priority**: Phase 1 (Python SDK core) should be completed first to validate API design.

**Dependencies**: 
- Reuse CLI code where possible (manifest parsing, Qwen engine)
- Coordinate with CLI team for shared components

**Testing**: 
- Use `expert-sql-qwen3-0-6b.v0.2.0.expert` as primary test package
- Create smaller test packages for unit tests

**Performance**: 
- Target <5% overhead vs manual approach
- Optimize for single-query latency first, then throughput

