# Optimal Binary Expert Format - Tasks

**Status**: Not Started  
**Estimated**: 12-14 weeks  
**Priority**: P4

---

## Phase 1: Format Specification (2 weeks)

### 1.1 Binary Format Design
- [ ] 1.1.1 Define header layout (MAGIC, version, TOC count, SHA256)
- [ ] 1.1.2 Design TOC entry structure (section_id, type, codec, offset, length)
- [ ] 1.1.3 Specify section types enum (params, router, grammar, etc)
- [ ] 1.1.4 Define alignment requirements (64/256 bytes)
- [ ] 1.1.5 Document endianness and platform compatibility

### 1.2 Metadata Schema
- [ ] 1.2.1 Design minimal JSON manifest (base_model, tokenizer, adapter)
- [ ] 1.2.2 Add compatibility fields (weights_hash, vocab_hash, shapes)
- [ ] 1.2.3 Define runtime hints (kv_cache, attn_impl, max_experts)
- [ ] 1.2.4 Create JSON schema validator

### 1.3 Rust Type Definitions
- [ ] 1.3.1 Implement `Header` struct with `#[repr(C)]`
- [ ] 1.3.2 Implement `TocEntry` struct
- [ ] 1.3.3 Implement `SectionType` enum
- [ ] 1.3.4 Implement adapter-specific structs (LoRA, DoRA, IA³)
- [ ] 1.3.5 Add serialization traits (bincode/serde)

### 1.4 Documentation
- [ ] 1.4.1 Write `BINARY_FORMAT_SPEC.md`
- [ ] 1.4.2 Create format diagram (hex editor view)
- [ ] 1.4.3 Document versioning strategy
- [ ] 1.4.4 Add examples and test fixtures

---

## Phase 2: Binary Writer (2 weeks)

### 2.1 Core Writer Implementation
- [ ] 2.1.1 Create `ExpertWriter` struct
- [ ] 2.1.2 Implement header writing with placeholder SHA256
- [ ] 2.1.3 Implement TOC building and writing
- [ ] 2.1.4 Add section alignment logic
- [ ] 2.1.5 Implement final SHA256 calculation and update

### 2.2 Compression Pipeline
- [ ] 2.2.1 Integrate zstd compression (levels 6-10)
- [ ] 2.2.2 Add uncompressed length tracking
- [ ] 2.2.3 Implement per-section SHA256 calculation
- [ ] 2.2.4 Add compression benchmarks
- [ ] 2.2.5 Optimize for memory usage (<500MB peak)

### 2.3 LoRA/DoRA Serialization
- [ ] 2.3.1 Convert safetensors to binary format
- [ ] 2.3.2 Serialize per-layer, per-module structure
- [ ] 2.3.3 Handle DoRA magnitude vectors
- [ ] 2.3.4 Add dtype conversion (f16/bf16/f32)
- [ ] 2.3.5 Validate shapes against base model

### 2.4 Additional Sections
- [ ] 2.4.1 Serialize router keywords/embeddings
- [ ] 2.4.2 Serialize grammar files (GBNF)
- [ ] 2.4.3 Serialize soft prompts
- [ ] 2.4.4 Add license and docs sections
- [ ] 2.4.5 Implement metrics serialization

### 2.5 CLI Integration
- [ ] 2.5.1 Add `--format binary` flag to `expert-cli package`
- [ ] 2.5.2 Add `--compression-level` option
- [ ] 2.5.3 Show progress bar for large experts
- [ ] 2.5.4 Report compression ratio and size reduction

---

## Phase 3: Binary Reader (2 weeks)

### 3.1 Memory-Mapped Reader
- [ ] 3.1.1 Implement `ExpertReader` with `memmap2`
- [ ] 3.1.2 Parse and validate header
- [ ] 3.1.3 Build TOC index for O(1) lookup
- [ ] 3.1.4 Implement section decompression on-demand
- [ ] 3.1.5 Add error handling for corrupted files

### 3.2 Validation
- [ ] 3.2.1 Verify MAGIC bytes and version
- [ ] 3.2.2 Validate global SHA256
- [ ] 3.2.3 Validate per-section SHA256
- [ ] 3.2.4 Check base_model and tokenizer hashes
- [ ] 3.2.5 Verify shapes compatibility

### 3.3 LoRA/DoRA Deserialization
- [ ] 3.3.1 Parse layer headers
- [ ] 3.3.2 Deserialize module deltas (A, B matrices)
- [ ] 3.3.3 Handle DoRA magnitude
- [ ] 3.3.4 Convert to runtime format (PyTorch/Candle tensors)
- [ ] 3.3.5 Apply dtype conversions

### 3.4 Streaming Support
- [ ] 3.4.1 Implement lazy section loading
- [ ] 3.4.2 Add layer-by-layer streaming
- [ ] 3.4.3 Implement LRU cache for frequently accessed layers
- [ ] 3.4.4 Measure bandwidth (SSD→RAM→GPU)
- [ ] 3.4.5 Optimize for <100ms first-layer latency

### 3.5 CLI Integration
- [ ] 3.5.1 Update `expert-cli validate` to support binary format
- [ ] 3.5.2 Add `--dump-metadata` to show JSON manifest
- [ ] 3.5.3 Add `--list-sections` to show TOC
- [ ] 3.5.4 Add `--verify-integrity` for full SHA256 check

---

## Phase 4: OSD Support (4 weeks)

### 4.1 Theory and Research
- [ ] 4.1.1 Study OSD paper (Alipour & Amiri 2025)
- [ ] 4.1.2 Implement SVD decomposition pipeline
- [ ] 4.1.3 Design importance scoring (singular values)
- [ ] 4.1.4 Determine sparsity threshold heuristics
- [ ] 4.1.5 Write `OSD_GUIDE.md` documentation

### 4.2 SVD Decomposition
- [ ] 4.2.1 Extract deltas from trained adapters
- [ ] 4.2.2 Compute SVD per matrix (scipy/numpy)
- [ ] 4.2.3 Select top-k singular values/vectors
- [ ] 4.2.4 Measure energy retention (cumulative %)
- [ ] 4.2.5 Create SVD factorization (U, Σ, Vᵀ)

### 4.3 Sparsity Mask
- [ ] 4.3.1 Rank singular vectors by importance
- [ ] 4.3.2 Apply threshold to create sparse mask
- [ ] 4.3.3 Convert to CSR/CSC format
- [ ] 4.3.4 Optimize storage (indices compression)
- [ ] 4.3.5 Validate reconstruction error

### 4.4 OSD Serialization
- [ ] 4.4.1 Implement `OSDModule` struct
- [ ] 4.4.2 Serialize U, Σ, Vᵀ matrices
- [ ] 4.4.3 Serialize sparse indices and values
- [ ] 4.4.4 Add blend weight (sparse_weight)
- [ ] 4.4.5 Compress with zstd

### 4.5 OSD Runtime Fusion
- [ ] 4.5.1 Implement delta reconstruction: Δ = UΣVᵀ
- [ ] 4.5.2 Apply sparse mask: Δ' = Δ ⊙ M
- [ ] 4.5.3 Blend with weight: Δ_final = (1-α)Δ + α·Δ'
- [ ] 4.5.4 Fuse with base weights: W' = W + Δ_final
- [ ] 4.5.5 Benchmark fusion latency

### 4.6 Hyperparameter Tuning
- [ ] 4.6.1 Grid search rank (8, 12, 16, 24)
- [ ] 4.6.2 Grid search sparsity (0.01, 0.05, 0.10, 0.20)
- [ ] 4.6.3 A/B test on SQL expert
- [ ] 4.6.4 A/B test on JSON expert
- [ ] 4.6.5 Create tuning guidelines per expert type

---

## Phase 5: Runtime Integration (2 weeks)

### 5.1 Hot-Swap with Streaming
- [ ] 5.1.1 Integrate binary reader into `ExpertManager`
- [ ] 5.1.2 Implement streaming load (layer-by-layer)
- [ ] 5.1.3 Add unload with cleanup
- [ ] 5.1.4 Support multi-expert (up to 10) simultaneously
- [ ] 5.1.5 Measure memory footprint

### 5.2 Backward Compatibility
- [ ] 5.2.1 Auto-detect format (binary vs tar.gz)
- [ ] 5.2.2 Implement unified `ExpertLoader` interface
- [ ] 5.2.3 Add fallback to tar.gz if binary fails
- [ ] 5.2.4 Log format used for debugging
- [ ] 5.2.5 Add deprecation warnings for tar.gz

### 5.3 Migration Tools
- [ ] 5.3.1 Create `expert-cli convert` command
- [ ] 5.3.2 Implement tar.gz → binary converter
- [ ] 5.3.3 Implement binary → tar.gz converter (for debug)
- [ ] 5.3.4 Add batch conversion for all experts
- [ ] 5.3.5 Write `MIGRATION.md` guide

### 5.4 Performance Benchmarks
- [ ] 5.4.1 Measure load time (cold cache)
- [ ] 5.4.2 Measure load time (hot cache)
- [ ] 5.4.3 Measure validation time
- [ ] 5.4.4 Measure VRAM overhead
- [ ] 5.4.5 Compare vs tar.gz baseline

### 5.5 Quality Benchmarks
- [ ] 5.5.1 A/B test OSD vs LoRA on SQL (exact-match)
- [ ] 5.5.2 A/B test on JSON (schema conformance)
- [ ] 5.5.3 A/B test on Neo4j (Cypher syntax)
- [ ] 5.5.4 A/B test on TypeScript (code quality)
- [ ] 5.5.5 Regression suite (ensure no degradation)

---

## Phase 6: Signing and Security (Optional, 1 week)

### 6.1 Ed25519 Signing
- [ ] 6.1.1 Integrate `ed25519-dalek` crate
- [ ] 6.1.2 Generate key pairs (`expert-cli keygen`)
- [ ] 6.1.3 Sign header + TOC + metadata
- [ ] 6.1.4 Store signature in header
- [ ] 6.1.5 Implement verification on load

### 6.2 Certificate Chain
- [ ] 6.2.1 Design certificate format
- [ ] 6.2.2 Serialize to SIGNING/CERT section
- [ ] 6.2.3 Implement trust chain validation
- [ ] 6.2.4 Add revocation support
- [ ] 6.2.5 Write security documentation

---

## Phase 7: Documentation and Release (1 week)

### 7.1 Technical Documentation
- [ ] 7.1.1 Finalize `BINARY_FORMAT_SPEC.md`
- [ ] 7.1.2 Write `OSD_GUIDE.md` with examples
- [ ] 7.1.3 Update `PACKAGING_GUIDE.md` for v2 format
- [ ] 7.1.4 Create `MIGRATION.md` for users
- [ ] 7.1.5 Add API documentation (rustdoc)

### 7.2 User Guides
- [ ] 7.2.1 Write "Getting Started with Binary Format"
- [ ] 7.2.2 Create troubleshooting guide
- [ ] 7.2.3 Add performance tuning tips
- [ ] 7.2.4 Document OSD hyperparameter selection
- [ ] 7.2.5 Create comparison table (LoRA/DoRA/IA³/OSD)

### 7.3 Examples
- [ ] 7.3.1 Example: Convert existing expert to binary
- [ ] 7.3.2 Example: Train with OSD adapter
- [ ] 7.3.3 Example: Custom section (extended metadata)
- [ ] 7.3.4 Example: Signed expert for marketplace
- [ ] 7.3.5 Example: Multi-expert hot-swap demo

### 7.4 Release
- [ ] 7.4.1 Tag version (e.g., v0.4.0-beta)
- [ ] 7.4.2 Update CHANGELOG.md
- [ ] 7.4.3 Create migration timeline (tar.gz deprecation)
- [ ] 7.4.4 Announce in README.md
- [ ] 7.4.5 Gather community feedback

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Load time (cold) | <200ms | Benchmark script |
| Load time (hot) | <50ms | Benchmark script |
| Validation time | <10ms | `expert-cli validate --time` |
| Package size (OSD) | 30-50 MB | File size on disk |
| Accuracy (OSD vs LoRA) | +5-10% | A/B test on SQL/JSON |
| VRAM overhead | <50 MB/expert | CUDA profiler |
| Backward compat | 100% | All tests pass with tar.gz |

---

## Risk Mitigation

| Risk | Mitigation | Owner |
|------|-----------|-------|
| OSD too complex | Implement LoRA/DoRA first, OSD optional | Phase 4 |
| Performance regression | Aggressive profiling, fallback to tar.gz | Phase 5 |
| Breaking changes | Keep tar.gz support, migration tool | Phase 5 |
| Security vulnerabilities | Optional signing, audit before release | Phase 6 |

---

## Dependencies

**Rust Crates**:
- `zstd = "0.13"` - Compression
- `memmap2 = "0.9"` - Memory mapping
- `sha2 = "0.10"` - Hashing
- `ed25519-dalek = "2.1"` - Signing (optional)
- `bincode = "1.3"` - Binary serialization
- `serde = { version = "1.0", features = ["derive"] }`

**Python Libraries**:
- `scipy >= 1.11.0` - SVD decomposition
- `numpy >= 1.24.0` - Matrix operations
- `safetensors >= 0.4.0` - Tensor I/O

---

**Total Tasks**: 126

**Completed**: 0

**In Progress**: 0

**Remaining**: 126

**Estimated Completion**: 12-14 weeks from start

