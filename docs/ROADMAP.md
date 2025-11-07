# Roadmap

> Phased implementation plan for the Expert System (P0-P6)

## Overview

The Expert System will be implemented in phases, each building on the previous one. Early phases focus on core functionality, while later phases add advanced features and optimizations.

**Timeline estimate**: P0-P5 over 6-12 months (depending on team size)

---

## P0: Minimal Runtime

**Goal**: Prove the concept with a working base model + single expert inference.

**Status**: ✅ **75% Complete** (2025-11-05)  
**Duration**: 4-6 weeks  
**Language**: Rust (core inference engine)

### Deliverables

#### 1. Base Model Loading
- [x] ✅ Load Qwen3-0.6B (Candle-based, `QwenEngine`)
- [x] ✅ Support INT4/INT8 quantization (Python training, Rust structure ready)
- [x] ✅ Basic inference (generate text via `QwenEngine.generate()`)
- [ ] ⏳ RoPE scaling setup (prepare for long context)

#### 2. LoRA Adapter Support
- [x] ✅ LoRA adapter structure (`LoraAdapter`, `AdapterType`)
- [x] ✅ Load LoRA weights from SafeTensors (`from_safetensors`)
- [x] ✅ Apply to target modules (`apply_lora` method)
- [ ] ⏳ Verify inference with adapter works correctly (integration pending)

#### 3. Paged KV Cache
- [x] ✅ Basic paged attention structure (`PagedKVCache`, `PagedKVCacheConfig`)
- [x] ✅ Block allocation and management (page tables, LRU eviction)
- [ ] ⏳ Full integration with Qwen3Model (structure exists, needs integration)
- [ ] ⏳ Support context up to 32k tokens (initial target)

#### 4. API
- [x] ✅ CLI chat interface (`expert-cli chat`)
- [x] ✅ Generation config (`GenerationConfig`, sampling functions)
- [ ] ⏳ Python API bindings (`attach_experts()`, `generate()`, `release()`)
- [ ] ⏳ Session management
- [x] ✅ Basic error handling

#### 5. CUDA Support
- [x] ✅ CUDA detection and device selection (`QwenEngine.from_local`)
- [x] ✅ GPU memory management (Candle Device abstraction)
- [x] ✅ Tested on NVIDIA GPUs (RTX 4090 verified)

### Success Criteria
- [ ] Load base model in <5 seconds
- [ ] Attach LoRA expert in <100ms
- [ ] Generate 100 tokens in <10 seconds (RTX 4090)
- [ ] VRAM usage <2GB total

### Non-Goals
- Multiple experts (P1)
- Router (P1)
- .expert format (P2)
- Long context >32k (P3)

---

## P1: Router & Multi-Expert

**Goal**: Add intelligent expert selection and support 2-10 experts per inference.

**Status**: ✅ **80% Complete** (2025-11-05)  
**Duration**: 6-8 weeks  
**Language**: Rust (inference + routing), Python (embeddings prototyping)

### Deliverables

#### 1. Expert Index
- [x] ✅ Expert registry structure (`ExpertManager`, `ExpertState`)
- [x] ✅ Expert metadata storage (via `Manifest` loading)
- [ ] ⏳ Embedding-based search index (FAISS or Vectorizer MCP - structure ready)

#### 2. Heuristic Router
- [x] ✅ Keyword matching (`KeywordRouter`)
- [x] ✅ Technology detection (keyword-based)
- [x] ✅ Format detection (keyword-based patterns)
- [ ] ⏳ Language detection (langdetect or fastText - not implemented)

#### 3. Embedding-Based Selection
- [x] ✅ Embedding router structure (`EmbeddingRouter`)
- [x] ✅ Simplified TF-IDF embedding (production: SentenceTransformers)
- [x] ✅ Cosine similarity scoring
- [x] ✅ Combined scoring (semantic + keyword + priority)
- [ ] ⏳ Full SentenceTransformers integration (currently TF-IDF)

#### 4. Mini-Policy (Optional)
- [ ] ⏳ Use base model to classify ambiguous prompts
- [ ] ⏳ Rank candidate experts
- [ ] ⏳ Fast inference mode (<100ms)

#### 5. Multi-Expert Composition
- [x] ✅ Expert hot-swap (`ExpertManager.load_expert`, `unload_expert`)
- [x] ✅ LRU eviction (`unload_lru`)
- [x] ✅ Priority-based pre-loading (`preload_priority`)
- [x] ✅ VRAM budget checking (max_loaded limit)
- [ ] ⏳ Incompatibility filtering (structure ready, needs implementation)

#### 6. Parameter Tuning
- [x] ✅ Temperature selection (via `GenerationConfig`)
- [x] ✅ Top-p, top-k, repetition penalty (sampling functions)
- [ ] ⏳ Automatic parameter selection by task type

### Success Criteria
- [ ] Router selects appropriate experts in <20ms (no mini-policy)
- [ ] Support up to 10 experts per inference
- [ ] VRAM usage <8GB with 10 experts
- [ ] Expert selection accuracy >80% on test cases

---

## P2: Expert Format & Marketplace

**Goal**: Standardize expert packaging and enable community distribution.

**Status**: ✅ **90% Complete** (2025-11-05)  
**Duration**: 6-8 weeks  
**Language**: Rust (CLI, packaging, signatures)

### Deliverables

#### 1. .expert Package Format
- [x] ✅ Manifest.json schema (v1.0 and v2.0 multi-model support)
- [x] ✅ SafeTensors integration (adapter weights)
- [x] ✅ Optional soft-prompt support (`.pt` files)
- [x] ✅ Compression (tar.gz with zstd planned)
- [x] ✅ Essential files inclusion (adapter_config.json, tokenizer files, grammar.gbnf)
- [x] ✅ Package validation (extract + verify structure)

#### 2. Signing & Verification
- [x] ✅ Ed25519 structure (`expert-cli sign`)
- [x] ✅ Package signing (sign command implemented)
- [x] ✅ Integrity checks (SHA-256 in package validation)
- [ ] ⏳ Signature verification on installation (structure ready)

#### 3. CLI Tooling
- [x] ✅ `expert-cli install <expert>` (install command)
- [x] ✅ `expert-cli list` (list command)
- [x] ✅ `expert-cli uninstall <expert>` (uninstall command)
- [x] ✅ `expert-cli package` (package command with v1.0/v2.0 support)
- [x] ✅ `expert-cli validate` (validate command with .expert extraction)
- [x] ✅ `expert-cli sign` (signing command)
- [x] ✅ `expert-cli route` (routing command)
- [x] ✅ `expert-cli chat` (interactive chat)
- [x] ✅ `expert-cli train` (training command)
- [x] ✅ `expert-cli dataset` (dataset commands)
- [ ] ⏳ `expert-cli search <query>` (local registry search)

#### 4. Local Registry
- [x] ✅ Install/uninstall experts (basic structure)
- [x] ✅ Version management (semver parsing in manifest)
- [ ] ⏳ Dependency resolution (structure ready, needs implementation)
- [x] ✅ Compatibility checking (base model hash in manifest)

#### 5. Marketplace (Local)
- [ ] ⏳ Local catalog (JSON - structure ready)
- [ ] ⏳ Update mechanism
- [ ] ⏳ Publisher trust model

### Success Criteria
- [ ] Package and install expert in <30 seconds
- [ ] Verify signature in <1 second
- [ ] Support 50+ experts in local registry
- [ ] Compatibility conflicts detected 100% of the time

### Non-Goals
- Online marketplace (future)
- Automatic updates (future)

---

## P3: Long Context

**Goal**: Extend context window to 128k-256k tokens.

**Duration**: 4-6 weeks  
**Language**: Rust (paged attention), Python (finetuning)

### Deliverables

#### 1. RoPE Scaling Implementation
- [ ] NTK-aware scaling
- [ ] YaRN implementation
- [ ] Configurable scaling factors
- [ ] Tested on Qwen3-0.6B

#### 2. Base Model Continual Finetuning
- [ ] Dataset: long-context examples (64k-128k)
- [ ] Short finetune (1 epoch, 10k examples)
- [ ] Evaluation suite (LongBench or similar)
- [ ] Target: 128k stable, 256k experimental

#### 3. Paged Attention Scaling
- [ ] Extend paging to 128k contexts
- [ ] Memory optimization (INT8 KV cache option)
- [ ] Block eviction for >256k contexts

#### 4. Position Curriculum for Experts
- [ ] Train experts on progressively longer contexts
- [ ] Curriculum: 16k → 32k → 64k → 128k
- [ ] Validation on long-context tasks

#### 5. Chunk Routing
- [ ] Pre-filter irrelevant context chunks
- [ ] Embedding-based relevance scoring
- [ ] Keep top-N most relevant chunks

### Success Criteria
- [ ] Stable inference at 128k context
- [ ] Experimental support for 256k
- [ ] <5% quality degradation vs 32k on long tasks
- [ ] VRAM usage <16GB for 128k context

---

## P4: Training Tooling & Binary Format

**Goal**: Enable easy expert training with synthetic data generation + optimize expert storage/loading.

**Status**: ✅ **70% Complete** (Training), ⏳ **0% Complete** (Binary Format) (2025-11-05)  
**Duration**: 12-14 weeks  
**Language**: Python (training pipelines, data generation), Rust (binary format)

### Deliverables

#### 1. OSD Binary Expert Format (🔬 Beta - High Priority)
- [x] ✅ Design binary layout (proposal complete in OpenSpec)
- [ ] ⏳ Implement OSD adapter (SVD + selective sparsity)
- [ ] ⏳ Memory-mapped reader with streaming
- [ ] ⏳ Multi-adapter support (LoRA/DoRA/IA³/OSD)
- [ ] ⏳ SHA256 integrity + Ed25519 signing
- [ ] ⏳ Migration tools (tar.gz ↔ binary)
- [ ] ⏳ Benchmarks: 2-5x faster load, 30-40% smaller
- [ ] ⏳ A/B test: OSD vs LoRA accuracy (+5-10% target)
- **See**: `openspec/changes/implement-osd-binary-expert-format/` (Proposal ready)

#### 2. Synthetic Data Generation CLI
- [x] ✅ Dataset commands (`expert-cli dataset`)
- [x] ✅ Dataset stats and validation
- [ ] ⏳ DeepSeek Chat integration
- [ ] ⏳ Claude API integration
- [ ] ⏳ GPT-4o API integration
- [ ] ⏳ Batch generation (async, parallel)

#### 3. Quality Control Pipeline
- [x] ✅ Schema validation (preprocess.py with sqlglot)
- [x] ✅ Deduplication (exact by question field)
- [x] ✅ SQL validation (MySQL→PostgreSQL conversion)
- [ ] ⏳ Diversity metrics (embedding-based)
- [ ] ⏳ Difficulty scoring
- [ ] ⏳ Stratification (easy/medium/hard)

#### 4. Training Pipelines
- [x] ✅ `expert-cli train` (supervised fine-tuning)
- [x] ✅ Support for LoRA, DoRA, IA³ adapters
- [x] ✅ Unsloth integration (2x faster, 70% less VRAM)
- [x] ✅ Checkpointing system (save_strategy, eval_strategy)
- [x] ✅ Windows compatibility fixes
- [x] ✅ Training parameter optimization (LR, dropout, warmup_ratio)
- [ ] ⏳ `expert-train dpo` (direct preference optimization)
- [ ] ⏳ `expert-train distill` (knowledge distillation)
- [ ] ⏳ Soft-prompts training (structure ready)

#### 5. Validation Framework
- [x] ✅ Test structure (test_comparison.py, test_expert.py)
- [x] ✅ Quality regression tests (expert-sql, expert-json)
- [ ] ⏳ Task-specific metrics (EM, F1, BLEU, accuracy)
- [ ] ⏳ Multi-expert compatibility testing
- [ ] ⏳ Automated eval on held-out data

#### 6. Export & Packaging
- [x] ✅ Export trained adapter to SafeTensors (automatic via training)
- [x] ✅ Manifest.json generation (manual, structure standardized)
- [x] ✅ Package as .expert (`expert-cli package`)
- [x] ✅ Sign with Ed25519 (`expert-cli sign`)

### Success Criteria
- [ ] Binary format: <200ms load (cold), <50ms (hot), 2.5x faster than tar.gz
- [ ] OSD accuracy: +5-10% vs LoRA at same storage
- [ ] Package size: 30-50 MB (vs 50-80 MB tar.gz), 40% reduction
- [ ] Generate 10k synthetic examples in <1 hour
- [ ] Train expert end-to-end in <2 hours (RTX 4090)
- [ ] Expert quality: >90% on domain-specific tasks
- [ ] Package and sign in <1 minute

---

## P5: Marketplace & Scale

**Goal**: Distribute experts via marketplace and support production workloads.

**Duration**: 8-12 weeks  
**Language**: Rust (marketplace backend, multi-GPU), Python (monitoring dashboards)

### Deliverables

#### 1. Online Marketplace
- [ ] Central registry (API + web UI)
- [ ] Upload/download experts
- [ ] Search and discovery
- [ ] Ratings and reviews
- [ ] Publisher verification

#### 2. ROCm Support (AMD GPUs)
- [ ] Port CUDA kernels to ROCm
- [ ] Test on AMD GPUs (RX 7000 series)
- [ ] Performance parity with CUDA

#### 3. Multi-GPU Support
- [ ] Model parallelism (split base model across GPUs)
- [ ] Expert parallelism (different experts on different GPUs)
- [ ] Pipeline parallelism (inference stages)

#### 4. Monitoring & Telemetry
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Job queue monitoring
- [ ] Expert performance tracking
- [ ] Router accuracy analytics

#### 5. Production Hardening
- [ ] Error handling and recovery
- [ ] Rate limiting
- [ ] Resource quotas (VRAM, CPU, jobs)
- [ ] Graceful degradation
- [ ] Health checks

#### 6. Orchestrator Improvements
- [ ] Priority queues
- [ ] Preemption
- [ ] Expert hot-cache optimization (LRU)
- [ ] VRAM budgeting
- [ ] Job batching

### Success Criteria
- [ ] Marketplace has >100 experts
- [ ] ROCm performance within 10% of CUDA
- [ ] Multi-GPU scales linearly (2x GPU = 2x throughput)
- [ ] Production uptime >99.9%
- [ ] Router accuracy >85% on diverse workloads

---

## P6: Advanced Features (Optional)

**Goal**: Cutting-edge optimizations and research features.

**Duration**: Ongoing  
**Language**: Mixed (Rust for runtime features, Python for research)

### Speculative Decoding
- [ ] Base model as draft generator
- [ ] Base + experts as verifier
- [ ] Parallel verification
- [ ] Target: 1.5-2x speedup

### Grammar-Guided Generation
- [x] ✅ Grammar support (grammar.gbnf files in experts)
- [x] ✅ SQL syntax constraints (grammar validation in manifest)
- [ ] ⏳ JSON schema enforcement
- [ ] ⏳ Custom grammar DSL
- [ ] ⏳ Constrained beam search

### Adaptive VRAM Budgeting
- [ ] Dynamic expert count adjustment (K adaptive)
- [ ] Rank reduction for tight VRAM (r=16 → r=8)
- [ ] Automatic IA³ fallback

### Multi-Expert Training
- [ ] Train experts to work together
- [ ] Interaction learning (expert A + expert B)
- [ ] Conflict detection and resolution

### Expert Fusion
- [ ] Automatically merge frequently co-loaded experts
- [ ] Offline fusion (create new merged expert)
- [ ] Reduce loading overhead

### RNN/LSTM Integration
- [ ] Sequence modeling for router decisions
- [ ] Expert selection as sequence prediction
- [ ] Temporal patterns in job queues

### Federated Learning
- [ ] Users contribute training data
- [ ] Privacy-preserving aggregation
- [ ] Improve shared experts collaboratively

### Quantized Experts
- [ ] INT8 LoRA weights (2x compression)
- [ ] INT4 LoRA weights (4x compression, experimental)
- [ ] Dynamic quantization

### OSD Enhancements (Post-Binary Format)
- [ ] Auto-tuning rank + sparsity per expert type
- [ ] Adaptive importance thresholds
- [ ] Layer-wise sparsity budgets
- [ ] Quantized OSD (INT8 sparse masks)

---

## Timeline Summary

| Phase | Duration | Cumulative | Language | Key Features |
|-------|----------|------------|----------|--------------|
| P0 | 4-6 weeks | 1.5 months | Rust | Base + single expert |
| P1 | 6-8 weeks | 3.5 months | Rust + Python | Router + multi-expert |
| P2 | 6-8 weeks | 5.5 months | Rust | .expert format + marketplace (local) |
| P3 | 4-6 weeks | 7 months | Rust + Python | Long context (128k-256k) |
| P4 | 12-14 weeks | 11 months | Python + Rust | Training tools + OSD binary format |
| P5 | 8-12 weeks | 12 months | Rust + Python | Marketplace (online) + production |
| P6 | Ongoing | - | Mixed | Advanced features |

**Total**: 14 months for P0-P5 (minimum viable product with OSD optimization)

### Language Distribution by Component

```
┌─────────────────────────────────────────────────────────┐
│                   Rust Components                       │
├─────────────────────────────────────────────────────────┤
│  • Core inference engine (P0-P1)                        │
│  • Expert hot-swap loader (P0-P1)                       │
│  • Paged KV cache (P0, P3)                              │
│  • Router/reasoning (P1)                                │
│  • Marketplace CLI (P2)                                 │
│  • Signature verification (P2)                          │
│  • Multi-GPU orchestration (P5)                         │
│  • gRPC/HTTP API + bindings (P0-P5)                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Python Components                      │
├─────────────────────────────────────────────────────────┤
│  • Expert training pipelines (P4)                       │
│  • Synthetic data generation (P4)                       │
│  • LoRA/IA³/DoRA/soft-prompt training (P4)              │
│  • Validation & evaluation (P4)                         │
│  • Dataset curation (P4)                                │
│  • Base model finetuning (P3)                           │
│  • Monitoring dashboards (P5)                           │
│  • Research prototyping (P6)                            │
└─────────────────────────────────────────────────────────┘
```

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| LoRA composition quality | Extensive testing, fall back to single expert |
| Long context stability | Incremental scaling (32k → 64k → 128k) |
| VRAM constraints | Aggressive quantization, IA³ fallback |
| Router accuracy | Continuous learning from telemetry |
| Expert conflicts | Compatibility matrix, automated testing |

### Operational Risks

| Risk | Mitigation |
|------|------------|
| Marketplace abuse | Signature verification, reputation system |
| Expert quality | Community reviews, automated validation |
| Adoption | Clear documentation, example experts |
| Scaling | Horizontal scaling, multi-GPU |

---

## Dependencies

### External Libraries
- **PyTorch** >=2.0 (core)
- **Transformers** >=4.35 (model loading)
- **PEFT** >=0.7 (LoRA/adapter support)
- **SafeTensors** (weight storage)
- **FAISS** or **Vectorizer MCP** (embedding search)
- **SentenceTransformers** (embeddings)

### Optional
- **vLLM** (paged attention reference)
- **llama.cpp** (quantization, KV cache)
- **BitsAndBytes** (quantization)
- **Flash Attention** (speedup)

---

## Community Contributions

We welcome contributions in the following areas:

**P0-P2 (Core)**:
- LoRA implementation optimization
- Additional adapter types (LoHa, AdaLoRA)
- CPU inference optimization

**P3 (Long Context)**:
- Long-context evaluation benchmarks
- Position embedding research
- Memory-efficient attention variants

**P4 (Training)**:
- Synthetic data templates
- Domain-specific expert recipes
- Training optimization tricks

**P5-P6 (Advanced)**:
- ROCm kernels
- Alternative quantization methods
- Novel expert composition strategies

---

## Success Metrics (P0-P5 Complete)

### Performance
- [ ] Inference latency: <20s for 1024 tokens (RTX 4090)
- [ ] Router latency: <20ms
- [ ] Expert load (hot): <10ms
- [ ] VRAM usage: <8GB for typical workload

### Quality
- [ ] Router accuracy: >85% on test suite
- [ ] Expert quality: >90% on domain tasks
- [ ] Long context: <5% degradation at 128k

### Ecosystem
- [ ] Marketplace: >100 experts
- [ ] Active users: >1000
- [ ] Community contributions: >10 expert recipes

### Documentation
- [ ] Complete architecture docs
- [ ] Training guides
- [ ] API reference
- [ ] 10+ example experts

---

## Post-P5: Continuous Improvement

- Expand marketplace (1000+ experts)
- Performance optimizations (quantized experts, fusion)
- Research collaborations (novel adapter types, routing strategies)
- Enterprise features (SSO, audit logs, compliance)
- Additional base models (Llama, Mistral, etc.)

---

## Implementation Status Summary

**Last Updated**: 2025-11-05

| Phase | Status | Completion | Key Achievements |
|-------|--------|------------|------------------|
| **P0** | ✅ Mostly Complete | 75% | QwenEngine, LoRA structure, Paged KV-cache structure, CUDA support |
| **P1** | ✅ Mostly Complete | 80% | KeywordRouter, EmbeddingRouter, ExpertManager, hot-swap |
| **P2** | ✅ Mostly Complete | 90% | Package format, CLI commands, validation, signing structure |
| **P3** | ⏳ Not Started | 0% | Long context support planned |
| **P4** | ✅ Training Complete, Binary Format Pending | 70% | Training pipeline, Unsloth, dataset tools, quality control |
| **P5** | ⏳ Not Started | 0% | Marketplace features planned |
| **P6** | ⏳ Partial | 10% | Grammar support (gbnf files) |

### Recent Major Completions (2025-11-05)

1. **Training Infrastructure**: Complete Python training pipeline with Unsloth support (2x faster, 70% less VRAM)
2. **Expert Packaging**: Full `.expert` package format with validation, essential files, grammar support
3. **Routing System**: Keyword and embedding routers with confidence scoring
4. **Multi-Expert Runtime**: ExpertManager with hot-swap, LRU eviction, priority pre-loading
5. **CLI Tooling**: 13 commands implemented (train, package, validate, install, list, uninstall, sign, route, chat, dataset, etc.)

### Current Focus

- **OSD Binary Format**: Proposal complete, implementation pending (P4)
- **Paged KV-Cache Integration**: Structure exists, needs deep integration with Qwen3Model
- **LoRA Runtime Integration**: Structure ready, needs end-to-end testing

---

## Get Involved

**Current phase**: P0-P2 (Mostly complete), P4 Training (Complete), Binary Format (Pending)

**How to contribute**:
1. Check [STATUS.md](STATUS.md) for current progress
2. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the system
3. Pick a task from this roadmap
4. Submit PRs or discuss in issues

**Contact**: [To be determined]

