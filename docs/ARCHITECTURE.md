# Architecture

> Detailed breakdown of the Expert System's six core components

## System Overview

The Expert System is built on a **hot-swappable adapter architecture** where a small base model remains fixed in VRAM while lightweight specialist adapters are dynamically loaded and composed at runtime. This document details each component and their interactions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                           │
│                  (prompt + template + params)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Multi-Agent Orchestrator │
                │   - Job queue              │
                │   - Priority scheduling    │
                │   - Telemetry collection   │
                └────────────┬───────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │      Router/Reasoning (RG) - CPU/RAM      │
        │                                            │
        │  1. Analyze prompt (heuristics, embeddings)│
        │  2. Query expert index (Vectorizer)        │
        │  3. Select top-K experts (K ≤ 10)          │
        │  4. Determine composition order            │
        │  5. Tune generation params (temp, length)  │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │         Expert Storage (SSD)               │
        │                                            │
        │  Local registry of .expert packages        │
        │  - Signed, versioned, compatibility-checked│
        │  - Lazy loading (only fetch what's needed) │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │    Dynamic Loader (SSD → RAM → VRAM)      │
        │                                            │
        │  - Decompress .expert packages             │
        │  - Map safetensors to VRAM (or pinned RAM) │
        │  - Apply adapters to base model layers     │
        │  - LRU cache management (hot experts)      │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │  Inference Runtime (RI) - GPU              │
        │                                            │
        │  Base Model: Qwen3-0.6B (INT4, fixed)      │
        │  + Attached Experts: LoRA/DoRA/IA³/Soft    │
        │  + Paged KV cache (per-job, isolated)      │
        │  + CUDA streams (parallel execution)       │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────┐
        │          Post-Processing                   │
        │  - Validate output (JSON, schema, format)  │
        │  - Collect metrics (latency, success)      │
        │  - Feedback to Router for learning         │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
                ┌────────┐
                │ Result │
                └────────┘
```

## Component 1: Base Model (MB)

### Overview
The foundation model that remains permanently loaded in GPU VRAM. All expert adapters are applied on top of this model.

### Specifications

**Model**: Qwen3-0.6B  
**Quantization**: INT4 or INT8 (GPTQ/AWQ/GGUF-compatible)  
**VRAM footprint**: ~0.3-0.6 GB (quantized), ~1.2 GB (FP16)  
**Context window**: 128k-256k tokens (via RoPE scaling)  
**Attention**: Paged (vLLM-style or llama.cpp paged KV)

### RoPE Scaling Configuration

Long context support is achieved through RoPE (Rotary Position Embedding) scaling:

- **NTK-aware scaling**: Frequency interpolation for positions beyond training window
- **YaRN (Yet another RoPE extensioN)**: Hybrid approach with attention scaling
- **Target**: 128k baseline, 200k stretched (with eval validation)

### Architecture Details

```
Qwen3-0.6B Architecture (simplified):
- Vocabulary: ~150k tokens
- Hidden size: 896
- Intermediate size: 4864
- Layers: 28
- Attention heads: 14
- KV heads: 2 (GQA - Grouped Query Attention)
- Activation: SwiGLU
```

### Quantization Strategy

1. **INT4 (preferred)**: 4-bit weights, ~0.3-0.4 GB VRAM, minimal quality loss
2. **INT8**: 8-bit weights, ~0.5-0.6 GB VRAM, better quality for complex reasoning
3. **Mixed precision**: INT4 for MLP, INT8 for attention (balanced approach)

### Loading Process

```rust
// Conceptual loading flow
let base_model = BaseModel::load(
    "qwen3-0.6b-int4.safetensors",
    QuantizationConfig {
        bits: 4,
        group_size: 128,
        rope_scaling: RopeScaling::YaRN { scale: 8.0, alpha: 1.0 },
    }
)?;

// Fixed in VRAM for session duration
base_model.to_device(Device::Cuda(0))?;
```

---

## Component 2: Experts (EXPs)

### Overview
Lightweight, task-specific adapters that modify the base model's behavior without altering its weights. Experts are **never merged** into the base model—they're applied as runtime composition layers.

### Expert Types (by preference order)

#### 1. LoRA (Low-Rank Adaptation)
**Size**: 10-80 MB (depending on rank `r`)  
**Target modules**: `q_proj`, `v_proj`, `o_proj` (attention), `gate_proj`, `up_proj`, `down_proj` (MLP)  
**Rank (r)**: 8-32 (sweet spot: 16)  
**Alpha (α)**: Usually equal to `r` (standard scaling)

```
Original weight: W ∈ ℝ^(m×n)
LoRA adds: ΔW = BA where B ∈ ℝ^(m×r), A ∈ ℝ^(r×n)
Final: W' = W + (α/r) * BA
```

**Advantages**: Proven, stable, wide tooling support  
**Disadvantages**: Larger than IA³, requires more training data

#### 2. LoRA-FA (Frozen-A LoRA)
**Variant of LoRA** where matrix A is frozen (random init, not trained).  
**Advantage**: Half the trainable parameters, faster training  
**Use case**: When data is limited but quality is acceptable

#### 3. DoRA (Weight-Decomposed LoRA)
**Decomposes** weight updates into magnitude and direction components.  
**Advantage**: Better quality than standard LoRA for same rank  
**Disadvantage**: Slightly higher compute cost

#### 4. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
**Size**: 1-5 MB (extremely lightweight)  
**Mechanism**: Learned element-wise scaling vectors  
**Target**: Key, value, and feedforward activations

```
Modified activation: a' = a ⊙ v
where v ∈ ℝ^d is learned scaling vector
```

**Advantages**: Minimal VRAM, fast loading  
**Disadvantages**: Less expressive than LoRA for complex domains

#### 5. Soft Prompts (Prompt Tuning)
**Size**: <1 MB  
**Mechanism**: Prepend learned embeddings to input  
**Length**: 64-128 tokens worth of embeddings

```
Input: [soft_prompt_embeddings] + tokenized_text
where soft_prompt ∈ ℝ^(L×hidden_size), L = 64-128
```

**Advantages**: Zero parameter overhead on model layers  
**Use case**: Style/format control (e.g., "always output valid JSON")

#### 6. Custom Vocabulary Heads (rare)
**Size**: Varies (depends on new vocab size)  
**Use case**: Domain-specific tokens not in base vocabulary  
**Example**: Chemical formulas, programming language symbols

### Expert Composition

Multiple experts are applied **additively**:

```
For LoRA experts on same module:
W' = W + Σᵢ (αᵢ/rᵢ) * BᵢAᵢ

For IA³ experts:
a' = a ⊙ v₁ ⊙ v₂ ⊙ ... ⊙ vₙ
```

**Composition order matters** for semantic tasks:
```
Example: JSON → English → Neo4j
1. json-parser.expert (understands structure)
2. english.expert (language context)
3. neo4j-schema.expert (domain knowledge)
```

---

## Component 3: Router/Reasoning (RG)

### Overview
CPU-based decision engine that analyzes incoming prompts and selects the optimal set of experts for the task. Runs in parallel with previous inference to minimize latency.

### Architecture

```
Input: Prompt + Template + User Hints
  ↓
┌─────────────────────────────────────┐
│  Feature Extraction (parallel)      │
│  - Heuristics (regex, lang-id)      │
│  - Embeddings (MiniLM, fast)        │
│  - Semantic analysis (MB short run) │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Expert Index Query                 │
│  - ANN search (Vectorizer/FAISS)    │
│  - Filter by compatibility          │
│  - Rank by semantic match           │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Scoring & Selection                │
│  - Semantic relevance (0-1)         │
│  - VRAM cost (MB)                   │
│  - Historical success rate (0-1)    │
│  - Incompatibility checks           │
│  - Top-K selection (K ≤ 10)         │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Parameter Tuning                   │
│  - Temperature (task-dependent)     │
│  - Max output tokens                │
│  - Top-p, top-k, min-p              │
│  - Repetition penalty               │
└─────────────────┬───────────────────┘
                  ↓
Output: Expert Plan (experts, order, params)
```

### Heuristic Rules (Fast Path)

```python
# Language detection
if detect_language(prompt) == "pt-BR":
    experts.add("portuguese.expert")

# Format detection
if "{" in prompt or "json" in prompt.lower():
    experts.add("json-parser.expert")

# Technology keywords
if "neo4j" in prompt.lower() or "MATCH (" in prompt:
    experts.add("neo4j-schema.expert")

# Task classification
if any(word in prompt.lower() for word in ["classify", "categorize"]):
    experts.add("document-classifier.expert")
```

### Embedding-Based Selection

```python
# Embed prompt
prompt_embedding = embed_model.encode(prompt)  # MiniLM: ~10ms

# Query expert index (ANN)
candidates = expert_index.search(
    prompt_embedding,
    k=20,  # Retrieve more than needed
    threshold=0.7  # Semantic similarity cutoff
)

# Rank by combined score
for exp in candidates:
    score = (
        0.5 * semantic_similarity(prompt_embedding, exp.embedding) +
        0.3 * exp.historical_success_rate +
        0.2 * (1 - exp.vram_cost / max_vram)
    )
```

### Policy Model (Mini-Policy)

For complex tasks, use the base model itself to rank experts:

```
Prompt to MB:
"Given the task: '{user_prompt}'
Available experts: {expert_list_with_descriptions}
Select up to 10 experts and rank them by relevance.
Output format: expert1,expert2,expert3,..."

Parse output → top-K experts
```

### Parameter Tuning Rules

| Task Type | Temperature | Max Tokens | Top-p |
|-----------|-------------|------------|-------|
| JSON parsing | 0.1-0.3 | 512-2048 | 0.9 |
| Classification | 0.2-0.4 | 128-512 | 0.95 |
| Creative writing | 0.7-0.9 | 4096+ | 0.95 |
| Code generation | 0.3-0.5 | 2048-8192 | 0.95 |
| Summarization | 0.5-0.7 | 512-2048 | 0.9 |

### Caching & Learning

**Cache similar prompts**:
```python
# Hash prompt (fuzzy, remove variable parts)
prompt_hash = fuzzy_hash(normalize(prompt))

if prompt_hash in router_cache:
    return router_cache[prompt_hash]  # Instant decision
```

**Offline learning**:
```python
# Collect (prompt, experts, success) tuples
# Train lightweight classifier or update heuristics
# Or use bandit algorithm (Thompson sampling, UCB)
```

---

## Component 4: Inference Runtime (RI)

### Overview
GPU-based execution engine that loads the base model, attaches experts, manages KV cache, and generates output. Supports CUDA (NVIDIA), ROCm (AMD), and CPU fallback.

### Core Responsibilities

1. **Adapter hot-swapping**: Load/unload experts in <10ms
2. **Paged KV cache**: Efficient long-context memory management
3. **Parallel execution**: CUDA streams for concurrent inference
4. **Quantization**: INT4/INT8 compute kernels
5. **Speculative decoding** (optional): MB draft + MB+EXPs verification

### Adapter Application

```rust
// Conceptual adapter attachment
impl InferenceRuntime {
    fn attach_experts(&mut self, session: &Session, experts: &[Expert]) -> Result<()> {
        for expert in experts {
            match expert.adapter_type {
                AdapterType::LoRA => {
                    // Load LoRA weights from .expert package
                    let (B, A) = load_lora_weights(expert.path)?;
                    
                    // Attach to target modules
                    for module_name in expert.target_modules {
                        let module = self.base_model.get_module_mut(module_name)?;
                        module.attach_lora(B.clone(), A.clone(), expert.alpha, expert.rank);
                    }
                },
                AdapterType::IA3 => {
                    let scaling_vectors = load_ia3_weights(expert.path)?;
                    // Attach scaling to activations
                    // ...
                },
                AdapterType::SoftPrompt => {
                    let embeddings = load_soft_prompt(expert.path)?;
                    session.prepend_embeddings(embeddings);
                },
            }
        }
        Ok(())
    }
}
```

### Paged KV Cache

Inspired by vLLM's PagedAttention:

```
Physical KV cache: Contiguous GPU memory divided into blocks (e.g., 16 tokens per block)

Logical sequences: Map to non-contiguous physical blocks

Example:
Sequence 1 (prompt + generation):
  Logical: [0, 1, 2, 3, 4, 5, ...]
  Physical blocks: [Block 0, Block 3, Block 7, ...]

Sequence 2:
  Logical: [0, 1, 2, ...]
  Physical blocks: [Block 1, Block 4, ...]
```

**Advantages**:
- No memory fragmentation
- Dynamic allocation as generation proceeds
- Can evict cold blocks if memory pressure

**Critical constraint**: KV cache is **not portable** between different expert sets. If experts change mid-generation, cache must be invalidated.

### CUDA Streams for Parallelism

```python
# Multiple inference jobs with different expert sets
stream_1 = cuda.Stream()  # Job 1: JSON parsing experts
stream_2 = cuda.Stream()  # Job 2: Code generation experts

with cuda.stream(stream_1):
    output_1 = model.generate(prompt_1, experts=[json_parser, english])

with cuda.stream(stream_2):
    output_2 = model.generate(prompt_2, experts=[rust_expert, async_expert])

# Runs concurrently if VRAM permits
```

### Speculative Decoding (Optional, P6)

**Idea**: Use base model alone as fast draft generator, then verify with base+experts.

```
1. Draft: MB generates N tokens speculatively (fast, no experts)
2. Verify: MB+EXPs checks draft in parallel
3. Accept: Keep correct prefix, reject divergence
4. Repeat: Continue from accepted prefix

Speedup: ~1.5-2x if draft acceptance rate >70%
```

---

## Component 5: Marketplace

### Overview
Decentralized catalog of expert packages with signature verification, version control, and compatibility checking. Users can discover, download, and install experts created by the community.

### Expert Registry Structure

```
Local registry: ~/.expert/registry.json

{
  "experts": [
    {
      "name": "json-parser",
      "version": "2.1.3",
      "path": "~/.expert/store/json-parser-2.1.3.expert",
      "manifest_hash": "sha256:abc123...",
      "base_model_compatibility": "qwen3-0.6b:sha256:def456...",
      "installed_at": "2025-10-15T14:30:00Z",
      "publisher": "hivellm",
      "publisher_pubkey": "ed25519:...",
      "verified": true
    },
    // ...
  ]
}
```

### Signature Verification

Each `.expert` package includes a signature:

```python
# Package signing (publisher)
private_key = load_ed25519_private_key()
package_hash = sha256(manifest + weights + soft_prompts + ...)
signature = private_key.sign(package_hash)

# Verification (user)
public_key = load_publisher_pubkey(manifest.publisher)
assert public_key.verify(signature, package_hash), "Invalid signature"
```

### Compatibility Checking

```json
// manifest.json
{
  "base_model": {
    "name": "Qwen3-0.6B",
    "sha256": "base_model_weights_hash",
    "rope_scaling": "yarn-128k"
  },
  "constraints": {
    "max_chain": 10,
    "incompatible_with": ["neo4j-legacy<=0.9.0", "old-json-parser<1.0.0"]
  }
}
```

**Checks before installation**:
1. Base model hash matches (or compatible variant)
2. RoPE scaling method matches
3. No incompatibilities with installed experts
4. Version constraints satisfied

### Discovery & Search

```bash
# Search marketplace
expert-cli search "json parsing"

# Results:
# - json-parser (v2.1.3) by hivellm ⭐4.8 📦 15MB
# - advanced-json (v1.0.0) by community ⭐4.2 📦 25MB

# Install
expert-cli install json-parser@2.1.3

# Auto-installs dependencies if specified in manifest
```

---

## Component 6: Multi-Agent Orchestrator

### Overview
Manages concurrent inference jobs, each with its own expert configuration. Handles job queuing, priority scheduling, VRAM budgeting, and telemetry collection.

### Job Queue Architecture

```
┌─────────────────────────────────────────┐
│         Job Submission Queue            │
│  FIFO with priority levels (0-9)        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │   VRAM Budget Check  │
       │   Can we fit job?    │
       └──────┬───────────────┘
              │
         YES  │  NO
              │   │
              ▼   ▼
         ┌────────────┐
         │  Execute   │   Queue (wait for VRAM)
         └─────┬──────┘
               │
               ▼
       ┌────────────────────┐
       │  RG (CPU thread)   │
       │  Select experts    │
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  RI (GPU stream)   │
       │  Load + Infer      │
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  Cleanup           │
       │  Release VRAM      │
       │  Collect metrics   │
       └────────────────────┘
```

### VRAM Budgeting

```python
class VRAMBudget:
    def __init__(self, total_vram_gb=16):
        self.total = total_vram_gb * 1024  # MB
        self.base_model = 500  # MB (Qwen3-0.6B INT4)
        self.reserved = 100  # MB (system overhead)
        self.available = self.total - self.base_model - self.reserved
        self.allocated = {}  # {job_id: vram_mb}
    
    def can_allocate(self, job_id, experts):
        required = sum(exp.vram_mb for exp in experts)
        required += 200  # KV cache estimate per job
        
        currently_used = sum(self.allocated.values())
        return (currently_used + required) <= self.available
    
    def allocate(self, job_id, experts):
        self.allocated[job_id] = sum(exp.vram_mb for exp in experts) + 200
    
    def release(self, job_id):
        del self.allocated[job_id]
```

### Preemption Strategy

**Light preemption**: If new high-priority job arrives and VRAM is full:

1. **Reuse hot experts**: If incoming job uses same experts as running job, share them
2. **Evict LRU experts**: Remove least-recently-used experts from cache
3. **Queue low-priority**: If neither works, queue the new job

```python
if not budget.can_allocate(new_job.id, new_job.experts):
    # Try to find shared experts
    shared = find_shared_experts(new_job.experts, running_jobs)
    if shared:
        load_only_missing(new_job.experts - shared)
    else:
        # Evict LRU
        evict_lru_experts(required_space=new_job.vram_estimate())
```

### Telemetry

Collected metrics per job:

```json
{
  "job_id": "uuid",
  "submitted_at": "2025-11-02T10:30:00Z",
  "router_latency_ms": 15,
  "expert_load_latency_ms": 8,
  "inference_latency_ms": 1250,
  "total_latency_ms": 1273,
  "experts_used": ["json-parser", "english", "neo4j-schema"],
  "vram_peak_mb": 850,
  "tokens_generated": 420,
  "success": true,
  "error": null
}
```

**Aggregated analytics**:
- Router accuracy (did selected experts produce valid output?)
- Expert popularity (which are used most?)
- Latency percentiles (p50, p95, p99)
- VRAM utilization over time

---

## Inter-Component Communication

### Data Flow Example

```
User submits:
  prompt: "Classify this JSON about Neo4j"
  body: {...}

1. Orchestrator: Enqueue job → assign ID
2. Router (CPU):
   - Heuristic: Detects JSON, Neo4j, classification task
   - Embedding: Finds similar past tasks
   - Selects: [json-parser, english, neo4j-schema, classifier]
   - Tunes: temp=0.3, max_tokens=512
   - Returns: ExpertPlan

3. Orchestrator: Check VRAM budget (OK? proceed)

4. Loader:
   - Decompress json-parser.expert from SSD
   - Load safetensors → VRAM
   - Attach LoRA to base model layers
   - Repeat for other 3 experts
   - Total time: ~25ms (hot cache) or ~150ms (cold)

5. Inference Runtime (GPU):
   - Initialize paged KV cache
   - Run forward pass with attached adapters
   - Generate 420 tokens (~8s on RTX 4090)

6. Post-process:
   - Validate JSON output
   - Collect metrics
   - Feedback to Router cache

7. Orchestrator:
   - Release VRAM
   - Return result to user
   - Log telemetry
```

### Performance Characteristics

| Component | Latency | Bottleneck |
|-----------|---------|------------|
| Orchestrator (queue) | <1ms | Rarely a bottleneck |
| Router (CPU) | 10-50ms | Embedding + heuristics |
| Expert loader (hot) | 1-10ms | VRAM mapping |
| Expert loader (cold) | 50-200ms | SSD I/O |
| Inference (GPU) | 500ms-10s | Sequence length, batch size |
| Post-process | 1-10ms | Validation complexity |

**Optimization targets**:
- Keep Router < 20ms (parallel with previous inference)
- Keep loader < 10ms (hot cache hit rate >80%)
- Maximize GPU utilization (batch when possible)

---

## Technology Stack

### Language Selection: Python vs Rust

**When to use each:**

#### Python (Training & Tooling)

**Use for:**
- Training pipelines and expert development
- Data curation and ETL scripts
- Validation and evaluation frameworks

**Why:**
- Unbeatable ML ecosystem: PyTorch, PEFT/LoRA, TRL/DPO, datasets, bitsandbytes, safetensors
- Rapid iteration on experts (.expert creation, validation, export)
- Rich data processing libraries
- Easy prototyping and experimentation

**Trade-offs:**
- ✅ **Pros**: Productivity, extensive ML libraries, fast iteration
- ❌ **Cons**: GC overhead for runtime-critical paths, packaging complexity

#### Rust (Runtime & Orchestration)

**Use for:**
- Inference server core
- Expert loading/unloading engine
- Marketplace CLI and verification
- Production deployment

**Why:**
- Low latency + fine-grained memory control (hot-swap LoRA/IA³/soft-prompts)
- Predictable performance (no GC pauses)
- Easy packaging: single binary/CLI, native libs with bindings (Node NAPI / Python PyO3)
- Efficient paged KV cache, streaming, multi-thread/GPU streams
- Perfect for marketplace (ed25519 signatures, local index, LRU cache, SSD I/O)

**Trade-offs:**
- ✅ **Pros**: Performance, single binary, predictable behavior, memory safety
- ❌ **Cons**: Fewer ML libraries, steeper learning curve, slower initial development

---

### Recommended Architecture Split

#### Core Inference (Rust)
- Load quantized base model (INT4/INT8)
- Attach/detach up to 10 experts (.expert) per session in milliseconds
- Paged KV cache per job
- Speculative decoding (optional)
- Backends: CUDA (NVIDIA), ROCm (AMD), CPU fallback
- Expose: gRPC/HTTP API + Node bindings (NAPI) + Python bindings (PyO3)

**Stack:**
- **Rust** with Candle or Burn (GPU tensor ops)
- **llama.cpp** integration (quantization, paged KV reference)
- **CUDA** 12.x / **ROCm** 6.x
- **Safetensors** (weight storage)
- **Tokio** (async runtime)
- **tonic** (gRPC) or **axum** (HTTP/2)

#### Expert Training Tooling (Python)
- LoRA/IA³/DoRA/soft-prompt pipelines with PyTorch/PEFT
- Validation metrics + evaluation frameworks
- Export to .expert (manifest + safetensors + signature)
- Dataset curation and augmentation tools
- Synthetic data generation (DeepSeek, Claude, GPT integration)

**Stack:**
- **PyTorch** >=2.0
- **Transformers** >=4.35
- **PEFT** >=0.7
- **TRL** (DPO/RLHF)
- **datasets**
- **bitsandbytes**
- **safetensors**

#### Router/Reasoning (Hybrid)
- **Production**: Rust implementation for minimal latency (<20ms)
- **Research**: Train/pilot policy in Python, port final model to Rust
- Use lightweight models (MiniLM for embeddings)

**Stack:**
- **Rust**: Candle for inference, ort for ONNX runtime
- **Python**: SentenceTransformers, scikit-learn for prototyping
- **FAISS** or Vectorizer MCP (ANN search)

#### Marketplace & CLI (Rust)
- `expert install/verify/list/prune` commands
- Ed25519 signatures and verification
- Compatibility checks (base model hash)
- SSD cache with LRU eviction
- Local registry management

**Stack:**
- **ed25519-dalek** (signatures)
- **sha2** (hashing)
- **semver** (versioning)
- **tar** / **zstd** (compression)
- **clap** (CLI framework)

---

## Security Considerations

1. **Signature verification**: Prevent malicious expert injection
2. **Sandboxing**: Experts are pure weights—no code execution
3. **VRAM limits**: Prevent OOM attacks via huge experts
4. **Rate limiting**: Marketplace downloads, API requests
5. **Audit logs**: Track expert installations, updates

---

## Future Extensions

- **Automatic expert merging**: If user always loads same 3 experts, offer to merge them
- **Expert chaining**: Define workflows (expert A → expert B → expert C)
- **Multi-GPU**: Split base model + experts across GPUs
- **Federated learning**: Users contribute training data to improve shared experts
- **Expert marketplace incentives**: Reward high-quality expert creators

---

## References

- LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- vLLM: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)
- Qwen2/Qwen3 technical reports
- DoRA: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)
- IA³: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (Liu et al., 2022)

