# Expert Package Format

> Complete specification of the `.expert` package format, manifest schema, and integrity verification

## Overview

Expert packages (`.expert` files) are self-contained, signed archives that bundle adapter weights, metadata, and optional components for runtime composition with the base model. They are designed to be:

- **Portable**: Work across systems with compatible base models
- **Verifiable**: Cryptographically signed for authenticity
- **Versioned**: Semantic versioning for compatibility tracking
- **Efficient**: Compressed for fast distribution and storage
- **Composable**: Multiple experts can be loaded together

## Package Structure

A `.expert` file is a **tar.gz** or **zip** archive with the following structure:

```
english-basic.v1.2.0.expert
├── manifest.json          # Required: Metadata, routing, integrity
├── weights.safetensors    # Required: Adapter weights (LoRA/IA³/DoRA)
├── soft_prompts/          # Optional: Prompt tuning embeddings
│   ├── intro.pt
│   └── style.pt
├── tokenizer_delta.json   # Optional: Additional vocabulary/rules
├── calibration.json       # Optional: Logit scaling recommendations
├── license.txt            # Optional: License terms
├── README.md              # Optional: Usage documentation
└── signature.sig          # Required: Ed25519 signature of package
```

### Required Files

1. **manifest.json**: Core metadata and configuration
2. **weights.safetensors**: Adapter parameters (one or more)
3. **signature.sig**: Cryptographic signature

### Optional Files

- **soft_prompts/*.pt**: PyTorch tensors for prompt tuning
- **tokenizer_delta.json**: Custom tokens or tokenization rules
- **calibration.json**: Recommended scaling factors for logits
- **license.txt**: License under which expert is distributed
- **README.md**: Human-readable documentation

---

## Manifest Schema

The `manifest.json` file contains all metadata about the expert. It follows this schema:

### Complete Example

```json
{
  "name": "english-basic",
  "version": "1.2.0",
  "description": "English language understanding and generation specialist",
  "author": "hivellm",
  "homepage": "https://experts.hivellm.dev/english-basic",
  
  "base_model": {
    "name": "Qwen3-0.6B",
    "sha256": "a1b2c3d4e5f6...base_model_hash",
    "quantization": "int4",
    "rope_scaling": "yarn-128k"
  },
  
  "adapters": [
    {
      "type": "lora",
      "target_modules": [
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
      ],
      "r": 16,
      "alpha": 16,
      "scaling": "standard",
      "dropout": 0.05,
      "path": "weights.safetensors",
      "size_bytes": 8388608,
      "sha256": "abc123...weights_hash"
    }
  ],
  
  "soft_prompts": [
    {
      "name": "english-style",
      "description": "Enforces formal English style",
      "path": "soft_prompts/intro.pt",
      "length": 64,
      "size_bytes": 229376,
      "sha256": "def456...soft_prompt_hash"
    }
  ],
  
  "capabilities": [
    "language:en",
    "language:en-US",
    "language:en-GB",
    "grammar",
    "style:formal",
    "comprehension"
  ],
  
  "routing": {
    "keywords": [
      "english",
      "en-US",
      "grammar",
      "spelling",
      "formal writing"
    ],
    "router_hint": "lang=en OR task=writing OR formality>0.7",
    "priority": 0.8
  },
  
  "constraints": {
    "max_chain": 10,
    "incompatible_with": [
      "english-legacy<=0.9.0",
      "slang-english@*"
    ],
    "requires": []
  },
  
  "perf": {
    "latency_ms_overhead": 1.8,
    "vram_mb_overhead": 65,
    "supported_batch_sizes": [1, 2, 4, 8]
  },
  
  "training": {
    "dataset": "synthetic-english-10k",
    "method": "sft",
    "epochs": 3,
    "learning_rate": 0.0003,
    "trained_on": "2025-10-15",
    "base_model_version": "qwen3-0.6b-int4-v1"
  },
  
  "integrity": {
    "created_at": "2025-10-30T12:00:00Z",
    "publisher": "hivellm",
    "pubkey": "ed25519:AQAB...public_key_here",
    "signature_algorithm": "ed25519",
    "signature": "hex:9f86d081884c7d659a2feaa0c55ad015a3bf4f...",
    "files": {
      "manifest.json": "sha256:e3b0c44298fc1c149afbf4c8996fb...",
      "weights.safetensors": "sha256:abc123...",
      "soft_prompts/intro.pt": "sha256:def456..."
    }
  },
  
  "license": "Apache-2.0",
  "tags": ["language", "english", "grammar", "production-ready"]
}
```

### Field Descriptions

#### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier (lowercase, hyphens) |
| `version` | string | Yes | Semantic version (e.g., "1.2.0") |
| `description` | string | Yes | Short description of expert's purpose |
| `author` | string | Yes | Creator/organization name |
| `homepage` | string | No | URL to expert's documentation/repo |
| `quality_metrics` | object | **Yes** | Standardized quality and benchmarking metrics (see below) |
| `perf` | object | **Yes** | Performance characteristics (see below) |
| `limitations` | array | No | Known limitations (strings or structured objects, see below) |

#### `base_model` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Base model name (e.g., "Qwen3-0.6B") |
| `sha256` | string | Yes | Hash of base model weights |
| `quantization` | string | Yes | Quantization scheme ("int4", "int8", "fp16") |
| `rope_scaling` | string | Yes | RoPE config ("yarn-128k", "ntk-256k") |

#### `adapters[]` Array

Each adapter object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | enum | Yes | "lora", "lora-fa", "dora", "ia3" |
| `target_modules` | string[] | Yes | Module names to attach adapter |
| `r` | int | Conditional | Rank (required for LoRA/DoRA) |
| `alpha` | int | Conditional | Scaling parameter (LoRA/DoRA) |
| `scaling` | string | No | "standard", "rsqrt", "custom" |
| `dropout` | float | No | Dropout rate during training |
| `path` | string | Yes | Relative path to safetensors file |
| `size_bytes` | int | Yes | File size in bytes |
| `sha256` | string | Yes | File hash |

**Target modules** (Qwen3-0.6B standard names):
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (MLP)
- `embed_tokens` (rare: vocabulary adaptation)
- `lm_head` (rare: custom output head)

#### `soft_prompts[]` Array

Each soft prompt object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Identifier for this soft prompt |
| `description` | string | No | What this prompt does |
| `path` | string | Yes | Relative path to .pt file |
| `length` | int | Yes | Number of virtual tokens (64-128) |
| `size_bytes` | int | Yes | File size |
| `sha256` | string | Yes | File hash |

#### `capabilities[]` Array

Structured capability tags:

- **Languages**: `language:en`, `language:pt-BR`, `language:zh`
- **Formats**: `format:json`, `format:xml`, `format:yaml`
- **Technologies**: `tech:neo4j`, `tech:postgres`, `tech:rust`
- **Tasks**: `task:classification`, `task:parsing`, `task:generation`
- **Domains**: `domain:medical`, `domain:legal`, `domain:code`

#### `routing` Object

Hints for router to select this expert:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `keywords` | string[] | Yes | Keywords that trigger this expert |
| `router_hint` | string | No | Boolean expression for advanced routing |
| `priority` | float | No | Base priority (0.0-1.0, default 0.5) |

**Router hint syntax** (example):
```
"lang=en AND (task=writing OR formality>0.7) AND NOT domain=slang"
```

#### `constraints` Object

Compatibility and composition rules:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_chain` | int | No | Max experts in same inference (default 10) |
| `incompatible_with` | string[] | No | Experts that conflict (name@version pattern) |
| `requires` | string[] | No | **Experts that must also be loaded** |
| `load_order` | int | No | Priority in loading sequence (lower loads first) |

**Version patterns**:
- `english-legacy@1.0.0` (exact)
- `english-legacy@<=0.9.0` (range)
- `slang-english@*` (any version)

**Expert Dependencies Example**:
```json
{
  "name": "document-classifier",
  "constraints": {
    "requires": [
      "english-basic@>=1.0.0",
      "json-parser@>=2.0.0"
    ],
    "load_order": 5
  }
}
```

**Dependency Resolution**:
When a user selects `document-classifier`, the router automatically:
1. Checks if `english-basic` and `json-parser` are installed
2. Loads dependencies first (by `load_order`)
3. Then loads `document-classifier`
4. Ensures total expert count ≤ 10

**Load Order Guidelines**:
- **1-2**: Format parsers (JSON, XML, YAML)
- **3-4**: Language models (English, Portuguese)
- **5-6**: Technology experts (Neo4j, Python, Rust)
- **7-8**: Domain experts (Medical, Legal)
- **9-10**: Task experts (Classification, Summarization)

#### `perf` Object

Performance characteristics (**required**):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `latency_ms_overhead` | float | **Yes** | Additional latency when expert is loaded (milliseconds) |
| `vram_mb_overhead` | int | **Yes** | Additional VRAM usage (megabytes). LoRA: ~15MB, DoRA: ~18MB, IA³: ~2MB |
| `supported_batch_sizes` | int[] | **Yes** | Batch sizes that work without OOM (minimum 1 item) |

**Example**:
```json
{
  "perf": {
    "latency_ms_overhead": 2.5,
    "vram_mb_overhead": 20,
    "supported_batch_sizes": [1, 2, 4, 8],
    "_comment": "DoRA r=14 needs 20MB VRAM. Grammar validation adds 0.5ms latency."
  }
}
```

#### `quality_metrics` Object

Standardized quality and benchmarking metrics (**required**):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `benchmark_score` | float | **Yes** | Overall benchmark score for the expert (0-10 scale recommended) |
| `base_model_score` | float | **Yes** | Score achieved by the base model for comparison (0-10 scale recommended) |
| `improvement_percent` | float | **Yes** | Percentage improvement versus base model |
| `win_rate_vs_base` | float | **Yes** | Win rate versus the base model (0.0-1.0 range, where 1.0 = 100% win rate) |
| `test_queries` | int | **Yes** | Number of queries evaluated in the benchmark |
| `checkpoint` | string | **Yes** | Identifier for the evaluated checkpoint (e.g., 'final', 'checkpoint-500') |
| `training_steps` | int | No | Training steps completed for the evaluated checkpoint |
| `test_date` | string | **Yes** | Date of the quality evaluation (ISO 8601 format: YYYY-MM-DD) |

**Note**: For experts that haven't been fully evaluated yet, use `0.0` for numeric metrics and `0` for `test_queries`, but include a `_comment` explaining that metrics are pending.

**Example**:
```json
{
  "quality_metrics": {
    "benchmark_score": 9.13,
    "base_model_score": 6.64,
    "improvement_percent": 37.5,
    "win_rate_vs_base": 0.85,
    "test_queries": 20,
    "checkpoint": "final",
    "training_steps": 655,
    "test_date": "2025-11-06",
    "_comment": "Qualitative analysis on 20 diverse queries. Strengths: MATCH patterns (10/10), aggregations (10/10). Weaknesses: AVG GROUP BY (4.2/10)."
  }
}
```

**Pending Metrics Example**:
```json
{
  "quality_metrics": {
    "benchmark_score": 0.0,
    "base_model_score": 0.0,
    "improvement_percent": 0.0,
    "win_rate_vs_base": 0.0,
    "test_queries": 0,
    "checkpoint": "adapter",
    "training_steps": 0,
    "test_date": "2025-11-08",
    "_comment": "Quality metrics pending - expert not yet fully evaluated. Dataset: 207,283 examples. Training configuration: DoRA r=12, 3 epochs."
  }
}
```

#### `limitations` Array

Known limitations or scenarios where the expert underperforms. Can be either simple strings (legacy format, backward compatible) or structured objects (recommended):

**Legacy Format** (backward compatible):
```json
{
  "limitations": [
    "no_recursive_cte",
    "no_union_operations"
  ]
}
```

**Structured Format** (recommended):
```json
{
  "limitations": [
    {
      "pattern": "no_recursive_cte",
      "description": "Recursive CTEs (WITH RECURSIVE) remain unreliable - rewrites into self-joins or subqueries instead of proper recursion",
      "example": "Query: 'Find all ancestors' → Generates self-joins instead of WITH RECURSIVE",
      "workaround": "Use iterative queries or provide explicit depth limit in prompt"
    }
  ]
}
```

**Structured Limitation Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pattern` | string | **Yes** | Identifier for the limitation pattern (e.g., 'recursive_cte', 'avg_group_by') |
| `description` | string | **Yes** | Human-readable description of the limitation |
| `example` | string | No | Example query or scenario that demonstrates the limitation |
| `workaround` | string | No | Suggested workaround or alternative approach |

#### `training.alternative_checkpoints` Object

Alternative checkpoint performance summaries (optional):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | **Yes** | Relative path to the checkpoint artifacts |
| `step` | int | **Yes** | Training step number |
| `score` | float | **Yes** | Quality score for this checkpoint |
| `win_rate` | float | **Yes** | Win rate versus base or reference checkpoint |
| `best_for` | string[] | No | Scenarios where this checkpoint excels |

**Example**:
```json
{
  "training": {
    "alternative_checkpoints": {
      "checkpoint-500": {
        "path": "weights/qwen3-06b/checkpoint-500",
        "step": 500,
        "score": 4.0,
        "win_rate": 0.40,
        "best_for": ["syntax_fixes", "stable_outputs"],
        "_comment": "Better overall score (6/15 = 40%) but still confuses complex schemas."
      }
    }
  }
}
```

#### `training` Object

Provenance information:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset` | string | No | Dataset name/ID |
| `method` | enum | No | "sft", "dpo", "rlhf", "distillation" |
| `epochs` | int | No | Training epochs |
| `learning_rate` | float | No | Learning rate used |
| `trained_on` | string | No | Date (ISO 8601) |
| `base_model_version` | string | No | Base model variant trained on |

#### `integrity` Object

Cryptographic verification:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `created_at` | string | Yes | ISO 8601 timestamp |
| `publisher` | string | Yes | Publisher identifier |
| `pubkey` | string | Yes | Ed25519 public key (base64 or hex) |
| `signature_algorithm` | string | Yes | "ed25519" |
| `signature` | string | Yes | Hex-encoded signature of all files |
| `files` | object | Yes | Map of filename → sha256 hash |

---

## Adapter Type Specifications

### LoRA (Low-Rank Adaptation)

**File format**: SafeTensors with keys:
```
base_model.model.layers.0.self_attn.q_proj.lora_A  # [r, in_features]
base_model.model.layers.0.self_attn.q_proj.lora_B  # [out_features, r]
base_model.model.layers.0.self_attn.v_proj.lora_A
base_model.model.layers.0.self_attn.v_proj.lora_B
...
```

**Manifest config**:
```json
{
  "type": "lora",
  "target_modules": ["q_proj", "v_proj"],
  "r": 16,
  "alpha": 16,
  "scaling": "standard"
}
```

**Runtime formula**:
```
W' = W + (alpha / r) * B @ A
```

### LoRA-FA (Frozen-A)

Same as LoRA, but matrix A is **not** included in safetensors (randomly initialized at load time).

**Manifest config**:
```json
{
  "type": "lora-fa",
  "r": 16,
  "alpha": 16,
  "init_std": 0.02  // For random A initialization
}
```

### DoRA (Weight-Decomposed LoRA)

**Additional keys** in safetensors:
```
base_model.model.layers.0.self_attn.q_proj.lora_magnitude  # [out_features]
```

**Manifest config**:
```json
{
  "type": "dora",
  "r": 16,
  "alpha": 16,
  "magnitude_scaling": true
}
```

### IA³ (Infused Adapter)

**File format**: SafeTensors with scaling vectors:
```
base_model.model.layers.0.self_attn.k_proj.ia3_scale  # [hidden_size]
base_model.model.layers.0.self_attn.v_proj.ia3_scale
base_model.model.layers.0.mlp.up_proj.ia3_scale
...
```

**Manifest config**:
```json
{
  "type": "ia3",
  "target_modules": ["k_proj", "v_proj", "up_proj"],
  "init_value": 1.0
}
```

**Runtime formula**:
```
activation' = activation ⊙ scale_vector
```

---

## Soft Prompts

**File format**: PyTorch tensor saved with `torch.save()`:
```python
# Shape: [num_virtual_tokens, hidden_size]
# Example: [64, 896] for 64 tokens on Qwen3-0.6B
soft_prompt = torch.randn(64, 896)
torch.save(soft_prompt, "intro.pt")
```

**Usage**:
- Prepended to input embeddings before attention
- Acts as task-specific "prefix" that guides generation
- No modification to model weights

---

## Signature Generation & Verification

### Signing Process (Publisher)

```python
import hashlib
import ed25519

# 1. Hash all files
file_hashes = {}
for filename in ["manifest.json", "weights.safetensors", ...]:
    with open(filename, "rb") as f:
        file_hashes[filename] = hashlib.sha256(f.read()).hexdigest()

# 2. Create canonical message to sign
message = "\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items()))

# 3. Sign with private key
private_key = ed25519.SigningKey(publisher_private_key_bytes)
signature = private_key.sign(message.encode())

# 4. Add to manifest
manifest["integrity"] = {
    "publisher": "hivellm",
    "pubkey": f"ed25519:{publisher_public_key.hex()}",
    "signature": signature.hex(),
    "files": file_hashes,
    ...
}

# 5. Create signature.sig file
with open("signature.sig", "w") as f:
    f.write(signature.hex())
```

### Verification Process (User)

```python
import hashlib
import ed25519

# 1. Load manifest
with open("manifest.json") as f:
    manifest = json.load(f)

# 2. Verify file hashes match
for filename, expected_hash in manifest["integrity"]["files"].items():
    with open(filename, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == expected_hash, f"Hash mismatch: {filename}"

# 3. Reconstruct signed message
message = "\n".join(
    f"{k}:{v}" 
    for k, v in sorted(manifest["integrity"]["files"].items())
)

# 4. Verify signature
public_key_hex = manifest["integrity"]["pubkey"].split(":")[1]
public_key = ed25519.VerifyingKey(bytes.fromhex(public_key_hex))

signature = bytes.fromhex(manifest["integrity"]["signature"])
public_key.verify(signature, message.encode())  # Raises if invalid

print("✓ Signature valid, expert is authentic")
```

---

## Compatibility Checking

Before loading an expert, verify:

### 1. Base Model Hash Match

```python
installed_base_hash = get_base_model_hash()
required_hash = manifest["base_model"]["sha256"]

if installed_base_hash != required_hash:
    # Check if variant is compatible (e.g., different quantization)
    if not is_compatible_variant(installed_base_hash, required_hash):
        raise IncompatibleBaseModel()
```

### 2. RoPE Scaling Match

```python
installed_rope = get_rope_config()
required_rope = manifest["base_model"]["rope_scaling"]

if installed_rope != required_rope:
    print(f"Warning: RoPE mismatch ({installed_rope} vs {required_rope})")
    # May still work but context length could be affected
```

### 3. No Conflicts with Installed Experts

```python
for incompatible_pattern in manifest["constraints"]["incompatible_with"]:
    if any(matches_pattern(exp, incompatible_pattern) for exp in installed_experts):
        raise IncompatibilityError(f"Conflicts with: {incompatible_pattern}")
```

### 4. Version Constraints

```python
import semver

for required_pattern in manifest["constraints"]["requires"]:
    # Parse "expert-name@>=1.0.0,<2.0.0"
    name, version_spec = parse_requirement(required_pattern)
    
    installed_version = get_installed_version(name)
    if not semver.match(installed_version, version_spec):
        raise MissingDependency(f"Requires: {required_pattern}")
```

---

## Packaging & Distribution

### Creating a Package

```bash
# Directory structure:
expert-build/
├── manifest.json
├── weights.safetensors
├── soft_prompts/intro.pt
└── license.txt

# Sign and package
expert-cli package \
  --input expert-build/ \
  --output english-basic.v1.2.0.expert \
  --sign-key ~/.expert/publisher.key \
  --compress gzip
```

### Publishing to Marketplace

```bash
expert-cli publish \
  --package english-basic.v1.2.0.expert \
  --registry https://registry.hivellm.dev \
  --api-key $PUBLISHER_API_KEY
```

### Installing from Marketplace

```bash
# Install latest version
expert-cli install english-basic

# Install specific version
expert-cli install english-basic@1.2.0

# Install from local file
expert-cli install ./custom-expert.v0.1.0.expert
```

---

## Size Optimization

### Typical Sizes

| Expert Type | Rank/Params | Approx Size |
|-------------|-------------|-------------|
| LoRA (r=8, attn only) | ~2M params | 8-12 MB |
| LoRA (r=16, attn+mlp) | ~8M params | 30-40 MB |
| LoRA (r=32, attn+mlp) | ~16M params | 60-80 MB |
| IA³ (attn+mlp) | ~100k params | 1-5 MB |
| Soft-prompt (64 tokens) | ~57k params | <1 MB |

### Compression

```bash
# gzip (default, good balance)
tar -czf expert.tar.gz expert-build/

# zstd (better compression, slower)
tar --zstd -cf expert.tar.zst expert-build/

# lz4 (faster decompression, larger files)
tar --lz4 -cf expert.tar.lz4 expert-build/
```

**Recommendation**: Use gzip for most cases, zstd for large experts (>100MB).

---

## Example Manifests

### Minimal Expert (IA³)

```json
{
  "name": "json-parser-tiny",
  "version": "1.0.0",
  "description": "Lightweight JSON parsing specialist",
  "author": "community",
  "base_model": {
    "name": "Qwen3-0.6B",
    "sha256": "abc123...",
    "quantization": "int4",
    "rope_scaling": "yarn-128k"
  },
  "adapters": [{
    "type": "ia3",
    "target_modules": ["k_proj", "v_proj"],
    "path": "weights.safetensors",
    "size_bytes": 2097152,
    "sha256": "def456..."
  }],
  "capabilities": ["format:json", "parsing"],
  "routing": {
    "keywords": ["json", "parse"],
    "priority": 0.9
  },
  "integrity": {
    "created_at": "2025-11-01T00:00:00Z",
    "publisher": "community",
    "pubkey": "ed25519:...",
    "signature": "...",
    "files": {
      "manifest.json": "sha256:...",
      "weights.safetensors": "sha256:..."
    }
  },
  "license": "MIT"
}
```

### Complex Expert (LoRA + Soft Prompts)

```json
{
  "name": "medical-specialist",
  "version": "2.0.1",
  "description": "Medical terminology and clinical reasoning expert",
  "author": "med-ai-lab",
  "homepage": "https://github.com/med-ai-lab/medical-expert",
  "base_model": {
    "name": "Qwen3-0.6B",
    "sha256": "abc123...",
    "quantization": "int8",
    "rope_scaling": "yarn-128k"
  },
  "adapters": [{
    "type": "dora",
    "target_modules": ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    "r": 32,
    "alpha": 32,
    "magnitude_scaling": true,
    "path": "weights.safetensors",
    "size_bytes": 67108864,
    "sha256": "..."
  }],
  "soft_prompts": [
    {
      "name": "clinical-context",
      "description": "Activates clinical reasoning mode",
      "path": "soft_prompts/clinical.pt",
      "length": 128,
      "size_bytes": 458752,
      "sha256": "..."
    }
  ],
  "capabilities": [
    "domain:medical",
    "domain:clinical",
    "language:en",
    "task:diagnosis",
    "task:terminology"
  ],
  "routing": {
    "keywords": ["medical", "clinical", "diagnosis", "symptom", "ICD", "SNOMED"],
    "router_hint": "domain=medical OR vocab=clinical",
    "priority": 0.95
  },
  "constraints": {
    "max_chain": 8,
    "incompatible_with": ["medical-legacy@<1.0.0"],
    "requires": ["english-basic@>=1.0.0"]
  },
  "perf": {
    "latency_ms_overhead": 3.2,
    "vram_mb_overhead": 120,
    "supported_batch_sizes": [1, 2, 4]
  },
  "training": {
    "dataset": "pubmed-synthetic-50k",
    "method": "dpo",
    "epochs": 5,
    "learning_rate": 0.0001,
    "trained_on": "2025-09-20",
    "base_model_version": "qwen3-0.6b-int8-v2"
  },
  "integrity": {
    "created_at": "2025-10-01T10:30:00Z",
    "publisher": "med-ai-lab",
    "pubkey": "ed25519:...",
    "signature_algorithm": "ed25519",
    "signature": "...",
    "files": {
      "manifest.json": "sha256:...",
      "weights.safetensors": "sha256:...",
      "soft_prompts/clinical.pt": "sha256:..."
    }
  },
  "license": "CC-BY-NC-4.0",
  "tags": ["medical", "healthcare", "clinical", "production-ready"]
}
```

---

## Best Practices

1. **Always sign packages**: Unsigned experts should trigger warnings
2. **Use semantic versioning**: Breaking changes → major version bump
3. **Document capabilities clearly**: Help router make good decisions
4. **Test compatibility**: Verify expert works with 2-10 other experts
5. **Optimize size**: Prefer IA³ for simple tasks, LoRA for complex
6. **Include provenance**: Document training dataset and method
7. **Specify constraints**: Mark incompatibilities explicitly
8. **Compress efficiently**: Balance load time vs download size

---

## Multi-Model Base Support

Starting from version 2.0 of the manifest schema, experts can support multiple base models. This allows a single expert to be distributed with weights for different model variants (e.g., Qwen3-0.6B and Qwen3-1.5B).

### Schema Changes

When supporting multiple models, the manifest uses `base_models` (array) instead of `base_model` (object):

```json
{
  "name": "english-basic",
  "version": "2.0.0",
  "schema_version": "2.0",
  
  "base_models": [
    {
      "name": "Qwen3-0.6B",
      "sha256": "abc123...",
      "quantization": "int4",
      "rope_scaling": "yarn-128k",
      "adapters": [
        {
          "type": "lora",
          "target_modules": ["q_proj", "v_proj"],
          "r": 16,
          "alpha": 16,
          "path": "weights/qwen3-0.6b/adapter.safetensors",
          "size_bytes": 8388608,
          "sha256": "def456..."
        }
      ]
    },
    {
      "name": "Qwen3-1.5B",
      "sha256": "xyz789...",
      "quantization": "int4",
      "rope_scaling": "yarn-128k",
      "adapters": [
        {
          "type": "lora",
          "target_modules": ["q_proj", "v_proj"],
          "r": 16,
          "alpha": 16,
          "path": "weights/qwen3-1.5b/adapter.safetensors",
          "size_bytes": 16777216,
          "sha256": "uvw012..."
        }
      ]
    }
  ]
}
```

### Directory Structure

```
expert-build/
├── manifest.json
├── weights/
│   ├── qwen3-0.6b/
│   │   └── adapter.safetensors
│   └── qwen3-1.5b/
│       └── adapter.safetensors
├── soft_prompts/          # Shared across all models
│   └── intro.pt
└── license.txt
```

### Packaging Strategy

When packaging a multi-model expert, generate **one .expert file per base model**:

```bash
# Package for Qwen3-0.6B
expert-cli package \
  --input expert-build/ \
  --output english-basic-qwen3-0.6b.v2.0.0.expert \
  --model qwen3-0.6b \
  --sign-key ~/.expert/publisher.key

# Package for Qwen3-1.5B  
expert-cli package \
  --input expert-build/ \
  --output english-basic-qwen3-1.5b.v2.0.0.expert \
  --model qwen3-1.5b \
  --sign-key ~/.expert/publisher.key
```

Each package contains:
- Manifest with **only** the selected base model entry
- Weights for that specific model
- Shared components (soft prompts, license, etc.)

### Installation Behavior

The CLI automatically selects the correct package variant:

```bash
# Auto-detects installed base model
expert-cli install english-basic@2.0.0

# Or specify explicitly
expert-cli install english-basic@2.0.0 --base-model qwen3-1.5b
```

### Backward Compatibility

For backward compatibility with schema v1.0:
- If `base_model` (singular) exists, treat as single-model expert
- If `base_models` (plural) exists, treat as multi-model expert
- Schema version field determines parsing strategy

```json
{
  "schema_version": "2.0",  // Required for multi-model
  "base_models": [...]      // Use array
}
```

vs

```json
{
  "schema_version": "1.0",  // Or omitted (defaults to 1.0)
  "base_model": {...}       // Single object
}
```

### Validation Rules

1. **Mutually exclusive**: Cannot have both `base_model` and `base_models`
2. **Consistent capabilities**: All model variants must have same capabilities
3. **Path uniqueness**: Each model must have distinct weight paths
4. **Shared resources**: Soft prompts, tokenizer deltas are shared across models
5. **Signature per package**: Each generated .expert has its own signature

### Benefits

- ✅ Single expert repository for multiple models
- ✅ Shared training configuration and datasets
- ✅ Consistent capabilities across model sizes
- ✅ Smaller distribution (one package per user's model)
- ✅ Easier maintenance (update once, publish for all models)

### Migration Path

Existing v1.0 experts continue to work. To upgrade to multi-model:

1. Change `schema_version` to `"2.0"`
2. Rename `base_model` → `base_models` (array)
3. Move `adapters` inside each base model entry
4. Update weight paths to include model-specific directories
5. Retrain or convert weights for additional models
6. Package separately for each model variant

---

## Future Extensions

- **Multi-language support**: Bundle translations of README/description
- **Differential updates**: Publish patches instead of full packages
- **Expert fusion**: Automatically merge frequently co-loaded experts
- **Lazy loading**: Stream weights on-demand for huge experts
- **Hardware profiles**: Specify min GPU, CUDA compute capability, etc.
- **Model auto-conversion**: Automatically adapt weights between similar architectures

