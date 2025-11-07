# Expert CLI

> Unified command-line interface for all expert operations

## Overview

The `expert-cli` is the **single source of truth** for all expert operations. No custom scripts per expert - all functionality is standardized and centralized in the CLI.

**Why centralized CLI:**
- ✅ Consistent commands across all experts
- ✅ Standardized workflows
- ✅ All configuration in `manifest.json`
- ✅ No "script soup" in every expert repository
- ✅ Easy to maintain and upgrade

---

## Installation

```bash
# Install expert-cli (Rust binary)
cargo install expert-cli

# Or download binary
curl -sSL https://expert.hivellm.dev/cli/install.sh | bash

# Verify
expert-cli --version
```

---

## Commands

### Dataset Generation

```bash
expert-cli dataset generate \
  --manifest manifest.json \
  --output datasets/data.jsonl
```

**What it does:**
1. Reads `training.dataset.generation` from manifest.json
2. Connects to specified LLM provider (DeepSeek, Claude, GPT-4o)
3. Generates synthetic examples based on domain/task/count
4. Applies quality filters (diversity, validation, deduplication)
5. Saves as JSONL

**Manifest configuration:**
```json
{
  "training": {
    "dataset": {
      "generation": {
        "domain": "json-parsing",
        "task": "parse and validate JSON documents",
        "count": 8000,
        "provider": "gpt-4o",
        "temperature": 0.8,
        "diversity_threshold": 0.85
      }
    }
  }
}
```

**Options:**
```bash
--manifest <path>        # Path to manifest.json (default: ./manifest.json)
--output <path>          # Output JSONL file
--provider <name>        # Override provider (deepseek, claude, gpt-4o)
--count <n>              # Override count
--append                 # Append to existing dataset
--validate               # Validate after generation
```

---

### Training

```bash
expert-cli train \
  --manifest manifest.json \
  --dataset datasets/data.jsonl \
  --output weights/
```

**What it does:**
1. Reads `training.config` from manifest.json
2. Loads base model (Qwen3-0.6B with specified quantization)
3. Configures adapter (LoRA/DoRA/IA³) from manifest
4. Trains on dataset with specified hyperparameters
5. Saves adapter weights to output directory

**Manifest configuration:**
```json
{
  "training": {
    "config": {
      "method": "sft",
      "adapter_type": "lora",
      "rank": 16,
      "alpha": 16,
      "target_modules": ["q_proj", "v_proj", "o_proj"],
      "epochs": 3,
      "learning_rate": 0.0003,
      "batch_size": 4,
      "gradient_accumulation_steps": 4,
      "warmup_steps": 100,
      "lr_scheduler": "cosine"
    }
  }
}
```

**Options:**
```bash
--manifest <path>        # Path to manifest.json
--dataset <path>         # Path to training dataset
--output <path>          # Output directory for weights
--base-model <path>      # Override base model path
--epochs <n>             # Override epochs
--resume <checkpoint>    # Resume from checkpoint
--eval-split <ratio>     # Validation split (default: 0.1)
--device <device>        # cuda, cpu, or auto (default)
```

---

### Validation

```bash
expert-cli validate \
  --expert weights/expert-name.v1.0.0
```

**What it does:**
1. Loads trained adapter
2. Runs test cases (from `tests/` directory)
3. Measures accuracy metrics
4. Validates manifest.json structure
5. Checks compatibility with base model

**Options:**
```bash
--expert <path>          # Path to expert directory or .expert file
--test-set <path>        # Custom test set (JSONL)
--metrics <metrics>      # Metrics to compute (accuracy,f1,bleu)
--verbose                # Show individual test results
```

---

### Packaging

```bash
expert-cli package \
  --manifest manifest.json \
  --weights weights/expert-name.v1.0.0 \
  --output weights/expert-name.v1.0.0.expert
```

**What it does:**
1. Reads manifest.json
2. Collects adapter weights (safetensors)
3. Collects soft prompts (if any)
4. Bundles into tar.gz with manifest
5. Computes file hashes
6. Creates `.expert` package

**Options:**
```bash
--manifest <path>        # Path to manifest.json
--weights <path>         # Path to weights directory
--output <path>          # Output .expert file
--compression <type>     # gzip (default), zstd, lz4
--include-license        # Include LICENSE file
```

---

### Multi-Model Support

Starting from schema v2.0, experts can support multiple base models (e.g., Qwen3-0.6B and Qwen3-1.5B) within a single expert repository.

#### Schema v2.0 Manifest

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
      "adapters": [{
        "type": "lora",
        "path": "weights/qwen3-0.6b/adapter.safetensors",
        "r": 16,
        "alpha": 16
      }]
    },
    {
      "name": "Qwen3-1.5B",
      "sha256": "xyz789...",
      "quantization": "int4",
      "rope_scaling": "yarn-128k",
      "adapters": [{
        "type": "lora",
        "path": "weights/qwen3-1.5b/adapter.safetensors",
        "r": 16,
        "alpha": 16
      }]
    }
  ]
}
```

#### Training for Specific Model

```bash
# Train for Qwen3-0.6B
expert-cli train \
  --manifest manifest.json \
  --model qwen3-0.6b \
  --dataset datasets/data.jsonl

# Train for Qwen3-1.5B
expert-cli train \
  --manifest manifest.json \
  --model qwen3-1.5b \
  --dataset datasets/data.jsonl
```

#### Packaging Multi-Model Experts

```bash
# Package for Qwen3-0.6B (creates separate .expert file)
expert-cli package \
  --manifest manifest.json \
  --model qwen3-0.6b \
  --output english-basic-qwen3-0.6b.v2.0.0.expert

# Package for Qwen3-1.5B
expert-cli package \
  --manifest manifest.json \
  --model qwen3-1.5b \
  --output english-basic-qwen3-1.5b.v2.0.0.expert
```

**What packaging does:**
1. Filters manifest to include only selected model
2. Copies model-specific weights from `weights/<model-name>/`
3. Copies shared resources (soft prompts, license)
4. Generates model-specific filename: `expert-name-model.vX.Y.Z.expert`
5. Computes hashes for included files only

#### Installation

```bash
# Auto-detects installed base model and selects matching variant
expert-cli install https://github.com/user/english-basic@v2.0.0

# Or specify explicitly
expert-cli install \
  https://github.com/user/english-basic@v2.0.0 \
  --base-model qwen3-1.5b
```

#### Validation

```bash
# Validates both v1.0 and v2.0 manifests
expert-cli validate --expert expert-directory/

# Checks performed:
# - Schema version detection
# - For v1.0: Ensures base_model and adapters at root
# - For v2.0: Ensures base_models array exists and is non-empty
# - Validates weight paths are unique across models
# - Prevents both base_model and base_models
```

#### Directory Structure

```
expert-multi-model/
├── manifest.json              # Schema v2.0
├── weights/
│   ├── qwen3-0.6b/
│   │   └── adapter.safetensors
│   └── qwen3-1.5b/
│       └── adapter.safetensors
├── soft_prompts/              # Shared across models
│   └── intro.pt
└── LICENSE
```

#### Benefits

- ✅ Single repository for multiple model variants
- ✅ Shared datasets and training configuration
- ✅ Consistent capabilities across model sizes
- ✅ Users download only their model variant
- ✅ Easier maintenance (update once, package for all)

#### Backward Compatibility

Schema v1.0 manifests continue to work without changes. To upgrade:

```bash
# v1.0 manifest
{
  "schema_version": "1.0",  # or omit (defaults to 1.0)
  "base_model": {...},
  "adapters": [...]
}

# v2.0 manifest
{
  "schema_version": "2.0",
  "base_models": [
    {
      "name": "...",
      "adapters": [...]
    }
  ]
}
```

---

### Signing

```bash
expert-cli sign \
  --expert weights/expert.v1.0.0.expert \
  --key ~/.expert/keys/publisher.pem
```

**What it does:**
1. Computes SHA-256 of all files in package
2. Signs with Ed25519 private key
3. Updates manifest.json with signature
4. Re-packages .expert file

**Key generation:**
```bash
# Generate signing key (once)
expert-cli keygen \
  --output ~/.expert/keys/publisher.pem \
  --name "Your Name" \
  --email "you@example.com"

# This creates:
# ~/.expert/keys/publisher.pem (private key - keep secret!)
# ~/.expert/keys/publisher.pub (public key - distribute)
```

**Options:**
```bash
--expert <path>          # Path to .expert file
--key <path>             # Path to private key
--algorithm <alg>        # ed25519 (default)
```

---

### Installation

```bash
# Install from Git repository
expert-cli install https://github.com/hivellm/expert-json-parser

# Install specific version
expert-cli install https://github.com/hivellm/expert-json-parser@v2.0.0

# Install from local path
expert-cli install /path/to/expert-directory

# Install from .expert file
expert-cli install /path/to/expert.v1.0.0.expert
```

**What it does:**
1. Clones Git repository (or copies local files)
2. Checks manifest.json for pre-built weights
3. If no weights, optionally trains locally
4. Verifies signature (if present)
5. Checks compatibility with base model
6. Resolves and installs dependencies (from `requires` field)
7. Registers in local expert registry

**Options:**
```bash
--no-deps                # Don't install dependencies
--no-train               # Don't train if weights missing
--force                  # Force reinstall
--verify                 # Verify signature (fail if missing/invalid)
--skip-verify            # Skip signature verification
```

---

### Management

```bash
# List installed experts
expert-cli list
expert-cli list --verbose  # Show details

# Show expert info
expert-cli info json-parser

# Update expert (git pull)
expert-cli update json-parser
expert-cli update --all

# Remove expert
expert-cli remove json-parser

# Verify expert integrity
expert-cli verify json-parser
```

---

### Marketplace

```bash
# Update marketplace index (git pull on marketplace repo)
expert-cli marketplace update

# Search experts
expert-cli marketplace search "json"
expert-cli marketplace search --category formats

# Browse categories
expert-cli marketplace categories

# Submit your expert
expert-cli marketplace submit https://github.com/you/your-expert
```

---

### Inference

```bash
# Run inference with expert
expert-cli infer \
  --expert json-parser \
  --prompt "Parse this JSON" \
  --input data.json

# Use multiple experts
expert-cli infer \
  --experts json-parser,english-basic,classifier \
  --prompt "Classify this document" \
  --input document.txt

# Auto-select experts (router)
expert-cli infer \
  --auto \
  --prompt "Parse and classify this JSON" \
  --input data.json
```

**Options:**
```bash
--expert <name>          # Single expert
--experts <list>         # Multiple experts (comma-separated)
--auto                   # Let router select experts
--prompt <text>          # Instruction prompt
--input <file>           # Input file
--temperature <float>    # Generation temperature
--max-tokens <int>       # Max output tokens
--output <file>          # Save output to file
--stream                 # Stream output (real-time)
```

---

## Configuration

### Global Config (~/.expert/config.json)

```json
{
  "base_model": {
    "path": "~/.expert/models/qwen3-0.6b-int4",
    "quantization": "int4"
  },
  "providers": {
    "deepseek": {
      "api_key_env": "DEEPSEEK_API_KEY",
      "endpoint": "https://api.deepseek.com"
    },
    "openai": {
      "api_key_env": "OPENAI_API_KEY"
    },
    "anthropic": {
      "api_key_env": "ANTHROPIC_API_KEY"
    }
  },
  "marketplace": {
    "index_url": "https://github.com/hivellm/expert-marketplace",
    "update_interval_hours": 24,
    "trusted_publishers": ["hivellm", "verified-org"]
  },
  "runtime": {
    "device": "cuda",
    "max_vram_gb": 16,
    "cache_dir": "~/.expert/cache"
  }
}
```

### Set Config

```bash
# Set values
expert-cli config set providers.deepseek.api_key "your-key"
expert-cli config set runtime.max_vram_gb 12

# Get values
expert-cli config get providers.deepseek.api_key

# Show all
expert-cli config list
```

---

## Complete Workflow Example

### Create and Publish Expert

```bash
# 1. Create expert directory
mkdir expert-myexpert
cd expert-myexpert

# 2. Create manifest.json
cat > manifest.json << 'EOF'
{
  "name": "myexpert",
  "version": "1.0.0",
  "description": "My custom expert",
  "base_model": {...},
  "training": {
    "dataset": {
      "generation": {
        "domain": "my-domain",
        "task": "my task description",
        "count": 5000,
        "provider": "deepseek"
      }
    },
    "config": {
      "method": "sft",
      "adapter_type": "lora",
      "rank": 16,
      "alpha": 16,
      "target_modules": ["q_proj", "v_proj"],
      "epochs": 3
    }
  },
  ...
}
EOF

# 3. Generate dataset (reads manifest.json)
expert-cli dataset generate --manifest manifest.json

# 4. Train (reads manifest.json)
expert-cli train --manifest manifest.json

# 5. Validate
expert-cli validate --expert weights/myexpert.v1.0.0

# 6. Package
expert-cli package --manifest manifest.json

# 7. Sign
expert-cli sign --expert weights/myexpert.v1.0.0.expert

# 8. Publish to Git
git init
git add .
git commit -m "Initial release v1.0.0"
git tag v1.0.0
git remote add origin https://github.com/you/expert-myexpert.git
git push -u origin main v1.0.0

# 9. Submit to marketplace
expert-cli marketplace submit https://github.com/you/expert-myexpert
```

### Install and Use Expert

```bash
# Install
expert-cli install https://github.com/you/expert-myexpert

# Use
expert-cli infer \
  --expert myexpert \
  --prompt "Do the task" \
  --input data.txt
```

---

## CLI Architecture

```
expert-cli (Rust binary)
├── Commands
│   ├── dataset (subcommands: generate, validate, split)
│   ├── train (SFT, DPO, distillation)
│   ├── validate (test expert)
│   ├── package (create .expert)
│   ├── sign (ed25519 signature)
│   ├── install (from Git/local/.expert)
│   ├── list/info/update/remove (management)
│   ├── marketplace (search, submit, update)
│   ├── infer (run inference)
│   └── config (get/set config)
│
├── Core Modules
│   ├── dataset_generator (calls LLM APIs)
│   ├── trainer (Python binding via PyO3)
│   ├── validator (run tests, measure metrics)
│   ├── packager (tar.gz creation)
│   ├── signer (ed25519 signing)
│   ├── installer (Git clone, dependency resolution)
│   ├── registry (local expert index)
│   └── inference_engine (load model + experts)
│
└── Python Bindings (via PyO3)
    ├── PyTorch/PEFT integration
    ├── Transformers model loading
    └── Training loop execution
```

---

## Environment Variables

```bash
# LLM Provider API Keys
export DEEPSEEK_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Hugging Face (for model downloads)
export HF_TOKEN="your-token"

# Expert CLI
export EXPERT_HOME="~/.expert"
export EXPERT_CACHE_DIR="~/.expert/cache"
export EXPERT_DEVICE="cuda"  # or "cpu", "cuda:0"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Missing dependencies |
| 4 | Compatibility error |
| 5 | Signature verification failed |
| 10 | Training failed |
| 11 | Validation failed |
| 12 | Packaging failed |

---

## Next Steps

- See [GIT_DISTRIBUTION.md](GIT_DISTRIBUTION.md) for Git-based distribution
- See [EXPERT_FORMAT.md](EXPERT_FORMAT.md) for manifest.json spec
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training details

