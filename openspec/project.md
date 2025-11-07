<!-- EXPERT_SYSTEM_PROJECT:START -->
# Expert System - Project Context

> OpenSpec project configuration for AI agent assistance

**CRITICAL**: This file provides essential context for AI agents working on the Expert System.

## Project Overview

**Name**: Expert System  
**Type**: ML Infrastructure / LLM Framework  
**Status**: Documentation phase (implementation starting)  
**Goal**: Dynamic expert composition system for specialized AI inference on consumer GPUs

## Description

The Expert System is a novel architecture that dynamically composes lightweight expert adapters (LoRA/DoRA/IA³) on top of a compact base model (Qwen3-0.6B). Instead of deploying massive models, this system loads task-specific experts on-demand, enabling fast, specialized inference on domestic GPUs (8-16GB VRAM).

**Key Innovation**: Experts are never merged into the base model—they're applied as runtime-composable sub-tensors that can be loaded/unloaded in milliseconds.

---

## Tech Stack

### Languages
- **Rust**: Core inference engine, CLI, marketplace (production runtime)
- **Python**: Training pipelines, synthetic data generation, tooling

### Rust Stack
- **Tensor ops**: candle or burn
- **Async runtime**: tokio
- **API**: tonic (gRPC) or axum (HTTP/2)
- **Bindings**: napi-rs (Node), pyo3 (Python)
- **Crypto**: ed25519-dalek (signatures)
- **CLI**: clap

### Python Stack
- **ML framework**: PyTorch >=2.0
- **Model loading**: transformers >=4.35
- **Adapters**: PEFT >=0.7 (LoRA, DoRA, IA³)
- **Training**: TRL (DPO/RLHF)
- **Data**: datasets, safetensors
- **Quantization**: bitsandbytes

### Models
- **Base model**: Qwen3-0.6B (quantized INT4/INT8)
- **Context**: 120k-200k tokens via RoPE scaling (YaRN/NTK)
- **Experts**: LoRA adapters (5-80MB each, max 10 per inference)

---

## Architecture

### Core Components

1. **Base Model (MB)**: Qwen3-0.6B quantized, fixed in VRAM
2. **Experts (EXPs)**: Lightweight adapters (LoRA/DoRA/IA³/soft-prompts)
3. **Router (RG)**: CPU-based expert selection engine
4. **Inference Runtime (RI)**: GPU engine with hot-swap adapters
5. **Marketplace**: Git-based distribution (no NPM/PyPI)
6. **Orchestrator**: Multi-agent job queue

### Data Flow

```
Prompt → Router (CPU) → Select Experts → Load Experts (SSD→VRAM) 
→ Inference (GPU) → Output
```

---

## File Structure

```
expert/
├── README.md              # Project overview
├── STATUS.md              # Current progress
├── ROADMAP.md            # Implementation phases (P0-P6)
├── QUICKSTART.md         # 4-week practical guide
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CLI.md
│   ├── EXPERT_FORMAT.md
│   ├── GIT_DISTRIBUTION.md
│   ├── EXECUTION_PIPELINE.md
│   ├── TRAINING_GUIDE.md
│   ├── SYNTHETIC_DATA.md
│   ├── ROUTING_REASONING.md
│   └── PERFORMANCE.md
├── experts/              # Official experts (real, not examples)
│   ├── expert-json-parser/
│   └── README.md
└── examples/
    └── expert-repository-template/
```

---

## Coding Conventions

### Rust

```rust
// Naming
pub struct ExpertLoader { }    // PascalCase for types
pub fn load_expert() { }       // snake_case for functions
const MAX_EXPERTS: usize = 10; // SCREAMING_SNAKE for constants

// Edition
edition = "2024"  // Use Rust 2024 edition

// Error handling
use anyhow::{Result, Context};
fn load() -> Result<Expert> {
    // Use ? operator
}

// Async
use tokio::task;
async fn process() { }
```

### Python

```python
# Naming
class ExpertTrainer:  # PascalCase for classes
def train_expert():   # snake_case for functions
MAX_EXPERTS = 10      # SCREAMING_SNAKE for constants

# Type hints
def load_model(path: str) -> Model:
    ...

# Error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Failed: {e}")
    raise
```

### JSON/JSONL

- Use 2-space indentation
- Keys in snake_case
- Manifest schema strictly enforced
- All timestamps ISO 8601

---

## Key Constraints

1. **Max 10 experts per inference**: Hard limit for VRAM and performance
2. **No expert merging**: Experts never modify base model weights
3. **KV cache isolation**: Not portable between different expert sets
4. **Load order matters**: Experts must load in sequence (format → language → tech → domain → task)
5. **Git-only distribution**: No centralized package managers (NPM, PyPI)
6. **CLI-only operations**: No custom scripts in expert repositories

---

## Development Workflow

### Creating New Expert

```bash
# 1. Create from template
git clone expert-repository-template expert-myexpert

# 2. Edit manifest.json
vim expert-myexpert/manifest.json

# 3. Generate, train, validate, package
expert-cli dataset generate --manifest manifest.json
expert-cli train --manifest manifest.json
expert-cli validate --expert weights/myexpert.v1.0.0
expert-cli package --manifest manifest.json
```

### Testing

```bash
# Validate single expert
expert-cli validate --expert expert-name

# Test multi-expert composition
expert-cli test-composition --experts json-parser,english-basic,classifier
```

---

## Documentation Standards

- **All docs in English** (user preference)
- Markdown files with proper formatting
- Code examples in all docs
- No unnecessary .md files (keep to: README, STATUS, ROADMAP, CHANGELOG, and essential docs)
- Update existing docs instead of creating new ones

---

## Git Workflow

- **No git reset** allowed
- **No git push** by agent (requires SSH password)
- After commits/tags, return commands for user to push manually
- Always use `wsl -d Ubuntu-24.04 -- bash -l -c` for commands
- Commit successful implementations with documentation updates

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Expert load (hot) | <10ms |
| Expert load (cold) | <200ms |
| Router latency | <20ms |
| Inference (1024 tokens) | <20s (RTX 4090) |
| VRAM usage | <8GB (10 experts) |
| Context window | 120k-200k tokens |

---

## Current Phase

**Phase**: Documentation complete  
**Next**: Implementation (Path A: Quick Start or Path B: Full Rust runtime)  
**Blockers**: None

---

## Dependencies

### Required on User's System
- **GPU**: NVIDIA (CUDA) or AMD (ROCm)
- **VRAM**: 8-16GB minimum
- **Storage**: SSD recommended for expert loading
- **RAM**: 16GB+ for training

### API Keys (for synthetic data)
- `DEEPSEEK_API_KEY`: DeepSeek Chat
- `OPENAI_API_KEY`: GPT-4o
- `ANTHROPIC_API_KEY`: Claude
- `HF_TOKEN`: Hugging Face (for gated models)

---

## Community

- **License**: TBD (likely MIT or Apache-2.0)
- **Repository**: TBD (https://github.com/hivellm/expert)
- **Contributions**: Welcome after P0 milestone
- **Marketplace**: Decentralized (Git-based)

---

## Notes for AI Agents

- User prefers **honesty over flattery**
- **Don't agree blindly** - be critical and realistic
- If unsure, **say you don't know** instead of making things up
- **Focus on what's requested** - don't over-engineer
- User wants **maximum realism** in responses
- User is experienced - no need to explain basics

---

## Next Milestones

1. **Immediate**: Finalize expert-json-parser structure
2. **Week 1-2**: Train first 2 experts (JSON, English)
3. **Week 3-4**: Complete 6 experts
4. **Month 2**: Build simple Python CLI prototype
5. **Month 3+**: Rust runtime implementation (P0)

---

Last Updated: 2025-11-02

<!-- EXPERT_SYSTEM_PROJECT:END -->
