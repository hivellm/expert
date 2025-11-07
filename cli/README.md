# Expert CLI

Command-line interface for training and managing HiveLLM experts.

## Features

- ✅ **Auto-detects Python venv** - No manual activation needed
- ✅ **Automatic DLL management** - Copies Python DLLs on Windows
- ✅ **CUDA support** - GPU training with automatic detection
- ✅ **HuggingFace integration** - Load datasets directly from HF Hub
- ✅ **LoRA fine-tuning** - Parameter-efficient training with PEFT
- ✅ **INT4 quantization** - Train on consumer GPUs (8GB+ VRAM)
- ✅ **Multi-task training** - Train on multiple datasets simultaneously
- ✅ **Resume from checkpoint** - Continue interrupted training

## Installation

### Prerequisites

1. **Rust** (nightly 1.85+):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly
```

2. **Python** (3.11+) with CUDA support:
```bash
# Create virtual environment (recommended)
python -m venv venv_windows  # Windows
python -m venv venv          # Linux/Mac

# Activate venv
.\venv_windows\Scripts\Activate.ps1  # Windows PowerShell
source venv/bin/activate              # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For CUDA support, ensure you have:
# - CUDA Toolkit 12.1+
# - cuDNN 8.9+
# - NVIDIA drivers
```

### Build from Source

**Windows:**
```powershell
# Quick build
.\rebuild-quick.ps1

# Or build with DLL auto-copy
.\rebuild-with-dlls.ps1
```

**Linux/Mac:**
```bash
cargo build --release
```

The binary will be in `target/release/expert-cli`.

> **Note:** The CLI automatically detects and activates the Python venv (`venv_windows` or `venv`) when running. No manual activation needed!

## Usage

### Chat with Qwen3 Model

The CLI includes a native Rust/Candle inference engine for Qwen3-0.6B:

```bash
# Interactive chat
expert-cli chat

# Single prompt
expert-cli chat --prompt "The capital of Brazil is"

# With experts (future)
expert-cli chat --experts neo4j,sql
```

**Parameters:**
- `--prompt`: Single prompt for one-shot generation (optional, interactive mode if not provided)
- `--experts`: Comma-separated list of experts to load (optional)
- `--device`: Device to use (cuda, cpu, auto) - default: auto
- `--base-model`: Path to base model (default: F:/Node/hivellm/expert/models/Qwen3-0.6B)

**Generation Settings** (hardcoded in chat.rs, customizable in future):
- Temperature: 0.7 (controls randomness)
- Top-p: 0.9 (nucleus sampling threshold)
- Max tokens: 50 (maximum generation length)

**Examples:**

```bash
# Code completion
expert-cli chat --prompt "def fibonacci(n):"

# Factual completion
expert-cli chat --prompt "The capital of Brazil is"

# Natural language
expert-cli chat --prompt "Hello, my name is"
```

**Quality Notes:**
- ✅ Uses native Rust/Candle (equivalent or better quality than Python/Transformers)
- ✅ CUDA acceleration with BF16 precision
- ✅ Implements NTK-by-parts RoPE scaling for long contexts (>32k tokens)
- ✅ Temperature + top-p nucleus sampling for diverse outputs
- ⚠️ BASE model (without instruction-tuning) - best for completions, not Q&A
- 💡 For better chat: use instruction-tuned variant or load LoRA experts

### Training an Expert

```bash
expert-cli train \
  --manifest ../experts/expert-json-parser/manifest.json \
  --dataset ../experts/expert-json-parser/datasets/json_8k.jsonl \
  --output ../experts/expert-json-parser/weights \
  --device cuda
```

**Options:**
- `--manifest`: Path to manifest.json (default: manifest.json)
- `--dataset`: Path to training dataset (JSONL format)
- `--output`: Output directory for weights
- `--epochs`: Override epochs from manifest
- `--device`: Device to use (cuda, cpu, auto)
- `--resume`: Resume from checkpoint

### Validate Expert

```bash
expert-cli validate --expert weights/json-parser.v0.0.1
```

### Package Expert

```bash
expert-cli package \
  --manifest manifest.json \
  --weights weights/json-parser.v0.0.1 \
  --output weights/json-parser.v0.0.1.expert
```

### Sign Expert

```bash
expert-cli sign \
  --expert weights/json-parser.v0.0.1.expert \
  --key ~/.expert/keys/publisher.pem
```

## Dataset Format

Training datasets must be in JSONL format with `instruction` and `response` fields:

```jsonl
{"instruction": "Parse this JSON: {\"name\": \"Alice\"}", "response": "{\"name\": \"Alice\"}"}
{"instruction": "Validate: {invalid json}", "response": "Invalid JSON: unexpected token"}
```

## Configuration

Global configuration is stored in `~/.expert/config.json`:

```json
{
  "base_model": {
    "path": "~/.expert/models/qwen3-0.6b-int4",
    "quantization": "int4"
  },
  "runtime": {
    "device": "cuda",
    "max_vram_gb": 16,
    "cache_dir": "~/.expert/cache"
  }
}
```

## Training Pipeline

The training process:

1. **Load Manifest**: Parse configuration from manifest.json
2. **Load Base Model**: Download/load Qwen3-0.6B with INT4 quantization
3. **Setup Adapter**: Configure LoRA with rank/alpha from manifest
4. **Load Dataset**: Parse JSONL and tokenize
5. **Train**: Fine-tune using PEFT
6. **Save Weights**: Save adapter to output directory

### Performance Optimizations (v0.2.3)

The training pipeline includes several optimizations for 2x speedup:

#### 1. SDPA (Flash Attention v2) + QLoRA

**Previously**: SDPA was disabled when using quantization, causing GPU underutilization.

**Now**: SDPA works with QLoRA INT4/INT8 for maximum throughput.

**Enable in manifest**:
```json
{
  "training": {
    "config": {
      "use_sdpa": true,
      "bf16": true,
      "use_tf32": true
    }
  }
}
```

**Impact**: +15-20% throughput, 85-95% GPU utilization

#### 2. Sequence Packing (SFTTrainer)

**Previously**: Standard Trainer with 30-40% padding waste.

**Now**: Auto-enables SFTTrainer when dataset has "text" field.

**Requirements**:
- Dataset format: `{"text": "complete_formatted_example"}`
- `trl>=0.7.0` installed

**Impact**: +30-40% tokens/s (reduces padding waste)

**Note**: Falls back to standard Trainer for datasets without "text" field.

#### 3. Combined Results

- **Training speed**: 2x faster (4hrs → 2hrs for 3 epochs on SQL dataset)
- **GPU utilization**: 85-95% (vs 60-70% before)
- **VRAM usage**: unchanged (~8GB for Qwen3-0.6B INT4 + DoRA r=12)

#### 4. Recommended Training Config

```json
{
  "training": {
    "config": {
      "bf16": true,
      "use_tf32": true,
      "use_sdpa": true,
      "max_seq_length": 2048,
      "dataloader_num_workers": 8,
      "dataloader_pin_memory": true,
      "dataloader_prefetch_factor": 8,
      "dataloader_persistent_workers": true,
      "optim": "adamw_torch_fused"
    }
  }
}
```

## Requirements

### Python Dependencies

See `requirements.txt`:
- torch (with CUDA)
- transformers
- peft
- datasets
- bitsandbytes

### Rust Dependencies

See `Cargo.toml`:
- clap (CLI parsing)
- pyo3 (Python bridge)
- serde (JSON)
- tokio (async)

## Development

### Run Tests

```bash
cargo test
```

### Format Code

```bash
cargo +nightly fmt
```

### Lint

```bash
cargo clippy -- -D warnings
```

## Troubleshooting

### Chat Issues

**Gibberish or repetitive output:**
- ✅ Fixed in v0.2.1+ (was due to mock implementation)
- Ensure you're using latest version: `expert-cli --version`

**Poor contextual relevance:**
- BASE Qwen3-0.6B lacks instruction-tuning
- Use completion-style prompts instead of questions
- Example: "The answer is" instead of "What is the answer?"
- Future: Load instruction-tuned experts with `--experts`

**CUDA not working:**
- Build with CUDA: `.\scripts\build-cuda.ps1` (Windows) or `cargo build --release --features cuda` (Linux)
- Check GPU: Model should show `Device: Cuda(...)`
- Fallback to CPU: Will show `Device: Cpu` if CUDA unavailable

**Sampling parameters:**
- Temperature: Controls randomness (0.0 = greedy, 0.7 = balanced, 1.0 = creative)
- Top-p: Nucleus sampling (0.9 = default, 1.0 = no filtering)
- Modify in `src/commands/chat.rs` lines 138-144 (customizable CLI flags in future)

### Diagnostic Tool

Run the diagnostic script to check your environment:
```bash
python diagnose_training.py
```

This checks:
- Python version and packages
- CUDA availability
- Model accessibility
- Dataset format
- Training configuration

### CUDA Not Available

If CUDA is not detected:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall torch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Out of Memory

If training fails with OOM:
1. Reduce batch size in manifest.json
2. Increase gradient accumulation steps
3. Use lower quantization (int4 instead of int8)
4. Reduce max sequence length

### DLL Not Found (Windows)

If you get exit code -1073741515 on Windows:
```powershell
# Copy Python DLLs manually
.\copy-python-dlls.ps1
```

### Python Import Errors

The CLI auto-detects venv, but if you still have issues:
1. Ensure venv exists: `venv_windows` (Windows) or `venv` (Linux/Mac)
2. Verify dependencies: `pip list | grep torch`
3. Reinstall if needed: `pip install -r requirements.txt`

### Module 'expert_trainer' Not Found

This should be fixed automatically, but if it persists:
1. Ensure `expert_trainer.py` exists in `expert/cli/`
2. Rebuild: `cargo build --release`
3. The CLI will add the correct path automatically

## Architecture

```
expert-cli (Rust)
├── CLI parsing (clap)
├── Manifest loader (serde)
├── Python bridge (PyO3)
│   └── expert_trainer.py
│       ├── Model loading (transformers)
│       ├── Dataset processing (datasets)
│       ├── LoRA setup (peft)
│       └── Training loop (Trainer)
└── Output management
```

The Rust CLI handles argument parsing, validation, and progress reporting, while delegating the actual ML training to Python via PyO3.

## License

MIT

