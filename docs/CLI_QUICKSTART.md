# Expert CLI Quick Start

Get started training your first expert in 5 minutes.

## Prerequisites

1. **Rust nightly** (1.85+):
```bash
rustup default nightly
rustup update nightly
```

2. **Python 3.11+** with virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

For CUDA support (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Build the CLI

```bash
cargo build --release
```

Binary will be in `target/release/expert-cli` (or `expert-cli.exe` on Windows).

## Train Your First Expert

### 1. Verify the Dataset

Check that the JSON parser dataset exists:
```bash
ls ../experts/expert-json-parser/datasets/json_8k.jsonl
```

### 2. Run Training

```bash
./target/release/expert-cli train \
  --manifest ../experts/expert-json-parser/manifest.json \
  --dataset ../experts/expert-json-parser/datasets/json_8k.jsonl \
  --output ../experts/expert-json-parser/weights \
  --device auto
```

**On Windows PowerShell**:
```powershell
.\target\release\expert-cli.exe train `
  --manifest ..\experts\expert-json-parser\manifest.json `
  --dataset ..\experts\expert-json-parser\datasets\json_8k.jsonl `
  --output ..\experts\expert-json-parser\weights `
  --device auto
```

### 3. Monitor Progress

The CLI will display:
- Training configuration
- GPU information (if CUDA available)
- Real-time training progress
- Loss metrics
- Estimated time remaining

Training time:
- **GPU (NVIDIA RTX 3090/4090)**: ~30-60 minutes
- **CPU**: ~4-6 hours (not recommended)

### 4. Output

After training completes, you'll find:
```
expert-json-parser/
└── weights/
    ├── adapter/           # LoRA adapter weights only
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── final/             # Full model + adapter
        ├── config.json
        ├── model.safetensors
        └── tokenizer files...
```

## Next Steps

### Validate the Expert

```bash
./target/release/expert-cli validate \
  --expert ../experts/expert-json-parser/weights/adapter
```

### Package the Expert

```bash
./target/release/expert-cli package \
  --manifest ../experts/expert-json-parser/manifest.json \
  --weights ../experts/expert-json-parser/weights/adapter \
  --output ../experts/expert-json-parser/weights/json-parser.v0.0.1.expert
```

### Sign the Expert

```bash
# First, generate signing key (once)
./target/release/expert-cli keygen \
  --output ~/.expert/keys/publisher.pem \
  --name "Your Name"

# Then sign
./target/release/expert-cli sign \
  --expert ../experts/expert-json-parser/weights/json-parser.v0.0.1.expert \
  --key ~/.expert/keys/publisher.pem
```

## Troubleshooting

### CUDA Not Detected

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

If not available, reinstall PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

If you get CUDA OOM errors, edit the manifest.json:
```json
{
  "training": {
    "config": {
      "batch_size": 2,  // Reduce from 4 to 2
      "gradient_accumulation_steps": 8  // Increase from 4 to 8
    }
  }
}
```

### Python Module Not Found

Make sure you're running from the `cli/` directory and Python can find `expert_trainer.py`:
```bash
cd expert/cli
export PYTHONPATH=".:$PYTHONPATH"
./target/release/expert-cli train ...
```

## Testing Without Training

To test the CLI without full training, set epochs to 1:
```bash
./target/release/expert-cli train \
  --manifest ../experts/expert-json-parser/manifest.json \
  --dataset ../experts/expert-json-parser/datasets/json_8k.jsonl \
  --output ../experts/expert-json-parser/weights \
  --epochs 1 \
  --device cpu
```

## Configuration

### Override Epochs

```bash
--epochs 5  # Override epochs from manifest
```

### Resume Training

```bash
--resume ../experts/expert-json-parser/weights/checkpoint-1000
```

### Force CPU

```bash
--device cpu
```

## Performance Tips

1. **Use CUDA**: Training on GPU is 10-20x faster
2. **Increase batch size**: If you have more VRAM (16GB+), increase batch_size
3. **Use mixed precision**: Already enabled by default (FP16 on CUDA)
4. **Monitor VRAM**: Use `nvidia-smi` to check GPU memory usage

## Full Workflow Example

```bash
# 1. Build CLI
cargo build --release

# 2. Activate Python environment
source venv/bin/activate

# 3. Train expert
./target/release/expert-cli train \
  --manifest ../experts/expert-json-parser/manifest.json \
  --dataset ../experts/expert-json-parser/datasets/json_8k.jsonl \
  --output ../experts/expert-json-parser/weights

# 4. Validate
./target/release/expert-cli validate \
  --expert ../experts/expert-json-parser/weights/adapter

# 5. Package
./target/release/expert-cli package \
  --manifest ../experts/expert-json-parser/manifest.json \
  --weights ../experts/expert-json-parser/weights/adapter \
  --output json-parser.v0.0.1.expert

# 6. Sign
./target/release/expert-cli sign \
  --expert json-parser.v0.0.1.expert \
  --key ~/.expert/keys/publisher.pem
```

## Help

Get help on any command:
```bash
./target/release/expert-cli --help
./target/release/expert-cli train --help
./target/release/expert-cli package --help
```

## Working with Multi-Model Experts (Schema v2.0)

### What are Multi-Model Experts?

Multi-model experts support **multiple base models** in a single manifest, allowing you to:
- Train the same expert on different model sizes (e.g., Qwen3-0.6B, Qwen3-1.5B, Qwen3-7B)
- Package model-specific adapters separately
- Maintain a single source of truth for expert capabilities

### Migration to Schema v2.0

All current experts use schema v2.0. The main differences:

**Schema v1.0 (Legacy)**:
```json
{
  "name": "expert-json",
  "version": "0.0.1",
  "base_model": {
    "name": "Qwen3-0.6B",
    "quantization": "int4"
  },
  "adapters": [...]
}
```

**Schema v2.0 (Current)**:
```json
{
  "name": "expert-json",
  "version": "0.0.1",
  "schema_version": "2.0",
  "base_models": [
    {
      "name": "Qwen3-0.6B",
      "quantization": "int4",
      "prompt_template": "chatml",
      "adapters": [...]
    }
  ]
}
```

### Training Multi-Model Experts

Training works the same way - the CLI automatically uses the first model in `base_models`:

```bash
./target/release/expert-cli train \
  --manifest ../experts/expert-neo4j/manifest.json \
  --output ../experts/expert-neo4j/weights \
  --device cuda
```

### Packaging Multi-Model Experts

For schema v2.0, you **must** specify which model to package:

```bash
# Package for Qwen3-0.6B
./target/release/expert-cli package \
  --manifest ../experts/expert-neo4j/manifest.json \
  --weights ../experts/expert-neo4j/weights \
  --model qwen3-0.6b
```

This creates: `expert-neo4j-qwen306b.v0.0.1.expert`

### Multi-Model Workflow

If your manifest has multiple models:

```json
{
  "schema_version": "2.0",
  "base_models": [
    {
      "name": "Qwen3-0.6B",
      "adapters": [{"path": "qwen3-06b/adapter", ...}]
    },
    {
      "name": "Qwen3-1.5B",
      "adapters": [{"path": "qwen3-15b/adapter", ...}]
    }
  ]
}
```

Train and package each model:

```bash
# Train for 0.6B (uses first model by default)
./target/release/expert-cli train \
  --manifest manifest.json \
  --output weights \
  --device cuda

# Package for 0.6B
./target/release/expert-cli package \
  --manifest manifest.json \
  --weights weights \
  --model qwen3-0.6b

# Train for 1.5B (TODO: add --model flag to train command)
# Currently, you need to reorder base_models array or edit manifest

# Package for 1.5B
./target/release/expert-cli package \
  --manifest manifest.json \
  --weights weights \
  --model qwen3-1.5b
```

### Prompt Templates

Schema v2.0 includes prompt template support in `base_model`:

```json
{
  "base_models": [
    {
      "name": "Qwen3-0.6B",
      "prompt_template": "chatml",  // Uses ChatML format
      "adapters": [...]
    }
  ]
}
```

Supported templates:
- `chatml` - ChatML format (Qwen, Yi, Mistral)
- `alpaca` - Alpaca format (universal default)
- `llama` - Llama 2/3, CodeLlama
- `phi` - Phi-2, Phi-3, Phi-3.5
- `deepseek` - DeepSeek models
- `gemma` - Google Gemma
- `mistral` - Mistral format

Auto-detection: If not specified, the trainer auto-detects based on model name.

### Troubleshooting Multi-Model

**Error: "Schema v2.0 requires --model flag"**
- Solution: Add `--model qwen3-0.6b` to package command

**Error: "Model 'xyz' not found in manifest"**
- Check spelling: model names are normalized (lowercase, no slashes)
- List available models in your manifest

**Backward Compatibility**
- Schema v1.0 manifests still work
- CLI automatically detects and handles both versions
- No breaking changes for existing experts

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [expert_trainer.py](expert_trainer.py) to understand the training pipeline
- See [../docs/TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md) for advanced topics
- View [../docs/EXPERT_FORMAT.md](../docs/EXPERT_FORMAT.md) for schema v2.0 details
- Check [../examples/multi-model-expert/](../examples/multi-model-expert/) for complete example

