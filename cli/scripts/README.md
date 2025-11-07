# Expert CLI Scripts

Python utility scripts for the Expert System CLI.

## Training Pipeline Scripts

### `pretokenize_datasets.py`
**Purpose**: Pre-tokenize datasets and save as Arrow format for fast loading during training.

**Usage**:
```bash
python scripts/pretokenize_datasets.py
```

**What it does**:
- Loads datasets from HuggingFace cache or local files
- Tokenizes with parallel processing (8 workers)
- Applies sequence packing (concatenate + chunk to 2048 tokens)
- Saves as Arrow format to `datasets_optimized/{expert}/tokenized/`
- Generates statistics (token counts, distributions)

**Output**: Pre-tokenized datasets in `datasets_optimized/` directory

---

### `find_datasets.py`
**Purpose**: Locate and list HuggingFace dataset cache.

**Usage**:
```bash
python scripts/find_datasets.py
```

**What it does**:
- Finds HuggingFace cache directory (default: `C:\Users\[user]\.cache\huggingface\datasets`)
- Lists all downloaded datasets with sizes
- Maps datasets to experts (json, typescript, neo4j, sql)

**Output**: Console report of dataset locations and sizes

---

### `benchmark_training.py`
**Purpose**: Monitor and benchmark training performance.

**Usage**:
```bash
# Start training in one terminal, then run:
python scripts/benchmark_training.py --duration 60 --output benchmark.json
```

**Options**:
- `--duration N`: Monitor for N seconds (default: 60)
- `--interval N`: Sample every N seconds (default: 2)
- `--output FILE`: Save report to JSON file

**What it does**:
- Monitors GPU utilization, VRAM usage, temperature
- Calculates statistics (mean, min, max)
- Provides performance assessment
- Exports JSON report

**Output**: Console monitoring + JSON report file

---

## Dataset Management Scripts

### `dataset_loader.py`
**Purpose**: Multi-task dataset loading utilities.

**Used by**: `expert_trainer.py` (imported)

**Features**:
- `MultiTaskDatasetLoader`: Handles multi-task dataset configurations
- Combines multiple dataset sources with weights
- Supports validation and deduplication

---

### `dataset_stats.py`
**Purpose**: Generate and display dataset statistics.

**Used by**: `expert_trainer.py` (imported)

**Functions**:
- `generate_stats()`: Compute dataset statistics
- `print_stats()`: Display statistics in formatted output

---

### `dataset_validator.py`
**Purpose**: Validate dataset format and quality.

**Used by**: `expert_trainer.py` (imported)

**Features**:
- `DatasetValidator`: Validates JSONL format
- Checks for required fields
- Validates JSON structure
- Deduplication checks

---

## Diagnostic Scripts

### `diagnose_training.py`
**Purpose**: Diagnose training issues and performance problems.

**Usage**:
```bash
python scripts/diagnose_training.py
```

**What it does**:
- Checks CUDA availability
- Verifies model and dataset paths
- Tests tokenizer
- Validates configuration

---

### `check_cuda.py`
**Purpose**: Check CUDA installation and GPU availability.

**Usage**:
```bash
python scripts/check_cuda.py
```

**Output**: CUDA version, GPU info, PyTorch compatibility

---

### `clear_cuda_cache.py`
**Purpose**: Clear CUDA cache to free GPU memory.

**Usage**:
```bash
python scripts/clear_cuda_cache.py
```

**When to use**: Before starting training to ensure clean GPU state

---

## Development Scripts

### `test_hf_dataset.py`
**Purpose**: Test HuggingFace dataset loading.

**Usage**:
```bash
python scripts/test_hf_dataset.py
```

**What it does**: Validates that HuggingFace datasets library works correctly

---

### `validate_grammar.py`
**Purpose**: Validate grammar files for constrained generation.

**Usage**:
```bash
python scripts/validate_grammar.py <grammar_file.gbnf>
```

**What it does**: Checks GBNF grammar syntax

---

## Utility Scripts

### `download_model.py`
**Purpose**: Download base models from HuggingFace.

**Usage**:
```bash
python scripts/download_model.py <model_name>
```

**Example**:
```bash
python scripts/download_model.py Qwen/Qwen3-0.6B
```

**What it does**: Downloads model to local cache for offline use

---

## Script Organization

```
cli/
├── expert_trainer.py          # Main trainer (called by Rust)
├── scripts/
│   ├── README.md              # This file
│   │
│   ├── # Training Pipeline (Python)
│   ├── pretokenize_datasets.py
│   ├── find_datasets.py
│   ├── benchmark_training.py
│   │
│   ├── # Dataset Management (Python)
│   ├── dataset_loader.py
│   ├── dataset_stats.py
│   ├── dataset_validator.py
│   │
│   ├── # Diagnostics (Python)
│   ├── diagnose_training.py
│   ├── check_cuda.py
│   ├── clear_cuda_cache.py
│   │
│   ├── # Utilities (Python)
│   ├── test_hf_dataset.py
│   ├── validate_grammar.py
│   ├── download_model.py
│   │
│   └── # Build & Setup (PowerShell)
│       ├── rebuild-force.ps1
│       ├── rebuild-quick.ps1
│       ├── rebuild-with-dlls.ps1
│       ├── copy-python-dlls.ps1
│       ├── setup_windows.ps1
│       ├── train_windows.ps1
│       └── train.ps1
```

---

## Dependencies

All scripts require dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `torch` - PyTorch for CUDA and model operations
- `transformers` - HuggingFace Transformers library
- `datasets` - HuggingFace Datasets library
- `peft` - Parameter-Efficient Fine-Tuning
- `bitsandbytes` - Quantization support

---

## Common Workflows

### 1. Before Training (First Time)

```bash
# Check CUDA
python scripts/check_cuda.py

# Find datasets
python scripts/find_datasets.py

# Pre-tokenize for speed
python scripts/pretokenize_datasets.py
```

### 2. Training Workflow

```bash
# Clear GPU cache
python scripts/clear_cuda_cache.py

# Train (in one terminal)
cd ../experts/expert-json
expert-cli train --manifest manifest.json

# Monitor (in another terminal)
cd ../../cli
python scripts/benchmark_training.py --duration 120
```

### 3. Troubleshooting

```bash
# Diagnose issues
python scripts/diagnose_training.py

# Check specific components
python scripts/check_cuda.py
python scripts/test_hf_dataset.py
```

---

## Notes

- **expert_trainer.py** stays in `cli/` root (called directly by Rust code)
- All utility scripts are in `cli/scripts/` for organization
- Scripts in `scripts/` can import each other (same directory)
- `expert_trainer.py` dynamically adds `scripts/` to Python path when importing these modules

---

Last Updated: November 3, 2025

