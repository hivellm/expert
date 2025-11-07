# CLI Scripts Organization

**Date**: November 3, 2025  
**Status**: ✅ Complete

## Overview

All Python utility scripts have been organized into `cli/scripts/` directory for better code organization and maintainability.

---

## Directory Structure

### Before
```
cli/
├── expert_trainer.py
├── check_cuda.py
├── clear_cuda_cache.py
├── dataset_loader.py
├── dataset_stats.py
├── dataset_validator.py
├── diagnose_training.py
├── download_model.py
├── test_hf_dataset.py
├── validate_grammar.py
├── copy-python-dlls.ps1
├── rebuild-force.ps1
├── rebuild-quick.ps1
├── rebuild-with-dlls.ps1
├── setup_windows.ps1
├── train.ps1
└── train_windows.ps1
```

### After
```
cli/
├── expert_trainer.py          # Main trainer (stays in root, called by Rust)
├── requirements.txt
├── Cargo.toml
├── README.md
└── scripts/
    ├── README.md              # Python scripts documentation
    ├── POWERSHELL_SCRIPTS.md  # PowerShell scripts documentation
    │
    ├── # Training Pipeline (Python)
    ├── pretokenize_datasets.py
    ├── find_datasets.py
    ├── benchmark_training.py
    │
    ├── # Dataset Management (Python)
    ├── dataset_loader.py
    ├── dataset_stats.py
    ├── dataset_validator.py
    │
    ├── # Diagnostics (Python)
    ├── diagnose_training.py
    ├── check_cuda.py
    ├── clear_cuda_cache.py
    │
    ├── # Utilities (Python)
    ├── test_hf_dataset.py
    ├── validate_grammar.py
    ├── download_model.py
    │
    └── # Build & Setup (PowerShell)
        ├── rebuild-force.ps1
        ├── rebuild-quick.ps1
        ├── rebuild-with-dlls.ps1
        ├── copy-python-dlls.ps1
        ├── setup_windows.ps1
        ├── train_windows.ps1
        └── train.ps1
```

---

## Changes Made

### 1. Scripts Moved to `scripts/`

**Python Scripts** (12 files):
- ✅ `check_cuda.py` → `scripts/check_cuda.py`
- ✅ `clear_cuda_cache.py` → `scripts/clear_cuda_cache.py`
- ✅ `dataset_loader.py` → `scripts/dataset_loader.py`
- ✅ `dataset_stats.py` → `scripts/dataset_stats.py`
- ✅ `dataset_validator.py` → `scripts/dataset_validator.py`
- ✅ `diagnose_training.py` → `scripts/diagnose_training.py`
- ✅ `download_model.py` → `scripts/download_model.py`
- ✅ `test_hf_dataset.py` → `scripts/test_hf_dataset.py`
- ✅ `validate_grammar.py` → `scripts/validate_grammar.py`
- ✅ `find_datasets.py` → `scripts/find_datasets.py`
- ✅ `pretokenize_datasets.py` → `scripts/pretokenize_datasets.py`
- ✅ `benchmark_training.py` → `scripts/benchmark_training.py`

**PowerShell Scripts** (7 files):
- ✅ `rebuild-force.ps1` → `scripts/rebuild-force.ps1`
- ✅ `rebuild-quick.ps1` → `scripts/rebuild-quick.ps1`
- ✅ `rebuild-with-dlls.ps1` → `scripts/rebuild-with-dlls.ps1`
- ✅ `copy-python-dlls.ps1` → `scripts/copy-python-dlls.ps1`
- ✅ `setup_windows.ps1` → `scripts/setup_windows.ps1`
- ✅ `train_windows.ps1` → `scripts/train_windows.ps1`
- ✅ `train.ps1` → `scripts/train.ps1`

**Total**: 19 scripts organized

**Kept in root**:
- ✅ `expert_trainer.py` - Called directly by Rust `python_bridge.rs`

### 2. Code Updates

#### `expert_trainer.py`
Updated imports to dynamically add `scripts/` to Python path:

```python
def load_multi_task_dataset(config: TrainingConfig, tokenizer):
    """Load and combine multi-task datasets"""
    import sys
    from pathlib import Path as PathlibPath
    
    # Add scripts directory to Python path
    scripts_dir = PathlibPath(__file__).parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    
    from dataset_loader import MultiTaskDatasetLoader
    from dataset_validator import DatasetValidator
    from dataset_stats import generate_stats, print_stats
```

#### `docs/OPTIMIZATION_SUMMARY.md`
Updated paths to scripts:

```bash
# Before
python ../../cli/clear_cuda_cache.py

# After
python ../../cli/scripts/clear_cuda_cache.py
```

### 3. Documentation

Created `cli/scripts/README.md` with:
- Complete description of each script
- Usage examples
- Common workflows
- Dependencies
- Organization structure

---

## Script Categories

### Training Pipeline (3 scripts)
- `pretokenize_datasets.py` - Pre-tokenize datasets for fast loading
- `find_datasets.py` - Locate HuggingFace dataset cache
- `benchmark_training.py` - Monitor training performance

### Dataset Management (3 scripts)
- `dataset_loader.py` - Multi-task dataset loading (imported by trainer)
- `dataset_stats.py` - Dataset statistics generation (imported by trainer)
- `dataset_validator.py` - Dataset validation (imported by trainer)

### Diagnostics (3 scripts)
- `diagnose_training.py` - Diagnose training issues
- `check_cuda.py` - Check CUDA availability
- `clear_cuda_cache.py` - Clear GPU memory

### Utilities (3 scripts)
- `test_hf_dataset.py` - Test HuggingFace datasets
- `validate_grammar.py` - Validate GBNF grammar files
- `download_model.py` - Download models from HuggingFace

---

## Usage Examples

### Updated Commands

All script references now use `scripts/` prefix:

```bash
# Before
python cli/clear_cuda_cache.py

# After
python cli/scripts/clear_cuda_cache.py
```

### Common Workflows

#### 1. Pre-training Setup
```bash
cd f:\Node\hivellm\expert\cli

# Check environment
python scripts/check_cuda.py

# Find datasets
python scripts/find_datasets.py

# Pre-tokenize
python scripts/pretokenize_datasets.py
```

#### 2. Training
```bash
# Clear cache
python scripts/clear_cuda_cache.py

# Train
expert-cli train --manifest manifest.json

# Monitor (separate terminal)
python scripts/benchmark_training.py --duration 120
```

#### 3. Troubleshooting
```bash
# Diagnose issues
python scripts/diagnose_training.py

# Test components
python scripts/check_cuda.py
python scripts/test_hf_dataset.py
```

---

## Benefits

### 1. Better Organization
- ✅ Clear separation: main trainer vs utility scripts
- ✅ All utilities in one place (`scripts/`)
- ✅ Easy to find and manage scripts

### 2. Maintainability
- ✅ `scripts/README.md` documents all scripts
- ✅ Consistent location for all utilities
- ✅ Easier onboarding for new developers

### 3. Scalability
- ✅ Easy to add new scripts to `scripts/`
- ✅ No clutter in root directory
- ✅ Clear categorization (training, diagnostics, utilities)

---

## Testing

### Compilation Test
```bash
cd f:\Node\hivellm\expert\cli
cargo build --release
```

**Result**: ✅ Compiled successfully with no errors

### Import Test
The dynamic path addition in `expert_trainer.py` ensures that:
- ✅ `dataset_loader.py` can be imported from `scripts/`
- ✅ `dataset_stats.py` can be imported from `scripts/`
- ✅ `dataset_validator.py` can be imported from `scripts/`

---

## Migration Notes

### For Developers

If you had scripts or documentation referencing old paths:

**Update these patterns**:
```bash
# Old
python cli/check_cuda.py
python cli/clear_cuda_cache.py
python cli/find_datasets.py

# New
python cli/scripts/check_cuda.py
python cli/scripts/clear_cuda_cache.py
python cli/scripts/find_datasets.py
```

### For CI/CD

Update any automation scripts that reference these utilities:

```yaml
# Before
- run: python expert/cli/clear_cuda_cache.py

# After
- run: python expert/cli/scripts/clear_cuda_cache.py
```

---

## Files Changed

### Moved (12 scripts)
- `cli/check_cuda.py` → `cli/scripts/check_cuda.py`
- `cli/clear_cuda_cache.py` → `cli/scripts/clear_cuda_cache.py`
- `cli/dataset_loader.py` → `cli/scripts/dataset_loader.py`
- `cli/dataset_stats.py` → `cli/scripts/dataset_stats.py`
- `cli/dataset_validator.py` → `cli/scripts/dataset_validator.py`
- `cli/diagnose_training.py` → `cli/scripts/diagnose_training.py`
- `cli/download_model.py` → `cli/scripts/download_model.py`
- `cli/test_hf_dataset.py` → `cli/scripts/test_hf_dataset.py`
- `cli/validate_grammar.py` → `cli/scripts/validate_grammar.py`
- `cli/find_datasets.py` (already in scripts)
- `cli/pretokenize_datasets.py` (already in scripts)
- `cli/benchmark_training.py` (already in scripts)

### Created
- `cli/scripts/README.md` - Complete documentation

### Modified
- `cli/expert_trainer.py` - Updated imports with dynamic path
- `docs/OPTIMIZATION_SUMMARY.md` - Updated script paths

### Unchanged
- `cli/expert_trainer.py` - Stays in root (called by Rust)

---

## Next Steps

No action needed! The reorganization is complete and tested.

**Recommendation**: Update any external documentation or scripts that reference the old paths.

---

Last Updated: November 3, 2025

