# Python Tests

This directory contains Python unit tests for the training modules.

## Structure

- `test_config.py` - Tests for configuration loading and parsing
- `test_model_loader.py` - Tests for model loading functions
- `test_dataset_loader.py` - Tests for dataset loading functions
- `test_adapter_setup.py` - Tests for adapter setup (LoRA, DoRA, IA³)
- `test_callbacks.py` - Tests for training callbacks
- `test_progress_testing.py` - Tests for progress testing system
- `test_trainer.py` - Integration tests for trainer module

## Running Tests

### Run all tests
```bash
cd cli
python -m pytest tests_python/ -v
```

### Run specific test file
```bash
python -m pytest tests_python/test_config.py -v
```

### Run with coverage
```bash
python -m pytest tests_python/ --cov=train --cov-report=html
```

## Requirements

Tests require:
- pytest
- pytest-cov (for coverage)
- torch (mocked in most tests)
- transformers (mocked in most tests)

Install with:
```bash
pip install pytest pytest-cov
```

