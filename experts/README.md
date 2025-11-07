# HiveLLM Experts

> Guide for creating and testing experts in the HiveLLM Expert System

## Overview

This directory contains expert repositories for the Expert System. Each subdirectory is a complete expert repository that can be:

- Used directly from this monorepo
- Published as standalone Git repository
- Installed via `expert-cli install <git-url>`

**All operations are done via `expert-cli`** - no custom scripts per expert. Configuration is entirely in `manifest.json`.

---

## Standard Structure

Each expert repository follows this structure:

```
expert-<name>/
├── README.md                    # Expert documentation
├── manifest.json                # Metadata + training config (REQUIRED)
├── LICENSE                      # License file
├── .gitignore                   # Git ignore
├── .gitattributes              # Git LFS config
│
├── datasets/                    # Training datasets
│   ├── train.jsonl              # Training examples
│   └── synthetic_fixes.jsonl   # Additional synthetic data (optional)
│
├── weights/                     # Training outputs
│   ├── adapter/                 # Raw adapter weights
│   └── <name>.v<version>.expert # Packaged .expert file
│
└── tests/                       # Test suite
    ├── test_<name>.py          # Test implementation
    └── test_cases.json         # Test cases and expected outputs
```

**Note**: No `scripts/` directory! All commands via `expert-cli`.

---

## Creating a New Expert

### Step 1: Create Directory Structure

```bash
# Create expert directory
mkdir expert-<name>
cd expert-<name>

# Create required directories
mkdir -p datasets weights tests
```

### Step 2: Create manifest.json

The `manifest.json` file contains all configuration for your expert:

```json
{
  "name": "myexpert",
  "version": "1.0.0",
  "description": "My expert does X",
  "base_model": {
    "name": "Qwen3-0.6B",
    "quantization": "int4"
  },
  "capabilities": ["task:my-task"],
  "load_order": 6,
  "requires": [],
  "training": {
    "dataset": {
      "path": "datasets/train.jsonl",
      "format": "jsonl"
    },
    "config": {
      "method": "sft",
      "adapter_type": "lora",
      "rank": 16,
      "alpha": 16,
      "epochs": 3,
      "learning_rate": 0.0003,
      "batch_size": 4
    }
  }
}
```

**Key fields:**
- `name`: Unique identifier for your expert
- `version`: Semantic version (major.minor.patch)
- `load_order`: Loading priority (lower = loads first)
- `requires`: Array of expert names this depends on
- `capabilities`: Array of task identifiers this expert handles

### Step 3: Prepare Dataset

Create your training dataset in JSONL format (`datasets/train.jsonl`):

```jsonl
{"instruction": "Task description", "input": "Input example", "output": "Expected output"}
{"instruction": "Another task", "input": "Another input", "output": "Another output"}
```

**Dataset best practices:**
- Minimum 1,000 examples recommended
- Include diverse examples covering all use cases
- Validate JSONL format before training
- Consider adding synthetic data for edge cases

### Step 4: Create Test Cases

Create `tests/test_cases.json` with test scenarios:

```json
{
  "test_cases": [
    {
      "name": "basic_functionality",
      "input": "Example input",
      "expected_output": "Expected output",
      "description": "Tests basic functionality"
    },
    {
      "name": "edge_case",
      "input": "Edge case input",
      "expected_output": "Expected output",
      "description": "Tests edge case handling"
    }
  ]
}
```

### Step 5: Train the Expert

```bash
# Train the expert
expert-cli train \
  --manifest manifest.json \
  --dataset datasets/train.jsonl \
  --output weights

# Training will create weights/adapter/ with the trained model
```

**Training tips:**
- Monitor loss during training
- Save checkpoints at regular intervals
- Adjust learning rate if loss doesn't decrease
- Use validation split to prevent overfitting

### Step 6: Test the Expert

See [Testing an Expert](#testing-an-expert) section below for detailed testing instructions.

### Step 7: Package the Expert

```bash
# Package the trained adapter into .expert file
expert-cli package \
  --manifest manifest.json \
  --weights weights/adapter \
  --output weights/myexpert.v1.0.0.expert
```

### Step 8: Sign the Expert

```bash
# Sign the expert package for distribution
expert-cli sign --expert weights/myexpert.v1.0.0.expert
```

---

## Testing an Expert

Testing is critical to ensure your expert works correctly. Follow these steps:

### 1. Create Test Cases

Define test cases in `tests/test_cases.json`:

```json
{
  "test_cases": [
    {
      "name": "test_basic",
      "input": "Your test input",
      "expected_output": "Expected output",
      "description": "What this test validates",
      "category": "basic"
    },
    {
      "name": "test_edge_case",
      "input": "Edge case input",
      "expected_output": "Expected output",
      "description": "Edge case validation",
      "category": "edge"
    }
  ]
}
```

### 2. Implement Test Script

Create `tests/test_<name>.py`:

```python
import json
import sys
from pathlib import Path

def load_test_cases():
    """Load test cases from JSON file"""
    test_file = Path(__file__).parent / "test_cases.json"
    with open(test_file) as f:
        return json.load(f)["test_cases"]

def test_expert(expert_path, test_case):
    """Test a single test case against the expert"""
    # Load expert
    # Run inference with test_case["input"]
    # Compare output with test_case["expected_output"]
    # Return pass/fail
    pass

def run_all_tests(expert_path):
    """Run all test cases"""
    test_cases = load_test_cases()
    results = []
    
    for case in test_cases:
        result = test_expert(expert_path, case)
        results.append({
            "name": case["name"],
            "passed": result["passed"],
            "output": result["output"],
            "expected": case["expected_output"]
        })
    
    return results

if __name__ == "__main__":
    expert_path = sys.argv[1] if len(sys.argv) > 1 else "weights/adapter"
    results = run_all_tests(expert_path)
    
    # Print results
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Tests: {passed}/{total} passed")
    
    # Exit with error code if any test failed
    sys.exit(0 if passed == total else 1)
```

### 3. Run Tests During Development

```bash
# Test adapter during training (before packaging)
python tests/test_<name>.py weights/adapter

# Test packaged expert
python tests/test_<name>.py weights/<name>.v1.0.0.expert
```

### 4. Validate with expert-cli

```bash
# Validate expert structure and integrity
expert-cli validate --expert weights/adapter

# Validate packaged expert
expert-cli validate --expert weights/<name>.v1.0.0.expert
```

### 5. Test Categories

Organize tests by category:

- **Basic**: Core functionality tests
- **Edge**: Edge cases and boundary conditions
- **Error**: Error handling and invalid inputs
- **Performance**: Response time and resource usage
- **Integration**: Works with other experts (if applicable)

### 6. Continuous Testing

Run tests at key points:

```bash
# After each training checkpoint
python tests/test_<name>.py weights/adapter

# Before packaging
expert-cli validate --expert weights/adapter
python tests/test_<name>.py weights/adapter

# After packaging
expert-cli validate --expert weights/<name>.v1.0.0.expert
python tests/test_<name>.py weights/<name>.v1.0.0.expert
```

### 7. Test Coverage Goals

- **Minimum**: 80% of use cases covered
- **Recommended**: 95%+ coverage
- **Critical**: All edge cases and error conditions tested

---

## Training Workflow

### Complete Training Cycle

```bash
cd expert-<name>

# 1. Prepare dataset
# Edit datasets/train.jsonl

# 2. Train
expert-cli train \
  --manifest manifest.json \
  --dataset datasets/train.jsonl \
  --output weights

# 3. Test adapter
python tests/test_<name>.py weights/adapter

# 4. Validate
expert-cli validate --expert weights/adapter

# 5. Package
expert-cli package \
  --manifest manifest.json \
  --weights weights/adapter \
  --output weights/<name>.v<version>.expert

# 6. Test packaged expert
python tests/test_<name>.py weights/<name>.v<version>.expert

# 7. Sign
expert-cli sign --expert weights/<name>.v<version>.expert

# 8. Ready for distribution!
```

### Training Best Practices

1. **Start small**: Begin with a small dataset to validate the approach
2. **Iterate**: Add more examples based on failure cases
3. **Monitor**: Watch training loss and validation metrics
4. **Checkpoint**: Save checkpoints regularly to compare results
5. **Test early**: Test during training, not just at the end
6. **Document**: Record training parameters and results

---

## Publishing to Git

### Initial Setup

```bash
cd expert-<name>

# Initialize Git
git init
git add .
git commit -m "Initial release v1.0.0"

# Tag version
git tag -a v1.0.0 -m "Release v1.0.0"

# Push to remote
git remote add origin https://github.com/hivellm/expert-<name>.git
git push -u origin main v1.0.0
```

### Updating Releases

```bash
# Make changes and test
# Update version in manifest.json

# Commit changes
git add .
git commit -m "Release v1.1.0"

# Tag new version
git tag -a v1.1.0 -m "Release v1.1.0"

# Push
git push origin main v1.1.0
```

---

## Best Practices

1. **All config in manifest.json**: Training params, dataset generation, everything
2. **No custom scripts**: Use `expert-cli` commands only
3. **Test thoroughly**: Include comprehensive test cases in `tests/`
4. **Version properly**: Semantic versioning (major.minor.patch)
5. **Document well**: Clear README with examples and limitations
6. **Sign releases**: Build trust with cryptographic signatures
7. **Set load_order**: Ensure correct loading sequence
8. **Declare dependencies**: Use `requires` field in manifest
9. **Test before packaging**: Always validate adapter before creating .expert
10. **Iterate on failures**: Use test failures to improve dataset

---

## See Also

- [Expert System Documentation](../README.md)
- [CLI Documentation](../docs/CLI.md)
- [Expert Format Specification](../docs/EXPERT_FORMAT.md)
- [Git Distribution Guide](../docs/GIT_DISTRIBUTION.md)
- [Quick Start Guide](../QUICKSTART.md)
