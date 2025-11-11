# Expert Creation Guide

> Complete guide for creating, training, testing, and packaging new experts in the HiveLLM Expert System

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Creation Process](#step-by-step-creation-process)
4. [Manifest Configuration](#manifest-configuration)
5. [Dataset Creation](#dataset-creation)
6. [Training Configuration](#training-configuration)
7. [Testing](#testing)
8. [Packaging and Distribution](#packaging-and-distribution)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

An expert is a specialized adapter trained on a specific domain or task. Experts are:

- **Modular**: Each expert handles a specific capability
- **Composable**: Multiple experts can work together
- **Versioned**: Semantic versioning for compatibility
- **Tested**: Comprehensive test suites ensure quality
- **Documented**: Clear capabilities and limitations

### Expert Types

- **Format Experts**: Handle specific data formats (JSON, XML, YAML)
- **Language Experts**: Handle natural languages (English, Spanish, etc.)
- **Technology Experts**: Handle specific technologies (SQL, Cypher, TypeScript)
- **Task Experts**: Handle specific tasks (classification, extraction, etc.)

---

## Prerequisites

Before creating a new expert, ensure you have:

1. **expert-cli installed and configured**
   ```bash
   # Verify installation
   expert-cli --version
   ```

2. **Base model available**
   - Qwen3-0.6B (recommended for most experts)
   - Path configured in manifest

3. **Python environment**
   - Python 3.10+
   - Required packages: `transformers`, `datasets`, `peft`, `torch`

4. **Understanding of the domain**
   - Clear definition of what the expert should do
   - Examples of inputs and expected outputs
   - Known limitations and edge cases

---

## Step-by-Step Creation Process

### Step 1: Create Directory Structure

```bash
# Navigate to experts directory
cd experts

# Create expert directory
mkdir expert-<name>
cd expert-<name>

# Create required directories
mkdir -p datasets/raw datasets/processed weights tests scripts docs

# Create initial files
touch README.md manifest.json LICENSE preprocess.py .gitignore
```

**Directory Structure:**
```
expert-<name>/
├── README.md              # Expert documentation
├── manifest.json          # Expert configuration (REQUIRED)
├── LICENSE                # License file
├── preprocess.py          # Dataset preprocessing script
├── grammar.gbnf           # Grammar file (if needed)
├── .gitignore            # Git ignore rules
│
├── datasets/              # Training datasets
│   ├── raw/              # Raw collected data
│   ├── processed/        # Processed/validated data
│   └── train.jsonl       # Final training dataset
│
├── weights/               # Training outputs
│   └── <model-name>/     # Model-specific weights
│
├── tests/                 # Test suite
│   ├── test_expert.py    # Main test script
│   ├── test_hard.py      # Hard/edge case tests
│   ├── test_comparison.py # Base model comparison
│   └── test_cases.json   # Test cases definition
│
└── scripts/               # Utility scripts (optional)
    └── collect_data.py   # Data collection scripts
```

---

## Manifest Configuration

The `manifest.json` is the **core configuration file** for your expert. It defines everything about the expert: capabilities, training, constraints, and metadata.

### Complete Manifest Template

```json
{
  "name": "expert-<name>",
  "version": "0.1.0",
  "schema_version": "2.0",
  "description": "Clear description of what this expert does and its purpose",
  "author": "hivellm",
  "homepage": "https://github.com/hivellm/expert-<name>",
  
  "base_models": [
    {
      "name": "F:/Node/hivellm/expert/models/Qwen3-0.6B",
      "sha256": "",
      "quantization": "int4",
      "rope_scaling": {
        "type": "ntk-by-parts",
        "factor": 8.0,
        "max_position_embeddings": 32768,
        "original_max_position_embeddings": 8192,
        "fine_grained": true,
        "_comment": "Qwen3-specific NTK-by-parts scaling"
      },
      "prompt_template": "chatml",  # Note: Uses Qwen3 native format (<|im_start|>/<|im_end|>)
      "adapters": []
    }
  ],
  
  "soft_prompts": [],
  
  "constraints": {
    "max_chain": 10,
    "load_order": 6,
    "incompatible_with": [],
    "requires": []
  },
  
  "capabilities": [
    "task:<task-name>",
    "feature:<feature-name>",
    "usecase:<usecase-name>",
    "language:<lang-code>"
  ],
  
  "limitations": [
    {
      "pattern": "limitation_id",
      "description": "Clear description of the limitation",
      "example": "Example input that triggers the limitation",
      "workaround": "How to work around this limitation"
    }
  ],
  
  "training": {
    "dataset": {
      "type": "single",
      "path": "datasets/train.jsonl",
      "validation_path": "datasets/validation.jsonl",
      "test_path": "datasets/test.jsonl",
      "field_mapping": {
        "instruction": "instruction",
        "input": "input",
        "response": "output"
      },
      "format": "jsonl"
    },
    "config": {
      "method": "sft",
      "adapter_type": "dora",
      "rank": 12,
      "alpha": 24,
      "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj"
      ],
      "epochs": 3,
      "learning_rate": 0.0001,
      "batch_size": 4,
      "gradient_accumulation_steps": 4,
      "warmup_steps": 100,
      "lr_scheduler": "cosine",
      "save_strategy": "steps",
      "save_steps": 500,
      "evaluation_strategy": "steps",
      "eval_steps": 500,
      "load_best_model_at_end": true,
      "metric_for_best_model": "eval_loss",
      "save_total_limit": 3,
      "logging_steps": 10,
      "max_seq_length": 2048,
      "fp16": true,
      "use_unsloth": true,
      "gradient_checkpointing": true
    },
    "alternative_checkpoints": []
  },
  
  "quality_metrics": {
    "benchmark_score": 0.0,
    "base_model_score": 0.0,
    "improvement_percent": 0.0,
    "win_rate_vs_base": 0.0,
    "test_queries": 0,
    "checkpoint": "",
    "training_steps": 0,
    "test_date": ""
  },
  
  "perf": {
    "latency_ms_overhead": 0,
    "vram_mb_overhead": 0,
    "supported_batch_sizes": [1, 4, 8]
  }
}
```

### Key Fields Explained

#### Basic Metadata

- **`name`**: Unique identifier (format: `expert-<name>`)
- **`version`**: Semantic version (start with `0.1.0` for initial release)
- **`schema_version`**: Always `"2.0"` for new experts
- **`description`**: Clear, concise description of expert's purpose

#### Base Models

- **`name`**: Path to base model directory
- **`quantization`**: `"int4"` (recommended) or `"int8"`
- **`prompt_template`**: `"chatml"` for Qwen3 models (uses Qwen3 native format `<|im_start|>/<|im_end|>`)
- **`adapters`**: Empty array initially (filled after training)

#### Constraints

- **`load_order`**: Priority for loading (lower = loads first)
  - Foundation experts: 1-3
  - Format experts: 4-5
  - Technology experts: 6-7
  - Task experts: 8-10
- **`requires`**: Array of expert names this depends on
- **`incompatible_with`**: Experts that cannot be loaded together

#### Capabilities

Use consistent prefixes:
- **`task:`** - Main task (e.g., `task:sql_generation`)
- **`feature:`** - Specific features (e.g., `feature:join_queries`)
- **`usecase:`** - Use cases (e.g., `usecase:ecommerce`)
- **`language:`** - Language codes (e.g., `language:en`)

#### Limitations

Document known limitations in structured format:
```json
{
  "pattern": "unique_id",
  "description": "What doesn't work",
  "example": "Example that fails",
  "workaround": "How to avoid or fix"
}
```

#### Training Configuration

- **`adapter_type`**: `"dora"` (recommended), `"lora"`, or `"ia3"`
- **`rank`**: 12-16 for DoRA, 16-32 for LoRA
- **`alpha`**: Usually `rank * 2`
- **`target_modules`**: Model layers to adapt
- **`use_unsloth`**: `true` for faster training (requires int4/int8)

---

## Dataset Creation

### Dataset Format

Training datasets use JSONL format (one JSON object per line) with a `text` field containing Qwen3-formatted conversations:

```jsonl
{"text": "<|im_start|>system\nDialect: sql\nSchema:\n...<|im_end|>\n<|im_start|>user\nQuestion...<|im_end|>\n<|im_start|>assistant\nAnswer...<|im_end|>\n"}
```

**Important**: All experts must use **Qwen3 native format** (`<|im_start|>/<|im_end|>`) instead of generic ChatML (`<|system|>/<|end|>`). This ensures compatibility with Qwen3 models and optimal training performance.

**Format Structure:**
```
<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
```

**Example:**
```jsonl
{"text": "<|im_start|>system\nDialect: sql\nSchema:\nusers(id, name, email)<|im_end|>\n<|im_start|>user\nFind all users<|im_end|>\n<|im_start|>assistant\nSELECT * FROM users;<|im_end|>\n"}
```

### Dataset Collection Strategies

#### 1. Manual Collection

Create examples manually for high-quality, domain-specific data:

```python
# scripts/collect_examples.py
examples = [
    {
        "instruction": "Generate SQL query",
        "input": "Schema: users(id, name, email)\nFind all users",
        "output": "SELECT * FROM users;"
    },
    # ... more examples
]

# Save to JSONL
import json
with open("datasets/raw/manual.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
```

#### 2. Documentation Extraction

Extract examples from documentation:

```python
# scripts/collect_documentation.py
import json
from pathlib import Path

def extract_examples_from_docs(doc_path):
    """Extract code examples from documentation"""
    # Parse documentation
    # Extract examples
    # Format as training examples
    pass
```

#### 3. Synthetic Generation

Generate synthetic examples using LLMs:

```python
# scripts/generate_synthetic.py
from openai import OpenAI

client = OpenAI()

def generate_examples(prompt, count=100):
    """Generate synthetic training examples"""
    examples = []
    for i in range(count):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        # Parse and format response
        examples.append(format_example(response))
    return examples
```

#### 4. Public Datasets

Use existing public datasets:

```python
# scripts/load_public_dataset.py
from datasets import load_dataset

dataset = load_dataset("dataset_name", split="train")
# Convert to JSONL format
# Filter and clean
# Save to datasets/raw/
```

### Dataset Preprocessing

Create `preprocess.py` to process raw data and format it using **Qwen3 native format**:

```python
#!/usr/bin/env python3
"""
Dataset preprocessing script for expert-<name>
Formats examples using Qwen3 native format (<|im_start|>/<|im_end|>)
"""

import json
import re
from pathlib import Path
from typing import List, Dict

def format_qwen3(system: str, user: str, assistant: str) -> str:
    """Format example with Qwen3 native format (<|im_start|>/<|im_end|>)"""
    # Qwen3 format: <|im_start|>role\ncontent<|im_end|>
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )

def load_raw_data(raw_dir: Path) -> List[Dict]:
    """Load all raw data files"""
    examples = []
    for jsonl_file in raw_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                examples.append(json.loads(line))
    return examples

def format_example(example: Dict) -> Dict:
    """Format example into Qwen3 format"""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()
    
    # Build system message
    system_content = f"Task: {example.get('task', 'general')}"
    if input_text:
        system_content += f"\nContext: {input_text}"
    
    # Format using Qwen3 format
    text = format_qwen3(system_content, instruction, output)
    
    return {"text": text}

def validate_example(example: Dict) -> bool:
    """Validate example structure and content"""
    required_fields = ["instruction", "output"]
    if not all(field in example for field in required_fields):
        return False
    
    # Additional validation
    if len(example["output"]) < 10:
        return False  # Too short
    
    return True

def deduplicate_examples(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate examples"""
    seen = set()
    unique = []
    for ex in examples:
        # Use instruction + output as deduplication key
        key = (ex["instruction"], ex.get("output", ""))
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique

def main():
    """Main preprocessing pipeline"""
    raw_dir = Path("datasets/raw")
    output_file = Path("datasets/train.jsonl")
    
    # Load raw data
    print("Loading raw data...")
    examples = load_raw_data(raw_dir)
    print(f"Loaded {len(examples)} raw examples")
    
    # Validate
    print("Validating examples...")
    examples = [ex for ex in examples if validate_example(ex)]
    print(f"Validated {len(examples)} examples")
    
    # Deduplicate
    print("Deduplicating...")
    examples = deduplicate_examples(examples)
    print(f"After deduplication: {len(examples)} examples")
    
    # Format examples using Qwen3 format
    print("Formatting examples with Qwen3 format...")
    formatted_examples = [format_example(ex) for ex in examples]
    
    # Split train/validation/test
    train_size = int(len(formatted_examples) * 0.8)
    val_size = int(len(formatted_examples) * 0.1)
    
    train = formatted_examples[:train_size]
    val = formatted_examples[train_size:train_size + val_size]
    test = formatted_examples[train_size + val_size:]
    
    # Save
    for split, data in [("train", train), ("validation", val), ("test", test)]:
        output_path = Path(f"datasets/{split}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} examples to {output_path}")

if __name__ == "__main__":
    main()
```

**Key Points:**
- Always use `format_qwen3()` function to ensure correct format
- Output must have a `text` field with Qwen3-formatted conversation
- Format: `<|im_start|>role\ncontent<|im_end|>` (no spaces after role name)
- **Qwen3 Hybrid Reasoning**: For Qwen3 models, use 75% reasoning + 25% direct outputs
  - Wrap 75% of examples in `<think>...</think>` blocks with brief reasoning statements
  - Keep 25% as direct query-only outputs
  - This maintains compatibility with Qwen3's hybrid reasoning capabilities
- Ensure compatibility: Functions should support both Qwen3 and legacy ChatML formats for backward compatibility

### Dataset Requirements

- **Minimum size**: 1,000 examples (recommended: 5,000+)
- **Quality over quantity**: Better to have fewer high-quality examples
- **Diversity**: Cover all use cases and edge cases
- **Validation**: All examples should be manually reviewed
- **Format consistency**: All examples must use Qwen3 native format (`<|im_start|>/<|im_end|>`)
- **Field requirement**: All examples must have a `text` field with Qwen3-formatted conversation
- **Qwen3 Hybrid Reasoning**: Use 75% reasoning + 25% direct outputs
  - 75% of examples should include `<think>...</think>` blocks with brief reasoning
  - 25% should be direct query-only outputs
  - This ensures compatibility with Qwen3's hybrid reasoning training approach

---

## Training Configuration

### Training Process

1. **Prepare dataset**:
   ```bash
   python preprocess.py
   ```

2. **Validate manifest**:
   ```bash
   expert-cli validate --manifest manifest.json
   ```

3. **Start training**:
   ```bash
   expert-cli train \
     --manifest manifest.json \
     --dataset datasets/train.jsonl \
     --output weights/
   ```

4. **Monitor training**:
   - Check logs for loss values
   - Monitor VRAM usage
   - Watch for overfitting (eval loss > train loss)

### Training Best Practices

- **Start with small dataset**: Test with 100-500 examples first
- **Use checkpoints**: Save every 500 steps to compare
- **Monitor validation**: Use validation split to detect overfitting
- **Adjust learning rate**: Lower if loss doesn't decrease
- **Use Unsloth**: Enable for 2x faster training (requires int4/int8)

### Hyperparameter Guidelines

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `adapter_type` | `dora` | Best balance of performance/size |
| `rank` | 12-16 | Higher = more capacity, larger size |
| `alpha` | `rank * 2` | Standard scaling |
| `learning_rate` | 0.0001 | Start here, adjust if needed |
| `batch_size` | 4-8 | Depends on VRAM |
| `epochs` | 3-5 | More for small datasets |
| `warmup_steps` | 100-500 | 10% of total steps |

---

## Testing

### Test Structure

Create comprehensive test suite in `tests/`:

```
tests/
├── test_expert.py       # Basic functionality tests
├── test_hard.py         # Edge cases and hard scenarios
├── test_comparison.py   # Compare with base model
└── test_cases.json      # Test case definitions
```

### Test Cases Format

`tests/test_cases.json`:

```json
{
  "test_cases": [
    {
      "id": "test_basic_1",
      "name": "Basic functionality test",
      "input": "Test input",
      "expected_output": "Expected output",
      "description": "Tests basic functionality",
      "category": "basic"
    },
    {
      "id": "test_edge_1",
      "name": "Edge case test",
      "input": "Edge case input",
      "expected_output": "Expected output",
      "description": "Tests edge case handling",
      "category": "edge"
    }
  ]
}
```

### Test Implementation

`tests/test_expert.py`:

```python
#!/usr/bin/env python3
"""
Basic functionality tests for expert-<name>
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_test_cases():
    """Load test cases from JSON file"""
    test_file = Path(__file__).parent / "test_cases.json"
    with open(test_file) as f:
        return json.load(f)["test_cases"]

def test_expert(expert_path, test_case):
    """Test a single test case"""
    # Load expert
    # Run inference
    # Compare output
    # Return result
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
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Tests: {passed}/{total} passed")
    
    sys.exit(0 if passed == total else 1)
```

### Running Tests

```bash
# Test during development
python tests/test_expert.py weights/adapter

# Test packaged expert
python tests/test_expert.py weights/expert-<name>.v0.1.0.expert

# Run all test suites
python tests/test_expert.py
python tests/test_hard.py
python tests/test_comparison.py
```

---

## Packaging and Distribution

### Package the Expert

After training and testing:

```bash
# Package adapter into .expert file
expert-cli package \
  --manifest manifest.json \
  --weights weights/qwen3-06b/checkpoint-500 \
  --output weights/expert-<name>-qwen3-0-6b.v0.1.0.expert
```

### Sign the Expert

```bash
# Sign for distribution
expert-cli sign --expert weights/expert-<name>-qwen3-0-6b.v0.1.0.expert
```

This creates a `.sha256` file for integrity verification.

### Update Manifest

After packaging, update `manifest.json` with:

1. **Adapter path**: Update `adapters[0].path` to checkpoint used
2. **Version**: Update version if releasing new version
3. **Quality metrics**: Add benchmark results
4. **Performance metrics**: Add latency/VRAM measurements

---

## Best Practices

### 1. Start Small

- Begin with 100-500 examples
- Test basic functionality
- Iterate and improve

### 2. Document Everything

- Clear README.md
- Document limitations
- Provide examples

### 3. Test Thoroughly

- Test all capabilities
- Test edge cases
- Compare with base model

### 4. Version Carefully

- Use semantic versioning
- Document changes in CHANGELOG.md
- Test before releasing

### 5. Monitor Quality

- Track benchmark scores
- Monitor regressions
- Update metrics regularly

### 6. Follow Conventions

- Use consistent naming
- Follow directory structure
- Use standard capabilities format

---

## Troubleshooting

### Common Issues

#### Training Fails

- **VRAM OOM**: Reduce batch_size or use gradient checkpointing
- **Loss not decreasing**: Lower learning rate or check dataset quality
- **Overfitting**: Add more data or reduce epochs

#### Tests Fail

- **Wrong output format**: Check prompt template
- **Grammar errors**: Add grammar.gbnf file
- **Inconsistent results**: Check temperature settings

#### Packaging Issues

- **Manifest validation fails**: Check all required fields
- **Weights not found**: Verify checkpoint path
- **Signature fails**: Check signing key configuration

### Getting Help

1. Check existing experts for examples
2. Review `docs/EXPERT_FORMAT.md`
3. Check `schemas/expert-manifest.schema.json`
4. Review test templates in `experts/tests/template/`

---

## Quick Reference

### Directory Structure
```
expert-<name>/
├── manifest.json          # Configuration
├── datasets/train.jsonl   # Training data
├── weights/               # Trained models
└── tests/                 # Test suite
```

### Essential Commands
```bash
# Validate manifest
expert-cli validate --manifest manifest.json

# Train expert
expert-cli train --manifest manifest.json

# Test expert
python tests/test_expert.py weights/adapter

# Package expert
expert-cli package --manifest manifest.json --weights weights/...

# Sign expert
expert-cli sign --expert weights/...expert
```

### Key Files
- `manifest.json` - Expert configuration
- `preprocess.py` - Dataset preprocessing
- `tests/test_cases.json` - Test definitions
- `README.md` - Documentation

---

## Next Steps

After creating your expert:

1. ✅ Train and validate
2. ✅ Run comprehensive tests
3. ✅ Package and sign
4. ✅ Update experts/README.md
5. ✅ Document capabilities and limitations
6. ✅ Share with community

---

**Last Updated**: 2025-11-08  
**Version**: 1.0.0

