# Multi-Task Dataset System

The Expert System supports combining multiple task-specific datasets with weighted sampling, validation, and deduplication. This allows training experts on diverse task families with optimal data distribution.

## Overview

Instead of a single monolithic dataset, organize training data into task families:

```
datasets/
├── schema_generate/    # 40% weight - Generate JSON from schemas
├── json_repair/        # 20% weight - Fix broken JSON
├── text_to_json/       # 20% weight - Extract JSON from text
├── json_transform/     # 10% weight - Transform JSON structures
└── json_style_strict/  # 10% weight - Normalize JSON formatting
```

Each task has independent train/valid/test splits, combined during training with configurable weights.

## Dataset Organization

### Directory Structure

```
expert/experts/expert-json-parser/datasets/
├── schema_generate/
│   ├── train.jsonl      # LLM-generated examples
│   ├── valid.jsonl
│   └── test.jsonl
├── json_repair/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── text_to_json/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── json_transform/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── json_style_strict/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── meta/
    └── stats.json       # Generated statistics
```

### JSONL Format

**SFT (Supervised Fine-Tuning) Format:**
```json
{"task": "schema_generate", "input": {"instruction": "Generate JSON following this schema", "schema": "{...}"}, "output": "{...}"}
{"task": "json_repair", "input": {"instruction": "Fix this broken JSON", "broken_json": "{\"a\": 1,}"}, "output": "{\"a\": 1}"}
{"task": "text_to_json", "input": {"instruction": "Extract JSON from text", "text": "Ana has 27 years"}, "output": "{\"name\": \"Ana\", \"age\": 27}"}
```

**DPO (Direct Preference Optimization) Format:**
```json
{"task": "json_repair", "input": {"instruction": "Fix this JSON", "broken_json": "{\"a\": 1,}"}, "chosen": "{\"a\": 1}", "rejected": "{\"a\": 1,}"}
```

**Required Fields:**
- `task`: Task family name (string)
- `input`: Dictionary with:
  - `instruction`: Prompt text (string)
  - Context fields (schema, broken_json, text, json, etc.)
- `output`: Expected response (SFT) OR
- `chosen` + `rejected`: Preference pair (DPO)

## Manifest Configuration

### Multi-Task Format

```json
{
  "training": {
    "dataset": {
      "type": "multi_task",
      "tasks": {
        "schema_generate": {
          "train": "datasets/schema_generate/train.jsonl",
          "valid": "datasets/schema_generate/valid.jsonl",
          "test": "datasets/schema_generate/test.jsonl",
          "weight": 0.4,
          "format": "sft"
        },
        "json_repair": {
          "train": "datasets/json_repair/train.jsonl",
          "valid": "datasets/json_repair/valid.jsonl",
          "test": "datasets/json_repair/test.jsonl",
          "weight": 0.2,
          "format": "dpo"
        },
        "text_to_json": {
          "train": "datasets/text_to_json/train.jsonl",
          "valid": "datasets/text_to_json/valid.jsonl",
          "test": "datasets/text_to_json/test.jsonl",
          "weight": 0.2,
          "format": "sft"
        }
      },
      "validation": {
        "validate_json": true,
        "validate_schema": false,
        "deduplicate": true,
        "min_length": 10,
        "max_length": 2048
      }
    },
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

### Configuration Options

**Task Configuration:**
- `train/valid/test`: Paths to JSONL files (relative to expert directory)
- `weight`: Sampling weight (0.0-1.0+), controls task proportion in final dataset
- `format`: "sft" or "dpo" (informational, both work)

**Validation Configuration:**
- `validate_json`: Check if output is valid JSON (default: true)
- `validate_schema`: Validate against schema if present (default: false)
- `deduplicate`: Remove duplicate outputs by hash (default: true)
- `min_length`: Minimum output length in characters (default: 10)
- `max_length`: Maximum output length in characters (default: 2048)

### Weights Explained

Weights control how many examples from each task appear in the final dataset:

- `weight: 1.0` - Include 100% of examples
- `weight: 0.5` - Include 50% (random sample)
- `weight: 0.2` - Include 20%
- `weight: 2.0` - Include 200% (duplicate examples)

**Example Distribution:**
```
schema_generate: 10,000 examples × 0.4 weight = 4,000 in final dataset
json_repair:      5,000 examples × 0.2 weight = 1,000 in final dataset
text_to_json:     5,000 examples × 0.2 weight = 1,000 in final dataset
```

## Training Pipeline

### 1. File Validation

Script checks all task files exist before training:

```powershell
.\train_windows.ps1

Dataset type: Multi-Task

  Task: schema_generate (weight=0.4)
    train : 10000 examples
    valid : 500 examples
    test : 500 examples

  Task: json_repair (weight=0.2)
    train : 5000 examples
    valid : 250 examples
    test : 250 examples

  Total train examples (before weighting): 15000
```

### 2. Loading and Combining

Datasets are loaded, weighted, and combined:

```python
Loading multi-task dataset...

   Loading train split...
   Sampled 4000/10000 examples for 'schema_generate' (weight=0.4)
   Sampled 1000/5000 examples for 'json_repair' (weight=0.2)

   Raw counts: 5000 train, 750 valid
```

### 3. Validation

Each example is validated:

```python
   Validating train examples...
   Filtered out 25 invalid train examples
      Example 42: Invalid JSON in output
      Example 103: Output too long (2150 > 2048)
      Example 201: Missing 'task' field

   After validation: 4975 train, 748 valid
```

### 4. Deduplication

Duplicate outputs removed by hash:

```python
   Deduplicating train set...
   Removed 120 duplicate examples

   After deduplication: 4855 train
```

### 5. Statistics

Quality metrics displayed:

```
============================================================
Train Set Statistics
============================================================

Total Examples: 4855

By Task:
  schema_generate     :   3920 ( 80.7%)
  json_repair         :    935 ( 19.3%)

By Format:
  SFT       :   3920 ( 80.7%)
  DPO       :    935 ( 19.3%)

Output Length:
  Average: 156.3 chars
  Min:     12 chars
  Max:     1998 chars

Quality:
  Valid JSON: 99.8%
============================================================
```

### 6. Training

Combined dataset used for training:

```python
   Tokenizing datasets...
   Tokenizing train: 100%|██████████| 4855/4855

   Train examples: 4855
   Eval examples: 748
```

## Backward Compatibility

Old formats still work:

**Single File:**
```json
{
  "dataset": {
    "path": "datasets/local.jsonl"
  }
}
```

**HuggingFace:**
```json
{
  "dataset": {
    "path": "tatsu-lab/alpaca"
  }
}
```

**Multi-Task** (new):
```json
{
  "dataset": {
    "type": "multi_task",
    "tasks": {...}
  }
}
```

## Generating Datasets

Datasets are generated by premium LLMs (DeepSeek/Claude/GPT-4). See [LLM_GENERATION_GUIDE.md](LLM_GENERATION_GUIDE.md) for prompts and formats.

### Quality Checklist

Before training, ensure:

- ✅ All files exist (train/valid/test for each task)
- ✅ JSONL format (one JSON per line)
- ✅ Required fields present (task, input, output/chosen)
- ✅ Valid JSON in outputs
- ✅ Reasonable lengths (10-2048 chars)
- ✅ Diverse examples (no duplicates)
- ✅ Balanced distribution (check weights)

## Troubleshooting

### Error: Missing file for task

```
ERROR: Missing train file: datasets/schema_generate/train.jsonl
```

**Solution**: Generate the missing file using LLM or create empty file to skip task.

### Error: Filtered out many examples

```
Filtered out 500 invalid train examples
```

**Solution**: Check validation rules in manifest. Common issues:
- Invalid JSON in `output`
- Outputs too long/short
- Missing required fields

### Warning: Weight results in 0 samples

```
Warning: Weight 0.01 results in 0 samples for 'task_name'
```

**Solution**: Increase weight or add more examples to the task.

### Low Valid JSON rate

```
Valid JSON: 45.2%
```

**Solution**: Fix LLM generation prompts to produce valid JSON, or disable validation:
```json
{"validation": {"validate_json": false}}
```

## Best Practices

1. **Start Small**: Test with 100-1000 examples per task before scaling
2. **Balance Tasks**: Weights should roughly match importance
3. **Validate First**: Run validation on small sample before generating 100k examples
4. **Deduplicate**: Always enable deduplication to remove redundant training
5. **Monitor Stats**: Check distribution matches expectations
6. **Version Control**: Track dataset versions in git (use Git LFS for large files)
7. **Document Prompts**: Save LLM prompts used to generate each task

## Example: Complete Setup

### 1. Create Directory Structure

```bash
mkdir -p datasets/{schema_generate,json_repair,text_to_json}/{train,valid,test}
```

### 2. Generate with LLM

Use DeepSeek/Claude to generate each task (see LLM_GENERATION_GUIDE.md).

### 3. Update Manifest

```json
{
  "training": {
    "dataset": {
      "type": "multi_task",
      "tasks": {
        "schema_generate": {
          "train": "datasets/schema_generate/train.jsonl",
          "valid": "datasets/schema_generate/valid.jsonl",
          "test": "datasets/schema_generate/test.jsonl",
          "weight": 0.6
        },
        "json_repair": {
          "train": "datasets/json_repair/train.jsonl",
          "valid": "datasets/json_repair/valid.jsonl",
          "test": "datasets/json_repair/test.jsonl",
          "weight": 0.4
        }
      },
      "validation": {
        "validate_json": true,
        "deduplicate": true
      }
    }
  }
}
```

### 4. Train

```powershell
cd expert/cli
.\train_windows.ps1
```

Training will automatically:
- Load and combine tasks
- Apply weights
- Validate examples
- Deduplicate
- Show statistics
- Train the expert

## See Also

- [LLM_GENERATION_GUIDE.md](LLM_GENERATION_GUIDE.md) - How to generate datasets with premium LLMs
- [HUGGINGFACE_DATASETS.md](HUGGINGFACE_DATASETS.md) - Using public HuggingFace datasets
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training guide

