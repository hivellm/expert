# Using HuggingFace Datasets

The Expert System supports using datasets directly from HuggingFace Hub, eliminating the need to generate synthetic data for every expert.

## Benefits

- ✅ High-quality public datasets
- ✅ No synthetic generation costs
- ✅ Community-vetted data
- ✅ Faster setup
- ✅ Reproducible training

## Configuration Formats

### 1. Basic HuggingFace Dataset

```json
{
  "training": {
    "dataset": {
      "path": "dataunitylab/json-schema-store"
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

### 2. Specify Split and Config

Use `::` separator to specify split and configuration:

```json
{
  "training": {
    "dataset": {
      "path": "tatsu-lab/alpaca::train"
    }
  }
}
```

With config name:

```json
{
  "training": {
    "dataset": {
      "path": "google/boolq::train::default"
    }
  }
}
```

### 3. Custom Field Mapping

For datasets with non-standard field names:

```json
{
  "training": {
    "dataset": {
      "path": "databricks/databricks-dolly-15k"
    },
    "field_mapping": {
      "instruction": "instruction",
      "input": "context",
      "response": "response"
    }
  }
}
```

### 4. Single Text Field

For datasets with just one text column (like language modeling):

```json
{
  "training": {
    "dataset": {
      "path": "wikitext::train::wikitext-2-raw-v1"
    },
    "text_field": "text"
  }
}
```

### 5. Local JSONL (Backward Compatible)

The old format still works:

```json
{
  "training": {
    "dataset": {
      "path": "datasets/json_8k.jsonl"
    }
  }
}
```

## Auto-Detection

The system automatically detects common field names:

**Instruction fields:**
- `instruction`
- `prompt`
- `question`
- `text`

**Response fields:**
- `response`
- `output`
- `answer`
- `completion`
- `text`

**Input fields:**
- `input`
- `context`

## Examples

### JSON Parsing Expert

```json
{
  "name": "json-parser",
  "training": {
    "dataset": {
      "path": "dataunitylab/json-schema-store::train"
    },
    "text_field": "schema",
    "config": {
      "epochs": 3,
      "learning_rate": 0.0003
    }
  }
}
```

### SQL Expert

```json
{
  "name": "sql-expert",
  "training": {
    "dataset": {
      "path": "b-mc2/sql-create-context"
    },
    "field_mapping": {
      "instruction": "question",
      "input": "context",
      "response": "answer"
    },
    "config": {
      "epochs": 3
    }
  }
}
```

### Code Generation Expert

```json
{
  "name": "code-expert",
  "training": {
    "dataset": {
      "path": "bigcode/the-stack-dedup::train::python"
    },
    "text_field": "content",
    "config": {
      "epochs": 2
    }
  }
}
```

### Instruction Following

```json
{
  "name": "instruct-expert",
  "training": {
    "dataset": {
      "path": "tatsu-lab/alpaca"
    },
    "config": {
      "epochs": 3
    }
  }
}
```

## Finding Datasets

Popular HuggingFace dataset hubs:

- **General Instruction**: `tatsu-lab/alpaca`, `yahma/alpaca-cleaned`
- **Code**: `bigcode/the-stack-dedup`, `codeparrot/github-code`
- **JSON/Data**: `dataunitylab/json-schema-store`
- **SQL**: `b-mc2/sql-create-context`, `Clinton/Text-to-sql-v1`
- **Math**: `competition_math`, `gsm8k`
- **Reasoning**: `openai/gsm8k`, `tau/commonsense_qa`

Search more at: https://huggingface.co/datasets

## Dataset Format Requirements

Your HuggingFace dataset should have one of these structures:

### Instruction-Response Format

```json
{
  "instruction": "Parse this JSON",
  "input": "{\"key\": \"value\"}",
  "response": "The JSON contains one key..."
}
```

### Single Text Format

```json
{
  "text": "Complete document or code here..."
}
```

### Custom Format

```json
{
  "question": "What is 2+2?",
  "context": "Math problem",
  "answer": "4"
}
```

Use `field_mapping` to map to standard fields.

## Training Process

1. **Dataset Download**: Automatically cached by HuggingFace
2. **Field Detection**: Auto-detects or uses custom mapping
3. **Tokenization**: Formats with instruction template
4. **Split**: 90% train, 10% validation
5. **Training**: Standard LoRA fine-tuning

## Best Practices

1. ✅ **Start with public datasets** - Proven quality
2. ✅ **Check dataset license** - Ensure commercial use allowed
3. ✅ **Verify field names** - Use dataset viewer on HF
4. ✅ **Test with small subset** - Add `::train[:1000]` for testing
5. ✅ **Monitor training** - Watch eval loss

## Combining Datasets

To use multiple datasets, create a local JSONL that combines them:

```python
from datasets import load_dataset, concatenate_datasets

ds1 = load_dataset("dataset1")
ds2 = load_dataset("dataset2")
combined = concatenate_datasets([ds1, ds2])
combined.to_json("datasets/combined.jsonl")
```

Then use local path in manifest.

## Troubleshooting

**Issue**: Field not found error

```
ValueError: Text field 'schema' not found in dataset
```

**Solution**: Check column names on HuggingFace dataset page, update `text_field` or `field_mapping`

---

**Issue**: Dataset too large / slow download

**Solution**: Use subset:
```json
{
  "path": "bigcode/the-stack-dedup::train[:10000]::python"
}
```

---

**Issue**: Format doesn't match instruction template

**Solution**: Use `text_field` for pre-formatted text or `field_mapping` for custom structures

## Migration from Synthetic Generation

**Before** (synthetic generation):
```json
{
  "dataset": {
    "path": "datasets/json_8k.jsonl",
    "generation": {
      "domain": "json-parsing",
      "task": "parse JSON",
      "count": 8000,
      "provider": "cursor"
    }
  }
}
```

**After** (HuggingFace):
```json
{
  "dataset": {
    "path": "dataunitylab/json-schema-store"
  }
}
```

Benefits:
- No generation cost
- Better quality
- Larger dataset
- Faster setup

