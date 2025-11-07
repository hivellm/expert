# Dataset Format Configuration Guide

This document explains how to configure `field_mapping` in the manifest for different dataset formats.

## Overview

Different HuggingFace datasets use different column names for instruction, input/context, and response. The `field_mapping` in the manifest tells the trainer how to map these fields to the standard training format:

```
### Instruction:
{instruction_field}

### Schema/Input/Context:
{input_field}

### Response:
{response_field}
```

## Field Mapping Structure

In `manifest.json`:

```json
{
  "training": {
    "dataset": {
      "path": "huggingface/dataset-name",
      "format": "huggingface",
      "field_mapping": {
        "instruction": "question",    // Maps to dataset's question column
        "input": "schema",             // Maps to dataset's schema column (optional)
        "response": "cypher"           // Maps to dataset's cypher column
      }
    }
  }
}
```

## Expert-Specific Configurations

### 1. expert-neo4j

**Dataset**: `neo4j/text2cypher-2025v1`

**Columns**: `question`, `schema`, `cypher`

**Field Mapping**:
```json
{
  "field_mapping": {
    "instruction": "question",
    "input": "schema",
    "response": "cypher"
  }
}
```

**Training Format**:
```
### Instruction:
{question}

### Schema:
{schema}

### Response:
{cypher}
```

**Example**:
```
### Instruction:
Find all persons named John

### Schema:
Node properties:
- **Person**
  - `name`: STRING

### Response:
MATCH (p:Person {name: 'John'}) RETURN p
```

---

### 2. expert-sql

**Dataset**: `b-mc2/sql-create-context`

**Columns**: `question`, `context`, `answer`

**Field Mapping**:
```json
{
  "field_mapping": {
    "instruction": "question",
    "input": "context",
    "response": "answer"
  }
}
```

**Training Format**:
```
### Instruction:
{question}

### Schema:
{context}

### Response:
{answer}
```

**Note**: The `context` field contains `CREATE TABLE` statements, so it's labeled as "Schema" in the training prompt (auto-detected by presence of "CREATE TABLE").

**Example**:
```
### Instruction:
How many heads of the departments are older than 56?

### Schema:
CREATE TABLE head (age INTEGER)

### Response:
SELECT COUNT(*) FROM head WHERE age > 56
```

---

### 3. expert-typescript

**Dataset**: `mhhmm/typescript-instruct-20k-v2c`

**Columns**: `instruction`, `output`

**Field Mapping**:
```json
{
  "field_mapping": {
    "instruction": "instruction",
    "response": "output"
  }
}
```

**Training Format**:
```
### Instruction:
{instruction}

### Response:
{output}
```

**Note**: No `input` field - TypeScript dataset only has instruction and code output.

**Example**:
```
### Instruction:
Generate code that sets up aliases for different directories using the path module

### Response:
```typescript
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@localtypes": path.resolve(__dirname, "src/core/types"),
    }
  }
});
```
```

---

### 4. expert-json (Multi-task)

**Dataset**: Local JSONL files (synthetic/generated)

**Type**: `multi_task`

**No field mapping needed** - Local datasets are already in the correct format:
```json
{"instruction": "...", "input": "...", "response": "..."}
```

---

## Auto-Detection Fallbacks

If `field_mapping` is not specified, the trainer attempts auto-detection:

**Instruction field** (tries in order):
1. `instruction`
2. `prompt`
3. `question`
4. `text`
5. `sql_prompt`

**Input/Context field** (tries in order):
1. `input`
2. `context`
3. `schema`

**Response field** (tries in order):
1. `response`
2. `output`
3. `answer`
4. `completion`
5. `text`
6. `cypher`
7. `sql`

## Context Label Detection

The trainer automatically determines the label for the input field:

- If content contains `CREATE TABLE` → **"Schema"** label
- If field name contains `schema` → **"Schema"** label
- Otherwise → **"Input"** label

## Best Practices

1. **Always specify `field_mapping`** - Don't rely on auto-detection
2. **Test format before training** - Use a quick validation script
3. **Document in manifest** - Add `_comment` to explain dataset structure
4. **Verify after training** - Test with same format used in training

## Adding New Datasets

When adding a new dataset:

1. **Inspect the dataset**:
   ```python
   from datasets import load_dataset
   ds = load_dataset('org/dataset-name', split='train')
   print('Columns:', ds.column_names)
   print('Example:', ds[0])
   ```

2. **Determine field mapping**:
   - What field contains the question/instruction?
   - What field contains the context/schema/input?
   - What field contains the answer/response?

3. **Add to manifest**:
   ```json
   {
     "dataset": {
       "path": "org/dataset-name",
       "format": "huggingface",
       "field_mapping": {
         "instruction": "your_instruction_field",
         "input": "your_context_field",
         "response": "your_response_field"
       }
     }
   }
   ```

4. **Test with a few examples** before full training

## Common Dataset Formats

| Dataset | Instruction | Input/Context | Response |
|---------|-------------|---------------|----------|
| Alpaca-style | `instruction` | `input` | `output` |
| OpenAI-style | `prompt` | - | `completion` |
| Text2SQL | `question` | `context` | `answer` |
| Text2Cypher | `question` | `schema` | `cypher` |
| Code-style | `instruction` | - | `output` |

## Troubleshooting

### Issue: Model generates wrong format

**Solution**: Check if test prompt matches training format exactly.

### Issue: Auto-detection fails

**Solution**: Add explicit `field_mapping` to manifest.

### Issue: Context not included in training

**Solution**: Verify `input` field in mapping points to correct column.

### Issue: Special tokens in output

**Solution**: May need to adjust tokenizer or response extraction logic.

---

Last Updated: 2025-11-03

