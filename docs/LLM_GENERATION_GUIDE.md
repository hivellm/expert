# LLM Generation Guide

Guide for generating high-quality datasets using premium LLMs (DeepSeek, Claude, GPT-4) for the Multi-Task Dataset System.

## Overview

Instead of manually creating datasets, use premium LLMs to generate thousands of examples across task families. This approach:

- ✅ Scales to 100k+ examples
- ✅ Ensures diversity and quality
- ✅ Covers edge cases systematically
- ✅ Reduces human effort by 99%

## Task Families for JSON Parser Expert

### 1. Schema Generate (40% weight)

**Goal**: Generate valid JSON that follows a given schema.

**Prompt Template:**

```
You are generating training data for a JSON expert model.

Task: Generate valid JSON examples that strictly follow JSON schemas.

For each schema provided, generate 5 diverse, valid JSON examples.

Output format (JSONL - one JSON per line):
{"task": "schema_generate", "input": {"instruction": "Generate JSON following this schema. Respond only with JSON.", "schema": "<SCHEMA>"}, "output": "<VALID_JSON>"}

Requirements:
- Output must be valid JSON
- Output must pass schema validation
- Vary optional fields (present/absent)
- Vary enum values
- Vary array lengths (empty, 1, many)
- Vary nesting depth
- Minify output (no extra whitespace)
- Use consistent formatting (sorted keys)

Example schemas to use:
1. package.json (npm package)
2. tsconfig.json (TypeScript config)
3. .eslintrc.json (ESLint config)
4. OpenAPI 3.0 (API spec fragments)
5. JSON Schema meta-schema

Generate 50 examples total (10 per schema type).
```

**Example Output:**

```jsonl
{"task": "schema_generate", "input": {"instruction": "Generate JSON following this schema. Respond only with JSON.", "schema": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"version\":{\"type\":\"string\"}}}"}, "output": "{\"name\":\"my-package\",\"version\":\"1.0.0\"}"}
{"task": "schema_generate", "input": {"instruction": "Generate JSON following this schema. Respond only with JSON.", "schema": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"version\":{\"type\":\"string\"},\"dependencies\":{\"type\":\"object\"}}}"}, "output": "{\"dependencies\":{},\"name\":\"example\",\"version\":\"2.1.0\"}"}
```

### 2. JSON Repair (20% weight)

**Goal**: Fix broken JSON (for SFT) or create preference pairs (for DPO).

**Prompt Template (SFT):**

```
Generate training data for fixing broken JSON.

Task: Create pairs of broken JSON → fixed JSON.

Output format (JSONL):
{"task": "json_repair", "input": {"instruction": "Fix this broken JSON. Respond only with valid JSON.", "broken_json": "<BROKEN>"}, "output": "<FIXED>"}

Common errors to inject:
- Trailing comma: {"a": 1, "b": 2,}
- Missing comma: {"a": 1 "b": 2}
- Missing quote: {"a": value}
- Wrong bracket: ["a": 1]
- Malformed escape: {"a": "text\"bad}
- Missing closing bracket: {"a": [1, 2}

Difficulty levels:
- Easy: 1 error
- Medium: 2-3 errors
- Hard: Multiple errors + unicode + deep nesting

Generate 40 examples (20 easy, 15 medium, 5 hard).
```

**Prompt Template (DPO):**

```
Generate DPO pairs for JSON repair.

Output format (JSONL):
{"task": "json_repair", "input": {"instruction": "Fix this broken JSON. Respond only with valid JSON.", "broken_json": "<BROKEN>"}, "chosen": "<CORRECT_FIX>", "rejected": "<INCORRECT_FIX_OR_ORIGINAL>"}

For "rejected", use:
- Original broken JSON (model should not just echo input)
- Partially fixed (some errors remain)
- Over-fixed (changed data unnecessarily)

Generate 40 preference pairs.
```

**Example Output:**

```jsonl
{"task": "json_repair", "input": {"instruction": "Fix this broken JSON. Respond only with valid JSON.", "broken_json": "{\"name\": \"Ana\", \"age\": 27,}"}, "output": "{\"age\":27,\"name\":\"Ana\"}"}
{"task": "json_repair", "input": {"instruction": "Fix this broken JSON. Respond only with valid JSON.", "broken_json": "{\"a\": 1 \"b\": 2}"}, "output": "{\"a\":1,\"b\":2}"}
```

### 3. Text to JSON (20% weight)

**Goal**: Extract structured JSON from natural language text.

**Prompt Template:**

```
Generate training data for extracting JSON from text.

Task: Convert natural language descriptions into structured JSON.

Output format (JSONL):
{"task": "text_to_json", "input": {"instruction": "Extract {fields} from the text. Respond only with JSON.", "text": "<NATURAL_TEXT>"}, "output": "<JSON>"}

Domains:
1. Person data: name, age, tags/interests
2. Product data: name, price, category
3. Event data: type, timestamp, severity
4. Location data: city, country, coordinates
5. Log entries: level, message, timestamp

Variations:
- Different word order
- Numbers as words ("twenty-seven" vs 27)
- Dates in various formats
- Missing fields (use null or omit)
- Extra noise/irrelevant text

Generate 50 examples (10 per domain).
```

**Example Output:**

```jsonl
{"task": "text_to_json", "input": {"instruction": "Extract {name:string, age:int, tags:string[]} from the text. Respond only with JSON.", "text": "Ana tem 27 anos e gosta de Neo4j e JSON."}, "output": "{\"age\":27,\"name\":\"Ana\",\"tags\":[\"Neo4j\",\"JSON\"]}"}
{"task": "text_to_json", "input": {"instruction": "Extract {product:string, price:float, currency:string} from text. Respond only with JSON.", "text": "Laptop Dell costs $1299.99 USD"}, "output": "{\"currency\":\"USD\",\"price\":1299.99,\"product\":\"Laptop Dell\"}"}
```

### 4. JSON Transform (10% weight)

**Goal**: Transform JSON structure according to rules.

**Prompt Template:**

```
Generate training data for JSON transformations.

Task: Apply transformation rules to JSON objects.

Output format (JSONL):
{"task": "json_transform", "input": {"instruction": "Transform the JSON: <RULES>. Respond only with JSON.", "json": "<INPUT_JSON>"}, "output": "<TRANSFORMED_JSON>"}

Transformation types:
- Rename keys: "name" → "fullName"
- Filter fields: Only keep specific keys
- Type casting: "123" → 123
- Case normalization: "FirstName" → "firstName"
- Flatten nested: {"user": {"name": "x"}} → {"userName": "x"}
- Group by property

Generate 30 examples (5 per transformation type).
```

**Example Output:**

```jsonl
{"task": "json_transform", "input": {"instruction": "Transform: rename 'name' to 'fullName'. Respond only with JSON.", "json": "{\"name\": \"Ana\", \"age\": 27}"}, "output": "{\"age\":27,\"fullName\":\"Ana\"}"}
{"task": "json_transform", "input": {"instruction": "Transform: keep only 'name' and 'age' fields. Respond only with JSON.", "json": "{\"name\": \"Bob\", \"age\": 30, \"city\": \"NYC\", \"country\": \"USA\"}"}, "output": "{\"age\":30,\"name\":\"Bob\"}"}
```

### 5. JSON Style/Strict (10% weight)

**Goal**: Normalize JSON formatting (minify, sort keys, consistent escapes).

**Prompt Template:**

```
Generate training data for JSON normalization.

Task: Standardize JSON formatting.

Output format (JSONL):
{"task": "json_style_strict", "input": {"instruction": "Normalize this JSON: minify and sort keys alphabetically. Respond only with JSON.", "json": "<MESSY_JSON>"}, "output": "<NORMALIZED_JSON>"}

Input variations:
- Pretty-printed with indentation
- Random key order
- Extra whitespace
- Mixed quote styles (ensure valid JSON)
- Different number formats (1.0 vs 1)

Output requirements:
- Minified (no whitespace except in strings)
- Keys sorted alphabetically
- Consistent number format
- Proper escapes

Generate 30 examples.
```

**Example Output:**

```jsonl
{"task": "json_style_strict", "input": {"instruction": "Normalize: minify and sort keys. Respond only with JSON.", "json": "{\n  \"name\": \"Ana\",\n  \"age\": 27\n}"}, "output": "{\"age\":27,\"name\":\"Ana\"}"}
{"task": "json_style_strict", "input": {"instruction": "Normalize: minify and sort keys. Respond only with JSON.", "json": "{\"z\": 3, \"a\": 1, \"m\": 2}"}, "output": "{\"a\":1,\"m\":2,\"z\":3}"}
```

## Generation Workflow

### Step 1: Choose LLM

**Recommended:**
- **DeepSeek V3**: Best cost/performance ($0.27/M tokens)
- **Claude 3.5 Sonnet**: Highest quality
- **GPT-4o**: Good balance

### Step 2: Generate by Task

Generate each task separately:

```bash
# Schema Generate
deepseek chat --system "You generate training data" --prompt "$(cat prompts/schema_generate.txt)" > datasets/schema_generate/train.jsonl

# JSON Repair
deepseek chat --prompt "$(cat prompts/json_repair.txt)" > datasets/json_repair/train.jsonl

# Continue for other tasks...
```

### Step 3: Split Train/Valid/Test

```python
import random
import json

def split_dataset(input_file, train_ratio=0.9, valid_ratio=0.05):
    with open(input_file) as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    total = len(lines)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    
    train = lines[:train_size]
    valid = lines[train_size:train_size+valid_size]
    test = lines[train_size+valid_size:]
    
    # Write splits
    with open('train.jsonl', 'w') as f:
        f.writelines(train)
    with open('valid.jsonl', 'w') as f:
        f.writelines(valid)
    with open('test.jsonl', 'w') as f:
        f.writelines(test)
    
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

split_dataset('all_examples.jsonl')
```

### Step 4: Validate

Check generated data:

```python
import json

def validate_jsonl(file_path):
    errors = []
    valid = 0
    
    with open(file_path) as f:
        for i, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
                
                # Check required fields
                assert "task" in ex
                assert "input" in ex
                assert "output" in ex or ("chosen" in ex and "rejected" in ex)
                
                # Check output is valid JSON
                if "output" in ex:
                    json.loads(ex["output"])
                if "chosen" in ex:
                    json.loads(ex["chosen"])
                
                valid += 1
            except Exception as e:
                errors.append((i, str(e)))
    
    print(f"Valid: {valid}/{valid+len(errors)}")
    if errors:
        print("First 5 errors:")
        for line, error in errors[:5]:
            print(f"  Line {line}: {error}")
    
    return len(errors) == 0

validate_jsonl('datasets/schema_generate/train.jsonl')
```

## Quality Checklist

Before using generated datasets:

### Format Validation
- ✅ Valid JSONL (one JSON per line)
- ✅ No empty lines
- ✅ All required fields present
- ✅ Outputs are valid JSON

### Content Quality
- ✅ Diverse examples (no duplicates)
- ✅ Varying difficulty levels
- ✅ Edge cases covered
- ✅ Realistic data (not generic "example" everywhere)

### Distribution
- ✅ Train/Valid/Test splits balanced
- ✅ Task weights match importance
- ✅ Sufficient examples per task (min 100)

### Validation
```bash
# Check file exists
ls -lh datasets/*/train.jsonl

# Count examples
wc -l datasets/*/train.jsonl

# Validate format
python validate_datasets.py

# Check for duplicates
sort datasets/*/train.jsonl | uniq -d
```

## Cost Estimation

**Example:** 50,000 examples across 5 tasks

**DeepSeek V3** ($0.27/M input, $1.10/M output):
- Prompts: ~500 tokens × 10 batches = 5k tokens input
- Output: ~200 tokens × 50k examples = 10M tokens output
- Cost: $0.001 (input) + $11 (output) = **~$11 total**

**Claude 3.5 Sonnet** ($3/M input, $15/M output):
- Same calculation: **~$150 total**

**Recommendation**: Start with DeepSeek for bulk generation, use Claude for quality refinement.

## Prompt Engineering Tips

1. **Be Specific**: Define exact output format with examples
2. **Request Diversity**: "Vary X, Y, Z" explicitly
3. **Set Constraints**: Min/max lengths, character sets
4. **Batch Generation**: Generate 100s at once for consistency
5. **Validate Output**: Parse JSON in prompt to catch errors early

## Troubleshooting

### LLM generates invalid JSON

**Solution**: Add validation to prompt:
```
CRITICAL: Output must be valid JSON. Test with json.loads() before returning.
```

### Examples too similar

**Solution**: Request explicit variation:
```
Vary: field names, nesting depth, data types, edge cases (empty arrays, null values).
No repetition - each example must be unique.
```

### Wrong format

**Solution**: Show exact example in prompt:
```
Example of CORRECT output:
{"task": "...", "input": {...}, "output": "..."}

Example of WRONG output (do not use):
{instruction: "...", response: "..."}
```

## See Also

- [MULTI_TASK_DATASETS.md](MULTI_TASK_DATASETS.md) - Dataset organization and training
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training guide

