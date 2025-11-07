# Dataset Expansion Scripts - Expert TypeScript

Scripts to expand the expert-typescript dataset from 20k to 100k+ examples.

## Available Scripts

### 1. extract_typescript_docs.py
Extracts examples from the official TypeScript Handbook.

```bash
# Install dependencies
pip install requests beautifulsoup4

# Run extraction
python scripts/extract_typescript_docs.py
```

**Output**: `datasets/typescript_docs.jsonl`

### 2. integrate_the_stack.py
Integrates TypeScript code from the bigcode/the-stack dataset.

```bash
# Install dependencies
pip install datasets

# Run integration (limit to 10k files for testing)
python scripts/integrate_the_stack.py --limit 10000

# Full integration (may take hours)
python scripts/integrate_the_stack.py --limit 100000
```

**Output**: `datasets/the_stack_typescript.jsonl`

### 3. merge_datasets.py
Combines multiple datasets into a single training file.

```bash
# Combine all datasets
python scripts/merge_datasets.py \
  --sources datasets/typescript_docs.jsonl \
            datasets/the_stack_typescript.jsonl \
            datasets/synthetic_fixes.jsonl \
  --output datasets/train.jsonl \
  --backup
```

## Recommended Workflow

### Phase 1: Official Documentation
```bash
# 1. Extract documentation
python scripts/extract_typescript_docs.py

# 2. Verify result
wc -l datasets/typescript_docs.jsonl
```

### Phase 2: The Stack
```bash
# 1. Integrate the-stack (start with small limit)
python scripts/integrate_the_stack.py --limit 5000

# 2. Verify quality
head -5 datasets/the_stack_typescript.jsonl

# 3. If satisfactory, increase limit
python scripts/integrate_the_stack.py --limit 50000
```

### Phase 3: Merge
```bash
# Combine all datasets
python scripts/merge_datasets.py \
  --sources datasets/typescript_docs.jsonl \
            datasets/the_stack_typescript.jsonl \
  --output datasets/train.jsonl \
  --backup
```

## Next Steps

1. ✅ Extract official documentation
2. ⏳ Integrate the-stack TypeScript
3. ⏳ Create synthetic_fixes.jsonl with manual examples
4. ⏳ Validate and train expanded model

## Notes

- **TypeScript Validation**: Scripts can use `tsc --noEmit` to validate syntax (optional, can be slow)
- **Deduplication**: merge_datasets.py automatically removes duplicates
- **Format**: All datasets must be in ChatML JSONL format
