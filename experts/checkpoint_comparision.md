# Qualitative Checkpoint Comparison Template

This template provides a standardized structure for comparing training checkpoints across all experts.

## Objective

The script does **NOT evaluate quality automatically**. It only:
- Runs the same prompts on all checkpoints
- Displays formatted results in the terminal
- Allows qualitative analysis by an external LLM (like you in chat)

## How to Use

### 1. Copy the Template

Copy the template to your expert root directory:

```bash
cp expert/experts/compare_checkpoints_template.py expert/experts/expert-{name}/compare.py
```

### 2. Configure the Script

Edit the script and adjust:

```python
# Base model configuration
BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"

# Checkpoint directory (relative to expert)
CHECKPOINT_DIR = "weights/qwen3-06b"

# Generation configuration
GEN_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
}

# Define expert-specific test cases
test_cases = [
    {
        "id": "test_001",
        "category": "category",
        "system_prompt": "Task: task_name\nDialect: dialect",
        "user_prompt": "Your prompt here",
        "expected_type": "json"  # or "text", "kql", "eql", etc.
    },
    # Add more cases...
]
```

### 3. Define Test Cases

Create representative test cases that cover:
- Basic expert cases
- Intermediate cases
- Complex/edge cases
- Different supported task types

**Example for expert-elastic:**
- Query DSL (term, match, bool, range, aggregations)
- KQL (simple, boolean operators)
- EQL (event queries, sequences)
- Mappings (ECS, simple, complex)
- Pipelines (geoip, rename, grok)

### 4. Execute

**IMPORTANT:** Always use CLI's `venv_windows` (has CUDA):

```powershell
cd F:/Node/hivellm/expert/experts/expert-{name}
F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe compare.py
```

### 5. Analyze Results

The script will display:
- Each test with its prompts
- Base model output
- Each checkpoint output
- Final summary

**Manual Analysis (by LLM in chat):**
1. Compare outputs between checkpoints
2. Identify which checkpoint has best quality
3. Check if there's progress between checkpoints
4. Identify issues (syntax, structure, completeness)
5. Recommend which checkpoint to use for package generation

## Template Structure

```
compare_checkpoints_template.py
├── Configuration (BASE_MODEL_PATH, CHECKPOINT_DIR, GEN_CONFIG)
├── Test cases (test_cases)
└── Main functions:
    ├── detect_device() - Detects CUDA/CPU
    ├── find_checkpoints() - Finds checkpoints automatically
    ├── load_base_model() - Loads base model
    ├── load_checkpoints() - Loads all checkpoints
    ├── generate_output() - Generates output for a prompt
    └── main() - Runs all tests
```

## Example Output

```
====================================================================================================
QUALITATIVE CHECKPOINT COMPARISON - EXPERT ELASTIC
This script generates outputs for external LLM analysis
Does not evaluate quality automatically
====================================================================================================

Checkpoints found: [50, 100, 150]
Total tests: 14
Device: cuda

[1/3] Loading Base Model...
[OK] Base Model loaded (device: cuda)

[2/3] Loading 3 checkpoints...
  Loading checkpoint-50... [OK]
  Loading checkpoint-100... [OK]
  Loading checkpoint-150... [OK]

[3/3] Running 14 tests...
====================================================================================================

TEST 1/14: dsl_001
Category: query_dsl
Expected type: json
----------------------------------------------------------------------------------------------------

[SYSTEM PROMPT]
Task: query_dsl
Dialect: elasticsearch

[USER PROMPT]
Search for documents where status equals 'active'.

----------------------------------------------------------------------------------------------------

[BASE MODEL]
{"query": {"term": {"status": "active"}}}

[CHECKPOINT-50]
{"query": {"term": {"status": "active"}}}

[CHECKPOINT-100]
{"query": {"term": {"status": "active"}}}

[CHECKPOINT-150]
{"query": {"term": {"status": "active"}}}

====================================================================================================
...
```

## Checklist for Creating Scripts

- [ ] Copy template to `expert-{name}/compare.py` (root directory)
- [ ] Configure correct `BASE_MODEL_PATH`
- [ ] Configure correct `CHECKPOINT_DIR`
- [ ] Define `test_cases` with representative cases
- [ ] Test execution with CLI's venv_windows
- [ ] Verify outputs are displayed correctly
- [ ] Document in expert README how to run

## Important Notes

1. **Always use CLI's venv_windows** - Has CUDA and correct dependencies
2. **Don't try to evaluate automatically** - Leave analysis to external LLM
3. **Use representative test cases** - Cover different scenarios
4. **Save results to JSON** - For later analysis if needed
5. **Document in README** - How to run and interpret results

## References

- See `expert/experts/expert-elastic/compare.py` for complete example
- See `expert/AGENTS.md` for rules about using venv_windows
