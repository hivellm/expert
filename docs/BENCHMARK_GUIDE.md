# Expert Benchmark Guide

This guide explains how to run benchmarks for experts to obtain `quality_metrics` for the manifest.

## Overview

The `quality_metrics` field in `manifest.json` provides standardized evaluation metrics that help users understand expert performance. This guide covers:

1. Setting up benchmark infrastructure
2. Creating test cases
3. Running benchmarks
4. Calculating metrics
5. Updating manifests

## Benchmark Structure

A benchmark consists of:

- **Test cases**: Domain-specific queries/tasks
- **Base model evaluation**: Baseline performance
- **Expert evaluation**: Expert performance
- **Comparison**: Calculate improvement metrics

## Test Case Format

Test cases should be stored in JSON format:

```json
{
  "test_suite": "expert-sql-benchmark",
  "version": "1.0.0",
  "test_cases": [
    {
      "id": "sql_001",
      "prompt": "Generate SQL query to find all customers with more than 5 orders",
      "schema": "CREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT);",
      "expected_keywords": ["SELECT", "JOIN", "COUNT", "GROUP BY", "HAVING"],
      "category": "aggregation",
      "difficulty": "medium"
    }
  ]
}
```

## Running Benchmarks

### Step 1: Prepare Test Cases

Create a test cases file (e.g., `tests/test_cases.json`) with domain-specific queries:

```python
# Example: tests/prepare_test_cases.py
import json

test_cases = [
    {
        "id": "test_001",
        "prompt": "Your test prompt here",
        "expected_keywords": ["keyword1", "keyword2"],
        "category": "basic"
    }
]

with open("tests/test_cases.json", "w") as f:
    json.dump({"test_cases": test_cases}, f, indent=2)
```

### Step 2: Evaluate Base Model

Run base model evaluation:

```python
# Example: tests/evaluate_base.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def evaluate_model(model, tokenizer, test_cases):
    results = []
    for case in test_cases:
        prompt = case["prompt"]
        output = generate_output(model, tokenizer, prompt)
        
        # Evaluate output (domain-specific)
        score = evaluate_output(output, case)
        results.append({
            "id": case["id"],
            "score": score,
            "output": output
        })
    return results

# Load base model
base_model, base_tokenizer = load_base_model()

# Load test cases
with open("tests/test_cases.json") as f:
    test_data = json.load(f)

# Evaluate
base_results = evaluate_model(base_model, base_tokenizer, test_data["test_cases"])

# Save results
with open("tests/base_results.json", "w") as f:
    json.dump(base_results, f, indent=2)
```

### Step 3: Evaluate Expert

Run expert evaluation (same process, but load expert adapter):

```python
# Example: tests/evaluate_expert.py
from peft import PeftModel

# Load expert
base_model, tokenizer = load_base_model()
expert_model = PeftModel.from_pretrained(base_model, "weights/qwen3-06b/adapter")

# Evaluate
expert_results = evaluate_model(expert_model, tokenizer, test_data["test_cases"])

# Save results
with open("tests/expert_results.json", "w") as f:
    json.dump(expert_results, f, indent=2)
```

### Step 4: Calculate Metrics

Calculate quality metrics:

```python
# Example: tests/calculate_metrics.py
import json

def calculate_metrics(base_results, expert_results):
    # Calculate benchmark scores (0-10 scale)
    base_scores = [r["score"] for r in base_results]
    expert_scores = [r["score"] for r in expert_results]
    
    base_score = sum(base_scores) / len(base_scores) if base_scores else 0.0
    expert_score = sum(expert_scores) / len(expert_scores) if expert_scores else 0.0
    
    # Calculate improvement
    improvement_percent = ((expert_score - base_score) / base_score * 100) if base_score > 0 else 0.0
    
    # Calculate win rate
    wins = sum(1 for e, b in zip(expert_scores, base_scores) if e > b)
    win_rate = wins / len(expert_scores) if expert_scores else 0.0
    
    return {
        "benchmark_score": expert_score,
        "base_model_score": base_score,
        "improvement_percent": improvement_percent,
        "win_rate_vs_base": win_rate,
        "test_queries": len(expert_results)
    }

# Load results
with open("tests/base_results.json") as f:
    base_results = json.load(f)

with open("tests/expert_results.json") as f:
    expert_results = json.load(f)

# Calculate
metrics = calculate_metrics(base_results, expert_results)

# Add metadata
metrics["checkpoint"] = "final"  # or "checkpoint-500", etc.
metrics["training_steps"] = 655  # from training logs
metrics["test_date"] = "2025-11-06"  # ISO 8601 format

# Save
with open("tests/quality_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

## Domain-Specific Evaluation

Different experts require different evaluation methods:

### SQL Expert

```python
def evaluate_sql_output(output, test_case):
    """Evaluate SQL query output"""
    score = 0.0
    
    # Check syntax validity
    if is_valid_sql(output):
        score += 3.0
    
    # Check expected keywords
    for keyword in test_case.get("expected_keywords", []):
        if keyword.lower() in output.lower():
            score += 1.0
    
    # Check semantic correctness (if schema provided)
    if test_case.get("schema"):
        if matches_schema(output, test_case["schema"]):
            score += 3.0
    
    # Check query structure
    if has_required_clauses(output, test_case):
        score += 3.0
    
    return min(score, 10.0)  # Cap at 10.0
```

### JSON Expert

```python
def evaluate_json_output(output, test_case):
    """Evaluate JSON output"""
    score = 0.0
    
    # Check JSON validity
    try:
        json.loads(output)
        score += 5.0
    except json.JSONDecodeError:
        return score
    
    # Check schema compliance (if schema provided)
    if test_case.get("schema"):
        if validates_against_schema(output, test_case["schema"]):
            score += 5.0
    
    return min(score, 10.0)
```

### Cypher Expert

```python
def evaluate_cypher_output(output, test_case):
    """Evaluate Cypher query output"""
    score = 0.0
    
    # Check syntax validity
    if is_valid_cypher(output):
        score += 4.0
    
    # Check expected patterns
    for pattern in test_case.get("expected_patterns", []):
        if pattern in output:
            score += 2.0
    
    # Check semantic correctness
    if matches_intent(output, test_case["prompt"]):
        score += 4.0
    
    return min(score, 10.0)
```

## Updating Manifest

After calculating metrics, update `manifest.json`:

```bash
# Use expert-cli or manually edit manifest.json
expert-cli manifest update --quality-metrics tests/quality_metrics.json
```

Or manually:

```json
{
  "quality_metrics": {
    "benchmark_score": 9.13,
    "base_model_score": 6.64,
    "improvement_percent": 37.5,
    "win_rate_vs_base": 0.85,
    "test_queries": 20,
    "checkpoint": "final",
    "training_steps": 655,
    "test_date": "2025-11-06",
    "_comment": "Qualitative analysis on 20 diverse queries. Strengths: MATCH patterns (10/10), aggregations (10/10). Weaknesses: AVG GROUP BY (4.2/10)."
  }
}
```

## Benchmark Report Template

Create a benchmark report documenting your evaluation:

```markdown
# Expert Benchmark Report

**Expert**: expert-sql  
**Version**: 0.3.0  
**Checkpoint**: checkpoint-500  
**Test Date**: 2025-11-06  
**Test Cases**: 15

## Results

- **Benchmark Score**: 8.8/10
- **Base Model Score**: 5.2/10
- **Improvement**: 69.2%
- **Win Rate**: 0.87 (13/15)

## Strengths

- ✅ JOIN operations: 10/10
- ✅ Aggregations: 10/10
- ✅ Subqueries: 9/10

## Weaknesses

- ⚠️ Recursive CTEs: 2/10
- ⚠️ UNION operations: 4/10
- ⚠️ Complex CASE WHEN: 6/10

## Test Cases

| ID | Category | Base Score | Expert Score | Improvement |
|----|----------|-----------|--------------|-------------|
| sql_001 | JOIN | 4.0 | 10.0 | +150% |
| sql_002 | Aggregation | 5.0 | 10.0 | +100% |
| sql_003 | Recursive CTE | 3.0 | 2.0 | -33% |
```

## Best Practices

1. **Use consistent test cases**: Reuse test cases across versions for comparison
2. **Document methodology**: Explain how scores are calculated
3. **Include edge cases**: Test known limitations
4. **Update regularly**: Re-run benchmarks after training improvements
5. **Compare checkpoints**: Evaluate multiple checkpoints to find best one
6. **Document failures**: Note which test cases fail and why

## Automation

Consider automating benchmark runs:

```bash
# scripts/run_benchmark.sh
#!/bin/bash

# Run base model evaluation
python tests/evaluate_base.py

# Run expert evaluation
python tests/evaluate_expert.py

# Calculate metrics
python tests/calculate_metrics.py

# Update manifest
expert-cli manifest update --quality-metrics tests/quality_metrics.json

# Generate report
python tests/generate_report.py > docs/BENCHMARK_REPORT.md
```

## See Also

- `experts/tests/template/` - Test template files
- `docs/EXPERT_FORMAT.md` - Manifest schema documentation
- `experts/expert-sql/tests/` - Example test suite

