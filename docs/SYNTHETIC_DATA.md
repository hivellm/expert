# Synthetic Data Generation

> Generate high-quality training datasets using premium LLMs instead of manual curation

## Overview

One of the Expert System's key innovations is leveraging premium LLMs (DeepSeek, Claude, GPT-4, Cursor Agent) to automatically generate training data for expert specialists. This eliminates the bottleneck of manual dataset curation and enables rapid expert development.

## Core Concept

**Traditional approach:**
```
Humans manually create examples → Train expert (weeks of work)
```

**Synthetic approach:**
```
Define task + constraints → LLM generates 10k examples (hours) → Train expert
```

**Benefits:**
- **Speed**: Generate 10k examples in minutes vs weeks
- **Quality**: Premium LLMs produce high-quality, diverse examples
- **Cost**: $5-50 to generate dataset vs $1000s for human labeling
- **Iteration**: Easy to regenerate with different constraints
- **Privacy**: No need for sensitive real-world data

---

## CLI Workflow

### Basic Generation

```bash
# Generate instruction-following dataset for SFT
expert-gen dataset create \
  --domain "json-parsing" \
  --task "parse and validate JSON documents" \
  --count 10000 \
  --provider "deepseek" \
  --output "datasets/json_10k.jsonl"
```

### Preference Pairs for RLHF/DPO

```bash
# Generate (prompt, chosen, rejected) tuples
expert-gen dataset create \
  --domain "neo4j-schema" \
  --task "classify Neo4j schema patterns" \
  --count 5000 \
  --format "preference_pairs" \
  --provider "claude" \
  --output "datasets/neo4j_rlhf.jsonl"
```

### Advanced Options

```bash
expert-gen dataset create \
  --domain "code-generation" \
  --task "generate Rust async functions" \
  --count 20000 \
  --provider "gpt-4o" \
  --temperature 0.8 \
  --format "instruction" \
  --difficulty "mixed" \
  --diversity-threshold 0.85 \
  --output "datasets/rust_async_20k.jsonl" \
  --seed 42 \
  --batch-size 100 \
  --max-workers 10
```

---

## Supported LLM Providers

### DeepSeek Chat (Recommended)

**Pros:**
- Extremely cost-effective ($0.14 per million tokens)
- High quality reasoning
- Good at following complex instructions
- Fast inference

**Cons:**
- API rate limits (can be worked around with batching)

**Best for:**
- Large datasets (10k-100k examples)
- Technical domains (code, parsing, formats)
- Budget-conscious projects

```bash
export DEEPSEEK_API_KEY="your-key"

expert-gen dataset create \
  --domain "sql-optimization" \
  --task "optimize PostgreSQL queries" \
  --count 15000 \
  --provider "deepseek"
```

### Claude 3.5 Sonnet

**Pros:**
- Excellent structured output
- Strong reasoning capabilities
- High creativity and diversity
- Long context (200k tokens)

**Cons:**
- More expensive (~$3 per million input tokens)
- Slower than DeepSeek

**Best for:**
- Complex reasoning tasks
- Creative domains (writing, storytelling)
- High-quality preference pairs

```bash
export ANTHROPIC_API_KEY="your-key"

expert-gen dataset create \
  --domain "medical-diagnosis" \
  --task "generate differential diagnoses from symptoms" \
  --count 5000 \
  --provider "claude" \
  --temperature 0.7
```

### GPT-4o / GPT-4 Turbo

**Pros:**
- Consistent, reliable output
- Good at following specific formats
- Wide domain knowledge

**Cons:**
- Expensive (~$2.50-5 per million tokens)
- Sometimes less creative than Claude

**Best for:**
- Format-strict tasks (JSON, XML, structured data)
- When consistency is critical
- Moderate dataset sizes (5k-10k)

```bash
export OPENAI_API_KEY="your-key"

expert-gen dataset create \
  --domain "json-validation" \
  --task "validate and repair JSON documents" \
  --count 8000 \
  --provider "gpt-4o"
```

### Cursor Agent Integration

**Pros:**
- Direct integration with development workflow
- Understands your codebase context
- Can generate domain-specific examples

**Cons:**
- Limited to Cursor IDE environment
- No batch API

**Best for:**
- Project-specific experts
- Small datasets (<1k) for rapid iteration
- Tasks related to your active codebase

```typescript
// Via Cursor Agent API (conceptual)
const dataset = await cursorAgent.generateDataset({
  domain: "project-specific-patterns",
  task: "refactor legacy code to new patterns",
  examples: 500,
  context: workspace.currentFiles
});
```

### Local Models (Privacy Mode)

**Pros:**
- Complete privacy (no data leaves your machine)
- No API costs
- Unlimited generation

**Cons:**
- Much slower (10-100x)
- Lower quality than premium models
- Requires local GPU

**Best for:**
- Sensitive domains (healthcare, legal)
- Offline development
- Experimentation

```bash
# Use local Llama 3.1 70B via Ollama
expert-gen dataset create \
  --domain "patient-records" \
  --task "anonymize medical records" \
  --count 2000 \
  --provider "local" \
  --model "llama3.1:70b"
```

---

## Generation Strategies

### 1. Supervised Fine-Tuning (SFT)

Generate instruction-response pairs for standard fine-tuning.

**Prompt template:**

```
You are a data generation expert. Generate 100 diverse examples for the following task:

Domain: {domain}
Task: {task}

Output format (JSONL):
{"instruction": "...", "input": "...", "output": "..."}

Requirements:
- High diversity in examples
- Cover edge cases
- Vary difficulty levels
- Realistic inputs
- Correct outputs

Generate examples now:
```

**Example output:**

```jsonl
{"instruction": "Parse JSON and extract name field", "input": "{\"name\": \"Alice\", \"age\": 30}", "output": "Alice"}
{"instruction": "Validate JSON syntax", "input": "{invalid json}", "output": "Error: Invalid JSON at position 1"}
{"instruction": "Extract nested field from JSON", "input": "{\"user\": {\"profile\": {\"email\": \"a@b.com\"}}}", "output": "a@b.com"}
```

**Quality filters:**

```python
def filter_sft_examples(examples):
    filtered = []
    
    for ex in examples:
        # Skip empty or malformed
        if not all(k in ex for k in ["instruction", "input", "output"]):
            continue
        
        # Skip too short
        if len(ex["output"]) < 3:
            continue
        
        # Skip duplicates (fuzzy)
        if is_duplicate(ex, filtered):
            continue
        
        filtered.append(ex)
    
    return filtered
```

### 2. Preference Pairs (RLHF/DPO)

Generate (prompt, chosen, rejected) tuples for preference learning.

**Two-step generation:**

Step 1: Generate multiple responses per prompt

```
Generate 3 different responses to this prompt, varying in quality:

Prompt: {prompt}

Output format:
1. Best response (correct, complete, well-formatted)
2. Mediocre response (partially correct, some issues)
3. Poor response (incorrect or incomplete)
```

Step 2: Convert to preference pairs

```python
def create_preference_pairs(prompt, responses):
    # Rank responses (can also ask LLM to rank them)
    best = responses[0]
    mediocre = responses[1]
    poor = responses[2]
    
    pairs = [
        {"prompt": prompt, "chosen": best, "rejected": mediocre},
        {"prompt": prompt, "chosen": best, "rejected": poor},
        {"prompt": prompt, "chosen": mediocre, "rejected": poor}
    ]
    
    return pairs
```

**Example:**

```jsonl
{"prompt": "Parse this JSON: {\"key\": \"value\"}", "chosen": "{\"key\": \"value\"}\nValid JSON with 1 key.", "rejected": "JSON is valid."}
{"prompt": "Parse this JSON: {invalid}", "chosen": "Error: Invalid JSON. Missing closing brace.", "rejected": "Error."}
```

### 3. Distillation

Use premium LLM as teacher to generate reasoning traces + answers.

**Prompt template:**

```
You are an expert in {domain}. For each task below, provide:
1. Your step-by-step reasoning (thinking process)
2. The final answer

This will be used to train a smaller model to mimic your reasoning.

Task: {task}
Input: {input}

Format:
<reasoning>
[Your detailed thought process]
</reasoning>

<answer>
[Final answer]
</answer>
```

**Example:**

```jsonl
{
  "input": "Optimize this SQL: SELECT * FROM users WHERE age > 18 AND age < 65",
  "reasoning": "1. SELECT * is inefficient, specify needed columns\n2. Age range can use BETWEEN\n3. Add index hint if available",
  "answer": "SELECT id, name, email FROM users WHERE age BETWEEN 19 AND 64"
}
```

**Training the expert:**

```python
# Train expert to output both reasoning and answer
# Then at inference, optionally include reasoning or just answer
```

### 4. Adversarial / Hard Negatives

Generate challenging examples to improve robustness.

**Prompt template:**

```
Generate 100 DIFFICULT examples for this task:

Domain: {domain}
Task: {task}

Make examples challenging by including:
- Edge cases
- Ambiguous inputs
- Malformed data
- Rare patterns
- Counter-intuitive scenarios

Examples should be hard but still have correct answers.
```

**Example (JSON parsing):**

```jsonl
{"input": "{\"key\": \"value with \\\"escaped quotes\\\"\"}", "output": "Escaped quotes handled correctly"}
{"input": "{\"unicode\": \"\\u0048\\u0065\\u006c\\u006c\\u006f\"}", "output": "Decoded: Hello"}
{"input": "{\"nested\": {\"deep\": {\"very\": {\"much\": \"so\"}}}}", "output": "4 levels of nesting"}
```

---

## Quality Control Pipeline

### 1. Schema Validation

Ensure generated data matches expected format:

```python
def validate_schema(examples, schema):
    valid = []
    
    for ex in examples:
        try:
            jsonschema.validate(instance=ex, schema=schema)
            valid.append(ex)
        except jsonschema.ValidationError:
            # Log and skip
            logger.warning(f"Invalid schema: {ex}")
    
    return valid
```

**JSON Schema example:**

```json
{
  "type": "object",
  "properties": {
    "instruction": {"type": "string", "minLength": 10},
    "input": {"type": "string"},
    "output": {"type": "string", "minLength": 1}
  },
  "required": ["instruction", "input", "output"]
}
```

### 2. Diversity Metrics

Measure and enforce diversity:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_diversity(examples, threshold=0.85):
    # Embed all examples
    texts = [ex["instruction"] + " " + ex["input"] for ex in examples]
    embeddings = model.encode(texts)
    
    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings)
    
    # Remove duplicates (similarity > threshold)
    keep = []
    for i, ex in enumerate(examples):
        # Check against already kept examples
        if not keep:
            keep.append(ex)
            continue
        
        keep_embeddings = [embeddings[j] for j, e in enumerate(examples) if e in keep]
        current_embedding = embeddings[i]
        
        max_sim = max(cosine_similarity([current_embedding], keep_embeddings)[0])
        
        if max_sim < threshold:
            keep.append(ex)
    
    diversity_score = len(keep) / len(examples)
    return keep, diversity_score
```

### 3. Difficulty Scoring

Balance easy/medium/hard examples:

```python
def score_difficulty(example, model):
    # Use perplexity as proxy for difficulty
    text = example["output"]
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        perplexity = torch.exp(outputs.loss)
    
    # Higher perplexity = harder example
    if perplexity < 5:
        return "easy"
    elif perplexity < 15:
        return "medium"
    else:
        return "hard"

def stratify_dataset(examples, distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2}):
    # Score all examples
    scored = [(ex, score_difficulty(ex, model)) for ex in examples]
    
    # Group by difficulty
    easy = [ex for ex, diff in scored if diff == "easy"]
    medium = [ex for ex, diff in scored if diff == "medium"]
    hard = [ex for ex, diff in scored if diff == "hard"]
    
    # Sample according to distribution
    total = len(examples)
    stratified = (
        random.sample(easy, int(total * distribution["easy"])) +
        random.sample(medium, int(total * distribution["medium"])) +
        random.sample(hard, int(total * distribution["hard"]))
    )
    
    return stratified
```

### 4. Deduplication

Remove exact and near-duplicates:

```python
import hashlib

def deduplicate(examples):
    seen_hashes = set()
    unique = []
    
    for ex in examples:
        # Exact deduplication (hash)
        content = json.dumps(ex, sort_keys=True)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique.append(ex)
    
    # Fuzzy deduplication (already done in diversity step)
    
    return unique
```

### 5. Human Spot-Check

Sample 1% for manual review:

```python
def human_review_sample(examples, sample_rate=0.01):
    sample_size = max(10, int(len(examples) * sample_rate))
    sample = random.sample(examples, sample_size)
    
    print(f"Review {sample_size} examples:")
    for i, ex in enumerate(sample):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {ex['instruction']}")
        print(f"Input: {ex['input']}")
        print(f"Output: {ex['output']}")
        
        response = input("Is this correct? (y/n/skip): ")
        if response == 'n':
            print(f"⚠ Found issue in example {i+1}")
            # Log for review
    
    print("\n✓ Spot-check complete")
```

---

## Training Integration

### SFT on Synthetic Data

```bash
# Generate dataset
expert-gen dataset create \
  --domain "json-parsing" \
  --task "parse and validate JSON" \
  --count 10000 \
  --provider "deepseek" \
  --output "datasets/json_10k.jsonl"

# Train expert
expert-train sft \
  --base qwen3-0.6b \
  --dataset datasets/json_10k.jsonl \
  --method lora \
  --r 16 \
  --epochs 3 \
  --output experts/json-parser-v1
```

### DPO on Preference Pairs

```bash
# Generate preference pairs
expert-gen dataset create \
  --domain "code-quality" \
  --task "improve code quality" \
  --count 5000 \
  --format "preference_pairs" \
  --provider "claude" \
  --output "datasets/code_quality_rlhf.jsonl"

# Train with DPO
expert-train dpo \
  --base qwen3-0.6b \
  --dataset datasets/code_quality_rlhf.jsonl \
  --method lora \
  --beta 0.1 \
  --epochs 2 \
  --output experts/code-quality-v2
```

### Distillation

```bash
# Generate reasoning traces
expert-gen dataset create \
  --domain "math" \
  --task "solve algebra problems with reasoning" \
  --count 8000 \
  --format "distillation" \
  --provider "gpt-4o" \
  --output "datasets/math_distill.jsonl"

# Train expert to mimic teacher
expert-train distill \
  --base qwen3-0.6b \
  --dataset datasets/math_distill.jsonl \
  --teacher-model "gpt-4o" \
  --temperature 2.0 \
  --output experts/math-v1
```

---

## Cost Optimization

### Batch Requests

```python
# Generate in batches to reduce API overhead
async def generate_batched(prompts, provider, batch_size=100):
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Single API call for batch
        batch_results = await provider.batch_generate(batch)
        results.extend(batch_results)
        
        # Rate limiting
        await asyncio.sleep(1)
    
    return results
```

### Use Cheaper Models for Simple Domains

```python
def select_provider(domain_complexity):
    if domain_complexity == "simple":
        return "deepseek"  # $0.14/M tokens
    elif domain_complexity == "medium":
        return "gpt-4o-mini"  # $0.15/M tokens
    else:
        return "claude"  # $3/M tokens, worth it for complex tasks
```

### Cache and Reuse

```python
# Cache generated examples
cache = {}

def generate_with_cache(domain, task, count, provider):
    cache_key = f"{domain}:{task}:{provider}"
    
    if cache_key in cache:
        print(f"Using cached dataset (found {len(cache[cache_key])} examples)")
        existing = cache[cache_key]
        
        if len(existing) >= count:
            return existing[:count]
        else:
            # Generate only the difference
            needed = count - len(existing)
            new_examples = generate(domain, task, needed, provider)
            cache[cache_key].extend(new_examples)
            return cache[cache_key]
    
    # Generate fresh
    examples = generate(domain, task, count, provider)
    cache[cache_key] = examples
    return examples
```

### Progressive Generation

```python
# Start small, expand if needed
def progressive_generation(domain, task, provider):
    # Start with 1k examples
    examples = generate(domain, task, 1000, provider)
    
    # Train quick test expert
    test_expert = train_expert(examples, epochs=1)
    
    # Evaluate
    score = evaluate(test_expert)
    
    if score < 0.8:
        # Generate more data
        additional = generate(domain, task, 4000, provider)
        examples.extend(additional)
        
        # Retrain
        final_expert = train_expert(examples, epochs=3)
    else:
        # 1k was enough
        final_expert = train_expert(examples, epochs=3)
    
    return final_expert
```

### Mix Synthetic + Real Data

```python
# 80% synthetic + 20% real for best results
def create_hybrid_dataset(synthetic_count=8000, real_count=2000):
    # Generate synthetic
    synthetic = generate_synthetic(count=synthetic_count)
    
    # Load real-world data (if available)
    real = load_real_data(count=real_count)
    
    # Combine
    dataset = synthetic + real
    random.shuffle(dataset)
    
    return dataset
```

---

## Advanced Techniques

### Self-Improvement Loop

```python
# Expert generates data → Teacher LLM critiques → Retrain
def self_improvement(expert, teacher_llm, iterations=3):
    for i in range(iterations):
        # Expert generates examples
        expert_examples = expert.generate(prompts=test_prompts)
        
        # Teacher critiques and generates corrections
        critiques = []
        for ex in expert_examples:
            critique = teacher_llm.critique(ex)
            if critique["needs_improvement"]:
                corrected = teacher_llm.correct(ex)
                critiques.append({
                    "prompt": ex["prompt"],
                    "rejected": ex["output"],
                    "chosen": corrected
                })
        
        # Retrain expert on critiques (DPO)
        expert = train_dpo(expert, critiques)
        
        print(f"Iteration {i+1}: {len(critiques)} improvements")
    
    return expert
```

### Multi-Teacher Ensemble

```python
# Combine outputs from multiple LLMs for diversity
def multi_teacher_generation(domain, task, count):
    # Split across teachers
    per_teacher = count // 3
    
    deepseek_examples = generate(domain, task, per_teacher, "deepseek")
    claude_examples = generate(domain, task, per_teacher, "claude")
    gpt4_examples = generate(domain, task, per_teacher, "gpt-4o")
    
    # Combine and deduplicate
    all_examples = deepseek_examples + claude_examples + gpt4_examples
    unique_examples = deduplicate(all_examples)
    
    return unique_examples
```

### Constitutional AI

```python
# Apply principles/rules during generation
principles = [
    "Outputs must be factually correct",
    "Avoid biased or offensive content",
    "Prioritize clarity over brevity",
    "Include error handling for edge cases"
]

prompt_with_principles = f"""
Generate examples for this task while following these principles:

{chr(10).join(f"- {p}" for p in principles)}

Domain: {domain}
Task: {task}

Generate examples:
"""
```

### Curriculum Learning

```python
# Generate data with progressive difficulty
def curriculum_dataset(domain, task, total_count=10000):
    # Easy examples (40%)
    easy = generate(
        domain, task, int(total_count * 0.4),
        difficulty="easy"
    )
    
    # Medium examples (40%)
    medium = generate(
        domain, task, int(total_count * 0.4),
        difficulty="medium"
    )
    
    # Hard examples (20%)
    hard = generate(
        domain, task, int(total_count * 0.2),
        difficulty="hard"
    )
    
    # Training: Start with easy, gradually introduce harder
    curriculum = {
        "phase1": easy[:2000],
        "phase2": easy[2000:] + medium[:2000],
        "phase3": medium[2000:] + hard
    }
    
    return curriculum
```

---

## Best Practices

1. **Start with DeepSeek**: Cost-effective and high quality for most domains
2. **Generate 2-3x more than needed**: Allows filtering and deduplication
3. **Always validate schema**: Catch malformed examples early
4. **Enforce diversity**: Use embedding-based deduplication
5. **Balance difficulty**: Mix easy/medium/hard examples
6. **Spot-check samples**: Human review catches systemic issues
7. **Version datasets**: Track which data produced which expert version
8. **Iterate quickly**: Small dataset → test → expand if needed
9. **Mix synthetic + real**: Real data adds authenticity
10. **Document prompts**: Save generation prompts with dataset for reproducibility

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Low diversity | Repetitive LLM outputs | Increase temperature, add diversity prompts |
| Format errors | LLM didn't follow schema | Add stricter schema validation, show examples in prompt |
| Too expensive | Using premium provider for simple task | Switch to DeepSeek or GPT-4o-mini |
| Low quality | Task too complex for automated generation | Use better teacher model (Claude/GPT-4), add human review |
| API rate limits | Too many parallel requests | Implement backoff, batch requests |

---

## Example: Complete JSON Parser Dataset

```bash
# 1. Generate initial dataset
expert-gen dataset create \
  --domain "json-parsing" \
  --task "parse, validate, and extract fields from JSON documents" \
  --count 12000 \
  --provider "deepseek" \
  --temperature 0.8 \
  --diversity-threshold 0.85 \
  --output "datasets/json_raw.jsonl"

# 2. Quality filter
expert-gen dataset filter \
  --input "datasets/json_raw.jsonl" \
  --output "datasets/json_filtered.jsonl" \
  --min-output-length 3 \
  --max-similarity 0.85 \
  --validate-json

# 3. Stratify by difficulty
expert-gen dataset stratify \
  --input "datasets/json_filtered.jsonl" \
  --output "datasets/json_10k.jsonl" \
  --distribution easy:0.3,medium:0.5,hard:0.2 \
  --total 10000

# 4. Train expert
expert-train sft \
  --base qwen3-0.6b \
  --dataset datasets/json_10k.jsonl \
  --method lora --r 16 \
  --epochs 3 \
  --output experts/json-parser-v1.0.0
```

**Result**: Production-ready JSON parsing expert trained in ~2 hours with $2 of API costs.

---

## Next Steps

- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for how to train experts on synthetic data
- See [EXPERT_FORMAT.md](EXPERT_FORMAT.md) for packaging trained experts
- See [ROADMAP.md](../ROADMAP.md) for upcoming synthetic data features

