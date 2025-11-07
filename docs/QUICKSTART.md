# Quick Start: First Experts

> Practical guide to train your first 6 experts and test the system

## Overview

Instead of following the full P0-P6 roadmap, this guide takes a **practical-first approach**:

1. Train 6 fundamental experts (Python tooling)
2. Build simple CLI to test them (Python prototype)
3. Benchmark performance
4. Validate the architecture before building Rust runtime

**Time estimate**: 2-4 weeks to working prototype

---

## The First 6 Experts

### Expert Dependency Tree

```
document-classifier (task expert)
    ├── english-basic (language)
    ├── json-parser (format)
    └── neo4j-cypher (technology)

python-code-expert (technology)
    └── english-basic (language)

rust-code-expert (technology)
    └── english-basic (language)
```

### 1. English-Basic (Foundation)

**Purpose**: English language understanding  
**Dependencies**: None  
**Load Order**: 3

**Training dataset**:
```bash
# Generate 10k English language examples
expert-gen dataset create \
  --domain "english-language" \
  --task "understand and generate English text" \
  --count 10000 \
  --provider "deepseek" \
  --difficulty "mixed" \
  --output datasets/english_10k.jsonl
```

**Training**:
```bash
python scripts/train_expert.py \
  --name "english-basic" \
  --dataset datasets/english_10k.jsonl \
  --method lora \
  --r 16 \
  --target-modules q_proj,v_proj,o_proj \
  --epochs 3 \
  --output experts/english-basic-v1
```

**Manifest** (experts/english-basic-v1/manifest.json):
```json
{
  "name": "english-basic",
  "version": "1.0.0",
  "description": "English language understanding and generation",
  "capabilities": ["language:en", "language:en-US", "language:en-GB"],
  "constraints": {
    "load_order": 3,
    "requires": []
  },
  "routing": {
    "keywords": ["english", "en", "grammar", "writing"],
    "priority": 0.9
  }
}
```

### 2. JSON-Parser (Format)

**Purpose**: Parse and validate JSON  
**Dependencies**: None  
**Load Order**: 1

**Training dataset**:
```bash
expert-gen dataset create \
  --domain "json-parsing" \
  --task "parse, validate, and extract fields from JSON documents" \
  --count 8000 \
  --provider "gpt-4o" \
  --format "instruction" \
  --output datasets/json_8k.jsonl
```

**Training**:
```bash
python scripts/train_expert.py \
  --name "json-parser" \
  --dataset datasets/json_8k.jsonl \
  --method lora \
  --r 16 \
  --epochs 3 \
  --output experts/json-parser-v1
```

**Manifest**:
```json
{
  "name": "json-parser",
  "version": "2.0.0",
  "description": "JSON parsing and validation specialist",
  "capabilities": ["format:json", "parsing", "validation"],
  "constraints": {
    "load_order": 1,
    "requires": []
  },
  "routing": {
    "keywords": ["json", "parse", "validate", "{", "}"],
    "priority": 0.95
  }
}
```

### 3. Neo4j-Cypher (Technology)

**Purpose**: Neo4j Cypher query understanding  
**Dependencies**: english-basic  
**Load Order**: 6

**Training dataset**:
```bash
expert-gen dataset create \
  --domain "neo4j-cypher" \
  --task "understand and generate Neo4j Cypher queries and schema patterns" \
  --count 6000 \
  --provider "claude" \
  --output datasets/neo4j_6k.jsonl
```

**Training**:
```bash
python scripts/train_expert.py \
  --name "neo4j-cypher" \
  --dataset datasets/neo4j_6k.jsonl \
  --method lora \
  --r 16 \
  --epochs 3 \
  --output experts/neo4j-cypher-v1
```

**Manifest**:
```json
{
  "name": "neo4j-cypher",
  "version": "1.0.0",
  "description": "Neo4j Cypher query and schema specialist",
  "capabilities": ["tech:neo4j", "language:cypher", "graph-database"],
  "constraints": {
    "load_order": 6,
    "requires": ["english-basic@>=1.0.0"]
  },
  "routing": {
    "keywords": ["neo4j", "cypher", "graph", "MATCH", "CREATE"],
    "priority": 0.85
  }
}
```

### 4. Python-Code (Technology)

**Purpose**: Python code understanding  
**Dependencies**: english-basic  
**Load Order**: 6

**Training dataset**:
```bash
expert-gen dataset create \
  --domain "python-code" \
  --task "understand, analyze, and generate Python code" \
  --count 8000 \
  --provider "deepseek" \
  --output datasets/python_8k.jsonl
```

**Training**:
```bash
python scripts/train_expert.py \
  --name "python-code" \
  --dataset datasets/python_8k.jsonl \
  --method lora \
  --r 16 \
  --epochs 3 \
  --output experts/python-code-v1
```

**Manifest**:
```json
{
  "name": "python-code",
  "version": "1.0.0",
  "description": "Python code analysis and generation specialist",
  "capabilities": ["tech:python", "code-analysis", "language:python"],
  "constraints": {
    "load_order": 6,
    "requires": ["english-basic@>=1.0.0"]
  },
  "routing": {
    "keywords": ["python", ".py", "import", "def", "class"],
    "priority": 0.9
  }
}
```

### 5. Rust-Code (Technology)

**Purpose**: Rust code understanding  
**Dependencies**: english-basic  
**Load Order**: 6

**Training dataset**:
```bash
expert-gen dataset create \
  --domain "rust-code" \
  --task "understand, analyze, and generate Rust code" \
  --count 7000 \
  --provider "claude" \
  --output datasets/rust_7k.jsonl
```

**Training**:
```bash
python scripts/train_expert.py \
  --name "rust-code" \
  --dataset datasets/rust_7k.jsonl \
  --method lora \
  --r 16 \
  --epochs 3 \
  --output experts/rust-code-v1
```

**Manifest**:
```json
{
  "name": "rust-code",
  "version": "1.0.0",
  "description": "Rust code analysis and generation specialist",
  "capabilities": ["tech:rust", "code-analysis", "language:rust"],
  "constraints": {
    "load_order": 6,
    "requires": ["english-basic@>=1.0.0"]
  },
  "routing": {
    "keywords": ["rust", ".rs", "fn", "struct", "impl", "cargo"],
    "priority": 0.9
  }
}
```

### 6. Document-Classifier (Task)

**Purpose**: Classify documents (like classify project)  
**Dependencies**: english-basic, json-parser, neo4j-cypher  
**Load Order**: 9

**Training dataset**:
```bash
# Use real examples from classify project + synthetic data
expert-gen dataset create \
  --domain "document-classification" \
  --task "classify documents by type, language, and technology" \
  --count 10000 \
  --provider "gpt-4o" \
  --format "preference_pairs" \
  --output datasets/classifier_10k.jsonl
```

**Training** (using DPO for better classification):
```bash
python scripts/train_expert.py \
  --name "document-classifier" \
  --dataset datasets/classifier_10k.jsonl \
  --method dpo \
  --r 16 \
  --epochs 2 \
  --output experts/document-classifier-v1
```

**Manifest**:
```json
{
  "name": "document-classifier",
  "version": "1.0.0",
  "description": "Document classification specialist (type, language, tech)",
  "capabilities": ["task:classification", "document-analysis"],
  "constraints": {
    "load_order": 9,
    "requires": [
      "english-basic@>=1.0.0",
      "json-parser@>=2.0.0",
      "neo4j-cypher@>=1.0.0"
    ]
  },
  "routing": {
    "keywords": ["classify", "classification", "categorize", "type"],
    "priority": 0.8
  }
}
```

---

## Project Structure

```
expert/
├── scripts/
│   ├── train_expert.py          # Training script
│   ├── generate_dataset.py      # Synthetic data generation
│   └── test_cli.py              # Simple test CLI
├── datasets/
│   ├── english_10k.jsonl
│   ├── json_8k.jsonl
│   ├── neo4j_6k.jsonl
│   ├── python_8k.jsonl
│   ├── rust_7k.jsonl
│   └── classifier_10k.jsonl
├── experts/
│   ├── english-basic-v1/
│   │   ├── manifest.json
│   │   ├── weights.safetensors
│   │   └── metadata.json
│   ├── json-parser-v1/
│   ├── neo4j-cypher-v1/
│   ├── python-code-v1/
│   ├── rust-code-v1/
│   └── document-classifier-v1/
└── tests/
    ├── test_english.py
    ├── test_json.py
    └── test_classifier.py
```

---

## Week-by-Week Plan

### Week 1: Setup & First 2 Experts

**Days 1-2**: Setup
```bash
# Create project structure
mkdir -p expert/{scripts,datasets,experts,tests}

# Install dependencies
pip install torch transformers peft datasets trl bitsandbytes safetensors

# Download base model
python scripts/download_base.py qwen2.5-0.5b
```

**Days 3-4**: Train English-Basic
```bash
# Generate dataset
python scripts/generate_dataset.py \
  --domain english-language \
  --count 10000 \
  --output datasets/english_10k.jsonl

# Train expert
python scripts/train_expert.py \
  --name english-basic \
  --dataset datasets/english_10k.jsonl
```

**Days 5-7**: Train JSON-Parser
```bash
# Generate dataset
python scripts/generate_dataset.py \
  --domain json-parsing \
  --count 8000 \
  --output datasets/json_8k.jsonl

# Train expert
python scripts/train_expert.py \
  --name json-parser \
  --dataset datasets/json_8k.jsonl

# Test both experts
python scripts/test_cli.py --experts english-basic,json-parser
```

### Week 2: Technology Experts

**Days 8-10**: Neo4j + Python experts
```bash
# Generate datasets
python scripts/generate_dataset.py --domain neo4j-cypher --count 6000
python scripts/generate_dataset.py --domain python-code --count 8000

# Train
python scripts/train_expert.py --name neo4j-cypher --dataset datasets/neo4j_6k.jsonl
python scripts/train_expert.py --name python-code --dataset datasets/python_8k.jsonl
```

**Days 11-14**: Rust + Classifier
```bash
# Generate datasets
python scripts/generate_dataset.py --domain rust-code --count 7000
python scripts/generate_dataset.py --domain document-classification --count 10000

# Train
python scripts/train_expert.py --name rust-code --dataset datasets/rust_7k.jsonl
python scripts/train_expert.py --name document-classifier --dataset datasets/classifier_10k.jsonl --method dpo
```

### Week 3: Integration & Testing

**Days 15-17**: Build Simple CLI
```python
# scripts/test_cli.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

class SimpleExpertCLI:
    def __init__(self, base_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_4bit=True,
            device_map="auto"
        )
        self.loaded_experts = {}
    
    def load_expert(self, expert_path):
        """Load a single expert (LoRA adapter)"""
        manifest = json.load(open(f"{expert_path}/manifest.json"))
        name = manifest["name"]
        
        # Load dependencies first
        for dep in manifest.get("constraints", {}).get("requires", []):
            dep_name = dep.split("@")[0]
            if dep_name not in self.loaded_experts:
                self.load_expert(f"experts/{dep_name}-v1")
        
        # Load this expert
        model = PeftModel.from_pretrained(
            self.base_model,
            expert_path,
            adapter_name=name
        )
        self.loaded_experts[name] = model
        print(f"✓ Loaded: {name}")
        
        return model
    
    def classify_document(self, file_content):
        """Classify document using all loaded experts"""
        # Load classifier + dependencies
        model = self.load_expert("experts/document-classifier-v1")
        
        prompt = f"""Classify this document:

{file_content[:500]}

Output format (JSON):
{{"type": "...", "language": "...", "technologies": [...]}}

Classification:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

# Usage
cli = SimpleExpertCLI("qwen2.5-0.5b")
result = cli.classify_document(open("test.py").read())
print(result)
```

**Days 18-19**: Benchmark Performance
```python
# tests/benchmark.py
import time
import psutil
import torch

def benchmark_expert_loading():
    """Measure expert load time"""
    cli = SimpleExpertCLI("qwen2.5-0.5b")
    
    start = time.time()
    cli.load_expert("experts/json-parser-v1")
    load_time = time.time() - start
    
    print(f"Expert load time: {load_time*1000:.1f}ms")

def benchmark_inference():
    """Measure inference latency"""
    cli = SimpleExpertCLI("qwen2.5-0.5b")
    cli.load_expert("experts/document-classifier-v1")
    
    test_doc = open("test.py").read()
    
    start = time.time()
    result = cli.classify_document(test_doc)
    latency = time.time() - start
    
    print(f"Inference latency: {latency:.2f}s")
    print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

def benchmark_multi_expert():
    """Test multiple expert composition"""
    cli = SimpleExpertCLI("qwen2.5-0.5b")
    
    # Load 5 experts
    experts = ["english-basic", "json-parser", "neo4j-cypher", "python-code", "document-classifier"]
    
    start = time.time()
    for exp in experts:
        cli.load_expert(f"experts/{exp}-v1")
    total_load_time = time.time() - start
    
    print(f"5 experts loaded in: {total_load_time:.2f}s")
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

if __name__ == "__main__":
    benchmark_expert_loading()
    benchmark_inference()
    benchmark_multi_expert()
```

**Days 20-21**: Validation & Iteration
- Test on real files from classify project
- Measure accuracy vs current classify implementation
- Fine-tune experts if needed
- Document results

### Week 4: Documentation & Planning

**Days 22-24**: Document findings
- Performance metrics (latency, VRAM, accuracy)
- Expert interaction patterns
- Dependency resolution issues
- Bottlenecks discovered

**Days 25-28**: Plan Rust runtime
- Based on Python prototype learnings
- Decide on tensor ops library (Candle vs Burn)
- Design hot-swap mechanism
- Plan migration path

---

## Success Criteria

### Functionality
- [ ] All 6 experts train successfully
- [ ] Dependencies resolve correctly
- [ ] CLI can classify documents
- [ ] Accuracy ≥ 85% on classify test set

### Performance (Python Prototype)
- [ ] Expert load time <500ms (cold, Python overhead)
- [ ] Inference <30s (1024 tokens, Python)
- [ ] VRAM usage <8GB (5 experts loaded)
- [ ] No crashes with 6+ experts

### Quality
- [ ] English expert: coherent text generation
- [ ] JSON expert: valid JSON output
- [ ] Neo4j expert: correct Cypher queries
- [ ] Classifier: accurate document types

---

## Next Steps After Week 4

Based on prototype results:

1. **If successful**: Begin Rust runtime (P0 from roadmap)
2. **If performance issues**: Optimize Python, then migrate
3. **If accuracy issues**: Improve datasets, retrain
4. **If dependency issues**: Refine manifest schema

---

## Quick Commands Reference

```bash
# Generate all datasets (run once)
bash scripts/generate_all_datasets.sh

# Train all experts (sequential, ~12 hours on RTX 4090)
bash scripts/train_all_experts.sh

# Test classification
python scripts/test_cli.py --file test.py --experts document-classifier

# Benchmark
python tests/benchmark.py

# Validate expert
python scripts/validate_expert.py experts/json-parser-v1
```

---

## Troubleshooting

**Issue**: CUDA OOM during training  
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: Expert dependencies not resolving  
**Solution**: Check manifest.json `requires` field and versions

**Issue**: Poor classification accuracy  
**Solution**: Generate more synthetic data or add DPO preference pairs

**Issue**: Slow inference  
**Solution**: This is Python prototype - expect 10x speedup in Rust runtime

---

## Cost Estimate

**Synthetic data generation** (DeepSeek Chat @ $0.14/M tokens):
- 6 experts × 8k examples avg × ~500 tokens = ~24M tokens
- **Cost**: ~$3-5

**GPU time** (RTX 4090 or equivalent):
- 6 experts × 2 hours avg = 12 hours
- **Cost**: Free (local) or ~$12-20 (cloud)

**Total**: $15-25 for complete prototype

---

## Files to Create

See [/scripts/README.md](scripts/README.md) for implementation of:
- `generate_dataset.py`
- `train_expert.py`
- `test_cli.py`
- `benchmark.py`
- `validate_expert.py`

All scripts will be created in Week 1 setup phase.

