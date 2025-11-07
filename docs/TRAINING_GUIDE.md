# Training Guide

> Step-by-step guide to training custom expert specialists using LoRA, IA³, DoRA, and soft-prompts

## Overview

Training experts for the Expert System involves fine-tuning lightweight adapters on top of the frozen Qwen3-0.6B base model. This guide covers the entire workflow from dataset preparation through training, validation, and packaging for distribution.

## Training Philosophy

**Key Principles:**
1. **Base model stays frozen**: Never modify Qwen3-0.6B weights
2. **Small, focused datasets**: 1k-50k high-quality examples per domain
3. **Synthetic data preferred**: Use premium LLMs to generate training data (see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md))
4. **Fast iteration**: Train experts in minutes to hours, not days
5. **Composition over scale**: Multiple small experts > one large expert

---

## Prerequisites

### Hardware Requirements

| Configuration | Training Time (10k examples, LoRA r=16) | Notes |
|---------------|----------------------------------------|-------|
| RTX 3060 12GB | ~2-3 hours | Minimum viable |
| RTX 4070 12GB | ~1-2 hours | Recommended |
| RTX 4090 24GB | ~30-60 min | Optimal |
| A100 40GB | ~15-30 min | Overkill but fast |

**CPU-only**: Possible but slow (~10-20x slower). Not recommended for iterative development.

### Software Stack

```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install peft>=0.7.0  # Parameter-Efficient Fine-Tuning
pip install datasets
pip install accelerate
pip install safetensors
pip install bitsandbytes  # For quantization

# Optional but recommended
pip install wandb  # Experiment tracking
pip install sentencepiece  # Tokenizer
```

---

## Training Pipeline Overview

```
1. Dataset Preparation
   ↓
2. Base Model Setup
   ↓
3. Adapter Configuration
   ↓
4. Training Loop
   ↓
5. Validation & Evaluation
   ↓
6. Export to SafeTensors
   ↓
7. Package as .expert
   ↓
8. Sign & Distribute
```

---

## Step 1: Dataset Preparation

### Data Format

Experts are trained on **instruction-following** datasets:

```jsonl
{"instruction": "Parse this JSON and extract the name field", "input": "{\"name\": \"Alice\", \"age\": 30}", "output": "Alice"}
{"instruction": "Validate JSON syntax", "input": "{\"key\": \"value\"}", "output": "Valid JSON"}
{"instruction": "Extract all keys from JSON", "input": "{\"a\": 1, \"b\": 2}", "output": "[\"a\", \"b\"]"}
```

**For preference tuning (DPO/RLHF)**:

```jsonl
{"prompt": "Parse this JSON", "chosen": "Correct parse result", "rejected": "Incorrect or incomplete parse"}
```

### Dataset Size Guidelines

| Task Complexity | Min Examples | Recommended | Max Useful |
|-----------------|--------------|-------------|------------|
| Simple format parsing | 500 | 2,000 | 5,000 |
| Language grammar | 2,000 | 10,000 | 30,000 |
| Domain knowledge | 5,000 | 20,000 | 50,000 |
| Complex reasoning | 10,000 | 30,000 | 100,000 |

**Quality > Quantity**: 1,000 high-quality examples > 10,000 noisy examples

### Creating Datasets

**Option 1: Manual curation** (traditional, slow)
```python
import jsonlines

examples = [
    {
        "instruction": "Parse JSON",
        "input": '{"key": "value"}',
        "output": "Parsed successfully"
    },
    # ... more examples
]

with jsonlines.open("dataset.jsonl", "w") as f:
    f.write_all(examples)
```

**Option 2: Synthetic generation** (recommended, see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md))
```bash
expert-gen dataset create \
  --domain "json-parsing" \
  --task "parse and validate JSON documents" \
  --count 5000 \
  --provider "deepseek" \
  --output datasets/json_5k.jsonl
```

---

## Step 2: Base Model Setup

### Load Quantized Base Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",  # Or Qwen3-0.6B when available
    load_in_4bit=True,  # INT4 quantization for memory efficiency
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for training (enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)
```

### Verify Base Model

```python
# Test inference
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# Should produce coherent text (not gibberish)
```

---

## Step 3: Adapter Configuration

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank (8, 16, 32 typical)
    lora_alpha=16,  # Scaling factor (usually = r)
    target_modules=[
        "q_proj",  # Query projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection (attention)
        "gate_proj",  # Gate projection (MLP)
        "up_proj",   # Up projection (MLP)
        "down_proj"  # Down projection (MLP)
    ],
    lora_dropout=0.05,  # Dropout rate
    bias="none",  # Don't train biases
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Expected: ~0.5-2% of total parameters
```

**Rank selection**:
- `r=8`: Smallest, fastest, ~10MB. Good for simple tasks (format parsing)
- `r=16`: Balanced, ~30MB. Recommended default
- `r=32`: Larger, ~60MB. For complex domains (medical, legal)

### LoRA-FA Configuration

```python
lora_fa_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    init_lora_weights="frozen",  # Freeze matrix A
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

**Advantage**: Half the trainable params, faster training  
**Use case**: Limited data (<2k examples) or rapid iteration

### DoRA Configuration

```python
from peft import DoRAConfig

dora_config = DoRAConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "o_proj"],
    magnitude_scaling=True,  # Key difference from LoRA
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, dora_config)
```

**Advantage**: Better quality than LoRA for same rank  
**Disadvantage**: Slightly slower training (~10-15%)

### IA³ Configuration

```python
from peft import IA3Config

ia3_config = IA3Config(
    target_modules=["k_proj", "v_proj", "up_proj"],
    feedforward_modules=["up_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, ia3_config)
```

**Advantage**: Extremely lightweight (~1-5MB), fast training  
**Use case**: Simple format/style adjustments, limited VRAM

### Soft Prompt Configuration

```python
from peft import PromptTuningConfig

soft_prompt_config = PromptTuningConfig(
    num_virtual_tokens=64,  # 64-128 typical
    prompt_tuning_init="RANDOM",  # or "TEXT" with init_text
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, soft_prompt_config)
```

**Advantage**: Zero overhead on model layers, <1MB  
**Use case**: Style enforcement (e.g., "always output valid JSON")

---

## Step 4: Training Loop

### Training Configuration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output/json-parser-lora",
    
    # Training hyperparameters
    num_train_epochs=3,  # 2-5 typical
    per_device_train_batch_size=4,  # Adjust based on VRAM
    gradient_accumulation_steps=4,  # Effective batch size = 4*4=16
    learning_rate=2e-4,  # 1e-4 to 5e-4 for LoRA
    lr_scheduler_type="cosine",
    warmup_steps=100,
    
    # Optimization
    optim="adamw_torch",  # or "paged_adamw_8bit" for low VRAM
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Logging
    logging_steps=10,
    logging_dir="./logs",
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 checkpoints
    
    # Hardware
    fp16=True,  # Use mixed precision (or bf16 on Ampere GPUs)
    dataloader_num_workers=4,
    
    # Experiment tracking
    report_to="wandb",  # or "tensorboard", "none"
    run_name="json-parser-lora-r16"
)
```

### Dataset Preprocessing

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="datasets/json_5k.jsonl", split="train")

# Train/val split
dataset = dataset.train_test_split(test_size=0.1)

# Tokenization function
def preprocess_function(examples):
    # Format as instruction-following
    prompts = [
        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        for inst, inp in zip(examples["instruction"], examples["input"])
    ]
    
    targets = examples["output"]
    
    # Tokenize
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
```

### Start Training

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# Train!
trainer.train()

# Save final model
trainer.save_model("./output/json-parser-lora-final")
```

### Monitoring Training

```python
# View logs in Weights & Biases
# https://wandb.ai/your-username/your-project

# Or TensorBoard
# tensorboard --logdir ./logs
```

**Key metrics to watch**:
- **Training loss**: Should decrease smoothly
- **Eval loss**: Should track training loss (not diverge)
- **Perplexity**: Lower is better
- **Learning rate**: Should decay over time (if using scheduler)

---

## Step 5: Validation & Evaluation

### Task-Specific Evaluation

#### JSON Parsing Expert

```python
import json

def evaluate_json_parser(model, tokenizer, test_cases):
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        prompt = f"Parse this JSON: {case['input']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=128)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if output is valid JSON
        try:
            json.loads(generated)
            correct += 1
        except json.JSONDecodeError:
            pass
    
    accuracy = correct / total
    print(f"JSON parsing accuracy: {accuracy:.2%}")
    return accuracy
```

#### Classification Expert

```python
from sklearn.metrics import accuracy_score, f1_score

def evaluate_classifier(model, tokenizer, test_dataset):
    predictions = []
    ground_truth = []
    
    for example in test_dataset:
        prompt = f"Classify: {example['input']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=50)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predictions.append(pred.strip())
        ground_truth.append(example['label'])
    
    acc = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted')
    
    print(f"Accuracy: {acc:.2%}")
    print(f"F1 Score: {f1:.3f}")
```

#### Language Expert

```python
from nltk.translate.bleu_score import sentence_bleu

def evaluate_language_expert(model, tokenizer, test_cases):
    bleu_scores = []
    
    for case in test_cases:
        prompt = case['prompt']
        reference = case['expected_output']
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        score = sentence_bleu([reference.split()], generated.split())
        bleu_scores.append(score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu:.3f}")
```

### Multi-Expert Compatibility Testing

```python
def test_expert_composition(experts_list):
    """Test that multiple experts work together without conflicts"""
    
    # Load base model
    model = load_base_model()
    
    # Attach all experts
    for expert in experts_list:
        attach_expert(model, expert)
    
    # Test inference
    test_prompt = "Parse this JSON and classify it"
    result = model.generate(test_prompt)
    
    # Verify no NaN, inf, or crashes
    assert result is not None
    assert not torch.isnan(result).any()
    
    print(f"✓ Successfully composed {len(experts_list)} experts")
```

---

## Step 6: Export to SafeTensors

### Extract LoRA Weights

```python
from safetensors.torch import save_file
import torch

# Load trained model
model = AutoPeftModelForCausalLM.from_pretrained("./output/json-parser-lora-final")

# Extract LoRA weights
lora_weights = {}
for name, param in model.named_parameters():
    if "lora" in name.lower():
        lora_weights[name] = param.cpu()

# Save as SafeTensors
save_file(lora_weights, "weights.safetensors")

print(f"Saved {len(lora_weights)} LoRA parameters")
```

### Verify SafeTensors

```python
from safetensors.torch import load_file

# Load and verify
weights = load_file("weights.safetensors")

for name, tensor in weights.items():
    print(f"{name}: {tensor.shape} ({tensor.dtype})")
    
    # Check for NaN/Inf
    assert not torch.isnan(tensor).any(), f"NaN in {name}"
    assert not torch.isinf(tensor).any(), f"Inf in {name}"

print("✓ SafeTensors file is valid")
```

---

## Step 7: Package as .expert

### Create Manifest

```python
import json
import hashlib
from datetime import datetime

def create_manifest(expert_name, version, author, adapter_config):
    # Compute file hashes
    with open("weights.safetensors", "rb") as f:
        weights_hash = hashlib.sha256(f.read()).hexdigest()
    
    manifest = {
        "name": expert_name,
        "version": version,
        "description": f"{expert_name} specialist expert",
        "author": author,
        "base_model": {
            "name": "Qwen3-0.6B",
            "sha256": "base_model_hash_here",  # Replace with actual
            "quantization": "int4",
            "rope_scaling": "yarn-128k"
        },
        "adapters": [{
            "type": "lora",
            "target_modules": adapter_config.target_modules,
            "r": adapter_config.r,
            "alpha": adapter_config.lora_alpha,
            "scaling": "standard",
            "dropout": adapter_config.lora_dropout,
            "path": "weights.safetensors",
            "size_bytes": os.path.getsize("weights.safetensors"),
            "sha256": weights_hash
        }],
        "capabilities": ["format:json", "parsing"],  # Customize
        "routing": {
            "keywords": ["json", "parse"],
            "priority": 0.8
        },
        "training": {
            "dataset": "synthetic-json-5k",
            "method": "sft",
            "epochs": 3,
            "learning_rate": 2e-4,
            "trained_on": datetime.now().isoformat()
        },
        "integrity": {
            "created_at": datetime.now().isoformat(),
            "publisher": author,
            "files": {
                "weights.safetensors": f"sha256:{weights_hash}"
            }
        },
        "license": "MIT"
    }
    
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest
```

### Create Package

```bash
# Create directory
mkdir json-parser-build
cp weights.safetensors json-parser-build/
cp manifest.json json-parser-build/
echo "MIT License..." > json-parser-build/license.txt

# Package as tar.gz
tar -czf json-parser.v1.0.0.expert json-parser-build/

# Verify package
tar -tzf json-parser.v1.0.0.expert
```

---

## Step 8: Sign & Distribute

### Generate Signing Key (One-time)

```python
import ed25519

# Generate keypair
signing_key, verifying_key = ed25519.create_keypair()

# Save private key (keep secret!)
with open("~/.expert/publisher.key", "wb") as f:
    f.write(signing_key.to_bytes())

# Save public key (distribute with expert)
with open("publisher.pub", "wb") as f:
    f.write(verifying_key.to_bytes())

print(f"Public key: {verifying_key.to_ascii(encoding='hex')}")
```

### Sign Package

```python
import hashlib
import ed25519

# Load private key
with open("~/.expert/publisher.key", "rb") as f:
    signing_key = ed25519.SigningKey(f.read())

# Hash all files in package
file_hashes = {}
for filename in ["manifest.json", "weights.safetensors"]:
    with open(f"json-parser-build/{filename}", "rb") as f:
        file_hashes[filename] = hashlib.sha256(f.read()).hexdigest()

# Create canonical message
message = "\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items()))

# Sign
signature = signing_key.sign(message.encode())

# Save signature
with open("json-parser-build/signature.sig", "w") as f:
    f.write(signature.hex())

# Update manifest with signature
manifest = json.load(open("json-parser-build/manifest.json"))
manifest["integrity"]["pubkey"] = f"ed25519:{signing_key.get_verifying_key().to_ascii(encoding='hex')}"
manifest["integrity"]["signature"] = signature.hex()
json.dump(manifest, open("json-parser-build/manifest.json", "w"), indent=2)

# Re-package
os.system("tar -czf json-parser.v1.0.0.expert json-parser-build/")
```

### Distribute

```bash
# Publish to marketplace (future)
expert-cli publish json-parser.v1.0.0.expert

# Or distribute manually
cp json-parser.v1.0.0.expert /path/to/distribution/
```

---

## Advanced Training Techniques

### DPO (Direct Preference Optimization)

For alignment without RLHF:

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./output/json-parser-dpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    beta=0.1,  # KL penalty strength
    max_length=512,
    max_prompt_length=256
)

# Dataset: {"prompt": ..., "chosen": ..., "rejected": ...}
dpo_dataset = load_dataset("json", data_files="preference_pairs.jsonl")

dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset["train"],
    tokenizer=tokenizer
)

dpo_trainer.train()
```

### Knowledge Distillation

Train expert to mimic a larger teacher model:

```python
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_student = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    
    loss = torch.nn.functional.kl_div(
        soft_student,
        soft_targets,
        reduction="batchmean"
    ) * (temperature ** 2)
    
    return loss
```

### Continual Learning

Train expert on multiple tasks sequentially without forgetting:

```python
# Elastic Weight Consolidation (EWC)
def compute_fisher_information(model, dataset):
    fisher = {}
    model.eval()
    
    for batch in dataset:
        logits = model(**batch).logits
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] = param.grad.data.clone() ** 2
    
    return fisher

# Use Fisher info as regularization in next task
```

---

## Best Practices

1. **Start with LoRA r=16**: Good default for most tasks
2. **Use synthetic data**: Faster than manual curation, see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md)
3. **Validate on held-out data**: Never train on your test set
4. **Test multi-expert composition**: Ensure your expert doesn't break when combined with others
5. **Monitor for overfitting**: If eval loss increases while train loss decreases, reduce epochs
6. **Version your datasets**: Track which data produced which expert version
7. **Document thoroughly**: Good manifest.json helps router make better decisions
8. **Sign your experts**: Build trust with cryptographic signatures

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss is NaN | Learning rate too high | Reduce LR to 1e-5 |
| No improvement | Dataset too small | Generate more synthetic data |
| VRAM OOM | Batch size too large | Reduce batch size, increase grad accumulation |
| Slow training | Inefficient data loading | Increase num_workers, use faster storage |
| Expert doesn't load | SafeTensors corruption | Re-export from checkpoint |
| Poor composition | Experts conflict | Check target_modules overlap, adjust |

---

## Next Steps

- See [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) for dataset generation strategies
- See [EXPERT_FORMAT.md](EXPERT_FORMAT.md) for complete manifest specification
- See [PERFORMANCE.md](PERFORMANCE.md) for optimization tips
- See [ROADMAP.md](../ROADMAP.md) for upcoming training features

