# Expert TypeScript

TypeScript code generation and instruction expert trained on multiple high-quality sources including synthetic datasets, official TypeScript documentation, and real-world code samples.

**Version:** 0.0.1 | **Base Model:** Qwen3-0.6B

## Quick Start

```bash
# 1. Download package (when available)
# wget https://github.com/hivellm/expert-typescript/releases/download/v0.0.1/expert-typescript-qwen3-0-6b.v0.0.1.expert

# 2. Install
expert-cli install expert-typescript-qwen3-0-6b.v0.0.1.expert

# 3. Use
expert-cli chat --experts typescript
> Create a function that validates email addresses using regex
```

**Works best for:** Code generation, refactoring, type definitions, modern TypeScript patterns  
**Limitations:** See Known Limitations below

## Features

- ✅ **Code Generation**: Generate TypeScript code from natural language descriptions
- ✅ **Code Explanation**: Explain TypeScript code logic and patterns
- ✅ **Refactoring**: Suggest improvements and refactorings for TypeScript code
- ✅ **Type Safety**: Understand and apply TypeScript type system
- ✅ **Modern Patterns**: Knowledge of modern TypeScript features and best practices
- ✅ **DoRA adapter (r=12)** optimized for code generation tasks
- ✅ **318,627 validated examples** from multiple high-quality sources
- ✅ **Comprehensive dataset** including GPT-3.5-turbo generated examples, TypeScript Handbook, and the-stack TypeScript subset

## What It Can Do ✅

**Excellent Support:**
- ✅ Function generation from natural language
- ✅ Interface and type definitions
- ✅ Class implementations
- ✅ Code refactoring suggestions
- ✅ Code explanation and documentation
- ✅ Modern TypeScript features (generics, async/await, decorators)
- ✅ Type inference and type safety
- ✅ Common patterns (singletons, factories, builders)

**Example Outputs:**

```typescript
// Code Generation
Input: "Create a function that validates email addresses using regex"
Output: 
function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Type Generation
Input: "Generate TypeScript interface for a user profile"
Output:
interface UserProfile {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  isActive: boolean;
  roles: string[];
}

// Code Explanation
Input: "Explain the logic of the useDispatcher function"
Output: The useDispatcher function takes an argument that can be a Dispatcher, 
DispatcherRegisterOptions, or an array of these types. It registers dispatchers 
with the store system...
```

## Known Limitations ⚠️

**These patterns may have lower success rates:**
- ⚠️ **Very complex type inference** - May struggle with deeply nested generics
- ⚠️ **Advanced decorators** - Limited support for complex decorator patterns
- ⚠️ **Framework-specific code** - May not follow specific framework conventions perfectly
- ⚠️ **Large codebases** - Best for single functions/interfaces, not entire modules

**Recommendation:** 
- ✅ Use for generating individual functions, types, and small code snippets
- ✅ Provide clear, specific instructions for best results
- ⚠️ Review generated code for framework-specific patterns

## Dataset

### Current Dataset
- **Total Examples**: 318,627 instruction-output pairs
- **Sources**:
  - [mhhmm/typescript-instruct-20k-v2c](https://huggingface.co/datasets/mhhmm/typescript-instruct-20k-v2c): 19,994 examples
  - TypeScript Handbook (extracted): 155 examples
  - [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) TypeScript subset: 298,543 examples (batches 1 & 2)
- **Format**: Instruction-tuning dataset for TypeScript code generation (ChatML format)
- **Preprocessing**: Deduplication applied (50,026 duplicates removed)
- **Location**: `datasets/train.jsonl`
- **Integration Date**: 2025-01-07

### Dataset Statistics
- **Expansion**: Successfully expanded from 20k to 318k+ examples (1,493% increase)
- **Quality**: Mix of GPT-3.5-turbo generated examples, official documentation, and real-world code samples
- **Coverage**: Comprehensive TypeScript patterns, modern features, and best practices

## Training

### Quick Start

```bash
# From expert-typescript directory
cd F:/Node/hivellm/expert/experts/expert-typescript

# Run training (CLI will download dataset automatically)
../../cli/target/release/expert-cli train
```

### Training Configuration

- **Base Model**: Qwen3-0.6B (INT4 quantized)
- **Adapter**: DoRA (rank=12, alpha=24)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
- **Epochs**: 3
- **Learning Rate**: 0.0003
- **Batch Size**: 16
- **Gradient Accumulation**: 4 steps
- **Max Sequence Length**: 2048

### Dataset Information

The dataset has been successfully expanded and merged. Current dataset includes:
- Base instruction-tuning examples (19,994)
- TypeScript Handbook documentation examples (155)
- the-stack TypeScript code samples (298,543)

All datasets have been merged into `datasets/train.jsonl` with deduplication applied. The dataset is ready for training.

## Testing

Run the test suite to validate the expert:

```bash
# Windows
.\test.ps1

# Linux/macOS
./test.sh
```

## Usage Examples

### Code Generation

```typescript
Input: "Create a function that validates email addresses using regex"
Output: 
function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}
```

### Code Explanation

```typescript
Input: "Explain the logic of the useDispatcher function"
Output: The useDispatcher function takes an argument that can be a Dispatcher, 
DispatcherRegisterOptions, or an array of these types. It registers dispatchers 
with the store system...
```

### Type Generation

```typescript
Input: "Generate TypeScript interface for a user profile"
Output:
interface UserProfile {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  isActive: boolean;
  roles: string[];
}
```

### Python Integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load expert adapter
adapter_path = "experts/expert-typescript"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Generate TypeScript code
prompt = "Create a function that validates email addresses using regex"

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )

code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(code)
```

## Package

Once trained, package the expert:

```bash
expert-cli package \
  --manifest manifest.json \
  --weights weights/qwen3-06b/adapter \
  --output expert-typescript-qwen3-0-6b.v0.0.1.expert
```

## Performance Metrics

| Metric | Current |
|--------|---------|
| Adapter Size | ~26 MB |
| VRAM Overhead | ~18 MB |
| Load Time | <10ms (hot) |
| Training Time | ~8-12 hours (RTX 4090) |
| Dataset Size | 318,627 examples |

## License

MIT License - Same as the source dataset

## Credits

- **Datasets**: 
  - [mhhmm/typescript-instruct-20k-v2c](https://huggingface.co/datasets/mhhmm/typescript-instruct-20k-v2c) (19,994 examples)
  - TypeScript Handbook (Microsoft) - Official documentation (155 examples)
  - [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) - TypeScript subset (298,543 examples)
- **Base Model**: Qwen3-0.6B by Alibaba Cloud
- **Training Framework**: HuggingFace PEFT + Transformers
