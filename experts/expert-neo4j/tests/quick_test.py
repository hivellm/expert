"""Quick test - apenas uma query para verificar evolução rápida"""
import sys
sys.path.append('..')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINTS = [
    "../weights/qwen3-06b/checkpoint-250",
    "../weights/qwen3-06b/checkpoint-500"
]

schema = """Node properties:
- **Person**
  - `name`: STRING
  - `age`: INTEGER"""

question = "Find all people older than 30"

print("="*60)
print("TESTE RÁPIDO: Base vs checkpoint-250 vs checkpoint-500")
print("="*60)
print(f"Query: {question}")
print()

# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device}")
print()

# Load base
print("[1/3] Loading Base Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device,
    dtype=dtype,
    trust_remote_code=True,
    local_files_only=True
)
print("[OK] Base loaded")

# Function to generate
def generate(model, prompt_text):
    messages = [
        {"role": "system", "content": f"Dialect: cypher\nSchema:\n{schema}"},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return result.strip().split("<|im_end|>")[0].strip()

# Test base
print("\n[BASE OUTPUT]")
print("-"*60)
base_out = generate(base_model, question)
print(base_out)
print()

# Test checkpoints
for i, ckpt in enumerate(CHECKPOINTS, 2):
    print(f"[{i}/3] Loading {ckpt.split('/')[-1]}...")
    model = PeftModel.from_pretrained(base_model, ckpt)
    print(f"[OK] Loaded")
    
    print(f"\n[{ckpt.split('/')[-1].upper()} OUTPUT]")
    print("-"*60)
    out = generate(model, question)
    print(out)
    print()
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

print("="*60)
print("[OK] Teste completo!")

