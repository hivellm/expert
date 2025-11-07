#!/usr/bin/env python3
"""
Test generation with Python for comparison
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def test_generation(model_path: str, prompt: str, max_tokens: int = 10):
    print("="*60)
    print("Python Generation Test")
    print("="*60)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\nFull output: {generated}")
    print(f"\nGenerated only: {generated_only}")
    print(f"Generated tokens: {len(outputs[0]) - inputs['input_ids'].shape[1]}")
    
    return generated_only

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    test_generation(model_path, prompt, max_tokens)

