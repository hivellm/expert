#!/usr/bin/env python3
"""
Debug: Generate 3 tokens step-by-step to see KV cache behavior
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_step_by_step(model_path: str, prompt: str):
    print("="*60)
    print("Step-by-Step Generation Debug")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt tokens: {input_ids[0].tolist()}")
    
    # Step 1: Process prompt
    print("\n[STEP 1] Process prompt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
    next_token = logits[0, -1, :].argmax().item()
    print(f"  Logits shape: {logits.shape}")
    print(f"  Cache layers: {len(past_key_values)}")
    print(f"  Cache[0] K shape: {past_key_values[0][0].shape}")
    print(f"  Next token: {next_token} ('{tokenizer.decode([next_token])}')")
    
    generated = [next_token]
    
    # Step 2: Generate token 2
    print("\n[STEP 2] Generate token 2")
    input_ids = torch.tensor([[next_token]], device="cuda")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
    next_token = logits[0, -1, :].argmax().item()
    print(f"  Logits shape: {logits.shape}")
    print(f"  Cache[0] K shape: {past_key_values[0][0].shape}")
    print(f"  Next token: {next_token} ('{tokenizer.decode([next_token])}')")
    generated.append(next_token)
    
    # Step 3: Generate token 3
    print("\n[STEP 3] Generate token 3")
    input_ids = torch.tensor([[next_token]], device="cuda")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
    next_token = logits[0, -1, :].argmax().item()
    print(f"  Logits shape: {logits.shape}")
    print(f"  Cache[0] K shape: {past_key_values[0][0].shape}")
    print(f"  Next token: {next_token} ('{tokenizer.decode([next_token])}')")
    generated.append(next_token)
    
    print(f"\n{'='*60}")
    print(f"Generated tokens: {generated}")
    print(f"Generated text: '{tokenizer.decode(generated)}'")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    
    debug_step_by_step(model_path, prompt)

