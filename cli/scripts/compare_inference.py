#!/usr/bin/env python3
"""
Compare Rust vs Python inference with same prompts
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def test_python_generation(model_path: str, prompt: str, max_tokens: int = 50):
    """Test with Python/Transformers (reference implementation)"""
    print("Python/Transformers Generation")
    print("=" * 80)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_token_count = inputs['input_ids'].shape[1]
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_token_count}")
    print(f"Max new tokens: {max_tokens}")
    print(f"Temperature: 0.7")
    print(f"Top-p: 0.9")
    print()
    
    # Generate
    print("Output: ", end='', flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_only = tokenizer.decode(outputs[0][prompt_token_count:], skip_special_tokens=True)
    print(generated_only)
    print()
    print(f"Generated tokens: {len(outputs[0]) - prompt_token_count}")
    print()
    
    return generated_only

if __name__ == "__main__":
    model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    
    # Test prompts
    test_prompts = [
        "The capital of Brazil is",
        "def fibonacci(n):",
        "Hello world",
        "What is 2+2?",
    ]
    
    if len(sys.argv) > 1:
        # Single prompt mode
        prompt = sys.argv[1]
        max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        test_python_generation(model_path, prompt, max_tokens)
    else:
        # Multiple prompts mode
        print("=" * 80)
        print("PYTHON/TRANSFORMERS INFERENCE TESTS")
        print("=" * 80)
        print()
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"TEST {i}/{len(test_prompts)}")
            print("-" * 80)
            test_python_generation(model_path, prompt, 50)
            print()

