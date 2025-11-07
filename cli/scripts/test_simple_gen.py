#!/usr/bin/env python3
"""
Simple text generation for CLI
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def main():
    if len(sys.argv) < 6:
        print("Usage: python test_simple_gen.py <model_path> <prompt> <max_tokens> <temperature> <top_p>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2]
    max_tokens = int(sys.argv[3])
    temperature = float(sys.argv[4])
    top_p = float(sys.argv[5])

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        do_sample = temperature > 0.0
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only the generated part
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only = generated[len(prompt):].strip()

    print(f"Generated: \"{generated_only}\"")

if __name__ == "__main__":
    main()

