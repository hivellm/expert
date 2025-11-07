"""
Template for Qualitative Checkpoint Comparison

This script runs the same prompts on all available checkpoints
and displays results for qualitative analysis by an external LLM.

USAGE:
1. Copy this file to expert-{name}/compare.py
2. Configure BASE_MODEL_PATH and CHECKPOINT_DIR
3. Define expert-specific test_cases
4. Run: python compare.py

The script does NOT evaluate quality - it only generates outputs for external analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - ADJUST THESE VALUES FOR YOUR EXPERT
# ============================================================================

# Base model path
BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"

# Directory where checkpoints are located (relative to expert directory)
CHECKPOINT_DIR = "weights/qwen3-06b"

# Generation configuration
GEN_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
}

# ============================================================================
# TEST CASES - DEFINE YOUR EXPERT-SPECIFIC PROMPTS
# ============================================================================

# Example structure - adapt for your expert:
test_cases = [
    {
        "id": "test_001",
        "category": "basic_query",
        "system_prompt": "Task: query_dsl\nDialect: elasticsearch",
        "user_prompt": "Search for documents where status equals 'active'.",
        "expected_type": "json"  # or "text", "kql", "eql", etc.
    },
    # Add more test cases here...
]

# ============================================================================
# MAIN CODE - DO NOT MODIFY FROM HERE
# ============================================================================

def detect_device():
    """Detects available device (CUDA or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def find_checkpoints(checkpoint_dir):
    """Finds all available checkpoints"""
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                try:
                    step = int(item.replace("checkpoint-", ""))
                    checkpoints.append((step, checkpoint_path))
                except ValueError:
                    continue
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def load_base_model(base_model_path, device):
    """Loads base model"""
    print(f"\n[1/3] Loading Base Model from: {base_model_path}")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"[OK] Base Model loaded (device: {device})")
    return model, tokenizer

def load_checkpoints(base_model_path, checkpoints, device):
    """Loads all checkpoints"""
    checkpoint_models = {}
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None
    
    print(f"\n[2/3] Loading {len(checkpoints)} checkpoints...")
    for step, checkpoint_path in checkpoints:
        print(f"  Loading checkpoint-{step}...", end=" ", flush=True)
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, checkpoint_path)
        checkpoint_models[step] = model
        print(f"[OK]")
    
    return checkpoint_models

def generate_output(model, tokenizer, system_prompt, user_prompt, gen_config, device):
    """Generates output for a prompt"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    gen_params = {
        **gen_config,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_params)
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
    
    return generated_text

def print_separator(char="=", width=100):
    """Prints visual separator"""
    print(char * width)

def print_test_header(test_case, test_num, total):
    """Prints test header"""
    print_separator()
    print(f"\nTEST {test_num}/{total}: {test_case.get('id', f'test_{test_num}')}")
    print(f"Category: {test_case.get('category', 'N/A')}")
    print(f"Expected type: {test_case.get('expected_type', 'N/A')}")
    print_separator("-")
    print(f"\n[SYSTEM PROMPT]")
    print(test_case['system_prompt'])
    print(f"\n[USER PROMPT]")
    print(test_case['user_prompt'])
    print_separator("-")

def print_output(label, output, max_length=500):
    """Prints formatted output"""
    print(f"\n[{label}]")
    if len(output) > max_length:
        print(output[:max_length])
        print(f"\n... (truncated, total: {len(output)} characters)")
    else:
        print(output)

def main():
    """Main function"""
    device = detect_device()
    
    print_separator()
    print("QUALITATIVE CHECKPOINT COMPARISON")
    print("This script generates outputs for external LLM analysis")
    print("Does not evaluate quality automatically")
    print_separator()
    
    # Check if there are test cases
    if not test_cases:
        print("ERROR: No test cases defined!")
        print("Configure test_cases at the beginning of the script.")
        sys.exit(1)
    
    # Find checkpoints
    checkpoints = find_checkpoints(CHECKPOINT_DIR)
    if not checkpoints:
        print(f"ERROR: No checkpoints found in: {CHECKPOINT_DIR}")
        sys.exit(1)
    
    print(f"\nCheckpoints found: {[c[0] for c in checkpoints]}")
    print(f"Total tests: {len(test_cases)}")
    print(f"Device: {device}")
    
    # Load models
    base_model, tokenizer = load_base_model(BASE_MODEL_PATH, device)
    checkpoint_models = load_checkpoints(BASE_MODEL_PATH, checkpoints, device)
    
    # Run tests
    print(f"\n[3/3] Running {len(test_cases)} tests...")
    print_separator()
    
    results = []
    
    for test_idx, test_case in enumerate(test_cases, 1):
        print_test_header(test_case, test_idx, len(test_cases))
        
        # Generate with base model
        base_output = generate_output(
            base_model, tokenizer,
            test_case['system_prompt'],
            test_case['user_prompt'],
            GEN_CONFIG,
            device
        )
        print_output("BASE MODEL", base_output)
        
        # Generate with each checkpoint
        checkpoint_outputs = {}
        for step, model in checkpoint_models.items():
            ckp_output = generate_output(
                model, tokenizer,
                test_case['system_prompt'],
                test_case['user_prompt'],
                GEN_CONFIG,
                device
            )
            checkpoint_outputs[step] = ckp_output
            print_output(f"CHECKPOINT-{step}", ckp_output)
        
        # Store result
        results.append({
            "test_id": test_case.get('id', f'test_{test_idx}'),
            "category": test_case.get('category', 'N/A'),
            "expected_type": test_case.get('expected_type', 'N/A'),
            "system_prompt": test_case['system_prompt'],
            "user_prompt": test_case['user_prompt'],
            "base_output": base_output,
            "checkpoint_outputs": checkpoint_outputs
        })
        
        print_separator()
    
    # Final summary
    print_separator()
    print("\nEXECUTION SUMMARY")
    print_separator()
    print(f"Total tests executed: {len(test_cases)}")
    print(f"Checkpoints tested: {[c[0] for c in checkpoints]}")
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"\nAll outputs have been displayed above.")
    print("Analyze the results above to determine:")
    print("  1. Which checkpoint produces best quality outputs")
    print("  2. Which checkpoint should be used to generate the package")
    print("  3. If training is progressing correctly")
    print_separator()
    
    # Save results to JSON for later analysis (optional)
    output_file = "checkpoint_comparison_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "base_model": BASE_MODEL_PATH,
                "checkpoints_tested": [c[0] for c in checkpoints],
                "device": device,
                "test_config": GEN_CONFIG,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to JSON: {e}")

if __name__ == "__main__":
    main()
