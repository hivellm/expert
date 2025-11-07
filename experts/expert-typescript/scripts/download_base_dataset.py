#!/usr/bin/env python3
"""
Download and convert the base TypeScript dataset from HuggingFace.

This script:
1. Downloads mhhmm/typescript-instruct-20k-v2c from HuggingFace
2. Converts to ChatML format
3. Saves to datasets/base_typescript.jsonl
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def format_to_chatml(instruction: str, output: str) -> str:
    """Format example to ChatML format."""
    chatml = f"<|system|>\nDialect: typescript\n<|end|>\n<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{output}\n<|end|>"
    return chatml


def load_base_dataset() -> List[Dict[str, Any]]:
    """Load base TypeScript dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("="*80)
        print("Downloading mhhmm/typescript-instruct-20k-v2c from HuggingFace")
        print("="*80)
        
        dataset = load_dataset("mhhmm/typescript-instruct-20k-v2c", split="train")
        print(f"\n[OK] Loaded {len(dataset):,} examples")
        
        examples = []
        for example in tqdm(dataset, desc="Converting"):
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            
            if instruction and output:
                chatml_text = format_to_chatml(instruction, output)
                examples.append({"text": chatml_text})
        
        return examples
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("\nMake sure 'datasets' is installed:")
        print("  pip install datasets")
        return []


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "datasets" / "base_typescript.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and convert dataset
    examples = load_base_dataset()
    
    if not examples:
        print("[ERROR] No examples to save")
        return
    
    # Save to JSONL
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(examples, desc="Writing"):
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n[OK] Saved {len(examples):,} examples to {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

