#!/usr/bin/env python3
"""
Download TypeScript examples from WizardLM dataset.

This script:
1. Downloads WizardLM_evol_instruct_V2_196k from HuggingFace
2. Filters TypeScript-related examples
3. Converts to ChatML format
4. Saves to datasets/wizardlm_typescript.jsonl
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def format_to_chatml(instruction: str, output: str) -> str:
    """Format example to ChatML format."""
    chatml = f"<|system|>\nDialect: typescript\n<|end|>\n<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{output}\n<|end|>"
    return chatml


def is_typescript_related(text: str) -> bool:
    """Check if text is TypeScript-related."""
    text_lower = text.lower()
    typescript_keywords = [
        "typescript", "tsx", ".ts", "interface", "type ", "declare",
        ": string", ": number", ": boolean", "as ", "extends", "implements"
    ]
    
    # Check for TypeScript keywords
    keyword_count = sum(1 for keyword in typescript_keywords if keyword in text_lower)
    
    # Also check for TypeScript patterns
    has_typescript_pattern = bool(
        re.search(r':\s*(string|number|boolean|any|void|object|Array<)', text) or
        re.search(r'interface\s+\w+', text) or
        re.search(r'type\s+\w+\s*=', text) or
        re.search(r'\.tsx?', text_lower)
    )
    
    return keyword_count >= 2 or has_typescript_pattern


def load_wizardlm_dataset(limit: int = None) -> List[Dict[str, Any]]:
    """Load WizardLM dataset and filter TypeScript examples."""
    try:
        from datasets import load_dataset
        print("="*80)
        print("Downloading WizardLM_evol_instruct_V2_196k from HuggingFace")
        print("="*80)
        
        # Try to load dataset
        try:
            dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
        except Exception as e:
            print(f"[WARNING] Failed to load full dataset: {e}")
            print("[INFO] Trying alternative dataset name...")
            try:
                dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True)
            except Exception as e2:
                print(f"[ERROR] Failed to load dataset: {e2}")
                return []
        
        print(f"\n[OK] Dataset loaded")
        
        examples = []
        typescript_count = 0
        total_checked = 0
        
        iterator = dataset
        if limit:
            # For streaming, manually limit
            iterator = iter(dataset)
            count = 0
        
        print("\nFiltering TypeScript examples...")
        for example in tqdm(iterator, desc="Processing"):
            if limit and count >= limit:
                break
            
            total_checked += 1
            
            # Get instruction and output
            instruction = example.get("instruction", "") or example.get("input", "") or ""
            output = example.get("output", "") or example.get("response", "") or ""
            
            # Combine text for checking
            combined_text = f"{instruction} {output}"
            
            if is_typescript_related(combined_text):
                typescript_count += 1
                chatml_text = format_to_chatml(instruction, output)
                examples.append({"text": chatml_text})
            
            if limit:
                count += 1
            
            # Stop if we have enough examples
            if len(examples) >= 10000:
                print(f"\n[INFO] Reached 10,000 TypeScript examples, stopping...")
                break
        
        print(f"\n[OK] Found {typescript_count:,} TypeScript examples out of {total_checked:,} checked")
        
        return examples
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download TypeScript examples from WizardLM")
    parser.add_argument("--limit", type=int, default=50000, help="Limit number of examples to check")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "datasets" / "wizardlm_typescript.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and filter dataset
    examples = load_wizardlm_dataset(limit=args.limit)
    
    if not examples:
        print("[WARNING] No TypeScript examples found")
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

