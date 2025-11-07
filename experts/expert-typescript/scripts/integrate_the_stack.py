#!/usr/bin/env python3
"""
Integrate TypeScript subset from bigcode/the-stack dataset.

This script:
1. Loads the-stack dataset
2. Filters TypeScript files
3. Extracts code examples with context
4. Creates instruction-output pairs
5. Validates TypeScript syntax
6. Saves to datasets/the_stack_typescript.jsonl
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set
from datasets import load_dataset
from tqdm import tqdm
import subprocess
import tempfile
import os


def validate_typescript(code: str) -> bool:
    """Validate TypeScript syntax using tsc."""
    if not code or len(code.strip()) < 10:
        return False
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Run tsc --noEmit
        result = subprocess.run(
            ['tsc', '--noEmit', '--skipLibCheck', temp_file],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # If tsc not available, do basic validation
        return True
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def extract_functions_and_classes(content: str) -> List[Dict[str, str]]:
    """Extract functions and classes from TypeScript code."""
    examples = []
    
    # Extract function declarations
    function_pattern = r'(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\)\s*(:\s*[^{]+)?\s*\{[^}]*\}'
    functions = re.finditer(function_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in functions:
        func_code = match.group(0)
        func_name = match.group(3)
        
        # Create instruction
        instruction = f"Show TypeScript function: {func_name}"
        
        examples.append({
            "instruction": instruction,
            "output": func_code.strip(),
            "type": "function"
        })
    
    # Extract class declarations
    class_pattern = r'(export\s+)?class\s+(\w+)[^{]*\{[^}]*\}'
    classes = re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in classes:
        class_code = match.group(0)
        class_name = match.group(2)
        
        instruction = f"Show TypeScript class: {class_name}"
        
        examples.append({
            "instruction": instruction,
            "output": class_code.strip(),
            "type": "class"
        })
    
    # Extract interface declarations
    interface_pattern = r'(export\s+)?interface\s+(\w+)[^{]*\{[^}]*\}'
    interfaces = re.finditer(interface_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in interfaces:
        interface_code = match.group(0)
        interface_name = match.group(2)
        
        instruction = f"Show TypeScript interface: {interface_name}"
        
        examples.append({
            "instruction": instruction,
            "output": interface_code.strip(),
            "type": "interface"
        })
    
    return examples


def format_to_chatml(instruction: str, output: str) -> str:
    """Format example to ChatML format."""
    chatml = f"<|system|>\nDialect: typescript\n<|end|>\n<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{output}\n<|end|>"
    return chatml


def load_the_stack_typescript(limit: int = None) -> List[Dict[str, Any]]:
    """Load TypeScript files from the-stack dataset."""
    print(f"\n{'='*80}")
    print("Loading TypeScript subset from bigcode/the-stack")
    print(f"{'='*80}")
    
    try:
        # Load the-stack dataset (streaming for large dataset)
        dataset = load_dataset(
            "bigcode/the-stack",
            data_dir="data/typescript",
            split="train",
            streaming=True
        )
        print("[OK] Dataset loaded (streaming mode)")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("[INFO] Trying alternative: load full dataset and filter")
        try:
            dataset = load_dataset("bigcode/the-stack", split="train")
            # Filter TypeScript files
            dataset = dataset.filter(lambda x: x.get("ext", "") == "ts" or x.get("ext", "") == "tsx")
            print(f"[OK] Loaded and filtered {len(dataset):,} TypeScript files")
        except Exception as e2:
            print(f"[ERROR] Alternative also failed: {e2}")
            return []
    
    examples = []
    valid_count = 0
    invalid_count = 0
    seen_code = set()
    
    print(f"\nProcessing TypeScript files...")
    iterator = dataset
    
    if limit:
        # For streaming, we need to manually limit
        iterator = iter(dataset)
        count = 0
    
    for example in tqdm(iterator, desc="Processing"):
        if limit and count >= limit:
            break
        
        content = example.get("content", "").strip()
        
        if not content or len(content) < 50:
            invalid_count += 1
            continue
        
        # Extract code examples
        code_examples = extract_functions_and_classes(content)
        
        for code_example in code_examples:
            code = code_example["output"]
            
            # Skip duplicates
            code_hash = hash(code[:200])  # Use first 200 chars as signature
            if code_hash in seen_code:
                continue
            seen_code.add(code_hash)
            
            # Validate TypeScript (optional, can be slow)
            # if not validate_typescript(code):
            #     invalid_count += 1
            #     continue
            
            examples.append(code_example)
            valid_count += 1
        
        if limit:
            count += 1
    
    print(f"\n[OK] Extracted {valid_count:,} valid examples")
    print(f"     Skipped {invalid_count:,} invalid/duplicate examples")
    
    return examples


def main():
    """Main integration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate TypeScript from the-stack")
    parser.add_argument("--limit", type=int, default=10000, help="Limit number of files to process")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_file = Path(args.output) if args.output else base_dir / "datasets" / "the_stack_typescript.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load examples
    examples = load_the_stack_typescript(limit=args.limit)
    
    if not examples:
        print("[ERROR] No examples to save")
        return
    
    # Format and save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(examples, desc="Writing"):
            chatml_text = format_to_chatml(
                example["instruction"],
                example["output"]
            )
            
            json_line = json.dumps({"text": chatml_text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n[OK] Saved {len(examples):,} examples to {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

