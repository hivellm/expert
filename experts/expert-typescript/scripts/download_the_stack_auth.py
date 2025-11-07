#!/usr/bin/env python3
"""
Download TypeScript code from bigcode/the-stack with authentication.

This script requires a HuggingFace token:
1. Get token from https://huggingface.co/settings/tokens
2. Set HF_TOKEN environment variable or use huggingface-cli login
3. Accept the dataset terms at https://huggingface.co/datasets/bigcode/the-stack

Usage:
    export HF_TOKEN=your_token_here
    python scripts/download_the_stack_auth.py --limit 10000
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Set
from tqdm import tqdm
import argparse


def format_to_chatml(instruction: str, output: str) -> str:
    """Format example to ChatML format."""
    chatml = f"<|system|>\nDialect: typescript\n<|end|>\n<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{output}\n<|end|>"
    return chatml


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


def load_the_stack_typescript(limit: int = None, token: str = None) -> List[Dict[str, Any]]:
    """Load TypeScript files from the-stack dataset."""
    try:
        from datasets import load_dataset
        
        print("="*80)
        print("Loading TypeScript subset from bigcode/the-stack")
        print("="*80)
        
        # Check for token
        if not token:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Try to get token from HuggingFace cache
        if not token:
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
            except Exception:
                pass
        
        # Try to read from token file (check both Windows and WSL paths)
        if not token:
            # Get WSL home path via subprocess if available
            wsl_home = None
            try:
                import subprocess
                result = subprocess.run(
                    ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-c", "echo $HOME"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    wsl_home = result.stdout.strip()
            except Exception:
                pass
            
            token_paths = [
                os.path.expanduser("~/.cache/huggingface/token"),
                os.path.expanduser("~/.huggingface/token"),
            ]
            
            # Add WSL paths if available
            if wsl_home:
                token_paths.extend([
                    os.path.join(wsl_home, ".cache", "huggingface", "token"),
                    os.path.join(wsl_home, ".huggingface", "token"),
                ])
            
            # Also try reading via WSL command
            if not token:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-c", "cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.huggingface/token 2>/dev/null || echo ''"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        token = result.stdout.strip()
                        print("[INFO] Token loaded from WSL")
                except Exception:
                    pass
            
            # Try file paths
            for token_path in token_paths:
                if token_path and os.path.exists(token_path):
                    try:
                        with open(token_path, 'r') as f:
                            token = f.read().strip()
                            if token:
                                print(f"[INFO] Token loaded from: {token_path[:50]}...")
                                break
                    except Exception:
                        continue
        
        if not token:
            print("[ERROR] HuggingFace token required!")
            print("\nTo use this script:")
            print("1. Get token from https://huggingface.co/settings/tokens")
            print("2. Accept dataset terms at https://huggingface.co/datasets/bigcode/the-stack")
            print("3. Set HF_TOKEN environment variable or use: huggingface-cli login")
            return []
        
        # Set token in environment for datasets library
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_TOKEN"] = token
            print(f"[INFO] Using token: {token[:10]}...{token[-4:]}")
        
        # Verify token and access
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"[INFO] Authenticated as: {user_info.get('name', 'unknown')}")
        except Exception as e:
            print(f"[WARNING] Could not verify token: {e}")
        
        # Try to load with token
        try:
            # Try streaming first (more memory efficient)
            print("\n[INFO] Attempting to load dataset (streaming mode)...")
            dataset = load_dataset(
                "bigcode/the-stack",
                data_dir="data/typescript",
                split="train",
                streaming=True,
                token=token
            )
            print("[OK] Dataset loaded (streaming mode)")
        except Exception as e:
            error_msg = str(e)
            if "gated dataset" in error_msg.lower() or "ask for access" in error_msg.lower():
                print(f"\n[ERROR] Dataset access required!")
                print("\n" + "="*80)
                print("ACTION REQUIRED: Accept dataset terms")
                print("="*80)
                print("\n1. Visit: https://huggingface.co/datasets/bigcode/the-stack")
                print("2. Click 'Agree and access repository'")
                print("3. Accept the terms and conditions")
                print("4. Run this script again")
                print("\n" + "="*80)
            else:
                print(f"[WARNING] Streaming failed: {e}")
                print("[INFO] Trying full dataset load...")
                try:
                    dataset = load_dataset(
                        "bigcode/the-stack",
                        data_dir="data/typescript",
                        split="train",
                        token=token
                    )
                    print(f"[OK] Loaded {len(dataset):,} TypeScript files")
                except Exception as e2:
                    print(f"[ERROR] Failed to load dataset: {e2}")
                    if "gated dataset" in str(e2).lower():
                        print("\n[ACTION REQUIRED] Please accept the dataset terms:")
                        print("   https://huggingface.co/datasets/bigcode/the-stack")
            return []
        
        examples = []
        valid_count = 0
        invalid_count = 0
        seen_code = set()
        
        print(f"\nProcessing TypeScript files...")
        iterator = dataset
        
        if limit:
            iterator = iter(dataset)
            count = 0
        
        for example in tqdm(iterator, desc="Processing"):
            if limit and count >= limit:
                break
            
            content = example.get("content", "").strip()
            
            if not content or len(content) < 50:
                invalid_count += 1
                if limit:
                    count += 1
                continue
            
            # Extract code examples
            code_examples = extract_functions_and_classes(content)
            
            for code_example in code_examples:
                code = code_example["output"]
                
                # Skip duplicates
                code_hash = hash(code[:200])
                if code_hash in seen_code:
                    continue
                seen_code.add(code_hash)
                
                # Format to ChatML
                chatml_text = format_to_chatml(
                    code_example["instruction"],
                    code_example["output"]
                )
                
                examples.append({"text": chatml_text})
                valid_count += 1
            
            if limit:
                count += 1
            
            # Stop if we have enough examples (only if no limit specified, otherwise use limit)
            # For large limits, continue processing until limit is reached
        
        print(f"\n[OK] Extracted {valid_count:,} valid examples")
        print(f"     Skipped {invalid_count:,} invalid/duplicate examples")
        
        return examples
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download TypeScript from the-stack (requires auth)")
    parser.add_argument("--limit", type=int, default=10000, help="Limit number of files to process")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or use HF_TOKEN env var)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_file = Path(args.output) if args.output else base_dir / "datasets" / "the_stack_typescript.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load examples
    examples = load_the_stack_typescript(limit=args.limit, token=args.token)
    
    if not examples:
        print("[ERROR] No examples to save")
        return
    
    # Format and save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(examples, desc="Writing"):
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n[OK] Saved {len(examples):,} examples to {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

