#!/usr/bin/env python3
"""
Merge multiple TypeScript datasets into a single training dataset.

This script:
1. Loads multiple JSONL files
2. Removes duplicates
3. Validates format
4. Combines into single train.jsonl
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
import argparse


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    
    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        return examples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON at line {line_num} in {file_path}: {e}")
                continue
    
    return examples


def deduplicate_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate examples based on text content."""
    seen_texts: Set[str] = set()
    unique_examples = []
    
    for example in examples:
        text = example.get("text", "").strip()
        
        if not text:
            continue
        
        # Use first 500 chars as signature (faster than full hash)
        signature = text[:500]
        
        if signature not in seen_texts:
            seen_texts.add(signature)
            unique_examples.append(example)
    
    return unique_examples


def validate_chatml_format(text: str) -> bool:
    """Validate ChatML format."""
    required_tags = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    
    for tag in required_tags:
        if tag not in text:
            return False
    
    return True


def main():
    """Main merge function."""
    parser = argparse.ArgumentParser(description="Merge TypeScript datasets")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="List of JSONL files to merge"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/train.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of existing output file"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file
    if args.backup and output_file.exists():
        backup_file = output_file.with_suffix('.jsonl.backup')
        import shutil
        shutil.copy2(output_file, backup_file)
        print(f"[OK] Backup created: {backup_file}")
    
    print("="*80)
    print("Merging TypeScript Datasets")
    print("="*80)
    
    # Load all source files
    all_examples = []
    
    for source_path in args.sources:
        source_file = base_dir / source_path
        print(f"\nLoading {source_file}...")
        
        examples = load_jsonl(source_file)
        print(f"  Loaded {len(examples):,} examples")
        
        # Validate format
        valid_count = 0
        for example in examples:
            text = example.get("text", "")
            if validate_chatml_format(text):
                valid_count += 1
        
        print(f"  Valid ChatML format: {valid_count:,}/{len(examples):,}")
        
        all_examples.extend(examples)
    
    print(f"\nTotal examples before deduplication: {len(all_examples):,}")
    
    # Deduplicate
    print("\nDeduplicating...")
    unique_examples = deduplicate_examples(all_examples)
    
    print(f"Total examples after deduplication: {len(unique_examples):,}")
    print(f"Duplicates removed: {len(all_examples) - len(unique_examples):,}")
    
    # Save merged dataset
    print(f"\nSaving merged dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(unique_examples, desc="Writing"):
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n[OK] Merged dataset saved!")
    print(f"     Total examples: {len(unique_examples):,}")
    print(f"     Output file: {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

