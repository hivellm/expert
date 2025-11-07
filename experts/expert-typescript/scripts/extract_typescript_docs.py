#!/usr/bin/env python3
"""
Extract TypeScript documentation examples and convert to instruction-output pairs.

This script:
1. Scrapes TypeScript Handbook (typescriptlang.org/docs)
2. Extracts code examples with explanations
3. Converts to instruction-output format
4. Saves to datasets/typescript_docs.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# TypeScript Handbook sections to extract
# Using the actual TypeScript documentation structure
HANDBOOK_SECTIONS = [
    "handbook/typescript-in-5-minutes",
    "handbook/basic-types",
    "handbook/variable-declarations",
    "handbook/interfaces",
    "handbook/classes",
    "handbook/functions",
    "handbook/generics",
    "handbook/enums",
    "handbook/modules",
    "handbook/namespaces",
    "handbook/symbols",
    "handbook/iterators-and-generators",
    "handbook/decorators",
    "handbook/type-manipulation/creating-types-from-types",
    "handbook/type-manipulation/keyof-types",
    "handbook/type-manipulation typeof-types",
    "handbook/type-manipulation/indexed-access-types",
    "handbook/type-manipulation/conditional-types",
    "handbook/type-manipulation/mapped-types",
    "handbook/type-manipulation/template-literal-types",
]


def extract_code_blocks(html_content: str) -> List[Dict[str, str]]:
    """Extract code blocks and their context from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    examples = []
    
    # Find code blocks in <pre><code> tags (most common)
    pre_code_blocks = soup.find_all('pre')
    
    for pre_block in pre_code_blocks:
        code_block = pre_block.find('code')
        if not code_block:
            continue
            
        code_text = code_block.get_text().strip()
        
        # Skip if too short
        if len(code_text) < 20:
            continue
        
        # Check if it looks like TypeScript/JavaScript code
        is_typescript = (
            'typescript' in code_text.lower() or
            ': string' in code_text or ': number' in code_text or
            'interface ' in code_text or 'type ' in code_text or
            'export ' in code_text or 'import ' in code_text or
            re.search(r':\s*(string|number|boolean|any|void)', code_text) or
            re.search(r'interface\s+\w+', code_text) or
            re.search(r'type\s+\w+\s*=', code_text)
        )
        
        if not is_typescript:
            continue
        
        # Try to find preceding explanation (look for headings, paragraphs)
        explanation = ""
        for prev_tag in ['h1', 'h2', 'h3', 'h4', 'p', 'div']:
            prev_elem = pre_block.find_previous(prev_tag)
            if prev_elem:
                text = prev_elem.get_text().strip()
                if len(text) > 20 and len(text) < 500:
                    explanation = text
                    break
        
        # Also try to get the section title
        section_title = ""
        h2_elem = pre_block.find_previous('h2')
        if h2_elem:
            section_title = h2_elem.get_text().strip()
        
        # Create instruction from explanation or generate from code
        if explanation:
            instruction = f"{explanation[:150]}"
        elif section_title:
            instruction = f"Show TypeScript example for {section_title}"
        else:
            # Generate instruction from code context
            if 'function' in code_text:
                func_match = re.search(r'function\s+(\w+)', code_text)
                if func_match:
                    instruction = f"Show TypeScript function: {func_match.group(1)}"
                else:
                    instruction = "Show TypeScript function example"
            elif 'interface' in code_text:
                interface_match = re.search(r'interface\s+(\w+)', code_text)
                if interface_match:
                    instruction = f"Show TypeScript interface: {interface_match.group(1)}"
                else:
                    instruction = "Show TypeScript interface example"
            elif 'class' in code_text:
                class_match = re.search(r'class\s+(\w+)', code_text)
                if class_match:
                    instruction = f"Show TypeScript class: {class_match.group(1)}"
                else:
                    instruction = "Show TypeScript class example"
            else:
                instruction = "Show TypeScript code example"
        
        examples.append({
            "instruction": instruction,
            "output": code_text,
            "source": "typescript-handbook"
        })
    
    return examples


def fetch_handbook_section(section: str) -> List[Dict[str, str]]:
    """Fetch a TypeScript Handbook section and extract examples."""
    url = f"https://www.typescriptlang.org/docs/{section}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        examples = extract_code_blocks(response.text)
        return examples
    except Exception as e:
        print(f"[WARNING] Failed to fetch {section}: {e}")
        return []


def format_to_chatml(instruction: str, output: str) -> str:
    """Format example to ChatML format."""
    chatml = f"<|system|>\nDialect: typescript\n<|end|>\n<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{output}\n<|end|>"
    return chatml


def main():
    """Main extraction function."""
    print("="*80)
    print("Extracting TypeScript Handbook Examples")
    print("="*80)
    
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "datasets" / "typescript_docs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    
    print(f"\nFetching {len(HANDBOOK_SECTIONS)} handbook sections...")
    for section in tqdm(HANDBOOK_SECTIONS, desc="Sections"):
        examples = fetch_handbook_section(section)
        all_examples.extend(examples)
        print(f"  {section}: {len(examples)} examples")
    
    print(f"\nTotal examples extracted: {len(all_examples)}")
    
    # Format to ChatML and save
    print(f"\nFormatting and saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(all_examples, desc="Writing"):
            chatml_text = format_to_chatml(
                example["instruction"],
                example["output"]
            )
            
            json_line = json.dumps({"text": chatml_text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n[OK] Saved {len(all_examples)} examples to {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

