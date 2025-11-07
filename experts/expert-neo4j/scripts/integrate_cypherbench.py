#!/usr/bin/env python3
"""
Integrate megagonlabs/cypherbench dataset into Neo4j expert training data
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from datasets import load_dataset
from collections import defaultdict

def extract_schema_from_cypher(cypher: str) -> Dict[str, any]:
    """
    Extract node labels, relationships, and properties from Cypher query
    Returns a simplified schema representation
    """
    schema = {
        'nodes': defaultdict(set),  # label -> set of properties
        'relationships': set()
    }
    
    # Extract node patterns: (n:Label) or (n:Label {prop: value})
    node_patterns = re.findall(r'\(([^)]+)\)', cypher)
    for pattern in node_patterns:
        # Extract label
        label_match = re.search(r':(\w+)', pattern)
        if label_match:
            label = label_match.group(1)
            # Extract properties from {prop: value}
            props_match = re.search(r'\{([^}]+)\}', pattern)
            if props_match:
                props_str = props_match.group(1)
                # Simple property extraction (just names, not types)
                props = re.findall(r'(\w+)\s*:', props_str)
                for prop in props:
                    schema['nodes'][label].add(prop)
            else:
                # No explicit properties, but label exists
                schema['nodes'][label].add('name')  # Common default
    
    # Extract relationship patterns: -[:TYPE]->
    rel_patterns = re.findall(r'\[[^]]*:(\w+)[^]]*\]', cypher)
    for rel_type in rel_patterns:
        schema['relationships'].add(rel_type)
    
    # Also check RETURN clause for properties
    return_match = re.search(r'RETURN\s+(.+?)(?:\s+ORDER|\s+LIMIT|$)', cypher, re.IGNORECASE)
    if return_match:
        return_clause = return_match.group(1)
        # Extract n.prop patterns
        prop_patterns = re.findall(r'(\w+)\.(\w+)', return_clause)
        for var, prop in prop_patterns:
            # Try to find which label this variable refers to
            # This is simplified - in real Cypher, we'd need full parsing
            for label in schema['nodes'].keys():
                schema['nodes'][label].add(prop)
    
    return schema

def schema_to_string(schema: Dict) -> str:
    """Convert schema dict to string format matching current dataset"""
    lines = []
    
    # Node properties
    if schema['nodes']:
        lines.append("Node properties:")
        for label in sorted(schema['nodes'].keys()):
            props = sorted(schema['nodes'][label])
            if props:
                lines.append(f"- **{label}**")
                for prop in props:
                    lines.append(f"  - `{prop}`: STRING")
            else:
                lines.append(f"- **{label}**")
    
    # Relationships
    if schema['relationships']:
        lines.append("Relationships:")
        for rel_type in sorted(schema['relationships']):
            # We don't know the exact node types, so use generic pattern
            # This is a limitation - we'll use a generic pattern
            lines.append(f"(:Node)-[:{rel_type}]->(:Node)")
    
    return "\n".join(lines) if lines else "Node properties:\n- **Node**\nRelationships:\n(:Node)-[:RELATED_TO]->(:Node)"

def format_to_chatml(question: str, cypher: str, schema_str: str) -> str:
    """Format example to ChatML format"""
    return f"""<|system|>
Dialect: cypher
Schema:
{schema_str}
<|end|>
<|user|>
{question}
<|end|>
<|assistant|>
{cypher}
<|end|>"""

def validate_cypher_basic(cypher: str) -> bool:
    """Basic Cypher syntax validation"""
    # Check balanced parentheses, brackets, braces
    paren_count = cypher.count('(') - cypher.count(')')
    bracket_count = cypher.count('[') - cypher.count(']')
    brace_count = cypher.count('{') - cypher.count('}')
    
    if paren_count != 0 or bracket_count != 0 or brace_count != 0:
        return False
    
    # Check for basic Cypher keywords
    if not re.search(r'\b(MATCH|CREATE|MERGE|RETURN)\b', cypher, re.IGNORECASE):
        return False
    
    return True

def load_current_dataset(train_file: Path) -> Set[str]:
    """Load current dataset and extract questions for deduplication"""
    questions = set()
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        # Extract question from ChatML format
                        user_match = re.search(r'<\|user\|>\s*(.+?)\s*<\|end\|>', text, re.DOTALL)
                        if user_match:
                            question = user_match.group(1).strip()
                            questions.add(question.lower())
                    except:
                        pass
    return questions

def main():
    print("="*60)
    print("Integrating megagonlabs/cypherbench dataset")
    print("="*60)
    
    # Paths
    datasets_dir = Path(__file__).parent.parent / "datasets"
    train_file = datasets_dir / "train.jsonl"
    backup_file = datasets_dir / "train.jsonl.backup"
    
    # Load current dataset for deduplication
    print("\n1. Loading current dataset for deduplication...")
    current_questions = load_current_dataset(train_file)
    print(f"   Current dataset: {len(current_questions)} unique questions")
    
    # Backup current dataset
    if train_file.exists():
        print("\n2. Creating backup...")
        import shutil
        shutil.copy2(train_file, backup_file)
        print(f"   Backup created: {backup_file}")
    
    # Load cypherbench dataset
    print("\n3. Loading megagonlabs/cypherbench...")
    dataset = load_dataset("megagonlabs/cypherbench", split="train")
    print(f"   Total examples: {len(dataset):,}")
    
    # Process examples
    print("\n4. Processing examples...")
    new_examples = []
    skipped_duplicates = 0
    skipped_invalid = 0
    
    for i, example in enumerate(dataset):
        if (i + 1) % 1000 == 0:
            print(f"   Processed: {i + 1}/{len(dataset)}")
        
        question = example.get('nl_question', '').strip()
        cypher = example.get('gold_cypher', '').strip()
        graph_name = example.get('graph', '')
        
        # Skip if empty
        if not question or not cypher:
            skipped_invalid += 1
            continue
        
        # Check for duplicates
        if question.lower() in current_questions:
            skipped_duplicates += 1
            continue
        
        # Validate Cypher
        if not validate_cypher_basic(cypher):
            skipped_invalid += 1
            continue
        
        # Extract schema from Cypher
        schema_dict = extract_schema_from_cypher(cypher)
        schema_str = schema_to_string(schema_dict)
        
        # Format to ChatML
        chatml_text = format_to_chatml(question, cypher, schema_str)
        
        new_examples.append({
            'text': chatml_text
        })
        
        # Add to current questions to avoid duplicates within this batch
        current_questions.add(question.lower())
    
    print(f"\n   Processed: {len(dataset):,} examples")
    print(f"   New examples: {len(new_examples):,}")
    print(f"   Skipped (duplicates): {skipped_duplicates:,}")
    print(f"   Skipped (invalid): {skipped_invalid:,}")
    
    # Load current examples
    print("\n5. Loading current examples...")
    current_examples = []
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        current_examples.append(json.loads(line))
                    except:
                        pass
    
    print(f"   Current examples: {len(current_examples):,}")
    
    # Merge datasets
    print("\n6. Merging datasets...")
    merged_examples = current_examples + new_examples
    print(f"   Total examples after merge: {len(merged_examples):,}")
    
    # Write merged dataset
    print("\n7. Writing merged dataset...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in merged_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"   Dataset written to: {train_file}")
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION SUMMARY")
    print("="*60)
    print(f"Source dataset: megagonlabs/cypherbench")
    print(f"Examples loaded: {len(dataset):,}")
    print(f"New examples added: {len(new_examples):,}")
    print(f"Duplicates skipped: {skipped_duplicates:,}")
    print(f"Invalid skipped: {skipped_invalid:,}")
    print(f"Previous total: {len(current_examples):,}")
    print(f"New total: {len(merged_examples):,}")
    print(f"Increase: {len(new_examples):,} examples ({len(new_examples)/len(current_examples)*100:.1f}%)")
    print("\nNext steps:")
    print("1. Update LICENSE with megagonlabs/cypherbench license (Apache 2.0)")
    print("2. Update README.md with new dataset information")
    print("3. Update manifest.json with new dataset statistics")

if __name__ == "__main__":
    main()

