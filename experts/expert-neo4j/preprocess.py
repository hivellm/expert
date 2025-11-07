#!/usr/bin/env python3
"""
Neo4j Cypher Text2Query Dataset Preprocessing

Converts HuggingFace datasets to expert-compatible format:
- Validates Cypher syntax (optional)
- Formats with ChatML for Qwen3
- Deduplicates by question
- Canonicalizes schema format
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

def load_dataset(dataset_name: str, split: str = "train"):
    """Load dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        print(f"Loading dataset: {dataset_name} (split: {split})")
        ds = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(ds)} examples")
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nAvailable text2cypher datasets:")
        print("  - tomasonjo/text2cypher")
        print("  - gretelai/synthetic_text_to_sql (can be adapted)")
        print("\nAlternatively, use a local JSONL file with --local-file")
        return None

def extract_schema_from_cypher_simple(cypher: str) -> str:
    """Extract a simple schema representation from Cypher query"""
    nodes = set()
    relationships = set()
    
    # Extract node labels: (n:Label) or (:Label)
    node_patterns = re.findall(r'\([^)]*:(\w+)', cypher)
    nodes.update(node_patterns)
    
    # Extract relationship types: -[:TYPE]->
    rel_patterns = re.findall(r'\[[^]]*:(\w+)', cypher)
    relationships.update(rel_patterns)
    
    if not nodes and not relationships:
        return ""
    
    lines = ["Node properties:"]
    for label in sorted(nodes):
        lines.append(f"- **{label}**")
        lines.append("  - `name`: STRING")
    
    if relationships:
        lines.append("Relationships:")
        for rel_type in sorted(relationships):
            lines.append(f"(:Node)-[:{rel_type}]->(:Node)")
    
    return "\n".join(lines)

def canonicalize_schema(schema: str) -> str:
    """Normalize schema formatting"""
    # Remove extra whitespace
    schema = re.sub(r'\s+', ' ', schema.strip())
    # Normalize node/relationship patterns
    schema = re.sub(r'\(\s+', '(', schema)
    schema = re.sub(r'\s+\)', ')', schema)
    schema = re.sub(r'\[\s+', '[', schema)
    schema = re.sub(r'\s+\]', ']', schema)
    return schema

def validate_cypher(cypher: str) -> bool:
    """Basic Cypher validation (can be enhanced with neo4j driver)"""
    if not cypher or not cypher.strip():
        return False
    
    # Check for basic Cypher keywords
    cypher_upper = cypher.upper()
    has_keyword = any(kw in cypher_upper for kw in [
        'MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE',
        'RETURN', 'WITH', 'WHERE', 'ORDER BY', 'LIMIT'
    ])
    
    if not has_keyword:
        return False
    
    # Check for basic syntax errors
    if cypher.count('(') != cypher.count(')'):
        return False
    if cypher.count('[') != cypher.count(']'):
        return False
    if cypher.count('{') != cypher.count('}'):
        return False
    
    return True

def format_chatml(question: str, cypher: str, schema: str = "", dialect: str = "cypher") -> str:
    """Format example with ChatML for Qwen3"""
    system_content = f"Dialect: {dialect}"
    if schema:
        system_content += f"\nSchema:\n{schema}"
    
    return (
        f"<|system|>\n{system_content}\n<|end|>\n"
        f"<|user|>\n{question}\n<|end|>\n"
        f"<|assistant|>\n{cypher}\n<|end|>"
    )

def format_simple(question: str, cypher: str, schema: str = "") -> Dict[str, str]:
    """Format example with simple instruction format"""
    instruction = question
    if schema:
        instruction = f"Schema: {schema}\n\nQuestion: {question}"
    
    return {
        "instruction": instruction,
        "input": "",
        "output": cypher
    }

def extract_cypher_from_chatml(text: str) -> str:
    """Extract Cypher query from ChatML text."""
    # Try standard ChatML format
    match = re.search(r'<\|assistant\|>\s*\n(.*?)\n<\|end\|>', text, re.DOTALL)
    if match:
        cypher = match.group(1).strip()
        if cypher:
            return cypher
    
    # Try without newline after assistant tag
    match = re.search(r'<\|assistant\|>(.*?)<\|end\|>', text, re.DOTALL)
    if match:
        cypher = match.group(1).strip()
        if cypher:
            return cypher
    
    # Try to find Cypher-like content after assistant tag (fallback)
    match = re.search(r'<\|assistant\|>(.*)', text, re.DOTALL)
    if match:
        cypher = match.group(1).strip()
        cypher = re.sub(r'<\|end\|>.*', '', cypher, flags=re.DOTALL).strip()
        if cypher and (cypher.upper().startswith('MATCH') or cypher.upper().startswith('CREATE') or 
                      cypher.upper().startswith('MERGE') or cypher.upper().startswith('DELETE') or
                      cypher.upper().startswith('RETURN') or cypher.upper().startswith('WITH')):
            return cypher
    
    return ""

def detect_cypher_type(cypher: str) -> str:
    """Detect Cypher command type."""
    cypher_upper = cypher.upper().strip()
    
    # Remove comments and whitespace
    cypher_upper = re.sub(r'//.*', '', cypher_upper)
    cypher_upper = re.sub(r'/\*.*?\*/', '', cypher_upper, flags=re.DOTALL)
    cypher_upper = cypher_upper.strip()
    
    if not cypher_upper:
        return "EMPTY"
    
    # Check main command types (order matters - check more specific first)
    if cypher_upper.startswith('MATCH'):
        return "MATCH"
    elif cypher_upper.startswith('CREATE'):
        return "CREATE"
    elif cypher_upper.startswith('MERGE'):
        return "MERGE"
    elif cypher_upper.startswith('DELETE'):
        return "DELETE"
    elif cypher_upper.startswith('SET'):
        return "SET"
    elif cypher_upper.startswith('REMOVE'):
        return "REMOVE"
    elif cypher_upper.startswith('RETURN'):
        return "RETURN"
    elif cypher_upper.startswith('WITH'):
        return "WITH"
    elif cypher_upper.startswith('UNWIND'):
        return "UNWIND"
    elif cypher_upper.startswith('UNION'):
        return "UNION"
    elif cypher_upper.startswith('CALL'):
        return "CALL"
    elif cypher_upper.startswith('FOREACH'):
        return "FOREACH"
    else:
        return "OTHER"

def rebalance_cypher_types(examples: List[Dict[str, Any]], target_match_ratio: float = 0.90, target_call_ratio: float = 0.05, min_total_examples: int = 5000) -> List[Dict[str, Any]]:
    """
    Rebalance Cypher command types by reducing MATCH and CALL examples.
    
    Args:
        examples: List of examples with 'text' field
        target_match_ratio: Target ratio for MATCH (default: 0.90 = 90%)
        target_call_ratio: Target max ratio for CALL (default: 0.05 = 5%)
        min_total_examples: Minimum total examples to include (default: 5000)
    
    Returns:
        Rebalanced list of examples
    """
    import random
    from collections import defaultdict
    
    # Categorize examples
    examples_by_type = defaultdict(list)
    
    for example in examples:
        text = example.get("text", "")
        if not text:
            continue
        
        cypher = extract_cypher_from_chatml(text)
        if not cypher:
            continue
        
        cypher_type = detect_cypher_type(cypher)
        examples_by_type[cypher_type].append(example)
    
    # Separate examples by type
    match_examples = examples_by_type.get("MATCH", [])
    call_examples = examples_by_type.get("CALL", [])
    other_examples = []
    
    for cypher_type, ex_list in examples_by_type.items():
        if cypher_type not in ["MATCH", "CALL"]:
            other_examples.extend(ex_list)
    
    random.seed(42)  # For reproducibility
    
    other_count = len(other_examples)
    available_total = len(match_examples) + len(call_examples) + other_count
    
    # Calculate target counts to reach min_total_examples while maintaining ratios
    # We want: MATCH = 70%, CALL = 5%, Other = 25%
    
    if other_count > 0:
        # Strategy: Use all other examples, then add MATCH and CALL to reach min_total_examples
        # while maintaining approximate ratios
        
        # Start with all other examples
        target_other_count = other_count
        
        # Calculate how many MATCH and CALL we need to reach min_total_examples
        remaining_slots = min_total_examples - target_other_count
        
        if remaining_slots > 0:
            # Calculate MATCH and CALL counts to maintain ratios
            # MATCH should be ~70% of total, CALL ~5% of total
            # So: MATCH = 0.7 * total, CALL = 0.05 * total
            # remaining_slots = MATCH + CALL
            # remaining_slots = 0.7 * total + 0.05 * total = 0.75 * total
            # total = remaining_slots / 0.75
            # MATCH = 0.7 * total = 0.7 * (remaining_slots / 0.75) = remaining_slots * 0.7/0.75
            target_match_count = int(remaining_slots * (target_match_ratio / (target_match_ratio + target_call_ratio)))
            target_call_count = int(remaining_slots * (target_call_ratio / (target_match_ratio + target_call_ratio)))
            
            # Ensure we don't exceed available examples
            target_match_count = min(target_match_count, len(match_examples))
            target_call_count = min(target_call_count, len(call_examples))
            
            # If we still have room and more MATCH available, add more MATCH to reach min_total_examples
            current_total = target_match_count + target_call_count + target_other_count
            if current_total < min_total_examples and len(match_examples) > target_match_count:
                additional_match = min(min_total_examples - current_total, len(match_examples) - target_match_count)
                target_match_count += additional_match
            
            # Recalculate CALL to maintain ~5% ratio of final total
            final_total = target_match_count + target_call_count + target_other_count
            ideal_call_count = int(final_total * target_call_ratio)
            ideal_call_count = min(ideal_call_count, len(call_examples))
            if ideal_call_count > target_call_count:
                target_call_count = ideal_call_count
        else:
            # We have enough other examples, use ratios
            target_match_count = int(min_total_examples * target_match_ratio)
            target_call_count = int(min_total_examples * target_call_ratio)
            target_match_count = min(target_match_count, len(match_examples))
            target_call_count = min(target_call_count, len(call_examples))
        
        # Ensure minimum counts
        if len(match_examples) > 0:
            min_match_count = min(100, len(match_examples))
            target_match_count = max(target_match_count, min_match_count)
    else:
        # If no other examples, use all available
        target_match_count = len(match_examples)
        target_call_count = len(call_examples)
    
    # Sample examples
    selected_match = random.sample(match_examples, target_match_count) if match_examples and target_match_count > 0 else []
    selected_call = random.sample(call_examples, target_call_count) if call_examples and target_call_count > 0 else []
    
    # Combine all examples
    rebalanced = selected_match + selected_call + other_examples
    
    # Shuffle to mix types
    random.shuffle(rebalanced)
    
    return rebalanced

def load_documentation_examples(raw_dir: Path) -> List[Dict[str, Any]]:
    """Load Neo4j official documentation examples from datasets root
    
    Returns:
        List of examples with 'question', 'cypher', 'schema' fields
    """
    examples = []
    # Documentation is now in the root of datasets directory
    doc_file = raw_dir / "neo4j_documentation.jsonl"
    
    if not doc_file.exists():
        print(f"[DOCUMENTATION] File not found: {doc_file}")
        return examples
    
    print(f"[DOCUMENTATION] Loading examples from: {doc_file}")
    
    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                # Ensure required fields exist
                if example.get("question") and example.get("cypher"):
                    examples.append({
                        "question": example.get("question", ""),
                        "cypher": example.get("cypher", ""),
                        "schema": example.get("schema", "")
                    })
            except Exception as e:
                print(f"[DOCUMENTATION] Error loading line: {e}")
                continue
    
    print(f"[DOCUMENTATION] Loaded {len(examples):,} examples")
    return examples

def process_dataset(
    dataset_name: str = None,
    local_file: str = None,
    output_dir: str = "datasets/processed",
    dialect: str = "cypher",
    validate: bool = False,
    no_deduplicate: bool = False,
    format_type: str = "chatml",
    field_mapping: Dict[str, str] = None,
    include_documentation: bool = False,
    raw_dir: str = "datasets/raw",
    rebalance: bool = True,
    target_match_ratio: float = 0.75
):
    """Process dataset and save in expert format"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load main dataset
    examples = []
    
    if local_file:
        print(f"Loading local file: {local_file}")
        with open(local_file, 'r', encoding='utf-8') as f:
            if local_file.endswith('.jsonl'):
                examples = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue
                print(f"Loaded {len(examples):,} examples from JSONL file")
            else:
                examples = json.load(f)
    elif dataset_name:
        # Load multiple datasets if specified
        if dataset_name == "all" or dataset_name == "neo4j+cypherbench":
            # Load both neo4j/text2cypher-2025v1 and megagonlabs/cypherbench
            print("Loading multiple datasets from HuggingFace...")
            
            # Load neo4j/text2cypher-2025v1
            print("\n[1/2] Loading neo4j/text2cypher-2025v1...")
            ds1 = load_dataset("neo4j/text2cypher-2025v1")
            if ds1:
                examples1 = list(ds1)
                examples.extend(examples1)
                print(f"  Added {len(examples1):,} examples from neo4j/text2cypher-2025v1")
            
            # Load megagonlabs/cypherbench
            print("\n[2/2] Loading megagonlabs/cypherbench...")
            ds2 = load_dataset("megagonlabs/cypherbench")
            if ds2:
                examples2 = list(ds2)
                examples.extend(examples2)
                print(f"  Added {len(examples2):,} examples from megagonlabs/cypherbench")
            
            print(f"\nTotal examples loaded: {len(examples):,}")
        else:
            # Load single dataset
            ds = load_dataset(dataset_name)
            if ds is None:
                return
            examples = list(ds)
    else:
        print("Error: Must provide either --dataset or --local-file")
        return
    
    # Load documentation examples if requested
    if include_documentation:
        # Documentation is now in datasets root, not in raw subdirectory
        datasets_root = Path("datasets")
        doc_examples = load_documentation_examples(datasets_root)
        if doc_examples:
            print(f"[DOCUMENTATION] Adding {len(doc_examples):,} documentation examples")
            examples.extend(doc_examples)
    
    # Default field mapping - detect automatically based on available fields
    if field_mapping is None:
        # Try to detect field names from first example
        if examples:
            first_ex = examples[0]
            # Check for cypherbench format (nl_question, gold_cypher)
            if "nl_question" in first_ex and "gold_cypher" in first_ex:
                field_mapping = {
                    "question": "nl_question",
                    "cypher": "gold_cypher",
                    "schema": "schema"
                }
            # Check for standard format (question, cypher)
            elif "question" in first_ex and "cypher" in first_ex:
                field_mapping = {
                    "question": "question",
                    "cypher": "cypher",
                    "schema": "schema"
                }
            else:
                # Default fallback
                field_mapping = {
                    "question": "question",
                    "cypher": "cypher",
                    "schema": "schema"
                }
        else:
            field_mapping = {
                "question": "question",
                "cypher": "cypher",
                "schema": "schema"
            }
    
    # Process examples
    processed = []
    seen_questions = set()
    stats = defaultdict(int)
    
    for idx, example in enumerate(examples):
        try:
            # Check if already in ChatML format (has "text" field)
            if "text" in example and example["text"]:
                text = example["text"]
                # Extract question from ChatML for deduplication
                question_match = re.search(r'<\|user\|>\s*\n(.*?)\n<\|end\|>', text, re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip()
                else:
                    # Fallback: extract from user tag without newline
                    question_match = re.search(r'<\|user\|>(.*?)<\|end\|>', text, re.DOTALL)
                    if question_match:
                        question = question_match.group(1).strip()
                    else:
                        question = text[:100]  # Fallback: use first 100 chars
                
                # Extract cypher for validation
                cypher = extract_cypher_from_chatml(text)
                
                # Validate Cypher if requested
                if validate and cypher and not validate_cypher(cypher):
                    stats['invalid_cypher'] += 1
                    continue
                
                # Deduplicate by question
                if not no_deduplicate:
                    question_key = question.strip().lower()
                    if question_key in seen_questions:
                        stats['duplicates'] += 1
                        continue
                    seen_questions.add(question_key)
                
                # Format example (already in ChatML)
                if format_type == "chatml":
                    processed.append({"text": text})
                else:
                    # Convert ChatML to simple format if needed
                    formatted = format_simple(question, cypher, "")
                    processed.append(formatted)
                
                stats['processed'] += 1
                continue
            
            # Original formats - extract fields (detect format per example)
            # Check for cypherbench format first
            if "nl_question" in example and "gold_cypher" in example:
                question = example.get("nl_question", "")
                cypher = example.get("gold_cypher", "")
                schema = example.get("schema", "")
            # Check for standard format
            elif "question" in example and "cypher" in example:
                question = example.get("question", "")
                cypher = example.get("cypher", "")
                schema = example.get("schema", "")
            else:
                # Try field mapping
                question = example.get(field_mapping.get("question", "question"), "")
                cypher = example.get(field_mapping.get("cypher", "cypher"), "")
                schema = example.get(field_mapping.get("schema", "schema"), "")
            
            # For cypherbench, extract schema from Cypher if not provided
            if not schema and cypher and ("gold_cypher" in example or "nl_question" in example):
                schema = extract_schema_from_cypher_simple(cypher)
            
            if not question or not cypher:
                stats['missing_fields'] += 1
                continue
            
            # Validate Cypher
            if validate and not validate_cypher(cypher):
                stats['invalid_cypher'] += 1
                continue
            
            # Canonicalize schema
            if schema:
                schema = canonicalize_schema(schema)
            
            # Deduplicate by question
            if not no_deduplicate:
                question_key = question.strip().lower()
                if question_key in seen_questions:
                    stats['duplicates'] += 1
                    continue
                seen_questions.add(question_key)
            
            # Format example
            if format_type == "chatml":
                text = format_chatml(question, cypher, schema, dialect)
                processed.append({"text": text})
            else:
                formatted = format_simple(question, cypher, schema)
                processed.append(formatted)
            
            stats['processed'] += 1
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(examples)} examples...")
        
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] < 10:
                print(f"Error processing example {idx}: {e}")
    
    # Rebalance Cypher command types if requested
    if rebalance and len(processed) > 0:
        target_call_ratio = 0.05  # Max 5% for CALL
        print(f"\n[REBALANCING] Rebalancing Cypher command types (target MATCH ratio: {target_match_ratio*100:.1f}%, max CALL ratio: {target_call_ratio*100:.1f}%)...")
        # Convert to list for rebalancing
        examples_list = [{"text": ex["text"]} for ex in processed]
        
        if len(examples_list) == 0:
            print(f"      No examples to rebalance")
        else:
            # Count types before
            from collections import Counter
            before_types = Counter()
            
            for ex in examples_list:
                cypher = extract_cypher_from_chatml(ex["text"])
                if cypher:
                    cypher_type = detect_cypher_type(cypher)
                    before_types[cypher_type] += 1
            
            rebalanced_list = rebalance_cypher_types(examples_list, target_match_ratio, target_call_ratio, min_total_examples=5000)
            
            # Count types after
            after_types = Counter()
            
            for ex in rebalanced_list:
                cypher = extract_cypher_from_chatml(ex["text"])
                if cypher:
                    cypher_type = detect_cypher_type(cypher)
                    after_types[cypher_type] += 1
            
            if len(examples_list) > 0:
                before_match_pct = before_types.get('MATCH', 0) / len(examples_list) * 100
                before_call_pct = before_types.get('CALL', 0) / len(examples_list) * 100
                print(f"      Before: MATCH={before_types.get('MATCH', 0):,} ({before_match_pct:.1f}%), CALL={before_types.get('CALL', 0):,} ({before_call_pct:.1f}%)")
            if len(rebalanced_list) > 0:
                after_match_pct = after_types.get('MATCH', 0) / len(rebalanced_list) * 100
                after_call_pct = after_types.get('CALL', 0) / len(rebalanced_list) * 100
                print(f"      After:  MATCH={after_types.get('MATCH', 0):,} ({after_match_pct:.1f}%), CALL={after_types.get('CALL', 0):,} ({after_call_pct:.1f}%)")
            print(f"      Reduction: {len(examples_list):,} -> {len(rebalanced_list):,} examples")
            
            # Convert back to processed format
            processed = [{"text": ex["text"]} for ex in rebalanced_list]
    else:
        if not rebalance:
            print(f"\n[REBALANCING] Skipping rebalancing")
    
    # Save processed dataset
    output_file = output_path / "train.jsonl"
    print(f"\nSaving {len(processed)} examples to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in processed:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats_file = output_path / "preprocessing_stats.json"
    stats_data = {
        "total_examples": len(examples),
        "processed": stats['processed'],
        "skipped": {
            "missing_fields": stats['missing_fields'],
            "invalid_cypher": stats['invalid_cypher'],
            "duplicates": stats['duplicates'],
            "errors": stats['errors']
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total examples:     {len(examples)}")
    print(f"Processed:          {stats['processed']}")
    print(f"Missing fields:     {stats['missing_fields']}")
    if validate:
        print(f"Invalid Cypher:     {stats['invalid_cypher']}")
    if not no_deduplicate:
        print(f"Duplicates:         {stats['duplicates']}")
    print(f"Errors:             {stats['errors']}")
    print(f"\nOutput: {output_file}")
    print(f"Stats:  {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Neo4j Cypher datasets for expert training")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="neo4j+cypherbench",
        help="HuggingFace dataset name. Use 'neo4j+cypherbench' or 'all' to load both neo4j/text2cypher-2025v1 and megagonlabs/cypherbench (default: neo4j+cypherbench)"
    )
    
    parser.add_argument(
        "--local-file",
        type=str,
        help="Path to local JSONL or JSON file (alternative to --dataset)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="datasets",
        help="Output directory (default: datasets)"
    )
    
    parser.add_argument(
        "--dialect",
        type=str,
        default="cypher",
        help="Query dialect (default: cypher)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate Cypher syntax (basic validation)"
    )
    
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Skip deduplication by question"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["chatml", "simple"],
        default="chatml",
        help="Output format (default: chatml for Qwen3)"
    )
    
    parser.add_argument(
        "--field-question",
        type=str,
        default="question",
        help="Field name for question in source dataset"
    )
    
    parser.add_argument(
        "--field-cypher",
        type=str,
        default="cypher",
        help="Field name for Cypher query in source dataset"
    )
    
    parser.add_argument(
        "--field-schema",
        type=str,
        default="schema",
        help="Field name for schema in source dataset"
    )
    
    parser.add_argument(
        "--include-documentation",
        action="store_true",
        help="Include Neo4j official documentation examples (from datasets/raw/documentation/)"
    )
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="datasets/raw",
        help="Directory containing raw datasets (default: datasets/raw)"
    )
    
    parser.add_argument(
        "--no-rebalance",
        action="store_true",
        help="Skip Cypher command type rebalancing (default: enabled, reduces MATCH to ~75%)"
    )
    
    parser.add_argument(
        "--match-ratio",
        type=float,
        default=0.75,
        help="Target MATCH ratio for rebalancing (default: 0.75 = 75 percent)"
    )
    
    args = parser.parse_args()
    
    field_mapping = {
        "question": args.field_question,
        "cypher": args.field_cypher,
        "schema": args.field_schema
    }
    
    process_dataset(
        dataset_name=args.dataset if not args.local_file else None,
        local_file=args.local_file,
        output_dir=args.output,
        dialect=args.dialect,
        validate=args.validate,
        no_deduplicate=args.no_deduplicate,
        format_type=args.format,
        field_mapping=field_mapping,
        include_documentation=args.include_documentation,
        raw_dir=args.raw_dir,
        rebalance=not args.no_rebalance,
        target_match_ratio=args.match_ratio
    )

if __name__ == "__main__":
    main()

