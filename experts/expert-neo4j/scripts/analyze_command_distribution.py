#!/usr/bin/env python3
"""
Analyze Cypher command distribution in datasets to find those with more non-MATCH examples.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any
from datasets import load_dataset

def detect_cypher_type(cypher: str) -> str:
    """Detect Cypher command type."""
    if not cypher:
        return "EMPTY"
    
    cypher_upper = cypher.upper().strip()
    cypher_upper = re.sub(r'//.*', '', cypher_upper)
    cypher_upper = re.sub(r'/\*.*?\*/', '', cypher_upper, flags=re.DOTALL)
    cypher_upper = cypher_upper.strip()
    
    if not cypher_upper:
        return "EMPTY"
    
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

def analyze_dataset_commands(dataset_name: str, cypher_field: str, limit: int = None) -> Dict[str, Any]:
    """Analyze command distribution in a dataset."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        ds = load_dataset(dataset_name, split="train")
        total = len(ds)
        
        if limit:
            print(f"Analyzing first {limit:,} examples (of {total:,} total)")
            sample_size = min(limit, total)
        else:
            print(f"Analyzing all {total:,} examples")
            sample_size = total
        
        commands = Counter()
        non_match_count = 0
        
        for i in range(sample_size):
            try:
                example = ds[i]
                cypher = example.get(cypher_field, "")
                
                if isinstance(cypher, str) and cypher.strip():
                    cypher_type = detect_cypher_type(cypher)
                    commands[cypher_type] += 1
                    
                    if cypher_type != "MATCH":
                        non_match_count += 1
            except Exception as e:
                continue
        
        match_count = commands.get("MATCH", 0)
        match_ratio = match_count / sample_size if sample_size > 0 else 0
        non_match_ratio = non_match_count / sample_size if sample_size > 0 else 0
        
        print(f"\nCommand Distribution:")
        for cmd_type, count in sorted(commands.items(), key=lambda x: x[1], reverse=True):
            pct = count / sample_size * 100 if sample_size > 0 else 0
            print(f"  {cmd_type:15s}: {count:6,} ({pct:5.2f}%)")
        
        print(f"\nSummary:")
        print(f"  MATCH ratio:     {match_ratio*100:.2f}%")
        print(f"  Non-MATCH ratio: {non_match_ratio*100:.2f}%")
        print(f"  Non-MATCH count: {non_match_count:,}")
        
        return {
            "dataset": dataset_name,
            "total": total,
            "analyzed": sample_size,
            "commands": dict(commands),
            "match_ratio": match_ratio,
            "non_match_ratio": non_match_ratio,
            "non_match_count": non_match_count,
            "score": non_match_count * (1 - match_ratio)  # Higher score = better for balancing
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "dataset": dataset_name,
            "error": str(e)
        }

def main():
    # Datasets to analyze
    datasets = [
        ("neo4j/text2cypher-2025v1", "cypher", None),
        ("megagonlabs/cypherbench", "gold_cypher", None),
        ("jiuyuan/train_cypher", "output_text", None),
    ]
    
    results = []
    
    for dataset_name, cypher_field, limit in datasets:
        result = analyze_dataset_commands(dataset_name, cypher_field, limit)
        results.append(result)
    
    # Find best candidates for balancing
    print(f"\n{'='*70}")
    print("BALANCING SCORE RANKING")
    print(f"{'='*70}")
    print("\nRanked by potential to balance dataset (higher score = better):")
    print()
    
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x.get("score", 0), reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['dataset']}")
        print(f"   Non-MATCH: {result['non_match_count']:,} ({result['non_match_ratio']*100:.2f}%)")
        print(f"   MATCH: {result['match_ratio']*100:.2f}%")
        print(f"   Score: {result['score']:.1f}")
        print()
    
    # Recommendations
    print(f"{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    best = sorted_results[0] if sorted_results else None
    if best and best['non_match_ratio'] > 0.05:  # At least 5% non-MATCH
        print(f"\n[OK] Best candidate: {best['dataset']}")
        print(f"   Would add ~{best['non_match_count']:,} non-MATCH examples")
        print(f"   Non-MATCH ratio: {best['non_match_ratio']*100:.2f}%")
    else:
        print("\n[WARNING] No dataset found with significant non-MATCH examples")
        print("   Consider:")
        print("   1. Collecting more CREATE/MERGE/DELETE examples from documentation")
        print("   2. Generating synthetic examples for write operations")
        print("   3. Using The Stack dataset filtered for Cypher mutations")

if __name__ == "__main__":
    main()

