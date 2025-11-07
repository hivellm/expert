#!/usr/bin/env python3
"""
Generate distribution charts for Neo4j expert dataset.

Creates visualizations showing:
- Cypher command type distribution
- Complexity distribution
- Dataset source breakdown (if available)
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

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

def analyze_complexity(cypher: str) -> str:
    """Analyze Cypher query complexity."""
    cypher_upper = cypher.upper()
    
    # Count complexity indicators
    has_match = bool(re.search(r'\bMATCH\b', cypher_upper))
    has_where = bool(re.search(r'\bWHERE\b', cypher_upper))
    has_return = bool(re.search(r'\bRETURN\b', cypher_upper))
    has_with = bool(re.search(r'\bWITH\b', cypher_upper))
    has_order_by = bool(re.search(r'\bORDER\s+BY\b', cypher_upper))
    has_limit = bool(re.search(r'\bLIMIT\b', cypher_upper))
    has_skip = bool(re.search(r'\bSKIP\b', cypher_upper))
    
    # Pattern complexity
    has_relationship = bool(re.search(r'\[.*?\]', cypher_upper))
    has_variable_length = bool(re.search(r'\[.*?\*.*?\]', cypher_upper))
    has_property_filter = bool(re.search(r'\{.*?\}', cypher_upper))
    
    # Aggregations
    has_aggregation = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*\(', cypher_upper))
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', cypher_upper))
    
    # Advanced features
    has_case = bool(re.search(r'\bCASE\s+WHEN\b', cypher_upper))
    has_union = bool(re.search(r'\bUNION\b', cypher_upper))
    has_call = bool(re.search(r'\bCALL\b', cypher_upper))
    
    # Count complexity score
    complexity_score = sum([
        has_match, has_where, has_return, has_with,
        has_order_by, has_limit, has_skip,
        has_relationship, has_variable_length, has_property_filter,
        has_aggregation, has_group_by, has_case, has_union, has_call
    ])
    
    # Adjust for variable-length paths (more complex)
    if has_variable_length:
        complexity_score += 2
    
    if complexity_score <= 3:
        return "Simple"
    elif complexity_score <= 6:
        return "Medium"
    elif complexity_score <= 10:
        return "Complex"
    else:
        return "Very Complex"

def load_and_analyze_dataset(dataset_path: Path) -> Dict[str, Counter]:
    """Load dataset and analyze command types and complexity."""
    commands = Counter()
    complexity_levels = Counter()
    total = 0
    
    print("Analyzing dataset...")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                text = example.get("text", "")
                
                if not text:
                    continue
                
                cypher = extract_cypher_from_chatml(text)
                if not cypher:
                    continue
                
                cypher_type = detect_cypher_type(cypher)
                commands[cypher_type] += 1
                
                complexity = analyze_complexity(cypher)
                complexity_levels[complexity] += 1
                
                total += 1
                
                if (i + 1) % 5000 == 0:
                    print(f"  Processed {i + 1:,} examples...")
                    
            except Exception as e:
                continue
    
    print(f"Total examples analyzed: {total:,}")
    print()
    
    return {
        "commands": commands,
        "complexity": complexity_levels
    }

def plot_command_type_distribution(commands: Counter, output_dir: Path):
    """Plot Cypher command type distribution."""
    # Sort by count (descending)
    sorted_commands = sorted(commands.items(), key=lambda x: x[1], reverse=True)
    types = [item[0] for item in sorted_commands]
    counts = [item[1] for item in sorted_commands]
    
    # Calculate percentages
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    bars = ax1.bar(types, counts, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Cypher Command Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Cypher Command Type Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels(types, rotation=45, ha='right')
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(counts, labels=types, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       textprops={'fontsize': 10})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax2.set_title('Cypher Command Type Distribution (Percentage)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    plt.savefig(output_dir / "dataset_distribution.png", dpi=300, bbox_inches="tight")
    try:
        plt.savefig(output_dir / "dataset_distribution.pdf", bbox_inches="tight")
    except PermissionError:
        print(f"  Warning: Could not save PDF (file may be open)")
    
    plt.close()
    
    print(f"Command type charts saved:")
    print(f"  - {output_dir / 'dataset_distribution.png'}")
    print(f"  - {output_dir / 'dataset_distribution.pdf'}")

def plot_complexity_distribution(complexity: Counter, output_dir: Path):
    """Plot complexity distribution."""
    # Order complexity levels
    order = ["Simple", "Medium", "Complex", "Very Complex"]
    sorted_complexity = [(level, complexity.get(level, 0)) for level in order if level in complexity]
    
    if not sorted_complexity:
        print("No complexity data to plot")
        return
    
    levels = [item[0] for item in sorted_complexity]
    counts = [item[1] for item in sorted_complexity]
    
    # Calculate percentages
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    bars = ax1.bar(levels, counts, color=colors[:len(levels)], edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Query Complexity Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(counts, labels=levels, autopct='%1.1f%%',
                                       colors=colors[:len(levels)], startangle=90,
                                       textprops={'fontsize': 11})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax2.set_title('Query Complexity Distribution (Percentage)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    plt.savefig(output_dir / "complexity_distribution.png", dpi=300, bbox_inches="tight")
    try:
        plt.savefig(output_dir / "complexity_distribution.pdf", bbox_inches="tight")
    except PermissionError:
        print(f"  Warning: Could not save PDF (file may be open)")
    
    plt.close()
    
    print(f"Complexity charts saved:")
    print(f"  - {output_dir / 'complexity_distribution.png'}")
    print(f"  - {output_dir / 'complexity_distribution.pdf'}")

def main():
    dataset_path = Path("datasets/train.jsonl")
    output_dir = Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    print("="*80)
    print("NEO4J DATASET DISTRIBUTION ANALYSIS")
    print("="*80)
    print()
    print(f"Loading dataset: {dataset_path}")
    print()
    
    # Load and analyze
    analysis_results = load_and_analyze_dataset(dataset_path)
    commands = analysis_results["commands"]
    complexity = analysis_results["complexity"]
    
    # Print summary statistics
    print("="*80)
    print("COMMAND TYPE DISTRIBUTION")
    print("="*80)
    total_commands = sum(commands.values())
    for cmd_type, count in sorted(commands.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_commands * 100
        print(f"  {cmd_type:20s}: {count:6,} ({pct:5.2f}%)")
    print()
    
    print("="*80)
    print("COMPLEXITY DISTRIBUTION")
    print("="*80)
    total_complexity = sum(complexity.values())
    order = ["Simple", "Medium", "Complex", "Very Complex"]
    for level in order:
        if level in complexity:
            count = complexity[level]
            pct = count / total_complexity * 100
            print(f"  {level:20s}: {count:6,} ({pct:5.2f}%)")
    print()
    
    # Generate charts
    print("="*80)
    print("GENERATING CHARTS")
    print("="*80)
    print()
    
    plot_command_type_distribution(commands, output_dir)
    plot_complexity_distribution(complexity, output_dir)
    
    print()
    print("="*80)
    print("[OK] Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

