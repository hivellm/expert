#!/usr/bin/env python3
"""
Compare Python vs Rust Qwen3 outputs
Identifies where the implementations diverge
"""

import json
import sys

def compare_outputs(python_file: str, rust_file: str):
    """Compare debug outputs from Python and Rust"""
    
    with open(python_file) as f:
        py_data = json.load(f)
    
    with open(rust_file) as f:
        rust_data = json.load(f)
    
    print("="*70)
    print("Qwen3 Python vs Rust Comparison")
    print("="*70)
    
    # Compare inputs
    print("\n1. INPUT TOKENS:")
    py_tokens = py_data["input"].get("tokens") or py_data["input"].get("input_ids")
    rust_tokens = rust_data["input"].get("tokens") or rust_data["input"].get("input_ids")
    
    if py_tokens == rust_tokens:
        print("  [OK] MATCH - Tokens identical")
    else:
        print(f"  [FAIL] MISMATCH")
        print(f"     Python:  {py_tokens}")
        print(f"     Rust:    {rust_tokens}")
        return
    
    # Compare logits
    print("\n2. FINAL LOGITS:")
    py_logits = py_data.get("logits") or py_data["final"]["logits"]
    rust_logits = rust_data["logits"]
    
    print(f"  Shape:")
    print(f"     Python:  {py_logits['shape']}")
    print(f"     Rust:    {rust_logits['shape']}")
    
    print(f"  Mean:")
    print(f"     Python:  {py_logits['mean']:.6f}")
    print(f"     Rust:    {rust_logits['mean']:.6f}")
    print(f"     Diff:    {abs(py_logits['mean'] - rust_logits['mean']):.6f}")
    
    print(f"  Min:")
    print(f"     Python:  {py_logits['min']:.6f}")
    print(f"     Rust:    {rust_logits['min']:.6f}")
    
    print(f"  Max:")
    print(f"     Python:  {py_logits['max']:.6f}")
    print(f"     Rust:    {rust_logits['max']:.6f}")
    
    print(f"  Argmax token:")
    print(f"     Python:  {py_logits['argmax']}")
    print(f"     Rust:    {rust_logits['argmax']}")
    
    if py_logits['argmax'] == rust_logits['argmax']:
        print("     [OK] MATCH - Same predicted token!")
    else:
        print("     [FAIL] MISMATCH - Different predictions")
    
    print(f"  Top 5 tokens:")
    py_top5 = py_logits.get('top_5_tokens') or py_logits.get('top_5', [])
    rust_top5 = rust_logits.get('top_5', [])
    print(f"     Python:  {py_top5}")
    print(f"     Rust:    {rust_top5}")
    
    matches = sum(1 for p, r in zip(py_top5, rust_top5) if p == r)
    print(f"     Overlap: {matches}/5")
    
    print(f"\n  Predicted text:")
    py_pred = py_logits.get('predicted_text', 'N/A')
    rust_pred = rust_logits.get('predicted_text', 'N/A')
    print(f"     Python:  '{py_pred}'")
    print(f"     Rust:    '{rust_pred}'")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    
    if py_logits['argmax'] == rust_logits['argmax']:
        print("  [OK] Outputs MATCH - Rust implementation correct!")
    else:
        mean_diff = abs(py_logits['mean'] - rust_logits['mean'])
        if mean_diff < 0.01:
            print("  [WARN] Argmax differs but logits are close (may be sampling variance)")
        else:
            print("  [FAIL] Outputs DIVERGE - Rust implementation has bugs")
            print(f"     Mean difference: {mean_diff:.6f}")
    
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_outputs.py <python.json> <rust.json>")
        sys.exit(1)
    
    compare_outputs(sys.argv[1], sys.argv[2])

