#!/usr/bin/env python3
"""
Quality Benchmarking for Expert Adapters

Compares expert quality across adapter types and configurations.
"""

import json
from pathlib import Path
from typing import Dict, List

def load_manifest(expert_path: Path) -> Dict:
    """Load expert manifest"""
    manifest_path = expert_path / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)

def analyze_expert(expert_path: Path) -> Dict:
    """Analyze a single expert configuration"""
    manifest = load_manifest(expert_path)
    
    # Extract key info
    config = manifest.get("training", {}).get("config", {})
    adapter_type = config.get("adapter_type", "unknown")
    rank = config.get("rank", "N/A")
    alpha = config.get("alpha", "N/A")
    
    # Check for weights
    weights_exist = False
    adapter_size_mb = 0
    
    # Try v1.0 location
    weights_path = expert_path / "weights" / "adapter_model.safetensors"
    if not weights_path.exists():
        # Try v2.0 location
        weights_path = expert_path / "qwen3-06b" / "adapter" / "adapter_model.safetensors"
    
    if weights_path.exists():
        weights_exist = True
        adapter_size_mb = weights_path.stat().st_size / (1024 * 1024)
    
    return {
        "name": manifest.get("name", "unknown"),
        "adapter_type": adapter_type,
        "rank": rank,
        "alpha": alpha,
        "weights_exist": weights_exist,
        "adapter_size_mb": round(adapter_size_mb, 2),
        "soft_prompts": len(manifest.get("soft_prompts", [])),
        "temperature": manifest.get("training", {}).get("decoding", {}).get("temperature", "N/A"),
    }

def main():
    experts_dir = Path("F:/Node/hivellm/expert/experts")
    
    print("="*70)
    print("Expert Quality Benchmarking - v0.2.3")
    print("="*70)
    print()
    
    experts = []
    for expert_path in sorted(experts_dir.glob("expert-*")):
        if expert_path.is_dir():
            try:
                info = analyze_expert(expert_path)
                experts.append(info)
            except Exception as e:
                print(f"[SKIP] {expert_path.name}: {e}")
    
    # Display results
    print(f"{'Expert':<20} {'Adapter':<10} {'Rank':<6} {'Size (MB)':<10} {'Soft':<5} {'Temp':<6} {'Trained':<8}")
    print("-"*70)
    
    for expert in experts:
        trained = "YES" if expert["weights_exist"] else "NO"
        size_str = f"{expert['adapter_size_mb']:.2f}" if expert["weights_exist"] else "-"
        
        print(f"{expert['name']:<20} "
              f"{expert['adapter_type']:<10} "
              f"{str(expert['rank']):<6} "
              f"{size_str:<10} "
              f"{expert['soft_prompts']:<5} "
              f"{str(expert['temperature']):<6} "
              f"{trained:<8}")
    
    print()
    print("="*70)
    print("Summary")
    print("="*70)
    
    # Count by adapter type
    adapter_counts = {}
    for expert in experts:
        adapter_type = expert["adapter_type"]
        adapter_counts[adapter_type] = adapter_counts.get(adapter_type, 0) + 1
    
    print("\nAdapter Type Distribution:")
    for adapter, count in sorted(adapter_counts.items()):
        print(f"  {adapter}: {count} experts")
    
    # Count trained
    trained_count = sum(1 for e in experts if e["weights_exist"])
    print(f"\nTrained: {trained_count}/{len(experts)} experts")
    
    # Total size
    total_size = sum(e["adapter_size_mb"] for e in experts if e["weights_exist"])
    print(f"Total adapter size: {total_size:.2f} MB")
    
    # Soft prompts
    total_soft_prompts = sum(e["soft_prompts"] for e in experts)
    print(f"Total soft prompts: {total_soft_prompts}")
    
    print()
    
    # Expected performance (from manifests)
    print("Expected Performance (from manifests):")
    print("-"*70)
    
    perf_data = []
    for expert_path in sorted(experts_dir.glob("expert-*")):
        try:
            manifest = load_manifest(expert_path)
            perf = manifest.get("perf", {})
            if perf:
                perf_data.append({
                    "name": manifest.get("name", "unknown"),
                    "vram_mb": perf.get("vram_mb_overhead", 0),
                    "latency_ms": perf.get("latency_ms_overhead", 0),
                })
        except:
            pass
    
    if perf_data:
        print(f"{'Expert':<20} {'VRAM (MB)':<12} {'Latency (ms)':<15}")
        print("-"*70)
        for p in perf_data:
            print(f"{p['name']:<20} {p['vram_mb']:<12} {p['latency_ms']:<15}")
        
        total_vram = sum(p["vram_mb"] for p in perf_data)
        print()
        print(f"Total VRAM overhead: {total_vram} MB (all experts loaded)")
    
    print()
    print("="*70)
    print("[OK] Benchmark complete")
    print("="*70)

if __name__ == "__main__":
    main()

