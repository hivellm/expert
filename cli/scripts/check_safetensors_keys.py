#!/usr/bin/env python3
"""
Script to inspect SafeTensors weight names in the Qwen3-0.6B model.
Used to verify that layer names match what the Rust code expects.
"""

import sys
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)

def check_model_keys(model_path: str):
    """Check the keys in a SafeTensors file."""
    safetensors_file = Path(model_path) / "model.safetensors"
    
    if not safetensors_file.exists():
        print(f"ERROR: File not found: {safetensors_file}")
        return
    
    print(f"📦 Inspecting: {safetensors_file}")
    print("=" * 80)
    
    with safe_open(safetensors_file, framework="pt") as f:
        keys = list(f.keys())
        
        # Check critical layers
        print("\n🔍 CRITICAL LAYERS:")
        print("-" * 80)
        
        # Embedding
        embed_keys = [k for k in keys if 'embed' in k.lower()]
        print(f"\n📌 Embeddings ({len(embed_keys)} found):")
        for key in embed_keys:
            shape = f.get_tensor(key).shape
            print(f"  ✓ {key}: {shape}")
        
        # LM Head (CRITICAL)
        lm_head_keys = [k for k in keys if 'lm_head' in k.lower()]
        print(f"\n📌 LM Head ({len(lm_head_keys)} found):")
        if lm_head_keys:
            for key in lm_head_keys:
                shape = f.get_tensor(key).shape
                print(f"  ✓ {key}: {shape}")
        else:
            print("  ⚠️  NO lm_head found! Checking if tied embeddings...")
            # Check config for tie_word_embeddings
            config_file = Path(model_path) / "config.json"
            if config_file.exists():
                import json
                with open(config_file) as cf:
                    config = json.load(cf)
                    tied = config.get("tie_word_embeddings", False)
                    print(f"  → tie_word_embeddings: {tied}")
                    if tied:
                        print("  → LM head should share weights with embed_tokens")
        
        # Norm layers
        norm_keys = [k for k in keys if 'norm' in k and 'model.norm' in k]
        print(f"\n📌 Final Norm ({len(norm_keys)} found):")
        for key in norm_keys[:3]:  # Show first 3
            shape = f.get_tensor(key).shape
            print(f"  ✓ {key}: {shape}")
        
        # Sample layer structure
        print("\n📌 Sample Layer Structure (layer 0):")
        layer_0_keys = [k for k in keys if 'layers.0.' in k]
        for key in sorted(layer_0_keys):
            shape = f.get_tensor(key).shape
            print(f"  ✓ {key}: {shape}")
        
        # Count layers
        layer_count = len([k for k in keys if 'layers.' in k and 'self_attn.q_proj.weight' in k])
        print(f"\n📊 STATISTICS:")
        print(f"  Total keys: {len(keys)}")
        print(f"  Number of layers: {layer_count}")
        
        # Check for potential issues
        print("\n🔍 VALIDATION:")
        issues = []
        
        if not lm_head_keys:
            # Check if it's supposed to be tied
            config_file = Path(model_path) / "config.json"
            if config_file.exists():
                import json
                with open(config_file) as cf:
                    config = json.load(cf)
                    if not config.get("tie_word_embeddings", False):
                        issues.append("❌ lm_head.weight missing and tie_word_embeddings=false")
            else:
                issues.append("⚠️  lm_head.weight missing (cannot verify if tied)")
        
        expected_patterns = [
            "model.embed_tokens.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.down_proj.weight",
            "model.norm.weight",
        ]
        
        for pattern in expected_patterns:
            if '*' in pattern:
                # Check pattern with layer 0
                check_pattern = pattern.replace('*', '0')
                if not any(check_pattern in k for k in keys):
                    issues.append(f"⚠️  Pattern not found: {pattern}")
            else:
                if not any(pattern in k for k in keys):
                    issues.append(f"⚠️  Key not found: {pattern}")
        
        if issues:
            print("\n⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✅ All expected patterns found!")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    check_model_keys(model_path)

