#!/usr/bin/env python3
"""
Expert Training Script - PyTorch/PEFT Integration
Called from Rust via PyO3

This module now imports from the refactored train/ module.
Maintains backward compatibility with existing Rust code.
"""

# Disable torch.compile on Windows (Triton incompatible)
import os
import platform
from pathlib import Path

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# CRITICAL: Windows stability configurations (must be set before any CUDA operations)
if platform.system() == "Windows":
    # CUDA memory allocation optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,backend:cudaMallocAsync"
    
    # For debugging (can be removed in production for better performance)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Avoid conflicts with Intel MKL
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Centralize Unsloth compiled cache (avoid duplicating in each expert)
# This cache stores compiled CUDA/Triton extensions (~5-10MB per expert)
CENTRALIZED_CACHE = Path(__file__).parent.parent / "cache" / "unsloth_compiled"
CENTRALIZED_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = str(CENTRALIZED_CACHE)
print(f"[CACHE] Unsloth compiled cache: {CENTRALIZED_CACHE}")

# Import from refactored train module
from train import train_expert
import json
import sys

# Re-export for backward compatibility
__all__ = ["train_expert"]

# When called as script (from Rust subprocess), read config JSON file
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Config JSON file path required as argument", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Call training function
        train_expert(config_dict)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
