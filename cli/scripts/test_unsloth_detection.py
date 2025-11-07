#!/usr/bin/env python3
"""
Test script to verify Unsloth detection
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("UNSLOTH DETECTION TEST")
print("=" * 80)

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("[OK] Unsloth is available")
    print("     FastLanguageModel:", FastLanguageModel)
except ImportError as e:
    USE_UNSLOTH = False
    print(f"[INFO] Unsloth not available: {e}")

print(f"\nUSE_UNSLOTH = {USE_UNSLOTH}")

# Try to import PyTorch and check CUDA
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 80)
if USE_UNSLOTH and torch.cuda.is_available():
    print("STATUS: Ready for 2x faster training with Unsloth!")
elif torch.cuda.is_available():
    print("STATUS: CUDA ready, but Unsloth not installed (will use standard PyTorch)")
else:
    print("STATUS: No CUDA detected")
print("=" * 80)

