import torch

print("="*60)
print("CUDA CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA NOT AVAILABLE!")
    print("\nPossible issues:")
    print("  - PyTorch not compiled with CUDA")
    print("  - CUDA drivers not installed")
    print("  - CUDA version mismatch")

print("\nTrying to import Unsloth...")
try:
    from unsloth import FastLanguageModel
    print("✓ Unsloth imported successfully!")
except Exception as e:
    print(f"✗ Unsloth import failed: {e}")

