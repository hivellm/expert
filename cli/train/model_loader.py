"""
Model Loading Module

Handles loading of base models and tokenizers with quantization and optimizations.
"""

import os
import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Import Unsloth if available (must be imported before transformers)
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False

from .config import TrainingConfig


def get_quantization_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """Get BitsAndBytes quantization config"""
    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def load_model_and_tokenizer(config: TrainingConfig) -> Tuple:
    """Load base model with quantization and optimizations (Unsloth if available)"""
    print(f"\nLoading model: {config.base_model_name}")
    print(f"   Quantization: {config.quantization}")
    
    # Check if Unsloth should be used (manifest flag + availability)
    use_unsloth_flag = getattr(config, 'use_unsloth', None)
    
    # Use Unsloth if: flag enabled in manifest AND library available AND quantization compatible
    if use_unsloth_flag and USE_UNSLOTH and config.quantization in ["int4", "int8"]:
        print(f"   [UNSLOTH] Enabled via manifest (use_unsloth: true)")
        print(f"   [UNSLOTH] Using FastLanguageModel (2x faster, 70% less VRAM)")
        return load_model_with_unsloth(config)
    else:
        # Explain why not using Unsloth
        if use_unsloth_flag and not USE_UNSLOTH:
            print(f"   [WARNING] use_unsloth=true in manifest, but Unsloth not installed")
            print(f"   [INFO] Install with: pip install 'unsloth[windows] @ git+https://github.com/unslothai/unsloth.git'")
        elif use_unsloth_flag and config.quantization not in ["int4", "int8"]:
            print(f"   [INFO] Unsloth requires int4/int8 quantization (current: {config.quantization})")
        elif not use_unsloth_flag:
            print(f"   [INFO] Unsloth disabled in manifest (use_unsloth: false)")
        
        print(f"   [STANDARD] Using PyTorch/Transformers")
        print(f"   [INFO] Applying Unsloth-inspired parameters (LR 5e-5, temp 0.7, dropout 0.1)")
        return load_model_standard(config)


def load_model_with_unsloth(config: TrainingConfig) -> Tuple:
    """Load model using Unsloth optimizations"""
    max_seq_length = config.max_seq_length or 2048
    
    # Unsloth automatically handles quantization
    load_in_4bit = (config.quantization == "int4")
    load_in_8bit = (config.quantization == "int8")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect (will use bfloat16)
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
    )
    
    print(f"   [OK] Model loaded with Unsloth optimizations")
    print(f"   [OK] Max sequence length: {max_seq_length}")
    
    return model, tokenizer


def load_model_standard(config: TrainingConfig) -> Tuple:
    """Load model using standard transformers (fallback)"""
    
    # Enable fast kernels if requested
    if config.use_tf32 and config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("   [OK] TF32 enabled for matrix operations")
    
    quantization_config = get_quantization_config(config.quantization)
    
    # Check if model path is local or HuggingFace repo
    is_local_path = os.path.exists(config.base_model_name) or (
        os.path.sep in config.base_model_name or '/' in config.base_model_name
    )
    
    print(f"   Model source: {'Local' if is_local_path else 'HuggingFace Hub'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        local_files_only=is_local_path,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype
    use_bf16 = config.bf16 if config.bf16 is not None else (config.device == "cuda")
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    if use_bf16:
        print("   [OK] Using BF16 dtype")
    else:
        print("   [OK] Using FP16 dtype")
    
    # Build model kwargs
    # With QLoRA (bitsandbytes), model MUST be loaded with device_map
    # Without quantization, we'll move manually
    if quantization_config is not None:
        # QLoRA requires device_map for proper GPU placement
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "cuda:0" if config.device == "cuda" else "cpu",
            "trust_remote_code": True,
            "local_files_only": is_local_path,
            "low_cpu_mem_usage": True,
        }
    else:
        # Without quantization, load normally and move manually
        model_kwargs = {
            "device_map": None,
            "trust_remote_code": True,
            "local_files_only": is_local_path,
            "low_cpu_mem_usage": True,
            "dtype": model_dtype,
        }
    
    # Set matmul precision for TF32 (Qwen3 optimization)
    if config.use_tf32:
        torch.set_float32_matmul_precision('high')
        print("   [OK] TF32 matmul precision set to 'high'")
    
    # Add Flash Attention if requested
    if config.flash_attention_2 and config.device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("   [OK] Flash Attention 2 enabled (30% faster than SDPA on Ampere+)")
    elif config.use_sdpa and config.device == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"
        if quantization_config is not None:
            print("   [OK] SDPA Flash Attention enabled (with QLoRA INT4)")
        else:
            print("   [OK] SDPA Flash Attention enabled")
    
    # Load model (standard transformers)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        **model_kwargs
    )
    
    # Move to GPU if not quantized (quantized models are already on GPU via device_map)
    if config.device == "cuda" and torch.cuda.is_available() and quantization_config is None:
        model = model.to("cuda")
        print("   [OK] Model moved to CUDA")
    elif quantization_config is not None:
        print("   [OK] Model loaded on CUDA (via QLoRA device_map)")
    
    # Prepare for k-bit training if quantized
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

