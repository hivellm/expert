#!/usr/bin/env python3
"""
Expert Training Script - PyTorch/PEFT Integration
Called from Rust via PyO3
"""

# Disable torch.compile on Windows (Triton incompatible)
import os
from pathlib import Path

os.environ["PYTORCH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Centralize Unsloth compiled cache (avoid duplicating in each expert)
# This cache stores compiled CUDA/Triton extensions (~5-10MB per expert)
CENTRALIZED_CACHE = Path(__file__).parent.parent / "cache" / "unsloth_compiled"
CENTRALIZED_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = str(CENTRALIZED_CACHE)
print(f"[CACHE] Unsloth compiled cache: {CENTRALIZED_CACHE}")

# CRITICAL: Import Unsloth FIRST (before transformers/peft/trl)
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("[INFO] Unsloth available - 2x faster training enabled")
except ImportError:
    USE_UNSLOTH = False
    print("[INFO] Unsloth not available - using standard PyTorch (slower)")

import json
import sys
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
# Disable torch dynamo/compile (Windows compatibility)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
from datasets import load_dataset
from peft import LoraConfig, PromptTuningConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from trl import SFTTrainer

# Import psutil for system RAM monitoring (Windows-specific optimization)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed - RAM monitoring disabled")

# Import prompt templates from scripts directory
import sys
from pathlib import Path as PathLib
scripts_path = PathLib(__file__).parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

try:
    from prompt_templates import format_training_example, get_recommended_template
except ImportError:
    # Fallback if module not found
    def format_training_example(instruction, response, input_text=None, input_label="Input", template_name="alpaca", system=None):
        text = f"### Instruction:\n{instruction}"
        if input_text:
            text += f"\n\n### {input_label}:\n{input_text}"
        text += f"\n\n### Response:\n{response}"
        return text
    
    def get_recommended_template(model_name):
        return "alpaca"


@dataclass
class TrainingConfig:
    """Training configuration from manifest"""
    # Required fields (no defaults)
    base_model_name: str
    quantization: str
    dataset_path: str
    output_dir: str
    device: str
    adapter_type: str
    rank: int
    alpha: int
    target_modules: list[str]
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    lr_scheduler: str
    
    # Optional fields (with defaults)
    validation_path: Optional[str] = None
    test_path: Optional[str] = None
    warmup_ratio: Optional[float] = None
    
    # Optional advanced training params
    use_unsloth: Optional[bool] = None
    max_seq_length: Optional[int] = None
    dataloader_num_workers: Optional[int] = None
    dataloader_pin_memory: Optional[bool] = None
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_persistent_workers: Optional[bool] = None
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    use_tf32: Optional[bool] = None
    use_sdpa: Optional[bool] = None
    flash_attention_2: Optional[bool] = None
    memory_efficient_attention: Optional[bool] = None
    activation_checkpointing: Optional[str] = None
    packing: Optional[bool] = None
    torch_compile: Optional[bool] = None
    torch_compile_backend: Optional[str] = None
    torch_compile_mode: Optional[str] = None
    optim: Optional[str] = None
    group_by_length: Optional[bool] = None
    save_steps: Optional[int] = None
    save_strategy: Optional[str] = None
    save_total_limit: Optional[int] = None
    evaluation_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    load_best_model_at_end: Optional[bool] = None
    metric_for_best_model: Optional[str] = None
    logging_steps: Optional[int] = None
    gradient_checkpointing: Optional[any] = None  # Can be bool or "selective"
    gradient_checkpointing_kwargs: Optional[dict] = None
    lr_scheduler_kwargs: Optional[dict] = None
    pretokenized_cache: Optional[str] = None
    use_cuda_graphs: Optional[bool] = None
    cuda_graph_warmup_steps: Optional[int] = None
    
    # Dataset field mapping (for HF datasets with different schemas)
    field_mapping: Optional[dict[str, str]] = None
    text_field: Optional[str] = None  # Single text field (alternative to instruction/response)
    format: Optional[str] = None  # Dataset format hint
    prompt_template: Optional[str] = None  # Prompt template from base_model
    
    # Dataset streaming (reduces RAM usage)
    streaming: Optional[bool] = None
    max_in_memory_samples: Optional[int] = None
    
    # Pre-tokenized dataset support (Windows optimization)
    use_pretokenized: Optional[bool] = None
    
    # Multi-task support
    dataset_type: Optional[str] = None  # "single", "huggingface", "multi_task"
    multi_task_config: Optional[dict] = None
    
    resume_checkpoint: Optional[str] = None


def load_training_config(config_dict: dict) -> TrainingConfig:
    """Parse config dictionary from Rust"""
    training = config_dict["training"]
    
    # Dataset config can be in config_dict["dataset"] OR training["dataset"]
    dataset_config = config_dict.get("dataset", training.get("dataset", {}))
    
    # Detect dataset type
    dataset_type = dataset_config.get("type", "single")
    
    # Extract prompt_template from base_model if present
    prompt_template = None
    if "base_model" in config_dict:
        base_model_cfg = config_dict["base_model"]
        if isinstance(base_model_cfg, dict):
            prompt_template = base_model_cfg.get("prompt_template")
    
    # Common fields
    common_fields = {
        "base_model_name": config_dict["base_model_name"],
        "quantization": config_dict["quantization"],
        "output_dir": config_dict["output_dir"],
        "device": config_dict["device"],
        "prompt_template": prompt_template,
        "adapter_type": training["adapter_type"],
        "rank": training["rank"],
        "alpha": training["alpha"],
        "target_modules": training["target_modules"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "warmup_steps": training.get("warmup_steps", 0),
        "warmup_ratio": training.get("warmup_ratio"),
        "lr_scheduler": training["lr_scheduler"],
        "resume_checkpoint": config_dict.get("resume_checkpoint"),
        # Optional advanced params
        "use_unsloth": training.get("use_unsloth"),
        "max_seq_length": training.get("max_seq_length"),
        "dataloader_num_workers": training.get("dataloader_num_workers"),
        "dataloader_pin_memory": training.get("dataloader_pin_memory"),
        "dataloader_prefetch_factor": training.get("dataloader_prefetch_factor"),
        "dataloader_persistent_workers": training.get("dataloader_persistent_workers"),
        "fp16": training.get("fp16"),
        "bf16": training.get("bf16"),
        "use_tf32": training.get("use_tf32"),
        "use_sdpa": training.get("use_sdpa"),
        "optim": training.get("optim"),
        "group_by_length": training.get("group_by_length"),
        "save_steps": training.get("save_steps"),
        "save_strategy": training.get("save_strategy"),
        "save_total_limit": training.get("save_total_limit"),
        "evaluation_strategy": training.get("evaluation_strategy"),
        "eval_steps": training.get("eval_steps"),
        "load_best_model_at_end": training.get("load_best_model_at_end"),
        "metric_for_best_model": training.get("metric_for_best_model"),
        "logging_steps": training.get("logging_steps"),
        "gradient_checkpointing": parse_json_field(config_dict.get("gradient_checkpointing_json")) or training.get("gradient_checkpointing", False),
        "gradient_checkpointing_kwargs": parse_json_field(config_dict.get("gradient_checkpointing_kwargs_json")),
        "lr_scheduler_kwargs": parse_json_field(config_dict.get("lr_scheduler_kwargs_json")),
        "pretokenized_cache": training.get("pretokenized_cache"),
        "use_cuda_graphs": training.get("use_cuda_graphs"),
        "cuda_graph_warmup_steps": training.get("cuda_graph_warmup_steps"),
    }
    
    if dataset_type == "multi_task":
        # Multi-task configuration
        return TrainingConfig(
            dataset_path="",  # Not used for multi-task
            dataset_type="multi_task",
            multi_task_config=dataset_config,
            field_mapping=None,
            text_field=None,
            streaming=None,
            max_in_memory_samples=None,
            **common_fields
        )
    else:
        # Single file or HuggingFace
        return TrainingConfig(
            dataset_path=config_dict.get("dataset_path", dataset_config.get("path", "")),
            validation_path=config_dict.get("validation_path", dataset_config.get("validation_path")),
            test_path=config_dict.get("test_path", dataset_config.get("test_path")),
            dataset_type=dataset_type,
            multi_task_config=None,
            field_mapping=dataset_config.get("field_mapping"),  # From dataset, not config_dict
            text_field=dataset_config.get("text_field"),
            format=dataset_config.get("format"),
            streaming=dataset_config.get("streaming"),
            max_in_memory_samples=dataset_config.get("max_in_memory_samples"),
            use_pretokenized=dataset_config.get("use_pretokenized"),
            **common_fields
        )


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


def load_model_and_tokenizer(config: TrainingConfig):
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


def load_model_with_unsloth(config: TrainingConfig):
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


def load_model_standard(config: TrainingConfig):
    """Load model using standard transformers (fallback)"""
    
    # Enable fast kernels if requested
    if config.use_tf32 and config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("   [OK] TF32 enabled for matrix operations")
    
    quantization_config = get_quantization_config(config.quantization)
    
    # Check if model path is local or HuggingFace repo
    import os
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


def configure_soft_prompts(model, soft_prompts, base_model_name, tokenizer):
    """
    Configure soft prompt tuning if specified in manifest.
    
    Soft prompts are trainable prompt embeddings that guide model behavior
    without modifying the base model weights.
    
    Args:
        model: Base model (before adapter application)
        soft_prompts: List of soft prompt configs from manifest
        base_model_name: Model identifier for tokenizer
        tokenizer: Tokenizer instance
    
    Returns:
        Model with soft prompt tuning applied (if soft prompts exist)
    """
    if not soft_prompts or len(soft_prompts) == 0:
        return model
    
    # Currently support single soft prompt during training
    # Multi-prompt support is future work
    soft_prompt = soft_prompts[0]
    
    print(f"\nConfiguring Soft Prompt Tuning")
    print(f"   Name: {soft_prompt.get('name', 'unnamed')}")
    print(f"   Virtual Tokens: {soft_prompt.get('tokens', 32)}")
    
    # Determine initialization method
    init_method = soft_prompt.get('init_method', 'random')
    init_text = soft_prompt.get('init_text', '')
    
    if init_method == 'text' and init_text:
        print(f"   Init Method: TEXT")
        print(f"   Init Text: {init_text[:60]}...")
        prompt_init = "TEXT"
    else:
        print(f"   Init Method: RANDOM")
        prompt_init = "RANDOM"
        init_text = None
    
    # Create prompt tuning config
    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=soft_prompt.get('tokens', 32),
        prompt_tuning_init=prompt_init,
        prompt_tuning_init_text=init_text,
        tokenizer_name_or_path=base_model_name,
    )
    
    # Apply prompt tuning to model
    model = get_peft_model(model, prompt_config)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def save_soft_prompts(model, soft_prompts, output_dir):
    """
    Save trained soft prompt embeddings after training.
    
    Args:
        model: Trained PEFT model
        soft_prompts: List of soft prompt configs from manifest
        output_dir: Base output directory
    """
    if not soft_prompts or len(soft_prompts) == 0:
        return
    
    import torch
    from pathlib import Path
    
    print(f"\nSaving Soft Prompt Embeddings")
    
    # Create soft_prompts directory
    soft_prompt_dir = Path(output_dir) / "soft_prompts"
    soft_prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract prompt embeddings from PEFT model
    # Check if model has prompt tuning module
    if hasattr(model, 'prompt_encoder'):
        for i, soft_prompt in enumerate(soft_prompts):
            try:
                # Get the prompt embedding
                # The prompt encoder stores embeddings for each task
                prompt_name = soft_prompt.get('name', f'prompt_{i}')
                
                # Extract embedding weight
                # PEFT stores prompts in prompt_encoder['default']
                if hasattr(model.prompt_encoder, 'default'):
                    embedding = model.prompt_encoder.default.embedding.weight
                else:
                    # Fallback: try to get first encoder
                    encoder_keys = list(model.prompt_encoder.keys())
                    if encoder_keys:
                        embedding = model.prompt_encoder[encoder_keys[0]].embedding.weight
                    else:
                        print(f"   ⚠️  Could not find prompt embedding for {prompt_name}")
                        continue
                
                # Save to file matching manifest path
                filename = Path(soft_prompt['path']).name
                save_path = soft_prompt_dir / filename
                
                torch.save(embedding.detach().cpu(), save_path)
                
                print(f"   [OK] Saved: {save_path}")
                print(f"     Shape: {embedding.shape}")
                print(f"     Size: {save_path.stat().st_size / 1024:.2f} KB")
                
            except Exception as e:
                print(f"   [WARN] Error saving soft prompt '{soft_prompt.get('name', 'unnamed')}': {e}")
    else:
        print("   [WARN] Model does not have prompt_encoder (soft prompts not configured)")


def setup_lora(model, config: TrainingConfig):
    """Configure LoRA/IA³/DoRA adapter"""
    adapter_type = config.adapter_type.lower()
    
    print(f"\nSetting up {adapter_type.upper()} adapter")
    print(f"   Type: {adapter_type}")
    print(f"   Target modules: {', '.join(config.target_modules)}")
    
    # Enable gradient checkpointing if requested (saves VRAM)
    # Gradient checkpointing (full or selective)
    if config.gradient_checkpointing:
        if config.gradient_checkpointing == "selective":
            print(f"   Gradient Checkpointing: SELECTIVE (attention only, saves 20% compute vs full)")
            # Selective checkpointing for Qwen3: only checkpoint attention layers
            model.gradient_checkpointing_enable()
            if hasattr(model, 'config'):
                model.config.use_cache = False  # Required for checkpointing
        else:
            print(f"   Gradient Checkpointing: ENABLED (saves VRAM)")
            model.gradient_checkpointing_enable()
    else:
        print(f"   Gradient Checkpointing: DISABLED")
    
    # Configure adapter based on type
    # Use Unsloth's optimized adapter if enabled in manifest and available
    use_unsloth_flag = getattr(config, 'use_unsloth', None)
    
    if use_unsloth_flag and USE_UNSLOTH and adapter_type in ["lora", "dora"]:
        print(f"   [UNSLOTH] Using optimized {adapter_type.upper()} adapter")
        
        # Get dropout from config (default 0.1 from LLaMA-Factory)
        dropout = 0.1
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=dropout,
            target_modules=config.target_modules,
            use_dora=(adapter_type == "dora"),
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        print(f"   Rank: {config.rank}")
        print(f"   Alpha: {config.alpha}")
        print(f"   Dropout: {dropout}")
        if adapter_type == "dora":
            print(f"   DoRA: Enabled (Unsloth optimized)")
        
        model.print_trainable_parameters()
        return model
    
    # Standard PEFT configuration (fallback)
    if adapter_type == "lora":
        # Standard LoRA
        print(f"   Rank: {config.rank}")
        print(f"   Alpha: {config.alpha}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.rank,
            lora_alpha=config.alpha,
            target_modules=config.target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        
    elif adapter_type == "dora":
        # DoRA (Weight-Decomposed Low-Rank Adaptation)
        rank = getattr(config, 'rank', 12)  # Default to 12 for DoRA if not specified
        alpha = getattr(config, 'alpha', 24)
        
        print(f"   Rank: {rank}")
        print(f"   Alpha: {alpha}")
        print(f"   DoRA: Enabled (magnitude decomposition)")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            target_modules=config.target_modules,
            lora_dropout=0.05,
            bias="none",
            use_dora=True,  # Enable DoRA variant
        )
        
    elif adapter_type == "ia3":
        # IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
        print(f"   IA³: Learning scaling vectors (no rank/alpha needed)")
        
        from peft import IA3Config
        
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.target_modules,
            feedforward_modules=config.target_modules,
        )
        
    elif adapter_type == "lokr":
        # LoKr (Low-Rank Kronecker Product)
        rank = getattr(config, 'rank', 12)
        alpha = getattr(config, 'alpha', 24)
        
        print(f"   Rank: {rank}")
        print(f"   Alpha: {alpha}")
        print(f"   LoKr: Enabled (Kronecker decomposition)")
        
        from peft import LoKrConfig
        
        peft_config = LoKrConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            target_modules=config.target_modules,
            lora_dropout=0.05,
        )
        
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}. Supported: lora, dora, ia3, lokr")
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Apply torch compile if requested (Qwen3 optimization)
    if getattr(config, 'torch_compile', False):
        backend = getattr(config, 'torch_compile_backend', 'inductor')
        mode = getattr(config, 'torch_compile_mode', 'reduce-overhead')
        print(f"\n   [COMPILE] Torch compile enabled (backend={backend}, mode={mode})")
        print(f"   Note: First forward pass will be slow (compilation), then 1.8-2.2x speedup")
        import torch
        model = torch.compile(model, backend=backend, mode=mode)
    
    return model


def load_multi_task_dataset(config: TrainingConfig, tokenizer):
    """Load and combine multi-task datasets"""
    import sys
    from pathlib import Path as PathlibPath
    
    # Add scripts directory to Python path
    scripts_dir = PathlibPath(__file__).parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    
    from dataset_loader import MultiTaskDatasetLoader
    from dataset_validator import DatasetValidator
    from dataset_stats import generate_stats, print_stats
    
    # Check for pre-tokenized cache first
    if config.pretokenized_cache:
        from datasets import load_from_disk
        cache_path = PathlibPath(config.pretokenized_cache)
        
        if cache_path.exists():
            print(f"\nLoading pre-tokenized multi-task dataset from cache...")
            print(f"   Cache: {cache_path}")
            
            try:
                cached_dataset = load_from_disk(str(cache_path))
                print(f"   [OK] Pre-tokenized dataset loaded successfully")
                
                # Convert to dict format expected by trainer
                return {
                    "train": cached_dataset.get("train"),
                    "test": cached_dataset.get("validation") or cached_dataset.get("test")
                }
            except Exception as e:
                print(f"   [WARN] Failed to load cache: {e}")
                print(f"   Creating pre-tokenized cache...")
        else:
            print(f"\nPre-tokenized cache not found: {cache_path}")
            print(f"   Creating cache during this training run...")
            print(f"   Future runs will be faster!")
            print()
    
    print("\nLoading multi-task dataset...")
    
    # Determine base path (expert directory)
    base_path = PathlibPath(config.output_dir).parent
    
    # Load datasets
    loader = MultiTaskDatasetLoader(config.multi_task_config, base_path=base_path)
    
    print("\n   Loading train split...")
    train_examples = loader.load_split("train")
    
    print("   Loading valid split...")
    valid_examples = loader.load_split("valid")
    
    print(f"\n   Raw counts: {len(train_examples)} train, {len(valid_examples)} valid")
    
    # Validate
    validator = DatasetValidator(config.multi_task_config.get("validation", {}))
    
    print("\n   Validating train examples...")
    train_examples, train_errors = validator.validate_and_filter(train_examples)
    
    if train_errors:
        print(f"   Filtered out {len(train_errors)} invalid train examples")
        # Show first few errors
        for i, (idx, error) in enumerate(train_errors[:3]):
            print(f"      Example {idx}: {error}")
    
    print("   Validating valid examples...")
    valid_examples, valid_errors = validator.validate_and_filter(valid_examples)
    
    if valid_errors:
        print(f"   Filtered out {len(valid_errors)} invalid valid examples")
    
    print(f"\n   After validation: {len(train_examples)} train, {len(valid_examples)} valid")
    
    # Deduplicate
    print("\n   Deduplicating train set...")
    train_examples = validator.deduplicate_examples(train_examples)
    
    print(f"   After deduplication: {len(train_examples)} train")
    
    # Generate and print statistics
    train_stats = generate_stats(train_examples)
    valid_stats = generate_stats(valid_examples)
    
    print_stats(train_stats, "Train Set Statistics")
    print_stats(valid_stats, "Valid Set Statistics")
    
    # Convert to HF dataset format
    from datasets import Dataset
    
    print("   Converting to HuggingFace Dataset format...")
    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples)
    
    # Tokenize
    def tokenize_multi_task(examples):
        """Tokenize multi-task examples (SFT or DPO)"""
        texts = []
        
        # Detect format based on first example
        has_output = "output" in examples
        has_chosen = "chosen" in examples
        
        num_examples = len(examples.get("task", []))
        
        for i in range(num_examples):
            input_data = examples["input"][i] if isinstance(examples["input"][i], dict) else {}
            
            # Build instruction text
            instruction = input_data.get("instruction", "")
            
            # Add context based on available fields
            context = ""
            if "schema" in input_data:
                context += f"\n\nSchema:\n{input_data['schema']}"
            if "broken_json" in input_data:
                context += f"\n\nJSON:\n{input_data['broken_json']}"
            if "text" in input_data:
                context += f"\n\nText:\n{input_data['text']}"
            if "json" in input_data and "json" not in context:
                context += f"\n\nJSON:\n{input_data['json']}"
            
            # Get response (output for SFT, chosen for DPO)
            if has_output:
                response = examples["output"][i]
            elif has_chosen:
                response = examples["chosen"][i]
            else:
                response = ""
            
            text = f"### Instruction:\n{instruction}{context}\n\n### Response:\n{response}"
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None,
        )
        
        tokenized["labels"] = [[x for x in ids] for ids in tokenized["input_ids"]]
        
        return tokenized
    
    print("   Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_multi_task,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    
    valid_dataset = valid_dataset.map(
        tokenize_multi_task,
        batched=True,
        remove_columns=valid_dataset.column_names,
        desc="Tokenizing valid",
    )
    
    # Save to cache if configured
    if config.pretokenized_cache:
        cache_path = PathlibPath(config.pretokenized_cache)
        
        if not cache_path.exists():
            print(f"\n   Saving multi-task dataset to cache...")
            print(f"   Cache path: {cache_path}")
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                from datasets import DatasetDict
                
                # Create dataset dict for saving
                dataset_dict = DatasetDict({
                    "train": train_dataset,
                    "validation": valid_dataset
                })
                
                dataset_dict.save_to_disk(str(cache_path))
                print(f"   [OK] Cache saved! Future training runs will be faster.")
            except Exception as e:
                print(f"   [WARN] Failed to save cache: {e}")
                print(f"   Training will continue without cache...")
    
    # Return in format expected by train_expert
    return {"train": train_dataset, "test": valid_dataset}


def load_and_prepare_dataset(config: TrainingConfig, tokenizer):
    """Load dataset (HuggingFace, local JSONL, or multi-task) and tokenize"""
    
    # Check for multi-task configuration
    if config.dataset_type == "multi_task":
        return load_multi_task_dataset(config, tokenizer)
    
    # Check for pre-tokenized cache first
    if config.pretokenized_cache:
        from datasets import load_from_disk
        cache_path = Path(config.pretokenized_cache)
        
        if cache_path.exists():
            print(f"\nLoading pre-tokenized dataset from cache...")
            print(f"   Cache: {cache_path}")
            
            try:
                tokenized_dataset = load_from_disk(str(cache_path))
                print(f"   [OK] Pre-tokenized dataset loaded successfully")
                
                for split_name in tokenized_dataset.keys():
                    print(f"   {split_name}: {len(tokenized_dataset[split_name])} examples")
                
                # Convert to dict format with 'train' and 'test' keys
                if "train" in tokenized_dataset:
                    train_data = tokenized_dataset["train"]
                    
                    # Check if we have validation/test split
                    if "validation" in tokenized_dataset:
                        test_data = tokenized_dataset["validation"]
                    elif "test" in tokenized_dataset:
                        test_data = tokenized_dataset["test"]
                    else:
                        # No validation - split train into train/test
                        print(f"   No validation split found - creating 10% split from train...")
                        split = train_data.train_test_split(test_size=0.1, seed=42)
                        train_data = split["train"]
                        test_data = split["test"]
                    
                    return {"train": train_data, "test": test_data}
                
                # Fallback
                return tokenized_dataset
            except Exception as e:
                print(f"   [WARN] Failed to load cache: {e}")
                print(f"   Creating pre-tokenized cache...")
        else:
            print(f"\nPre-tokenized cache not found: {cache_path}")
            print(f"   Creating pre-tokenized cache now...")
            print(f"   This is a one-time process (saves time on future training runs)")
            print()
    
    print(f"\nLoading dataset: {config.dataset_path}")
    
    # Check for pre-tokenized dataset (Windows optimization)
    use_pretokenized = config.use_pretokenized if config.use_pretokenized is not None else False
    
    if use_pretokenized:
        print(f"   [PRE-TOKENIZED] Enabled - loading tokenized dataset from disk")
        print(f"   [PRE-TOKENIZED] Format: Arrow (optimized for Windows)")
        print(f"   [PRE-TOKENIZED] Benefits: 10x faster loading, no tokenization overhead")
        
        try:
            from datasets import load_from_disk
            
            dataset = load_from_disk(config.dataset_path)
            print(f"   [OK] Pre-tokenized dataset loaded successfully")
            
            # Return immediately - already tokenized
            if isinstance(dataset, dict):
                return dataset
            else:
                # Split if needed
                split = dataset.train_test_split(test_size=0.1, seed=42)
                return {"train": split["train"], "test": split["test"]}
                
        except Exception as e:
            print(f"   [WARN] Failed to load pre-tokenized dataset: {e}")
            print(f"   Falling back to regular loading...")
            use_pretokenized = False
    
    # Check if it's a HuggingFace dataset (no file extension) or local file
    is_hf_dataset = not config.dataset_path.endswith('.jsonl') and not os.path.exists(config.dataset_path)
    
    # Determine if streaming is enabled
    use_streaming = config.streaming if config.streaming is not None else False
    
    if use_streaming:
        print(f"   [STREAMING] Enabled - loading examples on-demand")
        if config.max_in_memory_samples:
            print(f"   [STREAMING] Max in-memory samples: {config.max_in_memory_samples}")
        print(f"   [STREAMING] RAM usage: ~2-3GB (vs ~12-15GB without streaming)")
    
    if is_hf_dataset:
        # Load from HuggingFace Hub
        # Format: "dataset_name" or "dataset_name::split" or "dataset_name::split::config"
        parts = config.dataset_path.split("::")
        dataset_name = parts[0]
        split = parts[1] if len(parts) > 1 else "train"
        config_name = parts[2] if len(parts) > 2 else None
        
        print(f"   Source: HuggingFace Hub")
        print(f"   Dataset: {dataset_name}")
        print(f"   Split: {split}")
        if config_name:
            print(f"   Config: {config_name}")
        
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=use_streaming)
    else:
        # Load local JSONL
        print(f"   Source: Local file")
        
        # Check if we have separate validation/test files
        if config.validation_path and os.path.exists(config.validation_path):
            data_files = {"train": config.dataset_path}
            
            if config.validation_path:
                data_files["validation"] = config.validation_path
                print(f"   Validation: {config.validation_path}")
            
            if config.test_path and os.path.exists(config.test_path):
                data_files["test"] = config.test_path
                print(f"   Test: {config.test_path}")
            
            dataset = load_dataset("json", data_files=data_files, streaming=use_streaming)
        else:
            # Single file - will split later
            dataset = load_dataset("json", data_files=config.dataset_path, split="train", streaming=use_streaming)
    
    # For streaming datasets, we can't use len() directly
    if use_streaming:
        print(f"   Mode: Streaming (on-demand loading)")
        # Try to peek at column names if available
        try:
            first_example = next(iter(dataset))
            print(f"   Columns: {list(first_example.keys())}")
        except:
            print(f"   Columns: Unknown (streaming)")
    else:
        print(f"   Examples: {len(dataset)}")
        print(f"   Columns: {dataset.column_names}")
    
    # Auto-detect if dataset has "text" field (for SFTTrainer)
    # For streaming datasets, check first example; for regular datasets, use column_names
    has_text_field = False
    if use_streaming:
        try:
            first_example = next(iter(dataset))
            has_text_field = "text" in first_example
        except:
            pass
    else:
        has_text_field = "text" in dataset.column_names
    
    if has_text_field and not config.text_field:
        print(f"   [OK] Auto-detected 'text' field - will use SFTTrainer with packing")
        config.text_field = "text"
        
        # Streaming datasets can't be split easily - return as-is
        if use_streaming:
            print(f"   [WARN] Streaming mode: using entire dataset for training (no validation split)")
            return {"train": dataset, "test": []}
        
        # Return raw dataset without tokenization (SFTTrainer handles it)
        if "validation" in dataset:
            print(f"   Using pre-split dataset (train + validation)")
            return {"train": dataset["train"], "test": dataset["validation"]}
        elif "test" in dataset:
            print(f"   Using pre-split dataset (train + test)")
            return {"train": dataset["train"], "test": dataset["test"]}
        else:
            # Split for eval
            print(f"   Creating 10% validation split from train...")
            split = dataset.train_test_split(test_size=0.1, seed=42)
            return {"train": split["train"], "test": split["test"]}
    
    # Determine field mapping
    if config.field_mapping:
        print(f"   Using custom field mapping: {config.field_mapping}")
        instruction_field = config.field_mapping.get("instruction", "instruction")
        input_field = config.field_mapping.get("input", "input")
        response_field = config.field_mapping.get("response", "response")
    else:
        instruction_field = "instruction"
        input_field = "input"
        response_field = "response"
    
    # Determine template to use from base_model config
    template_name = "alpaca"  # Default
    if hasattr(config, 'prompt_template') and config.prompt_template:
        template_name = config.prompt_template
        print(f"   Using template from manifest: {template_name}")
    else:
        # Auto-detect based on model name
        template_name = get_recommended_template(config.base_model_name)
        print(f"   Auto-detected template: {template_name} (from model name)")
    
    # Tokenization function
    def tokenize_function(examples):
        texts = []
        
        # Mode 1: Single text field (just tokenize as-is)
        if config.text_field:
            if config.text_field not in examples:
                raise ValueError(f"Text field '{config.text_field}' not found in dataset. Available: {list(examples.keys())}")
            texts = examples[config.text_field]
        
        # Mode 2: Instruction-Response format with field mapping
        else:
            # Use field mapping from config if available
            if config.field_mapping:
                instruction_field = config.field_mapping.get("instruction", "instruction")
                input_field = config.field_mapping.get("input", "input")
                response_field = config.field_mapping.get("response", "response")
            
            # Get fields with fallbacks
            instructions = examples.get(instruction_field, [])
            if not instructions:
                # Try common alternatives
                for alt in ["instruction", "prompt", "question", "text", "sql_prompt"]:
                    if alt in examples:
                        instructions = examples[alt]
                        print(f"   Auto-detected instruction field: {alt}")
                        break
            
            responses = examples.get(response_field, [])
            if not responses:
                # Try common alternatives
                for alt in ["response", "output", "answer", "completion", "text", "cypher", "sql"]:
                    if alt in examples:
                        responses = examples[alt]
                        print(f"   Auto-detected response field: {alt}")
                        break
            
            inputs = examples.get(input_field, [])
            
            # Build formatted texts using template
            for i, instruction in enumerate(instructions):
                # Get input/context
                input_text = None
                input_label = "Input"
                
                if inputs and i < len(inputs) and inputs[i]:
                    input_text = inputs[i]
                    # Determine label based on content or field name
                    if "CREATE TABLE" in input_text or "schema" in input_field.lower():
                        input_label = "Schema"
                    elif "context" in input_field.lower():
                        input_label = "Context"
                
                # Get response
                response = responses[i] if responses and i < len(responses) else ""
                
                # Format using template
                text = format_training_example(
                    instruction=instruction,
                    response=response,
                    input_text=input_text,
                    input_label=input_label,
                    template_name=template_name
                )
                texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,  # Reduce from 2048 for faster training
            return_tensors=None,
        )
        
        # Set labels for causal LM (copy input_ids)
        tokenized["labels"] = [[x for x in ids] for ids in tokenized["input_ids"]]
        
        return tokenized
    
    # Tokenize dataset
    print("   Tokenizing...")
    
    # For streaming datasets, we need to handle differently
    from datasets import DatasetDict
    
    if use_streaming:
        # Streaming datasets don't support remove_columns with column_names
        # We'll manually remove columns in the tokenize function
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing",
        )
    elif isinstance(dataset, DatasetDict):
        # Multiple splits - tokenize each separately
        tokenized_dataset = {}
        for split_name in dataset.keys():
            tokenized_dataset[split_name] = dataset[split_name].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset[split_name].column_names,
                desc=f"Tokenizing {split_name}",
            )
        tokenized_dataset = DatasetDict(tokenized_dataset)
    else:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
    
    # Save to cache if configured (not supported with streaming)
    if config.pretokenized_cache and not use_streaming:
        cache_path = Path(config.pretokenized_cache)
        
        # Only save if cache doesn't exist (we're creating it now)
        if not cache_path.exists():
            print(f"\n   Saving pre-tokenized dataset to cache...")
            print(f"   Cache path: {cache_path}")
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                from datasets import DatasetDict
                
                # Convert to DatasetDict if it's not already
                if isinstance(tokenized_dataset, DatasetDict):
                    dataset_to_save = tokenized_dataset
                else:
                    # Single dataset - wrap in DatasetDict
                    dataset_to_save = DatasetDict({"train": tokenized_dataset})
                
                dataset_to_save.save_to_disk(str(cache_path))
                print(f"   [OK] Cache saved! Future training runs will be faster.")
            except Exception as e:
                print(f"   [WARN] Failed to save cache: {e}")
                print(f"   Training will continue without cache...")
    elif config.pretokenized_cache and use_streaming:
        print(f"\n   [INFO] Pre-tokenized cache disabled (incompatible with streaming mode)")
    
    return tokenized_dataset


def parse_json_field(json_str):
    """Parse JSON string field from Rust"""
    if json_str is None:
        return None
    import json
    try:
        return json.loads(json_str)
    except:
        return None


class MemoryCleanupCallback(TrainerCallback):
    """
    Dynamically clean up VRAM during training to prevent GPU memory leaks.
    
    Problem: Dataset batches are loaded into VRAM and never released.
    PyTorch keeps references to intermediate tensors for autograd.
    
    Strategy:
    1. Monitor VRAM usage every step (fast check)
    2. If VRAM > 80% of total, trigger immediate cleanup
    3. Also cleanup periodically as fallback (every N steps)
    
    Key differences from CPU GC:
    - torch.cuda.empty_cache() releases VRAM blocks back to CUDA
    - gc.collect() only helps with Python object references
    - Need both: gc.collect() to release Python refs, then empty_cache() for VRAM
    """
    
    def __init__(self, cleanup_steps=50, vram_threshold=0.80):
        self.cleanup_steps = cleanup_steps
        self.vram_threshold = vram_threshold  # 80% threshold
        self.step_count = 0
        self.total_vram = None
        self.last_cleanup_step = 0
        
        # Get total VRAM at initialization
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.total_vram = props.total_memory / 1024**3  # GB
            print(f"   [VRAM Monitor] Total VRAM: {self.total_vram:.2f}GB")
            print(f"   [VRAM Monitor] Cleanup threshold: {self.vram_threshold*100:.0f}% ({self.total_vram * self.vram_threshold:.2f}GB)")
        
    def _cleanup_vram(self, step, reason="periodic"):
        """Perform aggressive VRAM and RAM cleanup (Windows-optimized)"""
        # Get memory before cleanup
        ram_before = psutil.virtual_memory().used / 1024**3 if HAS_PSUTIL else 0
        vram_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        # Step 1: Force Python garbage collection (aggressive)
        for _ in range(3):  # Multiple passes to catch circular references
            gc.collect()
        
        # Step 2: Clear CUDA cache (THIS is what frees VRAM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Windows-specific: Reset peak memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
        
        # Get memory after cleanup
        ram_after = psutil.virtual_memory().used / 1024**3 if HAS_PSUTIL else 0
        vram_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        ram_freed = ram_before - ram_after
        vram_freed = vram_before - vram_after
        
        # Log if freed significant memory (> 100MB)
        if vram_freed > 0.1 or ram_freed > 0.1:
            print(f"\n[CLEANUP] Step {step} ({reason})")
            if vram_freed > 0.1:
                print(f"  VRAM: {vram_before:.2f}GB -> {vram_after:.2f}GB (freed {vram_freed:.2f}GB)")
            if HAS_PSUTIL and ram_freed > 0.1:
                print(f"  RAM:  {ram_before:.2f}GB -> {ram_after:.2f}GB (freed {ram_freed:.2f}GB)")
        
        self.last_cleanup_step = step
        
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor VRAM and cleanup if needed"""
        self.step_count += 1
        
        if not torch.cuda.is_available():
            return
        
        # Check current VRAM usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        usage_percent = allocated / self.total_vram if self.total_vram else 0
        
        # TRIGGER 1: VRAM above threshold (80%)
        if usage_percent >= self.vram_threshold:
            # Avoid cleaning too frequently (minimum 10 steps between cleanups)
            if state.global_step - self.last_cleanup_step >= 10:
                self._cleanup_vram(state.global_step, f"threshold {usage_percent*100:.1f}%")
        
        # TRIGGER 2: Periodic cleanup (fallback)
        elif self.step_count % self.cleanup_steps == 0:
            self._cleanup_vram(state.global_step, "periodic")
        
        # Log memory status every 250 steps
        if self.step_count % 250 == 0:
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"\n[MEMORY] Step {state.global_step}")
            print(f"  VRAM: {allocated:.2f}GB / {self.total_vram:.2f}GB ({usage_percent*100:.1f}%) | Reserved: {reserved:.2f}GB")
            
            if HAS_PSUTIL:
                ram_used = psutil.virtual_memory().used / 1024**3
                ram_total = psutil.virtual_memory().total / 1024**3
                ram_percent = psutil.virtual_memory().percent
                print(f"  RAM:  {ram_used:.2f}GB / {ram_total:.2f}GB ({ram_percent:.1f}%)")


class OverfittingMonitorCallback(TrainerCallback):
    """
    Monitor for early overfitting signals in SQL training.
    
    Detects when train loss diverges from eval loss (sign of overfitting),
    especially critical for synthetic datasets that may not generalize to real queries.
    
    Triggers:
    - Train loss < 0.3 (converged) AND eval_loss > train_loss * 1.2 (20% gap)
    - Action: Reduce LR by 50% to stabilize and improve generalization
    """
    
    def __init__(self, overfitting_threshold=1.2, min_train_loss=0.3):
        self.overfitting_threshold = overfitting_threshold
        self.min_train_loss = min_train_loss
        self.lr_reduced = False
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check for overfitting after each evaluation"""
        if metrics is None:
            return
        
        eval_loss = metrics.get("eval_loss")
        train_loss = state.log_history[-1].get("loss") if state.log_history else None
        
        if eval_loss is None or train_loss is None:
            return
        
        # Check for overfitting signal
        if train_loss < self.min_train_loss and eval_loss > train_loss * self.overfitting_threshold:
            if not self.lr_reduced:
                print(f"\n⚠️  [OVERFITTING DETECTED]")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Eval Loss:  {eval_loss:.4f}")
                print(f"   Gap: {(eval_loss/train_loss - 1)*100:.1f}%")
                print(f"   Action: Reducing LR by 50% to improve generalization")
                
                # Reduce learning rate
                for param_group in kwargs.get("optimizer").param_groups:
                    param_group['lr'] *= 0.5
                    print(f"   New LR: {param_group['lr']:.6f}")
                
                self.lr_reduced = True
        
        # Log status
        if eval_loss and train_loss:
            gap_percent = (eval_loss / train_loss - 1) * 100
            status = "✓" if gap_percent < 20 else "⚠️"
            print(f"   {status} Train/Eval Gap: {gap_percent:+.1f}%")


def train_expert(config_dict: dict) -> None:
    """Main training function called from Rust"""
    
    # Windows-specific memory optimizations
    import platform
    if platform.system() == "Windows":
        print("\n[WINDOWS] Applying memory optimizations for Windows...")
        
        training = config_dict.get("training", {})
        
        # Force single-worker mode (critical for Windows - prevents worker memory copies)
        if training.get("dataloader_num_workers", 0) > 0:
            print(f"   Changing dataloader_num_workers: {training.get('dataloader_num_workers')} -> 0 (Windows fix)")
            training["dataloader_num_workers"] = 0
        
        training["dataloader_persistent_workers"] = False
        
        # Disable torch.compile on Windows (causes memory leaks)
        #if training.get("torch_compile"):
        #    print(f"   Disabling torch_compile (Windows memory leak fix)")
        #    training["torch_compile"] = False
        
        # Disable CUDA graphs on Windows (unstable)
        #if training.get("use_cuda_graphs"):
        #    print(f"   Disabling use_cuda_graphs (Windows unstable)")
        #    training["use_cuda_graphs"] = False
        
        # Disable Flash Attention 2 on Windows (can cause issues)
        if training.get("flash_attention_2"):
            print(f"   Disabling flash_attention_2 (Windows compatibility)")
            training["flash_attention_2"] = False
            training["use_sdpa"] = True  # Use SDPA instead
        
        print("[WINDOWS] Optimizations applied\n")
    
    # Handle device auto-detection FIRST
    if config_dict.get("device") == "auto":
        detected_device = "cuda" if torch.cuda.is_available() else "cpu"
        config_dict["device"] = detected_device
        print(f"   [AUTO] Device auto-detected: {detected_device}")
    
    # Disable all subprocess-based features to avoid conflicts
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ["ACCELERATE_TORCH_DEVICE"] = config_dict.get("device", "cuda")
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # CRITICAL: Disable Accelerate launcher completely
    os.environ["ACCELERATE_LAUNCH_PARAMS"] = ""
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU
    
    # Parse config
    config = load_training_config(config_dict)
    
    print("\n" + "=" * 60)
    print("Expert Training Pipeline")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Configure soft prompts (if specified in manifest)
    # Must be called BEFORE adapter setup
    soft_prompts = getattr(config, 'soft_prompts', [])
    if soft_prompts:
        model = configure_soft_prompts(model, soft_prompts, config.base_model_name, tokenizer)
    
    # Setup adapter
    model = setup_lora(model, config)
    
    # Load dataset
    dataset_result = load_and_prepare_dataset(config, tokenizer)
    
    # Clean up VRAM after loading dataset (dataset may have created temporary tensors)
    print(f"\n   [VRAM] Cleaning up after dataset load...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_after_load = torch.cuda.memory_allocated() / 1024**3
        print(f"   VRAM after cleanup: {allocated_after_load:.2f}GB")
    
    # Check if already split (multi-task) or needs splitting
    if isinstance(dataset_result, dict) and "train" in dataset_result:
        # Already split (multi-task or cached)
        train_dataset = dataset_result["train"]
        eval_dataset = dataset_result.get("test", [])
    else:
        # Single dataset - needs splitting (if not streaming)
        from datasets import IterableDataset
        
        if isinstance(dataset_result, IterableDataset):
            # Streaming dataset - can't split, use entire dataset for training
            print(f"\n   [STREAMING] Using entire dataset for training (no validation split)")
            train_dataset = dataset_result
            eval_dataset = []
        else:
            split_dataset = dataset_result.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
    
    # For streaming datasets, we can't use len()
    from datasets import IterableDataset
    if isinstance(train_dataset, IterableDataset):
        print(f"\n   Train examples: Streaming (on-demand)")
        print(f"   Eval examples: None (streaming mode)")
    else:
        print(f"\n   Train examples: {len(train_dataset)}")
        print(f"   Eval examples: {len(eval_dataset)}")
    
    # Determine model-specific output directory for schema v2.0
    # Extract model name from path like "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    model_base_name = os.path.basename(config.base_model_name)
    # Normalize: "Qwen3-0.6B" -> "qwen3-0.6b"
    model_dir_name = model_base_name.lower().replace(".", "").replace("_", "-")
    
    # Create model-specific directory: weights/qwen3-0.6b/
    base_output_dir = Path(config.output_dir)
    output_dir = base_output_dir / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Model directory: {model_dir_name}")
    print(f"   Output path: {output_dir}")
    print(f"   Checkpoints will be saved in: {output_dir}/checkpoint-XXX")
    
    # Check if we have eval data
    has_eval = len(eval_dataset) > 0
    
    if not has_eval:
        print("\n   WARNING: No validation data available - disabling evaluation")
    
    # Enable CUDA Graphs if requested (via torch.compile)
    if config.use_cuda_graphs and config.device == "cuda":
        print(f"\n   [CUDA GRAPHS] Enabled via torch.compile")
        print(f"   Warmup steps: {config.cuda_graph_warmup_steps or 100}")
        print(f"   Expected gain: +15-20% throughput")
        print(f"   VRAM overhead: ~50-100MB")
        
        # CUDA Graphs work best with torch.compile + inductor backend
        if not config.torch_compile:
            print(f"   [WARN] CUDA Graphs requires torch_compile=true for best performance")
            print(f"   Enabling torch.compile automatically...")
            config.torch_compile = True
            config.torch_compile_backend = "inductor"
            config.torch_compile_mode = "max-autotune"  # Best for CUDA graphs
    
    # Calculate max_steps for streaming datasets
    # Streaming datasets don't have __len__, so we need to specify max_steps
    max_steps = None
    if isinstance(train_dataset, IterableDataset):
        # Estimate based on known dataset size (78311 examples for expert-sql)
        # For now, use a reasonable estimate or require it in config
        # Formula: (dataset_size / (batch_size * grad_accumulation)) * epochs
        
        # Hardcoded for expert-sql dataset (will be configurable later)
        dataset_size = 78311  # TODO: Make this configurable in manifest
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        steps_per_epoch = dataset_size // effective_batch_size
        max_steps = int(steps_per_epoch * config.epochs)
        
        print(f"\n   [STREAMING] Calculated training steps:")
        print(f"   Dataset size: {dataset_size} examples (estimated)")
        print(f"   Effective batch size: {effective_batch_size} ({config.batch_size} * {config.gradient_accumulation_steps})")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Total epochs: {config.epochs}")
        print(f"   Max steps: {max_steps}")
    
    # Build training args with optional advanced params
    training_args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler,
        "warmup_steps": config.warmup_steps if config.warmup_ratio is None else 0,
        "warmup_ratio": config.warmup_ratio if config.warmup_ratio is not None else 0.0,
        "logging_steps": config.logging_steps or 10,
        "save_total_limit": config.save_total_limit or 3,
        "fp16": config.fp16 if config.fp16 is not None else (not config.bf16 and config.device == "cuda"),
        "bf16": config.bf16 if config.bf16 is not None else False,
        "optim": config.optim or "adamw_torch",
        "report_to": [],  # Disable all logging integrations
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "disable_tqdm": False,  # Keep tqdm for progress
        "use_cpu": False,
        "ddp_find_unused_parameters": False,
        "local_rank": -1,
        "deepspeed": None,
        "fsdp": "",  # Disable FSDP
        "torch_compile": config.use_cuda_graphs and config.device == "cuda",  # Enable for CUDA graphs
    }
    
    # Configure eval and save strategies based on streaming mode OR manifest config
    if max_steps is not None:
        # Streaming mode: use steps-based saving, no evaluation (unless overridden)
        training_args_dict["eval_strategy"] = config.evaluation_strategy or "no"
        training_args_dict["save_strategy"] = config.save_strategy or "steps"
        training_args_dict["save_steps"] = config.save_steps or 500
        training_args_dict["load_best_model_at_end"] = config.load_best_model_at_end if config.load_best_model_at_end is not None else False
        training_args_dict["metric_for_best_model"] = config.metric_for_best_model
        training_args_dict["greater_is_better"] = False
        if config.eval_steps:
            training_args_dict["eval_steps"] = config.eval_steps
    else:
        # Regular mode: use manifest config OR defaults
        training_args_dict["eval_strategy"] = config.evaluation_strategy or ("epoch" if has_eval else "no")
        training_args_dict["save_strategy"] = config.save_strategy or "epoch"
        training_args_dict["load_best_model_at_end"] = config.load_best_model_at_end if config.load_best_model_at_end is not None else has_eval
        training_args_dict["metric_for_best_model"] = config.metric_for_best_model or ("eval_loss" if has_eval else None)
        training_args_dict["greater_is_better"] = False
        
        # Add save_steps and eval_steps if specified
        if config.save_steps:
            training_args_dict["save_steps"] = config.save_steps
        if config.eval_steps:
            training_args_dict["eval_steps"] = config.eval_steps
    
    # Add max_steps for streaming datasets, or num_train_epochs for regular datasets
    if max_steps is not None:
        training_args_dict["max_steps"] = max_steps
        # Don't set num_train_epochs when using max_steps (they're mutually exclusive)
    else:
        training_args_dict["num_train_epochs"] = config.epochs
    
    # Add dataloader params if specified
    # Note: For VRAM optimization, we want to minimize buffering
    if config.dataloader_num_workers is not None:
        training_args_dict["dataloader_num_workers"] = config.dataloader_num_workers
        if config.dataloader_num_workers == 0:
            print(f"   [DATALOADER] num_workers=0 (single-process, Windows compatible)")
        print(f"   [DATALOADER] Workers: {config.dataloader_num_workers} (VRAM: less workers = less buffering)")
    
    # pin_memory keeps data in CUDA-accessible RAM, but can increase memory usage
    if config.dataloader_pin_memory is not None and config.dataloader_pin_memory:
        training_args_dict["dataloader_pin_memory"] = config.dataloader_pin_memory
        print(f"   [DATALOADER] pin_memory=True (faster GPU transfer, +RAM usage)")
    
    # prefetch_factor controls how many batches to load ahead
    # Lower = less VRAM buffering
    if config.dataloader_prefetch_factor is not None and config.dataloader_num_workers and config.dataloader_num_workers > 0:
        training_args_dict["dataloader_prefetch_factor"] = config.dataloader_prefetch_factor
        print(f"   [DATALOADER] prefetch_factor={config.dataloader_prefetch_factor} (lower = less buffering)")
    
    # persistent_workers keeps workers alive between epochs
    # False = workers restart each epoch = free memory
    if config.dataloader_persistent_workers is not None and config.dataloader_num_workers and config.dataloader_num_workers > 0:
        training_args_dict["dataloader_persistent_workers"] = config.dataloader_persistent_workers
        if not config.dataloader_persistent_workers:
            print(f"   [DATALOADER] persistent_workers=False (workers restart each epoch = free memory)")
    
    if config.group_by_length is not None:
        training_args_dict["group_by_length"] = config.group_by_length
    
    # Add optional scheduler kwargs (for cosine_with_restarts)
    if hasattr(config, 'lr_scheduler_kwargs') and config.lr_scheduler_kwargs:
        training_args_dict["lr_scheduler_kwargs"] = config.lr_scheduler_kwargs
        print(f"\n   LR Scheduler: {config.lr_scheduler} with {config.lr_scheduler_kwargs}")
    
    # Add gradient checkpointing kwargs if needed
    if config.gradient_checkpointing:
        if hasattr(config, 'gradient_checkpointing_kwargs') and config.gradient_checkpointing_kwargs:
            training_args_dict["gradient_checkpointing_kwargs"] = config.gradient_checkpointing_kwargs
        else:
            training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Debug: Log checkpoint configuration
    print(f"\n   [CHECKPOINTS] Configuration:")
    print(f"      Save strategy: {training_args.save_strategy}")
    print(f"      Save steps: {training_args.save_steps if training_args.save_strategy == 'steps' else 'N/A'}")
    print(f"      Save total limit: {training_args.save_total_limit}")
    print(f"      Output dir: {training_args.output_dir}")
    
    # Initialize memory cleanup callback (always enabled)
    memory_callback = MemoryCleanupCallback(cleanup_steps=50, vram_threshold=0.80)
    print(f"\n   [CALLBACK] Dynamic Memory Cleanup (Windows-optimized)")
    print(f"   Strategy: Monitor VRAM + RAM every step")
    print(f"   Trigger 1: VRAM > 80% -> immediate cleanup")
    print(f"   Trigger 2: Every 50 steps -> periodic cleanup")
    print(f"   Action: gc.collect(×3) + empty_cache() + reset_peak_stats()")
    if HAS_PSUTIL:
        print(f"   [OK] psutil detected - monitoring system RAM")
    
    # Initialize overfitting monitor callback (only if we have eval dataset)
    overfitting_callback = None
    if has_eval:
        overfitting_callback = OverfittingMonitorCallback(
            overfitting_threshold=1.2,  # Trigger if eval_loss > train_loss * 1.2
            min_train_loss=0.3  # Only monitor after train loss < 0.3
        )
        print(f"\n   [CALLBACK] Overfitting monitor enabled")
        print(f"   Trigger: train_loss < 0.3 AND eval_loss > train_loss * 1.2")
        print(f"   Action: Reduce LR by 50% to improve generalization")
    
    # Build callback list
    callbacks = [memory_callback]
    if overfitting_callback:
        callbacks.append(overfitting_callback)
    
    # Determine if dataset has "text" field for SFTTrainer packing
    # Check first example (works for both regular and streaming datasets)
    has_text_field = False
    from datasets import IterableDataset
    
    try:
        if isinstance(train_dataset, IterableDataset):
            # Streaming dataset - peek at first example
            first_example = next(iter(train_dataset))
            has_text_field = "text" in first_example
        elif len(train_dataset) > 0:
            # Regular dataset
            has_text_field = "text" in train_dataset[0]
    except:
        pass
    
    # Use SFTTrainer with packing if dataset is formatted correctly
    if has_text_field:
        print(f"\n   Using SFTTrainer with sequence packing")
        print(f"   Max sequence length: {config.max_seq_length or 2048}")
        print(f"   Packing: ENABLED (+30-40% tokens/s expected)")
        
        # For streaming datasets with max_in_memory_samples limit
        if config.streaming and config.max_in_memory_samples:
            print(f"   [STREAMING] Max in-memory samples: {config.max_in_memory_samples}")
        
        # SFTTrainer with basic parameters (packing is automatic with text field)
        # Note: max_seq_length is handled via tokenizer truncation in dataset preprocessing
        
        # SFTTrainer configuration
        sft_kwargs = {
            "model": model,
            "tokenizer": tokenizer,  # Required by Unsloth
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset if has_eval else None,
            "formatting_func": lambda x: x["text"],  # Extract text field
            "callbacks": callbacks,  # Include memory cleanup + overfitting monitor
            "dataset_num_proc": 1,  # Use single process to avoid spam logs (Windows compatibility)
        }
        
        # Unsloth-specific: Set dataloader workers to 0 for Windows compatibility
        use_unsloth_flag = getattr(config, 'use_unsloth', None)
        if use_unsloth_flag and USE_UNSLOTH:
            print(f"   [UNSLOTH] Dataloader optimizations applied")
        
        trainer = SFTTrainer(**sft_kwargs)
    else:
        # Fallback to standard Trainer if dataset not formatted for SFTTrainer
        print(f"\n   Using standard Trainer (dataset missing 'text' field)")
        print(f"   Note: For better performance, format dataset with 'text' field")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM
        )
        
        from transformers import Trainer
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if has_eval else None,
            data_collator=data_collator,
            callbacks=callbacks,  # Include memory cleanup + overfitting monitor
        )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    trainer.train(resume_from_checkpoint=config.resume_checkpoint)
    
    # Final VRAM cleanup after training
    print("\n[VRAM] Final cleanup after training...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_final = torch.cuda.memory_allocated() / 1024**3
        print(f"   VRAM after training: {allocated_final:.2f}GB")
    
    # Save final model
    print("\nSaving model...")
    final_dir = output_dir / "final"
    adapter_dir = output_dir / "adapter"
    
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    # Save soft prompts if configured
    save_soft_prompts(model, soft_prompts, output_dir)
    
    # Save adapter only
    model.save_pretrained(str(adapter_dir))
    
    print("\nTraining complete!")
    print(f"   Model: {model_base_name}")
    print(f"   Output: {output_dir}")
    print(f"   Adapter: {adapter_dir / 'adapter_model.safetensors'}")
    print(f"   Full model: {final_dir}")


if __name__ == "__main__":
    # For standalone testing
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        with open(config_file) as f:
            config = json.load(f)
        train_expert(config)
    else:
        print("Usage: python expert_trainer.py <config.json>")
        sys.exit(1)

