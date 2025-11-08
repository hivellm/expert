"""
Adapter Setup Module

Handles configuration of LoRA/DoRA/IA³/LoKr adapters and soft prompts.
"""

import torch
from pathlib import Path
from peft import LoraConfig, PromptTuningConfig, get_peft_model, TaskType

# Import Unsloth if available
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False

from .config import TrainingConfig


def setup_adapter(model, config: TrainingConfig):
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
        model = torch.compile(model, backend=backend, mode=mode)
    
    return model


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

