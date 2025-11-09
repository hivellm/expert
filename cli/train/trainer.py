"""
Main Training Module

Orchestrates the training pipeline by coordinating all modules.
"""

import gc
import os
import platform
from pathlib import Path
from datasets import IterableDataset, DatasetDict
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer
from trl import SFTTrainer

from .config import TrainingConfig, load_training_config
from .model_loader import load_model_and_tokenizer
from .dataset_loader import load_and_prepare_dataset
from .adapter_setup import setup_adapter, configure_soft_prompts, save_soft_prompts
from .callbacks import MemoryCleanupCallback, OverfittingMonitorCallback
from .progress_testing import ProgressTestCallback

# Import Unsloth if available
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
except ImportError:
    USE_UNSLOTH = False

# Import psutil for system RAM monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# Standalone formatting function for SFTTrainer (must be at module level for pickle)
# This avoids capturing non-serializable objects from Unsloth when using multiprocessing
# Note: For Windows, conversion happens during pre-tokenization. For non-Windows,
# the tokenizer's chat_template should handle the conversion automatically.
def _format_text_for_sft(examples):
    """Extract text field from examples - standalone function for multiprocessing compatibility"""
    if isinstance(examples, dict):
        return {"text": examples.get("text", "")}
    return {"text": examples["text"] if "text" in examples else ""}


def train_expert(config_dict: dict) -> None:
    """Main training function called from Rust"""
    
    # Windows-specific memory optimizations
    if platform.system() == "Windows":
        print("\n[WINDOWS] Applying memory optimizations for Windows...")
        
        training = config_dict.get("training", {})
        
        # Force single-worker mode (critical for Windows - prevents worker memory copies)
        if training.get("dataloader_num_workers", 0) > 0:
            print(f"   Changing dataloader_num_workers: {training.get('dataloader_num_workers')} -> 0 (Windows fix)")
            training["dataloader_num_workers"] = 0
        
        training["dataloader_persistent_workers"] = False
        
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
    
    # Extract expert name and version from config_dict (for progress testing)
    expert_name = config_dict.get("expert_name", "unknown")
    expert_version = config_dict.get("expert_version", "0.0.0")
    
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
    model = setup_adapter(model, config)
    
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
    
    # Initialize progress test callback (if test cases exist)
    progress_test_callback = None
    expert_dir = Path(config.output_dir).parent
    test_cases_path = expert_dir / "tests" / "test_cases.json"
    
    if test_cases_path.exists():
        progress_test_callback = ProgressTestCallback(
            expert_dir=expert_dir,
            base_model_path=config.base_model_name,
            expert_name=expert_name,
            expert_version=expert_version,
            test_cases_path=test_cases_path
        )
        print(f"\n   [CALLBACK] Progress testing enabled")
        print(f"   Test cases: {test_cases_path}")
        print(f"   Reports will be saved to: {expert_dir / 'weights' / 'training_reports'}")
        print(f"   Tests will run automatically at each checkpoint save")
    else:
        print(f"\n   [INFO] Progress testing disabled (test_cases.json not found)")
        print(f"   To enable: Create {test_cases_path}")
    
    # Build callback list
    callbacks = [memory_callback]
    if overfitting_callback:
        callbacks.append(overfitting_callback)
    if progress_test_callback:
        callbacks.append(progress_test_callback)
    
    # Determine if dataset has "text" field for SFTTrainer packing
    # Check first example (works for both regular and streaming datasets)
    has_text_field = False
    
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
        
        # Windows: Pre-tokenize dataset and use standard Trainer (SFTTrainer has pickle issues)
        use_standard_trainer = False
        if platform.system() == "Windows" and not isinstance(train_dataset, IterableDataset):
            print(f"   [WINDOWS] Pre-tokenizing dataset to avoid multiprocessing pickle issues...")
            
            # Convert ChatML format to Qwen3 format using apply_chat_template
            # Dataset uses <|system|>/<|end|> but Qwen3 expects <|im_start|>/<|im_end|>
            import re
            
            def convert_chatml_to_messages(text):
                """Convert ChatML format (<|system|>, <|user|>, <|assistant|>, <|end|>) to messages list"""
                messages = []
                
                # Extract system message
                system_match = re.search(r'<\|system\|>\n(.*?)<\|end\|>', text, re.DOTALL)
                if system_match:
                    messages.append({'role': 'system', 'content': system_match.group(1).strip()})
                
                # Extract user message
                user_match = re.search(r'<\|user\|>\n(.*?)<\|end\|>', text, re.DOTALL)
                if user_match:
                    messages.append({'role': 'user', 'content': user_match.group(1).strip()})
                
                # Extract assistant message (for training, include it)
                assistant_match = re.search(r'<\|assistant\|>\n(.*?)<\|end\|>', text, re.DOTALL)
                if assistant_match:
                    messages.append({'role': 'assistant', 'content': assistant_match.group(1).strip()})
                
                return messages
            
            # Tokenize function (standalone for pickle)
            def tokenize_examples(examples):
                """Tokenize examples - converts ChatML to Qwen3 format and tokenizes"""
                texts = examples["text"]
                converted_texts = []
                
                for text in texts:
                    # Check if already in Qwen3 format (has im_start)
                    if "<|im_start|>" in text:
                        # Already in correct format
                        converted_texts.append(text)
                    else:
                        # Convert from ChatML to Qwen3 format
                        messages = convert_chatml_to_messages(text)
                        if messages:
                            # Apply chat template to get Qwen3 format (<|im_start|>/<|im_end|>)
                            formatted = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False  # For training, include assistant response
                            )
                            converted_texts.append(formatted)
                        else:
                            # Fallback: use as-is
                            converted_texts.append(text)
                
                return tokenizer(
                    converted_texts,
                    truncation=True,
                    max_length=config.max_seq_length or 2048,
                    padding=False,
                )
            
            print(f"   Tokenizing {len(train_dataset)} train examples (single process)...")
            train_dataset = train_dataset.map(
                tokenize_examples,
                batched=True,
                num_proc=1,  # Single process for Windows
                remove_columns=[col for col in train_dataset.column_names if col != "text"],
            )
            
            if has_eval and len(eval_dataset) > 0:
                print(f"   Tokenizing {len(eval_dataset)} eval examples (single process)...")
                eval_dataset = eval_dataset.map(
                    tokenize_examples,
                    batched=True,
                    num_proc=1,  # Single process for Windows
                    remove_columns=[col for col in eval_dataset.column_names if col != "text"],
                )
            
            print(f"   [OK] Dataset pre-tokenized (Windows-compatible)")
            print(f"   [INFO] Using standard Trainer (SFTTrainer incompatible with pre-tokenized data)")
            use_standard_trainer = True
        
        if use_standard_trainer:
            # Use standard Trainer with pre-tokenized dataset
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if has_eval else None,
                data_collator=data_collator,
                callbacks=callbacks,  # Include all callbacks
            )
        else:
            # Use SFTTrainer with text formatting (non-Windows or streaming)
            sft_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset if has_eval else None,
                "formatting_func": _format_text_for_sft,  # Module-level function (pickle-safe)
                "callbacks": callbacks,  # Include all callbacks
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
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if has_eval else None,
            data_collator=data_collator,
            callbacks=callbacks,  # Include all callbacks
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

