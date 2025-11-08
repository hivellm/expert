"""
Dataset Loading Module

Handles loading and preprocessing of datasets (HuggingFace, local JSONL, multi-task).
"""

import os
from pathlib import Path
from datasets import load_dataset, DatasetDict, IterableDataset

from .config import TrainingConfig

# Import prompt templates from scripts directory
import sys
from pathlib import Path as PathLib
scripts_path = PathLib(__file__).parent.parent / "scripts"
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


def load_multi_task_dataset(config: TrainingConfig, tokenizer):
    """Load and combine multi-task datasets"""
    from pathlib import Path as PathlibPath
    
    # Add scripts directory to Python path
    scripts_dir = PathlibPath(__file__).parent.parent / "scripts"
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

