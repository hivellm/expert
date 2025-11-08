"""
Training Configuration Module

Handles parsing and validation of training configuration from manifest.
"""

from dataclasses import dataclass
from typing import Optional


def parse_json_field(json_str):
    """Parse JSON string field from Rust"""
    if json_str is None:
        return None
    import json
    try:
        return json.loads(json_str)
    except:
        return None


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

