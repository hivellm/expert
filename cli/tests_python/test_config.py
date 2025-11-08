#!/usr/bin/env python3
"""
Tests for config.py module
"""

import pytest
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.config import TrainingConfig, load_training_config, parse_json_field


class TestParseJsonField:
    """Test parse_json_field function"""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON string"""
        json_str = '{"key": "value", "number": 42}'
        result = parse_json_field(json_str)
        assert result == {"key": "value", "number": 42}
    
    def test_parse_none(self):
        """Test parsing None"""
        assert parse_json_field(None) is None
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None"""
        invalid_json = "{invalid json}"
        result = parse_json_field(invalid_json)
        assert result is None
    
    def test_parse_empty_string(self):
        """Test parsing empty string"""
        result = parse_json_field("")
        assert result is None


class TestTrainingConfig:
    """Test TrainingConfig dataclass"""
    
    def test_create_minimal_config(self):
        """Test creating config with minimal required fields"""
        config = TrainingConfig(
            base_model_name="test-model",
            quantization="none",
            dataset_path="test.jsonl",
            output_dir="output",
            device="cpu",
            adapter_type="lora",
            rank=16,
            alpha=16,
            target_modules=["q_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
        )
        
        assert config.base_model_name == "test-model"
        assert config.quantization == "none"
        assert config.adapter_type == "lora"
        assert config.rank == 16
    
    def test_config_with_optional_fields(self):
        """Test creating config with optional fields"""
        config = TrainingConfig(
            base_model_name="test-model",
            quantization="int4",
            dataset_path="test.jsonl",
            output_dir="output",
            device="cuda",
            adapter_type="dora",
            rank=12,
            alpha=24,
            target_modules=["q_proj", "v_proj"],
            epochs=3,
            learning_rate=0.00005,
            batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            lr_scheduler="cosine",
            validation_path="val.jsonl",
            test_path="test.jsonl",
            warmup_ratio=0.1,
            max_seq_length=2048,
            fp16=True,
            bf16=False,
        )
        
        assert config.validation_path == "val.jsonl"
        assert config.test_path == "test.jsonl"
        assert config.max_seq_length == 2048
        assert config.fp16 is True
        assert config.bf16 is False


class TestLoadTrainingConfig:
    """Test load_training_config function"""
    
    def test_load_single_file_config(self):
        """Test loading config for single file dataset"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {},
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "linear",
            },
            "dataset": {
                "type": "single",
                "path": "test.jsonl",
            },
            "dataset_path": "test.jsonl",
        }
        
        config = load_training_config(config_dict)
        
        assert config.base_model_name == "test-model"
        assert config.dataset_path == "test.jsonl"
        assert config.dataset_type == "single"
        assert config.adapter_type == "lora"
    
    def test_load_multi_task_config(self):
        """Test loading config for multi-task dataset"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {},
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "linear",
            },
            "dataset": {
                "type": "multi_task",
                "tasks": [
                    {
                        "name": "task1",
                        "path": "task1.jsonl",
                    }
                ],
            },
        }
        
        config = load_training_config(config_dict)
        
        assert config.dataset_type == "multi_task"
        assert config.multi_task_config is not None
        assert config.dataset_path == ""  # Not used for multi-task
    
    def test_load_config_with_prompt_template(self):
        """Test loading config with prompt template from base_model"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {
                "prompt_template": "qwen",
            },
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "linear",
            },
            "dataset": {
                "type": "single",
                "path": "test.jsonl",
            },
            "dataset_path": "test.jsonl",
        }
        
        config = load_training_config(config_dict)
        
        assert config.prompt_template == "qwen"
    
    def test_load_config_with_validation_path(self):
        """Test loading config with validation and test paths"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {},
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "linear",
            },
            "dataset": {
                "type": "single",
                "path": "train.jsonl",
                "validation_path": "val.jsonl",
                "test_path": "test.jsonl",
            },
            "dataset_path": "train.jsonl",
            "validation_path": "val.jsonl",
            "test_path": "test.jsonl",
        }
        
        config = load_training_config(config_dict)
        
        assert config.validation_path == "val.jsonl"
        assert config.test_path == "test.jsonl"
    
    def test_load_config_with_gradient_checkpointing_json(self):
        """Test loading config with gradient_checkpointing as JSON string"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {},
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "linear",
            },
            "dataset": {
                "type": "single",
                "path": "test.jsonl",
            },
            "dataset_path": "test.jsonl",
            "gradient_checkpointing_json": "true",
        }
        
        config = load_training_config(config_dict)
        
        assert config.gradient_checkpointing is True
    
    def test_load_config_with_lr_scheduler_kwargs(self):
        """Test loading config with lr_scheduler_kwargs as JSON"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {},
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "cosine",
            },
            "dataset": {
                "type": "single",
                "path": "test.jsonl",
            },
            "dataset_path": "test.jsonl",
            "lr_scheduler_kwargs_json": '{"num_cycles": 2}',
        }
        
        config = load_training_config(config_dict)
        
        assert config.lr_scheduler_kwargs == {"num_cycles": 2}
    
    def test_load_config_with_resume_checkpoint(self):
        """Test loading config with resume_checkpoint"""
        config_dict = {
            "base_model_name": "test-model",
            "quantization": "none",
            "output_dir": "output",
            "device": "cpu",
            "base_model": {},
            "training": {
                "adapter_type": "lora",
                "rank": 16,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "lr_scheduler": "linear",
            },
            "dataset": {
                "type": "single",
                "path": "test.jsonl",
            },
            "dataset_path": "test.jsonl",
            "resume_checkpoint": "checkpoint-500",
        }
        
        config = load_training_config(config_dict)
        
        assert config.resume_checkpoint == "checkpoint-500"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

