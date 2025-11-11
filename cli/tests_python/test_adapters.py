#!/usr/bin/env python3
"""
Tests for IA³ and DoRA adapter implementations
"""

import pytest
import torch
from pathlib import Path
import sys
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train.config import TrainingConfig
from train.adapter_setup import setup_adapter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, IA3Config, get_peft_model


class TestAdapterTypes:
    """Test different adapter types"""
    
    @pytest.fixture
    def base_model_path(self):
        """Get base model path - uses small model for testing"""
        # Use local Qwen3-0.6B if available, else skip
        model_path = Path("../models/Qwen3-0.6B")
        if not model_path.exists():
            pytest.skip("Base model not available")
        return str(model_path)
    
    @pytest.fixture
    def base_model(self, base_model_path):
        """Load base model for testing"""
        # Load with int8 for faster tests
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        
        return model
    
    def test_lora_config(self, base_model, base_model_path):
        """Test standard LoRA configuration"""
        config = TrainingConfig(
            base_model_name=base_model_path,
            quantization="int8",
            dataset_path="test.jsonl",
            output_dir="test_output",
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_type="lora",
            rank=16,
            alpha=16,
            target_modules=["q_proj", "v_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
            gradient_checkpointing=False,
        )
        
        model_with_adapter = setup_adapter(base_model, config)
        
        # Verify adapter was added
        assert hasattr(model_with_adapter, 'peft_config')
        assert 'default' in model_with_adapter.peft_config
        
        # Verify it's LoRA
        peft_config = model_with_adapter.peft_config['default']
        assert isinstance(peft_config, LoraConfig)
        assert peft_config.r == 16
        assert peft_config.lora_alpha == 16
        
        # Verify trainable parameters
        trainable, total = model_with_adapter.get_nb_trainable_parameters()
        assert trainable > 0
        assert trainable < total
        print(f"LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def test_dora_config(self, base_model, base_model_path):
        """Test DoRA configuration"""
        config = TrainingConfig(
            base_model_name=base_model_path,
            quantization="int8",
            dataset_path="test.jsonl",
            output_dir="test_output",
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_type="dora",
            rank=12,
            alpha=12,
            target_modules=["q_proj", "v_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
            gradient_checkpointing=False,
        )
        
        model_with_adapter = setup_adapter(base_model, config)
        
        # Verify adapter was added
        assert hasattr(model_with_adapter, 'peft_config')
        peft_config = model_with_adapter.peft_config['default']
        
        # Verify it's LoRA with DoRA
        assert isinstance(peft_config, LoraConfig)
        assert peft_config.use_dora == True
        assert peft_config.r == 12
        
        # Verify trainable parameters
        trainable, total = model_with_adapter.get_nb_trainable_parameters()
        assert trainable > 0
        print(f"DoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def test_ia3_config(self, base_model, base_model_path):
        """Test IA³ configuration"""
        config = TrainingConfig(
            base_model_name=base_model_path,
            quantization="int8",
            dataset_path="test.jsonl",
            output_dir="test_output",
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_type="ia3",
            rank=None,  # IA³ doesn't use rank
            alpha=None,  # IA³ doesn't use alpha
            target_modules=["q_proj", "v_proj", "o_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
            gradient_checkpointing=False,
        )
        
        model_with_adapter = setup_adapter(base_model, config)
        
        # Verify adapter was added
        assert hasattr(model_with_adapter, 'peft_config')
        peft_config = model_with_adapter.peft_config['default']
        
        # Verify it's IA³
        assert isinstance(peft_config, IA3Config)
        
        # Verify trainable parameters (should be much smaller than LoRA)
        trainable, total = model_with_adapter.get_nb_trainable_parameters()
        assert trainable > 0
        assert trainable < total
        print(f"IA³ trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # IA³ should have fewer trainable params than LoRA r=16
        # Typically <1% vs LoRA's 1-3%
        percentage = 100 * trainable / total
        assert percentage < 2.0, f"IA³ should be very parameter-efficient, got {percentage:.2f}%"
    
    def test_adapter_size_comparison(self, base_model, base_model_path):
        """Compare trainable parameter counts across adapter types"""
        results = {}
        
        # Test LoRA r=16
        config_lora = TrainingConfig(
            base_model_name=base_model_path,
            quantization="int8",
            dataset_path="test.jsonl",
            output_dir="test_output",
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_type="lora",
            rank=16,
            alpha=16,
            target_modules=["q_proj", "v_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
            gradient_checkpointing=False,
        )
        
        model_lora = setup_adapter(base_model, config_lora)
        trainable_lora, total = model_lora.get_nb_trainable_parameters()
        results['lora_r16'] = trainable_lora
        
        # Reload base model for next test
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        
        # Test IA³
        config_ia3 = TrainingConfig(
            base_model_name=base_model_path,
            quantization="int8",
            dataset_path="test.jsonl",
            output_dir="test_output",
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_type="ia3",
            rank=None,
            alpha=None,
            target_modules=["q_proj", "v_proj", "o_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
            gradient_checkpointing=False,
        )
        
        model_ia3 = setup_adapter(base_model, config_ia3)
        trainable_ia3, _ = model_ia3.get_nb_trainable_parameters()
        results['ia3'] = trainable_ia3
        
        # Verify IA³ is significantly smaller
        assert trainable_ia3 < trainable_lora, "IA³ should have fewer parameters than LoRA"
        
        print("\n=== Adapter Parameter Comparison ===")
        print(f"LoRA r=16: {results['lora_r16']:,} parameters")
        print(f"IA³:       {results['ia3']:,} parameters")
        print(f"IA³ is {100 * results['ia3'] / results['lora_r16']:.1f}% the size of LoRA")
        
        return results
    
    def test_invalid_adapter_type(self, base_model, base_model_path):
        """Test that invalid adapter types raise errors"""
        config = TrainingConfig(
            base_model_name=base_model_path,
            quantization="int8",
            dataset_path="test.jsonl",
            output_dir="test_output",
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_type="invalid_type",
            rank=16,
            alpha=16,
            target_modules=["q_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
            gradient_checkpointing=False,
        )
        
        with pytest.raises(ValueError, match="Unsupported adapter type"):
            setup_adapter(base_model, config)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

