#!/usr/bin/env python3
"""
Tests for model_loader.py module
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.config import TrainingConfig
from train.model_loader import get_quantization_config, load_model_and_tokenizer


class TestGetQuantizationConfig:
    """Test get_quantization_config function"""
    
    def test_get_int4_config(self):
        """Test getting int4 quantization config"""
        config = get_quantization_config("int4")
        
        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == torch.bfloat16
        assert config.bnb_4bit_use_double_quant is True
    
    def test_get_int8_config(self):
        """Test getting int8 quantization config"""
        config = get_quantization_config("int8")
        
        assert config is not None
        assert config.load_in_8bit is True
    
    def test_get_none_config(self):
        """Test getting config for no quantization"""
        config = get_quantization_config("none")
        
        assert config is None
    
    def test_get_invalid_config(self):
        """Test getting config for invalid quantization"""
        config = get_quantization_config("invalid")
        
        assert config is None


class TestLoadModelAndTokenizer:
    """Test load_model_and_tokenizer function"""
    
    @pytest.fixture
    def config(self):
        """Create test config"""
        return TrainingConfig(
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
    
    @patch('train.model_loader.load_model_standard')
    def test_load_without_unsloth(self, mock_load_standard, config):
        """Test loading model without Unsloth"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_standard.return_value = (mock_model, mock_tokenizer)
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_load_standard.assert_called_once_with(config)
    
    @patch('train.model_loader.load_model_with_unsloth')
    @patch('train.model_loader.USE_UNSLOTH', True)
    def test_load_with_unsloth(self, mock_load_unsloth, config):
        """Test loading model with Unsloth when available"""
        config.use_unsloth = True
        config.quantization = "int4"
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_unsloth.return_value = (mock_model, mock_tokenizer)
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_load_unsloth.assert_called_once_with(config)
    
    @patch('train.model_loader.load_model_standard')
    @patch('train.model_loader.USE_UNSLOTH', False)
    def test_load_unsloth_not_available(self, mock_load_standard, config):
        """Test loading when Unsloth is requested but not available"""
        config.use_unsloth = True
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_standard.return_value = (mock_model, mock_tokenizer)
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Should fallback to standard loader
        mock_load_standard.assert_called_once_with(config)
    
    @patch('train.model_loader.load_model_standard')
    @patch('train.model_loader.USE_UNSLOTH', True)
    def test_load_unsloth_wrong_quantization(self, mock_load_standard, config):
        """Test loading when Unsloth is available but quantization is incompatible"""
        config.use_unsloth = True
        config.quantization = "none"  # Unsloth requires int4/int8
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_standard.return_value = (mock_model, mock_tokenizer)
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Should fallback to standard loader
        mock_load_standard.assert_called_once_with(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

