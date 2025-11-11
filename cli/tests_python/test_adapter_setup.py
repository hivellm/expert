#!/usr/bin/env python3
"""
Tests for adapter_setup.py module
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.config import TrainingConfig
from train.adapter_setup import setup_adapter, configure_soft_prompts, save_soft_prompts


class TestSetupAdapter:
    """Test setup_adapter function"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.config = Mock()
        model.config.use_cache = True
        model.gradient_checkpointing_enable = Mock()
        model.parameters = Mock(return_value=[torch.randn(10, 10)])
        return model
    
    @pytest.fixture
    def lora_config(self):
        """Create LoRA config"""
        return TrainingConfig(
            base_model_name="test-model",
            quantization="none",
            dataset_path="test.jsonl",
            output_dir="output",
            device="cpu",
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
        )
    
    @pytest.fixture
    def dora_config(self):
        """Create DoRA config"""
        return TrainingConfig(
            base_model_name="test-model",
            quantization="none",
            dataset_path="test.jsonl",
            output_dir="output",
            device="cpu",
            adapter_type="dora",
            rank=12,
            alpha=24,
            target_modules=["q_proj", "v_proj"],
            epochs=1,
            learning_rate=0.0001,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            lr_scheduler="linear",
        )
    
    @pytest.fixture
    def ia3_config(self):
        """Create IA³ config"""
        return TrainingConfig(
            base_model_name="test-model",
            quantization="none",
            dataset_path="test.jsonl",
            output_dir="output",
            device="cpu",
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
        )
    
    @patch('train.adapter_setup.get_peft_model')
    def test_setup_lora_adapter(self, mock_get_peft_model, mock_model, lora_config):
        """Test setting up LoRA adapter"""
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters = Mock()
        mock_get_peft_model.return_value = mock_peft_model
        
        result = setup_adapter(mock_model, lora_config)
        
        assert result == mock_peft_model
        mock_get_peft_model.assert_called_once()
        call_args = mock_get_peft_model.call_args
        assert call_args[0][0] == mock_model
        peft_config = call_args[0][1]
        assert peft_config.r == 16
        assert peft_config.lora_alpha == 16
    
    @patch('train.adapter_setup.get_peft_model')
    def test_setup_dora_adapter(self, mock_get_peft_model, mock_model, dora_config):
        """Test setting up DoRA adapter"""
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters = Mock()
        mock_get_peft_model.return_value = mock_peft_model
        
        result = setup_adapter(mock_model, dora_config)
        
        assert result == mock_peft_model
        call_args = mock_get_peft_model.call_args
        peft_config = call_args[0][1]
        assert peft_config.use_dora is True
    
    @patch('train.adapter_setup.get_peft_model')
    def test_setup_ia3_adapter(self, mock_get_peft_model, mock_model, ia3_config):
        """Test setting up IA³ adapter"""
        mock_peft_model = Mock()
        mock_peft_model.print_trainable_parameters = Mock()
        mock_get_peft_model.return_value = mock_peft_model
        
        result = setup_adapter(mock_model, ia3_config)
        
        assert result == mock_peft_model
        call_args = mock_get_peft_model.call_args
        peft_config = call_args[0][1]
        from peft import IA3Config
        assert isinstance(peft_config, IA3Config)
    
    def test_setup_adapter_with_gradient_checkpointing(self, mock_model, lora_config):
        """Test adapter setup with gradient checkpointing enabled"""
        # gradient_checkpointing can be bool or "selective"
        lora_config.gradient_checkpointing = True
        
        # Ensure mock_model.config.use_cache is True initially
        mock_model.config.use_cache = True
        
        with patch('train.adapter_setup.get_peft_model') as mock_get_peft_model:
            mock_peft_model = Mock()
            mock_peft_model.print_trainable_parameters = Mock()
            mock_get_peft_model.return_value = mock_peft_model
            
            setup_adapter(mock_model, lora_config)
            
            # Verify gradient checkpointing was enabled
            # Note: gradient_checkpointing_enable is called before get_peft_model
            assert mock_model.gradient_checkpointing_enable.called
            # Note: use_cache is set to False when gradient_checkpointing is enabled
            # But only if it's "selective", otherwise it's not changed
            # For boolean True, use_cache is not modified
            if lora_config.gradient_checkpointing == "selective":
                assert mock_model.config.use_cache is False
    
    def test_setup_adapter_with_selective_checkpointing(self, mock_model, lora_config):
        """Test adapter setup with selective gradient checkpointing"""
        lora_config.gradient_checkpointing = "selective"
        
        with patch('train.adapter_setup.get_peft_model') as mock_get_peft_model:
            mock_peft_model = Mock()
            mock_peft_model.print_trainable_parameters = Mock()
            mock_get_peft_model.return_value = mock_peft_model
            
            setup_adapter(mock_model, lora_config)
            
            mock_model.gradient_checkpointing_enable.assert_called_once()
    
    def test_setup_adapter_invalid_type(self, mock_model):
        """Test that invalid adapter type raises error"""
        config = TrainingConfig(
            base_model_name="test-model",
            quantization="none",
            dataset_path="test.jsonl",
            output_dir="output",
            device="cpu",
            adapter_type="invalid",
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
        
        with pytest.raises(ValueError, match="Unsupported adapter type"):
            setup_adapter(mock_model, config)


class TestConfigureSoftPrompts:
    """Test configure_soft_prompts function"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.parameters = Mock(return_value=[torch.randn(10, 10)])
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer"""
        return Mock()
    
    def test_configure_soft_prompts_empty_list(self, mock_model, mock_tokenizer):
        """Test configuring with empty soft prompts list"""
        result = configure_soft_prompts(mock_model, [], "test-model", mock_tokenizer)
        assert result == mock_model
    
    def test_configure_soft_prompts_none(self, mock_model, mock_tokenizer):
        """Test configuring with None soft prompts"""
        result = configure_soft_prompts(mock_model, None, "test-model", mock_tokenizer)
        assert result == mock_model
    
    @patch('train.adapter_setup.get_peft_model')
    def test_configure_soft_prompts_random_init(self, mock_get_peft_model, mock_model, mock_tokenizer):
        """Test configuring soft prompts with random initialization"""
        mock_peft_model = Mock()
        mock_peft_model.parameters = Mock(return_value=[torch.randn(10, 10)])
        mock_get_peft_model.return_value = mock_peft_model
        
        soft_prompts = [{
            "name": "test_prompt",
            "tokens": 32,
            "init_method": "random",
        }]
        
        result = configure_soft_prompts(mock_model, soft_prompts, "test-model", mock_tokenizer)
        
        assert result == mock_peft_model
        mock_get_peft_model.assert_called_once()
        call_args = mock_get_peft_model.call_args
        peft_config = call_args[0][1]
        assert peft_config.num_virtual_tokens == 32
    
    @patch('train.adapter_setup.get_peft_model')
    def test_configure_soft_prompts_text_init(self, mock_get_peft_model, mock_model, mock_tokenizer):
        """Test configuring soft prompts with text initialization"""
        mock_peft_model = Mock()
        mock_peft_model.parameters = Mock(return_value=[torch.randn(10, 10)])
        mock_get_peft_model.return_value = mock_peft_model
        
        soft_prompts = [{
            "name": "test_prompt",
            "tokens": 32,
            "init_method": "text",
            "init_text": "This is a test prompt",
        }]
        
        result = configure_soft_prompts(mock_model, soft_prompts, "test-model", mock_tokenizer)
        
        assert result == mock_peft_model
        call_args = mock_get_peft_model.call_args
        peft_config = call_args[0][1]
        assert peft_config.prompt_tuning_init == "TEXT"
        assert peft_config.prompt_tuning_init_text == "This is a test prompt"


class TestSaveSoftPrompts:
    """Test save_soft_prompts function"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with prompt_encoder"""
        model = Mock()
        mock_embedding = Mock()
        mock_embedding.weight = torch.randn(32, 768)
        mock_encoder = Mock()
        mock_encoder.embedding = mock_embedding
        model.prompt_encoder = {"default": mock_encoder}
        return model
    
    def test_save_soft_prompts_empty_list(self, mock_model, tmp_path):
        """Test saving with empty soft prompts list"""
        save_soft_prompts(mock_model, [], tmp_path)
        # Should not raise error
    
    def test_save_soft_prompts_none(self, mock_model, tmp_path):
        """Test saving with None soft prompts"""
        save_soft_prompts(mock_model, None, tmp_path)
        # Should not raise error
    
    @patch('train.adapter_setup.torch.save')
    def test_save_soft_prompts_with_encoder(self, mock_torch_save, mock_model, tmp_path):
        """Test saving soft prompts when model has prompt_encoder"""
        soft_prompts = [{
            "name": "test_prompt",
            "path": "prompts/test.pt",
        }]
        
        save_soft_prompts(mock_model, soft_prompts, tmp_path)
        
        # Verify torch.save was called
        assert mock_torch_save.called
    
    def test_save_soft_prompts_no_encoder(self, tmp_path):
        """Test saving when model doesn't have prompt_encoder"""
        model = Mock()
        model.prompt_encoder = None
        
        soft_prompts = [{
            "name": "test_prompt",
            "path": "prompts/test.pt",
        }]
        
        # Should not raise error, just print warning
        save_soft_prompts(model, soft_prompts, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

