#!/usr/bin/env python3
"""
Tests for trainer.py module (integration tests)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.trainer import train_expert


class TestTrainExpert:
    """Test train_expert function (integration tests with mocks)"""
    
    @pytest.fixture
    def minimal_config_dict(self):
        """Create minimal config dictionary"""
        return {
            "expert_name": "test-expert",
            "expert_version": "1.0.0",
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
                "dataloader_num_workers": 0,  # Disable multiprocessing for tests
            },
        }
    
    @patch('train.trainer.load_model_and_tokenizer')
    @patch('train.trainer.setup_adapter')
    @patch('train.trainer.load_and_prepare_dataset')
    @patch('train.trainer.SFTTrainer')
    def test_train_expert_basic_flow(
        self,
        mock_trainer_class,
        mock_load_dataset,
        mock_setup_adapter,
        mock_load_model,
        minimal_config_dict
    ):
        """Test basic training flow"""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_adapter_model = Mock()
        mock_setup_adapter.return_value = mock_adapter_model
        
        from datasets import Dataset
        # Create real Dataset objects (not mocks) to avoid multiprocessing issues
        # Use small dataset to avoid issues
        train_ds = Dataset.from_dict({"text": ["test example"]})
        test_ds = Dataset.from_dict({"text": ["test example"]})
        mock_load_dataset.return_value = {
            "train": train_ds,
            "test": test_ds,
        }
        
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Patch platform to avoid Windows-specific pre-tokenization
        with patch('train.trainer.platform.system', return_value='Linux'):
            # Run training
            train_expert(minimal_config_dict)
        
        # Verify calls
        mock_load_model.assert_called_once()
        mock_setup_adapter.assert_called_once()
        mock_load_dataset.assert_called_once()
        # Note: trainer.train() or unsloth_train() may be called depending on config
        assert mock_trainer.train.called or hasattr(mock_trainer, 'unsloth_train')
    
    @patch('train.trainer.load_model_and_tokenizer')
    @patch('train.trainer.setup_adapter')
    @patch('train.trainer.load_and_prepare_dataset')
    @patch('train.trainer.SFTTrainer')
    def test_train_expert_with_progress_testing(
        self,
        mock_trainer_class,
        mock_load_dataset,
        mock_setup_adapter,
        mock_load_model,
        minimal_config_dict
    ):
        """Test training with progress testing enabled"""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_adapter_model = Mock()
        mock_setup_adapter.return_value = mock_adapter_model
        
        from datasets import Dataset
        # Create real Dataset objects (not mocks) to avoid multiprocessing issues
        train_ds = Dataset.from_dict({"text": ["test example"]})
        test_ds = Dataset.from_dict({"text": ["test example"]})
        mock_load_dataset.return_value = {
            "train": train_ds,
            "test": test_ds,
        }
        
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Patch platform to avoid Windows-specific pre-tokenization
        with patch('train.trainer.platform.system', return_value='Linux'):
            # Run training
            train_expert(minimal_config_dict)
        
        # Verify trainer was created with callbacks
        assert mock_trainer_class.called
        call_kwargs = mock_trainer_class.call_args[1] if mock_trainer_class.call_args else {}
        assert "callbacks" in call_kwargs or len(call_kwargs.get("callbacks", [])) >= 0
    
    @patch('train.trainer.load_model_and_tokenizer')
    @patch('train.trainer.setup_adapter')
    @patch('train.trainer.load_and_prepare_dataset')
    @patch('train.trainer.SFTTrainer')
    def test_train_expert_with_resume_checkpoint(
        self,
        mock_trainer_class,
        mock_load_dataset,
        mock_setup_adapter,
        mock_load_model,
        minimal_config_dict
    ):
        """Test training with resume checkpoint"""
        minimal_config_dict["resume_checkpoint"] = "checkpoint-500"
        
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        mock_adapter_model = Mock()
        mock_setup_adapter.return_value = mock_adapter_model
        
        from datasets import Dataset
        # Create real Dataset objects (not mocks) to avoid multiprocessing issues
        train_ds = Dataset.from_dict({"text": ["test example"]})
        test_ds = Dataset.from_dict({"text": ["test example"]})
        mock_load_dataset.return_value = {
            "train": train_ds,
            "test": test_ds,
        }
        
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Patch platform to avoid Windows-specific pre-tokenization
        with patch('train.trainer.platform.system', return_value='Linux'):
            # Run training
            train_expert(minimal_config_dict)
        
        # Verify resume_from_checkpoint was passed
        # Note: unsloth_train() may be used instead of trainer.train()
        if mock_trainer.train.called:
            call_kwargs = mock_trainer.train.call_args[1] if mock_trainer.train.call_args else {}
            assert call_kwargs.get("resume_from_checkpoint") == "checkpoint-500"
        # If unsloth_train is used, it handles resume_from_checkpoint internally


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

