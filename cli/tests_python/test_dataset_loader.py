#!/usr/bin/env python3
"""
Tests for dataset_loader.py module
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from datasets import Dataset, DatasetDict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.config import TrainingConfig
from train.dataset_loader import load_and_prepare_dataset, load_multi_task_dataset


class TestLoadAndPrepareDataset:
    """Test load_and_prepare_dataset function"""
    
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
            dataset_type="single",
        )
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        return tokenizer
    
    @patch('train.dataset_loader.load_from_disk')
    def test_load_pretokenized_cache(self, mock_load_disk, config, mock_tokenizer):
        """Test loading from pre-tokenized cache"""
        config.pretokenized_cache = "cache_path"
        
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3]]}),
            "validation": Dataset.from_dict({"input_ids": [[4, 5, 6]]}),
        })
        mock_load_disk.return_value = mock_dataset
        
        result = load_and_prepare_dataset(config, mock_tokenizer)
        
        assert "train" in result
        assert "test" in result
        mock_load_disk.assert_called_once()
    
    @patch('train.dataset_loader.load_dataset')
    def test_load_single_file(self, mock_load_dataset, config, mock_tokenizer):
        """Test loading single file dataset"""
        config.pretokenized_cache = None
        
        mock_dataset = Dataset.from_dict({
            "instruction": ["test instruction"],
            "response": ["test response"],
        })
        mock_load_dataset.return_value = mock_dataset
        
        # Mock tokenization
        with patch('train.dataset_loader.Dataset.map') as mock_map:
            mock_map.return_value = Dataset.from_dict({
                "input_ids": [[1, 2, 3]],
                "labels": [[1, 2, 3]],
            })
            
            result = load_and_prepare_dataset(config, mock_tokenizer)
            
            assert result is not None
    
    @patch('train.dataset_loader.load_dataset')
    def test_load_with_validation_path(self, mock_load_dataset, config, mock_tokenizer):
        """Test loading dataset with separate validation file"""
        config.validation_path = "val.jsonl"
        config.pretokenized_cache = None
        
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"instruction": ["train"]}),
            "validation": Dataset.from_dict({"instruction": ["val"]}),
        })
        mock_load_dataset.return_value = mock_dataset
        
        with patch('train.dataset_loader.DatasetDict') as mock_dict:
            mock_dict.return_value = mock_dataset
            
            with patch('train.dataset_loader.Dataset.map') as mock_map:
                mock_map.return_value = Dataset.from_dict({
                    "input_ids": [[1, 2, 3]],
                })
                
                result = load_and_prepare_dataset(config, mock_tokenizer)
                
                assert result is not None
    
    def test_load_multi_task_dataset_type(self, config, mock_tokenizer):
        """Test loading multi-task dataset"""
        config.dataset_type = "multi_task"
        config.multi_task_config = {
            "tasks": [
                {
                    "name": "task1",
                    "path": "task1.jsonl",
                }
            ],
        }
        
        with patch('train.dataset_loader.load_multi_task_dataset') as mock_load:
            mock_load.return_value = {
                "train": Dataset.from_dict({"input": ["test"]}),
                "test": Dataset.from_dict({"input": ["test"]}),
            }
            
            result = load_and_prepare_dataset(config, mock_tokenizer)
            
            assert result is not None
            mock_load.assert_called_once_with(config, mock_tokenizer)


class TestLoadMultiTaskDataset:
    """Test load_multi_task_dataset function"""
    
    @pytest.fixture
    def config(self):
        """Create test config"""
        return TrainingConfig(
            base_model_name="test-model",
            quantization="none",
            dataset_path="",
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
            dataset_type="multi_task",
            multi_task_config={
                "tasks": [
                    {
                        "name": "task1",
                        "path": "task1.jsonl",
                    }
                ],
            },
        )
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        return tokenizer
    
    @patch('train.dataset_loader.load_from_disk')
    def test_load_multi_task_from_cache(self, mock_load_disk, config, mock_tokenizer):
        """Test loading multi-task dataset from cache"""
        config.pretokenized_cache = "cache_path"
        
        mock_dataset = DatasetDict({
            "train": Dataset.from_dict({"input": ["test"]}),
            "validation": Dataset.from_dict({"input": ["test"]}),
        })
        mock_load_disk.return_value = mock_dataset
        
        result = load_multi_task_dataset(config, mock_tokenizer)
        
        assert "train" in result
        assert "test" in result
        mock_load_disk.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

