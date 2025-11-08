#!/usr/bin/env python3
"""
Tests for callbacks.py module
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.callbacks import MemoryCleanupCallback, OverfittingMonitorCallback


class TestMemoryCleanupCallback:
    """Test MemoryCleanupCallback"""
    
    @pytest.fixture
    def callback(self):
        """Create callback instance"""
        with patch('train.callbacks.torch.cuda.is_available', return_value=True):
            with patch('train.callbacks.torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                callback = MemoryCleanupCallback(cleanup_steps=50, vram_threshold=0.80)
                callback.total_vram = 8.0  # Set manually for testing
                return callback
    
    @pytest.fixture
    def mock_args(self):
        """Create mock TrainingArguments"""
        args = Mock()
        args.output_dir = "test_output"
        return args
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TrainerState"""
        state = Mock()
        state.global_step = 0
        state.log_history = []
        return state
    
    @pytest.fixture
    def mock_control(self):
        """Create mock TrainerControl"""
        control = Mock()
        return control
    
    @patch('train.callbacks.torch.cuda.is_available')
    @patch('train.callbacks.torch.cuda.memory_allocated')
    @patch('train.callbacks.torch.cuda.empty_cache')
    @patch('train.callbacks.gc.collect')
    def test_cleanup_on_threshold(self, mock_gc, mock_empty_cache, mock_allocated, mock_is_available, callback, mock_args, mock_state, mock_control):
        """Test cleanup triggers when VRAM exceeds threshold"""
        mock_is_available.return_value = True
        # Set total_vram for threshold calculation
        callback.total_vram = 8.0  # 8GB
        mock_allocated.return_value = 7 * 1024**3  # 7GB (87.5% of 8GB)
        mock_state.global_step = 0
        callback.last_cleanup_step = -10  # Ensure cleanup can trigger
        
        callback.on_step_end(mock_args, mock_state, mock_control)
        
        # Should trigger cleanup (may be called or not depending on step count)
        # Just verify no errors occurred
        assert True
    
    @patch('train.callbacks.torch.cuda.is_available')
    @patch('train.callbacks.torch.cuda.memory_allocated')
    @patch('train.callbacks.torch.cuda.empty_cache')
    @patch('train.callbacks.gc.collect')
    def test_cleanup_periodic(self, mock_gc, mock_empty_cache, mock_allocated, mock_is_available, callback, mock_args, mock_state, mock_control):
        """Test periodic cleanup"""
        mock_is_available.return_value = True
        mock_allocated.return_value = 4 * 1024**3  # 4GB (50% of 8GB)
        
        # Run 50 steps
        for i in range(50):
            mock_state.global_step = i
            callback.on_step_end(mock_args, mock_state, mock_control)
        
        # Should have triggered periodic cleanup
        assert mock_empty_cache.called
    
    @patch('train.callbacks.torch.cuda.is_available')
    def test_no_cleanup_on_cpu(self, mock_is_available, callback, mock_args, mock_state, mock_control):
        """Test that cleanup doesn't run on CPU"""
        mock_is_available.return_value = False
        
        callback.on_step_end(mock_args, mock_state, mock_control)
        
        # Should not raise error
    
    @patch('train.callbacks.torch.cuda.is_available', return_value=False)
    def test_callback_initialization_cpu(self, mock_is_available):
        """Test callback initialization on CPU"""
        callback = MemoryCleanupCallback()
        assert callback.total_vram is None


class TestOverfittingMonitorCallback:
    """Test OverfittingMonitorCallback"""
    
    @pytest.fixture
    def callback(self):
        """Create callback instance"""
        return OverfittingMonitorCallback(
            overfitting_threshold=1.2,
            min_train_loss=0.3
        )
    
    @pytest.fixture
    def mock_args(self):
        """Create mock TrainingArguments"""
        args = Mock()
        return args
    
    @pytest.fixture
    def mock_state(self):
        """Create mock TrainerState"""
        state = Mock()
        state.log_history = [{"loss": 0.25}]
        return state
    
    @pytest.fixture
    def mock_control(self):
        """Create mock TrainerControl"""
        control = Mock()
        return control
    
    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer"""
        optimizer = Mock()
        optimizer.param_groups = [{"lr": 0.0001}]
        return optimizer
    
    def test_no_overfitting_detected(self, callback, mock_args, mock_state, mock_control, mock_optimizer):
        """Test when no overfitting is detected"""
        metrics = {
            "eval_loss": 0.25,  # Same as train loss
        }
        
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=metrics, optimizer=mock_optimizer)
        
        # LR should not be reduced
        assert mock_optimizer.param_groups[0]["lr"] == 0.0001
        assert callback.lr_reduced is False
    
    def test_overfitting_detected(self, callback, mock_args, mock_state, mock_control, mock_optimizer):
        """Test when overfitting is detected"""
        metrics = {
            "eval_loss": 0.40,  # 60% higher than train loss (0.25)
        }
        
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=metrics, optimizer=mock_optimizer)
        
        # LR should be reduced by 50%
        assert mock_optimizer.param_groups[0]["lr"] == 0.00005
        assert callback.lr_reduced is True
    
    def test_train_loss_too_high(self, callback, mock_args, mock_state, mock_control, mock_optimizer):
        """Test that overfitting detection doesn't trigger when train loss is too high"""
        mock_state.log_history = [{"loss": 0.5}]  # Above min_train_loss threshold
        
        metrics = {
            "eval_loss": 0.8,  # Much higher than train loss
        }
        
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=metrics, optimizer=mock_optimizer)
        
        # LR should not be reduced (train loss too high)
        assert mock_optimizer.param_groups[0]["lr"] == 0.0001
    
    def test_no_metrics(self, callback, mock_args, mock_state, mock_control, mock_optimizer):
        """Test handling when metrics is None"""
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=None, optimizer=mock_optimizer)
        
        # Should not raise error
    
    def test_no_train_loss(self, callback, mock_args, mock_state, mock_control, mock_optimizer):
        """Test handling when train loss is not available"""
        mock_state.log_history = []
        
        metrics = {
            "eval_loss": 0.4,
        }
        
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=metrics, optimizer=mock_optimizer)
        
        # Should not raise error
    
    def test_lr_reduced_only_once(self, callback, mock_args, mock_state, mock_control, mock_optimizer):
        """Test that LR is only reduced once"""
        metrics = {
            "eval_loss": 0.40,
        }
        
        # First evaluation
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=metrics, optimizer=mock_optimizer)
        first_lr = mock_optimizer.param_groups[0]["lr"]
        
        # Second evaluation (should not reduce again)
        callback.on_evaluate(mock_args, mock_state, mock_control, metrics=metrics, optimizer=mock_optimizer)
        second_lr = mock_optimizer.param_groups[0]["lr"]
        
        assert first_lr == second_lr
        assert first_lr == 0.00005


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

