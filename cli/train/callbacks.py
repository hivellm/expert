"""
Training Callbacks Module

Contains callbacks for memory cleanup, overfitting monitoring, and progress testing.
"""

import gc
import torch
from transformers import TrainerCallback

# Import psutil for system RAM monitoring (Windows-specific optimization)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MemoryCleanupCallback(TrainerCallback):
    """
    Dynamically clean up VRAM during training to prevent GPU memory leaks.
    
    Problem: Dataset batches are loaded into VRAM and never released.
    PyTorch keeps references to intermediate tensors for autograd.
    
    Strategy:
    1. Monitor VRAM usage every step (fast check)
    2. If VRAM > 80% of total, trigger immediate cleanup
    3. Also cleanup periodically as fallback (every N steps)
    
    Key differences from CPU GC:
    - torch.cuda.empty_cache() releases VRAM blocks back to CUDA
    - gc.collect() only helps with Python object references
    - Need both: gc.collect() to release Python refs, then empty_cache() for VRAM
    """
    
    def __init__(self, cleanup_steps=50, vram_threshold=0.80):
        self.cleanup_steps = cleanup_steps
        self.vram_threshold = vram_threshold  # 80% threshold
        self.step_count = 0
        self.total_vram = None
        self.last_cleanup_step = 0
        
        # Get total VRAM at initialization
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.total_vram = props.total_memory / 1024**3  # GB
            print(f"   [VRAM Monitor] Total VRAM: {self.total_vram:.2f}GB")
            print(f"   [VRAM Monitor] Cleanup threshold: {self.vram_threshold*100:.0f}% ({self.total_vram * self.vram_threshold:.2f}GB)")
        
    def _cleanup_vram(self, step, reason="periodic"):
        """Perform aggressive VRAM and RAM cleanup (Windows-optimized)"""
        # Get memory before cleanup
        ram_before = psutil.virtual_memory().used / 1024**3 if HAS_PSUTIL else 0
        vram_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        # Step 1: Force Python garbage collection (aggressive)
        for _ in range(3):  # Multiple passes to catch circular references
            gc.collect()
        
        # Step 2: Clear CUDA cache (THIS is what frees VRAM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Windows-specific: Reset peak memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
        
        # Get memory after cleanup
        ram_after = psutil.virtual_memory().used / 1024**3 if HAS_PSUTIL else 0
        vram_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        ram_freed = ram_before - ram_after
        vram_freed = vram_before - vram_after
        
        # Log if freed significant memory (> 100MB)
        if vram_freed > 0.1 or ram_freed > 0.1:
            print(f"\n[CLEANUP] Step {step} ({reason})")
            if vram_freed > 0.1:
                print(f"  VRAM: {vram_before:.2f}GB -> {vram_after:.2f}GB (freed {vram_freed:.2f}GB)")
            if HAS_PSUTIL and ram_freed > 0.1:
                print(f"  RAM:  {ram_before:.2f}GB -> {ram_after:.2f}GB (freed {ram_freed:.2f}GB)")
        
        self.last_cleanup_step = step
        
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor VRAM and cleanup if needed"""
        self.step_count += 1
        
        if not torch.cuda.is_available():
            return
        
        # Check current VRAM usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        usage_percent = allocated / self.total_vram if self.total_vram else 0
        
        # TRIGGER 1: VRAM above threshold (80%)
        if usage_percent >= self.vram_threshold:
            # Avoid cleaning too frequently (minimum 10 steps between cleanups)
            if state.global_step - self.last_cleanup_step >= 10:
                self._cleanup_vram(state.global_step, f"threshold {usage_percent*100:.1f}%")
        
        # TRIGGER 2: Periodic cleanup (fallback)
        elif self.step_count % self.cleanup_steps == 0:
            self._cleanup_vram(state.global_step, "periodic")
        
        # Log memory status every 250 steps
        if self.step_count % 250 == 0:
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"\n[MEMORY] Step {state.global_step}")
            print(f"  VRAM: {allocated:.2f}GB / {self.total_vram:.2f}GB ({usage_percent*100:.1f}%) | Reserved: {reserved:.2f}GB")
            
            if HAS_PSUTIL:
                ram_used = psutil.virtual_memory().used / 1024**3
                ram_total = psutil.virtual_memory().total / 1024**3
                ram_percent = psutil.virtual_memory().percent
                print(f"  RAM:  {ram_used:.2f}GB / {ram_total:.2f}GB ({ram_percent:.1f}%)")


class OverfittingMonitorCallback(TrainerCallback):
    """
    Monitor for early overfitting signals in training.
    
    Detects when train loss diverges from eval loss (sign of overfitting),
    especially critical for synthetic datasets that may not generalize to real queries.
    
    Triggers:
    - Train loss < 0.3 (converged) AND eval_loss > train_loss * 1.2 (20% gap)
    - Action: Reduce LR by 50% to stabilize and improve generalization
    """
    
    def __init__(self, overfitting_threshold=1.2, min_train_loss=0.3):
        self.overfitting_threshold = overfitting_threshold
        self.min_train_loss = min_train_loss
        self.lr_reduced = False
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check for overfitting after each evaluation"""
        if metrics is None:
            return
        
        eval_loss = metrics.get("eval_loss")
        train_loss = state.log_history[-1].get("loss") if state.log_history else None
        
        if eval_loss is None or train_loss is None:
            return
        
        # Check for overfitting signal
        if train_loss < self.min_train_loss and eval_loss > train_loss * self.overfitting_threshold:
            if not self.lr_reduced:
                print(f"\n⚠️  [OVERFITTING DETECTED]")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Eval Loss:  {eval_loss:.4f}")
                print(f"   Gap: {(eval_loss/train_loss - 1)*100:.1f}%")
                print(f"   Action: Reducing LR by 50% to improve generalization")
                
                # Reduce learning rate
                optimizer = kwargs.get("optimizer")
                if optimizer:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        print(f"   New LR: {param_group['lr']:.6f}")
                
                self.lr_reduced = True
        
        # Log status
        if eval_loss and train_loss:
            gap_percent = (eval_loss / train_loss - 1) * 100
            status = "✓" if gap_percent < 20 else "⚠️"
            print(f"   {status} Train/Eval Gap: {gap_percent:+.1f}%")

