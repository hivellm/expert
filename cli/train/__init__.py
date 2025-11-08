"""
Expert Training Module - Refactored from expert_trainer.py

This module provides the main training function and exports it for use by Rust code.
"""

from .trainer import train_expert

__all__ = ["train_expert"]

