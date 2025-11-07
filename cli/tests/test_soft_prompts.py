#!/usr/bin/env python3
"""
Test soft prompt training functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from expert_trainer import configure_soft_prompts, save_soft_prompts
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class MockConfig:
    base_model_name: str = "gpt2"

def test_soft_prompt_config():
    """Test soft prompt configuration"""
    print("Testing soft prompt configuration...")
    
    # Test with TEXT initialization
    soft_prompts = [{
        'name': 'test_prompt',
        'tokens': 16,
        'init_method': 'text',
        'init_text': 'Test prompt initialization',
        'path': 'soft_prompts/test_16.pt'
    }]
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = MockConfig()
    
    # Configure soft prompts
    model = configure_soft_prompts(model, soft_prompts, config.base_model_name, tokenizer)
    
    # Verify prompt encoder exists
    assert hasattr(model, 'prompt_encoder'), "Model should have prompt_encoder"
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0, "Should have trainable parameters"
    
    print(f"  [OK] Soft prompt configured: {trainable} trainable params")
    
    # Test with RANDOM initialization
    soft_prompts_random = [{
        'name': 'test_random',
        'tokens': 32,
        'init_method': 'random',
        'path': 'soft_prompts/test_32.pt'
    }]
    
    model2 = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    model2 = configure_soft_prompts(model2, soft_prompts_random, config.base_model_name, tokenizer)
    
    assert hasattr(model2, 'prompt_encoder'), "Model should have prompt_encoder (random)"
    print("  [OK] Random initialization works")
    
    # Test empty soft prompts (should return model unchanged)
    model3 = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    model3 = configure_soft_prompts(model3, [], config.base_model_name, tokenizer)
    
    assert not hasattr(model3, 'prompt_encoder'), "Should not have prompt_encoder when empty"
    print("  [OK] Empty soft prompts handled correctly")
    
    return True

def test_soft_prompt_save(tmp_path="/tmp/test_soft_prompts"):
    """Test soft prompt saving"""
    print("\nTesting soft prompt saving...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = MockConfig()
    
    soft_prompts = [{
        'name': 'test_save',
        'tokens': 8,
        'init_method': 'random',
        'path': 'soft_prompts/test_save.pt'
    }]
    
    # Configure
    model = configure_soft_prompts(model, soft_prompts, config.base_model_name, tokenizer)
    
    # Save
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    save_soft_prompts(model, soft_prompts, tmp_path)
    
    # Verify file exists
    saved_file = Path(tmp_path) / "soft_prompts" / "test_save.pt"
    assert saved_file.exists(), f"Soft prompt file should be saved at {saved_file}"
    
    # Load and verify
    loaded = torch.load(saved_file)
    assert loaded.shape[0] == 8, f"Should have 8 tokens, got {loaded.shape[0]}"
    
    print(f"  [OK] Soft prompt saved: {saved_file}")
    print(f"  [OK] Shape verified: {loaded.shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)
    
    return True

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Soft Prompt Training Tests")
        print("=" * 60)
        
        test_soft_prompt_config()
        test_soft_prompt_save()
        
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

