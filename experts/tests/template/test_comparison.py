"""Comparison tests between base model and expert.

This template provides tests comparing base model vs expert performance.
Copy this file to your expert's tests/ directory and customize for your domain.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
EXPERT_PATH = "../weights/qwen3-06b/adapter"


def load_base_model():
    """Load base model without expert adapter"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    return model, tokenizer


def load_expert_model():
    """Load base model + expert adapter"""
    if not Path(EXPERT_PATH).exists():
        pytest.skip(f"Expert weights not found at {EXPERT_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, EXPERT_PATH)
    return model, tokenizer


def generate_output(model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
    """Generate output from model"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


class TestComparison:
    """Comparison test suite: base model vs expert"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        """Load base model once"""
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load expert model once"""
        return load_expert_model()
    
    def test_improvement_over_base(self, base_model, expert_model):
        """Test that expert performs better than base model"""
        base_m, base_t = base_model
        expert_m, expert_t = expert_model
        
        # TODO: Add test cases where expert should outperform base
        test_cases = [
            "Test case where expert should excel"
        ]
        
        for prompt in test_cases:
            base_output = generate_output(base_m, base_t, prompt)
            expert_output = generate_output(expert_m, expert_t, prompt)
            
            # TODO: Add domain-specific comparison logic
            # Example: Check if expert output is more accurate/formatted correctly
            assert expert_output, "Expert output should not be empty"
            assert base_output, "Base output should not be empty"
            
            # Expert should produce different (hopefully better) output
            # assert expert_output != base_output, "Expert should produce different output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

