"""Basic functionality tests for expert validation.

This template provides standard tests for basic expert capabilities.
Copy this file to your expert's tests/ directory and customize for your domain.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List

# Base model path (adjust as needed)
BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
EXPERT_PATH = "../weights/qwen3-06b/adapter"


def load_expert_model():
    """Load base model + expert adapter"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        if not Path(EXPERT_PATH).exists():
            pytest.skip(f"Expert weights not found at {EXPERT_PATH}")
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(model, EXPERT_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        
        return model, tokenizer
    except ImportError:
        pytest.skip("Required dependencies not installed (torch, transformers, peft)")


def generate_output(model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
    """Generate output from model"""
    import torch
    
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


class TestBasicFunctionality:
    """Basic functionality test suite"""
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load model once for all tests"""
        return load_expert_model()
    
    def test_simple_generation(self, expert_model):
        """Test basic generation capability"""
        model, tokenizer = expert_model
        
        # TODO: Customize prompt for your expert domain
        prompt = "Generate a simple example"
        output = generate_output(model, tokenizer, prompt)
        
        assert output, "Output should not be empty"
        assert len(output) > 10, "Output should have reasonable length"
    
    def test_format_compliance(self, expert_model):
        """Test that output follows expected format"""
        model, tokenizer = expert_model
        
        # TODO: Customize prompt and validation for your expert domain
        prompt = "Generate formatted output"
        output = generate_output(model, tokenizer, prompt)
        
        # TODO: Add format-specific validation
        # Example for JSON expert:
        # try:
        #     json.loads(output)
        # except json.JSONDecodeError:
        #     pytest.fail("Output should be valid JSON")
        
        assert output, "Output should not be empty"
    
    def test_basic_accuracy(self, expert_model):
        """Test basic accuracy on simple cases"""
        model, tokenizer = expert_model
        
        # TODO: Add domain-specific test cases
        test_cases = [
            {
                "prompt": "Simple test case 1",
                "expected_keywords": ["keyword1", "keyword2"]
            }
        ]
        
        for case in test_cases:
            output = generate_output(model, tokenizer, case["prompt"])
            
            # Check for expected keywords
            for keyword in case.get("expected_keywords", []):
                assert keyword.lower() in output.lower(), f"Expected keyword '{keyword}' not found in output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

