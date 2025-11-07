"""Basic tests for TypeScript expert."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "cli"))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    pytest.skip("PyTorch/Transformers not available", allow_module_level=True)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load base model, tokenizer, and adapter."""
    base_model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    adapter_path = Path(__file__).parent.parent / "weights" / "qwen3-06b" / "adapter"
    
    if not adapter_path.exists():
        pytest.skip(f"Adapter not found at {adapter_path}")
    
    print(f"\nLoading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """Generate response from model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()


def test_model_loads(model_and_tokenizer):
    """Test that model and adapter load successfully."""
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None
    print("✅ Model and adapter loaded successfully")


def test_basic_function_generation(model_and_tokenizer):
    """Test generating a simple TypeScript function."""
    model, tokenizer = model_and_tokenizer
    
    prompt = "Create a TypeScript function that checks if a string is a palindrome"
    response = generate_response(model, tokenizer, prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response:\n{response}\n")
    
    # Check for TypeScript keywords
    assert "function" in response.lower() or "const" in response.lower()
    assert "string" in response.lower()
    assert "return" in response.lower()


def test_type_definition(model_and_tokenizer):
    """Test generating TypeScript type definitions."""
    model, tokenizer = model_and_tokenizer
    
    prompt = "Create a TypeScript interface for a product in an e-commerce system"
    response = generate_response(model, tokenizer, prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response:\n{response}\n")
    
    # Check for interface or type keyword
    assert "interface" in response.lower() or "type" in response.lower()


def test_code_explanation(model_and_tokenizer):
    """Test explaining TypeScript code."""
    model, tokenizer = model_and_tokenizer
    
    code = """
const greet = (name: string): string => {
  return `Hello, ${name}!`;
};
"""
    
    prompt = f"Explain what this TypeScript code does:\n{code}"
    response = generate_response(model, tokenizer, prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response:\n{response}\n")
    
    # Check for explanation-related words
    assert len(response) > 20  # Should have meaningful explanation


def test_async_pattern(model_and_tokenizer):
    """Test generating async TypeScript code."""
    model, tokenizer = model_and_tokenizer
    
    prompt = "Create an async TypeScript function that fetches user data from an API"
    response = generate_response(model, tokenizer, prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response:\n{response}\n")
    
    # Check for async keywords
    assert "async" in response.lower()
    assert "await" in response.lower() or "promise" in response.lower()


def test_class_generation(model_and_tokenizer):
    """Test generating a TypeScript class."""
    model, tokenizer = model_and_tokenizer
    
    prompt = "Create a TypeScript class for managing a shopping cart"
    response = generate_response(model, tokenizer, prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response:\n{response}\n")
    
    # Check for class keyword
    assert "class" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

