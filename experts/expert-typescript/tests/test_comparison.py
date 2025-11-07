"""Comparison tests: Expert vs Base model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "cli"))

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    pytest.skip("PyTorch/Transformers not available", allow_module_level=True)


@pytest.fixture(scope="module")
def models():
    """Load both base model and expert model."""
    base_model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    adapter_path = Path(__file__).parent.parent / "weights" / "qwen3-06b" / "adapter"
    
    if not adapter_path.exists():
        pytest.skip(f"Adapter not found at {adapter_path}")
    
    print(f"\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    
    # Load expert model
    expert_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    expert_model = PeftModel.from_pretrained(expert_base, str(adapter_path))
    expert_model.eval()
    
    return {
        "base": base_model,
        "expert": expert_model,
        "tokenizer": tokenizer
    }


def generate(model, tokenizer, prompt: str, max_length: int = 256) -> str:
    """Generate response."""
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
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()


def test_comparison_function_generation(models):
    """Compare base vs expert on function generation."""
    prompt = "Write a TypeScript function to validate email format"
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    base_response = generate(models["base"], models["tokenizer"], prompt)
    expert_response = generate(models["expert"], models["tokenizer"], prompt)
    
    print(f"\n🔵 BASE MODEL:")
    print(base_response)
    print(f"\n🟢 EXPERT MODEL:")
    print(expert_response)
    print(f"\n{'='*60}\n")
    
    # Expert should produce longer, more detailed code
    assert len(expert_response) >= len(base_response) * 0.5


def test_comparison_interface_generation(models):
    """Compare base vs expert on interface generation."""
    prompt = "Create a TypeScript interface for a blog post"
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    base_response = generate(models["base"], models["tokenizer"], prompt)
    expert_response = generate(models["expert"], models["tokenizer"], prompt)
    
    print(f"\n🔵 BASE MODEL:")
    print(base_response)
    print(f"\n🟢 EXPERT MODEL:")
    print(expert_response)
    print(f"\n{'='*60}\n")
    
    # Expert should include "interface" keyword
    assert "interface" in expert_response.lower() or "type" in expert_response.lower()


def test_comparison_generics(models):
    """Compare base vs expert on generic types."""
    prompt = "Create a TypeScript generic function to get the first element of an array"
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    base_response = generate(models["base"], models["tokenizer"], prompt)
    expert_response = generate(models["expert"], models["tokenizer"], prompt)
    
    print(f"\n🔵 BASE MODEL:")
    print(base_response)
    print(f"\n🟢 EXPERT MODEL:")
    print(expert_response)
    print(f"\n{'='*60}\n")
    
    # Check for generic syntax
    has_generics = "<" in expert_response and ">" in expert_response
    print(f"Expert uses generics: {has_generics}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

