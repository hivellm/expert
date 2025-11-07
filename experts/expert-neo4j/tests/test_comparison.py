"""A/B comparison tests: Base model vs Neo4j Expert"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
EXPERT_PATH = "../weights/qwen3-06b/adapter"


def load_base_model():
    """Load base model without expert"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


def load_expert_model():
    """Load base model + expert adapter"""
    if not os.path.exists(EXPERT_PATH):
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


def generate_cypher(model, tokenizer, schema, question, max_tokens=200):
    """Generate Cypher query using ChatML format (Qwen3 native)"""
    # Qwen3 uses ChatML format with <|im_start|> and <|im_end|>
    prompt = f"""<|im_start|>user
{question}

Schema:
{schema}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the response part (after assistant marker)
    if "<|im_start|>assistant" in response:
        cypher = response.split("<|im_start|>assistant")[-1].strip()
        cypher = cypher.replace("<|im_end|>", "").strip()
    else:
        cypher = response.strip()
    
    return cypher


def validate_cypher_syntax(cypher):
    """Basic Cypher syntax validation"""
    cypher_upper = cypher.upper()
    
    # Must have at least one key command
    has_command = any(cmd in cypher_upper for cmd in ["MATCH", "CREATE", "MERGE", "RETURN"])
    
    # Check for common Neo4j elements
    has_nodes_or_rels = "(" in cypher or "[" in cypher
    
    return has_command and has_nodes_or_rels


class TestComparison:
    """Compare base model vs expert on Cypher generation"""
    
    @pytest.fixture(scope="class")
    def models(self):
        """Load both models"""
        base = load_base_model()
        expert = load_expert_model()
        return {"base": base, "expert": expert}
    
    def test_basic_query_comparison(self, models):
        """Compare basic query generation"""
        schema = """Node properties:
- **Person**
  - `name`: STRING
Relationships:
None"""
        
        question = "Find all people named John"
        
        # Base model
        base_model, base_tokenizer = models["base"]
        base_output = generate_cypher(base_model, base_tokenizer, schema, question)
        
        # Expert model
        expert_model, expert_tokenizer = models["expert"]
        expert_output = generate_cypher(expert_model, expert_tokenizer, schema, question)
        
        print(f"\nBase model output:\n{base_output}\n")
        print(f"Expert model output:\n{expert_output}\n")
        
        # Expert should produce more valid Cypher
        base_valid = validate_cypher_syntax(base_output)
        expert_valid = validate_cypher_syntax(expert_output)
        
        print(f"Base valid: {base_valid}, Expert valid: {expert_valid}")
        
        # Expert should be at least as good as base
        if base_valid:
            assert expert_valid, "Expert should produce valid Cypher when base does"
    
    def test_complex_query_comparison(self, models):
        """Compare complex query generation"""
        schema = """Node properties:
- **Movie**
  - `title`: STRING
  - `released`: INTEGER
- **Person**
  - `name`: STRING
  - `born`: INTEGER
Relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)"""
        
        question = "Find all movies directed by people born in the 1960s"
        
        # Base model
        base_model, base_tokenizer = models["base"]
        base_output = generate_cypher(base_model, base_tokenizer, schema, question)
        
        # Expert model
        expert_model, expert_tokenizer = models["expert"]
        expert_output = generate_cypher(expert_model, expert_tokenizer, schema, question)
        
        print(f"\nBase model output:\n{base_output}\n")
        print(f"Expert model output:\n{expert_output}\n")
        
        # Check for key elements
        base_has_directed = "DIRECTED" in base_output.upper()
        expert_has_directed = "DIRECTED" in expert_output.upper()
        
        base_has_filter = "WHERE" in base_output.upper() or "born" in base_output.lower()
        expert_has_filter = "WHERE" in expert_output.upper() or "born" in expert_output.lower()
        
        print(f"Base has DIRECTED: {base_has_directed}, has filter: {base_has_filter}")
        print(f"Expert has DIRECTED: {expert_has_directed}, has filter: {expert_has_filter}")
        
        # Expert should understand the query better
        assert expert_has_directed or expert_has_filter, "Expert should include relevant query elements"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

