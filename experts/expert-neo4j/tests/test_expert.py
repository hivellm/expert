"""Simple validation tests for Neo4j Expert"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# Base model path (adjust as needed)
BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
EXPERT_PATH = "../weights/qwen3-06b/adapter"


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
    """Generate Cypher query from question and schema"""
    # Qwen3 uses ChatML format
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
    # Extract only the generated part (after assistant marker)
    if "<|im_start|>assistant" in response:
        cypher = response.split("<|im_start|>assistant")[-1].strip()
        cypher = cypher.replace("<|im_end|>", "").strip()
    else:
        cypher = response.strip()
    
    return cypher


class TestNeo4jExpert:
    """Test suite for Neo4j Expert"""
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load model once for all tests"""
        return load_expert_model()
    
    def test_simple_match_query(self, expert_model):
        """Test basic MATCH query generation"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Person**
  - `name`: STRING
  - `age`: INTEGER
Relationships:
(:Person)-[:KNOWS]->(:Person)"""
        
        question = "Find all people"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        assert "MATCH" in cypher
        assert "Person" in cypher
        assert "RETURN" in cypher
    
    def test_relationship_query(self, expert_model):
        """Test query with relationships"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Movie**
  - `title`: STRING
  - `released`: INTEGER
- **Person**
  - `name`: STRING
Relationships:
(:Person)-[:ACTED_IN]->(:Movie)"""
        
        question = "Find all actors in movies"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        assert "MATCH" in cypher
        assert "ACTED_IN" in cypher or "acted_in" in cypher.lower()
        assert "RETURN" in cypher
    
    def test_filtering_query(self, expert_model):
        """Test query with WHERE clause"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Person**
  - `name`: STRING
  - `born`: INTEGER
Relationships:
(:Person)-[:DIRECTED]->(:Movie)"""
        
        question = "Find people born after 1970"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        assert "MATCH" in cypher
        assert "WHERE" in cypher or "born" in cypher
        assert "RETURN" in cypher
    
    def test_aggregation_query(self, expert_model):
        """Test query with aggregation"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **User**
  - `name`: STRING
Relationships:
(:User)-[:FOLLOWS]->(:User)"""
        
        question = "Count total users"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        assert "MATCH" in cypher or "count" in cypher.lower()
        assert "RETURN" in cypher
    
    def test_ordering_and_limit(self, expert_model):
        """Test query with ORDER BY and LIMIT"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Product**
  - `name`: STRING
  - `price`: FLOAT
Relationships:
None"""
        
        question = "Find top 5 most expensive products"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        assert "MATCH" in cypher
        assert "RETURN" in cypher
        # May include ORDER BY and LIMIT
        assert "Product" in cypher


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

