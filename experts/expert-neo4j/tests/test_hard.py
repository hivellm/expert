"""Comprehensive test suite for Neo4j Expert - Hard cases"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

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


def generate_cypher(model, tokenizer, schema, question, max_tokens=300):
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
    # Extract response part
    if "<|im_start|>assistant" in response:
        cypher = response.split("<|im_start|>assistant")[-1].strip()
        cypher = cypher.replace("<|im_end|>", "").strip()
    else:
        cypher = response.strip()
    
    return cypher


class TestHardCases:
    """Hard test cases for Neo4j Expert"""
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load model once for all tests"""
        return load_expert_model()
    
    def test_multi_hop_traversal(self, expert_model):
        """Test multi-hop relationship traversal"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Person**
  - `name`: STRING
- **Movie**
  - `title`: STRING
- **Genre**
  - `name`: STRING
Relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Movie)-[:HAS_GENRE]->(:Genre)"""
        
        question = "Find all actors who acted in Action movies"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "RETURN" in cypher
    
    def test_aggregation_with_grouping(self, expert_model):
        """Test aggregation with GROUP BY"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **User**
  - `name`: STRING
- **Post**
  - `title`: STRING
  - `likes`: INTEGER
Relationships:
(:User)-[:POSTED]->(:Post)"""
        
        question = "Count posts per user"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "count" in cypher.lower() or "COUNT" in cypher
        assert "RETURN" in cypher
    
    def test_pattern_with_properties(self, expert_model):
        """Test pattern matching with property constraints"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Product**
  - `name`: STRING
  - `price`: FLOAT
  - `category`: STRING
Relationships:
None"""
        
        question = "Find products with price greater than 100 in Electronics category"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "WHERE" in cypher or "price" in cypher
        assert "RETURN" in cypher
    
    def test_bidirectional_relationship(self, expert_model):
        """Test bidirectional relationship query"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Person**
  - `name`: STRING
Relationships:
(:Person)-[:FRIENDS_WITH]->(:Person)"""
        
        question = "Find all friends of John"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "FRIENDS_WITH" in cypher or "friends" in cypher.lower()
        assert "RETURN" in cypher
    
    def test_optional_match(self, expert_model):
        """Test OPTIONAL MATCH pattern"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Person**
  - `name`: STRING
- **Email**
  - `address`: STRING
Relationships:
(:Person)-[:HAS_EMAIL]->(:Email)"""
        
        question = "Find all people and their emails if they have one"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "RETURN" in cypher
    
    def test_path_finding(self, expert_model):
        """Test path finding query"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Airport**
  - `code`: STRING
  - `name`: STRING
Relationships:
(:Airport)-[:FLIGHT_TO]->(:Airport)"""
        
        question = "Find shortest path between LAX and JFK airports"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "Airport" in cypher
        assert "RETURN" in cypher
    
    def test_union_query(self, expert_model):
        """Test UNION query"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Movie**
  - `title`: STRING
- **TVShow**
  - `title`: STRING
Relationships:
None"""
        
        question = "Find all movie and TV show titles"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "RETURN" in cypher
    
    def test_complex_filtering(self, expert_model):
        """Test complex WHERE clause with multiple conditions"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Employee**
  - `name`: STRING
  - `salary`: INTEGER
  - `department`: STRING
  - `years_experience`: INTEGER
Relationships:
(:Employee)-[:REPORTS_TO]->(:Employee)"""
        
        question = "Find employees in Engineering with salary above 80000 and more than 5 years experience"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "WHERE" in cypher or "salary" in cypher
        assert "RETURN" in cypher
    
    def test_date_filtering(self, expert_model):
        """Test date/time filtering"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **Event**
  - `name`: STRING
  - `date`: DATE
Relationships:
None"""
        
        question = "Find events after 2020"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "RETURN" in cypher
    
    def test_string_operations(self, expert_model):
        """Test string operations in query"""
        model, tokenizer = expert_model
        
        schema = """Node properties:
- **User**
  - `username`: STRING
  - `email`: STRING
Relationships:
None"""
        
        question = "Find users whose email contains gmail"
        cypher = generate_cypher(model, tokenizer, schema, question)
        
        print(f"\nGenerated Cypher:\n{cypher}\n")
        
        assert "MATCH" in cypher
        assert "RETURN" in cypher


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

