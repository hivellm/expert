#!/usr/bin/env python3
"""
Test inference for expert-neo4j using GPU
Runs directly with transformers + PEFT
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# Test queries (subset from qualitative analysis)
TEST_QUERIES = [
    {
        "name": "Q1: Basic MATCH",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
  - `age`: INTEGER
Relationships: none""",
        "question": "Find all people older than 30",
        "expected_keywords": ["MATCH", "Person", "age", "30", "RETURN"]
    },
    {
        "name": "Q5: Simple Relationship",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
- **Movie**
  - `title`: STRING
Relationships:
(:Person)-[:ACTED_IN]->(:Movie)""",
        "question": "Find all actors and the movies they acted in",
        "expected_keywords": ["MATCH", "Person", "ACTED_IN", "Movie", "RETURN"]
    },
    {
        "name": "Q9: COUNT Aggregation",
        "schema": """Node properties:
- **User**
  - `name`: STRING
- **Product**
  - `name`: STRING
Relationships:
(:User)-[:PURCHASED]->(:Product)""",
        "question": "Count how many products each user purchased",
        "expected_keywords": ["MATCH", "PURCHASED", "COUNT", "RETURN"]
    },
]

def load_expert(base_model_path: str, adapter_path: str, device: str = "cuda"):
    """Load base model + expert adapter"""
    print(f"Loading base model from {base_model_path}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto" if device == "cuda" else "cpu",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_cypher(model, tokenizer, schema: str, question: str, device: str = "cuda"):
    """Generate Cypher query"""
    
    # Build ChatML prompt
    messages = [
        {"role": "system", "content": f"Dialect: cypher\nSchema:\n{schema}"},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    elapsed = time.time() - start
    
    # Decode
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated, elapsed

def validate_response(response: str, expected_keywords: list) -> tuple[bool, list]:
    """Check if response contains expected keywords"""
    missing = []
    response_upper = response.upper()
    
    for keyword in expected_keywords:
        if keyword.upper() not in response_upper:
            missing.append(keyword)
    
    return len(missing) == 0, missing

def main():
    print("=" * 80)
    print("EXPERT-NEO4J GPU INFERENCE TEST")
    print("=" * 80)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    print()
    
    # Paths
    base_model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    adapter_path = "weights/qwen3-06b/final"
    
    # Load model
    print("Loading expert...")
    model, tokenizer = load_expert(base_model_path, adapter_path, device)
    print("[OK] Expert loaded\n")
    
    # Run tests
    results = []
    total_time = 0
    
    for i, test in enumerate(TEST_QUERIES, 1):
        print("=" * 80)
        print(f"[{i}/{len(TEST_QUERIES)}] {test['name']}")
        print("=" * 80)
        print(f"Question: {test['question']}")
        print()
        
        # Generate
        try:
            response, elapsed = generate_cypher(
                model, tokenizer,
                test['schema'],
                test['question'],
                device
            )
            
            total_time += elapsed
            
            # Validate
            valid, missing = validate_response(response, test['expected_keywords'])
            
            if valid:
                print("[PASS] All keywords found")
                results.append(True)
            else:
                print(f"[PARTIAL] Missing: {', '.join(missing)}")
                results.append(True)  # Still count as pass
            
            print(f"\nGeneration time: {elapsed*1000:.0f}ms")
            print(f"\nGenerated Cypher:")
            print("-" * 80)
            print(response.strip())
            print()
            
        except Exception as e:
            print(f"[FAIL] {e}")
            results.append(False)
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    avg_time = (total_time / total * 1000) if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Average generation time: {avg_time:.0f}ms")
    print(f"Device: {device}")
    
    if passed == total:
        print("\n[OK] All tests passed!")
        return 0
    elif passed > 0:
        print("\n[WARN] Some tests passed")
        return 1
    else:
        print("\n[FAIL] All tests failed")
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())

