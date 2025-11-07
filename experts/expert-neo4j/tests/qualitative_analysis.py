"""
Análise Qualitativa: Modelo Base vs Expert Neo4j (múltiplos checkpoints)

Testa queries Cypher representativas e compara a qualidade dos outputs.
Compara Base Model, checkpoint-250 e checkpoint-500.
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datetime import datetime

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
CHECKPOINTS = [
    "../weights/qwen3-06b/checkpoint-250",
    "../weights/qwen3-06b/checkpoint-500",
    "../weights/qwen3-06b/checkpoint-592",
    "../weights/qwen3-06b/final"
]

# Check CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Test queries (from basic to advanced - 20 queries for comprehensive analysis)
TEST_QUERIES = [
    # BASIC QUERIES (1-4)
    {
        "name": "Q1: Basic MATCH",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
  - `age`: INTEGER
Relationships:
None""",
        "question": "Find all people older than 30",
        "expected_elements": ["MATCH", "Person", "WHERE", "age", "30", "RETURN"]
    },
    {
        "name": "Q2: Simple Property Filter",
        "schema": """Node properties:
- **Product**
  - `name`: STRING
  - `price`: FLOAT
  - `inStock`: BOOLEAN
Relationships:
None""",
        "question": "Find all products that are in stock",
        "expected_elements": ["MATCH", "Product", "WHERE", "inStock", "true", "RETURN"]
    },
    {
        "name": "Q3: Multiple Properties",
        "schema": """Node properties:
- **Employee**
  - `name`: STRING
  - `salary`: INTEGER
  - `department`: STRING
Relationships:
None""",
        "question": "Find all employees in IT department with salary over 80000",
        "expected_elements": ["MATCH", "Employee", "WHERE", "department", "IT", "salary", "80000", "RETURN"]
    },
    {
        "name": "Q4: String Pattern Matching",
        "schema": """Node properties:
- **Company**
  - `name`: STRING
  - `industry`: STRING
Relationships:
None""",
        "question": "Find all companies whose name contains 'Tech'",
        "expected_elements": ["MATCH", "Company", "WHERE", "name", "CONTAINS", "Tech", "RETURN"]
    },
    
    # RELATIONSHIP QUERIES (5-8)
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
        "expected_elements": ["MATCH", "Person", "ACTED_IN", "Movie", "RETURN"]
    },
    {
        "name": "Q6: Bidirectional Relationship",
        "schema": """Node properties:
- **User**
  - `username`: STRING
Relationships:
(:User)-[:FOLLOWS]->(:User)""",
        "question": "Find all users who follow each other",
        "expected_elements": ["MATCH", "User", "FOLLOWS", "RETURN"]
    },
    {
        "name": "Q7: Filter on Relationship",
        "schema": """Node properties:
- **Movie**
  - `title`: STRING
  - `released`: INTEGER
- **Person**
  - `name`: STRING
  - `born`: INTEGER
Relationships:
(:Person)-[:DIRECTED]->(:Movie)""",
        "question": "Find all movies directed by people born in the 1960s",
        "expected_elements": ["MATCH", "DIRECTED", "WHERE", "born", "1960", "RETURN"]
    },
    {
        "name": "Q8: Relationship with Properties",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
- **Company**
  - `name`: STRING
Relationships:
(:Person)-[:WORKS_AT {since: INTEGER, role: STRING}]->(:Company)""",
        "question": "Find all people who started working at companies after 2020",
        "expected_elements": ["MATCH", "WORKS_AT", "WHERE", "since", "2020", "RETURN"]
    },
    
    # AGGREGATION QUERIES (9-12)
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
        "expected_elements": ["MATCH", "PURCHASED", "RETURN", "COUNT"]
    },
    {
        "name": "Q10: SUM Aggregation",
        "schema": """Node properties:
- **Customer**
  - `name`: STRING
- **Order**
  - `total`: FLOAT
Relationships:
(:Customer)-[:PLACED]->(:Order)""",
        "question": "Calculate total order value for each customer",
        "expected_elements": ["MATCH", "PLACED", "RETURN", "SUM", "total"]
    },
    {
        "name": "Q11: AVG and GROUP BY",
        "schema": """Node properties:
- **Movie**
  - `title`: STRING
  - `rating`: FLOAT
- **Genre**
  - `name`: STRING
Relationships:
(:Movie)-[:IN_GENRE]->(:Genre)""",
        "question": "Find average rating for each genre",
        "expected_elements": ["MATCH", "IN_GENRE", "RETURN", "AVG", "rating"]
    },
    {
        "name": "Q12: COLLECT Aggregation",
        "schema": """Node properties:
- **Author**
  - `name`: STRING
- **Book**
  - `title`: STRING
Relationships:
(:Author)-[:WROTE]->(:Book)""",
        "question": "List all books written by each author",
        "expected_elements": ["MATCH", "WROTE", "RETURN", "COLLECT"]
    },
    
    # MULTI-HOP QUERIES (13-16)
    {
        "name": "Q13: Multi-hop Pattern",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
- **Movie**
  - `title`: STRING
Relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)""",
        "question": "Find actors who worked in movies directed by Steven Spielberg",
        "expected_elements": ["MATCH", "ACTED_IN", "DIRECTED", "WHERE", "name", "RETURN"]
    },
    {
        "name": "Q14: Chain of Relationships",
        "schema": """Node properties:
- **City**
  - `name`: STRING
- **State**
  - `name`: STRING
- **Country**
  - `name`: STRING
Relationships:
(:City)-[:IN_STATE]->(:State)
(:State)-[:IN_COUNTRY]->(:Country)""",
        "question": "Find all cities in Brazil",
        "expected_elements": ["MATCH", "City", "IN_STATE", "IN_COUNTRY", "WHERE", "name", "Brazil", "RETURN"]
    },
    {
        "name": "Q15: Triangle Pattern",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
- **Company**
  - `name`: STRING
- **Product**
  - `name`: STRING
Relationships:
(:Person)-[:WORKS_AT]->(:Company)
(:Company)-[:PRODUCES]->(:Product)
(:Person)-[:USES]->(:Product)""",
        "question": "Find people who use products made by their own company",
        "expected_elements": ["MATCH", "WORKS_AT", "PRODUCES", "USES", "WHERE", "RETURN"]
    },
    {
        "name": "Q16: Complex Multi-hop",
        "schema": """Node properties:
- **Customer**
  - `name`: STRING
- **Order**
  - `id`: STRING
- **Product**
  - `name`: STRING
  - `category`: STRING
Relationships:
(:Customer)-[:PLACED]->(:Order)
(:Order)-[:CONTAINS]->(:Product)""",
        "question": "Find customers who ordered products in the Electronics category",
        "expected_elements": ["MATCH", "Customer", "PLACED", "CONTAINS", "Product", "WHERE", "category", "Electronics", "RETURN"]
    },
    
    # ADVANCED QUERIES (17-20)
    {
        "name": "Q17: ORDER BY and LIMIT",
        "schema": """Node properties:
- **Movie**
  - `title`: STRING
  - `rating`: FLOAT
  - `released`: INTEGER
Relationships:
None""",
        "question": "Find the top 5 highest rated movies",
        "expected_elements": ["MATCH", "Movie", "RETURN", "ORDER BY", "rating", "LIMIT", "5"]
    },
    {
        "name": "Q18: Variable-length Path",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
Relationships:
(:Person)-[:KNOWS]->(:Person)""",
        "question": "Find all people connected to John within 3 hops",
        "expected_elements": ["MATCH", "KNOWS", "WHERE", "name", "RETURN"]
    },
    {
        "name": "Q19: OPTIONAL MATCH",
        "schema": """Node properties:
- **Person**
  - `name`: STRING
  - `email`: STRING
- **Address**
  - `street`: STRING
  - `city`: STRING
Relationships:
(:Person)-[:LIVES_AT]->(:Address)""",
        "question": "Find all people and their addresses if they have one",
        "expected_elements": ["MATCH", "Person", "OPTIONAL", "LIVES_AT", "Address", "RETURN"]
    },
    {
        "name": "Q20: Complex Filter Logic",
        "schema": """Node properties:
- **Transaction**
  - `amount`: FLOAT
  - `date`: STRING
  - `type`: STRING
- **Account**
  - `number`: STRING
  - `balance`: FLOAT
Relationships:
(:Account)-[:HAS_TRANSACTION]->(:Transaction)""",
        "question": "Find accounts with deposits over 1000 or withdrawals over 500 in the last month",
        "expected_elements": ["MATCH", "HAS_TRANSACTION", "WHERE", "amount", "type", "OR", "RETURN"]
    },
]


def load_model(checkpoint_path=None):
    """Load model (base or with checkpoint adapter)"""
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    
    # Adjust dtype based on device
    if DEVICE == "cuda":
        dtype = torch.bfloat16
        device_map = "cuda"
    else:
        dtype = torch.float32
        device_map = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model_name = f"Expert ({os.path.basename(checkpoint_path)})"
    else:
        model_name = "Base Model"
    
    return model, tokenizer, model_name


def generate_cypher(model, tokenizer, schema, question, max_tokens=300):
    """Generate Cypher query using ChatML format"""
    # Use Qwen3 ChatML format
    messages = [
        {"role": "system", "content": f"Dialect: cypher\nSchema:\n{schema}"},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only generated part
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean up (remove potential chat markers)
    generated = generated.strip()
    if "<|im_end|>" in generated:
        generated = generated.split("<|im_end|>")[0].strip()
    
    return generated


def analyze_output(output, expected_elements):
    """Analyze quality of generated output"""
    output_upper = output.upper()
    
    # Check for expected elements
    found_elements = []
    missing_elements = []
    
    for element in expected_elements:
        if element.upper() in output_upper:
            found_elements.append(element)
        else:
            missing_elements.append(element)
    
    # Basic syntax checks
    has_cypher_commands = any(cmd in output_upper for cmd in ["MATCH", "CREATE", "MERGE", "RETURN", "WITH"])
    has_graph_elements = "(" in output or "[" in output
    
    # Balanced parentheses/brackets
    balanced = (
        output.count("(") == output.count(")") and
        output.count("[") == output.count("]") and
        output.count("{") == output.count("}")
    )
    
    # Check for common mistakes
    has_sql_keywords = any(kw in output_upper for kw in ["SELECT", "FROM", "JOIN", "TABLE"])
    has_excess_noise = len(output) > 500  # Too long might indicate hallucination
    
    score = 0
    max_score = 10
    
    # Scoring
    if has_cypher_commands:
        score += 2
    if has_graph_elements:
        score += 1
    if balanced:
        score += 2
    if not has_sql_keywords:
        score += 2
    if not has_excess_noise:
        score += 1
    
    # Expected elements coverage
    coverage = len(found_elements) / len(expected_elements) if expected_elements else 0
    score += coverage * 2  # Up to 2 points for element coverage
    
    return {
        "score": score,
        "max_score": max_score,
        "found_elements": found_elements,
        "missing_elements": missing_elements,
        "has_cypher_commands": has_cypher_commands,
        "has_graph_elements": has_graph_elements,
        "balanced": balanced,
        "has_sql_keywords": has_sql_keywords,
        "has_excess_noise": has_excess_noise,
        "coverage": coverage
    }


def run_analysis():
    """Run complete qualitative analysis for all checkpoints"""
    print("=" * 80)
    print("QUALITATIVE ANALYSIS: Base vs All Checkpoints (250, 500, 592, final)")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Total Queries: {len(TEST_QUERIES)}")
    print(f"Expected Runtime: ~{len(TEST_QUERIES) * (len(CHECKPOINTS) + 1) * 3 / 60:.0f} minutes")
    print()
    
    # Load base model
    print("-" * 80)
    print("[1/3] Loading Base Model...")
    base_model, base_tokenizer, base_name = load_model()
    print(f"[OK] {base_name} loaded")
    print()
    
    # Load checkpoint models
    checkpoint_models = []
    for i, checkpoint_path in enumerate(CHECKPOINTS, 2):
        if not os.path.exists(checkpoint_path):
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"[{i}/{len(CHECKPOINTS)+1}] Loading {os.path.basename(checkpoint_path)}...")
        model, tokenizer, name = load_model(checkpoint_path)
        checkpoint_models.append((model, tokenizer, name, checkpoint_path))
        print(f"[OK] {name} loaded")
        print()
    
    if not checkpoint_models:
        print("[ERROR] No checkpoints found!")
        return
    
    # Run tests
    all_results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print("=" * 80)
        print(f"{query['name']} ({i}/{len(TEST_QUERIES)})")
        print("=" * 80)
        print(f"Question: {query['question']}")
        print()
        
        query_results = {
            "query": query['name'],
            "question": query['question'],
            "models": {}
        }
        
        # Generate with base model
        print("[BASE MODEL]")
        print("-" * 40)
        base_output = generate_cypher(base_model, base_tokenizer, query['schema'], query['question'])
        # Safe print with encoding handling
        safe_output = base_output.encode('ascii', errors='replace').decode('ascii')
        print(safe_output[:200] + ("..." if len(safe_output) > 200 else ""))
        base_analysis = analyze_output(base_output, query['expected_elements'])
        print(f"Score: {base_analysis['score']:.1f}/10 | Coverage: {base_analysis['coverage']*100:.0f}%")
        print()
        
        query_results["models"]["base"] = {
            "output": base_output,
            "score": base_analysis['score'],
            "analysis": base_analysis
        }
        
        # Generate with all checkpoint models
        for model, tokenizer, name, checkpoint_path in checkpoint_models:
            checkpoint_name = os.path.basename(checkpoint_path)
            print(f"[{checkpoint_name.upper()}]")
            print("-" * 40)
            output = generate_cypher(model, tokenizer, query['schema'], query['question'])
            # Safe print with encoding handling
            safe_output = output.encode('ascii', errors='replace').decode('ascii')
            print(safe_output[:200] + ("..." if len(safe_output) > 200 else ""))
            analysis = analyze_output(output, query['expected_elements'])
            improvement = analysis['score'] - base_analysis['score']
            
            status = "[+]" if improvement > 0 else "[-]" if improvement < 0 else "[=]"
            print(f"Score: {analysis['score']:.1f}/10 | Coverage: {analysis['coverage']*100:.0f}% | {status} {improvement:+.1f} vs base")
            print()
            
            query_results["models"][checkpoint_name] = {
                "output": output,
                "score": analysis['score'],
                "analysis": analysis,
                "improvement": improvement
            }
        
        all_results.append(query_results)
        print()
    
    # Summary
    print("=" * 80)
    print("[OVERALL SUMMARY - EVOLUTION COMPARISON]")
    print("=" * 80)
    print()
    
    # Calculate averages for each model
    base_avg = sum(r["models"]["base"]["score"] for r in all_results) / len(all_results)
    
    print(f"Average Scores Across {len(all_results)} Queries:")
    print(f"  Base Model:      {base_avg:.2f}/10")
    
    checkpoint_avgs = {}
    for model, tokenizer, name, checkpoint_path in checkpoint_models:
        checkpoint_name = os.path.basename(checkpoint_path)
        avg = sum(r["models"][checkpoint_name]["score"] for r in all_results) / len(all_results)
        checkpoint_avgs[checkpoint_name] = avg
        improvement = avg - base_avg
        status = "[+]" if improvement > 0 else "[-]" if improvement < 0 else "[=]"
        print(f"  {checkpoint_name:15s}: {avg:.2f}/10  {status} {improvement:+.2f} vs base ({(improvement/10)*100:+.1f}%)")
    
    print()
    
    # Head-to-head comparison for each checkpoint
    for model, tokenizer, name, checkpoint_path in checkpoint_models:
        checkpoint_name = os.path.basename(checkpoint_path)
        wins = sum(1 for r in all_results if r["models"][checkpoint_name]["improvement"] > 0)
        losses = sum(1 for r in all_results if r["models"][checkpoint_name]["improvement"] < 0)
        ties = sum(1 for r in all_results if r["models"][checkpoint_name]["improvement"] == 0)
        
        print(f"{checkpoint_name} vs Base:")
        print(f"  Wins:   {wins}/{len(all_results)} ({wins/len(all_results)*100:.0f}%)")
        print(f"  Losses: {losses}/{len(all_results)} ({losses/len(all_results)*100:.0f}%)")
        print(f"  Ties:   {ties}/{len(all_results)} ({ties/len(all_results)*100:.0f}%)")
        print()
    
    # Evolution progression across all checkpoints
    print("Evolution Progression:")
    print(f"  Base Model: {base_avg:.2f}")
    sorted_checkpoints = sorted(checkpoint_avgs.items(), key=lambda x: int(x[0].split('-')[-1]) if x[0].startswith('checkpoint-') else 9999)
    prev_score = base_avg
    for checkpoint_name, score in sorted_checkpoints:
        improvement = score - prev_score
        status = "[+]" if improvement > 0 else "[-]" if improvement < 0 else "[=]"
        print(f"  {checkpoint_name:15s}: {score:.2f}  {status} {improvement:+.2f} from previous")
        prev_score = score
    print()
    
    # Key evolution metrics
    if "checkpoint-250" in checkpoint_avgs and "final" in checkpoint_avgs:
        total_evolution = checkpoint_avgs["final"] - checkpoint_avgs["checkpoint-250"]
        print(f"Total Evolution (250 -> final): {total_evolution:+.2f} points ({(total_evolution/10)*100:+.1f}%)")
        print()
    
    print("Query-by-Query Comparison:")
    print("-" * 80)
    for r in all_results:
        line = f"{r['query']:30s}"
        line += f" Base={r['models']['base']['score']:.1f}"
        for checkpoint_name in sorted(checkpoint_avgs.keys()):
            score = r['models'][checkpoint_name]['score']
            improvement = r['models'][checkpoint_name]['improvement']
            status = "[+]" if improvement > 0 else "[-]" if improvement < 0 else "[=]"
            line += f"  {checkpoint_name}={score:.1f}{status}"
        print(line)
    print()
    
    # Verdict
    print("=" * 80)
    print("[FINAL VERDICT]")
    print("=" * 80)
    print()
    
    # Analyze best checkpoint
    if checkpoint_avgs:
        best_checkpoint = max(checkpoint_avgs.items(), key=lambda x: x[1])
        best_name, best_score = best_checkpoint
        best_improvement = best_score - base_avg
        
        print(f"Best Checkpoint: {best_name}")
        print(f"  Score: {best_score:.2f}/10 (base: {base_avg:.2f}/10)")
        print(f"  Improvement: {best_improvement:+.2f} points ({(best_improvement/10)*100:+.1f}%)")
        print()
        
        # Evolution assessment
        if "checkpoint-250" in checkpoint_avgs and "checkpoint-500" in checkpoint_avgs:
            evolution = checkpoint_avgs["checkpoint-500"] - checkpoint_avgs["checkpoint-250"]
            
            if evolution > 0.5:
                print("[OK] CLEAR POSITIVE EVOLUTION")
                print(f"   Model improving with more training (+{evolution:.2f} points from 250->500)")
                print(f"   Recommendation: Continue to final checkpoint (~655)")
            elif evolution > -0.3:
                print("[OK] STABLE EVOLUTION")
                print(f"   Model maintained performance ({evolution:+.2f} points from 250->500)")
                print(f"   Recommendation: Check final checkpoint to confirm")
            else:
                print("[WARN] POSSIBLE OVERFITTING")
                print(f"   Model got worse from 250->500 ({evolution:.2f} points)")
                print(f"   Recommendation: checkpoint-250 may be better than final")
            print()
        
        # Overall assessment
        if best_improvement > 1.5:
            print("[OK] TRAINING VERY SUCCESSFUL")
            print(f"   Significant improvement vs base (+{best_improvement:.1f} points)")
        elif best_improvement > 0.5:
            print("[OK] TRAINING SUCCESSFUL")
            print(f"   Moderate improvement vs base (+{best_improvement:.1f} points)")
        elif best_improvement > -0.5:
            print("[WARN] TRAINING WITH LIMITED EFFECT")
            print(f"   Small difference vs base ({best_improvement:+.1f} points)")
            print(f"   Recommendation: Wait for final checkpoint or adjust hyperparameters")
        else:
            print("[ERROR] POSSIBLE TRAINING PROBLEM")
            print(f"   Expert worse than base ({best_improvement:.1f} points)")
            print(f"   Recommendation: Check dataset and hyperparameters")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    run_analysis()

