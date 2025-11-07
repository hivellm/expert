#!/usr/bin/env python3
"""
Analyze additional Neo4j Cypher datasets from HuggingFace
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
import sys

def analyze_dataset(dataset_name: str, split: str = "train") -> Dict[str, Any]:
    """Analyze a HuggingFace dataset"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load dataset
        print(f"Loading dataset...")
        dataset = load_dataset(dataset_name, split=split)
        
        # Basic stats
        total_examples = len(dataset)
        print(f"Total examples: {total_examples}")
        
        # Analyze structure
        print(f"\nDataset structure:")
        print(f"Features: {dataset.features}")
        
        # Sample analysis
        sample = dataset[0] if total_examples > 0 else None
        if sample:
            print(f"\nSample entry:")
            for key, value in sample.items():
                try:
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
                except UnicodeEncodeError:
                    # Handle encoding issues
                    if isinstance(value, str):
                        safe_value = value.encode('ascii', 'ignore').decode('ascii')
                        print(f"  {key}: {safe_value[:100]}... (contains non-ASCII)")
                    else:
                        print(f"  {key}: <non-printable>")
        
        # Field detection
        question_field = None
        cypher_field = None
        schema_field = None
        
        for field in dataset.features.keys():
            field_lower = field.lower()
            # Question field detection (prioritize 'question', then 'nl_question', then 'input_text')
            if 'question' in field_lower:
                if question_field is None or 'nl_question' in field_lower:
                    question_field = field
            elif 'query' in field_lower and 'cypher' not in field_lower:
                if question_field is None:
                    question_field = field
            elif 'input_text' in field_lower or 'text' in field_lower:
                if question_field is None:
                    question_field = field
            
            # Cypher field detection (prioritize 'cypher', then 'gold_cypher', then 'output_text')
            if 'cypher' in field_lower:
                if cypher_field is None or 'gold_cypher' in field_lower:
                    cypher_field = field
            elif 'output_text' in field_lower:
                if cypher_field is None:
                    cypher_field = field
            
            # Schema field detection
            if 'schema' in field_lower or 'graph' in field_lower:
                schema_field = field
        
        print(f"\nDetected fields:")
        print(f"  Question: {question_field}")
        print(f"  Cypher: {cypher_field}")
        print(f"  Schema: {schema_field}")
        
        # Language detection (simple heuristic)
        if total_examples > 0 and question_field:
            try:
                sample_question = sample.get(question_field, "")
                if isinstance(sample_question, str):
                    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in sample_question)
                    language = "Chinese" if has_chinese else "English"
                else:
                    language = "Unknown"
            except:
                language = "Unknown"
            print(f"\nLanguage: {language}")
        else:
            language = "Unknown"
        
        # Cypher pattern analysis
        cypher_patterns = {
            'MATCH': 0,
            'CREATE': 0,
            'MERGE': 0,
            'RETURN': 0,
            'WHERE': 0,
            'WITH': 0,
            'COUNT': 0,
            'AVG': 0,
            'ORDER BY': 0,
            'LIMIT': 0,
            'shortestPath': 0,
            'relationships': 0
        }
        
        if cypher_field and total_examples > 0:
            print(f"\nAnalyzing Cypher patterns (first 100 examples)...")
            for i in range(min(100, total_examples)):
                cypher = dataset[i].get(cypher_field, "")
                if isinstance(cypher, str):
                    cypher_upper = cypher.upper()
                    for pattern in cypher_patterns.keys():
                        if pattern.upper() in cypher_upper:
                            cypher_patterns[pattern] += 1
            
            print(f"  Pattern distribution:")
            for pattern, count in sorted(cypher_patterns.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"    {pattern}: {count}/{min(100, total_examples)}")
        
        # Schema presence
        has_schema = schema_field is not None
        schema_count = 0
        if schema_field and total_examples > 0:
            for i in range(min(100, total_examples)):
                schema = dataset[i].get(schema_field)
                if schema and (isinstance(schema, str) and len(schema.strip()) > 0):
                    schema_count += 1
        
        print(f"\nSchema presence: {has_schema}")
        if has_schema:
            print(f"  Examples with schema: {schema_count}/{min(100, total_examples)}")
        
        return {
            'dataset_name': dataset_name,
            'total_examples': total_examples,
            'language': language,
            'question_field': question_field,
            'cypher_field': cypher_field,
            'schema_field': schema_field,
            'has_schema': has_schema,
            'schema_coverage': schema_count / min(100, total_examples) if total_examples > 0 else 0,
            'cypher_patterns': cypher_patterns,
            'features': list(dataset.features.keys()),
            'sample': sample
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        # Try to get partial info even on error
        try:
            # For Doraemon-AI dataset, it has known generation errors
            if 'Doraemon-AI' in dataset_name:
                return {
                    'dataset_name': dataset_name,
                    'error': error_msg,
                    'known_issue': 'Dataset generation error (schema casting issue)',
                    'partial_info': 'Dataset exists but has schema casting problems preventing full load'
                }
            # For johanhan dataset, encoding issues
            elif 'johanhan' in dataset_name:
                return {
                    'dataset_name': dataset_name,
                    'error': error_msg,
                    'known_issue': 'Encoding/Unicode issues',
                    'partial_info': 'Dataset contains Chinese characters causing encoding errors'
                }
        except:
            pass
        
        return {
            'dataset_name': dataset_name,
            'error': error_msg
        }

def main():
    datasets_to_analyze = [
        ("jiuyuan/train_cypher", "train"),
        ("megagonlabs/cypherbench", "train"),
        ("Doraemon-AI/text-to-neo4j-cypher-chinese", "train"),
        ("johanhan/neo4j-cypher-fixed", "train"),
    ]
    
    results = []
    
    for dataset_name, split in datasets_to_analyze:
        try:
            result = analyze_dataset(dataset_name, split)
            results.append(result)
        except Exception as e:
            print(f"Failed to analyze {dataset_name}: {e}")
            results.append({
                'dataset_name': dataset_name,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        if 'error' in result:
            print(f"\n{result['dataset_name']}: ERROR - {result['error']}")
        else:
            print(f"\n{result['dataset_name']}:")
            print(f"  Examples: {result['total_examples']}")
            print(f"  Language: {result['language']}")
            print(f"  Schema: {'Yes' if result['has_schema'] else 'No'}")
            print(f"  Question field: {result['question_field']}")
            print(f"  Cypher field: {result['cypher_field']}")
    
    # Save results
    output_file = Path(__file__).parent.parent / "docs" / "DATASET_ANALYSIS.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Additional Neo4j Cypher Datasets Analysis\n\n")
        f.write("Analysis of potential datasets for integration into the Neo4j expert training data.\n\n")
        
        for result in results:
            f.write(f"## {result['dataset_name']}\n\n")
            
            if 'error' in result:
                f.write(f"**Status:** ERROR\n\n")
                f.write(f"**Error:** {result['error']}\n\n")
            else:
                f.write(f"**Status:** Available\n\n")
                f.write(f"**Total Examples:** {result['total_examples']:,}\n\n")
                f.write(f"**Language:** {result['language']}\n\n")
                f.write(f"**Schema Support:** {'Yes' if result['has_schema'] else 'No'}\n\n")
                f.write(f"**Schema Coverage:** {result['schema_coverage']*100:.1f}%\n\n")
                f.write(f"**Fields:**\n")
                f.write(f"- Question: `{result['question_field']}`\n")
                f.write(f"- Cypher: `{result['cypher_field']}`\n")
                f.write(f"- Schema: `{result['schema_field']}`\n\n")
                
                if result['cypher_patterns']:
                    f.write(f"**Cypher Patterns (sample of 100):**\n")
                    for pattern, count in sorted(result['cypher_patterns'].items(), key=lambda x: x[1], reverse=True):
                        if count > 0:
                            f.write(f"- {pattern}: {count}\n")
                    f.write("\n")
            
            f.write("---\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        current_dataset_size = 29512  # From manifest.json
        
        for result in results:
            if 'error' not in result:
                f.write(f"### {result['dataset_name']}\n\n")
                
                recommendation = []
                
                if result['language'] == 'Chinese':
                    recommendation.append("⚠️ **Language mismatch**: Dataset is in Chinese, but current expert uses English-only dataset.")
                
                if result['total_examples'] < 1000:
                    recommendation.append("⚠️ **Small size**: Less than 1,000 examples may not significantly improve the model.")
                elif result['total_examples'] > 5000:
                    recommendation.append("✅ **Large size**: Good potential for significant dataset expansion.")
                
                if not result['has_schema']:
                    recommendation.append("⚠️ **No schema**: Missing graph schema context (critical for Neo4j queries).")
                elif result['schema_coverage'] < 0.8:
                    recommendation.append("⚠️ **Low schema coverage**: Many examples may lack schema context.")
                
                if result['total_examples'] > current_dataset_size * 0.1:
                    recommendation.append("✅ **Significant addition**: Would add substantial new examples.")
                
                if not recommendation:
                    recommendation.append("✅ **Potential candidate**: Worth evaluating for integration.")
                
                for rec in recommendation:
                    f.write(f"- {rec}\n")
                
                f.write("\n")
    
    print(f"\nAnalysis saved to: {output_file}")

if __name__ == "__main__":
    main()

