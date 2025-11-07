#!/usr/bin/env python3
"""
Neo4j Official Documentation Collector

Scrapes Cypher examples from Neo4j official documentation.
Extracts query examples, patterns, and best practices.

Source: https://neo4j.com/docs/cypher-manual/current/
Output: ~500-1,500 examples
"""

import json
import requests
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse

OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "raw" / "documentation"
DOC_BASE_URL = "https://neo4j.com/docs/cypher-manual/current/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Key documentation pages with Cypher examples
# Prioritized for write operations (CREATE, MERGE, DELETE, SET, REMOVE) to balance dataset
DOC_PAGES = [
    # Write operations (HIGH PRIORITY for balancing)
    "clauses/create/",
    "clauses/merge/",
    "clauses/set/",
    "clauses/delete/",
    "clauses/remove/",
    "clauses/foreach/",
    # Read operations
    "clauses/match/",
    "clauses/where/",
    "clauses/return/",
    "clauses/order-by/",
    "clauses/limit/",
    "clauses/skip/",
    "clauses/with/",
    "clauses/unwind/",
    "clauses/union/",
    "clauses/call-subquery/",
    # Functions and patterns (often contain examples with RETURN, WITH, etc.)
    "functions/",
    "functions/aggregating/",
    "functions/list/",
    "functions/string/",
    "functions/predicate/",
    "functions/scalar/",
    "functions/relationship/",
    "functions/node/",
    "functions/path/",
    "functions/pattern/",
    "functions/property/",
    "functions/type/",
    "operators/",
    "operators/comparison/",
    "operators/mathematical/",
    "operators/string/",
    "operators/boolean/",
    "patterns/",
    "patterns/pattern-matching/",
    "patterns/pattern-comprehension/",
    "patterns/variable-length-paths/",
    # Administration (often has CREATE/DELETE examples)
    "administration/indexes-for-search-performance/",
    "administration/constraints/",
    "administration/query-logging/",
    "query-tuning/",
    "query-tuning/using-indexes/",
    "query-tuning/using-explain/",
    "query-tuning/using-profile/",
    # Transactions (often has CREATE/MERGE/DELETE examples)
    "transactions/",
    "transactions/transaction-execution/",
    # Data modeling (often has CREATE examples)
    "data-modeling/",
    "data-modeling/graph-modeling/",
    # Best practices (often has comprehensive examples)
    "best-practices/",
    "best-practices/query-performance/",
    # Import/Export (often has CREATE examples)
    "import-export/",
    "import-export/import-data/",
    # Introduction and tutorials
    "introduction/",
    "introduction/getting-started/",
]


def extract_cypher_from_code_block(code_block: str) -> Optional[str]:
    """Extract Cypher query from code block"""
    if not code_block:
        return None
    
    # Remove HTML tags if present
    code_block = re.sub(r'<[^>]+>', '', code_block)
    
    # Look for Cypher keywords
    cypher_keywords = [
        'MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE',
        'RETURN', 'WITH', 'WHERE', 'ORDER BY', 'LIMIT', 'SKIP',
        'UNWIND', 'UNION', 'CALL', 'FOREACH'
    ]
    
    code_upper = code_block.upper().strip()
    has_cypher = any(kw in code_upper for kw in cypher_keywords)
    
    if not has_cypher:
        return None
    
    # Clean up the code block
    cypher = code_block.strip()
    
    # Remove common prefixes/suffixes
    cypher = re.sub(r'^cypher\s*', '', cypher, flags=re.IGNORECASE)
    cypher = re.sub(r'^neo4j\s*>\s*', '', cypher, flags=re.IGNORECASE)
    cypher = re.sub(r'^>\s*', '', cypher)
    
    # Basic validation - check for balanced parentheses, brackets, braces
    if cypher.count('(') != cypher.count(')'):
        return None
    if cypher.count('[') != cypher.count(']'):
        return None
    if cypher.count('{') != cypher.count('}'):
        return None
    
    # Minimum length check
    if len(cypher.strip()) < 10:
        return None
    
    return cypher.strip()


def extract_schema_from_context(text: str, cypher: str) -> Optional[str]:
    """Extract schema information from surrounding text"""
    schema_parts = []
    
    # Look for node labels in Cypher
    node_labels = re.findall(r'\([^:]*:(\w+)', cypher)
    if node_labels:
        unique_labels = list(set(node_labels))
        schema_parts.append(f"Node labels: {', '.join(unique_labels)}")
    
    # Look for relationship types
    rel_types = re.findall(r'\[[^:]*:(\w+)', cypher)
    if rel_types:
        unique_rels = list(set(rel_types))
        schema_parts.append(f"Relationship types: {', '.join(unique_rels)}")
    
    # Look for property patterns in text
    property_pattern = re.search(r'properties?\s*[:=]\s*\{([^}]+)\}', text, re.IGNORECASE)
    if property_pattern:
        schema_parts.append(f"Properties: {property_pattern.group(1)}")
    
    if schema_parts:
        return "\n".join(schema_parts)
    
    return None


def extract_examples_from_page(url: str) -> List[Dict[str, Any]]:
    """Extract Cypher examples from a documentation page"""
    examples = []
    
    try:
        print(f"  Fetching: {url}")
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find code blocks (usually in <pre><code> or <code> tags)
        # Also check for <pre> tags that might contain Cypher directly
        code_blocks = soup.find_all(['pre', 'code'])
        
        # Also look for divs with code classes
        code_divs = soup.find_all('div', class_=re.compile(r'code|example|snippet', re.I))
        for div in code_divs:
            code_text = div.get_text()
            if code_text.strip():
                code_blocks.append(type('MockElement', (), {'get_text': lambda: code_text})())
        
        page_text = soup.get_text()
        
        # Track seen Cypher queries to avoid duplicates within page
        seen_in_page = set()
        
        for code_block in code_blocks:
            code_text = code_block.get_text()
            
            # Extract Cypher query
            cypher = extract_cypher_from_code_block(code_text)
            if not cypher:
                continue
            
            # Skip if we've seen this exact query in this page
            cypher_key = cypher.strip().lower()
            if cypher_key in seen_in_page:
                continue
            seen_in_page.add(cypher_key)
            
            # Try to find question/description from surrounding text
            question = None
            
            # Look for preceding paragraph, heading, or list item
            prev_elem = code_block.find_previous(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
            if prev_elem:
                question_text = prev_elem.get_text().strip()
                # Clean up question text
                question_text = re.sub(r'\s+', ' ', question_text)
                # Remove common prefixes
                question_text = re.sub(r'^(Example|Example:|Example\s+\d+[:.]?\s*)', '', question_text, flags=re.IGNORECASE)
                question_text = question_text.strip()
                if len(question_text) > 15 and len(question_text) < 500:
                    question = question_text
            
            # Also check following paragraph for description
            if not question:
                next_elem = code_block.find_next(['p', 'li'])
                if next_elem:
                    question_text = next_elem.get_text().strip()
                    question_text = re.sub(r'\s+', ' ', question_text)
                    if len(question_text) > 15 and len(question_text) < 500:
                        question = question_text
            
            # Fallback: generate question from Cypher keywords
            if not question:
                cypher_upper = cypher.upper()
                if cypher_upper.startswith('CREATE'):
                    question = "Create nodes and relationships in the graph"
                elif cypher_upper.startswith('MERGE'):
                    question = "Merge nodes and relationships, creating them if they don't exist"
                elif cypher_upper.startswith('DELETE') or 'DELETE' in cypher_upper:
                    question = "Delete nodes or relationships from the graph"
                elif cypher_upper.startswith('SET'):
                    question = "Set properties or labels on nodes or relationships"
                elif cypher_upper.startswith('REMOVE'):
                    question = "Remove properties or labels from nodes or relationships"
                elif cypher_upper.startswith('UNWIND'):
                    question = "Unwind a list into individual rows"
                elif cypher_upper.startswith('WITH'):
                    question = "Chain query parts using WITH clause"
                elif cypher_upper.startswith('RETURN'):
                    question = "Return query results"
                elif 'MATCH' in cypher_upper:
                    question = "Query the graph database"
                else:
                    question = "Execute Cypher query"
            
            # Extract schema from context
            schema = extract_schema_from_context(page_text, cypher)
            
            example = {
                "question": question,
                "cypher": cypher,
                "schema": schema or "",
                "source": "neo4j_official_docs",
                "url": url,
                "category": "documentation"
            }
            
            examples.append(example)
        
        print(f"  Found {len(examples)} examples")
        
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
    
    return examples


def collect_documentation_examples() -> List[Dict[str, Any]]:
    """Collect all examples from Neo4j documentation"""
    all_examples = []
    
    print("="*70)
    print("Neo4j Official Documentation Collector")
    print("="*70)
    print(f"Base URL: {DOC_BASE_URL}")
    print(f"Pages to check: {len(DOC_PAGES)}")
    print()
    
    for page in DOC_PAGES:
        url = urljoin(DOC_BASE_URL, page)
        examples = extract_examples_from_page(url)
        all_examples.extend(examples)
        
        # Rate limiting
        time.sleep(1)
    
    # Deduplicate by Cypher query
    seen_cypher = set()
    unique_examples = []
    
    for example in all_examples:
        cypher_key = example["cypher"].strip().lower()
        if cypher_key not in seen_cypher:
            seen_cypher.add(cypher_key)
            unique_examples.append(example)
    
    print()
    print("="*70)
    print("Collection Summary")
    print("="*70)
    print(f"Total examples found: {len(all_examples)}")
    print(f"Unique examples: {len(unique_examples)}")
    print(f"Duplicates removed: {len(all_examples) - len(unique_examples)}")
    
    return unique_examples


def save_examples(examples: List[Dict[str, Any]], output_file: Path):
    """Save examples to JSONL file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(examples)} examples to: {output_file}")


def main():
    """Main collection function"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "neo4j_documentation.jsonl"
    
    examples = collect_documentation_examples()
    
    if examples:
        save_examples(examples, output_file)
        print(f"\n[OK] Collection complete!")
        print(f"Output: {output_file}")
    else:
        print("\n[WARNING] No examples collected!")


if __name__ == "__main__":
    main()

