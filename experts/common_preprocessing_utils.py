#!/usr/bin/env python3
"""
Common preprocessing utilities for query experts
Ensures query-only responses (no explanatory text)
"""

import re
from typing import Optional


def extract_query_only(text: str, query_type: str = "auto") -> str:
    """
    Extract only the query from response, removing any explanatory text.
    
    Args:
        text: Raw response text (may contain reasoning/explanation)
        query_type: Type of query to extract ("cypher", "sql", "json", "elastic", "auto")
    
    Returns:
        Clean query string (only the query, no explanation)
    """
    if not text or not text.strip():
        return ""
    
    # Remove reasoning blocks (Qwen3 uses <think>...</think>, legacy uses <think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Detect query type if auto
    if query_type == "auto":
        query_type = detect_query_type(text)
    
    # Extract query based on type
    if query_type == "cypher":
        return extract_cypher_only(text)
    elif query_type == "sql":
        return extract_sql_only(text)
    elif query_type == "json":
        return extract_json_only(text)
    elif query_type == "elastic":
        return extract_elastic_only(text)
    else:
        # Generic extraction - try all patterns
        for extractor in [extract_cypher_only, extract_sql_only, extract_json_only, extract_elastic_only]:
            result = extractor(text)
            if result:
                return result
        return text.strip()


def detect_query_type(text: str) -> str:
    """Detect query type from text content"""
    text_upper = text.upper().strip()
    
    # Cypher patterns
    if any(kw in text_upper for kw in ['MATCH', 'CREATE', 'MERGE', 'RETURN', 'WITH', 'UNWIND']):
        if 'MATCH' in text_upper or 'CREATE' in text_upper or 'MERGE' in text_upper:
            return "cypher"
    
    # SQL patterns
    if any(kw in text_upper for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE TABLE']):
        if text_upper.startswith('SELECT') or text_upper.startswith('WITH'):
            return "sql"
    
    # JSON patterns
    if text.strip().startswith('{') or text.strip().startswith('['):
        try:
            import json
            json.loads(text.strip())
            return "json"
        except:
            pass
    
    # Elastic patterns (KQL/EQL/DSL)
    if any(kw in text_upper for kw in ['WHERE', 'AND', 'OR']) and ('event.' in text.lower() or 'process.' in text.lower()):
        return "elastic"
    
    return "unknown"


def extract_cypher_only(text: str) -> str:
    """Extract only Cypher query, removing explanation"""
    # Find first Cypher keyword
    cypher_patterns = [
        r'(MATCH.*?)(?:\n\n|$)',
        r'(CREATE.*?)(?:\n\n|$)',
        r'(MERGE.*?)(?:\n\n|$)',
        r'(RETURN.*?)(?:\n\n|$)',
        r'(WITH.*?)(?:\n\n|$)',
    ]
    
    for pattern in cypher_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            # Stop at reasoning prefixes
            stop_pattern = r'(?i)\n\n(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|This query|The result)'
            stop_match = re.search(stop_pattern, query)
            if stop_match:
                query = query[:stop_match.start()].strip()
            return query
    
    # Fallback: extract lines with Cypher keywords
    lines = text.split('\n')
    cypher_lines = []
    for line in lines:
        line_upper = line.upper().strip()
        if any(kw in line_upper for kw in ['MATCH', 'CREATE', 'MERGE', 'RETURN', 'WITH', 'WHERE', 'ORDER BY', 'LIMIT']):
            cypher_lines.append(line.strip())
        elif cypher_lines and (line.strip().startswith('(') or line.strip().startswith('[') or ':' in line):
            cypher_lines.append(line.strip())
        elif cypher_lines and not line.strip():
            # Empty line might be end of query
            break
    
    if cypher_lines:
        return '\n'.join(cypher_lines)
    
    return text.strip()


def extract_sql_only(text: str) -> str:
    """Extract only SQL query, removing explanation"""
    # Find first SQL keyword
    sql_patterns = [
        r'(SELECT.*?)(?:\n\n|$)',
        r'(WITH.*?SELECT.*?)(?:\n\n|$)',
        r'(INSERT.*?)(?:\n\n|$)',
        r'(UPDATE.*?)(?:\n\n|$)',
        r'(DELETE.*?)(?:\n\n|$)',
        r'(CREATE TABLE.*?)(?:\n\n|$)',
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            # Stop at reasoning prefixes
            stop_pattern = r'(?i)\n\n(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|This query|The result|Explanation)'
            stop_match = re.search(stop_pattern, query)
            if stop_match:
                query = query[:stop_match.start()].strip()
            # Remove trailing semicolons if followed by text
            query = re.sub(r';\s*(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|This query|The result|Explanation).*', ';', query, flags=re.IGNORECASE)
            return query
    
    # Fallback: extract lines with SQL keywords
    lines = text.split('\n')
    sql_lines = []
    for line in lines:
        line_upper = line.upper().strip()
        if any(kw in line_upper for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT']):
            sql_lines.append(line.strip())
        elif sql_lines and (line.strip().startswith('(') or ',' in line or 'JOIN' in line_upper):
            sql_lines.append(line.strip())
        elif sql_lines and not line.strip() and len(sql_lines) > 1:
            # Empty line might be end of query
            break
    
    if sql_lines:
        return '\n'.join(sql_lines)
    
    return text.strip()


def extract_json_only(text: str) -> str:
    """Extract only JSON, removing explanation"""
    # Try to find JSON object/array
    json_patterns = [
        r'(\{.*?\})',  # JSON object
        r'(\[.*?\])',  # JSON array
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            # Take the first complete JSON
            for match in matches:
                json_str = match.group(1)
                try:
                    import json
                    json.loads(json_str)  # Validate
                    return json_str
                except:
                    continue
    
    # Fallback: try to extract from code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return text.strip()


def extract_elastic_only(text: str) -> str:
    """Extract only Elastic query (KQL/EQL/DSL), removing explanation"""
    # Try JSON first (Query DSL)
    json_match = re.search(r'(\{.*?"query".*?\})', text, re.DOTALL)
    if json_match:
        try:
            import json
            json.loads(json_match.group(1))
            return json_match.group(1).strip()
        except:
            pass
    
    # Try KQL (starts with WHERE or field patterns)
    kql_pattern = r'(WHERE\s+.*?)(?:\n\n|$)'
    match = re.search(kql_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try EQL (starts with process, file, network, etc.)
    eql_pattern = r'(process\s+.*?|file\s+.*?|network\s+.*?)(?:\n\n|$)'
    match = re.search(eql_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return text.strip()


def sanitize_chatml_response(text: str, query_type: str = "auto") -> str:
    """
    Sanitize ChatML response to ensure query-only output.
    This should be applied during preprocessing to clean training data.
    
    Note: For Qwen3 training, preprocessing scripts should use 75% reasoning + 25% direct outputs.
    Reasoning blocks use <think>...</think> tags (Qwen3 format).
    """
    # Extract from ChatML format if present
    if '<|im_start|>assistant' in text:
        match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    if '<|assistant|>' in text:
        match = re.search(r'<\|assistant\|>\s*\n(.*?)\n<\|end\|>', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # Remove reasoning blocks (Qwen3 uses <think>...</think>, legacy uses <think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Extract query only
    return extract_query_only(text, query_type)

