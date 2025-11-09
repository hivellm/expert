// GBNF grammar parser for generic validation
// Completely generic - works with any GBNF grammar file

use crate::error::Result;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Parsed GBNF grammar rules - completely generic
#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    pub root_rule: String,
    pub rules: HashMap<String, GbnfRule>,
    pub keywords: Vec<String>, // All quoted strings from grammar
    pub start_patterns: Vec<String>, // Patterns that indicate start of valid output
}

#[derive(Debug, Clone)]
pub struct GbnfRule {
    pub name: String,
    pub definition: String,
}

impl GbnfGrammar {
    /// Load and parse GBNF file
    pub fn from_file(grammar_file: &Path) -> Result<Self> {
        let content = fs::read_to_string(grammar_file)?;
        Self::from_str(&content)
    }

    /// Parse GBNF content - completely generic parser
    pub fn from_str(content: &str) -> Result<Self> {
        let mut rules = HashMap::new();
        let mut root_rule = String::new();
        let mut keywords = Vec::new();
        let mut start_patterns = HashSet::new();

        // Parse rules (name ::= definition)
        // Handles multi-line rules with | and continuation
        let mut current_rule: Option<(String, String)> = None;
        
        for line in content.lines() {
            let line = line.trim();
            
            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Extract root rule
            if line.starts_with("root ::=") {
                let rule_pattern = Regex::new(r"^root\s*::=\s*(.+)$")?;
                if let Some(caps) = rule_pattern.captures(line) {
                    if let Some(def) = caps.get(1) {
                        root_rule = def.as_str().trim().to_string();
                    }
                }
                continue;
            }

            // Check if this is a rule definition
            let rule_pattern = Regex::new(r"^([a-zA-Z0-9_-]+)\s*::=\s*(.+)$")?;
            if let Some(caps) = rule_pattern.captures(line) {
                // Save previous rule if exists
                if let Some((name, def)) = current_rule.take() {
                    rules.insert(name.clone(), GbnfRule {
                        name: name.clone(),
                        definition: def.trim().to_string(),
                    });
                }
                
                if let (Some(name), Some(def)) = (caps.get(1), caps.get(2)) {
                    let rule_name = name.as_str().to_string();
                    let definition = def.as_str().trim().to_string();
                    current_rule = Some((rule_name, definition));
                }
            } else if line.contains("|") || line.starts_with("\"") {
                // Continuation of rule (alternative or quoted string)
                if let Some((name, def)) = current_rule.as_mut() {
                    *def = format!("{} {}", def, line);
                }
            } else if current_rule.is_some() {
                // End of rule, save it
                if let Some((name, def)) = current_rule.take() {
                    rules.insert(name.clone(), GbnfRule {
                        name: name.clone(),
                        definition: def.trim().to_string(),
                    });
                }
            }
        }
        
        // Save last rule
        if let Some((name, def)) = current_rule {
            rules.insert(name.clone(), GbnfRule {
                name: name.clone(),
                definition: def.trim().to_string(),
            });
        }

        // Extract all keywords (quoted strings) from all rules
        let keyword_pattern = Regex::new(r#""([^"]+)""#)?;
        for rule in rules.values() {
            for cap in keyword_pattern.captures_iter(&rule.definition) {
                if let Some(kw) = cap.get(1) {
                    let keyword = kw.as_str().to_string();
                    if !keywords.contains(&keyword) {
                        keywords.push(keyword.clone());
                        // Keywords that are likely to start a query (uppercase, common commands)
                        if keyword.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) 
                            && keyword.len() > 2 {
                            start_patterns.insert(keyword.to_uppercase());
                        }
                    }
                }
            }
        }

        // If root_rule is empty, try to find it from rules
        if root_rule.is_empty() {
            if let Some(root) = rules.get("root") {
                root_rule = root.definition.clone();
            }
        }

        Ok(Self {
            root_rule,
            rules,
            keywords,
            start_patterns: start_patterns.into_iter().collect(),
        })
    }

    /// Generic validation - checks if text matches grammar patterns
    pub fn validate(&self, text: &str) -> Result<bool> {
        let text_trimmed = text.trim();
        if text_trimmed.is_empty() {
            return Ok(false);
        }

        let text_upper = text_trimmed.to_uppercase();
        
        // Check if text starts with any keyword from grammar
        let mut found_start = false;
        for keyword in &self.keywords {
            let keyword_upper = keyword.to_uppercase();
            // Check if text starts with keyword (common pattern for queries)
            if text_upper.starts_with(&keyword_upper) || text_upper.contains(&format!(" {} ", keyword_upper)) {
                found_start = true;
                break;
            }
        }

        // If no keyword match, check if it's JSON-like (starts with { or [)
        if !found_start {
            if text_trimmed.starts_with('{') || text_trimmed.starts_with('[') {
                // Try to parse as JSON
                return Ok(serde_json::from_str::<serde_json::Value>(text_trimmed).is_ok());
            }
        }

        // Basic syntax checks
        // Check balanced parentheses, brackets, braces
        let parens_balanced = text_trimmed.matches('(').count() == text_trimmed.matches(')').count();
        let brackets_balanced = text_trimmed.matches('[').count() == text_trimmed.matches(']').count();
        let braces_balanced = text_trimmed.matches('{').count() == text_trimmed.matches('}').count();

        Ok(found_start && parens_balanced && brackets_balanced && braces_balanced)
    }

    /// Generic query extraction - removes reasoning and extracts valid output
    pub fn extract_query(&self, text: &str) -> Result<String> {
        // Remove reasoning blocks
        let think_pattern = Regex::new(r"(?i)<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>")?;
        let mut cleaned = think_pattern.replace_all(text, "").to_string();
        
        // Remove standalone reasoning tags
        let standalone_pattern = Regex::new(r"(?i)<(?:think|redacted_reasoning)>.*")?;
        cleaned = standalone_pattern.replace_all(&cleaned, "").to_string();

        // Find the start of valid output based on grammar keywords
        let mut start_pos = 0;
        let cleaned_upper = cleaned.to_uppercase();
        
        // Try to find first occurrence of any grammar keyword
        for keyword in &self.keywords {
            let keyword_upper = keyword.to_uppercase();
            if let Some(pos) = cleaned_upper.find(&keyword_upper) {
                if start_pos == 0 || pos < start_pos {
                    start_pos = pos;
                }
            }
        }

        // If found keyword, start from there
        if start_pos > 0 {
            cleaned = cleaned[start_pos..].to_string();
        }

        // Try JSON first (common pattern)
        if cleaned.trim().starts_with('{') || cleaned.trim().starts_with('[') {
            let json_pattern = Regex::new(r"(\{.*?\}|\[.*?\])")?;
            if let Some(mat) = json_pattern.find(&cleaned) {
                let json_str = mat.as_str();
                if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                    return Ok(json_str.to_string());
                }
            }
        }

        // Remove explanatory text after the query
        // Common patterns: explanations start with "Okay", "Let me", "The user", etc.
        let stop_patterns = vec![
            r"(?i)\n\n(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|This query|The result|Explanation|Note|Remember)",
            r"(?i);\s*(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|This query|The result|Explanation)",
        ];

        for pattern_str in stop_patterns {
            if let Ok(stop_pattern) = Regex::new(pattern_str) {
                if let Some(stop_match) = stop_pattern.find(&cleaned) {
                    cleaned = cleaned[..stop_match.start()].trim().to_string();
                    break;
                }
            }
        }

        // Remove text before first keyword if present
        if !self.keywords.is_empty() {
            for keyword in &self.keywords {
                let keyword_upper = keyword.to_uppercase();
                let cleaned_upper = cleaned.to_uppercase();
                if let Some(pos) = cleaned_upper.find(&keyword_upper) {
                    if pos > 0 {
                        cleaned = cleaned[pos..].to_string();
                        break;
                    }
                }
            }
        }

        Ok(cleaned.trim().to_string())
    }
}
