// Grammar-based validation for query experts
// Validates generated output against language-specific grammars
// Uses GBNF files from experts for generic validation

use crate::error::Result;
use regex::Regex;
use std::path::Path;

pub use super::gbnf_parser::GbnfGrammar;

/// Grammar validator trait
pub trait GrammarValidator {
    fn validate(&self, text: &str) -> Result<bool>;
    fn extract_query(&self, text: &str) -> Result<String>;
}

/// Cypher grammar validator
pub struct CypherValidator;

impl GrammarValidator for CypherValidator {
    fn validate(&self, text: &str) -> Result<bool> {
        // Basic Cypher validation: check for required keywords and syntax
        let text_upper = text.to_uppercase();
        let text_upper = text_upper.trim();
        
        // Must have at least one key command
        let has_command = text_upper.contains("MATCH")
            || text_upper.contains("CREATE")
            || text_upper.contains("MERGE")
            || text_upper.contains("RETURN")
            || text_upper.contains("WITH");
        
        if !has_command {
            return Ok(false);
        }
        
        // Check for balanced parentheses, brackets, braces
        let parens_balanced = text.matches('(').count() == text.matches(')').count();
        let brackets_balanced = text.matches('[').count() == text.matches(']').count();
        let braces_balanced = text.matches('{').count() == text.matches('}').count();
        
        Ok(parens_balanced && brackets_balanced && braces_balanced)
    }
    
    fn extract_query(&self, text: &str) -> Result<String> {
        use regex::Regex;
        
        // Remove reasoning blocks
        let think_pattern = Regex::new(r"(?i)<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>")?;
        let mut cleaned = think_pattern.replace_all(text, "").to_string();
        
        // Extract Cypher query (starts with MATCH, CREATE, MERGE, etc.)
        let cypher_pattern = Regex::new(r"(?i)(MATCH|CREATE|MERGE|RETURN|WITH|UNWIND).*?$")?;
        if let Some(mat) = cypher_pattern.find(&cleaned) {
            let query = mat.as_str();
            // Stop at reasoning prefixes
            let stop_pattern = Regex::new(r"(?i)\n\n(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at)")?;
            if let Some(stop_match) = stop_pattern.find(query) {
                return Ok(query[..stop_match.start()].trim().to_string());
            }
            return Ok(query.trim().to_string());
        }
        
        Ok(cleaned.trim().to_string())
    }
}

/// SQL grammar validator
pub struct SqlValidator {
    dialect: String,
}

impl SqlValidator {
    pub fn new(dialect: &str) -> Self {
        Self {
            dialect: dialect.to_string(),
        }
    }
}

impl GrammarValidator for SqlValidator {
    fn validate(&self, text: &str) -> Result<bool> {
        // Basic SQL validation: check for required keywords
        let text_upper = text.to_uppercase();
        let text_upper = text_upper.trim();
        
        // Must have at least one key command
        let has_command = text_upper.starts_with("SELECT")
            || text_upper.starts_with("WITH")
            || text_upper.starts_with("INSERT")
            || text_upper.starts_with("UPDATE")
            || text_upper.starts_with("DELETE")
            || text_upper.starts_with("CREATE");
        
        if !has_command {
            return Ok(false);
        }
        
        // Check for balanced parentheses
        let parens_balanced = text.matches('(').count() == text.matches(')').count();
        
        Ok(parens_balanced)
    }
    
    fn extract_query(&self, text: &str) -> Result<String> {
        use regex::Regex;
        
        // Remove reasoning blocks
        let think_pattern = Regex::new(r"(?i)<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>")?;
        let mut cleaned = think_pattern.replace_all(text, "").to_string();
        
        // Extract SQL query
        let sql_pattern = Regex::new(r"(?i)(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE).*?$")?;
        if let Some(mat) = sql_pattern.find(&cleaned) {
            let query = mat.as_str();
            // Stop at reasoning prefixes
            let stop_pattern = Regex::new(r"(?i)\n\n(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|Explanation)")?;
            if let Some(stop_match) = stop_pattern.find(query) {
                return Ok(query[..stop_match.start()].trim().to_string());
            }
            // Remove trailing semicolons if followed by text
            let query = Regex::new(r";\s*(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at|This query|The result|Explanation).*")?
                .replace(&query, ";");
            return Ok(query.trim().to_string());
        }
        
        Ok(cleaned.trim().to_string())
    }
}

/// JSON grammar validator
pub struct JsonValidator;

impl GrammarValidator for JsonValidator {
    fn validate(&self, text: &str) -> Result<bool> {
        // Try to parse as JSON
        match serde_json::from_str::<serde_json::Value>(text.trim()) {
            Ok(_) => Ok(true),
            Err(_) => {
                // Try to extract JSON from code blocks
                use regex::Regex;
                let code_block_pattern = Regex::new(r"```(?:json)?\s*(\{.*?\}|\[.*?\])```")?;
                if let Some(caps) = code_block_pattern.captures(text) {
                    if let Some(json_str) = caps.get(1) {
                        return Ok(serde_json::from_str::<serde_json::Value>(json_str.as_str()).is_ok());
                    }
                }
                Ok(false)
            }
        }
    }
    
    fn extract_query(&self, text: &str) -> Result<String> {
        // Try to find JSON object/array
        let json_pattern = Regex::new(r"(\{.*?\}|\[.*?\])")?;
        if let Some(mat) = json_pattern.find(text) {
            let json_str = mat.as_str();
            // Validate it's actually JSON
            if serde_json::from_str::<serde_json::Value>(json_str).is_ok() {
                return Ok(json_str.to_string());
            }
        }
        
        // Try code blocks
        let code_block_pattern = Regex::new(r"```(?:json)?\s*(\{.*?\}|\[.*?\])```")?;
        if let Some(caps) = code_block_pattern.captures(text) {
            if let Some(json_str) = caps.get(1) {
                if serde_json::from_str::<serde_json::Value>(json_str.as_str()).is_ok() {
                    return Ok(json_str.as_str().to_string());
                }
            }
        }
        
        Ok(text.trim().to_string())
    }
}

/// Elastic grammar validator (supports JSON DSL, KQL, EQL)
pub struct ElasticValidator;

impl GrammarValidator for ElasticValidator {
    fn validate(&self, text: &str) -> Result<bool> {
        let text_trimmed = text.trim();
        
        // Try JSON first (Query DSL)
        if text_trimmed.starts_with('{') {
            return Ok(serde_json::from_str::<serde_json::Value>(text_trimmed).is_ok());
        }
        
        // Try KQL (starts with WHERE or field patterns)
        if text_trimmed.to_uppercase().starts_with("WHERE")
            || text_trimmed.contains(':')
        {
            return Ok(true); // Basic KQL validation
        }
        
        // Try EQL (starts with process, file, network, etc.)
        if text_trimmed.to_lowercase().starts_with("process")
            || text_trimmed.to_lowercase().starts_with("file")
            || text_trimmed.to_lowercase().starts_with("network")
        {
            return Ok(true); // Basic EQL validation
        }
        
        Ok(false)
    }
    
    fn extract_query(&self, text: &str) -> Result<String> {
        // Try JSON first (Query DSL)
        let json_pattern = Regex::new(r#"(\{.*?"query".*?\})"#)?;
        if let Some(mat) = json_pattern.find(text) {
            if serde_json::from_str::<serde_json::Value>(mat.as_str()).is_ok() {
                return Ok(mat.as_str().to_string());
            }
        }
        
        // Try KQL
        let kql_pattern = Regex::new(r"(?i)(WHERE\s+.*?)(?:\n\n|$)")?;
        if let Some(mat) = kql_pattern.find(text) {
            return Ok(mat.as_str().trim().to_string());
        }
        
        // Try EQL
        let eql_pattern = Regex::new(r"(?i)(process\s+.*?|file\s+.*?|network\s+.*?)(?:\n\n|$)")?;
        if let Some(mat) = eql_pattern.find(text) {
            return Ok(mat.as_str().trim().to_string());
        }
        
        Ok(text.trim().to_string())
    }
}

/// GBNF-based validator (generic, uses GBNF files)
pub struct GbnfValidator {
    grammar: GbnfGrammar,
}

impl GbnfValidator {
    pub fn new(grammar: GbnfGrammar) -> Self {
        Self { grammar }
    }
}

impl GrammarValidator for GbnfValidator {
    fn validate(&self, text: &str) -> Result<bool> {
        self.grammar.validate(text)
    }

    fn extract_query(&self, text: &str) -> Result<String> {
        self.grammar.extract_query(text)
    }
}

/// Create validator based on grammar type (deprecated - use GBNF instead)
pub fn create_validator(grammar_type: &str) -> Box<dyn GrammarValidator> {
    // For backwards compatibility, but prefer GBNF
    match grammar_type.to_lowercase().as_str() {
        "cypher" | "neo4j" => Box::new(CypherValidator),
        "sql" | "sql-postgres" | "sql-mysql" => Box::new(SqlValidator::new(grammar_type)),
        "json" => Box::new(JsonValidator),
        "elastic" | "elasticsearch" => Box::new(ElasticValidator),
        _ => Box::new(JsonValidator),
    }
}

/// Load grammar file and create validator - ALWAYS uses GBNF if available
pub fn load_grammar_validator(grammar_file: &Path) -> Result<Box<dyn GrammarValidator>> {
    // Always try GBNF first (generic approach)
    if grammar_file.exists() {
        let ext = grammar_file.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext == "gbnf" || grammar_file.file_name().and_then(|n| n.to_str()).unwrap_or("").ends_with(".gbnf") {
            let grammar = GbnfGrammar::from_file(grammar_file)?;
            return Ok(Box::new(GbnfValidator::new(grammar)));
        }
    }
    
    // Fallback: try to find grammar.gbnf in same directory
    if let Some(parent) = grammar_file.parent() {
        let gbnf_path = parent.join("grammar.gbnf");
        if gbnf_path.exists() {
            let grammar = GbnfGrammar::from_file(&gbnf_path)?;
            return Ok(Box::new(GbnfValidator::new(grammar)));
        }
    }
    
    // Last resort: use legacy validators based on file name
    let file_name = grammar_file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    
    if file_name.contains("cypher") || file_name.contains("neo4j") {
        Ok(Box::new(CypherValidator))
    } else if file_name.contains("sql") {
        Ok(Box::new(SqlValidator::new("postgres")))
    } else if file_name.contains("elastic") {
        Ok(Box::new(ElasticValidator))
    } else {
        Ok(Box::new(JsonValidator))
    }
}

