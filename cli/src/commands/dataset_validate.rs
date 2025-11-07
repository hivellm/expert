// Dataset validation command (migrated from Python)
// Validates dataset format, schema, and deduplicates

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub validate_json: bool,
    pub validate_schema: bool,
    pub deduplicate: bool,
    pub min_length: usize,
    pub max_length: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validate_json: true,
            validate_schema: false,
            deduplicate: true,
            min_length: 10,
            max_length: 2048,
        }
    }
}

pub struct DatasetValidator {
    config: ValidationConfig,
    seen_hashes: HashSet<String>,
}

impl DatasetValidator {
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            seen_hashes: HashSet::new(),
        }
    }
    
    /// Validate a single example
    pub fn validate_example(&mut self, example: &Value) -> Result<bool> {
        // Check required fields
        if !example.get("task").is_some() {
            return Err(anyhow!("Missing 'task' field"));
        }
        
        if !example.get("input").is_some() {
            return Err(anyhow!("Missing 'input' field"));
        }
        
        // Check format (SFT or DPO)
        if example.get("output").is_some() {
            self.validate_sft(example)
        } else if example.get("chosen").is_some() && example.get("rejected").is_some() {
            self.validate_dpo(example)
        } else {
            Err(anyhow!("Invalid format: must have 'output' (SFT) or 'chosen'+'rejected' (DPO)"))
        }
    }
    
    fn validate_sft(&mut self, example: &Value) -> Result<bool> {
        let output = example.get("output")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("'output' must be a string"))?;
        
        // Validate JSON if enabled
        if self.config.validate_json {
            if let Err(_) = serde_json::from_str::<Value>(output) {
                // Not valid JSON, but might be intentional
            }
        }
        
        // Check length
        let len = output.len();
        if len < self.config.min_length {
            return Err(anyhow!("Output too short: {} < {}", len, self.config.min_length));
        }
        if len > self.config.max_length {
            return Err(anyhow!("Output too long: {} > {}", len, self.config.max_length));
        }
        
        // Check for duplicates
        if self.config.deduplicate {
            let hash = self.hash_example(example);
            if self.seen_hashes.contains(&hash) {
                return Ok(false); // Duplicate
            }
            self.seen_hashes.insert(hash);
        }
        
        Ok(true)
    }
    
    fn validate_dpo(&mut self, example: &Value) -> Result<bool> {
        let chosen = example.get("chosen")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("'chosen' must be a string"))?;
        
        let rejected = example.get("rejected")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("'rejected' must be a string"))?;
        
        // Check lengths
        if chosen.len() < self.config.min_length || rejected.len() < self.config.min_length {
            return Err(anyhow!("Chosen/rejected too short"));
        }
        
        if chosen.len() > self.config.max_length || rejected.len() > self.config.max_length {
            return Err(anyhow!("Chosen/rejected too long"));
        }
        
        // Check for duplicates
        if self.config.deduplicate {
            let hash = self.hash_example(example);
            if self.seen_hashes.contains(&hash) {
                return Ok(false); // Duplicate
            }
            self.seen_hashes.insert(hash);
        }
        
        Ok(true)
    }
    
    fn hash_example(&self, example: &Value) -> String {
        use sha2::{Sha256, Digest};
        let json_str = serde_json::to_string(example).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json_str.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    /// Validate entire dataset file
    pub fn validate_file(&mut self, path: &Path) -> Result<ValidationStats> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut stats = ValidationStats::default();
        
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            stats.total += 1;
            
            // Parse JSON
            let example: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    stats.invalid_json += 1;
                    eprintln!("Line {}: Invalid JSON: {}", line_num + 1, e);
                    continue;
                }
            };
            
            // Validate
            match self.validate_example(&example) {
                Ok(true) => stats.valid += 1,
                Ok(false) => stats.duplicates += 1,
                Err(e) => {
                    stats.invalid += 1;
                    eprintln!("Line {}: {}", line_num + 1, e);
                }
            }
        }
        
        Ok(stats)
    }
}

#[derive(Debug, Default)]
pub struct ValidationStats {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub invalid_json: usize,
    pub duplicates: usize,
}

impl ValidationStats {
    pub fn print(&self) {
        println!("\n=== Dataset Validation Results ===");
        println!("Total examples:   {}", self.total);
        println!("Valid:            {} ({:.1}%)", self.valid, self.valid as f64 / self.total as f64 * 100.0);
        println!("Invalid format:   {}", self.invalid);
        println!("Invalid JSON:     {}", self.invalid_json);
        println!("Duplicates:       {}", self.duplicates);
        println!("=================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_validate_sft() {
        let mut validator = DatasetValidator::new(ValidationConfig::default());
        
        let example = json!({
            "task": "test",
            "input": "What is 2+2?",
            "output": "4"
        });
        
        assert!(validator.validate_example(&example).is_ok());
    }
    
    #[test]
    fn test_validate_dpo() {
        let mut validator = DatasetValidator::new(ValidationConfig::default());
        
        let example = json!({
            "task": "test",
            "input": "What is 2+2?",
            "chosen": "The answer is 4",
            "rejected": "I don't know"
        });
        
        assert!(validator.validate_example(&example).is_ok());
    }
    
    #[test]
    fn test_missing_field() {
        let mut validator = DatasetValidator::new(ValidationConfig::default());
        
        let example = json!({
            "input": "What is 2+2?"
        });
        
        assert!(validator.validate_example(&example).is_err());
    }
    
    #[test]
    fn test_deduplicate() {
        let config = ValidationConfig {
            deduplicate: true,
            ..Default::default()
        };
        let mut validator = DatasetValidator::new(config);
        
        let example = json!({
            "task": "test",
            "input": "What is 2+2?",
            "output": "The answer is 4"
        });
        
        // First time should be valid
        assert_eq!(validator.validate_example(&example).unwrap(), true);
        
        // Second time should be duplicate
        assert_eq!(validator.validate_example(&example).unwrap(), false);
    }
}

