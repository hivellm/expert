// Dataset statistics command (migrated from Python)
// Generate statistics and metadata for datasets

use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_examples: usize,
    pub by_task: HashMap<String, usize>,
    pub by_format: FormatStats,
    pub length_stats: LengthStats,
    pub valid_json_rate: f64,
    pub total_chars: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FormatStats {
    pub sft: usize,
    pub dpo: usize,
    pub unknown: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LengthStats {
    pub avg_output_length: f64,
    pub max_output_length: usize,
    pub min_output_length: usize,
}

impl Default for DatasetStats {
    fn default() -> Self {
        Self {
            total_examples: 0,
            by_task: HashMap::new(),
            by_format: FormatStats {
                sft: 0,
                dpo: 0,
                unknown: 0,
            },
            length_stats: LengthStats {
                avg_output_length: 0.0,
                max_output_length: 0,
                min_output_length: usize::MAX,
            },
            valid_json_rate: 0.0,
            total_chars: 0,
        }
    }
}

impl DatasetStats {
    /// Generate statistics from dataset file
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut stats = Self::default();
        let mut total_length = 0;
        let mut valid_json_count = 0;
        
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            // Parse JSON
            let example: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            
            stats.total_examples += 1;
            
            // Count by task
            if let Some(task) = example.get("task").and_then(|v| v.as_str()) {
                *stats.by_task.entry(task.to_string()).or_insert(0) += 1;
            }
            
            // Count format
            if example.get("output").is_some() {
                stats.by_format.sft += 1;
            } else if example.get("chosen").is_some() && example.get("rejected").is_some() {
                stats.by_format.dpo += 1;
            } else {
                stats.by_format.unknown += 1;
            }
            
            // Calculate length statistics
            if let Some(output) = example.get("output").and_then(|v| v.as_str()) {
                let len = output.len();
                total_length += len;
                stats.total_chars += len;
                
                if len > stats.length_stats.max_output_length {
                    stats.length_stats.max_output_length = len;
                }
                if len < stats.length_stats.min_output_length {
                    stats.length_stats.min_output_length = len;
                }
                
                // Check if valid JSON
                if serde_json::from_str::<Value>(output).is_ok() {
                    valid_json_count += 1;
                }
            }
        }
        
        // Calculate averages
        if stats.by_format.sft > 0 {
            stats.length_stats.avg_output_length = total_length as f64 / stats.by_format.sft as f64;
        }
        
        if stats.total_examples > 0 {
            stats.valid_json_rate = valid_json_count as f64 / stats.total_examples as f64 * 100.0;
        }
        
        // Fix min if no data
        if stats.length_stats.min_output_length == usize::MAX {
            stats.length_stats.min_output_length = 0;
        }
        
        Ok(stats)
    }
    
    /// Print statistics in human-readable format
    pub fn print(&self) {
        println!("\n=== Dataset Statistics ===");
        println!("Total examples: {}", self.total_examples);
        println!();
        
        println!("By Task:");
        for (task, count) in &self.by_task {
            let percentage = (*count as f64 / self.total_examples as f64) * 100.0;
            println!("  {}: {} ({:.1}%)", task, count, percentage);
        }
        println!();
        
        println!("By Format:");
        println!("  SFT:     {}", self.by_format.sft);
        println!("  DPO:     {}", self.by_format.dpo);
        println!("  Unknown: {}", self.by_format.unknown);
        println!();
        
        println!("Length Statistics:");
        println!("  Avg:  {:.1} chars", self.length_stats.avg_output_length);
        println!("  Min:  {} chars", self.length_stats.min_output_length);
        println!("  Max:  {} chars", self.length_stats.max_output_length);
        println!("  Total: {} chars", self.total_chars);
        println!();
        
        println!("Quality:");
        println!("  Valid JSON: {:.1}%", self.valid_json_rate);
        println!("=========================");
    }
    
    /// Save statistics to JSON file
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_dataset_stats() {
        // Create temporary dataset file
        let mut file = NamedTempFile::new().unwrap();
        
        writeln!(file, r#"{{"task":"test","input":"Q1","output":"A1"}}"#).unwrap();
        writeln!(file, r#"{{"task":"test","input":"Q2","output":"A2"}}"#).unwrap();
        writeln!(file, r#"{{"task":"other","input":"Q3","output":"A3"}}"#).unwrap();
        
        file.flush().unwrap();
        
        let stats = DatasetStats::from_file(file.path()).unwrap();
        
        assert_eq!(stats.total_examples, 3);
        assert_eq!(stats.by_format.sft, 3);
        assert_eq!(stats.by_task.len(), 2);
        assert_eq!(stats.by_task.get("test"), Some(&2));
        assert_eq!(stats.by_task.get("other"), Some(&1));
    }
}

