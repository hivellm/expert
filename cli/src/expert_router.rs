// Expert Router - Intelligent expert selection based on prompt analysis
// Uses keyword matching from manifest.routing to select best expert

use crate::manifest::Manifest;
use std::path::PathBuf;

#[derive(Clone)]
pub struct LoadedExpert {
    pub name: String,
    pub manifest: Manifest,
    pub adapter_path: PathBuf,
}

pub struct ExpertRouter {
    experts: Vec<LoadedExpert>,
}

impl ExpertRouter {
    pub fn new(experts: Vec<LoadedExpert>) -> Self {
        Self { experts }
    }
    
    /// Select best expert for the given prompt
    /// Returns None if prompt is generic (should use base model)
    pub fn select_expert(&self, prompt: &str) -> Option<&LoadedExpert> {
        // Check if this is a generic query first
        if Self::is_generic_query(prompt) {
            return None; // Use base model for generic queries
        }
        
        // Score each expert
        let mut scored_experts: Vec<(&LoadedExpert, f32)> = self.experts
            .iter()
            .map(|expert| (expert, Self::score_expert(expert, prompt)))
            .collect();
        
        // Sort by score (highest first)
        scored_experts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return highest scoring expert if score > 0
        if let Some((expert, score)) = scored_experts.first() {
            if *score > 0.0 {
                return Some(expert);
            }
        }
        
        None
    }
    
    /// Check if prompt is a generic/explanatory query
    fn is_generic_query(prompt: &str) -> bool {
        let prompt_lower = prompt.to_lowercase();
        
        let generic_patterns = vec![
            "what is",
            "what are",
            "who is",
            "who are",
            "where is",
            "where are",
            "when is",
            "when are",
            "how to",
            "how do",
            "why",
            "explain",
            "meaning of",
            "definition of",
            "describe",
            "tell me about",
            "can you explain",
        ];
        
        generic_patterns.iter().any(|pattern| prompt_lower.contains(pattern))
    }
    
    /// Score an expert based on keyword matches in prompt
    fn score_expert(expert: &LoadedExpert, prompt: &str) -> f32 {
        let prompt_lower = prompt.to_lowercase();
        
        // Get routing config from manifest
        let routing = match &expert.manifest.routing {
            Some(r) => r,
            None => return 0.0, // No routing config = no match
        };
        
        let mut score = 0.0;
        
        // Add points for matching keywords
        for keyword in &routing.keywords {
            if prompt_lower.contains(&keyword.to_lowercase()) {
                score += 1.0;
            }
        }
        
        // Subtract points for exclude keywords
        if let Some(ref exclude) = routing.exclude_keywords {
            for keyword in exclude {
                if prompt_lower.contains(&keyword.to_lowercase()) {
                    score -= 2.0; // Penalty is stronger than match
                }
            }
        }
        
        // Apply priority multiplier
        if let Some(priority) = routing.priority {
            score *= priority;
        }
        
        // Also check capabilities for additional context
        for capability in &expert.manifest.capabilities {
            if let Some(cap_type) = capability.split(':').nth(1) {
                if prompt_lower.contains(cap_type) {
                    score += 0.5; // Bonus for capability match
                }
            }
        }
        
        score
    }
    
    /// Get all loaded experts
    pub fn experts(&self) -> &[LoadedExpert] {
        &self.experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generic_detection() {
        assert!(ExpertRouter::is_generic_query("What is SQL?"));
        assert!(ExpertRouter::is_generic_query("Explain how Neo4j works"));
        assert!(ExpertRouter::is_generic_query("Tell me about databases"));
        
        assert!(!ExpertRouter::is_generic_query("SELECT * FROM users"));
        assert!(!ExpertRouter::is_generic_query("MATCH (n) RETURN n"));
        assert!(!ExpertRouter::is_generic_query("Find all products"));
    }
}
