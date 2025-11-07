// Routing confidence scorer
// Computes confidence scores for routing decisions

use crate::manifest::Manifest;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct ConfidenceScorer {
    // Historical statistics
    expert_success_rate: std::collections::HashMap<String, f32>,
    query_patterns: std::collections::HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct ConfidenceScore {
    pub score: f32,
    pub confidence: f32,
    pub factors: Vec<String>,
}

impl ConfidenceScorer {
    pub fn new() -> Self {
        Self {
            expert_success_rate: std::collections::HashMap::new(),
            query_patterns: std::collections::HashMap::new(),
        }
    }

    /// Compute confidence score for a routing decision
    pub fn score(
        &self,
        expert_name: &str,
        query: &str,
        matched_keywords: &[String],
        base_score: f32,
    ) -> ConfidenceScore {
        let mut confidence = base_score;
        let mut factors = Vec::new();
        
        // Factor 1: Keyword match quality
        let keyword_factor = if matched_keywords.is_empty() {
            0.5
        } else {
            (matched_keywords.len() as f32 / 5.0).min(1.0) // Normalize to max 5 keywords
        };
        confidence *= keyword_factor;
        factors.push(format!("keyword_match: {:.2}", keyword_factor));
        
        // Factor 2: Query length (longer queries = more context = higher confidence)
        let query_length_factor = (query.len() as f32 / 100.0).min(1.0);
        confidence *= 0.7 + 0.3 * query_length_factor;
        factors.push(format!("query_length: {:.2}", query_length_factor));
        
        // Factor 3: Historical success rate
        if let Some(&success_rate) = self.expert_success_rate.get(expert_name) {
            confidence *= 0.8 + 0.2 * success_rate;
            factors.push(format!("historical_success: {:.2}", success_rate));
        }
        
        // Factor 4: Query pattern match
        let query_pattern = self.extract_pattern(query);
        if let Some(&pattern_score) = self.query_patterns.get(&query_pattern) {
            confidence *= 0.9 + 0.1 * pattern_score;
            factors.push(format!("pattern_match: {:.2}", pattern_score));
        }
        
        // Normalize confidence to [0, 1]
        confidence = confidence.min(1.0).max(0.0);
        
        ConfidenceScore {
            score: base_score,
            confidence,
            factors,
        }
    }

    /// Extract query pattern (simplified)
    fn extract_pattern(&self, query: &str) -> String {
        // Extract first few words as pattern
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().take(3).collect();
        words.join("_")
    }

    /// Update success rate for an expert
    pub fn record_success(&mut self, expert_name: String, success: bool) {
        let entry = self.expert_success_rate.entry(expert_name).or_insert(0.5);
        // Exponential moving average
        *entry = *entry * 0.9 + (if success { 1.0 } else { 0.0 }) * 0.1;
    }

    /// Update pattern score
    pub fn record_pattern(&mut self, pattern: String, score: f32) {
        let entry = self.query_patterns.entry(pattern).or_insert(0.5);
        *entry = *entry * 0.9 + score * 0.1;
    }
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_scoring() {
        let scorer = ConfidenceScorer::new();
        
        let score = scorer.score(
            "expert-sql",
            "show all database tables",
            &["sql".to_string(), "database".to_string()],
            0.85,
        );
        
        assert!(score.confidence > 0.0);
        assert!(score.confidence <= 1.0);
        assert!(!score.factors.is_empty());
    }
}

