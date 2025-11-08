use crate::manifest::Manifest;
use std::collections::HashSet;

pub struct KeywordRouter {
    experts: Vec<ExpertMetadata>,
}

#[derive(Debug, Clone)]
pub struct ExpertMetadata {
    pub name: String,
    pub keywords: HashSet<String>,
    pub priority: f64,
}

#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub expert_name: String,
    pub score: f64,
    pub matched_keywords: Vec<String>,
}

impl KeywordRouter {
    pub fn new() -> Self {
        Self {
            experts: Vec::new(),
        }
    }

    pub fn add_expert(&mut self, manifest: &Manifest) {
        if let Some(routing) = &manifest.routing {
            let keywords: HashSet<String> =
                routing.keywords.iter().map(|k| k.to_lowercase()).collect();

            let priority = routing.priority.unwrap_or(0.5) as f64;

            self.experts.push(ExpertMetadata {
                name: manifest.name.clone(),
                keywords,
                priority,
            });
        }
    }

    pub fn route(&self, query: &str, top_n: usize) -> Vec<RoutingResult> {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<String> = query_lower
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let mut results: Vec<RoutingResult> = self
            .experts
            .iter()
            .map(|expert| {
                let matched: Vec<String> = expert
                    .keywords
                    .intersection(&query_words)
                    .cloned()
                    .collect();

                let keyword_score = if expert.keywords.is_empty() {
                    0.0
                } else {
                    matched.len() as f64 / expert.keywords.len() as f64
                };

                let final_score = keyword_score * expert.priority;

                RoutingResult {
                    expert_name: expert.name.clone(),
                    score: final_score,
                    matched_keywords: matched,
                }
            })
            .filter(|r| r.score > 0.0)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_n);
        results
    }

    pub fn route_single(&self, query: &str) -> Option<String> {
        self.route(query, 1).first().map(|r| r.expert_name.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Manifest, Routing};

    #[test]
    fn test_keyword_routing() {
        let mut router = KeywordRouter::new();

        let mut sql_manifest = Manifest::default();
        sql_manifest.name = "expert-sql".to_string();
        sql_manifest.routing = Some(Routing {
            keywords: vec![
                "sql".to_string(),
                "database".to_string(),
                "query".to_string(),
            ],
            exclude_keywords: None,
            router_hint: Some("database=sql".to_string()),
            priority: Some(0.8),
        });

        router.add_expert(&sql_manifest);

        let results = router.route("show all database tables", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].expert_name, "expert-sql");
        assert!(results[0].score > 0.0);
    }
}
