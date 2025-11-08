// End-to-end integration tests

use expert::manifest::Manifest;
use expert::routing::{ConfidenceScorer, EmbeddingRouter, KeywordRouter};
use expert::runtime::ExpertManager;
use std::path::PathBuf;

#[test]
fn test_end_to_end_routing() {
    // Setup router
    let mut keyword_router = KeywordRouter::new();
    let mut embedding_router = EmbeddingRouter::new();
    let mut confidence_scorer = ConfidenceScorer::new();

    // Add experts
    let mut sql_manifest = Manifest::default();
    sql_manifest.name = "expert-sql".to_string();
    sql_manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["sql".to_string(), "database".to_string()],
        router_hint: None,
        priority: Some(0.8),
    });

    keyword_router.add_expert(&sql_manifest);
    embedding_router.add_expert(&sql_manifest);

    // Route query
    let query = "show all database tables";

    // Keyword routing
    let keyword_results = keyword_router.route(query, 1);
    assert!(keyword_results.len() > 0);

    // Embedding routing
    let embedding_results = embedding_router.route(query, 1);
    assert!(embedding_results.len() > 0);

    // Confidence scoring
    let confidence = confidence_scorer.score(
        &keyword_results[0].expert_name,
        query,
        &keyword_results[0].matched_keywords,
        keyword_results[0].score,
    );

    assert!(confidence.confidence > 0.0);
    assert!(confidence.confidence <= 1.0);
}

#[test]
fn test_expert_manager_integration() {
    let mut manager = ExpertManager::new(2);

    // Register experts
    for name in &["expert-sql", "expert-json", "expert-typescript"] {
        let mut manifest = Manifest::default();
        manifest.name = name.to_string();
        manager.register_expert(
            name.to_string(),
            manifest,
            PathBuf::from(format!("test/{}", name)),
        );
    }

    // Verify integration
    let stats = manager.stats();
    assert_eq!(stats.total_experts, 3);
    assert_eq!(stats.loaded_experts, 0);

    // Verify manager can handle multiple experts
    assert!(stats.memory_mb >= 0.0);
}

#[test]
fn test_routing_consistency() {
    let mut keyword_router = KeywordRouter::new();
    let mut embedding_router = EmbeddingRouter::new();

    // Add same expert to both routers
    let mut manifest = Manifest::default();
    manifest.name = "expert-sql".to_string();
    manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["sql".to_string(), "database".to_string()],
        router_hint: None,
        priority: Some(0.8),
    });

    keyword_router.add_expert(&manifest);
    embedding_router.add_expert(&manifest);

    // Both should route to same expert for clear queries
    let query = "show all database tables";

    let keyword_result = keyword_router.route_single(query);
    let embedding_result = embedding_router.route_single(query);

    assert!(keyword_result.is_some());
    assert!(embedding_result.is_some());

    // Both should match expert-sql
    assert_eq!(keyword_result.unwrap(), "expert-sql");
    assert_eq!(embedding_result.unwrap().expert_name, "expert-sql");
}
