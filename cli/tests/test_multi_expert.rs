// Multi-expert load tests

// use expert::registry::ExpertRegistry;
use expert::routing::{EmbeddingRouter, KeywordRouter};
use expert::runtime::ExpertManager;
use std::path::PathBuf;

#[test]
fn test_multi_expert_registration() {
    let mut manager = ExpertManager::new(3);

    // Register multiple experts
    let mut sql_manifest = expert::manifest::Manifest::default();
    sql_manifest.name = "expert-sql".to_string();
    manager.register_expert(
        "expert-sql".to_string(),
        sql_manifest,
        PathBuf::from("test/sql"),
    );

    let mut json_manifest = expert::manifest::Manifest::default();
    json_manifest.name = "expert-json".to_string();
    manager.register_expert(
        "expert-json".to_string(),
        json_manifest,
        PathBuf::from("test/json"),
    );

    let stats = manager.stats();
    assert_eq!(stats.total_experts, 2);
    assert_eq!(stats.loaded_experts, 0);
}

#[test]
fn test_routing_multiple_experts() {
    let mut router = KeywordRouter::new();

    // Add multiple experts
    let mut sql_manifest = expert::manifest::Manifest::default();
    sql_manifest.name = "expert-sql".to_string();
    sql_manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["sql".to_string(), "database".to_string()],
        router_hint: None,
        priority: Some(0.8),
    });
    router.add_expert(&sql_manifest);

    let mut json_manifest = expert::manifest::Manifest::default();
    json_manifest.name = "expert-json".to_string();
    json_manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["json".to_string(), "format".to_string()],
        router_hint: None,
        priority: Some(0.7),
    });
    router.add_expert(&json_manifest);

    // Route SQL query
    let results = router.route("show all database tables", 2);
    assert!(results.len() > 0);
    assert_eq!(results[0].expert_name, "expert-sql");

    // Route JSON query
    let results = router.route("format json data", 2);
    assert!(results.len() > 0);
    assert_eq!(results[0].expert_name, "expert-json");
}

#[test]
fn test_embedding_router_multiple_experts() {
    let mut router = EmbeddingRouter::new();

    // Add multiple experts
    let mut sql_manifest = expert::manifest::Manifest::default();
    sql_manifest.name = "expert-sql".to_string();
    sql_manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["sql".to_string(), "database".to_string()],
        router_hint: None,
        priority: Some(0.8),
    });
    router.add_expert(&sql_manifest);

    let mut json_manifest = expert::manifest::Manifest::default();
    json_manifest.name = "expert-json".to_string();
    json_manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["json".to_string(), "format".to_string()],
        router_hint: None,
        priority: Some(0.7),
    });
    router.add_expert(&json_manifest);

    // Route SQL query
    let results = router.route("show all database tables", 2);
    assert!(results.len() > 0);

    // Route JSON query
    let results = router.route("format json data", 2);
    assert!(results.len() > 0);
}
