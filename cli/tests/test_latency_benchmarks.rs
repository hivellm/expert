// Latency benchmarks

use std::time::Instant;
use expert::routing::{KeywordRouter, EmbeddingRouter};
use expert::manifest::Manifest;

#[test]
fn test_keyword_router_latency() {
    let mut router = KeywordRouter::new();
    
    // Add expert
    let mut manifest = Manifest::default();
    manifest.name = "expert-sql".to_string();
    manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["sql".to_string(), "database".to_string()],
        router_hint: None,
        priority: Some(0.8),
    });
    router.add_expert(&manifest);
    
    // Benchmark routing
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = router.route("show all database tables", 1);
    }
    let elapsed = start.elapsed();
    
    let avg_latency_ms = elapsed.as_millis() as f64 / 1000.0;
    
    // Keyword router should be very fast (<1ms per query)
    assert!(avg_latency_ms < 1.0, "Keyword router too slow: {:.2}ms", avg_latency_ms);
}

#[test]
fn test_embedding_router_latency() {
    let mut router = EmbeddingRouter::new();
    
    // Add expert
    let mut manifest = Manifest::default();
    manifest.name = "expert-sql".to_string();
    manifest.routing = Some(expert::manifest::Routing {
        keywords: vec!["sql".to_string(), "database".to_string()],
        router_hint: None,
        priority: Some(0.8),
    });
    router.add_expert(&manifest);
    
    // Benchmark routing
    let start = Instant::now();
    for _ in 0..100 {
        let _ = router.route("show all database tables", 1);
    }
    let elapsed = start.elapsed();
    
    let avg_latency_ms = elapsed.as_millis() as f64 / 100.0;
    
    // Embedding router should be reasonably fast (<10ms per query)
    assert!(avg_latency_ms < 10.0, "Embedding router too slow: {:.2}ms", avg_latency_ms);
}

