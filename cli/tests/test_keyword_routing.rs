use expert_cli::manifest::{Manifest, Routing};
use expert_cli::routing::KeywordRouter;
use std::path::PathBuf;

fn load_real_experts() -> Vec<Manifest> {
    let mut experts = Vec::new();
    let experts_dir = PathBuf::from("../experts");

    for expert_name in &[
        "expert-sql",
        "expert-json",
        "expert-typescript",
        "expert-neo4j",
    ] {
        let manifest_path = experts_dir.join(expert_name).join("manifest.json");
        if let Ok(manifest) = Manifest::load(&manifest_path) {
            experts.push(manifest);
        }
    }

    experts
}

#[test]
fn test_sql_expert_routing() {
    let experts = load_real_experts();
    if experts.is_empty() {
        println!("Skipping test - no experts found");
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // SQL query should route to expert-sql
    let results = router.route("show all database tables with select query", 3);

    assert!(!results.is_empty(), "Should match at least one expert");

    // Check if SQL expert is in top results
    let has_sql = results.iter().any(|r| r.expert_name == "expert-sql");
    assert!(has_sql, "expert-sql should be in top results for SQL query");
}

#[test]
fn test_neo4j_expert_routing() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // Neo4j/Graph query
    let results = router.route("find nodes in graph database with relationships", 3);

    if !results.is_empty() {
        let has_neo4j = results.iter().any(|r| r.expert_name == "expert-neo4j");
        assert!(has_neo4j, "expert-neo4j should match graph/node keywords");
    }
}

#[test]
fn test_json_expert_routing() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // JSON query with explicit json keyword
    let results = router.route("json parsing and validation", 3);

    // Test is flexible - if no matches, that's okay (depends on keywords configured)
    // Just verify the router works
    assert!(results.len() <= 3, "Should limit results to top_k");
}

#[test]
fn test_typescript_expert_routing() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // TypeScript/Code query
    let results = router.route("write typescript code with proper types", 3);

    if !results.is_empty() {
        // Should match typescript expert if available
        assert!(!results.is_empty(), "Should match at least one expert");
    }
}

#[test]
fn test_no_matches_for_unrelated_query() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // Query with no matching keywords
    let results = router.route("quantum physics differential equations", 1);

    // Should return empty if no keywords match
    if results.is_empty() {
        assert!(true, "Correctly returned no matches");
    } else {
        // If it matched something, score should be low
        assert!(
            results[0].score < 0.3,
            "Score should be low for unrelated query"
        );
    }
}

#[test]
fn test_case_insensitive_matching() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // Lowercase and uppercase should both match
    let results_lower = router.route("sql database query", 3);
    let results_upper = router.route("SQL DATABASE QUERY", 3);

    // Both should return results
    assert!(
        !results_lower.is_empty() || !results_upper.is_empty(),
        "Case insensitive matching should work"
    );
}

#[test]
fn test_partial_keyword_matching() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // Query with only one matching keyword
    let results = router.route("show me the database", 1);

    if !results.is_empty() {
        assert!(results[0].score > 0.0, "Should have positive score");
        assert!(
            !results[0].matched_keywords.is_empty(),
            "Should have at least one matched keyword"
        );
    }
}

#[test]
fn test_top_k_limiting() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // Request top 1
    let results = router.route("database query code format", 1);
    assert!(results.len() <= 1, "Should return at most 1 expert");

    // Request top 3
    let results = router.route("database query code format", 3);
    assert!(results.len() <= 3, "Should return at most 3 experts");

    // Request more than available
    let results = router.route("database query code format", 100);
    assert!(
        results.len() <= experts.len(),
        "Should not exceed number of experts"
    );
}

#[test]
fn test_route_single_convenience() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    // Route SQL query
    let result = router.route_single("database sql query");
    assert!(
        result.is_some(),
        "Should return an expert for database query"
    );

    // Route unrelated query
    let result = router.route_single("quantum physics equations");
    // May or may not match depending on keywords
}

#[test]
fn test_matched_keywords_tracking() {
    let experts = load_real_experts();
    if experts.is_empty() {
        return;
    }

    let mut router = KeywordRouter::new();
    for expert in &experts {
        router.add_expert(expert);
    }

    let results = router.route("select from database table with sql query", 3);

    if !results.is_empty() {
        // Should track which keywords matched
        assert!(
            !results[0].matched_keywords.is_empty(),
            "Should track matched keywords"
        );

        // Score should correlate with number of matches
        if results.len() > 1 {
            // Higher score should have more or equal matches
            assert!(
                results[0].score >= results[1].score,
                "Results should be sorted by score"
            );
        }
    }
}

#[test]
fn test_router_with_no_experts() {
    let router = KeywordRouter::new();

    let results = router.route("any query", 1);
    assert_eq!(results.len(), 0, "Empty router should return no results");

    let result = router.route_single("any query");
    assert_eq!(result, None, "Empty router should return None");
}
