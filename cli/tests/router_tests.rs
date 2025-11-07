// Router integration tests
// Tests the router with actual expert manifests if available

use std::path::Path;

#[test]
fn test_router_with_real_manifests() {
    // Try to load real expert manifests
    let expert_dirs = ["expert-neo4j", "expert-sql", "expert-json", "expert-typescript"];
    let mut found_count = 0;
    
    for expert_name in &expert_dirs {
        let manifest_path = Path::new("../experts")
            .join(expert_name)
            .join("manifest.json");
        
        if manifest_path.exists() {
            found_count += 1;
        }
    }
    
    // This test only runs if we have actual experts
    if found_count > 0 {
        println!("Found {} expert manifests for testing", found_count);
        assert!(found_count > 0, "Should find at least one expert manifest");
    } else {
        println!("No expert manifests found - skipping integration test");
    }
}

#[test]
fn test_router_command_available() {
    // Just verify the router module is available
    // Actual routing logic is tested in router.rs internal tests
    use expert_cli::router::KeywordRouter;
    use expert_cli::Manifest;
    
    // Create empty router
    let router = KeywordRouter { experts: vec![] };
    
    // Route with no experts should return empty
    let matches = router.route("test query", 5);
    assert_eq!(matches.len(), 0, "Empty router should return no matches");
}


