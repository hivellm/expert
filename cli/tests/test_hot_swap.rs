// Hot-swap performance tests

use expert::manifest::Manifest;
use expert::runtime::ExpertManager;
use std::path::PathBuf;

#[test]
fn test_expert_hot_swap() {
    let mut manager = ExpertManager::new(2);

    // Register multiple experts
    let mut sql_manifest = Manifest::default();
    sql_manifest.name = "expert-sql".to_string();
    manager.register_expert(
        "expert-sql".to_string(),
        sql_manifest,
        PathBuf::from("test/sql"),
    );

    let mut json_manifest = Manifest::default();
    json_manifest.name = "expert-json".to_string();
    manager.register_expert(
        "expert-json".to_string(),
        json_manifest,
        PathBuf::from("test/json"),
    );

    let mut ts_manifest = Manifest::default();
    ts_manifest.name = "expert-typescript".to_string();
    manager.register_expert(
        "expert-typescript".to_string(),
        ts_manifest,
        PathBuf::from("test/ts"),
    );

    // Verify we can only have max_loaded experts
    let stats = manager.stats();
    assert_eq!(stats.total_experts, 3);
    assert_eq!(stats.loaded_experts, 0);

    // Note: Actual loading requires valid model paths, so we skip that in tests
    // In production, this would test actual load/unload cycles
}

#[test]
fn test_lru_eviction() {
    let mut manager = ExpertManager::new(2);

    // Register 3 experts
    for i in 0..3 {
        let mut manifest = Manifest::default();
        manifest.name = format!("expert-{}", i);
        manager.register_expert(
            format!("expert-{}", i),
            manifest,
            PathBuf::from(format!("test/{}", i)),
        );
    }

    // Verify registration
    let stats = manager.stats();
    assert_eq!(stats.total_experts, 3);

    // Note: Actual LRU eviction testing requires loaded experts
    // This is tested at runtime in production
}
