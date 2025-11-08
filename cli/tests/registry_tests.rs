use chrono::Utc;
use expert_cli::registry::{AdapterEntry, BaseModelEntry, ExpertEntry, ExpertRegistry};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_create_empty_registry() {
    let temp_dir = TempDir::new().unwrap();
    let registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    assert_eq!(registry.version, "1.0");
    assert_eq!(registry.experts.len(), 0);
    assert_eq!(registry.base_models.len(), 0);
}

#[test]
fn test_add_remove_expert() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    let expert = ExpertEntry {
        name: "expert-test".to_string(),
        version: "0.0.1".to_string(),
        base_model: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("experts/expert-test"),
        source: "file://./test".to_string(),
        installed_at: Utc::now(),
        adapters: vec![],
        capabilities: vec!["test:feature".to_string()],
        dependencies: vec![],
    };

    // Add expert
    registry.add_expert(expert.clone());
    assert_eq!(registry.experts.len(), 1);
    assert!(registry.has_expert("expert-test"));

    // Get expert
    let retrieved = registry.get_expert("expert-test");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().version, "0.0.1");

    // Remove expert
    let removed = registry.remove_expert("expert-test");
    assert!(removed.is_some());
    assert_eq!(registry.experts.len(), 0);
    assert!(!registry.has_expert("expert-test"));
}

#[test]
fn test_add_remove_base_model() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    let model = BaseModelEntry {
        name: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("models/Qwen3-0.6B"),
        sha256: Some("abc123".to_string()),
        quantization: Some("int4".to_string()),
        size_bytes: 1000000,
        installed_at: Utc::now(),
        source: "huggingface".to_string(),
    };

    // Add model
    registry.add_base_model(model.clone());
    assert_eq!(registry.base_models.len(), 1);
    assert!(registry.has_base_model("Qwen3-0.6B"));

    // Get model
    let retrieved = registry.get_base_model("Qwen3-0.6B");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().quantization.as_deref(), Some("int4"));

    // Remove model
    let removed = registry.remove_base_model("Qwen3-0.6B");
    assert!(removed.is_some());
    assert_eq!(registry.base_models.len(), 0);
}

#[test]
fn test_save_load_registry() {
    let temp_dir = TempDir::new().unwrap();
    let registry_path = temp_dir.path().join("test-registry.json");

    // Create and populate registry
    let mut registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    let expert = ExpertEntry {
        name: "expert-test".to_string(),
        version: "0.0.1".to_string(),
        base_model: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("experts/expert-test"),
        source: "file://./test".to_string(),
        installed_at: Utc::now(),
        adapters: vec![AdapterEntry {
            adapter_type: "lora".to_string(),
            path: PathBuf::from("adapter"),
            size_bytes: 1000,
            sha256: Some("hash".to_string()),
        }],
        capabilities: vec!["test".to_string()],
        dependencies: vec![],
    };

    registry.add_expert(expert);

    // Save to custom path
    let json = serde_json::to_string_pretty(&registry).unwrap();
    std::fs::write(&registry_path, json).unwrap();

    // Load from custom path
    let content = std::fs::read_to_string(&registry_path).unwrap();
    let loaded: ExpertRegistry = serde_json::from_str(&content).unwrap();

    assert_eq!(loaded.experts.len(), 1);
    assert_eq!(loaded.experts[0].name, "expert-test");
    assert_eq!(loaded.experts[0].adapters.len(), 1);
}

#[test]
fn test_update_existing_expert() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    let expert_v1 = ExpertEntry {
        name: "expert-test".to_string(),
        version: "0.0.1".to_string(),
        base_model: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("experts/expert-test"),
        source: "file://./test".to_string(),
        installed_at: Utc::now(),
        adapters: vec![],
        capabilities: vec![],
        dependencies: vec![],
    };

    registry.add_expert(expert_v1);
    assert_eq!(registry.experts.len(), 1);
    assert_eq!(registry.experts[0].version, "0.0.1");

    // Update with newer version
    let expert_v2 = ExpertEntry {
        name: "expert-test".to_string(),
        version: "0.0.2".to_string(),
        base_model: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("experts/expert-test"),
        source: "file://./test".to_string(),
        installed_at: Utc::now(),
        adapters: vec![],
        capabilities: vec![],
        dependencies: vec![],
    };

    registry.add_expert(expert_v2);

    // Should still have only 1 expert (updated, not duplicated)
    assert_eq!(registry.experts.len(), 1);
    assert_eq!(registry.experts[0].version, "0.0.2");
}

#[test]
fn test_list_experts_empty() {
    let temp_dir = TempDir::new().unwrap();
    let registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    let experts = registry.list_experts();
    assert_eq!(experts.len(), 0);
}

#[test]
fn test_list_base_models_empty() {
    let temp_dir = TempDir::new().unwrap();
    let registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    let models = registry.list_base_models();
    assert_eq!(models.len(), 0);
}

#[test]
fn test_multiple_experts() {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );

    for i in 1..=5 {
        let expert = ExpertEntry {
            name: format!("expert-{}", i),
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join(format!("experts/expert-{}", i)),
            source: "file://./test".to_string(),
            installed_at: Utc::now(),
            adapters: vec![],
            capabilities: vec![],
            dependencies: vec![],
        };
        registry.add_expert(expert);
    }

    assert_eq!(registry.experts.len(), 5);
    assert!(registry.has_expert("expert-1"));
    assert!(registry.has_expert("expert-5"));
    assert!(!registry.has_expert("expert-6"));
}
