use expert_cli::registry::{ExpertRegistry, ExpertEntry, BaseModelEntry, AdapterEntry};
use std::path::PathBuf;
use chrono::Utc;
use tempfile::TempDir;

fn create_test_registry() -> ExpertRegistry {
    let temp_dir = TempDir::new().unwrap();
    let mut registry = ExpertRegistry::new(
        temp_dir.path().join("experts"),
        temp_dir.path().join("models"),
    );
    
    // Add test expert 1
    registry.add_expert(ExpertEntry {
        name: "expert-neo4j".to_string(),
        version: "0.0.1".to_string(),
        base_model: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("experts/expert-neo4j"),
        source: "file://./neo4j".to_string(),
        installed_at: Utc::now(),
        adapters: vec![AdapterEntry {
            adapter_type: "lora".to_string(),
            path: PathBuf::from("adapter"),
            size_bytes: 1000000,
            sha256: Some("hash1".to_string()),
        }],
        capabilities: vec!["tech:neo4j".to_string(), "query:cypher".to_string()],
        dependencies: vec![],
    });
    
    // Add test expert 2
    registry.add_expert(ExpertEntry {
        name: "expert-sql".to_string(),
        version: "0.0.1".to_string(),
        base_model: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("experts/expert-sql"),
        source: "file://./sql".to_string(),
        installed_at: Utc::now(),
        adapters: vec![],
        capabilities: vec!["database:sql".to_string()],
        dependencies: vec![],
    });
    
    // Add test model
    registry.add_base_model(BaseModelEntry {
        name: "Qwen3-0.6B".to_string(),
        path: temp_dir.path().join("models/Qwen3-0.6B"),
        sha256: Some("model_hash".to_string()),
        quantization: Some("int4".to_string()),
        size_bytes: 500000000,
        installed_at: Utc::now(),
        source: "huggingface".to_string(),
    });
    
    registry
}

#[test]
fn test_list_all_experts() {
    let registry = create_test_registry();
    let experts = registry.list_experts();
    
    assert_eq!(experts.len(), 2);
    assert!(experts.iter().any(|e| e.name == "expert-neo4j"));
    assert!(experts.iter().any(|e| e.name == "expert-sql"));
}

#[test]
fn test_list_all_models() {
    let registry = create_test_registry();
    let models = registry.list_base_models();
    
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "Qwen3-0.6B");
    assert_eq!(models[0].quantization.as_deref(), Some("int4"));
}

#[test]
fn test_filter_experts_by_base_model() {
    let registry = create_test_registry();
    let experts = registry.list_experts();
    
    let filtered: Vec<_> = experts.iter()
        .filter(|e| e.base_model == "Qwen3-0.6B")
        .collect();
    
    assert_eq!(filtered.len(), 2);
}

#[test]
fn test_expert_has_adapters() {
    let registry = create_test_registry();
    let neo4j = registry.get_expert("expert-neo4j").unwrap();
    
    assert_eq!(neo4j.adapters.len(), 1);
    assert_eq!(neo4j.adapters[0].adapter_type, "lora");
    assert_eq!(neo4j.adapters[0].size_bytes, 1000000);
}

#[test]
fn test_expert_has_capabilities() {
    let registry = create_test_registry();
    let neo4j = registry.get_expert("expert-neo4j").unwrap();
    
    assert_eq!(neo4j.capabilities.len(), 2);
    assert!(neo4j.capabilities.contains(&"tech:neo4j".to_string()));
    assert!(neo4j.capabilities.contains(&"query:cypher".to_string()));
}

#[test]
fn test_registry_serialization() {
    let registry = create_test_registry();
    
    // Serialize to JSON
    let json = serde_json::to_string_pretty(&registry).unwrap();
    
    // Check contains expected fields
    assert!(json.contains("\"version\""));
    assert!(json.contains("\"experts\""));
    assert!(json.contains("\"base_models\""));
    assert!(json.contains("expert-neo4j"));
    assert!(json.contains("expert-sql"));
}

#[test]
fn test_registry_deserialization() {
    let json = r#"{
        "version": "1.0",
        "last_updated": "2025-11-03T12:00:00Z",
        "install_dir": "/tmp/experts",
        "models_dir": "/tmp/models",
        "base_models": [],
        "experts": []
    }"#;
    
    let registry: ExpertRegistry = serde_json::from_str(json).unwrap();
    
    assert_eq!(registry.version, "1.0");
    assert_eq!(registry.experts.len(), 0);
    assert_eq!(registry.base_models.len(), 0);
}

