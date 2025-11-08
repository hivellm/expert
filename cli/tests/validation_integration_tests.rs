use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_manifest_v2_structure() {
    let manifest_v2 = r#"{
        "name": "expert-test",
        "version": "0.0.1",
        "schema_version": "2.0",
        "base_models": [{
            "name": "Qwen3-0.6B",
            "prompt_template": "chatml",
            "adapters": [{
                "type": "lora",
                "path": "adapter",
                "r": 16,
                "alpha": 16
            }]
        }],
        "capabilities": ["test"],
        "training": {
            "dataset": {"path": "test"},
            "config": {"method": "sft"}
        }
    }"#;

    let parsed: serde_json::Value = serde_json::from_str(manifest_v2).unwrap();
    assert_eq!(parsed["schema_version"], "2.0");
    assert!(parsed["base_models"].is_array());
}

#[test]
fn test_adapter_path_resolution() {
    let base_path = PathBuf::from("./experts/expert-test");
    let weights_path = base_path.join("weights");
    let adapter_path = weights_path.join("qwen3-06b/adapter");

    assert_eq!(
        adapter_path.to_string_lossy(),
        "./experts/expert-test/weights/qwen3-06b/adapter"
    );
}

#[test]
fn test_size_bytes_option() {
    // Test that Option<u64> works correctly
    let size: Option<u64> = Some(14702472);

    if let Some(s) = size {
        assert!(s > 0);
        assert_eq!(s, 14702472);
    }

    let no_size: Option<u64> = None;
    assert!(no_size.is_none());
}

#[test]
fn test_sha256_option() {
    // Test that Option<String> works correctly
    let sha: Option<String> = Some("abc123".to_string());

    if let Some(ref s) = sha {
        assert!(!s.is_empty());
        assert_eq!(s, "abc123");
    }

    let no_sha: Option<String> = None;
    assert!(no_sha.is_none());
}

#[test]
fn test_capabilities_vec() {
    let caps = vec![
        "tech:neo4j".to_string(),
        "database:neo4j".to_string(),
        "query:cypher".to_string(),
    ];

    assert_eq!(caps.len(), 3);
    assert!(caps.contains(&"tech:neo4j".to_string()));
}

#[test]
fn test_model_name_normalization() {
    let model_names = vec![
        "Qwen3-0.6B",
        "F:/Node/hivellm/expert/models/Qwen3-0.6B",
        "qwen3-0.6b",
    ];

    for name in model_names {
        let normalized = name.to_lowercase().replace('/', "-").replace('\\', "-");
        assert!(normalized.contains("qwen3") || normalized.contains("qwen3-0.6b"));
    }
}

#[test]
fn test_package_naming_convention() {
    let expert_name = "expert-neo4j";
    let model_name = "qwen3-0.6b";
    let version = "0.0.1";

    let package_name = format!("{}-{}.v{}.expert", expert_name, model_name, version);

    assert_eq!(package_name, "expert-neo4j-qwen3-0.6b.v0.0.1.expert");
    assert!(package_name.ends_with(".expert"));
}
