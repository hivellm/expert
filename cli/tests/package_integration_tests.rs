use std::fs;
use std::path::PathBuf;

/// Integration tests for the package command with multi-model support

#[test]
fn test_package_v1_manifest_structure() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v1.json");
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v1.0 manifest");
    
    // Verify v1.0 structure
    assert!(content.contains("\"base_model\""));
    assert!(content.contains("\"adapters\""));
    assert!(!content.contains("\"base_models\""));
    
    // Verify single weight path
    let weight_count = content.matches("weights/").count();
    assert_eq!(weight_count, 1, "v1.0 should have single weight path");
}

#[test]
fn test_package_v2_manifest_structure() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v2.0 manifest");
    
    // Verify v2.0 structure
    assert!(content.contains("\"base_models\""));
    assert!(!content.contains("\"base_model\":"));
    
    // Verify multiple weight paths
    let weight_count = content.matches("weights/").count();
    assert_eq!(weight_count, 2, "v2.0 should have multiple weight paths");
    
    // Verify model-specific paths
    assert!(content.contains("weights/qwen3-0.6b/"));
    assert!(content.contains("weights/qwen3-1.5b/"));
}

#[test]
fn test_package_v2_model_names_are_distinct() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v2.0 manifest");
    
    // Should have two distinct model names
    assert!(content.contains("\"Qwen3-0.6B\""));
    assert!(content.contains("\"Qwen3-1.5B\""));
    
    // Count occurrences (appears twice: once in name, once in description potentially)
    let qwen_06b_count = content.matches("Qwen3-0.6B").count();
    let qwen_15b_count = content.matches("Qwen3-1.5B").count();
    
    assert!(qwen_06b_count >= 1, "Qwen3-0.6B should appear at least once");
    assert!(qwen_15b_count >= 1, "Qwen3-1.5B should appear at least once");
}

#[test]
fn test_package_v2_adapters_embedded_in_models() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v2.0 manifest");
    
    let lines: Vec<&str> = content.lines().collect();
    
    // Find base_models line
    let base_models_line = lines.iter()
        .position(|l| l.contains("\"base_models\""))
        .expect("base_models not found");
    
    // Find an adapters line
    let adapters_line = lines.iter()
        .position(|l| l.contains("\"adapters\""))
        .expect("adapters not found");
    
    // adapters should appear AFTER base_models (embedded)
    assert!(adapters_line > base_models_line, 
        "adapters should be embedded in base_models, not at root");
}

#[test]
fn test_package_v1_and_v2_have_compatible_fields() {
    let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json")
        .expect("Failed to read v1.0 manifest");
    let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json")
        .expect("Failed to read v2.0 manifest");
    
    // Both should have these common fields
    let common_fields = vec![
        "\"name\"",
        "\"version\"",
        "\"description\"",
        "\"capabilities\"",
        "\"constraints\"",
        "\"training\"",
    ];
    
    for field in common_fields {
        assert!(v1_content.contains(field), "v1.0 missing {}", field);
        assert!(v2_content.contains(field), "v2.0 missing {}", field);
    }
}

#[test]
fn test_package_v2_weight_paths_include_model_identifier() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v2.0 manifest");
    
    // Weight paths should include model identifier
    assert!(content.contains("qwen3-0.6b"), "Path should include model identifier");
    assert!(content.contains("qwen3-1.5b"), "Path should include model identifier");
    
    // Should NOT have generic "adapter.safetensors" path
    let generic_path = content.contains("\"path\": \"weights/adapter.safetensors\"");
    assert!(!generic_path, "v2.0 should not have generic weight paths");
}

#[test]
fn test_package_adapter_count_consistency() {
    let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json")
        .expect("Failed to read v1.0 manifest");
    let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json")
        .expect("Failed to read v2.0 manifest");
    
    // v1.0: Should have exactly 1 adapter at root level
    let v1_adapter_arrays = v1_content.matches("\"adapters\"").count();
    assert_eq!(v1_adapter_arrays, 1, "v1.0 should have 1 adapters array");
    
    // v2.0: Should have adapters arrays per model (2 models = 2 adapter arrays)
    let v2_adapter_arrays = v2_content.matches("\"adapters\"").count();
    assert_eq!(v2_adapter_arrays, 2, "v2.0 should have adapters arrays per model");
}

#[test]
fn test_json_formatting_validity() {
    // Both fixtures should be valid JSON
    let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json")
        .expect("Failed to read v1.0 manifest");
    let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json")
        .expect("Failed to read v2.0 manifest");
    
    // Parse as generic JSON first
    let v1_json: serde_json::Value = serde_json::from_str(&v1_content)
        .expect("v1.0 manifest is not valid JSON");
    let v2_json: serde_json::Value = serde_json::from_str(&v2_content)
        .expect("v2.0 manifest is not valid JSON");
    
    // Verify basic structure
    assert!(v1_json.is_object());
    assert!(v2_json.is_object());
    
    // Verify name field exists
    assert!(v1_json.get("name").is_some());
    assert!(v2_json.get("name").is_some());
}

#[test]
fn test_schema_version_field_presence() {
    let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json")
        .expect("Failed to read v1.0 manifest");
    let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json")
        .expect("Failed to read v2.0 manifest");
    
    let v1_json: serde_json::Value = serde_json::from_str(&v1_content).unwrap();
    let v2_json: serde_json::Value = serde_json::from_str(&v2_content).unwrap();
    
    // Both should explicitly declare schema_version
    assert_eq!(v1_json.get("schema_version").and_then(|v| v.as_str()), Some("1.0"));
    assert_eq!(v2_json.get("schema_version").and_then(|v| v.as_str()), Some("2.0"));
}

#[test]
fn test_training_config_presence() {
    let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").unwrap();
    let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();
    
    // Both should have training.config
    assert!(v1_content.contains("\"training\""));
    assert!(v1_content.contains("\"config\""));
    assert!(v2_content.contains("\"training\""));
    assert!(v2_content.contains("\"config\""));
    
    // Both should have same training config structure
    let required_training_fields = vec![
        "\"method\"",
        "\"adapter_type\"",
        "\"rank\"",
        "\"epochs\"",
        "\"learning_rate\"",
    ];
    
    for field in required_training_fields {
        assert!(v1_content.contains(field), "v1.0 missing {}", field);
        assert!(v2_content.contains(field), "v2.0 missing {}", field);
    }
}

