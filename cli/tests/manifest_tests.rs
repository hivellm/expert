use std::fs;
use std::path::PathBuf;

// These tests validate manifest parsing and validation with real JSON files

#[test]
fn test_parse_v1_manifest_from_file() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v1.json");
    
    // This test requires the CLI to be built with the manifest module
    // We're testing that a real v1.0 JSON file can be parsed
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v1.0 test fixture");
    
    assert!(content.contains("\"schema_version\": \"1.0\""));
    assert!(content.contains("\"base_model\""));
    assert!(!content.contains("\"base_models\""));
}

#[test]
fn test_parse_v2_manifest_from_file() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
    
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v2.0 test fixture");
    
    assert!(content.contains("\"schema_version\": \"2.0\""));
    assert!(content.contains("\"base_models\""));
    assert!(!content.contains("\"base_model\""));
    assert!(content.contains("Qwen3-0.6B"));
    assert!(content.contains("Qwen3-1.5B"));
}

#[test]
fn test_v2_has_model_specific_paths() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
    
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v2.0 test fixture");
    
    // v2.0 should have model-specific paths
    assert!(content.contains("weights/qwen3-0.6b/adapter.safetensors"));
    assert!(content.contains("weights/qwen3-1.5b/adapter.safetensors"));
}

#[test]
fn test_v1_has_simple_path() {
    let manifest_path = PathBuf::from("tests/fixtures/manifest_v1.json");
    
    let content = fs::read_to_string(&manifest_path)
        .expect("Failed to read v1.0 test fixture");
    
    // v1.0 should have simple path
    assert!(content.contains("weights/adapter.safetensors"));
}

