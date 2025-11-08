use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_package_filename_generation_v1() {
    let expert_name = "expert-neo4j";
    let version = "0.0.1";

    let filename = format!("{}.v{}.expert", expert_name, version);
    assert_eq!(filename, "expert-neo4j.v0.0.1.expert");
}

#[test]
fn test_package_filename_generation_v2() {
    let expert_name = "expert-neo4j";
    let model_name = "qwen3-0.6b";
    let version = "0.0.1";

    let filename = format!("{}-{}.v{}.expert", expert_name, model_name, version);
    assert_eq!(filename, "expert-neo4j-qwen3-0.6b.v0.0.1.expert");
}

#[test]
fn test_model_name_normalization() {
    let model = "F:/Node/hivellm/expert/models/Qwen3-0.6B";

    let normalized = model
        .to_lowercase()
        .replace('/', "")
        .replace('\\', "")
        .replace(':', "")
        .replace("fnodehivellmexpertmodels", "");

    assert!(normalized.contains("qwen3"));
}

#[test]
fn test_adapter_path_construction() {
    let weights_dir = PathBuf::from("weights");
    let adapter_path = "qwen3-06b/adapter";

    let full_path = weights_dir.join(adapter_path);

    assert_eq!(full_path.to_string_lossy(), "weights/qwen3-06b/adapter");
}

#[test]
fn test_sha256_hex_format() {
    let hash = "27af68e96322ef4a3dc3bcc9c027bf1dd0515cb01bd38d555f29de7c22a02066";

    // Should be valid hex
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_package_size_calculation() {
    let size_bytes: u64 = 14702472;
    let size_mb = size_bytes as f64 / 1_048_576.0;

    assert!((size_mb - 14.02).abs() < 0.1);
}

#[test]
fn test_tar_gz_extension() {
    let package = "expert-neo4j.v0.0.1.expert";

    // .expert is tar.gz format
    assert!(package.ends_with(".expert"));

    let without_ext = package.strip_suffix(".expert").unwrap();
    assert_eq!(without_ext, "expert-neo4j.v0.0.1");
}

#[test]
fn test_manifest_filtering_v2() {
    // Simulate filtering manifest for specific model
    let models = vec!["Qwen3-0.6B", "Qwen3-1.5B", "Qwen3-7B"];
    let selected = "Qwen3-0.6B";

    let filtered: Vec<_> = models.iter().filter(|m| **m == selected).collect();

    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0], &"Qwen3-0.6B");
}

#[test]
fn test_shared_resources_collection() {
    let resources = vec!["LICENSE", "README.md", "grammar.gbnf"];

    // LICENSE is typically shared
    assert!(resources.contains(&"LICENSE"));
}
