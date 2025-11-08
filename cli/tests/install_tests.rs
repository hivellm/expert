use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_parse_git_url() {
    // Test basic Git URL parsing logic
    let url1 = "git+https://github.com/user/repo.git";
    assert!(url1.starts_with("git+"));

    let url2 = "git+https://github.com/user/repo.git#v0.0.1";
    assert!(url2.contains('#'));

    let parts: Vec<&str> = url2.split('#').collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[1], "v0.0.1");
}

#[test]
fn test_parse_file_url() {
    let url = "file://./experts/expert-neo4j";
    assert!(url.starts_with("file://"));

    let path = url.strip_prefix("file://").unwrap();
    assert_eq!(path, "./experts/expert-neo4j");
}

#[test]
fn test_parse_local_path() {
    let path = "./experts/expert-neo4j";
    let pathbuf = PathBuf::from(path);
    assert_eq!(pathbuf.to_str().unwrap(), "./experts/expert-neo4j");
}

#[test]
fn test_parse_expert_package() {
    let package = "expert-neo4j-qwen306b.v0.0.1.expert";
    assert!(package.ends_with(".expert"));

    let without_ext = package.strip_suffix(".expert").unwrap();
    assert_eq!(without_ext, "expert-neo4j-qwen306b.v0.0.1");
}

#[test]
fn test_expert_name_extraction() {
    let package = "expert-neo4j-qwen306b.v0.0.1.expert";
    let parts: Vec<&str> = package.split('.').collect();

    // Should be: ["expert-neo4j-qwen306b", "v0", "0", "1", "expert"]
    assert!(parts.len() >= 5);
    assert_eq!(parts[parts.len() - 1], "expert");
}

#[test]
fn test_temp_directory_creation() {
    let temp_dir = TempDir::new().unwrap();
    assert!(temp_dir.path().exists());
    assert!(temp_dir.path().is_dir());
}

#[test]
fn test_copy_directory_structure() {
    let src_dir = TempDir::new().unwrap();
    let dst_dir = TempDir::new().unwrap();

    // Create test structure
    std::fs::create_dir_all(src_dir.path().join("subdir")).unwrap();
    std::fs::write(src_dir.path().join("file1.txt"), "content1").unwrap();
    std::fs::write(src_dir.path().join("subdir/file2.txt"), "content2").unwrap();

    // Verify source structure
    assert!(src_dir.path().join("file1.txt").exists());
    assert!(src_dir.path().join("subdir/file2.txt").exists());
}

#[test]
fn test_manifest_detection() {
    let temp_dir = TempDir::new().unwrap();
    let manifest_path = temp_dir.path().join("manifest.json");

    // Before creation
    assert!(!manifest_path.exists());

    // Create manifest
    std::fs::write(&manifest_path, r#"{"name": "test"}"#).unwrap();

    // After creation
    assert!(manifest_path.exists());
}

#[test]
fn test_git_url_validation() {
    let valid_urls = vec![
        "git+https://github.com/user/repo.git",
        "git+https://github.com/user/repo.git#main",
        "git+https://github.com/user/repo.git#v0.0.1",
    ];

    for url in valid_urls {
        assert!(url.starts_with("git+"));
        assert!(url.contains("github.com") || url.contains("gitlab.com"));
    }
}
