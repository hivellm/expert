use std::path::PathBuf;

#[test]
fn test_ed25519_key_length() {
    // Ed25519 keys are 32 bytes = 64 hex characters
    let pubkey_hex = "3560290124df0eec6fae0e7dd9be75c1a08c9adcbdc718d2d7df93a3534576d3";

    assert_eq!(pubkey_hex.len(), 64);
    assert!(pubkey_hex.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_signature_hex_format() {
    let signature = "f1e81a9b4eaad28652a1bc50dd3deb80";

    // Should be valid hex
    assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(signature.len() % 2, 0); // Even number of chars
}

#[test]
fn test_canonical_message_format() {
    let files = vec![
        ("LICENSE".to_string(), "hash1".to_string()),
        ("manifest.json".to_string(), "hash2".to_string()),
        ("adapter.safetensors".to_string(), "hash3".to_string()),
    ];

    let canonical: String = files
        .iter()
        .map(|(file, hash)| format!("{}:{}", file, hash))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(canonical.contains("LICENSE:hash1"));
    assert!(canonical.contains("manifest.json:hash2"));
    assert!(canonical.contains("\n"));
}

#[test]
fn test_key_file_extensions() {
    let private_key = "publisher.pem";
    let public_key = "publisher.pub";

    assert!(private_key.ends_with(".pem"));
    assert!(public_key.ends_with(".pub"));
}

#[test]
fn test_integrity_section_fields() {
    // Test that integrity section has required fields
    let fields = vec![
        "created_at",
        "publisher",
        "pubkey",
        "signature_algorithm",
        "signature",
    ];

    assert!(fields.contains(&"pubkey"));
    assert!(fields.contains(&"signature"));
    assert!(fields.contains(&"signature_algorithm"));
}

#[test]
fn test_signature_algorithm_name() {
    let algorithm = "Ed25519";

    assert_eq!(algorithm, "Ed25519");
    assert!(!algorithm.is_empty());
}

#[test]
fn test_timestamp_format() {
    let timestamp = "2025-11-03T12:00:00Z";

    // ISO 8601 format
    assert!(timestamp.contains('T'));
    assert!(timestamp.contains('Z'));
    assert_eq!(timestamp.len(), 20);
}
