use std::path::PathBuf;

#[test]
fn test_jsonl_extension_detection() {
    let files = vec![
        ("dataset.jsonl", true),
        ("dataset.json", true),
        ("dataset.txt", false),
        ("dataset.csv", false),
    ];

    for (filename, is_valid) in files {
        let path = PathBuf::from(filename);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let valid = ext == "jsonl" || ext == "json";

        assert_eq!(valid, is_valid);
    }
}

#[test]
fn test_json_line_parsing() {
    let lines = vec![
        r#"{"instruction": "test", "response": "answer"}"#,
        r#"{"question": "test", "answer": "answer"}"#,
        r#"invalid json"#,
    ];

    let mut valid = 0;
    let mut invalid = 0;

    for line in lines {
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(_) => valid += 1,
            Err(_) => invalid += 1,
        }
    }

    assert_eq!(valid, 2);
    assert_eq!(invalid, 1);
}

#[test]
fn test_field_detection_instruction() {
    let json = r#"{"instruction": "Do something", "response": "Done"}"#;
    let parsed: serde_json::Value = serde_json::from_str(json).unwrap();

    assert!(parsed.get("instruction").is_some());
    assert!(parsed.get("response").is_some());
}

#[test]
fn test_field_detection_question() {
    let json = r#"{"question": "What is X?", "answer": "X is Y"}"#;
    let parsed: serde_json::Value = serde_json::from_str(json).unwrap();

    assert!(parsed.get("question").is_some());
    assert!(parsed.get("answer").is_some());
}

#[test]
fn test_empty_line_handling() {
    let lines = vec!["", "  ", "\n", r#"{"valid": true}"#];

    let non_empty: Vec<_> = lines.iter().filter(|l| !l.trim().is_empty()).collect();

    assert_eq!(non_empty.len(), 1);
}

#[test]
fn test_dataset_statistics() {
    let total = 1000;
    let valid = 950;
    let invalid = 50;

    assert_eq!(total, valid + invalid);

    let success_rate = (valid as f64 / total as f64) * 100.0;
    assert!((success_rate - 95.0).abs() < 0.1);
}

#[test]
fn test_error_limit() {
    let max_errors_to_display = 5;
    let total_errors = 20;

    assert!(max_errors_to_display < total_errors);
}
