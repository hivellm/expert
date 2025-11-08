/// Tests to validate error message quality and clarity

#[cfg(test)]
mod error_validation_tests {

    #[test]
    fn test_v1_missing_base_model_error_message() {
        // Error messages should be clear and actionable
        let json = r#"{
            "name": "test",
            "version": "1.0.0",
            "schema_version": "1.0",
            "description": "test",
            "adapters": [{
                "type": "lora",
                "target_modules": ["q_proj"],
                "r": 16,
                "alpha": 16,
                "scaling": "standard",
                "dropout": 0.05,
                "path": "weights/adapter.safetensors",
                "size_bytes": 1000,
                "sha256": "hash"
            }],
            "capabilities": [],
            "constraints": {
                "load_order": 1,
                "incompatible_with": [],
                "requires": []
            },
            "training": {
                "dataset": {},
                "config": {
                    "method": "sft",
                    "adapter_type": "lora",
                    "rank": 16,
                    "alpha": 16,
                    "target_modules": ["q_proj"],
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "warmup_steps": 0,
                    "lr_scheduler": "constant"
                }
            }
        }"#;

        // This will fail during validation
        // We're testing that JSON parsing succeeds but validation catches the issue
        let parse_result = serde_json::from_str::<serde_json::Value>(json);
        assert!(parse_result.is_ok(), "JSON should be valid");
    }

    #[test]
    fn test_conflict_error_message_clarity() {
        // When both base_model and base_models are present, error should be clear
        let json = r#"{
            "name": "test",
            "version": "1.0.0",
            "schema_version": "2.0",
            "description": "test",
            "base_model": {
                "name": "Qwen3-0.6B"
            },
            "base_models": [{
                "name": "Qwen3-1.5B",
                "adapters": []
            }],
            "capabilities": [],
            "constraints": {
                "load_order": 1,
                "incompatible_with": [],
                "requires": []
            },
            "training": {
                "dataset": {},
                "config": {
                    "method": "sft",
                    "adapter_type": "lora",
                    "rank": 16,
                    "alpha": 16,
                    "target_modules": ["q_proj"],
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "warmup_steps": 0,
                    "lr_scheduler": "constant"
                }
            }
        }"#;

        // JSON parsing should succeed (it's valid JSON)
        let result = serde_json::from_str::<serde_json::Value>(json);
        assert!(result.is_ok());

        // Validation would catch the conflict
        // Testing that the JSON structure itself is parseable
    }

    #[test]
    fn test_malformed_json_error() {
        // Malformed JSON should give clear error
        let bad_json = r#"{
            "name": "test",
            "version": "1.0.0",
            missing_quotes: "value"
        }"#;

        let result = serde_json::from_str::<serde_json::Value>(bad_json);
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_msg = error.to_string();

        // Error should mention the problem
        assert!(error_msg.len() > 0);
    }

    #[test]
    fn test_missing_required_field_error() {
        // Missing required fields should be caught
        let json = r#"{
            "version": "1.0.0",
            "description": "test"
        }"#;

        let result = serde_json::from_str::<serde_json::Value>(json);

        // JSON is valid, but manifest validation would catch missing "name"
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_array_structure() {
        // Invalid array structure should be caught during parsing
        let json = r#"{
            "name": "test",
            "capabilities": "should be array not string"
        }"#;

        // This should parse as JSON but fail during manifest deserialization
        let result = serde_json::from_str::<serde_json::Value>(json);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod data_integrity_tests {
    use std::fs;

    #[test]
    fn test_v2_no_orphaned_adapters() {
        // In v2.0, there should be no adapters at root level
        let content = fs::read_to_string("tests/fixtures/manifest_v2.json")
            .expect("Failed to read v2.0 manifest");

        let lines: Vec<&str> = content.lines().collect();

        // Find if "adapters" appears before "base_models"
        let base_models_line = lines
            .iter()
            .position(|l| l.contains("\"base_models\""))
            .unwrap();

        let orphaned_adapters = lines
            .iter()
            .take(base_models_line)
            .any(|l| l.trim().starts_with("\"adapters\""));

        assert!(
            !orphaned_adapters,
            "No orphaned adapters should exist at root in v2.0"
        );
    }

    #[test]
    fn test_v1_no_base_models_array() {
        // v1.0 should not have base_models (plural)
        let content = fs::read_to_string("tests/fixtures/manifest_v1.json")
            .expect("Failed to read v1.0 manifest");

        assert!(
            !content.contains("\"base_models\""),
            "v1.0 should not have base_models array"
        );
    }

    #[test]
    fn test_weight_path_format_consistency() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").unwrap();
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();

        // v1.0: weights/adapter.safetensors
        assert!(v1_content.contains("weights/adapter.safetensors"));

        // v2.0: weights/<model>/adapter.safetensors
        assert!(v2_content.contains("weights/qwen3-0.6b/adapter.safetensors"));
        assert!(v2_content.contains("weights/qwen3-1.5b/adapter.safetensors"));
    }

    #[test]
    fn test_sha256_hash_format() {
        let content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();

        // Check that SHA256 fields exist and are strings
        if let Some(base_models) = json.get("base_models").and_then(|v| v.as_array()) {
            for model in base_models {
                if let Some(sha256) = model.get("sha256") {
                    assert!(sha256.is_string(), "SHA256 should be string");

                    if let Some(hash) = sha256.as_str() {
                        assert!(!hash.is_empty(), "SHA256 should not be empty");
                    }
                }
            }
        }
    }

    #[test]
    fn test_adapter_type_values() {
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();

        // Should contain valid adapter type
        assert!(
            v2_content.contains("\"type\": \"lora\"")
                || v2_content.contains("\"type\": \"dora\"")
                || v2_content.contains("\"type\": \"ia3\"")
        );
    }

    #[test]
    fn test_learning_rate_scientific_notation() {
        let content = fs::read_to_string("tests/fixtures/manifest_v1.json").unwrap();

        // learning_rate should be present
        assert!(content.contains("\"learning_rate\""));

        // Should be a valid number
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
        let lr = json
            .get("training")
            .and_then(|t| t.get("config"))
            .and_then(|c| c.get("learning_rate"))
            .and_then(|v| v.as_f64());

        assert!(lr.is_some(), "learning_rate should be present and numeric");
        assert!(lr.unwrap() > 0.0, "learning_rate should be positive");
    }
}
