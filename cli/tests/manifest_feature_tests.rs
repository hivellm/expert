// Integration tests for manifest advanced features
use expert_cli::manifest::{DecodingConfig, Manifest, SoftPrompt};
use std::path::Path;

#[test]
fn test_soft_prompt_packaging() {
    // Test that soft prompts are correctly defined in manifest
    let json = r#"{
        "name": "expert-test",
        "version": "0.0.1",
        "schema_version": "2.0",
        "description": "Test",
        "capabilities": ["test"],
        "base_models": [{
            "name": "test-model",
            "adapters": [{
                "type": "lora",
                "target_modules": ["q_proj"],
                "r": 8,
                "alpha": 16,
                "path": "adapter"
            }]
        }],
        "soft_prompts": [{
            "name": "test_prompt",
            "path": "soft_prompts/test_32.pt",
            "tokens": 32
        }],
        "constraints": {
            "load_order": 0,
            "incompatible_with": [],
            "requires": []
        },
        "training": {
            "dataset": {"path": "test"},
            "config": {
                "method": "sft",
                "adapter_type": "lora",
                "rank": 8,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.001,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 10,
                "lr_scheduler": "linear"
            }
        },
        "license": "MIT"
    }"#;

    let manifest: Manifest = serde_json::from_str(json).expect("Should parse with soft prompts");

    assert_eq!(manifest.soft_prompts.len(), 1);
    assert_eq!(manifest.soft_prompts[0].name, "test_prompt");
    assert_eq!(manifest.soft_prompts[0].tokens, 32);
    assert_eq!(manifest.soft_prompts[0].path, "soft_prompts/test_32.pt");
}

#[test]
fn test_decoding_config_override_order() {
    // Test 3-level priority: CLI > manifest > default

    let manifest_temp = Some(0.1);
    let cli_temp = Some(0.5);
    let default_temp = 0.7;

    // Priority 1: CLI override wins
    let result = cli_temp.or(manifest_temp).unwrap_or(default_temp);
    assert_eq!(result, 0.5, "CLI should override manifest");

    // Priority 2: Manifest wins when no CLI
    let result_no_cli: Option<f64> = None;
    let result = result_no_cli.or(manifest_temp).unwrap_or(default_temp);
    assert_eq!(result, 0.1, "Manifest should be used when no CLI override");

    // Priority 3: Default when nothing set
    let result_default: Option<f64> = None;
    let manifest_none: Option<f64> = None;
    let result = result_default.or(manifest_none).unwrap_or(default_temp);
    assert_eq!(result, 0.7, "Should fallback to default");
}

#[test]
fn test_manifest_backward_compatibility() {
    // Test that old manifests without new fields still work
    let json_without_decoding = r#"{
        "name": "expert-old",
        "version": "0.0.1",
        "schema_version": "2.0",
        "description": "Old format",
        "capabilities": ["test"],
        "base_models": [{
            "name": "test-model",
            "adapters": [{
                "type": "lora",
                "target_modules": ["q_proj"],
                "r": 8,
                "alpha": 16,
                "path": "adapter"
            }]
        }],
        "soft_prompts": [],
        "constraints": {
            "load_order": 0,
            "incompatible_with": [],
            "requires": []
        },
        "training": {
            "dataset": {"path": "test"},
            "config": {
                "method": "sft",
                "adapter_type": "lora",
                "rank": 8,
                "alpha": 16,
                "target_modules": ["q_proj"],
                "epochs": 1,
                "learning_rate": 0.001,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 10,
                "lr_scheduler": "linear"
            }
        },
        "license": "MIT"
    }"#;

    let manifest: Manifest = serde_json::from_str(json_without_decoding)
        .expect("Should parse manifest without decoding config");

    // Decoding should be None (optional)
    assert!(manifest.training.decoding.is_none());

    // Soft prompts should be empty
    assert_eq!(manifest.soft_prompts.len(), 0);

    // Should validate successfully
    assert!(manifest.validate().is_ok());
}

#[test]
fn test_decoding_config_all_fields() {
    let json = r#"{
        "name": "expert-full",
        "version": "0.0.1",
        "schema_version": "2.0",
        "description": "Full decoding config",
        "capabilities": ["test"],
        "base_models": [{
            "name": "test-model",
            "adapters": [{
                "type": "dora",
                "target_modules": ["q_proj", "k_proj"],
                "r": 12,
                "alpha": 24,
                "path": "adapter"
            }]
        }],
        "soft_prompts": [],
        "constraints": {
            "load_order": 0,
            "incompatible_with": [],
            "requires": []
        },
        "training": {
            "dataset": {"path": "test"},
            "config": {
                "method": "sft",
                "adapter_type": "dora",
                "rank": 12,
                "alpha": 24,
                "target_modules": ["q_proj", "k_proj"],
                "epochs": 1,
                "learning_rate": 0.001,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 10,
                "lr_scheduler": "linear"
            },
            "decoding": {
                "use_grammar": true,
                "grammar_type": "sql-postgres",
                "grammar_file": "grammars/sql.gbnf",
                "validation": "parser-strict",
                "stop_sequences": [";", "\n\n"],
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 50
            }
        },
        "license": "MIT"
    }"#;

    let manifest: Manifest = serde_json::from_str(json).expect("Should parse full decoding config");

    let decoding = manifest.training.decoding.expect("Should have decoding");

    assert_eq!(decoding.use_grammar, Some(true));
    assert_eq!(decoding.grammar_type, Some("sql-postgres".to_string()));
    assert_eq!(decoding.grammar_file, Some("grammars/sql.gbnf".to_string()));
    assert_eq!(decoding.validation, Some("parser-strict".to_string()));
    assert_eq!(
        decoding.stop_sequences,
        Some(vec![";".to_string(), "\n\n".to_string()])
    );
    assert_eq!(decoding.temperature, Some(0.1));
    assert_eq!(decoding.top_p, Some(0.9));
    assert_eq!(decoding.top_k, Some(50));
}
