// Tests for qwen.rs - model loading and generation
// Note: These tests require actual model files or are mocked

use expert_cli::inference::qwen::{Qwen3Config, QwenEngine};

#[test]
fn test_qwen3_config_creation() {
    let config = Qwen3Config {
        hidden_size: 896,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        intermediate_size: 4864,
        vocab_size: 151936,
        max_position_embeddings: 32768,
        rope_theta: 1000000.0,
    };

    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.vocab_size, 151936);
}

#[test]
fn test_qwen3_config_defaults() {
    // Test that config parsing handles missing fields gracefully
    let config_json = r#"{
        "hidden_size": 512,
        "num_hidden_layers": 12
    }"#;

    let config_value: serde_json::Value = serde_json::from_str(config_json).unwrap();

    let config = Qwen3Config {
        hidden_size: config_value["hidden_size"].as_u64().unwrap_or(896) as usize,
        num_hidden_layers: config_value["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        num_attention_heads: config_value["num_attention_heads"].as_u64().unwrap_or(14) as usize,
        num_key_value_heads: config_value["num_key_value_heads"].as_u64().unwrap_or(2) as usize,
        intermediate_size: config_value["intermediate_size"].as_u64().unwrap_or(4864) as usize,
        vocab_size: config_value["vocab_size"].as_u64().unwrap_or(151936) as usize,
        max_position_embeddings: config_value["max_position_embeddings"]
            .as_u64()
            .unwrap_or(32768) as usize,
        rope_theta: config_value["rope_theta"].as_f64().unwrap_or(1000000.0) as f32,
    };

    assert_eq!(config.hidden_size, 512);
    assert_eq!(config.num_hidden_layers, 12);
    assert_eq!(config.num_attention_heads, 14); // Default
    assert_eq!(config.vocab_size, 151936); // Default
}

#[test]
fn test_config_parsing_types() {
    // Test parsing different numeric types
    let config_json = r#"{
        "hidden_size": 512,
        "num_hidden_layers": "24",
        "vocab_size": 151936.0
    }"#;

    let config_value: serde_json::Value = serde_json::from_str(config_json).unwrap();

    // Test that as_u64() handles integers correctly
    let hidden_size = config_value["hidden_size"].as_u64().unwrap_or(0) as usize;
    assert_eq!(hidden_size, 512);

    // String should fail and use default
    let layers = config_value["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
    assert_eq!(layers, 24); // Uses default because string conversion fails

    // Float should work
    let vocab = config_value["vocab_size"].as_u64().unwrap_or(0) as usize;
    assert_eq!(vocab, 151936);
}

#[test]
fn test_rope_theta_parsing() {
    // Test rope_theta parsing (can be large float)
    let config_json = r#"{
        "rope_theta": 1000000.0
    }"#;

    let config_value: serde_json::Value = serde_json::from_str(config_json).unwrap();
    let rope_theta = config_value["rope_theta"].as_f64().unwrap_or(1000000.0) as f32;

    assert!((rope_theta - 1000000.0).abs() < 0.1);
}

#[test]
fn test_config_all_fields() {
    // Test config with all fields present
    let config_json = r#"{
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0
    }"#;

    let config_value: serde_json::Value = serde_json::from_str(config_json).unwrap();

    let config = Qwen3Config {
        hidden_size: config_value["hidden_size"].as_u64().unwrap_or(896) as usize,
        num_hidden_layers: config_value["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        num_attention_heads: config_value["num_attention_heads"].as_u64().unwrap_or(14) as usize,
        num_key_value_heads: config_value["num_key_value_heads"].as_u64().unwrap_or(2) as usize,
        intermediate_size: config_value["intermediate_size"].as_u64().unwrap_or(4864) as usize,
        vocab_size: config_value["vocab_size"].as_u64().unwrap_or(151936) as usize,
        max_position_embeddings: config_value["max_position_embeddings"]
            .as_u64()
            .unwrap_or(32768) as usize,
        rope_theta: config_value["rope_theta"].as_f64().unwrap_or(1000000.0) as f32,
    };

    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.intermediate_size, 4864);
    assert_eq!(config.vocab_size, 151936);
    assert_eq!(config.max_position_embeddings, 32768);
    assert!((config.rope_theta - 1000000.0).abs() < 0.1);
}

// Note: Tests for actual model loading (from_local, from_pretrained, generate)
// would require:
// 1. Actual model files (large, ~GB)
// 2. Mock SafeTensors files
// 3. Mock tokenizer
// These are integration tests that should be run separately with real models
//
// For unit tests, we focus on config parsing and data structures

#[test]
fn test_config_clone() {
    let config = Qwen3Config {
        hidden_size: 896,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        intermediate_size: 4864,
        vocab_size: 151936,
        max_position_embeddings: 32768,
        rope_theta: 1000000.0,
    };

    let cloned = config.clone();
    assert_eq!(cloned.hidden_size, config.hidden_size);
    assert_eq!(cloned.num_hidden_layers, config.num_hidden_layers);
}

#[test]
fn test_config_debug() {
    // Test that config can be formatted for debugging
    let config = Qwen3Config {
        hidden_size: 896,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        intermediate_size: 4864,
        vocab_size: 151936,
        max_position_embeddings: 32768,
        rope_theta: 1000000.0,
    };

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("896"));
    assert!(debug_str.contains("24"));
}

#[test]
fn test_vocab_size_variants() {
    // Test different vocab sizes
    let configs = vec![
        Qwen3Config {
            vocab_size: 32000,
            ..create_dummy_config()
        },
        Qwen3Config {
            vocab_size: 151936,
            ..create_dummy_config()
        },
        Qwen3Config {
            vocab_size: 200000,
            ..create_dummy_config()
        },
    ];

    for config in configs {
        assert!(config.vocab_size > 0);
    }
}

#[test]
fn test_layer_count_variants() {
    // Test different layer counts
    let configs = vec![
        Qwen3Config {
            num_hidden_layers: 12,
            ..create_dummy_config()
        },
        Qwen3Config {
            num_hidden_layers: 24,
            ..create_dummy_config()
        },
        Qwen3Config {
            num_hidden_layers: 48,
            ..create_dummy_config()
        },
    ];

    for config in configs {
        assert!(config.num_hidden_layers > 0);
    }
}

fn create_dummy_config() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 896,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        intermediate_size: 4864,
        vocab_size: 151936,
        max_position_embeddings: 32768,
        rope_theta: 1000000.0,
    }
}
