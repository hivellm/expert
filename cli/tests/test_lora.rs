// Tests for lora.rs - adapter loading and application

use candle_core::{DType, Device, Tensor};
use expert_cli::inference::lora::{AdapterType, LoraAdapter};
use std::collections::HashMap;
use std::path::Path;
use tempfile::TempDir;

// Helper function to assert tensors are close
fn assert_tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) {
    let a_vec = a.to_vec1::<f32>().unwrap();
    let b_vec = b.to_vec1::<f32>().unwrap();

    assert_eq!(a_vec.len(), b_vec.len(), "Tensor dimensions don't match");

    for (i, (&a_val, &b_val)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        let diff = (a_val - b_val).abs();
        assert!(
            diff <= tolerance,
            "Tensors differ at index {}: {} vs {} (diff: {})",
            i, a_val, b_val, diff
        );
    }
}

#[test]
fn test_adapter_type_enum() {
    // Test that AdapterType variants exist
    let lora = AdapterType::LoRA;
    let dora = AdapterType::DoRA;
    let ia3 = AdapterType::IA3;

    // Just verify they compile and can be compared
    match lora {
        AdapterType::LoRA => assert!(true),
        _ => assert!(false),
    }

    match dora {
        AdapterType::DoRA => assert!(true),
        _ => assert!(false),
    }

    match ia3 {
        AdapterType::IA3 => assert!(true),
        _ => assert!(false),
    }
}

#[test]
fn test_lora_adapter_creation() {
    // Create a mock adapter with dummy weights
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    // Create dummy LoRA weights
    let lora_a = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &device).unwrap();
    let lora_b = Tensor::new(&[[5.0, 6.0]], &device).unwrap();

    weights.insert("layer.lora_A.weight".to_string(), lora_a);
    weights.insert("layer.lora_B.weight".to_string(), lora_b);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: Some(2),
        alpha: Some(16),
        target_modules: vec!["layer".to_string()],
        weights,
    };

    assert_eq!(adapter.rank, Some(2));
    assert_eq!(adapter.alpha, Some(16));
    assert_eq!(adapter.target_modules.len(), 1);
    assert_eq!(adapter.weights.len(), 2);
}

#[test]
fn test_size_bytes() {
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    // Create tensors with known sizes
    let t1 = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
    let t2 = Tensor::new(&[[5.0f32, 6.0, 7.0]], &device).unwrap();

    weights.insert("t1".to_string(), t1);
    weights.insert("t2".to_string(), t2);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: None,
        alpha: None,
        target_modules: vec![],
        weights,
    };

    // t1: 4 elements, t2: 3 elements = 7 elements total
    // Each f32 is 4 bytes = 28 bytes
    let size = adapter.size_bytes();
    assert_eq!(size, 7 * std::mem::size_of::<f32>());
}

#[test]
fn test_num_parameters() {
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    let t1 = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &device).unwrap();
    let t2 = Tensor::new(&[[5.0, 6.0, 7.0]], &device).unwrap();

    weights.insert("t1".to_string(), t1);
    weights.insert("t2".to_string(), t2);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: None,
        alpha: None,
        target_modules: vec![],
        weights,
    };

    // 4 + 3 = 7 parameters
    assert_eq!(adapter.num_parameters(), 7);
}

#[test]
fn test_apply_lora_no_weights() {
    // Test applying LoRA when layer has no LoRA weights
    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: Some(2),
        alpha: Some(16),
        target_modules: vec![],
        weights: HashMap::new(),
    };

    let device = Device::Cpu;
    let base_weight = Tensor::new(&[[1.0, 2.0], [3.0, 4.0]], &device).unwrap();

    // Should return original weight unchanged
    let result = adapter.apply_lora("layer", &base_weight).unwrap();
    let base_vec = base_weight.to_vec2::<f32>().unwrap();
    let result_vec = result.to_vec2::<f32>().unwrap();

    assert_eq!(base_vec, result_vec);
}

#[test]
fn test_apply_lora_with_weights() {
    // Test applying LoRA with actual weights
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    // LoRA matrices: B (1x2) and A (2x3)
    // Final weight should be: W + (alpha/r) * BA
    let lora_a = Tensor::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).unwrap();
    let lora_b = Tensor::new(&[[7.0, 8.0]], &device).unwrap();

    weights.insert("layer.lora_A.weight".to_string(), lora_a);
    weights.insert("layer.lora_B.weight".to_string(), lora_b);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: Some(2),
        alpha: Some(16), // scale = 16/2 = 8.0
        target_modules: vec!["layer".to_string()],
        weights,
    };

    // Base weight: 1x3 (to match BA output shape)
    let base_weight = Tensor::new(&[[1.0, 2.0, 3.0]], &device).unwrap();

    // Apply LoRA
    let result = adapter.apply_lora("layer", &base_weight).unwrap();

    // Verify result shape matches base
    let base_shape = base_weight.shape();
    let result_shape = result.shape();
    assert_eq!(base_shape, result_shape);

    // Verify result is not identical to base (LoRA was applied)
    let base_vec = base_weight.to_vec2::<f32>().unwrap();
    let result_vec = result.to_vec2::<f32>().unwrap();
    assert_ne!(base_vec, result_vec);
}

#[test]
fn test_apply_lora_scaling() {
    // Test that alpha/r scaling is applied correctly
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    let lora_a = Tensor::new(&[[1.0, 1.0], [1.0, 1.0]], &device).unwrap();
    let lora_b = Tensor::new(&[[1.0, 1.0]], &device).unwrap();

    weights.insert("layer.lora_A.weight".to_string(), lora_a);
    weights.insert("layer.lora_B.weight".to_string(), lora_b);

    // alpha=16, rank=2 => scale = 8.0
    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: Some(2),
        alpha: Some(16),
        target_modules: vec![],
        weights,
    };

    let base_weight = Tensor::new(&[[0.0, 0.0]], &device).unwrap();
    let result = adapter.apply_lora("layer", &base_weight).unwrap();

    // BA = [[1,1]] * [[1,1],[1,1]] = [[2,2]]
    // Scaled: [[2,2]] * 8.0 = [[16,16]]
    // Result: [[0,0]] + [[16,16]] = [[16,16]]
    let result_vec = result.to_vec2::<f32>().unwrap();
    assert!((result_vec[0][0] - 16.0).abs() < 0.1);
}

#[test]
fn test_apply_lora_no_alpha_rank() {
    // Test with no alpha/rank (should use scale=1.0)
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    let lora_a = Tensor::new(&[[1.0, 1.0], [1.0, 1.0]], &device).unwrap();
    let lora_b = Tensor::new(&[[1.0, 1.0]], &device).unwrap();

    weights.insert("layer.lora_A.weight".to_string(), lora_a);
    weights.insert("layer.lora_B.weight".to_string(), lora_b);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: None,
        alpha: None,
        target_modules: vec![],
        weights,
    };

    let base_weight = Tensor::new(&[[0.0, 0.0]], &device).unwrap();
    let result = adapter.apply_lora("layer", &base_weight).unwrap();

    // With scale=1.0, BA should be [[2,2]]
    // Result: [[0,0]] + [[2,2]] = [[2,2]]
    let result_vec = result.to_vec2::<f32>().unwrap();
    assert!((result_vec[0][0] - 2.0).abs() < 0.1);
}

#[test]
fn test_empty_weights() {
    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: None,
        alpha: None,
        target_modules: vec![],
        weights: HashMap::new(),
    };

    assert_eq!(adapter.size_bytes(), 0);
    assert_eq!(adapter.num_parameters(), 0);
}

#[test]
fn test_multiple_target_modules() {
    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: Some(16),
        alpha: Some(32),
        target_modules: vec![
            "q_proj".to_string(),
            "v_proj".to_string(),
            "k_proj".to_string(),
        ],
        weights: HashMap::new(),
    };

    assert_eq!(adapter.target_modules.len(), 3);
    assert!(adapter.target_modules.contains(&"q_proj".to_string()));
    assert!(adapter.target_modules.contains(&"v_proj".to_string()));
    assert!(adapter.target_modules.contains(&"k_proj".to_string()));
}

#[test]
fn test_dora_adapter_type() {
    let adapter = LoraAdapter {
        adapter_type: AdapterType::DoRA,
        rank: Some(16),
        alpha: Some(32),
        target_modules: vec![],
        weights: HashMap::new(),
    };

    match adapter.adapter_type {
        AdapterType::DoRA => assert!(true),
        _ => assert!(false),
    }
}

#[test]
fn test_ia3_adapter_type() {
    let adapter = LoraAdapter {
        adapter_type: AdapterType::IA3,
        rank: None,  // IA³ doesn't use rank
        alpha: None, // IA³ doesn't use alpha
        target_modules: vec!["layer".to_string()],
        weights: HashMap::new(),
    };

    match adapter.adapter_type {
        AdapterType::IA3 => assert!(true),
        _ => assert!(false),
    }

    // IA³ should work without rank/alpha
    assert_eq!(adapter.rank, None);
    assert_eq!(adapter.alpha, None);
}

#[test]
fn test_lokr_adapter_type() {
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    // Create LoKR matrices (similar to LoRA but for Kronecker product)
    let lokr_a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
    let lokr_b_data = vec![5.0f32, 6.0]; // 2x1 matrix

    let lokr_a = Tensor::from_vec(lokr_a_data, (2, 2), &device).unwrap();
    let lokr_b = Tensor::from_vec(lokr_b_data, (2, 1), &device).unwrap();

    weights.insert("layer.lokr_A".to_string(), lokr_a);
    weights.insert("layer.lokr_B".to_string(), lokr_b);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoKR,
        rank: Some(2), // LoKR uses rank like LoRA
        alpha: Some(4),
        target_modules: vec!["layer".to_string()],
        weights,
    };

    match adapter.adapter_type {
        AdapterType::LoKR => assert!(true),
        _ => assert!(false),
    }

    // LoKR should have rank/alpha
    assert_eq!(adapter.rank, Some(2));
    assert_eq!(adapter.alpha, Some(4));
}

#[test]
fn test_apply_ia3_adapter() {
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    // Create base weight matrix [3, 4]
    let base_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let base_weight = Tensor::from_vec(base_data, (3, 4), &device).unwrap();

    // Create IA³ scaling vector [3] (matches output features)
    let ia3_data = vec![2.0f32, 0.5, 3.0]; // Scale first row by 2x, second by 0.5x, third by 3x
    let ia3_vector = Tensor::from_vec(ia3_data, (3,), &device).unwrap();

    weights.insert("layer.ia3_l".to_string(), ia3_vector);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::IA3,
        rank: None,
        alpha: None,
        target_modules: vec!["layer".to_string()],
        weights,
    };

    // Apply IA³
    let result = adapter.apply_adapter("layer", &base_weight).unwrap();

    // Check result dimensions
    assert_eq!(result.dims(), vec![3, 4]);

    // Expected: each row scaled by corresponding IA³ value
    // Row 0: [0,1,2,3] * 2.0 = [0,2,4,6]
    // Row 1: [4,5,6,7] * 0.5 = [2,2.5,3,3.5]
    // Row 2: [8,9,10,11] * 3.0 = [24,27,30,33]
    let expected_data = vec![
        0.0, 2.0, 4.0, 6.0,      // Row 0 scaled by 2.0
        2.0, 2.5, 3.0, 3.5,      // Row 1 scaled by 0.5
        24.0, 27.0, 30.0, 33.0   // Row 2 scaled by 3.0
    ];

    let expected = Tensor::from_vec(expected_data, (3, 4), &device).unwrap();
    assert_tensors_close(&result, &expected, 1e-6);
}

#[test]
fn test_soft_prompt_loading() {
    // Test loading and activating soft prompts
    let mut engine = QwenEngine {
        model: unsafe { std::mem::zeroed() }, // Mock model for testing
        tokenizer: unsafe { std::mem::zeroed() }, // Mock tokenizer
        device: Device::Cpu,
        config: Qwen3Config {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            vocab_size: 151936,
            max_position_embeddings: 131072,
            rope_theta: 10000.0,
        },
        logits_buffer: vec![0.0; 151936],
        soft_prompts: HashMap::new(),
        active_soft_prompt: None,
        base_weights: HashMap::new(),
        current_adapter: None,
    };

    // Test loading a soft prompt
    let prompt_text = "You are a helpful assistant.";
    engine.load_soft_prompt("test_prompt", prompt_text).unwrap();

    // Check that prompt was loaded
    assert!(engine.soft_prompts.contains_key("test_prompt"));
    assert!(engine.active_soft_prompt.is_none());

    // Test activating the prompt
    engine.activate_soft_prompt(Some("test_prompt"));
    assert_eq!(engine.active_soft_prompt, Some("test_prompt".to_string()));

    // Test getting active prompt tokens
    let tokens = engine.get_active_soft_prompt_tokens();
    assert!(tokens.is_some());
    assert!(!tokens.unwrap().is_empty());

    // Test deactivating
    engine.activate_soft_prompt(None);
    assert!(engine.active_soft_prompt.is_none());
    assert!(engine.get_active_soft_prompt_tokens().is_none());

    // Test getting available prompts
    let available = engine.get_soft_prompts();
    assert_eq!(available.len(), 1);
    assert_eq!(available[0], "test_prompt");
}

#[test]
fn test_large_adapter() {
    // Test with larger adapter (more realistic)
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    // Simulate larger LoRA matrices
    let large_size = 100;
    let lora_a_data: Vec<f32> = (0..large_size * 64).map(|i| i as f32).collect();
    let lora_b_data: Vec<f32> = (0..large_size).map(|i| i as f32).collect();

    let lora_a = Tensor::from_vec(lora_a_data, (64, large_size), &device).unwrap();
    let lora_b = Tensor::from_vec(lora_b_data, (1, large_size), &device).unwrap();

    weights.insert("layer.lora_A.weight".to_string(), lora_a);
    weights.insert("layer.lora_B.weight".to_string(), lora_b);

    let adapter = LoraAdapter {
        adapter_type: AdapterType::LoRA,
        rank: Some(64),
        alpha: Some(128),
        target_modules: vec![],
        weights,
    };

    // Should handle large adapters
    assert_eq!(adapter.num_parameters(), large_size * 64 + large_size);
    assert!(adapter.size_bytes() > 0);
}
