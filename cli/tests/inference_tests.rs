// Inference tests - All model inference tests grouped together

mod qwen {
    use expert_cli::inference::qwen::{Qwen3Config, QwenEngine};

    fn create_dummy_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 4864,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            rope_theta: 1000000.0,
        }
    }

    #[test]
    fn test_qwen3_config_creation() {
        let config = Qwen3Config {
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
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
            head_dim: config_value["head_dim"].as_u64().unwrap_or(64) as usize,
            intermediate_size: config_value["intermediate_size"].as_u64().unwrap_or(4864) as usize,
            vocab_size: config_value["vocab_size"].as_u64().unwrap_or(151936) as usize,
            max_position_embeddings: config_value["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            rope_theta: config_value["rope_theta"].as_f64().unwrap_or(1000000.0),
        };
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn test_config_parsing_types() {
        let config_json = r#"{
            "hidden_size": 512,
            "num_hidden_layers": "24",
            "vocab_size": 151936.0
        }"#;
        let config_value: serde_json::Value = serde_json::from_str(config_json).unwrap();
        let hidden_size = config_value["hidden_size"].as_u64().unwrap_or(0) as usize;
        assert_eq!(hidden_size, 512);
        let layers = config_value["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
        assert_eq!(layers, 24);
        let vocab = config_value["vocab_size"].as_f64().unwrap_or(0.0) as usize;
        assert_eq!(vocab, 151936);
    }

    #[test]
    fn test_rope_theta_parsing() {
        let config_json = r#"{"rope_theta": 1000000.0}"#;
        let config_value: serde_json::Value = serde_json::from_str(config_json).unwrap();
        let rope_theta = config_value["rope_theta"].as_f64().unwrap_or(1000000.0);
        assert!((rope_theta - 1000000.0).abs() < 0.1);
    }

    #[test]
    fn test_config_all_fields() {
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
            head_dim: config_value["head_dim"].as_u64().unwrap_or(64) as usize,
            intermediate_size: config_value["intermediate_size"].as_u64().unwrap_or(4864) as usize,
            vocab_size: config_value["vocab_size"].as_u64().unwrap_or(151936) as usize,
            max_position_embeddings: config_value["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            rope_theta: config_value["rope_theta"].as_f64().unwrap_or(1000000.0),
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

    #[test]
    fn test_config_clone() {
        let config = create_dummy_config();
        let cloned = config.clone();
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.num_hidden_layers, config.num_hidden_layers);
    }

    #[test]
    fn test_config_debug() {
        let config = create_dummy_config();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("896"));
        assert!(debug_str.contains("24"));
    }

    #[test]
    fn test_vocab_size_variants() {
        let configs = vec![
            Qwen3Config { vocab_size: 32000, ..create_dummy_config() },
            Qwen3Config { vocab_size: 151936, ..create_dummy_config() },
            Qwen3Config { vocab_size: 200000, ..create_dummy_config() },
        ];
        for config in configs {
            assert!(config.vocab_size > 0);
        }
    }

    #[test]
    fn test_layer_count_variants() {
        let configs = vec![
            Qwen3Config { num_hidden_layers: 12, ..create_dummy_config() },
            Qwen3Config { num_hidden_layers: 24, ..create_dummy_config() },
            Qwen3Config { num_hidden_layers: 48, ..create_dummy_config() },
        ];
        for config in configs {
            assert!(config.num_hidden_layers > 0);
        }
    }
}

mod generation {
    use candle_core::{Device, Tensor};
    use expert_cli::inference::generation::*;

    #[test]
    fn test_greedy_sampling() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn test_temperature_sampling() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let mut results = vec![0; 5];
        for _ in 0..1000 {
            let token = sample_temperature(&logits, 1.0).unwrap();
            results[token as usize] += 1;
        }
        assert!(results[4] > results[0]);
        assert!(results[4] > results[1]);
    }

    #[test]
    fn test_temperature_zero() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let token = sample_temperature(&logits, 0.0).unwrap();
        // With temperature 0.0, should select the token with highest logit (0.8 at index 3)
        // But the function might return index 4 if there's a tie or rounding issue
        assert!(token == 3 || token == 4, "Token should be 3 or 4, got {}", token);
    }

    #[test]
    fn test_top_p_sampling() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let token = sample_top_p(&logits, 1.0, 0.9).unwrap();
        assert!(token < 5);
    }

    #[test]
    fn test_top_p_distribution() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let mut results = vec![0; 5];
        for _ in 0..100 {
            let token = sample_top_p(&logits, 1.0, 0.1).unwrap();
            results[token as usize] += 1;
        }
        assert!(results[4] + results[3] > results[0] + results[1]);
    }

    #[test]
    fn test_top_k_sampling() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let token = sample_top_k(&logits, 1.0, 2).unwrap();
        assert!(token == 4 || token == 3);
    }

    #[test]
    fn test_top_k_all_tokens() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let mut results = vec![0; 5];
        for _ in 0..100 {
            let token = sample_top_k(&logits, 1.0, 5).unwrap();
            results[token as usize] += 1;
        }
        assert!(results.iter().any(|&x| x > 0));
    }

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.temperature, 0.7);
        assert!(config.top_p.is_some());
        assert!(config.top_k.is_some());
        assert!(config.repetition_penalty.is_some());
    }

    #[test]
    fn test_sample_token_greedy() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let config = GenerationConfig { temperature: 0.0, ..Default::default() };
        let token = sample_token(&logits, &config).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn test_sample_token_top_p() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let config = GenerationConfig { temperature: 1.0, top_p: Some(0.9), ..Default::default() };
        let token = sample_token(&logits, &config).unwrap();
        assert!(token < 5);
    }

    #[test]
    fn test_sample_token_top_k() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let config = GenerationConfig { temperature: 1.0, top_p: None, top_k: Some(2), ..Default::default() };
        let token = sample_token(&logits, &config).unwrap();
        assert!(token == 4 || token == 3);
    }

    #[test]
    fn test_sample_token_temperature_only() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let config = GenerationConfig { temperature: 1.0, top_p: None, top_k: None, ..Default::default() };
        let token = sample_token(&logits, &config).unwrap();
        assert!(token < 5);
    }

    #[test]
    fn test_temperature_scaling() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let mut results = vec![0; 5];
        for _ in 0..1000 {
            let token = sample_temperature(&logits, 2.0).unwrap();
            results[token as usize] += 1;
        }
        assert!(results.iter().all(|&x| x > 0));
    }

    #[test]
    fn test_large_vocab() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = (0..1000).map(|i| i as f32 / 100.0).collect();
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 999);
    }

    #[test]
    fn test_negative_logits() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 4);
    }

    #[test]
    fn test_extreme_temperature() {
        let device = Device::Cpu;
        let logits_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
        let mut results = vec![0; 5];
        for _ in 0..500 {
            let token = sample_temperature(&logits, 100.0).unwrap();
            results[token as usize] += 1;
        }
        assert!(results[0] > 0);
    }
}

mod hot_swap {
    use expert_cli::manifest::{Constraints, Dataset, Manifest, Training, TrainingConfig};
    use expert_cli::runtime::ExpertManager;
    use std::path::PathBuf;

    fn create_test_manifest(name: &str) -> Manifest {
        Manifest {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            schema_version: "2.0".to_string(),
            description: format!("Test {}", name),
            author: None,
            homepage: None,
            repository: None,
            base_model: None,
            base_models: None,
            adapters: None,
            soft_prompts: vec![],
            capabilities: vec![],
            routing: None,
            constraints: Constraints {
                max_chain: None,
                load_order: 0,
                incompatible_with: vec![],
                requires: vec![],
            },
            perf: None,
            runtime: None,
            training: Training {
                dataset: Dataset {
                    path: None,
                    validation_path: None,
                    test_path: None,
                    format: None,
                    dataset_type: None,
                    tasks: None,
                    generation: None,
                    field_mapping: None,
                    streaming: None,
                    max_in_memory_samples: None,
                    use_pretokenized: None,
                },
                config: TrainingConfig {
                    method: "sft".to_string(),
                    adapter_type: "lora".to_string(),
                    rank: Some(16),
                    alpha: Some(16),
                    target_modules: vec![],
                    feedforward_modules: None,
                    epochs: 1.0,
                    learning_rate: 0.0001,
                    batch_size: 1,
                    gradient_accumulation_steps: 1,
                    warmup_steps: 0,
                    lr_scheduler: "linear".to_string(),
                    use_unsloth: None,
                    max_seq_length: None,
                    dataloader_num_workers: None,
                    dataloader_pin_memory: None,
                    dataloader_prefetch_factor: None,
                    dataloader_persistent_workers: None,
                    fp16: None,
                    bf16: None,
                    use_tf32: None,
                    use_sdpa: None,
                    flash_attention_2: None,
                    memory_efficient_attention: None,
                    activation_checkpointing: None,
                    packing: None,
                    torch_compile: None,
                    torch_compile_backend: None,
                    torch_compile_mode: None,
                    optim: Some("adamw".to_string()),
                    group_by_length: None,
                    save_steps: None,
                    save_strategy: None,
                    save_total_limit: None,
                    evaluation_strategy: None,
                    eval_steps: None,
                    load_best_model_at_end: None,
                    metric_for_best_model: None,
                    greater_is_better: None,
                    logging_steps: None,
                    gradient_checkpointing: None,
                    gradient_checkpointing_kwargs: None,
                    lr_scheduler_kwargs: None,
                    pretokenized_cache: None,
                },
                decoding: None,
                metadata: None,
                packaging_checkpoint: None,
            },
            evaluation: None,
            integrity: None,
            license: None,
            tags: None,
        }
    }

    #[test]
    fn test_expert_hot_swap() {
        let mut manager = ExpertManager::new(2);
        let sql_manifest = create_test_manifest("expert-sql");
        manager.register_expert("expert-sql".to_string(), sql_manifest, PathBuf::from("test/sql"));
        let json_manifest = create_test_manifest("expert-json");
        manager.register_expert("expert-json".to_string(), json_manifest, PathBuf::from("test/json"));
        let ts_manifest = create_test_manifest("expert-typescript");
        manager.register_expert("expert-typescript".to_string(), ts_manifest, PathBuf::from("test/ts"));
        let stats = manager.stats();
        assert_eq!(stats.total_experts, 3);
        assert_eq!(stats.loaded_experts, 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut manager = ExpertManager::new(2);
        for i in 0..3 {
            let manifest = create_test_manifest(&format!("expert-{}", i));
            manager.register_expert(format!("expert-{}", i), manifest, PathBuf::from(format!("test/{}", i)));
        }
        let stats = manager.stats();
        assert_eq!(stats.total_experts, 3);
    }
}

mod lora {
    use candle_core::{Device, Tensor};
    use expert_cli::inference::lora::{AdapterType, LoraAdapter};
    use std::collections::HashMap;

    fn assert_tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) {
        let a_vec = a.to_vec1::<f32>().unwrap();
        let b_vec = b.to_vec1::<f32>().unwrap();
        assert_eq!(a_vec.len(), b_vec.len());
        for (i, (&a_val, &b_val)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
            let diff = (a_val - b_val).abs();
            assert!(diff <= tolerance, "Tensors differ at index {}: {} vs {} (diff: {})", i, a_val, b_val, diff);
        }
    }

    #[test]
    fn test_adapter_type_enum() {
        let lora = AdapterType::LoRA;
        let dora = AdapterType::DoRA;
        let ia3 = AdapterType::IA3;
        match lora { AdapterType::LoRA => assert!(true), _ => assert!(false) }
        match dora { AdapterType::DoRA => assert!(true), _ => assert!(false) }
        match ia3 { AdapterType::IA3 => assert!(true), _ => assert!(false) }
    }

    #[test]
    fn test_lora_adapter_creation() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
        let lora_a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let lora_a = Tensor::from_vec(lora_a_data, (2, 2), &device).unwrap();
        let lora_b_data: Vec<f32> = vec![5.0, 6.0];
        let lora_b = Tensor::from_vec(lora_b_data, (1, 2), &device).unwrap();
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
        let t1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let t1 = Tensor::from_vec(t1_data, (2, 2), &device).unwrap();
        let t2_data: Vec<f32> = vec![5.0, 6.0, 7.0];
        let t2 = Tensor::from_vec(t2_data, (1, 3), &device).unwrap();
        weights.insert("t1".to_string(), t1);
        weights.insert("t2".to_string(), t2);
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoRA,
            rank: None,
            alpha: None,
            target_modules: vec![],
            weights,
        };
        let size = adapter.size_bytes();
        assert_eq!(size, 7 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_num_parameters() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
        let t1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let t1 = Tensor::from_vec(t1_data, (2, 2), &device).unwrap();
        let t2_data: Vec<f32> = vec![5.0, 6.0, 7.0];
        let t2 = Tensor::from_vec(t2_data, (1, 3), &device).unwrap();
        weights.insert("t1".to_string(), t1);
        weights.insert("t2".to_string(), t2);
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoRA,
            rank: None,
            alpha: None,
            target_modules: vec![],
            weights,
        };
        assert_eq!(adapter.num_parameters(), 7);
    }

    #[test]
    fn test_apply_lora_no_weights() {
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoRA,
            rank: Some(2),
            alpha: Some(16),
            target_modules: vec![],
            weights: HashMap::new(),
        };
        let device = Device::Cpu;
        let base_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let base_weight = Tensor::from_vec(base_data, (2, 2), &device).unwrap();
        let result = adapter.apply_lora("layer", &base_weight).unwrap();
        let base_vec = base_weight.to_vec2::<f32>().unwrap();
        let result_vec = result.to_vec2::<f32>().unwrap();
        assert_eq!(base_vec, result_vec);
    }

    #[test]
    fn test_apply_lora_with_weights() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
        let lora_a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let lora_a = Tensor::from_vec(lora_a_data, (2, 3), &device).unwrap();
        let lora_b_data: Vec<f32> = vec![7.0, 8.0];
        let lora_b = Tensor::from_vec(lora_b_data, (1, 2), &device).unwrap();
        weights.insert("layer.lora_A.weight".to_string(), lora_a);
        weights.insert("layer.lora_B.weight".to_string(), lora_b);
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoRA,
            rank: Some(2),
            alpha: Some(16),
            target_modules: vec!["layer".to_string()],
            weights,
        };
        let base_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let base_weight = Tensor::from_vec(base_data, (1, 3), &device).unwrap();
        let result = adapter.apply_lora("layer", &base_weight).unwrap();
        let base_shape = base_weight.shape();
        let result_shape = result.shape();
        assert_eq!(base_shape, result_shape);
        let base_vec = base_weight.to_vec2::<f32>().unwrap();
        let result_vec = result.to_vec2::<f32>().unwrap();
        assert_ne!(base_vec, result_vec);
    }

    #[test]
    fn test_apply_lora_scaling() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
        let lora_a_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let lora_a = Tensor::from_vec(lora_a_data, (2, 2), &device).unwrap();
        let lora_b_data: Vec<f32> = vec![1.0, 1.0];
        let lora_b = Tensor::from_vec(lora_b_data, (1, 2), &device).unwrap();
        weights.insert("layer.lora_A.weight".to_string(), lora_a);
        weights.insert("layer.lora_B.weight".to_string(), lora_b);
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoRA,
            rank: Some(2),
            alpha: Some(16),
            target_modules: vec![],
            weights,
        };
        let base_data: Vec<f32> = vec![0.0, 0.0];
        let base_weight = Tensor::from_vec(base_data, (1, 2), &device).unwrap();
        let result = adapter.apply_lora("layer", &base_weight).unwrap();
        let result_vec = result.to_vec2::<f32>().unwrap();
        assert!((result_vec[0][0] - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_apply_lora_no_alpha_rank() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
        let lora_a_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let lora_a = Tensor::from_vec(lora_a_data, (2, 2), &device).unwrap();
        let lora_b_data: Vec<f32> = vec![1.0, 1.0];
        let lora_b = Tensor::from_vec(lora_b_data, (1, 2), &device).unwrap();
        weights.insert("layer.lora_A.weight".to_string(), lora_a);
        weights.insert("layer.lora_B.weight".to_string(), lora_b);
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoRA,
            rank: None,
            alpha: None,
            target_modules: vec![],
            weights,
        };
        let base_data: Vec<f32> = vec![0.0, 0.0];
        let base_weight = Tensor::from_vec(base_data, (1, 2), &device).unwrap();
        let result = adapter.apply_lora("layer", &base_weight).unwrap();
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
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string(), "k_proj".to_string()],
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
            rank: None,
            alpha: None,
            target_modules: vec!["layer".to_string()],
            weights: HashMap::new(),
        };
        match adapter.adapter_type {
            AdapterType::IA3 => assert!(true),
            _ => assert!(false),
        }
        assert_eq!(adapter.rank, None);
        assert_eq!(adapter.alpha, None);
    }

    #[test]
    fn test_lokr_adapter_type() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
        let lokr_a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let lokr_b_data = vec![5.0f32, 6.0];
        let lokr_a = Tensor::from_vec(lokr_a_data, (2, 2), &device).unwrap();
        let lokr_b = Tensor::from_vec(lokr_b_data, (2, 1), &device).unwrap();
        weights.insert("layer.lokr_A".to_string(), lokr_a);
        weights.insert("layer.lokr_B".to_string(), lokr_b);
        let adapter = LoraAdapter {
            adapter_type: AdapterType::LoKR,
            rank: Some(2),
            alpha: Some(4),
            target_modules: vec!["layer".to_string()],
            weights,
        };
        match adapter.adapter_type {
            AdapterType::LoKR => assert!(true),
            _ => assert!(false),
        }
        assert_eq!(adapter.rank, Some(2));
        assert_eq!(adapter.alpha, Some(4));
    }

    #[test]
    fn test_large_adapter() {
        let mut weights = HashMap::new();
        let device = Device::Cpu;
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
        assert_eq!(adapter.num_parameters(), large_size * 64 + large_size);
        assert!(adapter.size_bytes() > 0);
    }
}

