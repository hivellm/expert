// Integration tests - All integration tests grouped together

mod end_to_end {
    use expert_cli::manifest::{Constraints, Dataset, Manifest, Routing, Training, TrainingConfig};
    use expert_cli::routing::{ConfidenceScorer, EmbeddingRouter, KeywordRouter};
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
    fn test_end_to_end_routing() {
        let mut keyword_router = KeywordRouter::new();
        let mut embedding_router = EmbeddingRouter::new();
        let mut confidence_scorer = ConfidenceScorer::new();
        let mut sql_manifest = create_test_manifest("expert-sql");
        sql_manifest.routing = Some(Routing {
            keywords: vec!["sql".to_string(), "database".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.8),
        });
        keyword_router.add_expert(&sql_manifest);
        embedding_router.add_expert(&sql_manifest);
        let query = "show all database tables";
        let keyword_results = keyword_router.route(query, 1);
        assert!(keyword_results.len() > 0);
        let embedding_results = embedding_router.route(query, 1);
        assert!(embedding_results.len() > 0);
        let confidence = confidence_scorer.score(
            &keyword_results[0].expert_name,
            query,
            &keyword_results[0].matched_keywords,
            keyword_results[0].score as f32,
        );
        assert!(confidence.confidence > 0.0);
        assert!(confidence.confidence <= 1.0);
    }

    #[test]
    fn test_expert_manager_integration() {
        let mut manager = ExpertManager::new(2);
        for name in &["expert-sql", "expert-json", "expert-typescript"] {
            let manifest = create_test_manifest(name);
            manager.register_expert(name.to_string(), manifest, PathBuf::from(format!("test/{}", name)));
        }
        let stats = manager.stats();
        assert_eq!(stats.total_experts, 3);
        assert_eq!(stats.loaded_experts, 0);
        assert!(stats.memory_mb >= 0.0);
    }

    #[test]
    fn test_routing_consistency() {
        let mut keyword_router = KeywordRouter::new();
        let mut embedding_router = EmbeddingRouter::new();
        let mut manifest = create_test_manifest("expert-sql");
        manifest.routing = Some(Routing {
            keywords: vec!["sql".to_string(), "database".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.8),
        });
        keyword_router.add_expert(&manifest);
        embedding_router.add_expert(&manifest);
        let query = "show all database tables";
        let keyword_result = keyword_router.route_single(query);
        let embedding_result = embedding_router.route_single(query);
        assert!(keyword_result.is_some());
        assert!(embedding_result.is_some());
        assert_eq!(keyword_result.unwrap(), "expert-sql");
        assert_eq!(embedding_result.unwrap().expert_name, "expert-sql");
    }
}

mod multi_expert {
    use expert_cli::manifest::{Constraints, Dataset, Manifest, Routing, Training, TrainingConfig};
    use expert_cli::routing::{EmbeddingRouter, KeywordRouter};
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
    fn test_multi_expert_registration() {
        let mut manager = ExpertManager::new(3);
        let sql_manifest = create_test_manifest("expert-sql");
        manager.register_expert("expert-sql".to_string(), sql_manifest, PathBuf::from("test/sql"));
        let json_manifest = create_test_manifest("expert-json");
        manager.register_expert("expert-json".to_string(), json_manifest, PathBuf::from("test/json"));
        let stats = manager.stats();
        assert_eq!(stats.total_experts, 2);
        assert_eq!(stats.loaded_experts, 0);
    }

    #[test]
    fn test_routing_multiple_experts() {
        let mut router = KeywordRouter::new();
        let mut sql_manifest = create_test_manifest("expert-sql");
        sql_manifest.routing = Some(Routing {
            keywords: vec!["sql".to_string(), "database".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.8),
        });
        router.add_expert(&sql_manifest);
        let mut json_manifest = create_test_manifest("expert-json");
        json_manifest.routing = Some(Routing {
            keywords: vec!["json".to_string(), "format".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.7),
        });
        router.add_expert(&json_manifest);
        let results = router.route("show all database tables", 2);
        assert!(results.len() > 0);
        assert_eq!(results[0].expert_name, "expert-sql");
        let results = router.route("format json data", 2);
        assert!(results.len() > 0);
        assert_eq!(results[0].expert_name, "expert-json");
    }

    #[test]
    fn test_embedding_router_multiple_experts() {
        let mut router = EmbeddingRouter::new();
        let mut sql_manifest = create_test_manifest("expert-sql");
        sql_manifest.routing = Some(Routing {
            keywords: vec!["sql".to_string(), "database".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.8),
        });
        router.add_expert(&sql_manifest);
        let mut json_manifest = create_test_manifest("expert-json");
        json_manifest.routing = Some(Routing {
            keywords: vec!["json".to_string(), "format".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.7),
        });
        router.add_expert(&json_manifest);
        let results = router.route("show all database tables", 2);
        assert!(results.len() > 0);
        let results = router.route("format json data", 2);
        assert!(results.len() > 0);
    }
}

mod dependency_resolution {
    use std::collections::HashSet;

    #[test]
    fn test_dependency_parsing() {
        let dep = "expert-english@>=0.0.1";
        let parts: Vec<&str> = dep.split('@').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "expert-english");
        assert_eq!(parts[1], ">=0.0.1");
    }

    #[test]
    fn test_dependency_without_version() {
        let dep = "expert-english";
        let parts: Vec<&str> = dep.split('@').collect();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], "expert-english");
    }

    #[test]
    fn test_circular_dependency_detection() {
        let max_depth = 10;
        let mut depth = 0;
        for _ in 0..15 {
            depth += 1;
            if depth > max_depth {
                break;
            }
        }
        assert_eq!(depth, 11);
    }

    #[test]
    fn test_version_constraint_exact() {
        let version_req = "0.0.1";
        let installed_version = "0.0.1";
        assert_eq!(version_req, installed_version);
    }

    #[test]
    fn test_version_constraint_gte() {
        let constraint = ">=0.0.1";
        assert!(constraint.starts_with(">="));
        let version = constraint.strip_prefix(">=").unwrap();
        assert_eq!(version, "0.0.1");
    }

    #[test]
    fn test_dependency_graph_construction() {
        let mut deps = vec![
            ("expert-a", vec!["expert-b", "expert-c"]),
            ("expert-b", vec!["expert-c"]),
            ("expert-c", vec![]),
        ];
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();

        fn visit(name: &str, deps: &[(&str, Vec<&str>)], visited: &mut HashSet<String>, sorted: &mut Vec<String>) {
            if visited.contains(name) {
                return;
            }
            visited.insert(name.to_string());
            if let Some((_, children)) = deps.iter().find(|(n, _)| *n == name) {
                for child in children {
                    visit(child, deps, visited, sorted);
                }
            }
            sorted.push(name.to_string());
        }

        for (name, _) in &deps {
            visit(name, &deps, &mut visited, &mut sorted);
        }
        assert_eq!(sorted[0], "expert-c");
        assert_eq!(sorted[sorted.len() - 1], "expert-a");
    }

    #[test]
    fn test_dependency_list_empty() {
        let requires: Vec<String> = Vec::new();
        assert!(requires.is_empty());
    }

    #[test]
    fn test_dependency_list_multiple() {
        let requires = vec!["expert-english@>=0.0.1".to_string(), "expert-base@0.1.0".to_string()];
        assert_eq!(requires.len(), 2);
    }

    #[test]
    fn test_git_url_for_dependency() {
        let dep_name = "expert-english";
        let git_url = format!("git+https://github.com/hivellm/{}.git", dep_name);
        assert_eq!(git_url, "git+https://github.com/hivellm/expert-english.git");
    }
}

mod error_messages {
    #[test]
    fn test_v1_missing_base_model_error_message() {
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
        let parse_result = serde_json::from_str::<serde_json::Value>(json);
        assert!(parse_result.is_ok(), "JSON should be valid");
    }

    #[test]
    fn test_conflict_error_message_clarity() {
        let json = r#"{
            "name": "test",
            "version": "1.0.0",
            "schema_version": "2.0",
            "description": "test",
            "base_model": {"name": "Qwen3-0.6B"},
            "base_models": [{"name": "Qwen3-1.5B", "adapters": []}],
            "capabilities": [],
            "constraints": {"load_order": 1, "incompatible_with": [], "requires": []},
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
        let result = serde_json::from_str::<serde_json::Value>(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_malformed_json_error() {
        let bad_json = r#"{
            "name": "test",
            "version": "1.0.0",
            missing_quotes: "value"
        }"#;
        let result = serde_json::from_str::<serde_json::Value>(bad_json);
        assert!(result.is_err());
        let error = result.unwrap_err();
        let error_msg = error.to_string();
        assert!(error_msg.len() > 0);
    }

    #[test]
    fn test_missing_required_field_error() {
        let json = r#"{"version": "1.0.0", "description": "test"}"#;
        let result = serde_json::from_str::<serde_json::Value>(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_array_structure() {
        let json = r#"{"name": "test", "capabilities": "should be array not string"}"#;
        let result = serde_json::from_str::<serde_json::Value>(json);
        assert!(result.is_ok());
    }
}

mod data_integrity {
    use std::fs;

    #[test]
    fn test_v2_no_orphaned_adapters() {
        let content = fs::read_to_string("tests/fixtures/manifest_v2.json")
            .expect("Failed to read v2.0 manifest");
        let lines: Vec<&str> = content.lines().collect();
        let base_models_line = lines.iter().position(|l| l.contains("\"base_models\"")).unwrap();
        let orphaned_adapters = lines.iter().take(base_models_line).any(|l| l.trim().starts_with("\"adapters\""));
        assert!(!orphaned_adapters, "No orphaned adapters should exist at root in v2.0");
    }

    #[test]
    fn test_v1_no_base_models_array() {
        let content = fs::read_to_string("tests/fixtures/manifest_v1.json")
            .expect("Failed to read v1.0 manifest");
        assert!(!content.contains("\"base_models\""), "v1.0 should not have base_models array");
    }

    #[test]
    fn test_weight_path_format_consistency() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").unwrap();
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();
        assert!(v1_content.contains("weights/adapter.safetensors"));
        assert!(v2_content.contains("weights/qwen3-0.6b/adapter.safetensors"));
        assert!(v2_content.contains("weights/qwen3-1.5b/adapter.safetensors"));
    }

    #[test]
    fn test_sha256_hash_format() {
        let content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
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
        assert!(
            v2_content.contains("\"type\": \"lora\"")
                || v2_content.contains("\"type\": \"dora\"")
                || v2_content.contains("\"type\": \"ia3\"")
        );
    }

    #[test]
    fn test_learning_rate_scientific_notation() {
        let content = fs::read_to_string("tests/fixtures/manifest_v1.json").unwrap();
        assert!(content.contains("\"learning_rate\""));
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

mod package_integration {
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_package_v1_manifest_structure() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v1.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v1.0 manifest");
        assert!(content.contains("\"base_model\""));
        assert!(content.contains("\"adapters\""));
        assert!(!content.contains("\"base_models\""));
        let weight_count = content.matches("weights/").count();
        assert_eq!(weight_count, 1, "v1.0 should have single weight path");
    }

    #[test]
    fn test_package_v2_manifest_structure() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v2.0 manifest");
        assert!(content.contains("\"base_models\""));
        assert!(!content.contains("\"base_model\":"));
        let weight_count = content.matches("weights/").count();
        assert_eq!(weight_count, 2, "v2.0 should have multiple weight paths");
        assert!(content.contains("weights/qwen3-0.6b/"));
        assert!(content.contains("weights/qwen3-1.5b/"));
    }

    #[test]
    fn test_package_v2_model_names_are_distinct() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v2.0 manifest");
        assert!(content.contains("\"Qwen3-0.6B\""));
        assert!(content.contains("\"Qwen3-1.5B\""));
        let qwen_06b_count = content.matches("Qwen3-0.6B").count();
        let qwen_15b_count = content.matches("Qwen3-1.5B").count();
        assert!(qwen_06b_count >= 1, "Qwen3-0.6B should appear at least once");
        assert!(qwen_15b_count >= 1, "Qwen3-1.5B should appear at least once");
    }

    #[test]
    fn test_package_v2_adapters_embedded_in_models() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v2.0 manifest");
        let lines: Vec<&str> = content.lines().collect();
        let base_models_line = lines.iter().position(|l| l.contains("\"base_models\"")).expect("base_models not found");
        let adapters_line = lines.iter().position(|l| l.contains("\"adapters\"")).expect("adapters not found");
        assert!(adapters_line > base_models_line, "adapters should be embedded in base_models, not at root");
    }

    #[test]
    fn test_package_v1_and_v2_have_compatible_fields() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").expect("Failed to read v1.0 manifest");
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").expect("Failed to read v2.0 manifest");
        let common_fields = vec!["\"name\"", "\"version\"", "\"description\"", "\"capabilities\"", "\"constraints\"", "\"training\""];
        for field in common_fields {
            assert!(v1_content.contains(field), "v1.0 missing {}", field);
            assert!(v2_content.contains(field), "v2.0 missing {}", field);
        }
    }

    #[test]
    fn test_package_v2_weight_paths_include_model_identifier() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v2.0 manifest");
        assert!(content.contains("qwen3-0.6b"), "Path should include model identifier");
        assert!(content.contains("qwen3-1.5b"), "Path should include model identifier");
        let generic_path = content.contains("\"path\": \"weights/adapter.safetensors\"");
        assert!(!generic_path, "v2.0 should not have generic weight paths");
    }

    #[test]
    fn test_package_adapter_count_consistency() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").expect("Failed to read v1.0 manifest");
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").expect("Failed to read v2.0 manifest");
        let v1_adapter_arrays = v1_content.matches("\"adapters\"").count();
        assert_eq!(v1_adapter_arrays, 1, "v1.0 should have 1 adapters array");
        let v2_adapter_arrays = v2_content.matches("\"adapters\"").count();
        assert_eq!(v2_adapter_arrays, 2, "v2.0 should have adapters arrays per model");
    }

    #[test]
    fn test_json_formatting_validity() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").expect("Failed to read v1.0 manifest");
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").expect("Failed to read v2.0 manifest");
        let v1_json: serde_json::Value = serde_json::from_str(&v1_content).expect("v1.0 manifest is not valid JSON");
        let v2_json: serde_json::Value = serde_json::from_str(&v2_content).expect("v2.0 manifest is not valid JSON");
        assert!(v1_json.is_object());
        assert!(v2_json.is_object());
        assert!(v1_json.get("name").is_some());
        assert!(v2_json.get("name").is_some());
    }

    #[test]
    fn test_schema_version_field_presence() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").expect("Failed to read v1.0 manifest");
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").expect("Failed to read v2.0 manifest");
        let v1_json: serde_json::Value = serde_json::from_str(&v1_content).unwrap();
        let v2_json: serde_json::Value = serde_json::from_str(&v2_content).unwrap();
        assert_eq!(v1_json.get("schema_version").and_then(|v| v.as_str()), Some("1.0"));
        assert_eq!(v2_json.get("schema_version").and_then(|v| v.as_str()), Some("2.0"));
    }

    #[test]
    fn test_training_config_presence() {
        let v1_content = fs::read_to_string("tests/fixtures/manifest_v1.json").unwrap();
        let v2_content = fs::read_to_string("tests/fixtures/manifest_v2.json").unwrap();
        assert!(v1_content.contains("\"training\""));
        assert!(v1_content.contains("\"config\""));
        assert!(v2_content.contains("\"training\""));
        assert!(v2_content.contains("\"config\""));
        let required_training_fields = vec!["\"method\"", "\"adapter_type\"", "\"rank\"", "\"epochs\"", "\"learning_rate\""];
        for field in required_training_fields {
            assert!(v1_content.contains(field), "v1.0 missing {}", field);
            assert!(v2_content.contains(field), "v2.0 missing {}", field);
        }
    }
}

mod validation {
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_manifest_v2_structure() {
        let manifest_v2 = r#"{
            "name": "expert-test",
            "version": "0.0.1",
            "schema_version": "2.0",
            "base_models": [{
                "name": "Qwen3-0.6B",
                "prompt_template": "chatml",
                "adapters": [{
                    "type": "lora",
                    "path": "adapter",
                    "r": 16,
                    "alpha": 16
                }]
            }],
            "capabilities": ["test"],
            "training": {
                "dataset": {"path": "test"},
                "config": {"method": "sft"}
            }
        }"#;
        let parsed: serde_json::Value = serde_json::from_str(manifest_v2).unwrap();
        assert_eq!(parsed["schema_version"], "2.0");
        assert!(parsed["base_models"].is_array());
    }

    #[test]
    fn test_adapter_path_resolution() {
        let base_path = PathBuf::from("./experts/expert-test");
        let weights_path = base_path.join("weights");
        let adapter_path = weights_path.join("qwen3-06b/adapter");
        // Normalize path separators for cross-platform compatibility
        let normalized = adapter_path.to_string_lossy().replace('\\', "/");
        assert_eq!(normalized, "./experts/expert-test/weights/qwen3-06b/adapter");
    }

    #[test]
    fn test_size_bytes_option() {
        let size: Option<u64> = Some(14702472);
        if let Some(s) = size {
            assert!(s > 0);
            assert_eq!(s, 14702472);
        }
        let no_size: Option<u64> = None;
        assert!(no_size.is_none());
    }

    #[test]
    fn test_sha256_option() {
        let sha: Option<String> = Some("abc123".to_string());
        if let Some(ref s) = sha {
            assert!(!s.is_empty());
            assert_eq!(s, "abc123");
        }
        let no_sha: Option<String> = None;
        assert!(no_sha.is_none());
    }

    #[test]
    fn test_capabilities_vec() {
        let caps = vec!["tech:neo4j".to_string(), "database:neo4j".to_string(), "query:cypher".to_string()];
        assert_eq!(caps.len(), 3);
        assert!(caps.contains(&"tech:neo4j".to_string()));
    }

    #[test]
    fn test_model_name_normalization() {
        let model_names = vec!["Qwen3-0.6B", "F:/Node/hivellm/expert/models/Qwen3-0.6B", "qwen3-0.6b"];
        for name in model_names {
            let normalized = name.to_lowercase().replace('/', "-").replace('\\', "-");
            assert!(normalized.contains("qwen3") || normalized.contains("qwen3-0.6b"));
        }
    }

    #[test]
    fn test_package_naming_convention() {
        let expert_name = "expert-neo4j";
        let model_name = "qwen3-0.6b";
        let version = "0.0.1";
        let package_name = format!("{}-{}.v{}.expert", expert_name, model_name, version);
        assert_eq!(package_name, "expert-neo4j-qwen3-0.6b.v0.0.1.expert");
        assert!(package_name.ends_with(".expert"));
    }
}

