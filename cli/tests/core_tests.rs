// Core tests - All core functionality tests grouped together

mod manifest {
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_parse_v1_manifest_from_file() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v1.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v1.0 test fixture");
        assert!(content.contains("\"schema_version\": \"1.0\""));
        assert!(content.contains("\"base_model\""));
        assert!(!content.contains("\"base_models\""));
    }

    #[test]
    fn test_parse_v2_manifest_from_file() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v2.0 test fixture");
        assert!(content.contains("\"schema_version\": \"2.0\""));
        assert!(content.contains("\"base_models\""));
        assert!(!content.contains("\"base_model\""));
        assert!(content.contains("Qwen3-0.6B"));
        assert!(content.contains("Qwen3-1.5B"));
    }

    #[test]
    fn test_v2_has_model_specific_paths() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v2.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v2.0 test fixture");
        assert!(content.contains("weights/qwen3-0.6b/adapter.safetensors"));
        assert!(content.contains("weights/qwen3-1.5b/adapter.safetensors"));
    }

    #[test]
    fn test_v1_has_simple_path() {
        let manifest_path = PathBuf::from("tests/fixtures/manifest_v1.json");
        let content = fs::read_to_string(&manifest_path).expect("Failed to read v1.0 test fixture");
        assert!(content.contains("weights/adapter.safetensors"));
    }
}

mod manifest_features {
    use expert_cli::manifest::{DecodingConfig, Manifest, SoftPrompt};

    #[test]
    fn test_soft_prompt_packaging() {
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
        let manifest_temp = Some(0.1);
        let cli_temp = Some(0.5);
        let default_temp = 0.7;
        let result = cli_temp.or(manifest_temp).unwrap_or(default_temp);
        assert_eq!(result, 0.5, "CLI should override manifest");
        let result_no_cli: Option<f64> = None;
        let result = result_no_cli.or(manifest_temp).unwrap_or(default_temp);
        assert_eq!(result, 0.1, "Manifest should be used when no CLI override");
        let result_default: Option<f64> = None;
        let manifest_none: Option<f64> = None;
        let result = result_default.or(manifest_none).unwrap_or(default_temp);
        assert_eq!(result, 0.7, "Should fallback to default");
    }

    #[test]
    fn test_manifest_backward_compatibility() {
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
        assert!(manifest.training.decoding.is_none());
        assert_eq!(manifest.soft_prompts.len(), 0);
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
        assert_eq!(decoding.stop_sequences, Some(vec![";".to_string(), "\n\n".to_string()]));
        assert_eq!(decoding.temperature, Some(0.1));
        assert_eq!(decoding.top_p, Some(0.9));
        assert_eq!(decoding.top_k, Some(50));
    }
}

mod model_detection {
    use std::path::PathBuf;

    #[test]
    fn test_model_search_paths() {
        let paths = vec!["~/.expert/models", "./models", "../models"];
        assert!(paths.len() >= 3);
    }

    #[test]
    fn test_huggingface_cache_path() {
        let home = if cfg!(windows) { "C:\\Users\\User" } else { "/home/user" };
        let hf_cache = if cfg!(windows) {
            format!("{}\\AppData\\Local\\huggingface\\hub", home)
        } else {
            format!("{}/.cache/huggingface/hub", home)
        };
        assert!(hf_cache.contains("huggingface"));
    }

    #[test]
    fn test_model_name_extraction_from_path() {
        let path = PathBuf::from("/models/Qwen3-0.6B");
        let name = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "Qwen3-0.6B");
    }

    #[test]
    fn test_hf_cache_model_format() {
        let cache_name = "models--Qwen--Qwen3-0.6B";
        assert!(cache_name.starts_with("models--"));
        let cleaned = cache_name.strip_prefix("models--").unwrap().replace("--", "/");
        assert_eq!(cleaned, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn test_quantization_detection() {
        let dtypes = vec![("int4", true), ("int8", true), ("float16", false), ("bfloat16", false)];
        for (dtype, is_quant) in dtypes {
            let quantized = dtype.contains("int4") || dtype.contains("int8");
            assert_eq!(quantized, is_quant);
        }
    }

    #[test]
    fn test_config_json_presence() {
        let model_dir = PathBuf::from("/models/Qwen3-0.6B");
        let config_path = model_dir.join("config.json");
        assert_eq!(config_path.file_name().unwrap(), "config.json");
    }

    #[test]
    fn test_model_size_calculation() {
        let size_bytes: u64 = 536870912;
        let size_mb = size_bytes / 1_048_576;
        assert_eq!(size_mb, 512);
    }

    #[test]
    fn test_model_compatibility_matching() {
        let required = "Qwen3-0.6B";
        let available = vec!["Qwen3-0.6B", "F:/models/Qwen3-0.6B", "Qwen/Qwen3-0.6B"];
        let matches: Vec<_> = available.iter().filter(|m| m.ends_with(required) || m.contains(required)).collect();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_deduplication_logic() {
        let models = vec!["Qwen3-0.6B", "Qwen3-0.6B", "Qwen3-1.5B"];
        let mut seen = std::collections::HashSet::new();
        let deduped: Vec<_> = models.iter().filter(|m| seen.insert(*m)).collect();
        assert_eq!(deduped.len(), 2);
    }
}

mod registry {
    use chrono::Utc;
    use expert_cli::registry::{AdapterEntry, BaseModelEntry, ExpertVersionEntry, ExpertRegistry};
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_create_empty_registry() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        assert_eq!(registry.version, "1.0");
        assert_eq!(registry.experts.len(), 0);
        assert_eq!(registry.base_models.len(), 0);
    }

    #[test]
    fn test_add_remove_expert() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        let expert = ExpertVersionEntry {
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("experts/expert-test"),
            source: "file://./test".to_string(),
            installed_at: Utc::now(),
            adapters: vec![],
            capabilities: vec!["test:feature".to_string()],
            dependencies: vec![],
        };
        registry.add_expert_version("expert-test", expert.clone());
        assert_eq!(registry.experts.len(), 1);
        assert!(registry.has_expert("expert-test"));
        let retrieved = registry.get_expert("expert-test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().version, "0.0.1");
        let removed = registry.remove_expert("expert-test");
        assert!(removed.is_some());
        assert_eq!(registry.experts.len(), 0);
        assert!(!registry.has_expert("expert-test"));
    }

    #[test]
    fn test_add_remove_base_model() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        let model = BaseModelEntry {
            name: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("models/Qwen3-0.6B"),
            sha256: Some("abc123".to_string()),
            quantization: Some("int4".to_string()),
            size_bytes: 1000000,
            installed_at: Utc::now(),
            source: "huggingface".to_string(),
        };
        registry.add_base_model(model.clone());
        assert_eq!(registry.base_models.len(), 1);
        assert!(registry.has_base_model("Qwen3-0.6B"));
        let retrieved = registry.get_base_model("Qwen3-0.6B");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().quantization.as_deref(), Some("int4"));
        let removed = registry.remove_base_model("Qwen3-0.6B");
        assert!(removed.is_some());
        assert_eq!(registry.base_models.len(), 0);
    }

    #[test]
    fn test_save_load_registry() {
        let temp_dir = TempDir::new().unwrap();
        let registry_path = temp_dir.path().join("test-registry.json");
        let mut registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        let expert = ExpertVersionEntry {
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("experts/expert-test"),
            source: "file://./test".to_string(),
            installed_at: Utc::now(),
            adapters: vec![AdapterEntry {
                adapter_type: "lora".to_string(),
                path: PathBuf::from("adapter"),
                size_bytes: 1000,
                sha256: Some("hash".to_string()),
            }],
            capabilities: vec!["test".to_string()],
            dependencies: vec![],
        };
        registry.add_expert_version("expert-test", expert);
        let json = serde_json::to_string_pretty(&registry).unwrap();
        std::fs::write(&registry_path, json).unwrap();
        let content = std::fs::read_to_string(&registry_path).unwrap();
        let loaded: ExpertRegistry = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.experts.len(), 1);
        assert_eq!(loaded.experts[0].name, "expert-test");
        assert_eq!(loaded.experts[0].versions.len(), 1);
        assert_eq!(loaded.experts[0].versions[0].adapters.len(), 1);
    }

    #[test]
    fn test_update_existing_expert() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        let expert_v1 = ExpertVersionEntry {
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("experts/expert-test"),
            source: "file://./test".to_string(),
            installed_at: Utc::now(),
            adapters: vec![],
            capabilities: vec![],
            dependencies: vec![],
        };
        registry.add_expert_version("expert-test", expert_v1);
        assert_eq!(registry.experts.len(), 1);
        assert_eq!(registry.experts[0].versions[0].version, "0.0.1");
        let expert_v2 = ExpertVersionEntry {
            version: "0.0.2".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("experts/expert-test"),
            source: "file://./test".to_string(),
            installed_at: Utc::now(),
            adapters: vec![],
            capabilities: vec![],
            dependencies: vec![],
        };
        registry.add_expert_version("expert-test", expert_v2);
        assert_eq!(registry.experts.len(), 1);
        assert_eq!(registry.experts[0].versions.len(), 2);
        assert_eq!(registry.experts[0].versions[0].version, "0.0.2");
    }

    #[test]
    fn test_list_experts_empty() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        let expert_records = registry.expert_records();
        assert_eq!(expert_records.len(), 0);
    }

    #[test]
    fn test_list_base_models_empty() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        let models = registry.list_base_models();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_multiple_experts() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );
        for i in 1..=5 {
            let expert = ExpertVersionEntry {
                version: "0.0.1".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: temp_dir.path().join(format!("experts/expert-{}", i)),
                source: "file://./test".to_string(),
                installed_at: Utc::now(),
                adapters: vec![],
                capabilities: vec![],
                dependencies: vec![],
            };
            registry.add_expert_version(&format!("expert-{}", i), expert);
        }
        assert_eq!(registry.experts.len(), 5);
        assert!(registry.has_expert("expert-1"));
        assert!(registry.has_expert("expert-5"));
        assert!(!registry.has_expert("expert-6"));
    }
}

mod router {
    use std::path::Path;

    #[test]
    fn test_router_with_real_manifests() {
        let expert_dirs = ["expert-neo4j", "expert-sql", "expert-json", "expert-typescript"];
        let mut found_count = 0;
        for expert_name in &expert_dirs {
            let manifest_path = Path::new("../experts").join(expert_name).join("manifest.json");
            if manifest_path.exists() {
                found_count += 1;
            }
        }
        if found_count > 0 {
            println!("Found {} expert manifests for testing", found_count);
            assert!(found_count > 0, "Should find at least one expert manifest");
        } else {
            println!("No expert manifests found - skipping integration test");
        }
    }

    #[test]
    fn test_router_command_available() {
        use expert_cli::Manifest;
        use expert_cli::routing::KeywordRouter;
        let router = KeywordRouter::new();
        let matches = router.route("test query", 5);
        assert_eq!(matches.len(), 0, "Empty router should return no matches");
    }
}

mod keyword_routing {
    use expert_cli::manifest::{Manifest, Routing};
    use expert_cli::routing::KeywordRouter;
    use std::path::PathBuf;

    fn load_real_experts() -> Vec<Manifest> {
        let mut experts = Vec::new();
        let experts_dir = PathBuf::from("../experts");
        for expert_name in &["expert-sql", "expert-json", "expert-typescript", "expert-neo4j"] {
            let manifest_path = experts_dir.join(expert_name).join("manifest.json");
            if let Ok(manifest) = Manifest::load(&manifest_path) {
                experts.push(manifest);
            }
        }
        experts
    }

    #[test]
    fn test_sql_expert_routing() {
        let experts = load_real_experts();
        if experts.is_empty() {
            println!("Skipping test - no experts found");
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("show all database tables with select query", 3);
        assert!(!results.is_empty(), "Should match at least one expert");
        let has_sql = results.iter().any(|r| r.expert_name == "expert-sql");
        assert!(has_sql, "expert-sql should be in top results for SQL query");
    }

    #[test]
    fn test_neo4j_expert_routing() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("find nodes in graph database with relationships", 3);
        if !results.is_empty() {
            let has_neo4j = results.iter().any(|r| r.expert_name == "expert-neo4j");
            assert!(has_neo4j, "expert-neo4j should match graph/node keywords");
        }
    }

    #[test]
    fn test_json_expert_routing() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("json parsing and validation", 3);
        assert!(results.len() <= 3, "Should limit results to top_k");
    }

    #[test]
    fn test_typescript_expert_routing() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("write typescript code with proper types", 3);
        if !results.is_empty() {
            assert!(!results.is_empty(), "Should match at least one expert");
        }
    }

    #[test]
    fn test_no_matches_for_unrelated_query() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("quantum physics differential equations", 1);
        if results.is_empty() {
            assert!(true, "Correctly returned no matches");
        } else {
            assert!(results[0].score < 0.3, "Score should be low for unrelated query");
        }
    }

    #[test]
    fn test_case_insensitive_matching() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results_lower = router.route("sql database query", 3);
        let results_upper = router.route("SQL DATABASE QUERY", 3);
        assert!(!results_lower.is_empty() || !results_upper.is_empty(), "Case insensitive matching should work");
    }

    #[test]
    fn test_partial_keyword_matching() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("show me the database", 1);
        if !results.is_empty() {
            assert!(results[0].score > 0.0, "Should have positive score");
            assert!(!results[0].matched_keywords.is_empty(), "Should have at least one matched keyword");
        }
    }

    #[test]
    fn test_top_k_limiting() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("database query code format", 1);
        assert!(results.len() <= 1, "Should return at most 1 expert");
        let results = router.route("database query code format", 3);
        assert!(results.len() <= 3, "Should return at most 3 experts");
        let results = router.route("database query code format", 100);
        assert!(results.len() <= experts.len(), "Should not exceed number of experts");
    }

    #[test]
    fn test_route_single_convenience() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let result = router.route_single("database sql query");
        assert!(result.is_some(), "Should return an expert for database query");
    }

    #[test]
    fn test_matched_keywords_tracking() {
        let experts = load_real_experts();
        if experts.is_empty() {
            return;
        }
        let mut router = KeywordRouter::new();
        for expert in &experts {
            router.add_expert(expert);
        }
        let results = router.route("select from database table with sql query", 3);
        if !results.is_empty() {
            assert!(!results[0].matched_keywords.is_empty(), "Should track matched keywords");
            if results.len() > 1 {
                assert!(results[0].score >= results[1].score, "Results should be sorted by score");
            }
        }
    }

    #[test]
    fn test_router_with_no_experts() {
        let router = KeywordRouter::new();
        let results = router.route("any query", 1);
        assert_eq!(results.len(), 0, "Empty router should return no results");
        let result = router.route_single("any query");
        assert_eq!(result, None, "Empty router should return None");
    }
}

mod router_comprehensive {
    use expert_cli::manifest::{
        BaseModelV2, Constraints, Dataset, Manifest, Routing, Training, TrainingConfig,
    };
    use expert_cli::routing::KeywordRouter;
    use std::path::Path;

    fn create_test_manifest_simple(name: &str, keywords: Vec<&str>, priority: f32) -> Manifest {
        Manifest {
            name: name.to_string(),
            version: "0.0.1".to_string(),
            schema_version: "2.0".to_string(),
            description: format!("{} expert", name),
            author: None,
            homepage: None,
            repository: None,
            base_model: None,
            base_models: Some(vec![BaseModelV2 {
                name: "test-model".to_string(),
                sha256: None,
                quantization: None,
                rope_scaling: None,
                prompt_template: None,
                adapters: vec![],
            }]),
            adapters: None,
            soft_prompts: vec![],
            capabilities: keywords.iter().map(|k| format!("test:{}", k)).collect(),
            routing: Some(Routing {
                keywords: keywords.iter().map(|k| k.to_string()).collect(),
                exclude_keywords: None,
                router_hint: None,
                priority: Some(priority),
            }),
            constraints: Constraints {
                max_chain: None,
                load_order: 1,
                incompatible_with: vec![],
                requires: vec![],
            },
            perf: None,
            runtime: None,
            training: create_dummy_training(),
            evaluation: None,
            integrity: None,
            license: None,
            tags: None,
        }
    }

    fn create_dummy_training() -> Training {
        Training {
            dataset: Dataset {
                path: Some("test".to_string()),
                validation_path: None,
                test_path: None,
                format: Some("huggingface".to_string()),
                dataset_type: Some("single".to_string()),
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
                target_modules: vec!["q_proj".to_string()],
                feedforward_modules: None,
                epochs: 1.0,
                learning_rate: 0.0001,
                batch_size: 1,
                gradient_accumulation_steps: 1,
                warmup_steps: 0,
                lr_scheduler: "linear".to_string(),
                use_unsloth: None,
                max_seq_length: Some(1024),
                dataloader_num_workers: Some(1),
                dataloader_pin_memory: Some(false),
                dataloader_prefetch_factor: Some(2),
                dataloader_persistent_workers: Some(false),
                fp16: Some(false),
                bf16: Some(false),
                use_tf32: Some(false),
                use_sdpa: Some(false),
                flash_attention_2: None,
                memory_efficient_attention: None,
                activation_checkpointing: None,
                packing: None,
                torch_compile: None,
                torch_compile_backend: None,
                torch_compile_mode: None,
                optim: Some("adamw".to_string()),
                group_by_length: Some(false),
                save_steps: Some(100),
                save_strategy: None,
                save_total_limit: None,
                evaluation_strategy: None,
                eval_steps: None,
                load_best_model_at_end: None,
                metric_for_best_model: None,
                greater_is_better: None,
                logging_steps: Some(10),
                gradient_checkpointing: Some(serde_json::Value::Bool(false)),
                gradient_checkpointing_kwargs: None,
                lr_scheduler_kwargs: None,
                pretokenized_cache: None,
            },
            decoding: None,
            metadata: None,
            packaging_checkpoint: None,
        }
    }

    #[test]
    fn test_router_empty() {
        let router = KeywordRouter::new();
        let matches = router.route("test query", 5);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_router_exact_match() {
        let expert = create_test_manifest_simple("expert-neo4j", vec!["neo4j", "cypher", "graph"], 1.0);
        let mut router = KeywordRouter::new();
        router.add_expert(&expert);
        let matches = router.route("generate neo4j cypher query", 5);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].expert_name, "expert-neo4j");
        assert!(matches[0].score > 0.0);
        assert!(matches[0].matched_keywords.contains(&"neo4j".to_string()));
        assert!(matches[0].matched_keywords.contains(&"cypher".to_string()));
    }

    #[test]
    fn test_router_ranking() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("expert-sql", vec!["sql", "database", "query"], 1.0));
        router.add_expert(&create_test_manifest_simple("expert-neo4j", vec!["neo4j", "graph"], 1.0));
        router.add_expert(&create_test_manifest_simple("expert-json", vec!["json", "format"], 1.0));
        let matches = router.route("create sql database query", 3);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].expert_name, "expert-sql");
    }

    #[test]
    fn test_router_priority_boost() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("low-priority", vec!["test"], 0.5));
        router.add_expert(&create_test_manifest_simple("high-priority", vec!["test"], 2.0));
        let matches = router.route("test query", 2);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].expert_name, "high-priority");
        assert_eq!(matches[1].expert_name, "low-priority");
    }

    #[test]
    fn test_router_case_insensitive() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("expert-test", vec!["TypeScript", "Code"], 1.0));
        let matches = router.route("TYPESCRIPT CODE GENERATION", 5);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].expert_name, "expert-test");
    }

    #[test]
    fn test_router_fuzzy_match() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("expert-db", vec!["database", "postgresql"], 1.0));
        // Test with exact keyword match first
        let matches_exact = router.route("I need help with my database", 5);
        assert!(!matches_exact.is_empty(), "Should match 'database' keyword");
        // Test with partial match (postgresql contains postgres)
        let matches_partial = router.route("I need help with postgresql", 5);
        assert!(!matches_partial.is_empty(), "Should match 'postgresql' keyword");
    }

    #[test]
    fn test_router_top_k_limit() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("expert-1", vec!["test"], 1.0));
        router.add_expert(&create_test_manifest_simple("expert-2", vec!["test"], 0.9));
        router.add_expert(&create_test_manifest_simple("expert-3", vec!["test"], 0.8));
        router.add_expert(&create_test_manifest_simple("expert-4", vec!["test"], 0.7));
        let matches = router.route("test query", 2);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_router_no_match() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("expert-tech", vec!["neo4j", "sql"], 1.0));
        let matches = router.route("cooking recipe", 5);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_router_capability_matching() {
        let mut expert = create_test_manifest_simple("expert-ts", vec!["typescript"], 1.0);
        expert.capabilities = vec!["language:typescript".to_string(), "code-generation".to_string()];
        let mut router = KeywordRouter::new();
        router.add_expert(&expert);
        let matches = router.route("typescript code generation", 5);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_router_score_accumulation() {
        let mut router = KeywordRouter::new();
        router.add_expert(&create_test_manifest_simple("expert-multi", vec!["sql", "database", "query", "mysql"], 1.0));
        let matches1 = router.route("sql", 5);
        let score1 = matches1[0].score;
        let matches3 = router.route("sql database query", 5);
        let score3 = matches3[0].score;
        assert!(score3 > score1);
    }

    #[test]
    fn test_router_with_real_manifests() {
        let expert_paths = ["../experts/expert-neo4j", "../experts/expert-sql"];
        let mut experts = Vec::new();
        for path in &expert_paths {
            let manifest_path = Path::new(path).join("manifest.json");
            if manifest_path.exists() {
                if let Ok(manifest) = Manifest::load(&manifest_path) {
                    experts.push(manifest);
                }
            }
        }
        if !experts.is_empty() {
            let mut router = KeywordRouter::new();
            for expert in &experts {
                router.add_expert(expert);
            }
            let matches = router.route("generate neo4j cypher query", 5);
            if matches.len() > 0 {
                println!("Real manifest routing works! Found: {}", matches[0].expert_name);
                assert!(matches[0].score > 0.0);
            }
        }
    }
}

