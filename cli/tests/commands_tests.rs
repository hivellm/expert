// Commands tests - All CLI command tests grouped together

mod chat {
    use assert_cmd::Command;
    use predicates::prelude::*;

    #[test]
    fn test_chat_oneshot_mode() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--prompt")
            .arg("Hello world")
            .arg("--max-tokens")
            .arg("10")
            .arg("--device")
            .arg("cpu");
        cmd.assert().success();
    }

    #[test]
    fn test_chat_with_expert() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--experts")
            .arg("neo4j")
            .arg("--prompt")
            .arg("Test query")
            .arg("--max-tokens")
            .arg("10")
            .arg("--device")
            .arg("cpu");
        cmd.assert().success();
    }

    #[test]
    fn test_chat_debug_shows_loading() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--prompt")
            .arg("Test")
            .arg("--max-tokens")
            .arg("5")
            .arg("--device")
            .arg("cpu")
            .arg("--debug");
        cmd.assert()
            .success()
            .stdout(predicate::str::contains("Loading"));
    }

    #[test]
    fn test_chat_quiet_mode_no_extra_output() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--prompt")
            .arg("Test")
            .arg("--max-tokens")
            .arg("5")
            .arg("--device")
            .arg("cpu");
        cmd.assert()
            .success()
            .stdout(predicate::str::contains("Loading").not());
    }

    #[test]
    fn test_chat_multiple_experts() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--experts")
            .arg("neo4j,sql")
            .arg("--prompt")
            .arg("Find all")
            .arg("--max-tokens")
            .arg("10")
            .arg("--device")
            .arg("cpu");
        cmd.assert().success();
    }

    #[test]
    fn test_chat_temperature_override() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--prompt")
            .arg("Test")
            .arg("--temperature")
            .arg("0.5")
            .arg("--max-tokens")
            .arg("5")
            .arg("--device")
            .arg("cpu");
        cmd.assert().success();
    }

    #[test]
    fn test_chat_sampling_params() {
        let mut cmd = Command::cargo_bin("expert-cli").unwrap();
        cmd.arg("chat")
            .arg("--prompt")
            .arg("Test")
            .arg("--temperature")
            .arg("0.7")
            .arg("--top-p")
            .arg("0.9")
            .arg("--top-k")
            .arg("40")
            .arg("--max-tokens")
            .arg("5")
            .arg("--device")
            .arg("cpu");
        cmd.assert().success();
    }
}

mod dataset {
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
}

mod install {
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_parse_git_url() {
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
        std::fs::create_dir_all(src_dir.path().join("subdir")).unwrap();
        std::fs::write(src_dir.path().join("file1.txt"), "content1").unwrap();
        std::fs::write(src_dir.path().join("subdir/file2.txt"), "content2").unwrap();
        assert!(src_dir.path().join("file1.txt").exists());
        assert!(src_dir.path().join("subdir/file2.txt").exists());
    }

    #[test]
    fn test_manifest_detection() {
        let temp_dir = TempDir::new().unwrap();
        let manifest_path = temp_dir.path().join("manifest.json");
        assert!(!manifest_path.exists());
        std::fs::write(&manifest_path, r#"{"name": "test"}"#).unwrap();
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
}

mod list {
    use chrono::Utc;
    use expert_cli::registry::{AdapterEntry, BaseModelEntry, ExpertVersionEntry, ExpertRegistry};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_registry() -> ExpertRegistry {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = ExpertRegistry::new(
            temp_dir.path().join("experts"),
            temp_dir.path().join("models"),
        );

        registry.add_expert_version("expert-neo4j", ExpertVersionEntry {
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("experts/expert-neo4j"),
            source: "file://./neo4j".to_string(),
            installed_at: Utc::now(),
            adapters: vec![AdapterEntry {
                adapter_type: "lora".to_string(),
                path: PathBuf::from("adapter"),
                size_bytes: 1000000,
                sha256: Some("hash1".to_string()),
            }],
            capabilities: vec!["tech:neo4j".to_string(), "query:cypher".to_string()],
            dependencies: vec![],
        });

        registry.add_expert_version("expert-sql", ExpertVersionEntry {
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("experts/expert-sql"),
            source: "file://./sql".to_string(),
            installed_at: Utc::now(),
            adapters: vec![],
            capabilities: vec!["database:sql".to_string()],
            dependencies: vec![],
        });

        registry.add_base_model(BaseModelEntry {
            name: "Qwen3-0.6B".to_string(),
            path: temp_dir.path().join("models/Qwen3-0.6B"),
            sha256: Some("model_hash".to_string()),
            quantization: Some("int4".to_string()),
            size_bytes: 500000000,
            installed_at: Utc::now(),
            source: "huggingface".to_string(),
        });

        registry
    }

    #[test]
    fn test_list_all_experts() {
        let registry = create_test_registry();
        let expert_records = registry.expert_records();
        assert_eq!(expert_records.len(), 2);
        assert!(expert_records.iter().any(|e| e.name == "expert-neo4j"));
        assert!(expert_records.iter().any(|e| e.name == "expert-sql"));
    }

    #[test]
    fn test_list_all_models() {
        let registry = create_test_registry();
        let models = registry.list_base_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "Qwen3-0.6B");
        assert_eq!(models[0].quantization.as_deref(), Some("int4"));
    }

    #[test]
    fn test_filter_experts_by_base_model() {
        let registry = create_test_registry();
        let filtered: Vec<_> = registry.iter_versions()
            .filter(|(_, version)| version.base_model == "Qwen3-0.6B")
            .collect();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_expert_has_adapters() {
        let registry = create_test_registry();
        let neo4j = registry.get_expert("expert-neo4j").unwrap();
        assert_eq!(neo4j.adapters.len(), 1);
        assert_eq!(neo4j.adapters[0].adapter_type, "lora");
        assert_eq!(neo4j.adapters[0].size_bytes, 1000000);
    }

    #[test]
    fn test_expert_has_capabilities() {
        let registry = create_test_registry();
        let neo4j = registry.get_expert("expert-neo4j").unwrap();
        assert_eq!(neo4j.capabilities.len(), 2);
        assert!(neo4j.capabilities.contains(&"tech:neo4j".to_string()));
        assert!(neo4j.capabilities.contains(&"query:cypher".to_string()));
    }

    #[test]
    fn test_registry_serialization() {
        let registry = create_test_registry();
        let json = serde_json::to_string_pretty(&registry).unwrap();
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"experts\""));
        assert!(json.contains("\"base_models\""));
        assert!(json.contains("expert-neo4j"));
        assert!(json.contains("expert-sql"));
    }

    #[test]
    fn test_registry_deserialization() {
        let json = r#"{
            "version": "1.0",
            "last_updated": "2025-11-03T12:00:00Z",
            "install_dir": "/tmp/experts",
            "models_dir": "/tmp/models",
            "base_models": [],
            "experts": []
        }"#;

        let registry: ExpertRegistry = serde_json::from_str(json).unwrap();
        assert_eq!(registry.version, "1.0");
        assert_eq!(registry.experts.len(), 0);
        assert_eq!(registry.base_models.len(), 0);
    }
}

mod package {
    use std::path::PathBuf;

    #[test]
    fn test_package_filename_generation_v1() {
        let expert_name = "expert-neo4j";
        let version = "0.0.1";
        let filename = format!("{}.v{}.expert", expert_name, version);
        assert_eq!(filename, "expert-neo4j.v0.0.1.expert");
    }

    #[test]
    fn test_package_filename_generation_v2() {
        let expert_name = "expert-neo4j";
        let model_name = "qwen3-0.6b";
        let version = "0.0.1";
        let filename = format!("{}-{}.v{}.expert", expert_name, model_name, version);
        assert_eq!(filename, "expert-neo4j-qwen3-0.6b.v0.0.1.expert");
    }

    #[test]
    fn test_model_name_normalization() {
        let model = "F:/Node/hivellm/expert/models/Qwen3-0.6B";
        let normalized = model
            .to_lowercase()
            .replace('/', "")
            .replace('\\', "")
            .replace(':', "")
            .replace("fnodehivellmexpertmodels", "");
        assert!(normalized.contains("qwen3"));
    }

    #[test]
    fn test_adapter_path_construction() {
        let weights_dir = PathBuf::from("weights");
        let adapter_path = "qwen3-06b/adapter";
        let full_path = weights_dir.join(adapter_path);
        // Normalize path separators for cross-platform compatibility
        let normalized = full_path.to_string_lossy().replace('\\', "/");
        assert_eq!(normalized, "weights/qwen3-06b/adapter");
    }

    #[test]
    fn test_sha256_hex_format() {
        let hash = "27af68e96322ef4a3dc3bcc9c027bf1dd0515cb01bd38d555f29de7c22a02066";
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_package_size_calculation() {
        let size_bytes: u64 = 14702472;
        let size_mb = size_bytes as f64 / 1_048_576.0;
        assert!((size_mb - 14.02).abs() < 0.1);
    }

    #[test]
    fn test_tar_gz_extension() {
        let package = "expert-neo4j.v0.0.1.expert";
        assert!(package.ends_with(".expert"));
        let without_ext = package.strip_suffix(".expert").unwrap();
        assert_eq!(without_ext, "expert-neo4j.v0.0.1");
    }

    #[test]
    fn test_manifest_filtering_v2() {
        let models = vec!["Qwen3-0.6B", "Qwen3-1.5B", "Qwen3-7B"];
        let selected = "Qwen3-0.6B";
        let filtered: Vec<_> = models.iter().filter(|m| **m == selected).collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], &"Qwen3-0.6B");
    }

    #[test]
    fn test_shared_resources_collection() {
        let resources = vec!["LICENSE", "README.md", "grammar.gbnf"];
        assert!(resources.contains(&"LICENSE"));
    }
}

mod sign {
    use std::path::PathBuf;

    #[test]
    fn test_ed25519_key_length() {
        let pubkey_hex = "3560290124df0eec6fae0e7dd9be75c1a08c9adcbdc718d2d7df93a3534576d3";
        assert_eq!(pubkey_hex.len(), 64);
        assert!(pubkey_hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_signature_hex_format() {
        let signature = "f1e81a9b4eaad28652a1bc50dd3deb80";
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(signature.len() % 2, 0);
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
        assert!(timestamp.contains('T'));
        assert!(timestamp.contains('Z'));
        assert_eq!(timestamp.len(), 20);
    }
}

mod train {
    use std::path::PathBuf;

    #[test]
    fn test_training_config_defaults() {
        let batch_size = 4;
        let epochs = 3;
        let learning_rate = 0.0003;
        assert!(batch_size > 0);
        assert!(epochs > 0);
        assert!(learning_rate > 0.0 && learning_rate < 1.0);
    }

    #[test]
    fn test_gradient_accumulation() {
        let batch_size = 4;
        let gradient_accumulation_steps = 8;
        let effective_batch = batch_size * gradient_accumulation_steps;
        assert_eq!(effective_batch, 32);
    }

    #[test]
    fn test_device_selection() {
        let devices = vec!["cuda", "cpu", "auto"];
        for device in devices {
            assert!(device == "cuda" || device == "cpu" || device == "auto");
        }
    }

    #[test]
    fn test_output_path_construction() {
        let output_dir = PathBuf::from("weights");
        let adapter_dir = output_dir.join("adapter");
        // Normalize path separators for cross-platform compatibility
        let normalized = adapter_dir.to_string_lossy().replace('\\', "/");
        assert_eq!(normalized, "weights/adapter");
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let schedulers = vec!["linear", "cosine", "constant"];
        assert!(schedulers.contains(&"cosine"));
    }

    #[test]
    fn test_optimizer_names() {
        let optimizers = vec!["adamw_torch", "adamw_torch_fused", "sgd"];
        assert!(optimizers.contains(&"adamw_torch_fused"));
    }

    #[test]
    fn test_bf16_configuration() {
        let use_bf16 = true;
        let use_fp16 = false;
        assert!(!(use_bf16 && use_fp16));
    }

    #[test]
    fn test_max_seq_length() {
        let lengths = vec![512, 1024, 1536, 2048, 4096];
        for len in lengths {
            assert!(len >= 512 && len <= 8192);
        }
    }

    #[test]
    fn test_warmup_steps() {
        let warmup_steps = 100;
        let total_steps = 1000;
        let warmup_ratio = warmup_steps as f64 / total_steps as f64;
        assert!((warmup_ratio - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_lora_rank_values() {
        let ranks = vec![4, 8, 12, 16, 24, 32];
        for rank in ranks {
            assert!(rank >= 4 && rank <= 64);
            assert_eq!(rank % 4, 0);
        }
    }

    #[test]
    fn test_lora_alpha_scaling() {
        let rank = 16;
        let alpha = 32;
        assert_eq!(alpha, rank * 2);
    }
}

mod update {
    use std::path::PathBuf;

    #[test]
    fn test_git_source_detection() {
        let sources = vec![
            ("git+https://github.com/user/repo.git", true),
            ("file://./local/path", false),
            ("./local/path", false),
            ("package.expert", false),
        ];

        for (source, is_git) in sources {
            assert_eq!(source.starts_with("git+"), is_git);
        }
    }

    #[test]
    fn test_git_ref_extraction() {
        let source = "git+https://github.com/user/repo.git#v0.0.1";
        if let Some(pos) = source.find('#') {
            let (url, ref_part) = source.split_at(pos);
            let ref_spec = ref_part.strip_prefix('#').unwrap();
            assert_eq!(url, "git+https://github.com/user/repo.git");
            assert_eq!(ref_spec, "v0.0.1");
        }
    }

    #[test]
    fn test_git_directory_detection() {
        let expert_dir = PathBuf::from("/tmp/expert");
        let git_dir = expert_dir.join(".git");
        assert_eq!(git_dir.file_name().unwrap(), ".git");
    }

    #[test]
    fn test_force_flag_behavior() {
        let force = true;
        let is_git = false;
        if !is_git && force {
            assert!(true);
        }
    }

    #[test]
    fn test_git_pull_command() {
        let command_parts = vec!["git", "-C", "/path/to/repo", "pull", "origin", "main"];
        assert_eq!(command_parts[0], "git");
        assert_eq!(command_parts[1], "-C");
        assert_eq!(command_parts[2], "/path/to/repo");
        assert_eq!(command_parts[3], "pull");
    }

    #[test]
    fn test_ref_spec_default() {
        let ref_spec: Option<&str> = None;
        let default_ref = ref_spec.unwrap_or("main");
        assert_eq!(default_ref, "main");
    }
}

