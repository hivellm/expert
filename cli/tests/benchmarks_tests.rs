// Benchmarks tests - All performance benchmark tests grouped together

mod latency {
    use expert_cli::manifest::{Constraints, Dataset, Manifest, Routing, Training, TrainingConfig};
    use expert_cli::routing::{EmbeddingRouter, KeywordRouter};
    use std::time::Instant;

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
    fn test_keyword_router_latency() {
        let mut router = KeywordRouter::new();
        let mut manifest = create_test_manifest("expert-sql");
        manifest.routing = Some(Routing {
            keywords: vec!["sql".to_string(), "database".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.8),
        });
        router.add_expert(&manifest);
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = router.route("show all database tables", 1);
        }
        let elapsed = start.elapsed();
        let avg_latency_ms = elapsed.as_millis() as f64 / 1000.0;
        assert!(avg_latency_ms < 1.0, "Keyword router too slow: {:.2}ms", avg_latency_ms);
    }

    #[test]
    fn test_embedding_router_latency() {
        let mut router = EmbeddingRouter::new();
        let mut manifest = create_test_manifest("expert-sql");
        manifest.routing = Some(Routing {
            keywords: vec!["sql".to_string(), "database".to_string()],
            exclude_keywords: None,
            router_hint: None,
            priority: Some(0.8),
        });
        router.add_expert(&manifest);
        let start = Instant::now();
        for _ in 0..100 {
            let _ = router.route("show all database tables", 1);
        }
        let elapsed = start.elapsed();
        let avg_latency_ms = elapsed.as_millis() as f64 / 100.0;
        assert!(avg_latency_ms < 10.0, "Embedding router too slow: {:.2}ms", avg_latency_ms);
    }
}

mod vram {
    use candle_core::Device;
    use expert_cli::inference::QwenEngine;
    use expert_cli::inference::paged_kv_cache::{CacheStats, PagedKVCache, PagedKVCacheConfig};

    #[test]
    fn test_paged_kv_cache_memory() {
        let config = PagedKVCacheConfig {
            page_size: 16,
            max_pages: 512,
            num_layers: 28,
            num_heads: 8,
            head_dim: 128,
            dtype: candle_core::DType::BF16,
        };
        let device = Device::Cpu;
        let cache = PagedKVCache::new(config, device).unwrap();
        let stats = cache.stats();
        assert!(stats.memory_mb > 0.0);
        assert!(stats.total_pages == 512);
        assert!(stats.free_pages == 512);
        assert!(stats.used_pages == 0);
    }

    #[test]
    fn test_paged_kv_cache_usage() {
        let config = PagedKVCacheConfig {
            page_size: 16,
            max_pages: 512,
            num_layers: 28,
            num_heads: 8,
            head_dim: 128,
            dtype: candle_core::DType::BF16,
        };
        let device = Device::Cpu;
        let mut cache = PagedKVCache::new(config, device).unwrap();
        let page1 = cache.allocate_page(1).unwrap();
        let page2 = cache.allocate_page(2).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.used_pages, 2);
        assert_eq!(stats.free_pages, 510);
        cache.free_sequence(1);
        let stats = cache.stats();
        assert_eq!(stats.used_pages, 1);
        assert_eq!(stats.free_pages, 511);
    }
}

