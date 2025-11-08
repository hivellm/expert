// Comprehensive router tests
// Tests keyword matching, scoring, and ranking

use expert_cli::manifest::{
    BaseModelV2, Constraints, Dataset, Manifest, Routing, Training, TrainingConfig,
};
use expert_cli::router::KeywordRouter;
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
            path: "test".to_string(),
            format: Some("huggingface".to_string()),
            dataset_type: Some("single".to_string()),
            tasks: None,
            generation: None,
            field_mapping: None,
        },
        config: TrainingConfig {
            method: "sft".to_string(),
            adapter_type: "lora".to_string(),
            rank: 16,
            alpha: 16,
            target_modules: vec!["q_proj".to_string()],
            epochs: 1,
            learning_rate: 0.0001,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            lr_scheduler: "linear".to_string(),
            max_seq_length: Some(1024),
            dataloader_num_workers: Some(1),
            dataloader_pin_memory: Some(false),
            dataloader_prefetch_factor: Some(2),
            dataloader_persistent_workers: Some(false),
            fp16: Some(false),
            bf16: Some(false),
            use_tf32: Some(false),
            use_sdpa: Some(false),
            optim: Some("adamw".to_string()),
            group_by_length: Some(false),
            save_steps: Some(100),
            logging_steps: Some(10),
            gradient_checkpointing: Some(false),
            pretokenized_cache: None,
        },
        metadata: None,
    }
}

#[test]
fn test_router_empty() {
    let router = KeywordRouter { experts: vec![] };
    let matches = router.route("test query", 5);
    assert_eq!(matches.len(), 0);
}

#[test]
fn test_router_exact_match() {
    let experts = vec![create_test_manifest_simple(
        "expert-neo4j",
        vec!["neo4j", "cypher", "graph"],
        1.0,
    )];

    let router = KeywordRouter { experts };
    let matches = router.route("generate neo4j cypher query", 5);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].expert_name, "expert-neo4j");
    assert!(matches[0].score > 0.0);
    assert!(matches[0].matched_keywords.contains(&"neo4j".to_string()));
    assert!(matches[0].matched_keywords.contains(&"cypher".to_string()));
}

#[test]
fn test_router_ranking() {
    let experts = vec![
        create_test_manifest_simple("expert-sql", vec!["sql", "database", "query"], 1.0),
        create_test_manifest_simple("expert-neo4j", vec!["neo4j", "graph"], 1.0),
        create_test_manifest_simple("expert-json", vec!["json", "format"], 1.0),
    ];

    let router = KeywordRouter { experts };
    let matches = router.route("create sql database query", 3);

    assert!(!matches.is_empty());
    // SQL should rank first due to multiple matches
    assert_eq!(matches[0].expert_name, "expert-sql");
}

#[test]
fn test_router_priority_boost() {
    let experts = vec![
        create_test_manifest_simple("low-priority", vec!["test"], 0.5),
        create_test_manifest_simple("high-priority", vec!["test"], 2.0),
    ];

    let router = KeywordRouter { experts };
    let matches = router.route("test query", 2);

    assert_eq!(matches.len(), 2);
    // High priority should rank first
    assert_eq!(matches[0].expert_name, "high-priority");
    assert_eq!(matches[1].expert_name, "low-priority");
}

#[test]
fn test_router_case_insensitive() {
    let experts = vec![create_test_manifest_simple(
        "expert-test",
        vec!["TypeScript", "Code"],
        1.0,
    )];

    let router = KeywordRouter { experts };
    let matches = router.route("TYPESCRIPT CODE GENERATION", 5);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].expert_name, "expert-test");
}

#[test]
fn test_router_fuzzy_match() {
    let experts = vec![create_test_manifest_simple(
        "expert-db",
        vec!["database", "postgresql"],
        1.0,
    )];

    let router = KeywordRouter { experts };
    let matches = router.route("I need help with my postgres db", 5);

    // Should match "database" via "db" fuzzy match
    assert!(!matches.is_empty());
}

#[test]
fn test_router_top_k_limit() {
    let experts = vec![
        create_test_manifest_simple("expert-1", vec!["test"], 1.0),
        create_test_manifest_simple("expert-2", vec!["test"], 0.9),
        create_test_manifest_simple("expert-3", vec!["test"], 0.8),
        create_test_manifest_simple("expert-4", vec!["test"], 0.7),
    ];

    let router = KeywordRouter { experts };
    let matches = router.route("test query", 2);

    assert_eq!(matches.len(), 2); // Should limit to top 2
}

#[test]
fn test_router_no_match() {
    let experts = vec![create_test_manifest_simple(
        "expert-tech",
        vec!["neo4j", "sql"],
        1.0,
    )];

    let router = KeywordRouter { experts };
    let matches = router.route("cooking recipe", 5);

    assert_eq!(matches.len(), 0); // No relevant keywords
}

#[test]
fn test_router_capability_matching() {
    let mut expert = create_test_manifest_simple("expert-ts", vec!["typescript"], 1.0);
    expert.capabilities = vec![
        "language:typescript".to_string(),
        "code-generation".to_string(),
    ];

    let router = KeywordRouter {
        experts: vec![expert],
    };
    let matches = router.route("typescript code generation", 5);

    assert_eq!(matches.len(), 1);
    assert!(matches[0].matched_capabilities.len() > 0);
}

#[test]
fn test_router_score_accumulation() {
    let experts = vec![create_test_manifest_simple(
        "expert-multi",
        vec!["sql", "database", "query", "mysql"],
        1.0,
    )];

    let router = KeywordRouter { experts };

    // Query with 1 keyword
    let matches1 = router.route("sql", 5);
    let score1 = matches1[0].score;

    // Query with 3 keywords
    let matches3 = router.route("sql database query", 5);
    let score3 = matches3[0].score;

    // More keywords should give higher score
    assert!(score3 > score1);
}

#[test]
fn test_router_with_real_manifests() {
    // Try to load real manifests if available
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
        let router = KeywordRouter { experts };
        let matches = router.route("generate neo4j cypher query", 5);

        // Should find neo4j expert if manifest is valid
        if matches.len() > 0 {
            println!(
                "Real manifest routing works! Found: {}",
                matches[0].expert_name
            );
            assert!(matches[0].score > 0.0);
        }
    }
}
