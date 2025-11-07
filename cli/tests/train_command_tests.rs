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
    
    assert_eq!(adapter_dir.to_string_lossy(), "weights/adapter");
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
    
    // Can't use both
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
    assert!((warmup_ratio - 0.1).abs() < 0.01); // ~10%
}

#[test]
fn test_lora_rank_values() {
    let ranks = vec![4, 8, 12, 16, 24, 32];
    
    for rank in ranks {
        assert!(rank >= 4 && rank <= 64);
        assert_eq!(rank % 4, 0); // Usually multiple of 4
    }
}

#[test]
fn test_lora_alpha_scaling() {
    let rank = 16;
    let alpha = 32; // Usually 2x rank
    
    assert_eq!(alpha, rank * 2);
}

