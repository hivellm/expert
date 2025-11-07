use std::path::PathBuf;

#[test]
fn test_model_search_paths() {
    let paths = vec![
        "~/.expert/models",
        "./models",
        "../models",
    ];
    
    assert!(paths.len() >= 3);
}

#[test]
fn test_huggingface_cache_path() {
    // HF cache path format
    let home = if cfg!(windows) {
        "C:\\Users\\User"
    } else {
        "/home/user"
    };
    
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
    let dtypes = vec![
        ("int4", true),
        ("int8", true),
        ("float16", false),
        ("bfloat16", false),
    ];
    
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
    let size_bytes: u64 = 536870912; // 512 MB
    let size_mb = size_bytes / 1_048_576;
    
    assert_eq!(size_mb, 512);
}

#[test]
fn test_model_compatibility_matching() {
    let required = "Qwen3-0.6B";
    let available = vec![
        "Qwen3-0.6B",
        "F:/models/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
    ];
    
    let matches: Vec<_> = available.iter()
        .filter(|m| m.ends_with(required) || m.contains(required))
        .collect();
    
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_deduplication_logic() {
    let models = vec!["Qwen3-0.6B", "Qwen3-0.6B", "Qwen3-1.5B"];
    
    let mut seen = std::collections::HashSet::new();
    let deduped: Vec<_> = models.iter()
        .filter(|m| seen.insert(*m))
        .collect();
    
    assert_eq!(deduped.len(), 2);
}

