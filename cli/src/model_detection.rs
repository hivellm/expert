use crate::error::Error;
use crate::registry::{BaseModelEntry, ExpertRegistry};
use chrono::Utc;
use std::path::PathBuf;

/// Auto-detect installed base models from various locations
pub fn detect_base_models() -> Result<Vec<BaseModelEntry>, Error> {
    let mut models = Vec::new();

    // 1. Check registry first
    if let Ok(registry) = ExpertRegistry::load() {
        models.extend(registry.base_models);
    }

    // 2. Scan standard paths
    let search_paths = get_search_paths();
    for path in search_paths {
        if path.exists() {
            models.extend(scan_models_directory(&path)?);
        }
    }

    // 3. Check HuggingFace cache
    if let Ok(hf_models) = scan_huggingface_cache() {
        models.extend(hf_models);
    }

    // Deduplicate by name
    dedup_models(&mut models);

    Ok(models)
}

/// Get standard model search paths
fn get_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // User home .expert/models
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".expert").join("models"));
    }

    // Current directory models/
    paths.push(PathBuf::from("./models"));
    paths.push(PathBuf::from("../models"));

    // Environment variable
    if let Ok(custom_path) = std::env::var("EXPERT_MODELS_PATH") {
        for path in custom_path.split(':') {
            paths.push(PathBuf::from(path));
        }
    }

    paths
}

/// Scan a directory for HuggingFace models
fn scan_models_directory(dir: &PathBuf) -> Result<Vec<BaseModelEntry>, Error> {
    let mut models = Vec::new();

    if !dir.exists() || !dir.is_dir() {
        return Ok(models);
    }

    for entry in std::fs::read_dir(dir).map_err(|e| Error::Io(e))? {
        let entry = entry.map_err(|e| Error::Io(e))?;
        let path = entry.path();

        if path.is_dir() {
            // Check if it's a HuggingFace model (has config.json)
            let config_path = path.join("config.json");
            if config_path.exists() {
                if let Ok(model) = create_model_entry(&path) {
                    models.push(model);
                }
            }
        }
    }

    Ok(models)
}

/// Scan HuggingFace cache directory
fn scan_huggingface_cache() -> Result<Vec<BaseModelEntry>, Error> {
    let cache_dir = if let Some(home) = dirs::home_dir() {
        home.join(".cache").join("huggingface").join("hub")
    } else {
        return Ok(Vec::new());
    };

    if !cache_dir.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();

    for entry in std::fs::read_dir(&cache_dir).map_err(|e| Error::Io(e))? {
        let entry = entry.map_err(|e| Error::Io(e))?;
        let path = entry.path();

        if path.is_dir() {
            // HF cache uses format: models--org--model
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if dir_name.starts_with("models--") {
                // Check snapshots directory
                let snapshots = path.join("snapshots");
                if snapshots.exists() {
                    // Get latest snapshot
                    if let Ok(mut entries) = std::fs::read_dir(&snapshots) {
                        if let Some(Ok(snapshot)) = entries.next() {
                            let snapshot_path = snapshot.path();
                            if snapshot_path.join("config.json").exists() {
                                if let Ok(model) = create_model_entry(&snapshot_path) {
                                    models.push(model);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(models)
}

/// Create a BaseModelEntry from a model directory
fn create_model_entry(path: &PathBuf) -> Result<BaseModelEntry, Error> {
    let name = extract_model_name(path);
    let size = calculate_dir_size(path)?;

    // Try to read quantization from config
    let quantization = detect_quantization(path);

    Ok(BaseModelEntry {
        name,
        path: path.clone(),
        sha256: None,
        quantization,
        size_bytes: size,
        installed_at: Utc::now(),
        source: "auto-detected".to_string(),
    })
}

/// Extract model name from path
fn extract_model_name(path: &PathBuf) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| {
            // Clean up HF cache format: models--org--model -> org/model
            if s.starts_with("models--") {
                s.strip_prefix("models--").unwrap_or(s).replace("--", "/")
            } else {
                s.to_string()
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Detect quantization type from config
fn detect_quantization(path: &PathBuf) -> Option<String> {
    let config_path = path.join("config.json");

    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            // Check for quantization_config
            if let Some(quant_config) = json.get("quantization_config") {
                if let Some(quant_method) = quant_config.get("quant_method") {
                    return quant_method.as_str().map(|s| s.to_string());
                }
            }

            // Check torch_dtype for hints
            if let Some(dtype) = json.get("torch_dtype") {
                if let Some(dtype_str) = dtype.as_str() {
                    if dtype_str.contains("int4") || dtype_str.contains("4bit") {
                        return Some("int4".to_string());
                    } else if dtype_str.contains("int8") || dtype_str.contains("8bit") {
                        return Some("int8".to_string());
                    }
                }
            }
        }
    }

    None
}

/// Calculate total size of directory
fn calculate_dir_size(path: &PathBuf) -> Result<u64, Error> {
    let mut size = 0u64;

    for entry in std::fs::read_dir(path).map_err(|e| Error::Io(e))? {
        let entry = entry.map_err(|e| Error::Io(e))?;
        let metadata = entry.metadata().map_err(|e| Error::Io(e))?;

        if metadata.is_file() {
            size += metadata.len();
        } else if metadata.is_dir() {
            size += calculate_dir_size(&entry.path())?;
        }
    }

    Ok(size)
}

/// Remove duplicate models (keep first occurrence)
fn dedup_models(models: &mut Vec<BaseModelEntry>) {
    let mut seen = std::collections::HashSet::new();
    models.retain(|m| seen.insert(m.name.clone()));
}

/// Find best matching model for a requirement
pub fn find_compatible_model<'a>(
    required_name: &str,
    models: &'a [BaseModelEntry],
) -> Option<&'a BaseModelEntry> {
    // Exact match
    if let Some(model) = models.iter().find(|m| m.name == required_name) {
        return Some(model);
    }

    // Partial match (e.g., "Qwen3-0.6B" matches "Qwen/Qwen3-0.6B")
    models
        .iter()
        .find(|m| m.name.ends_with(required_name) || m.name.contains(required_name))
}
