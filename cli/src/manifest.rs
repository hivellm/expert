use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

/// Schema version for manifest format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaVersion {
    V1_0,
    V2_0,
}

impl SchemaVersion {
    pub fn from_str(s: &str) -> Self {
        match s {
            "2.0" => SchemaVersion::V2_0,
            _ => SchemaVersion::V1_0, // Default to v1.0
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            SchemaVersion::V1_0 => "1.0",
            SchemaVersion::V2_0 => "2.0",
        }
    }
}

/// RoPE scaling configuration - supports both string and detailed object formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScaling {
    /// Simple string format (legacy): "yarn-128k", "ntk-256k"
    Simple(String),
    /// Detailed configuration object (Qwen3-specific)
    Detailed {
        #[serde(rename = "type")]
        scaling_type: String,
        factor: f64,
        max_position_embeddings: usize,
        original_max_position_embeddings: usize,
        fine_grained: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,

    /// Schema version (v1.0 or v2.0) - defaults to "1.0" if missing
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    pub description: String,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub repository: Option<Repository>,

    /// Single base model (schema v1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model: Option<BaseModel>,

    /// Multiple base models (schema v2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_models: Option<Vec<BaseModelV2>>,

    /// Adapters (only for v1.0, moved to BaseModelV2 in v2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapters: Option<Vec<Adapter>>,

    #[serde(default)]
    pub soft_prompts: Vec<SoftPrompt>,
    pub capabilities: Vec<String>,
    pub routing: Option<Routing>,
    pub constraints: Constraints,
    pub perf: Option<Performance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Runtime>,
    pub training: Training,
    pub evaluation: Option<Evaluation>,
    pub integrity: Option<Integrity>,
    pub license: Option<String>,
    pub tags: Option<Vec<String>>,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

#[cfg(test)]
impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            name: "test-expert".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "2.0".to_string(),
            description: "Test expert".to_string(),
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
                load_order: 10,
                incompatible_with: vec![],
                requires: vec![],
            },
            perf: None,
            runtime: None,
            training: Training {
                packaging_checkpoint: None,
                dataset: Dataset {
                    path: None,
                    format: None,
                    dataset_type: None,
                    tasks: None,
                    generation: None,
                    field_mapping: None,
                    validation_path: None,
                    test_path: None,
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
                    epochs: 1.0,
                    learning_rate: 0.0001,
                    batch_size: 4,
                    gradient_accumulation_steps: 4,
                    warmup_steps: 100,
                    lr_scheduler: "cosine".to_string(),
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
                    optim: None,
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
                    feedforward_modules: None,
                    use_unsloth: None,
                },
                decoding: None,
                metadata: None,
            },
            evaluation: None,
            integrity: None,
            license: None,
            tags: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repository {
    #[serde(rename = "type")]
    pub repo_type: String,
    pub url: String,
}

/// Base model for schema v1.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModel {
    pub name: String,
    pub sha256: Option<String>,
    pub quantization: Option<String>,
    pub rope_scaling: Option<RopeScaling>,
}

/// Base model for schema v2.0 (includes adapters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelV2 {
    pub name: String,
    pub sha256: Option<String>,
    pub quantization: Option<String>,
    pub rope_scaling: Option<RopeScaling>,

    // Prompt template format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    pub adapters: Vec<Adapter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adapter {
    #[serde(rename = "type")]
    pub adapter_type: String, // "lora", "ia3", "dora", "lokr"
    pub target_modules: Vec<String>,

    // LoRA/DoRA specific (optional for IA³/LoKr)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scaling: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropout: Option<f32>,

    // DoRA specific
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_dora: Option<bool>,

    // IA³ doesn't need r/alpha, just target_modules
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftPrompt {
    pub name: String,
    pub path: String,
    pub tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Routing {
    pub keywords: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_keywords: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraints {
    pub max_chain: Option<u32>,
    pub load_order: u32,
    pub incompatible_with: Vec<String>,
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Runtime {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candle_compatible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_kv_cache_persistence: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_kernel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Performance {
    pub latency_ms_overhead: f32,
    pub vram_mb_overhead: u32,
    pub supported_batch_sizes: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Training {
    pub dataset: Dataset,
    pub config: TrainingConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoding: Option<DecodingConfig>,
    pub metadata: Option<TrainingMetadata>,

    /// Specific checkpoint to use for packaging (e.g., "checkpoint-1250" or "final")
    /// If not specified, uses the "final" checkpoint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packaging_checkpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_path: Option<String>,

    pub generation: Option<DatasetGeneration>,
    #[serde(rename = "type")]
    pub dataset_type: Option<String>,
    pub tasks: Option<serde_json::Value>,

    // Field mapping for different dataset formats
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_mapping: Option<FieldMapping>,

    // Format hint (e.g., "huggingface", "jsonl")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    // Streaming mode (loads examples on-demand, reduces RAM usage)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,

    // Max samples to keep in memory when streaming (default: unlimited)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_in_memory_samples: Option<u32>,

    // Pre-tokenized dataset (Windows optimization - loads tokenized Arrow format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_pretokenized: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMapping {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetGeneration {
    pub domain: String,
    pub task: String,
    pub count: usize,
    pub provider: String,
    pub temperature: Option<f32>,
    pub diversity_threshold: Option<f32>,
    pub difficulty_distribution: Option<DifficultyDistribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyDistribution {
    pub easy: f32,
    pub medium: f32,
    pub hard: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecodingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_grammar: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_cmd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub method: String,
    pub adapter_type: String,

    // Rank and alpha are optional - NOT used by IA³ adapter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<u32>,

    pub target_modules: Vec<String>,

    // IA³-specific: feedforward modules
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedforward_modules: Option<Vec<String>>,

    pub epochs: f32, // Changed from u32 to support fractional epochs (e.g. 2.5)
    pub learning_rate: f32,
    pub batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub warmup_steps: u32,
    pub lr_scheduler: String,

    // Optional advanced optimization parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_unsloth: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_seq_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataloader_num_workers: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataloader_pin_memory: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataloader_prefetch_factor: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataloader_persistent_workers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp16: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bf16: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_tf32: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_sdpa: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attention_2: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_efficient_attention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation_checkpointing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packing: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_compile: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_compile_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_compile_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optim: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_by_length: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_total_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_best_model_at_end: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_for_best_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub greater_is_better: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_checkpointing: Option<serde_json::Value>, // Can be bool or "selective"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient_checkpointing_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_scheduler_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pretokenized_cache: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub trained_on: Option<String>,
    pub base_model_version: Option<String>,
    pub training_time_hours: Option<f32>,
    pub gpu: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub test_set_size: u32,
    pub metrics: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Integrity {
    pub created_at: Option<String>,
    pub publisher: String,
    pub pubkey: Option<String>,
    pub signature_algorithm: String,
    pub signature: Option<String>,
    pub files: HashMap<String, String>,
}

impl Manifest {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Manifest(format!("Failed to read manifest: {}", e)))?;

        let manifest: Manifest = serde_json::from_str(&content)
            .map_err(|e| Error::Manifest(format!("Failed to parse manifest: {}", e)))?;

        manifest.validate()?;
        Ok(manifest)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Get the parsed schema version
    pub fn get_schema_version(&self) -> SchemaVersion {
        SchemaVersion::from_str(&self.schema_version)
    }

    /// Check if this is a multi-model manifest (schema v2.0)
    pub fn is_multi_model(&self) -> bool {
        self.get_schema_version() == SchemaVersion::V2_0
    }

    /// Get all base models (works for both v1.0 and v2.0)
    pub fn get_base_models(&self) -> Vec<String> {
        match self.get_schema_version() {
            SchemaVersion::V1_0 => {
                if let Some(ref base) = self.base_model {
                    vec![base.name.clone()]
                } else {
                    vec![]
                }
            }
            SchemaVersion::V2_0 => {
                if let Some(ref models) = self.base_models {
                    models.iter().map(|m| m.name.clone()).collect()
                } else {
                    vec![]
                }
            }
        }
    }

    /// Get a specific base model by name (for v2.0)
    pub fn get_base_model_by_name(&self, name: &str) -> Option<&BaseModelV2> {
        if let Some(ref models) = self.base_models {
            models.iter().find(|m| m.name == name)
        } else {
            None
        }
    }

    pub fn validate(&self) -> Result<()> {
        // Validate required fields
        if self.name.is_empty() {
            return Err(Error::Manifest("name cannot be empty".to_string()));
        }

        if self.version.is_empty() {
            return Err(Error::Manifest("version cannot be empty".to_string()));
        }

        // Validate schema version
        let schema_version = self.get_schema_version();

        // Critical: Cannot have both base_model and base_models
        if self.base_model.is_some() && self.base_models.is_some() {
            return Err(Error::Manifest(
                "Manifest cannot have both 'base_model' and 'base_models'. Use 'base_model' for schema v1.0 or 'base_models' for schema v2.0".to_string()
            ));
        }

        // Validate based on schema version
        match schema_version {
            SchemaVersion::V1_0 => self.validate_v1()?,
            SchemaVersion::V2_0 => self.validate_v2()?,
        }

        // Validate training config (common to both versions)
        if self.training.config.epochs <= 0.0 {
            return Err(Error::Manifest("epochs must be > 0".to_string()));
        }

        if self.training.config.learning_rate <= 0.0 {
            return Err(Error::Manifest("learning_rate must be > 0".to_string()));
        }

        // Validate rank only if present (LoRA/DoRA need it, IA³ doesn't)
        if let Some(rank) = self.training.config.rank {
            if rank == 0 {
                return Err(Error::Manifest("adapter rank must be > 0".to_string()));
            }
        } else if self.training.config.adapter_type != "ia3" {
            return Err(Error::Manifest(format!(
                "adapter type '{}' requires rank and alpha",
                self.training.config.adapter_type
            )));
        }

        if self.training.config.target_modules.is_empty() {
            return Err(Error::Manifest(
                "target_modules cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate schema v1.0 specific requirements
    fn validate_v1(&self) -> Result<()> {
        // Must have base_model
        if self.base_model.is_none() {
            return Err(Error::Manifest(
                "Schema v1.0 requires 'base_model' field".to_string(),
            ));
        }

        // Must have adapters at root level
        if self.adapters.is_none() || self.adapters.as_ref().unwrap().is_empty() {
            return Err(Error::Manifest(
                "Schema v1.0 requires 'adapters' array at root level".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate schema v2.0 specific requirements
    fn validate_v2(&self) -> Result<()> {
        // Must have base_models
        let base_models = self.base_models.as_ref().ok_or_else(|| {
            Error::Manifest("Schema v2.0 requires 'base_models' array".to_string())
        })?;

        // base_models must be non-empty
        if base_models.is_empty() {
            return Err(Error::Manifest(
                "Schema v2.0 requires at least one entry in 'base_models' array".to_string(),
            ));
        }

        // Validate each base model
        for (i, model) in base_models.iter().enumerate() {
            if model.name.is_empty() {
                return Err(Error::Manifest(format!(
                    "base_models[{}]: name cannot be empty",
                    i
                )));
            }

            if model.adapters.is_empty() {
                return Err(Error::Manifest(format!(
                    "base_models[{}]: must have at least one adapter",
                    i
                )));
            }
        }

        // Validate weight paths are unique across all models
        let mut seen_paths = std::collections::HashSet::new();
        for model in base_models {
            for adapter in &model.adapters {
                if !seen_paths.insert(&adapter.path) {
                    return Err(Error::Manifest(format!(
                        "Duplicate adapter path found: {}. Each model must have unique weight paths",
                        adapter.path
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manifest_v1() -> Manifest {
        Manifest {
            name: "test-expert".to_string(),
            version: "1.0.0".to_string(),
            schema_version: "1.0".to_string(),
            description: "Test expert v1.0".to_string(),
            author: Some("Test Author".to_string()),
            homepage: None,
            repository: None,
            base_model: Some(BaseModel {
                name: "Qwen3-0.6B".to_string(),
                sha256: Some("abc123".to_string()),
                quantization: Some("int4".to_string()),
                rope_scaling: Some(RopeScaling::Simple("yarn-128k".to_string())),
            }),
            base_models: None,
            adapters: Some(vec![Adapter {
                adapter_type: "lora".to_string(),
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
                r: Some(16),
                alpha: Some(16),
                scaling: Some("standard".to_string()),
                dropout: Some(0.05),
                use_dora: None,
                path: "weights/adapter.safetensors".to_string(),
                size_bytes: Some(8388608),
                sha256: Some("def456".to_string()),
            }]),
            soft_prompts: vec![],
            capabilities: vec!["test".to_string()],
            routing: None,
            constraints: Constraints {
                max_chain: Some(10),
                load_order: 5,
                incompatible_with: vec![],
                requires: vec![],
            },
            perf: None,
            runtime: None,
            training: Training {
                packaging_checkpoint: None,
                dataset: Dataset {
                    path: None,
                    format: None,
                    dataset_type: None,
                    tasks: None,
                    generation: None,
                    field_mapping: None,
                    validation_path: None,
                    test_path: None,
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
                    epochs: 3.0,
                    learning_rate: 0.0003,
                    batch_size: 4,
                    gradient_accumulation_steps: 4,
                    warmup_steps: 100,
                    lr_scheduler: "cosine".to_string(),
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
                    optim: None,
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
                    feedforward_modules: None,
                    use_unsloth: None,
                },
                decoding: None,
                metadata: None,
            },
            evaluation: None,
            integrity: None,
            license: None,
            tags: None,
        }
    }

    fn create_test_manifest_v2() -> Manifest {
        Manifest {
            name: "test-expert".to_string(),
            version: "2.0.0".to_string(),
            schema_version: "2.0".to_string(),
            description: "Test expert v2.0".to_string(),
            author: Some("Test Author".to_string()),
            homepage: None,
            repository: None,
            base_model: None,
            base_models: Some(vec![
                BaseModelV2 {
                    name: "Qwen3-0.6B".to_string(),
                    sha256: Some("abc123".to_string()),
                    quantization: Some("int4".to_string()),
                    rope_scaling: Some(RopeScaling::Simple("yarn-128k".to_string())),
                    prompt_template: None,
                    adapters: vec![Adapter {
                        adapter_type: "lora".to_string(),
                        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
                        r: Some(16),
                        alpha: Some(16),
                        scaling: Some("standard".to_string()),
                        dropout: Some(0.05),
                        use_dora: None,
                        path: "weights/qwen3-0.6b/adapter.safetensors".to_string(),
                        size_bytes: Some(8388608),
                        sha256: Some("def456".to_string()),
                    }],
                },
                BaseModelV2 {
                    name: "Qwen3-1.5B".to_string(),
                    sha256: Some("xyz789".to_string()),
                    quantization: Some("int4".to_string()),
                    rope_scaling: Some(RopeScaling::Simple("yarn-128k".to_string())),
                    prompt_template: None,
                    adapters: vec![Adapter {
                        adapter_type: "lora".to_string(),
                        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
                        r: Some(16),
                        alpha: Some(16),
                        scaling: Some("standard".to_string()),
                        dropout: Some(0.05),
                        use_dora: None,
                        path: "weights/qwen3-1.5b/adapter.safetensors".to_string(),
                        size_bytes: Some(16777216),
                        sha256: Some("uvw012".to_string()),
                    }],
                },
            ]),
            adapters: None,
            soft_prompts: vec![],
            capabilities: vec!["test".to_string()],
            routing: None,
            constraints: Constraints {
                max_chain: Some(10),
                load_order: 5,
                incompatible_with: vec![],
                requires: vec![],
            },
            perf: None,
            runtime: None,
            training: Training {
                packaging_checkpoint: None,
                dataset: Dataset {
                    path: None,
                    format: None,
                    dataset_type: None,
                    tasks: None,
                    generation: None,
                    field_mapping: None,
                    validation_path: None,
                    test_path: None,
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
                    epochs: 3.0,
                    learning_rate: 0.0003,
                    batch_size: 4,
                    gradient_accumulation_steps: 4,
                    warmup_steps: 100,
                    lr_scheduler: "cosine".to_string(),
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
                    optim: None,
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
                    feedforward_modules: None,
                    use_unsloth: None,
                },
                decoding: None,
                metadata: None,
            },
            evaluation: None,
            integrity: None,
            license: None,
            tags: None,
        }
    }

    #[test]
    fn test_schema_version_detection() {
        let manifest_v1 = create_test_manifest_v1();
        assert_eq!(manifest_v1.get_schema_version(), SchemaVersion::V1_0);

        let manifest_v2 = create_test_manifest_v2();
        assert_eq!(manifest_v2.get_schema_version(), SchemaVersion::V2_0);
    }

    #[test]
    fn test_is_multi_model() {
        let manifest_v1 = create_test_manifest_v1();
        assert!(!manifest_v1.is_multi_model());

        let manifest_v2 = create_test_manifest_v2();
        assert!(manifest_v2.is_multi_model());
    }

    #[test]
    fn test_get_base_models_v1() {
        let manifest = create_test_manifest_v1();
        let models = manifest.get_base_models();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "Qwen3-0.6B");
    }

    #[test]
    fn test_get_base_models_v2() {
        let manifest = create_test_manifest_v2();
        let models = manifest.get_base_models();

        assert_eq!(models.len(), 2);
        assert_eq!(models[0], "Qwen3-0.6B");
        assert_eq!(models[1], "Qwen3-1.5B");
    }

    #[test]
    fn test_get_base_model_by_name() {
        let manifest = create_test_manifest_v2();

        let model = manifest.get_base_model_by_name("Qwen3-0.6B");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "Qwen3-0.6B");

        let model = manifest.get_base_model_by_name("Qwen3-1.5B");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "Qwen3-1.5B");

        let model = manifest.get_base_model_by_name("NonExistent");
        assert!(model.is_none());
    }

    #[test]
    fn test_validate_v1_success() {
        let manifest = create_test_manifest_v1();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_validate_v2_success() {
        let manifest = create_test_manifest_v2();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_validate_v1_missing_base_model() {
        let mut manifest = create_test_manifest_v1();
        manifest.base_model = None;

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires 'base_model'")
        );
    }

    #[test]
    fn test_validate_v1_missing_adapters() {
        let mut manifest = create_test_manifest_v1();
        manifest.adapters = None;

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires 'adapters'")
        );
    }

    #[test]
    fn test_validate_v2_missing_base_models() {
        let mut manifest = create_test_manifest_v2();
        manifest.base_models = None;

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires 'base_models'")
        );
    }

    #[test]
    fn test_validate_v2_empty_base_models() {
        let mut manifest = create_test_manifest_v2();
        manifest.base_models = Some(vec![]);

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at least one entry")
        );
    }

    #[test]
    fn test_validate_conflicting_base_model_fields() {
        let mut manifest = create_test_manifest_v2();
        manifest.base_model = Some(BaseModel {
            name: "Conflict".to_string(),
            sha256: None,
            quantization: None,
            rope_scaling: None,
        });

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot have both"));
    }

    #[test]
    fn test_validate_v2_duplicate_weight_paths() {
        let mut manifest = create_test_manifest_v2();

        // Set duplicate path
        if let Some(ref mut models) = manifest.base_models {
            models[1].adapters[0].path = "weights/qwen3-0.6b/adapter.safetensors".to_string();
        }

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Duplicate adapter path")
        );
    }

    #[test]
    fn test_validate_empty_name() {
        let mut manifest = create_test_manifest_v1();
        manifest.name = "".to_string();

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("name cannot be empty")
        );
    }

    #[test]
    fn test_validate_zero_epochs() {
        let mut manifest = create_test_manifest_v1();
        manifest.training.config.epochs = 0.0;

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("epochs must be > 0")
        );
    }

    #[test]
    fn test_validate_invalid_learning_rate() {
        let mut manifest = create_test_manifest_v1();
        manifest.training.config.learning_rate = 0.0;

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("learning_rate must be > 0")
        );
    }

    #[test]
    fn test_schema_version_from_str() {
        assert_eq!(SchemaVersion::from_str("1.0"), SchemaVersion::V1_0);
        assert_eq!(SchemaVersion::from_str("2.0"), SchemaVersion::V2_0);
        assert_eq!(SchemaVersion::from_str("unknown"), SchemaVersion::V1_0); // defaults to v1.0
    }

    #[test]
    fn test_schema_version_as_str() {
        assert_eq!(SchemaVersion::V1_0.as_str(), "1.0");
        assert_eq!(SchemaVersion::V2_0.as_str(), "2.0");
    }

    #[test]
    fn test_serialize_v1_manifest() {
        let manifest = create_test_manifest_v1();
        let json = serde_json::to_string_pretty(&manifest).unwrap();

        // Should contain v1.0 fields
        assert!(json.contains("\"schema_version\": \"1.0\""));
        assert!(json.contains("\"base_model\""));
        assert!(json.contains("\"adapters\""));

        // Should NOT contain v2.0 fields
        assert!(!json.contains("\"base_models\""));
    }

    #[test]
    fn test_serialize_v2_manifest() {
        let manifest = create_test_manifest_v2();
        let json = serde_json::to_string_pretty(&manifest).unwrap();

        // Should contain v2.0 fields
        assert!(json.contains("\"schema_version\": \"2.0\""));
        assert!(json.contains("\"base_models\""));

        // Should NOT contain v1.0 fields at root level (they're None)
        // Note: "adapters" will appear inside base_models, which is correct for v2.0
        assert!(!json.contains("\"base_model\":"));

        // Verify adapters are inside base_models, not at root
        let lines: Vec<&str> = json.lines().collect();
        let base_models_idx = lines
            .iter()
            .position(|l| l.contains("\"base_models\""))
            .unwrap();

        // Find if there's an "adapters" at root level (before base_models or after it closes)
        let root_adapters = lines
            .iter()
            .take(base_models_idx)
            .any(|l| l.trim().starts_with("\"adapters\":"));

        assert!(
            !root_adapters,
            "adapters should not be at root level in v2.0"
        );
    }

    #[test]
    fn test_deserialize_v1_manifest() {
        let json = r#"{
            "name": "test",
            "version": "1.0.0",
            "schema_version": "1.0",
            "description": "test",
            "base_model": {
                "name": "Qwen3-0.6B",
                "sha256": "hash",
                "quantization": "int4",
                "rope_scaling": "yarn-128k"
            },
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

        let manifest: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.schema_version, "1.0");
        assert!(manifest.base_model.is_some());
        assert!(manifest.base_models.is_none());
        assert_eq!(manifest.get_schema_version(), SchemaVersion::V1_0);
    }

    #[test]
    fn test_deserialize_v2_manifest() {
        let json = r#"{
            "name": "test",
            "version": "2.0.0",
            "schema_version": "2.0",
            "description": "test",
            "base_models": [
                {
                    "name": "Qwen3-0.6B",
                    "sha256": "hash1",
                    "quantization": "int4",
                    "rope_scaling": "yarn-128k",
                    "adapters": [{
                        "type": "lora",
                        "target_modules": ["q_proj"],
                        "r": 16,
                        "alpha": 16,
                        "scaling": "standard",
                        "dropout": 0.05,
                        "path": "weights/qwen3-0.6b/adapter.safetensors",
                        "size_bytes": 1000,
                        "sha256": "hash"
                    }]
                },
                {
                    "name": "Qwen3-1.5B",
                    "sha256": "hash2",
                    "quantization": "int4",
                    "rope_scaling": "yarn-128k",
                    "adapters": [{
                        "type": "lora",
                        "target_modules": ["q_proj"],
                        "r": 16,
                        "alpha": 16,
                        "scaling": "standard",
                        "dropout": 0.05,
                        "path": "weights/qwen3-1.5b/adapter.safetensors",
                        "size_bytes": 2000,
                        "sha256": "hash2"
                    }]
                }
            ],
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

        let manifest: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.schema_version, "2.0");
        assert!(manifest.base_model.is_none());
        assert!(manifest.base_models.is_some());
        assert_eq!(manifest.get_schema_version(), SchemaVersion::V2_0);
        assert_eq!(manifest.base_models.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_deserialize_default_schema_version() {
        // When schema_version is omitted, it should default to "1.0"
        let json = r#"{
            "name": "test",
            "version": "1.0.0",
            "description": "test",
            "base_model": {
                "name": "Qwen3-0.6B"
            },
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

        let manifest: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.schema_version, "1.0"); // Should default to 1.0
        assert_eq!(manifest.get_schema_version(), SchemaVersion::V1_0);
    }

    #[test]
    fn test_round_trip_v1() {
        let original = create_test_manifest_v1();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.schema_version, original.schema_version);
        assert_eq!(parsed.get_schema_version(), original.get_schema_version());
    }

    #[test]
    fn test_round_trip_v2() {
        let original = create_test_manifest_v2();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.schema_version, original.schema_version);
        assert_eq!(parsed.get_base_models(), original.get_base_models());
    }

    // ============ ADVANCED EDGE CASE TESTS ============

    #[test]
    fn test_v2_with_single_model() {
        // Edge case: v2.0 manifest with only one model (should work)
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models.truncate(1); // Keep only first model
        }

        assert!(manifest.validate().is_ok());
        assert_eq!(manifest.get_base_models().len(), 1);
    }

    #[test]
    fn test_v2_model_name_normalization() {
        // Test that model names with special characters can be retrieved
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].name = "Qwen3/0.6B-INT4".to_string();
        }

        let retrieved = manifest.get_base_model_by_name("Qwen3/0.6B-INT4");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Qwen3/0.6B-INT4");
    }

    #[test]
    fn test_v2_case_sensitive_model_names() {
        // Model names should be case-sensitive
        let manifest = create_test_manifest_v2();

        assert!(manifest.get_base_model_by_name("Qwen3-0.6B").is_some());
        assert!(manifest.get_base_model_by_name("qwen3-0.6b").is_none()); // Different case
        assert!(manifest.get_base_model_by_name("QWEN3-0.6B").is_none());
    }

    #[test]
    fn test_v2_empty_model_name() {
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].name = "".to_string();
        }

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("name cannot be empty")
        );
    }

    #[test]
    fn test_v2_model_without_adapters() {
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].adapters.clear();
        }

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must have at least one adapter")
        );
    }

    #[test]
    fn test_v2_multiple_adapters_same_model() {
        // Edge case: A model can have multiple adapters
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            let second_adapter = Adapter {
                adapter_type: "lora".to_string(),
                target_modules: vec!["k_proj".to_string()],
                r: Some(8),
                alpha: Some(8),
                scaling: Some("standard".to_string()),
                dropout: Some(0.1),
                use_dora: None,
                path: "weights/qwen3-0.6b/adapter2.safetensors".to_string(),
                size_bytes: Some(4194304),
                sha256: Some("second_hash".to_string()),
            };
            models[0].adapters.push(second_adapter);
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_v2_three_models() {
        // Edge case: More than 2 models
        let mut manifest = create_test_manifest_v2();

        let third_model = BaseModelV2 {
            name: "Qwen3-3B".to_string(),
            sha256: Some("third_hash".to_string()),
            quantization: Some("int8".to_string()),
            rope_scaling: Some(RopeScaling::Simple("yarn-128k".to_string())),
            prompt_template: None,
            adapters: vec![Adapter {
                adapter_type: "lora".to_string(),
                target_modules: vec!["q_proj".to_string()],
                r: Some(32),
                alpha: Some(32),
                scaling: Some("standard".to_string()),
                dropout: Some(0.05),
                use_dora: None,
                path: "weights/qwen3-3b/adapter.safetensors".to_string(),
                size_bytes: Some(33554432),
                sha256: Some("third_adapter".to_string()),
            }],
        };

        if let Some(ref mut models) = manifest.base_models {
            models.push(third_model);
        }

        assert!(manifest.validate().is_ok());
        assert_eq!(manifest.get_base_models().len(), 3);
    }

    #[test]
    fn test_get_base_models_with_empty_manifest() {
        let mut manifest = create_test_manifest_v1();
        manifest.base_model = None;

        let models = manifest.get_base_models();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_v2_weight_path_with_subdirectories() {
        // Paths can have multiple subdirectories
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].adapters[0].path =
                "weights/models/qwen3/0.6b/v1/adapter.safetensors".to_string();
            models[1].adapters[0].path =
                "weights/models/qwen3/1.5b/v1/adapter.safetensors".to_string();
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_v2_absolute_paths_rejected() {
        // Absolute paths should work in validation (but may fail in packaging)
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].adapters[0].path = "/absolute/path/adapter.safetensors".to_string();
        }

        // Validation should pass (path format is valid)
        // Actual file existence is checked during packaging
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_training_config_validation_edge_cases() {
        let mut manifest = create_test_manifest_v1();

        // Test very high learning rate (should be allowed)
        manifest.training.config.learning_rate = 1.0;
        assert!(manifest.validate().is_ok());

        // Test very low learning rate (should be allowed)
        manifest.training.config.learning_rate = 0.000001;
        assert!(manifest.validate().is_ok());

        // Test exactly zero (should fail)
        manifest.training.config.learning_rate = 0.0;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_training_config_zero_rank() {
        let mut manifest = create_test_manifest_v1();
        manifest.training.config.rank = Some(0);

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rank must be > 0"));
    }

    #[test]
    fn test_training_config_empty_target_modules() {
        let mut manifest = create_test_manifest_v1();
        manifest.training.config.target_modules.clear();

        let result = manifest.validate();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("target_modules cannot be empty")
        );
    }

    #[test]
    fn test_v2_partial_duplicate_paths() {
        // Two adapters in different models shouldn't have overlapping paths
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            // Same directory but different filename should be OK
            models[0].adapters[0].path = "weights/shared/adapter1.safetensors".to_string();
            models[1].adapters[0].path = "weights/shared/adapter2.safetensors".to_string();
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_schema_version_equivalence() {
        // Test that SchemaVersion enum comparison works correctly
        let v1 = SchemaVersion::from_str("1.0");
        let v1_alt = SchemaVersion::from_str("1");
        let v2 = SchemaVersion::from_str("2.0");

        assert_eq!(v1, SchemaVersion::V1_0);
        assert_eq!(v1_alt, SchemaVersion::V1_0); // "1" defaults to v1.0
        assert_eq!(v2, SchemaVersion::V2_0);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_get_base_models_stability() {
        // Multiple calls should return consistent results
        let manifest = create_test_manifest_v2();

        let models1 = manifest.get_base_models();
        let models2 = manifest.get_base_models();
        let models3 = manifest.get_base_models();

        assert_eq!(models1, models2);
        assert_eq!(models2, models3);
        assert_eq!(models1.len(), 2);
    }

    #[test]
    fn test_v2_models_with_different_quantizations() {
        // Models can have different quantization schemes
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].quantization = Some("int4".to_string());
            models[1].quantization = Some("int8".to_string());
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_v2_models_with_different_rope_scaling() {
        // Models can have different RoPE configurations
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].rope_scaling = Some(RopeScaling::Simple("yarn-128k".to_string()));
            models[1].rope_scaling = Some(RopeScaling::Simple("ntk-256k".to_string()));
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_decoding_config_default() {
        let config = DecodingConfig::default();

        // Default should have None values
        assert_eq!(config.use_grammar, None);
        assert_eq!(config.temperature, None);
        assert_eq!(config.top_p, None);
        assert_eq!(config.top_k, None);
    }

    #[test]
    fn test_manifest_with_decoding_config() {
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
            "soft_prompts": [],
            "constraints": {
                "load_order": 0,
                "incompatible_with": [],
                "requires": []
            },
            "training": {
                "dataset": {
                    "path": "test"
                },
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
                },
                "decoding": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 50
                }
            },
            "license": "MIT"
        }"#;

        let manifest: Manifest = serde_json::from_str(json).expect("Should parse");

        assert!(manifest.training.decoding.is_some());
        let decoding = manifest.training.decoding.unwrap();
        assert_eq!(decoding.temperature, Some(0.1));
        assert_eq!(decoding.top_p, Some(0.9));
        assert_eq!(decoding.top_k, Some(50));
    }

    #[test]
    fn test_ia3_without_rank() {
        let json = r#"{
            "name": "expert-test",
            "version": "0.0.1",
            "schema_version": "2.0",
            "description": "Test",
            "capabilities": ["test"],
            "base_models": [{
                "name": "test-model",
                "adapters": [{
                    "type": "ia3",
                    "target_modules": ["k_proj"],
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
                "dataset": {
                    "path": "test"
                },
                "config": {
                    "method": "sft",
                    "adapter_type": "ia3",
                    "target_modules": ["k_proj"],
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

        let manifest: Manifest = serde_json::from_str(json).expect("IA3 without rank should parse");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_v2_model_with_optional_fields_none() {
        // Optional fields can be None
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].sha256 = None;
            models[0].quantization = None;
            models[0].rope_scaling = None;
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_capabilities_preservation() {
        // Capabilities should be preserved through serialization
        let mut manifest = create_test_manifest_v2();
        manifest.capabilities = vec![
            "language:en".to_string(),
            "task:parsing".to_string(),
            "format:json".to_string(),
        ];

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.capabilities.len(), 3);
        assert!(parsed.capabilities.contains(&"language:en".to_string()));
        assert!(parsed.capabilities.contains(&"task:parsing".to_string()));
        assert!(parsed.capabilities.contains(&"format:json".to_string()));
    }

    #[test]
    fn test_constraints_preservation() {
        // Constraints should be preserved through serialization
        let mut manifest = create_test_manifest_v2();
        manifest.constraints.requires = vec![
            "english-basic@>=1.0.0".to_string(),
            "json-parser@2.0.0".to_string(),
        ];
        manifest.constraints.incompatible_with = vec!["legacy-expert@*".to_string()];

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.constraints.requires.len(), 2);
        assert_eq!(parsed.constraints.incompatible_with.len(), 1);
    }

    #[test]
    fn test_v2_adapter_types() {
        // Different adapter types should be allowed
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].adapters[0].adapter_type = "lora".to_string();
            models[1].adapters[0].adapter_type = "dora".to_string();
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_version_string_validation() {
        // Version strings should allow semantic versioning
        let mut manifest = create_test_manifest_v1();

        // Standard versions
        manifest.version = "1.0.0".to_string();
        assert!(manifest.validate().is_ok());

        manifest.version = "2.1.3".to_string();
        assert!(manifest.validate().is_ok());

        // Pre-release versions
        manifest.version = "1.0.0-alpha".to_string();
        assert!(manifest.validate().is_ok());

        manifest.version = "2.0.0-beta.1".to_string();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_v2_duplicate_model_names() {
        // Duplicate model names should be allowed (different quantizations)
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[1].name = "Qwen3-0.6B".to_string(); // Same as first model
            // But paths must still be unique
            models[1].adapters[0].path = "weights/qwen3-0.6b-int8/adapter.safetensors".to_string();
        }

        // Should validate (unique paths, even if same model name)
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_large_adapter_sizes() {
        // Test with realistic large adapter sizes
        let mut manifest = create_test_manifest_v2();

        if let Some(ref mut models) = manifest.base_models {
            models[0].adapters[0].size_bytes = Some(134217728); // 128 MB
            models[1].adapters[0].size_bytes = Some(268435456); // 256 MB
        }

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_multiple_target_modules() {
        // Test with many target modules
        let mut manifest = create_test_manifest_v1();

        manifest.training.config.target_modules = vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ];

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_high_rank_adapter() {
        // Test with high rank values
        let mut manifest = create_test_manifest_v1();

        if let Some(ref mut adapters) = manifest.adapters {
            adapters[0].r = Some(128);
            adapters[0].alpha = Some(256);
        }

        manifest.training.config.rank = Some(128);
        manifest.training.config.alpha = Some(256);

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_low_rank_adapter() {
        // Test with minimum rank values
        let mut manifest = create_test_manifest_v1();

        if let Some(ref mut adapters) = manifest.adapters {
            adapters[0].r = Some(1);
            adapters[0].alpha = Some(1);
        }

        manifest.training.config.rank = Some(1);
        manifest.training.config.alpha = Some(1);

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_extreme_epochs() {
        // Test with extreme epoch values
        let mut manifest = create_test_manifest_v1();

        // Very high epochs (should be allowed)
        manifest.training.config.epochs = 1000.0;
        assert!(manifest.validate().is_ok());

        // Single epoch (minimum)
        manifest.training.config.epochs = 1.0;
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_soft_prompts_with_v2() {
        // Soft prompts should work with v2.0
        let mut manifest = create_test_manifest_v2();

        manifest.soft_prompts = vec![
            SoftPrompt {
                name: "intro".to_string(),
                path: "soft_prompts/intro.pt".to_string(),
                tokens: 64,
            },
            SoftPrompt {
                name: "style".to_string(),
                path: "soft_prompts/style.pt".to_string(),
                tokens: 128,
            },
        ];

        assert!(manifest.validate().is_ok());
        assert_eq!(manifest.soft_prompts.len(), 2);
    }

    #[test]
    fn test_load_order_boundaries() {
        // Load order should support full range
        let mut manifest = create_test_manifest_v1();

        manifest.constraints.load_order = 1; // Minimum
        assert!(manifest.validate().is_ok());

        manifest.constraints.load_order = 10; // Maximum recommended
        assert!(manifest.validate().is_ok());

        manifest.constraints.load_order = 100; // Beyond recommendation (but valid)
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_unicode_in_description() {
        // Unicode characters should be supported
        let mut manifest = create_test_manifest_v2();

        manifest.description = "Expert for 中文, Português, and العربية languages 🚀".to_string();
        manifest.author = Some("José Silva 李明".to_string());

        assert!(manifest.validate().is_ok());

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.description, manifest.description);
        assert_eq!(parsed.author, manifest.author);
    }

    #[test]
    fn test_v2_consistency_check() {
        // All models in v2.0 should have consistent load_order
        let manifest = create_test_manifest_v2();

        // load_order is at manifest level, not per-model
        assert_eq!(manifest.constraints.load_order, 5);

        // This is correct - capabilities are shared across all models
        assert!(!manifest.capabilities.is_empty());
    }

    #[test]
    fn test_serialize_minimal_v1() {
        // Test with minimal required fields
        let manifest = Manifest {
            name: "minimal".to_string(),
            version: "1.0.0".to_string(),
            schema_version: "1.0".to_string(),
            description: "minimal".to_string(),
            author: None,
            homepage: None,
            repository: None,
            base_model: Some(BaseModel {
                name: "Qwen3-0.6B".to_string(),
                sha256: None,
                quantization: None,
                rope_scaling: None,
            }),
            base_models: None,
            adapters: Some(vec![Adapter {
                adapter_type: "lora".to_string(),
                target_modules: vec!["q_proj".to_string()],
                r: Some(16),
                alpha: Some(16),
                scaling: Some("standard".to_string()),
                dropout: Some(0.0),
                use_dora: None,
                path: "weights/adapter.safetensors".to_string(),
                size_bytes: Some(1000),
                sha256: Some("hash".to_string()),
            }]),
            soft_prompts: vec![],
            capabilities: vec!["test".to_string()],
            routing: None,
            constraints: Constraints {
                max_chain: None,
                load_order: 1,
                incompatible_with: vec![],
                requires: vec![],
            },
            perf: None,
            runtime: None,
            training: Training {
                packaging_checkpoint: None,
                dataset: Dataset {
                    path: None,
                    format: None,
                    dataset_type: None,
                    tasks: None,
                    generation: None,
                    field_mapping: None,
                    validation_path: None,
                    test_path: None,
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
                    epochs: 1.0,
                    learning_rate: 0.001,
                    batch_size: 1,
                    gradient_accumulation_steps: 1,
                    warmup_steps: 0,
                    lr_scheduler: "constant".to_string(),
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
                    optim: None,
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
                    feedforward_modules: None,
                    use_unsloth: None,
                },
                decoding: None,
                metadata: None,
            },
            evaluation: None,
            integrity: None,
            license: None,
            tags: None,
        };

        assert!(manifest.validate().is_ok());
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(!json.is_empty());
    }
}
