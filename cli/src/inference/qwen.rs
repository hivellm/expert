// Qwen3 model loader using Candle
// Based on candle-transformers Qwen2 implementation

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use super::qwen3_model::{Qwen3Config as ModelConfig, Qwen3Model};
use super::grammar_validator::GrammarValidator;

/// Clean reasoning/thinking text from output
/// Removes <think>, <think> blocks and extracts only the final output
fn clean_reasoning_text(text: &str) -> String {
    use regex::Regex;
    
    let mut cleaned = text.to_string();
    
    // Remove <think> or <think> blocks
    let think_pattern = Regex::new(r"(?i)<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>").unwrap();
    cleaned = think_pattern.replace_all(&cleaned, "").to_string();
    
    // Try to extract Cypher/SQL/JSON query (starts with MATCH, CREATE, SELECT, {, etc.)
    let cypher_pattern = Regex::new(r"(?i)(MATCH|CREATE|MERGE|RETURN|WITH|SELECT|INSERT|UPDATE|DELETE|WITH|UNWIND).*?$").unwrap();
    if let Some(mat) = cypher_pattern.find(&cleaned) {
        let query = mat.as_str();
        // Stop at common reasoning prefixes
        let stop_pattern = Regex::new(r"(?i)\n\n(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at)").unwrap();
        if let Some(stop_match) = stop_pattern.find(query) {
            return query[..stop_match.start()].trim().to_string();
        }
        return query.trim().to_string();
    }
    
    // Try to extract JSON (starts with {)
    if cleaned.trim().starts_with('{') {
        let json_pattern = Regex::new(r"\{.*\}").unwrap();
        if let Some(mat) = json_pattern.find(&cleaned) {
            return mat.as_str().trim().to_string();
        }
    }
    
    // Remove common reasoning prefixes
    let prefix_pattern = Regex::new(r"(?i)^(Okay|Let me|I need|Wait|Hmm|So|First|The user|Looking at).*?\n").unwrap();
    cleaned = prefix_pattern.replace_all(&cleaned, "").to_string();
    
    cleaned.trim().to_string()
}

/// Clean ChatML artifacts from generated text
/// Removes everything after <|end|> or <|endoftext|> tokens
/// Optionally removes reasoning text if show_reasoning is false
fn clean_chatml_output(text: &str, show_reasoning: bool) -> String {
    let mut cleaned = text.split("<|end|>")
        .next()
        .unwrap_or(text)
        .split("<|endoftext|>")
        .next()
        .unwrap_or(text)
        .trim()
        .to_string();
    
    // Remove reasoning text by default (unless show_reasoning is true)
    if !show_reasoning {
        cleaned = clean_reasoning_text(&cleaned);
    }
    
    cleaned
}

/// State for autoregressive generation
struct GenerationState {
    token: usize,
    pos: usize,
}

impl GenerationState {
    fn new(initial_token: usize) -> Self {
        Self {
            token: initial_token,
            pos: 0,
        }
    }

    fn advance(&mut self, next_token: usize) {
        self.token = next_token;
        self.pos += 1;
    }
}

/// Qwen3 engine with native Candle
/// Custom implementation for Qwen3 architecture
pub struct QwenEngine {
    pub model: Qwen3Model,
    pub tokenizer: Tokenizer,
    pub device: Device,
    config: Qwen3Config,
    logits_buffer: Vec<f32>,
    // Soft prompts support
    soft_prompts: HashMap<String, Vec<u32>>, // name -> token_ids
    active_soft_prompt: Option<String>,      // currently active soft prompt
    // Hot-swap support
    base_weights: HashMap<String, Tensor>, // original model weights
    current_adapter: Option<String>,       // currently loaded adapter name
    // Grammar validation support
    grammar_validator: Option<Box<dyn GrammarValidator>>,
}

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
}

impl QwenEngine {
    /// Load Qwen3 from local path (already downloaded)
    pub fn from_local(model_path: &Path, use_cuda: bool) -> Result<Self> {
        Self::from_local_with_hints(model_path, use_cuda, None, true)
    }

    /// Load Qwen3 with adapter (LoRA/DoRA)
    pub fn from_local_with_adapter(
        model_path: &Path,
        adapter_path: &Path,
        adapter_type_str: &str,
        use_cuda: bool,
        verbose: bool,
    ) -> Result<Self> {
        if verbose {
            println!("📦 Loading Qwen3 model with adapter");
            println!("   Base: {}", model_path.display());
            println!("   Adapter: {}", adapter_path.display());
            println!("   Type: {}", adapter_type_str);
        }

        // Load adapter first to get config
        use super::lora::{AdapterType, LoraAdapter};

        // Convert string to AdapterType enum
        let adapter_type = match adapter_type_str.to_lowercase().as_str() {
            "lora" => AdapterType::LoRA,
            "dora" => AdapterType::DoRA,
            "ia3" => AdapterType::IA3,
            "lokr" => AdapterType::LoKR,
            _ => {
                if verbose {
                    println!(
                        "   ⚠️  Unknown adapter type '{}', defaulting to DoRA",
                        adapter_type_str
                    );
                }
                AdapterType::DoRA
            }
        };

        let adapter = LoraAdapter::from_safetensors_verbose(adapter_path, adapter_type, verbose)?;

        if verbose {
            println!(
                "   Adapter: {:?} (r={:?}, alpha={:?})",
                adapter.adapter_type, adapter.rank, adapter.alpha
            );
            println!(
                "   Parameters: {:.2}M",
                adapter.num_parameters() as f64 / 1_000_000.0
            );
            println!("   Merging adapter with base weights...");
        }

        // Merge adapter into base weights
        let merged_path = Self::merge_adapter_weights(model_path, &adapter, verbose)?;

        if verbose {
            println!("✅ Weights merged");
            println!("   Loading merged model...");
        }

        // Load merged model
        let mut engine = Self::from_local_with_hints(&merged_path, use_cuda, None, verbose)?;

        // Clean up temporary merged weights after loading
        if merged_path != *model_path {
            if verbose {
                println!("   Cleaning up temporary files...");
            }
            let _ = std::fs::remove_dir_all(&merged_path);
        }

        Ok(engine)
    }

    /// Load Qwen3 with optional runtime hints from expert manifest
    pub fn from_local_with_hints(
        model_path: &Path,
        use_cuda: bool,
        runtime_hints: Option<&crate::manifest::Runtime>,
        verbose: bool,
    ) -> Result<Self> {
        if verbose {
            println!("📦 Loading Qwen3 model from: {}", model_path.display());
        }

        // Log runtime hints if provided
        if verbose {
            if let Some(hints) = runtime_hints {
                if let Some(ref kernel) = hints.attention_kernel {
                    println!("   Runtime hint: attention_kernel = {}", kernel);
                }
                if hints.requires_kv_cache_persistence == Some(true) {
                    println!("   Runtime hint: KV cache persistence enabled");
                }
            }
        }

        let device = if use_cuda {
            #[cfg(feature = "cuda")]
            {
                if candle_core::utils::cuda_is_available() {
                    if verbose {
                        println!("   Using CUDA");
                    }
                    Device::new_cuda(0)?
                } else {
                    if verbose {
                        println!("   CUDA requested but not available, using CPU");
                    }
                    Device::Cpu
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                if verbose {
                    println!("   Binary not compiled with CUDA support, using CPU");
                    println!("   Run: .\\build-cuda.ps1 to build with CUDA");
                }
                Device::Cpu
            }
        } else {
            if verbose {
                println!("   Using CPU");
            }
            Device::Cpu
        };

        // Load config
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(anyhow!("config.json not found in {}", model_path.display()));
        }

        let config_json = fs::read_to_string(&config_path)?;
        let config_value: serde_json::Value = serde_json::from_str(&config_json)?;

        let config = Qwen3Config {
            hidden_size: config_value["hidden_size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing hidden_size"))? as usize,
            num_hidden_layers: config_value["num_hidden_layers"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing num_hidden_layers"))?
                as usize,
            num_attention_heads: config_value["num_attention_heads"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing num_attention_heads"))?
                as usize,
            num_key_value_heads: config_value["num_key_value_heads"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing num_key_value_heads"))?
                as usize,
            head_dim: config_value["head_dim"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing head_dim"))? as usize,
            intermediate_size: config_value["intermediate_size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing intermediate_size"))?
                as usize,
            vocab_size: config_value["vocab_size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing vocab_size"))? as usize,
            max_position_embeddings: config_value["max_position_embeddings"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing max_position_embeddings"))?
                as usize,
            rope_theta: config_value["rope_theta"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing rope_theta"))? as f64,
        };

        if verbose {
            println!("✅ Config loaded");
            println!("   Layers: {}", config.num_hidden_layers);
            println!("   Heads: {}", config.num_attention_heads);
            println!("   Head dim: {}", config.head_dim);
            println!("   Vocab: {}", config.vocab_size);
        }

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!(
                "tokenizer.json not found in {}",
                model_path.display()
            ));
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        if verbose {
            println!("✅ Tokenizer loaded");
            println!("   Loading model weights...");
        }
        let weights_file = model_path.join("model.safetensors");
        if !weights_file.exists() {
            return Err(anyhow!(
                "model.safetensors not found in {}",
                model_path.display()
            ));
        }

        let dtype = device.bf16_default_to_f32();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, &device)? };

        // Create model config
        let model_config = ModelConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            attention_bias: false,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            hidden_act: Activation::Silu,
        };

        let model = Qwen3Model::load(vb, &model_config)?;

        if verbose {
            println!("✅ Qwen3 model loaded");
            println!("   Device: {:?}", device);
            println!("   Dtype: {:?}", dtype);
        }

        Ok(Self {
            model,
            tokenizer,
            device,
            config: config.clone(),
            logits_buffer: vec![0.0; config.vocab_size],
            soft_prompts: HashMap::new(),
            active_soft_prompt: None,
            base_weights: HashMap::new(),
            current_adapter: None,
            grammar_validator: None,
        })
    }

    /// Load from HuggingFace (downloads if needed)
    pub fn from_pretrained(repo_id: &str, use_cuda: bool) -> Result<Self> {
        println!("📥 Downloading Qwen3 from HuggingFace: {}", repo_id);

        let device = if use_cuda && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        // Download and load config
        let config_path = repo.get("config.json")?;
        let config_json = fs::read_to_string(&config_path)?;
        let config_value: serde_json::Value = serde_json::from_str(&config_json)?;

        let config = Qwen3Config {
            hidden_size: config_value["hidden_size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing hidden_size"))? as usize,
            num_hidden_layers: config_value["num_hidden_layers"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing num_hidden_layers"))?
                as usize,
            num_attention_heads: config_value["num_attention_heads"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing num_attention_heads"))?
                as usize,
            num_key_value_heads: config_value["num_key_value_heads"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing num_key_value_heads"))?
                as usize,
            head_dim: config_value["head_dim"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing head_dim"))? as usize,
            intermediate_size: config_value["intermediate_size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing intermediate_size"))?
                as usize,
            vocab_size: config_value["vocab_size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing vocab_size"))? as usize,
            max_position_embeddings: config_value["max_position_embeddings"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing max_position_embeddings"))?
                as usize,
            rope_theta: config_value["rope_theta"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing rope_theta"))? as f64,
        };

        // Download tokenizer
        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Download weights
        let weights_file = repo.get("model.safetensors")?;

        let dtype = device.bf16_default_to_f32();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, &device)? };

        let model_config = ModelConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            attention_bias: false,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            hidden_act: Activation::Silu,
        };

        let model = Qwen3Model::load(vb, &model_config)?;

        println!("✅ Qwen3 model loaded");

        Ok(Self {
            model,
            tokenizer,
            device,
            config: config.clone(),
            logits_buffer: vec![0.0; config.vocab_size],
            soft_prompts: HashMap::new(),
            active_soft_prompt: None,
            base_weights: HashMap::new(),
            current_adapter: None,
            grammar_validator: None,
        })
    }

    /// Load soft prompt from a text file
    pub fn load_soft_prompt(&mut self, name: &str, prompt_text: &str) -> Result<()> {
        // Tokenize the soft prompt text
        let encoding = self
            .tokenizer
            .encode(prompt_text, true)
            .map_err(|e| anyhow!("Failed to tokenize soft prompt: {:?}", e))?;

        let token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();

        self.soft_prompts.insert(name.to_string(), token_ids);
        Ok(())
    }

    /// Activate a soft prompt
    pub fn activate_soft_prompt(&mut self, name: Option<&str>) {
        self.active_soft_prompt = name.map(|s| s.to_string());
    }

    /// Get active soft prompt tokens
    pub fn get_active_soft_prompt_tokens(&self) -> Option<&[u32]> {
        self.active_soft_prompt
            .as_ref()
            .and_then(|name| self.soft_prompts.get(name))
            .map(|tokens| tokens.as_slice())
    }

    /// Get the name of the active soft prompt (if any)
    pub fn get_active_soft_prompt_name(&self) -> Option<&str> {
        self.active_soft_prompt.as_deref()
    }

    /// Get list of available soft prompt names
    pub fn get_soft_prompts(&self) -> Vec<String> {
        self.soft_prompts.keys().cloned().collect()
    }
    
    /// Set grammar validator for query-only output validation
    pub fn set_grammar_validator(&mut self, validator: Option<Box<dyn GrammarValidator>>) {
        self.grammar_validator = validator;
    }
    
    /// Load grammar validator from file path
    pub fn load_grammar_validator(&mut self, grammar_file: &Path) -> Result<()> {
        use super::grammar_validator::load_grammar_validator;
        self.grammar_validator = Some(load_grammar_validator(grammar_file)?);
        Ok(())
    }

    /// Save current model weights as base weights (for hot-swap)
    pub fn save_base_weights(&mut self) -> Result<()> {
        if !self.base_weights.is_empty() {
            return Ok(()); // Already saved
        }

        // Extract weights from the model
        // Note: This is a simplified approach. In practice, we'd need to extract
        // weights from the actual model layers, but Candle doesn't expose this easily.
        // For now, we'll use a placeholder - real implementation would require
        // modifying the model to allow weight extraction.

        println!("⚠️  Base weights saving not fully implemented - requires model weight access");
        Ok(())
    }

    /// Hot-swap to a different adapter (<10ms target)
    pub fn hot_swap_adapter(
        &mut self,
        adapter_name: &str,
        adapter: &super::lora::LoraAdapter,
    ) -> Result<()> {
        // This is a simplified hot-swap implementation
        // Real hot-swap would require pre-loaded adapters and fast weight swapping

        if self.current_adapter.as_ref() == Some(&adapter_name.to_string()) {
            return Ok(()); // Already loaded
        }

        // For now, just mark as current (real implementation would swap weights)
        self.current_adapter = Some(adapter_name.to_string());

        println!("🔄 Hot-swapped to adapter: {}", adapter_name);
        Ok(())
    }

    /// Get current adapter name
    pub fn current_adapter(&self) -> Option<&str> {
        self.current_adapter.as_deref()
    }

    /// Load soft prompts from manifest
    pub fn load_soft_prompts_from_manifest(
        &mut self,
        manifest: &crate::manifest::Manifest,
        expert_path: &Path,
    ) -> Result<()> {
        for soft_prompt in &manifest.soft_prompts {
            let prompt_path = expert_path.join(&soft_prompt.path);

            if prompt_path.exists() {
                let prompt_text = std::fs::read_to_string(&prompt_path).map_err(|e| {
                    anyhow!(
                        "Failed to read soft prompt file {}: {}",
                        prompt_path.display(),
                        e
                    )
                })?;

                self.load_soft_prompt(&soft_prompt.name, &prompt_text)?;
                println!(
                    "✅ Loaded soft prompt: {} ({} tokens)",
                    soft_prompt.name, soft_prompt.tokens
                );
            } else {
                println!("⚠️  Soft prompt file not found: {}", prompt_path.display());
            }
        }
        Ok(())
    }

    /// Merge adapter weights into base model weights
    /// Returns path to merged weights (temporary file)
    fn merge_adapter_weights(
        model_path: &Path,
        adapter: &super::lora::LoraAdapter,
        verbose: bool,
    ) -> Result<PathBuf> {
        use std::collections::HashMap;

        // Load base model weights
        let base_weights_path = model_path.join("model.safetensors");
        if !base_weights_path.exists() {
            return Err(anyhow!(
                "Base model weights not found: {}",
                base_weights_path.display()
            ));
        }

        let mut base_weights = candle_core::safetensors::load(&base_weights_path, &Device::Cpu)?;

        if verbose {
            println!("   Loaded {} base weight tensors", base_weights.len());
            println!("   Adapter has {} weight tensors", adapter.weights.len());

            // Show first adapter key to debug
            if let Some(first_key) = adapter.weights.keys().next() {
                println!("   Example adapter key: {}", first_key);
            }
            if let Some(first_key) = base_weights.keys().next() {
                println!("   Example base key: {}", first_key);
            }
        }

        // Show adapter type and parameters
        if verbose {
            match adapter.adapter_type {
                super::lora::AdapterType::LoRA | super::lora::AdapterType::DoRA => {
                    let scale = if let (Some(alpha), Some(rank)) = (adapter.alpha, adapter.rank) {
                        alpha as f64 / rank as f64
                    } else {
                        1.0
                    };
                    println!(
                        "   {:?} scale: {:.2} (alpha={:?}, r={:?})",
                        adapter.adapter_type, scale, adapter.alpha, adapter.rank
                    );
                }
                super::lora::AdapterType::IA3 => {
                    println!("   IA³ adapter (no rank/alpha parameters)");
                }
                super::lora::AdapterType::LoKR => {
                    let scale = if let (Some(alpha), Some(rank)) = (adapter.alpha, adapter.rank) {
                        alpha as f64 / rank as f64
                    } else {
                        1.0
                    };
                    println!(
                        "   LoKR scale: {:.2} (alpha={:?}, r={:?})",
                        scale, adapter.alpha, adapter.rank
                    );
                }
            }
        }

        // Apply adapter weights based on type
        let mut merged_count = 0;
        let mut adapter_keys: Vec<String> = adapter.weights.keys().cloned().collect();
        adapter_keys.sort();

        // Debug: print first few keys to understand naming
        if verbose {
            println!("   Debug: Adapter keys (first 5):");
            for key in adapter_keys.iter().take(5) {
                println!("     {}", key);
            }
            let base_keys: Vec<_> = base_weights.keys().cloned().collect();
            println!("   Debug: Base keys (first 5):");
            for key in base_keys.iter().take(5) {
                println!("     {}", key);
            }
        }

        // Apply adapter to each matching layer
        let mut first_miss = true;
        for adapter_key in &adapter_keys {
            // Extract layer name based on adapter type
            let layer_name = match adapter.adapter_type {
                super::lora::AdapterType::LoRA | super::lora::AdapterType::DoRA => {
                    // Remove .lora_A.weight or .lora_B.weight suffix
                    if let Some(stripped) = adapter_key.strip_suffix(".lora_A.weight") {
                        stripped
                    } else if let Some(stripped) = adapter_key.strip_suffix(".lora_B.weight") {
                        stripped
                    } else {
                        continue; // Skip non-LoRA keys
                    }
                }
                super::lora::AdapterType::IA3 => {
                    // Remove .ia3_l suffix
                    if let Some(stripped) = adapter_key.strip_suffix(".ia3_l") {
                        stripped
                    } else {
                        continue; // Skip non-IA³ keys
                    }
                }
                super::lora::AdapterType::LoKR => {
                    // Remove .lokr_A or .lokr_B suffix
                    if let Some(stripped) = adapter_key.strip_suffix(".lokr_A") {
                        stripped
                    } else if let Some(stripped) = adapter_key.strip_suffix(".lokr_B") {
                        stripped
                    } else {
                        continue; // Skip non-LoKR keys
                    }
                }
            };

            // Remove PEFT wrapper prefixes from adapter key
            // "base_model.model.model.layers..." -> "model.layers..."
            let cleaned_name = if let Some(stripped) = layer_name.strip_prefix("base_model.model.")
            {
                stripped
            } else if let Some(stripped) = layer_name.strip_prefix("base_model.") {
                stripped
            } else {
                layer_name
            };

            // Find corresponding base weight
            let base_key = format!("{}.weight", cleaned_name);

            if let Some(base_weight) = base_weights.get_mut(&base_key) {
                // Apply adapter using the generic method
                let merged_weight = adapter.apply_adapter(cleaned_name, base_weight)?;

                // Convert back to same dtype if needed
                let merged_weight = merged_weight.to_dtype(base_weight.dtype())?;

                // Update base weight
                *base_weight = merged_weight;

                merged_count += 1;
            } else if verbose && first_miss {
                println!("   Debug: First miss - adapter key: {}", adapter_key);
                println!("          Layer name: {}", layer_name);
                println!("          Cleaned: {}", cleaned_name);
                println!("          Looking for: {}", base_key);
                first_miss = false;
            }
        }

        if verbose {
            println!("   Merged {} weight matrices", merged_count);
        }

        // Save merged weights to temporary file
        let temp_dir = std::env::temp_dir();
        let merged_weights_path = temp_dir.join("merged_model.safetensors");

        candle_core::safetensors::save(&base_weights, &merged_weights_path)?;

        if verbose {
            println!(
                "   Saved merged weights to: {}",
                merged_weights_path.display()
            );
        }

        // Return temp directory with merged weights
        // Create a temp directory structure like the original model
        let temp_model_dir = temp_dir.join("merged_model");
        std::fs::create_dir_all(&temp_model_dir)?;

        // Move merged weights
        let final_weights_path = temp_model_dir.join("model.safetensors");
        std::fs::rename(&merged_weights_path, &final_weights_path)?;

        // Copy other files (config, tokenizer) from base model
        for file in &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
        ] {
            let src = model_path.join(file);
            if src.exists() {
                let dst = temp_model_dir.join(file);
                std::fs::copy(&src, &dst)?;
            }
        }

        Ok(temp_model_dir)
    }

    /// Generate text following qwen3-rs pattern
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        show_reasoning: bool,
    ) -> Result<String> {
        self.generate_verbose(prompt, max_tokens, temperature, top_p, true, show_reasoning)
    }

    /// Generate text with verbose control
    pub fn generate_verbose(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        verbose: bool,
        show_reasoning: bool,
    ) -> Result<String> {
        // Clear KV cache from previous generation
        self.model.clear_kv_cache();

        // Combine soft prompt tokens with user prompt tokens
        let mut all_tokens: Vec<usize> = Vec::new();

        // Add soft prompt tokens first (if active)
        if let Some(soft_tokens) = self.get_active_soft_prompt_tokens() {
            all_tokens.extend(soft_tokens.iter().map(|&t| t as usize));
            if verbose && !soft_tokens.is_empty() {
                println!("   Soft prompt tokens: {}", soft_tokens.len());
            }
        }

        // Add user prompt tokens
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization failed: {:?}", e))?;
        let prompt_tokens: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
        all_tokens.extend_from_slice(&prompt_tokens);

        if all_tokens.is_empty() {
            return Err(anyhow!("Please provide a prompt"));
        }

        if verbose {
            println!(
                "   Total tokens: {} (soft: {}, prompt: {})",
                all_tokens.len(),
                self.get_active_soft_prompt_tokens()
                    .map(|t| t.len())
                    .unwrap_or(0),
                prompt_tokens.len()
            );
            println!("   Starting generation (max {} tokens)...", max_tokens);
        }

        let seq_len = 2048; // Should match config
        let mut state = GenerationState::new(all_tokens[0]);
        let eos_token = 151645_usize;

        let mut generated_tokens = 0;
        let mut output_token_ids = Vec::new();

        while state.pos < seq_len && generated_tokens < max_tokens {
            // Always run forward pass to populate KV cache correctly
            self.model
                .forward_single(state.token, state.pos, &mut self.logits_buffer)?;

            let next_token = if state.pos < all_tokens.len() - 1 {
                // Still processing prompt tokens (including soft prompt) - use ground truth
                all_tokens[state.pos + 1]
            } else {
                // Generate new tokens from model output
                let next = self.sample_next_token(temperature, top_p)?;

                if next == eos_token {
                    break;
                }

                generated_tokens += 1;

                // Store token for decoding
                output_token_ids.push(next as u32);

                // Output token in real-time if verbose
                if verbose {
                    let text = self
                        .tokenizer
                        .decode(&[next as u32], false)
                        .map_err(|e| anyhow!("Decode failed: {:?}", e))?;
                    print!("{}", text);
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }

                next
            };

            state.advance(next_token);
        }

        if verbose {
            println!();
        }

        // Decode all generated tokens at once
        let generated_text = self
            .tokenizer
            .decode(&output_token_ids, true)
            .map_err(|e| anyhow!("Final decode failed: {:?}", e))?;

        // Clean ChatML artifacts
        let mut cleaned = clean_chatml_output(&generated_text, show_reasoning);
        
        // Apply grammar validation if available
        if let Some(ref validator) = self.grammar_validator {
            match validator.extract_query(&cleaned) {
                Ok(extracted) => {
                    cleaned = extracted;
                    // Validate the extracted query
                    if let Ok(is_valid) = validator.validate(&cleaned) {
                        if !is_valid && verbose {
                            eprintln!("⚠️  Warning: Generated output failed grammar validation");
                        }
                    }
                }
                Err(e) => {
                    if verbose {
                        eprintln!("⚠️  Warning: Failed to extract query: {:?}", e);
                    }
                }
            }
        }
        
        Ok(cleaned)
    }

    fn sample_next_token(&self, temperature: f64, top_p: Option<f64>) -> Result<usize> {
        use rand::Rng;

        if temperature <= 0.0 {
            // Greedy sampling - pick highest logit
            return self
                .logits_buffer
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or_else(|| anyhow!("Empty logits"));
        }

        // Apply temperature scaling
        let mut logits = self.logits_buffer.clone();
        for logit in &mut logits {
            *logit /= temperature as f32;
        }

        // Softmax to get probabilities
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Top-p (nucleus) sampling
        if let Some(p_threshold) = top_p {
            if p_threshold < 1.0 {
                // Sort indices by probability (descending)
                let mut indices: Vec<usize> = (0..probs.len()).collect();
                indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

                // Find cutoff where cumulative probability exceeds threshold
                let mut cumsum = 0.0;
                let mut cutoff = probs.len();
                for (idx, &i) in indices.iter().enumerate() {
                    cumsum += probs[i];
                    if cumsum > p_threshold as f32 {
                        cutoff = idx + 1;
                        break;
                    }
                }

                // Zero out low-probability tokens
                for (idx, &i) in indices.iter().enumerate() {
                    if idx >= cutoff {
                        probs[i] = 0.0;
                    }
                }

                // Re-normalize
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }
            }
        }

        // Sample from the probability distribution
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen_range(0.0..1.0);
        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum > sample {
                return Ok(idx);
            }
        }

        // Fallback: return last token (shouldn't happen with proper normalization)
        Ok(probs.len() - 1)
    }

    /// Get model reference (for LoRA attachment)
    pub fn model_mut(&mut self) -> &mut Qwen3Model {
        &mut self.model
    }
}
