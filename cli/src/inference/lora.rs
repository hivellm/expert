// LoRA/IA³/DoRA adapter loading for Candle
// Loads SafeTensors adapters and applies to model

use anyhow::{Result, anyhow};
use candle_core::{Tensor, Device};
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum AdapterType {
    LoRA,
    DoRA,
    IA3,
}

#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub adapter_type: AdapterType,
    pub rank: Option<u32>,
    pub alpha: Option<u32>,
    pub target_modules: Vec<String>,
    pub weights: HashMap<String, Tensor>,
}

impl LoraAdapter {
    /// Load LoRA adapter from SafeTensors
    pub fn from_safetensors(path: &Path, adapter_type: AdapterType) -> Result<Self> {
        Self::from_safetensors_verbose(path, adapter_type, true)
    }
    
    /// Load LoRA adapter from SafeTensors with verbose control
    pub fn from_safetensors_verbose(path: &Path, adapter_type: AdapterType, verbose: bool) -> Result<Self> {
        if verbose {
            println!("Loading {:?} adapter from: {}", adapter_type, path.display());
        }
        
        // Load SafeTensors file
        let adapter_path = if path.is_dir() {
            // Look for adapter_model.safetensors in directory
            let safetensors_path = path.join("adapter_model.safetensors");
            if !safetensors_path.exists() {
                return Err(anyhow!("adapter_model.safetensors not found in {}", path.display()));
            }
            safetensors_path
        } else {
            path.to_path_buf()
        };
        
        // Load tensors
        let weights = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;
        
        if verbose {
            println!("✅ Loaded {} weight tensors", weights.len());
        }
        
        // Parse config (if adapter_config.json exists)
        let config_path = if path.is_dir() {
            path.join("adapter_config.json")
        } else {
            path.parent().unwrap().join("adapter_config.json")
        };
        
        let (rank, alpha, target_modules) = if config_path.exists() {
            let config_json = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_json)?;
            
            let r = config["r"].as_u64().map(|v| v as u32);
            let a = config["lora_alpha"].as_u64().map(|v| v as u32);
            let modules = config["target_modules"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            
            (r, a, modules)
        } else {
            (None, None, vec![])
        };
        
        Ok(Self {
            adapter_type,
            rank,
            alpha,
            target_modules,
            weights,
        })
    }
    
    /// Apply LoRA to a layer's weight tensor
    /// W' = W + (alpha/r) * BA
    pub fn apply_lora(&self, layer_name: &str, base_weight: &Tensor) -> Result<Tensor> {
        let lora_a_key = format!("{}.lora_A.weight", layer_name);
        let lora_b_key = format!("{}.lora_B.weight", layer_name);
        
        // Check if this layer has LoRA weights
        if !self.weights.contains_key(&lora_a_key) || !self.weights.contains_key(&lora_b_key) {
            // No LoRA for this layer, return original
            return Ok(base_weight.clone());
        }
        
        let lora_a = &self.weights[&lora_a_key];
        let lora_b = &self.weights[&lora_b_key];
        
        // Compute BA
        let ba = lora_b.matmul(lora_a)?;
        
        // Scale by alpha/r
        let scale = if let (Some(alpha), Some(rank)) = (self.alpha, self.rank) {
            alpha as f64 / rank as f64
        } else {
            1.0
        };
        
        let scaled_ba = ba.affine(scale, 0.0)?;
        
        // Add to base weight
        let updated_weight = base_weight.add(&scaled_ba)?;
        
        Ok(updated_weight)
    }
    
    /// Get adapter size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weights.values()
            .map(|t| t.elem_count() * std::mem::size_of::<f32>())
            .sum()
    }
    
    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.values()
            .map(|t| t.elem_count())
            .sum()
    }
}

/// Load multiple adapters for multi-expert scenarios
pub fn load_adapters(paths: &[&Path]) -> Result<Vec<LoraAdapter>> {
    let mut adapters = Vec::new();
    
    for path in paths {
        // Detect adapter type from config
        let adapter = LoraAdapter::from_safetensors(path, AdapterType::LoRA)?;
        adapters.push(adapter);
    }
    
    Ok(adapters)
}

