// LoRA/IA³/DoRA adapter loading for Candle
// Loads SafeTensors adapters and applies to model

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum AdapterType {
    LoRA,
    DoRA,
    IA3,
    LoKR,
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
    pub fn from_safetensors_verbose(
        path: &Path,
        adapter_type: AdapterType,
        verbose: bool,
    ) -> Result<Self> {
        if verbose {
            println!(
                "Loading {:?} adapter from: {}",
                adapter_type,
                path.display()
            );
        }

        // Load SafeTensors file
        let adapter_path = if path.is_dir() {
            // Look for adapter_model.safetensors in directory
            let safetensors_path = path.join("adapter_model.safetensors");
            if !safetensors_path.exists() {
                return Err(anyhow!(
                    "adapter_model.safetensors not found in {}",
                    path.display()
                ));
            }
            safetensors_path
        } else {
            path.to_path_buf()
        };

        // Load tensors
        let weights = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;

        if verbose {
            println!("[OK] Loaded {} weight tensors", weights.len());
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
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
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

    /// Apply IA³ to a layer's weight tensor
    /// IA³ multiplies activations by learned scaling vectors
    /// W' = W * scaling_vector (reshaped to match weight dimensions)
    pub fn apply_ia3(&self, layer_name: &str, base_weight: &Tensor) -> Result<Tensor> {
        let ia3_key = format!("{}.ia3_l", layer_name);

        // Check if this layer has IA³ weights
        if !self.weights.contains_key(&ia3_key) {
            // No IA³ for this layer, return original
            return Ok(base_weight.clone());
        }

        let ia3_vector = &self.weights[&ia3_key];

        // IA³ scaling vector needs to be reshaped to match weight dimensions
        // For linear layers: [out_features] -> [out_features, in_features]
        let (out_features, in_features) = base_weight.dims2()?;

        // Reshape IA³ vector to broadcast with weight matrix
        let scaling_matrix = if ia3_vector.dims1()? == out_features {
            // Vector matches output features - expand to [out_features, 1] for broadcasting
            ia3_vector.unsqueeze(1)? // [out_features] -> [out_features, 1]
        } else if ia3_vector.dims1()? == in_features {
            // Vector matches input features - expand to [1, in_features] for broadcasting
            ia3_vector.unsqueeze(0)? // [in_features] -> [1, in_features]
        } else {
            return Err(anyhow!(
                "IA³ vector for layer {} has wrong dimensions. Expected {} or {}, got {}",
                layer_name,
                out_features,
                in_features,
                ia3_vector.dims1()?
            ));
        };

        // Apply scaling: element-wise multiplication
        let scaled_weight = base_weight.mul(&scaling_matrix)?;

        Ok(scaled_weight)
    }

    /// Apply LoKR to a layer's weight tensor
    /// LoKR uses Kronecker products for efficient low-rank adaptation
    pub fn apply_lokr(&self, layer_name: &str, base_weight: &Tensor) -> Result<Tensor> {
        let lokr_a_key = format!("{}.lokr_A", layer_name);
        let lokr_b_key = format!("{}.lokr_B", layer_name);

        // Check if this layer has LoKR weights
        if !self.weights.contains_key(&lokr_a_key) || !self.weights.contains_key(&lokr_b_key) {
            // No LoKR for this layer, return original
            return Ok(base_weight.clone());
        }

        let lokr_a = &self.weights[&lokr_a_key];
        let lokr_b = &self.weights[&lokr_b_key];

        // LoKR uses Kronecker product: W' = W + kron(A, B) * scale
        // For efficiency, we compute this as: W + (A ⊗ B) where ⊗ is Kronecker product
        let kron_product = Self::kronecker_product(lokr_a, lokr_b)?;

        // Scale by alpha/r (same as LoRA)
        let scale = if let (Some(alpha), Some(rank)) = (self.alpha, self.rank) {
            alpha as f64 / rank as f64
        } else {
            1.0
        };

        let scaled_kron = kron_product.affine(scale, 0.0)?;

        // Add to base weight
        let updated_weight = base_weight.add(&scaled_kron)?;

        Ok(updated_weight)
    }

    /// Compute Kronecker product of two matrices
    fn kronecker_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (a_rows, a_cols) = a.dims2()?;
        let (b_rows, b_cols) = b.dims2()?;

        // Reshape A to [a_rows, 1, a_cols, 1]
        let a_expanded = a.unsqueeze(1)?.unsqueeze(3)?;

        // Reshape B to [1, b_rows, 1, b_cols]
        let b_expanded = b.unsqueeze(0)?.unsqueeze(2)?;

        // Element-wise multiplication gives Kronecker product
        let kron = a_expanded.mul(&b_expanded)?;

        // Reshape to final dimensions [a_rows * b_rows, a_cols * b_cols]
        let result = kron.reshape((a_rows * b_rows, a_cols * b_cols))?;

        Ok(result)
    }

    /// Generic apply method that dispatches to the correct adapter type
    pub fn apply_adapter(&self, layer_name: &str, base_weight: &Tensor) -> Result<Tensor> {
        match self.adapter_type {
            AdapterType::LoRA => self.apply_lora(layer_name, base_weight),
            AdapterType::DoRA => self.apply_lora(layer_name, base_weight), // DoRA uses same logic as LoRA for now
            AdapterType::IA3 => self.apply_ia3(layer_name, base_weight),
            AdapterType::LoKR => self.apply_lokr(layer_name, base_weight),
        }
    }

    /// Get adapter size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weights
            .values()
            .map(|t| t.elem_count() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.values().map(|t| t.elem_count()).sum()
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
