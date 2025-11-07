# Adapter Merging Implementation TODO

## Current Status

**✅ Implemented:**
- Adapter loading from SafeTensors
- Adapter type detection from manifest (LoRA/DoRA/IA3)
- Adapter weight validation (504 tensors, 8.52M params)
- One-shot mode with clean output
- Debug mode showing adapter details

**❌ NOT Implemented:**
- **Actual weight merging** into model layers

## The Problem

`candle_nn::Linear` layers are immutable after creation. We can't modify weights post-loading.

**Current code:**
```rust
// Loads adapter weights ✅
let adapter = LoraAdapter::from_safetensors_verbose(adapter_path, adapter_type, verbose)?;

// Loads base model ✅  
let mut engine = Self::from_local_with_hints(model_path, use_cuda, None, verbose)?;

// Try to apply adapter ❌
engine.apply_adapter(&adapter, verbose)?; // Just validates, doesn't merge
```

## Solutions

### Option 1: Pre-merge Weights (BEST)

Merge adapter into base weights BEFORE creating VarBuilder:

```rust
// 1. Load base model weights as HashMap<String, Tensor>
let base_weights = candle_core::safetensors::load(&base_path, &device)?;

// 2. Load adapter weights
let adapter_weights = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;

// 3. Merge: W' = W + (alpha/r) * B * A
let merged_weights = merge_lora_weights(base_weights, adapter_weights, alpha, rank)?;

// 4. Create VarBuilder from merged weights
let vb = VarBuilder::from_tensors(merged_weights, dtype, &device);

// 5. Build model with merged weights
let model = Qwen3Model::load(vb, &config)?;
```

**Pros:**
- Clean architecture
- No runtime overhead
- Works with any model

**Cons:**
- Need to implement `VarBuilder::from_tensors()` or similar
- More complex loading logic

### Option 2: Runtime Adapter (SLOWER)

Apply adapter during forward pass:

```rust
impl Qwen3Attention {
    fn forward(&mut self, x: &Tensor, adapter: Option<&LoraAdapter>) -> Result<Tensor> {
        let q = x.apply(&self.q_proj)?;
        
        // Apply adapter if present
        let q = if let Some(ada) = adapter {
            ada.apply_lora("q_proj", &q)?
        } else {
            q
        };
        
        // ... rest of forward
    }
}
```

**Pros:**
- Easy to implement
- Can swap adapters dynamically

**Cons:**
- Slower inference (adapter applied every forward)
- Messy code (adapter checks everywhere)

### Option 3: Rebuild Layers (COMPLEX)

Recreate Linear layers with merged weights:

```rust
// Extract weight tensor from Linear (NOT POSSIBLE - no getter)
let q_weight = layer.self_attn.q_proj.weight(); // ❌ Doesn't exist

// Merge weights
let merged_weight = adapter.apply_lora("q_proj", &q_weight)?;

// Create new Linear
let new_q_proj = Linear::new(merged_weight, bias);

// Replace (requires making layers mutable)
layer.self_attn.q_proj = new_q_proj;
```

**Pros:**
- Clean at runtime

**Cons:**
- `Linear` doesn't expose weights
- Can't replace layers easily

## Recommended: Option 1

Implement pre-merge before model creation.

**Implementation steps:**
1. Load base model.safetensors as `HashMap<String, Tensor>`
2. Load adapter_model.safetensors as `HashMap<String, Tensor>`
3. For each layer, merge:
   ```
   W_q' = W_q + (alpha/r) * matmul(lora_B.q_proj, lora_A.q_proj)
   ```
4. Save merged weights to temp file or pass directly to VarBuilder
5. Build model from merged weights

**Estimated effort:** 2-3 hours

## Alternative: Use Python for Now

Python + transformers + PEFT already handles this perfectly:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path)
# Adapter automatically merged ✅
```

For production, keep using Python inference until Rust merging is ready.

## Testing

Run `scripts/test-deterministic.ps1` after implementing:
- Should show DIFFERENT outputs between base and expert
- Expert should generate correct Cypher/SQL queries

