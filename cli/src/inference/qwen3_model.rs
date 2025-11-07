// Custom Qwen3 model implementation for Candle
// Based on official Candle Qwen3 implementation

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, Linear, RmsNorm, kv_cache::KvCache, Activation};
use half::f16;
use std::sync::Arc;

/// Repeats a key or value tensor for grouped query attention
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Ok(Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let base = cfg.rope_theta as f32;
        
        // NTK-by-parts scaling for Qwen3 (critical for >32k context)
        // Applies exponential scaling when context exceeds 32768 tokens
        let scaled_base = if max_seq_len > 32768 {
            let beta = 0.25; // Qwen3-specific parameter
            let scale_factor = (max_seq_len as f32 / 32768.0).powf(beta);
            base * scale_factor
        } else {
            base
        };
        
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / scaled_base.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

pub struct Qwen3Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,
}

impl Qwen3Attention {
    fn new(cfg: &Qwen3Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        
        let q_proj = if cfg.attention_bias {
            candle_nn::linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?
        };
        
        let k_proj = if cfg.attention_bias {
            candle_nn::linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?
        };
        
        let v_proj = if cfg.attention_bias {
            candle_nn::linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?
        };
        
        let o_proj = if cfg.attention_bias {
            candle_nn::linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?
        };
        
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        
        let hidden_size = head_dim * cfg.num_attention_heads;
        let kv_cache = KvCache::new(2, 512);
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb: rotary,
            kv_cache,
        })
    }
    
    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        
        // Projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        // Reshape to (B, L, H, D) -> (B, H, L, D)
        let q = q.reshape((b, l, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        
        // Apply q_norm and k_norm per-head
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;
        
        // RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        
        // KV cache
        let k_cont = k.contiguous()?;
        let v_cont = v.contiguous()?;
        let (k, v) = self.kv_cache.append(&k_cont, &v_cont)?;
        
        // GQA repeat_kv
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;
        
        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        
        // Output projection
        Ok(ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)?)
    }
    
    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

pub struct Qwen3MLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub struct Qwen3DecoderLayer {
    pub self_attn: Qwen3Attention,
    pub mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Qwen3DecoderLayer {
    fn new(cfg: &Qwen3Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }
    
    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        Ok((x + h2)?)
    }

    /// Forward for single token (qwen3-rs pattern)
    fn forward_single(&mut self, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        // Convert to tensor with batch and sequence dimensions: [1, 1, hidden_size]
        let device = self.self_attn.q_proj.weight().device();
        let x_tensor = Tensor::new(x, device)?.reshape((1, 1, x.len()))?;

        // Attention with norm
        let h = self.ln1.forward(&x_tensor)?;
        let h = self.self_attn.forward(&h, pos)?;
        let x_after_attn = (&x_tensor + h)?;

        // Feed forward with norm
        let h2 = self.ln2.forward(&x_after_attn)?;
        let h2 = h2.apply(&self.mlp)?;
        let x_final = (&x_after_attn + h2)?;

        // Convert back to vec: squeeze to [hidden_size]
        let x_final = x_final.squeeze(0)?.squeeze(0)?;
        Ok(x_final.to_vec1::<f32>()?)
    }
    
    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    pub layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Qwen3Config,
}

impl Qwen3Model {
    pub fn load(vb: VarBuilder, cfg: &Qwen3Config) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        
        // Load LM head - Qwen3 uses tied embeddings (shares weights with embed_tokens)
        // Since tie_word_embeddings=true, we reuse the embedding weight matrix transposed
        let embed_weight = embed_tokens.embeddings();
        let lm_head = candle_nn::Linear::new(embed_weight.clone(), None);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: cfg.clone(),
        })
    }

    /// Forward pass for a single token - REAL INFERENCE
    pub fn forward_single(&mut self, token: usize, pos: usize, logits_out: &mut [f32]) -> Result<()> {
        let device = self.embed_tokens.embeddings().device();
        
        // 1. Create token tensor with batch and sequence dimensions [1]
        let token_tensor = Tensor::new(&[token as u32], device)?;
        
        // 2. Embed token → [1, hidden_size]
        let mut hidden = self.embed_tokens.forward(&token_tensor)?;
        
        // 3. Add batch and sequence dimensions for transformer layers → [1, 1, hidden_size]
        hidden = hidden.unsqueeze(0)?;
        
        // 4. Pass through all transformer layers
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, pos)?;
        }
        
        // 5. Final layer norm
        hidden = self.norm.forward(&hidden)?;
        
        // 6. LM head projection to vocabulary → [1, 1, vocab_size]
        let logits = hidden.apply(&self.lm_head)?;
        
        // 7. Convert to F32 and flatten to output buffer [vocab_size]
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let logits_vec = logits_f32.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        logits_out.copy_from_slice(&logits_vec);
        
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
    
    /// Get mutable reference to a specific layer for LoRA injection
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> Option<&mut Qwen3DecoderLayer> {
        self.layers.get_mut(layer_idx)
    }
    
    /// Get the number of layers in the model
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Returns the target module names for LoRA adapters (Qwen3 best practices)
    /// 
    /// LoRA should be applied to attention projections and MLP layers.
    /// Do NOT apply to normalization layers (q_norm, k_norm, input_layernorm, post_attention_layernorm)
    /// as this can cause training instability.
    pub fn lora_target_modules() -> Vec<&'static str> {
        vec![
            "q_proj",       // Query projection (attention)
            "k_proj",       // Key projection (attention)
            "v_proj",       // Value projection (attention)
            "o_proj",       // Output projection (attention)
            "gate_proj",    // Gate projection (MLP)
            "up_proj",      // Up projection (MLP)
            "down_proj",    // Down projection (MLP)
        ]
    }
}

