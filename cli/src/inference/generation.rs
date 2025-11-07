// Generation utilities - sampling, top-p, temperature

use anyhow::Result;
use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: Some(50),
            repetition_penalty: Some(1.1),
        }
    }
}

/// Sample token with temperature
pub fn sample_temperature(logits: &Tensor, temperature: f64) -> Result<u32> {
    let logits = logits.to_vec1::<f32>()?;
    let mut logits: Vec<f32> = logits.iter().map(|&x| x / temperature as f32).collect();
    
    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    logits.iter_mut().for_each(|x| *x = (*x - max).exp());
    let sum: f32 = logits.iter().sum();
    logits.iter_mut().for_each(|x| *x /= sum);
    
    // Sample
    let rand_val: f32 = rand::random();
    let mut cumsum = 0.0;
    for (idx, &prob) in logits.iter().enumerate() {
        cumsum += prob;
        if cumsum >= rand_val {
            return Ok(idx as u32);
        }
    }
    
    Ok((logits.len() - 1) as u32)
}

/// Sample token with top-p (nucleus sampling)
pub fn sample_top_p(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    let logits = logits.to_vec1::<f32>()?;
    let mut logits: Vec<f32> = logits.iter().map(|&x| x / temperature as f32).collect();
    
    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    logits.iter_mut().for_each(|x| *x = (*x - max).exp());
    let sum: f32 = logits.iter().sum();
    logits.iter_mut().for_each(|x| *x /= sum);
    
    // Sort by probability descending
    let mut probs_with_idx: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    probs_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Find top-p threshold
    let mut cumsum = 0.0;
    let mut top_p_idx = 0;
    for (i, (_, prob)) in probs_with_idx.iter().enumerate() {
        cumsum += prob;
        top_p_idx = i;
        if cumsum >= top_p as f32 {
            break;
        }
    }
    
    // Sample from top-p subset
    let top_p_probs = &probs_with_idx[..=top_p_idx];
    let rand_val: f32 = rand::random::<f32>() * cumsum;
    
    let mut cumsum = 0.0;
    for (idx, prob) in top_p_probs {
        cumsum += prob;
        if cumsum >= rand_val {
            return Ok(*idx as u32);
        }
    }
    
    Ok(probs_with_idx[top_p_idx].0 as u32)
}

/// Sample token with top-k filtering
pub fn sample_top_k(logits: &Tensor, temperature: f64, k: usize) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;
    let logits: Vec<f32> = logits_vec.iter().map(|&x| x / temperature as f32).collect();
    
    // Get top-k indices
    let mut logits_with_idx: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    logits_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Zero out non-top-k
    let mut filtered = vec![f32::NEG_INFINITY; logits.len()];
    for (idx, logit) in logits_with_idx.iter().take(k) {
        filtered[*idx] = *logit;
    }
    
    // Softmax on filtered
    let max = filtered.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    filtered.iter_mut().for_each(|x| *x = (*x - max).exp());
    let sum: f32 = filtered.iter().sum();
    filtered.iter_mut().for_each(|x| *x /= sum);
    
    // Sample
    let rand_val: f32 = rand::random();
    let mut cumsum = 0.0;
    for (idx, &prob) in filtered.iter().enumerate() {
        cumsum += prob;
        if cumsum >= rand_val {
            return Ok(idx as u32);
        }
    }
    
    Ok((filtered.len() - 1) as u32)
}

/// Greedy decoding (argmax)
pub fn sample_greedy(logits: &Tensor) -> Result<u32> {
    Ok(logits.argmax(0)?.to_vec0::<u32>()?)
}

/// Sample with all strategies combined
pub fn sample_token(
    logits: &Tensor,
    config: &GenerationConfig,
) -> Result<u32> {
    if config.temperature <= 0.0 {
        return sample_greedy(logits);
    }
    
    if let Some(p) = config.top_p {
        return sample_top_p(logits, config.temperature, p);
    }
    
    if let Some(k) = config.top_k {
        return sample_top_k(logits, config.temperature, k);
    }
    
    sample_temperature(logits, config.temperature)
}

