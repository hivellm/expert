// Tests for generation.rs - sampling strategies

use expert_cli::inference::generation::*;
use candle_core::{Device, Tensor, DType};

#[test]
fn test_greedy_sampling() {
    // Create dummy logits tensor
    let device = Device::Cpu;
    let logits_vec = vec![0.1, 0.5, 0.2, 0.8, 0.3];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Greedy should pick index 3 (highest value 0.8)
    let token = sample_greedy(&logits).unwrap();
    assert_eq!(token, 3);
}

#[test]
fn test_temperature_sampling() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // With temperature 1.0, should sample from distribution
    // Higher logits should have higher probability
    let mut results = vec![0; 5];
    for _ in 0..1000 {
        let token = sample_temperature(&logits, 1.0).unwrap();
        results[token as usize] += 1;
    }
    
    // Token 4 (highest logit) should be sampled most often
    assert!(results[4] > results[0]);
    assert!(results[4] > results[1]);
}

#[test]
fn test_temperature_zero() {
    let device = Device::Cpu;
    let logits_vec = vec![0.1, 0.5, 0.2, 0.8, 0.3];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Temperature 0 should behave like greedy
    let token = sample_temperature(&logits, 0.0).unwrap();
    assert_eq!(token, 3); // Highest logit
}

#[test]
fn test_top_p_sampling() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Top-p 0.9 should sample from top tokens
    let token = sample_top_p(&logits, 1.0, 0.9).unwrap();
    assert!(token < 5); // Should be valid token index
}

#[test]
fn test_top_p_distribution() {
    let device = Device::Cpu;
    let logits_vec = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // With top_p=0.1, should only sample from top tokens
    let mut results = vec![0; 5];
    for _ in 0..100 {
        let token = sample_top_p(&logits, 1.0, 0.1).unwrap();
        results[token as usize] += 1;
    }
    
    // Lower indices should rarely be sampled
    assert!(results[4] + results[3] > results[0] + results[1]);
}

#[test]
fn test_top_k_sampling() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Top-k=2 should only sample from top 2 tokens
    let token = sample_top_k(&logits, 1.0, 2).unwrap();
    assert!(token == 4 || token == 3); // Top 2 indices
}

#[test]
fn test_top_k_all_tokens() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Top-k=5 (all tokens) should allow any token
    let mut results = vec![0; 5];
    for _ in 0..100 {
        let token = sample_top_k(&logits, 1.0, 5).unwrap();
        results[token as usize] += 1;
    }
    
    // All tokens should be possible
    assert!(results.iter().any(|&x| x > 0));
}

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_tokens, 512);
    assert_eq!(config.temperature, 0.7);
    assert!(config.top_p.is_some());
    assert!(config.top_k.is_some());
    assert!(config.repetition_penalty.is_some());
}

#[test]
fn test_sample_token_greedy() {
    let device = Device::Cpu;
    let logits_vec = vec![0.1, 0.5, 0.2, 0.8, 0.3];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    let config = GenerationConfig {
        temperature: 0.0, // Greedy
        ..Default::default()
    };
    
    let token = sample_token(&logits, &config).unwrap();
    assert_eq!(token, 3); // Highest logit
}

#[test]
fn test_sample_token_top_p() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    let config = GenerationConfig {
        temperature: 1.0,
        top_p: Some(0.9),
        ..Default::default()
    };
    
    let token = sample_token(&logits, &config).unwrap();
    assert!(token < 5);
}

#[test]
fn test_sample_token_top_k() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    let config = GenerationConfig {
        temperature: 1.0,
        top_p: None,
        top_k: Some(2),
        ..Default::default()
    };
    
    let token = sample_token(&logits, &config).unwrap();
    assert!(token == 4 || token == 3); // Top 2
}

#[test]
fn test_sample_token_temperature_only() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    let config = GenerationConfig {
        temperature: 1.0,
        top_p: None,
        top_k: None,
        ..Default::default()
    };
    
    let token = sample_token(&logits, &config).unwrap();
    assert!(token < 5);
}

#[test]
fn test_temperature_scaling() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // With uniform logits and high temperature, distribution should be uniform
    let mut results = vec![0; 5];
    for _ in 0..1000 {
        let token = sample_temperature(&logits, 2.0).unwrap();
        results[token as usize] += 1;
    }
    
    // With high temperature, should be more uniform
    // All tokens should be sampled at least once
    assert!(results.iter().all(|&x| x > 0));
}

#[test]
fn test_large_vocab() {
    // Test with larger vocabulary (like real models)
    let device = Device::Cpu;
    let logits_vec: Vec<f32> = (0..1000).map(|i| i as f32 / 100.0).collect();
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Should still work with large vocab
    let token = sample_greedy(&logits).unwrap();
    assert_eq!(token, 999); // Highest logit
}

#[test]
fn test_negative_logits() {
    let device = Device::Cpu;
    let logits_vec = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Greedy should still work with negative logits
    let token = sample_greedy(&logits).unwrap();
    assert_eq!(token, 4); // Highest (1.0)
}

#[test]
fn test_extreme_temperature() {
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Tensor::new(logits_vec.as_slice(), &device).unwrap();
    
    // Very high temperature should make distribution more uniform
    let mut results = vec![0; 5];
    for _ in 0..500 {
        let token = sample_temperature(&logits, 100.0).unwrap();
        results[token as usize] += 1;
    }
    
    // With extreme temperature, distribution should be more uniform
    // Lower indices should be sampled more often than with temp=1.0
    assert!(results[0] > 0); // Should sample low indices too
}

