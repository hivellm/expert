// Rust-native inference engine using Candle
// Replaces Python inference for 10-50x speedup

pub mod qwen;
pub mod qwen3_model;
pub mod lora;
pub mod generation;
pub mod paged_kv_cache;

pub use qwen::QwenEngine;
pub use lora::{LoraAdapter, AdapterType};
pub use generation::{GenerationConfig, sample_token};
pub use paged_kv_cache::{PagedKVCache, PagedKVCacheConfig, CacheStats};

