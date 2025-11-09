// Rust-native inference engine using Candle
// Replaces Python inference for 10-50x speedup

pub mod generation;
pub mod grammar_validator;
pub mod gbnf_parser;
pub mod lora;
pub mod paged_kv_cache;
pub mod qwen;
pub mod qwen3_model;

pub use generation::{GenerationConfig, sample_token};
pub use grammar_validator::{create_validator, load_grammar_validator, GrammarValidator, GbnfGrammar};
pub use lora::{AdapterType, LoraAdapter};
pub use paged_kv_cache::{CacheStats, PagedKVCache, PagedKVCacheConfig};
pub use qwen::QwenEngine;
