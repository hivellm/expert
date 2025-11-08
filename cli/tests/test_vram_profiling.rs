// VRAM profiling tests

use candle_core::Device;
use expert::inference::QwenEngine;
use expert::inference::paged_kv_cache::{CacheStats, PagedKVCache, PagedKVCacheConfig};

#[test]
fn test_paged_kv_cache_memory() {
    let config = PagedKVCacheConfig {
        page_size: 16,
        max_pages: 512,
        num_layers: 28,
        num_heads: 8,
        head_dim: 128,
        dtype: candle_core::DType::BF16,
    };

    let device = Device::Cpu;
    let cache = PagedKVCache::new(config, device).unwrap();

    let stats = cache.stats();

    // Verify memory estimation
    assert!(stats.memory_mb > 0.0);
    assert!(stats.total_pages == 512);
    assert!(stats.free_pages == 512);
    assert!(stats.used_pages == 0);
}

#[test]
fn test_paged_kv_cache_usage() {
    let config = PagedKVCacheConfig {
        page_size: 16,
        max_pages: 512,
        num_layers: 28,
        num_heads: 8,
        head_dim: 128,
        dtype: candle_core::DType::BF16,
    };

    let device = Device::Cpu;
    let mut cache = PagedKVCache::new(config, device).unwrap();

    // Allocate pages
    let page1 = cache.allocate_page(1).unwrap();
    let page2 = cache.allocate_page(2).unwrap();

    let stats = cache.stats();
    assert_eq!(stats.used_pages, 2);
    assert_eq!(stats.free_pages, 510);

    // Free pages
    cache.free_sequence(1);
    let stats = cache.stats();
    assert_eq!(stats.used_pages, 1);
    assert_eq!(stats.free_pages, 511);
}
