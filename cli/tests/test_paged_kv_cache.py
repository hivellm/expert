#!/usr/bin/env python3
"""
Tests for Paged KV-Cache implementation
"""

import pytest
import torch
from pathlib import Path
import sys

# Add parent and scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from paged_kv_cache import PagedKVCache, PagedKVCacheConfig, create_paged_kv_cache


class TestPagedKVCache:
    """Test paged KV cache functionality"""
    
    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration"""
        return PagedKVCacheConfig(
            page_size=16,
            max_pages=64,  # 1024 tokens max
            num_layers=4,  # Small for testing
            num_heads=8,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    @pytest.fixture
    def cache(self, cache_config):
        """Create cache instance"""
        return PagedKVCache(cache_config)
    
    def test_cache_initialization(self, cache, cache_config):
        """Test cache initializes correctly"""
        assert cache.config == cache_config
        assert len(cache.physical_pages) == 0  # Nothing allocated yet
        assert len(cache.free_pages) == 0
        assert cache.access_counter == 0
    
    def test_page_allocation(self, cache):
        """Test page allocation works"""
        layer_idx = 0
        
        # Allocate first page
        page_idx = cache.allocate_page(layer_idx)
        assert page_idx is not None
        assert page_idx >= 0
        assert page_idx < cache.config.max_pages
        
        # Physical memory should be allocated
        assert layer_idx in cache.physical_pages
        
        # Free pages should be tracked
        assert layer_idx in cache.free_pages
        assert page_idx not in cache.free_pages[layer_idx]
    
    def test_write_and_read_kv(self, cache):
        """Test writing and reading K/V cache"""
        layer_idx = 0
        logical_page = 0
        
        # Create test K/V tensors
        k_cache = torch.randn(
            cache.config.num_heads,
            cache.config.page_size,
            cache.config.head_dim,
            dtype=cache.config.dtype,
            device=cache.device,
        )
        v_cache = torch.randn(
            cache.config.num_heads,
            cache.config.page_size,
            cache.config.head_dim,
            dtype=cache.config.dtype,
            device=cache.device,
        )
        
        # Write
        cache.write_kv(layer_idx, logical_page, k_cache, v_cache)
        
        # Read back
        k_read, v_read = cache.read_kv(layer_idx, [logical_page])
        
        # Verify
        assert k_read is not None
        assert v_read is not None
        assert torch.allclose(k_read, k_cache, rtol=1e-3)
        assert torch.allclose(v_read, v_cache, rtol=1e-3)
    
    def test_multiple_pages(self, cache):
        """Test handling multiple pages"""
        layer_idx = 0
        
        # Write 3 pages
        k_caches = []
        v_caches = []
        
        for logical_page in range(3):
            k = torch.randn(
                cache.config.num_heads,
                cache.config.page_size,
                cache.config.head_dim,
                dtype=cache.config.dtype,
                device=cache.device,
            )
            v = torch.randn(
                cache.config.num_heads,
                cache.config.page_size,
                cache.config.head_dim,
                dtype=cache.config.dtype,
                device=cache.device,
            )
            
            k_caches.append(k)
            v_caches.append(v)
            
            cache.write_kv(layer_idx, logical_page, k, v)
        
        # Read all pages
        k_read, v_read = cache.read_kv(layer_idx, [0, 1, 2])
        
        # Should be concatenated
        expected_k = torch.cat(k_caches, dim=1)  # Along sequence dim
        expected_v = torch.cat(v_caches, dim=1)
        
        assert k_read.shape[1] == cache.config.page_size * 3
        assert torch.allclose(k_read, expected_k, rtol=1e-3)
        assert torch.allclose(v_read, expected_v, rtol=1e-3)
    
    def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full"""
        layer_idx = 0
        
        # Allocate all pages
        for i in range(cache.config.max_pages):
            page_idx = cache.allocate_page(layer_idx)
            assert page_idx is not None
        
        # All pages should be allocated
        initial_count = len([k for k in cache.page_access_time.keys() if k[0] == layer_idx])
        assert initial_count == cache.config.max_pages
        
        # Try to allocate one more - should evict LRU and reuse
        new_page = cache.allocate_page(layer_idx)
        assert new_page is not None
        
        # Should still have max_pages allocated (one evicted, one added)
        final_count = len([k for k in cache.page_access_time.keys() if k[0] == layer_idx])
        assert final_count == cache.config.max_pages
        
        # The new page should be marked as most recently used
        latest_access = max(cache.page_access_time.values())
        assert cache.page_access_time[(layer_idx, new_page)] == latest_access
    
    def test_multi_layer_allocation(self, cache):
        """Test allocation across multiple layers"""
        # Allocate pages for different layers
        for layer_idx in range(cache.config.num_layers):
            page_idx = cache.allocate_page(layer_idx)
            assert page_idx is not None
            assert layer_idx in cache.physical_pages
    
    def test_cache_stats(self, cache):
        """Test cache statistics"""
        # Initially empty
        stats = cache.get_stats()
        assert stats['pages_allocated'] == 0
        assert stats['utilization'] == 0.0
        
        # Allocate some pages and write data
        for layer in range(2):
            for page in range(3):
                k = torch.randn(
                    cache.config.num_heads,
                    cache.config.page_size,
                    cache.config.head_dim,
                    dtype=cache.config.dtype,
                    device=cache.device,
                )
                v = torch.randn(
                    cache.config.num_heads,
                    cache.config.page_size,
                    cache.config.head_dim,
                    dtype=cache.config.dtype,
                    device=cache.device,
                )
                cache.write_kv(layer, page, k, v)
        
        # Check stats
        stats = cache.get_stats()
        assert stats['pages_allocated'] == 6  # 2 layers * 3 pages
        assert stats['num_layers'] == 2
        assert stats['utilization'] > 0
    
    def test_create_from_model_config(self):
        """Test creating cache from model config"""
        model_config = {
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "hidden_size": 1024,
        }
        
        cache = create_paged_kv_cache(model_config, page_size=16, max_context=4096)
        
        assert cache.config.num_layers == 24
        assert cache.config.num_heads == 16
        assert cache.config.head_dim == 64  # 1024 / 16
        assert cache.config.max_pages == 256  # 4096 / 16
    
    def test_memory_efficiency(self, cache):
        """Test that paged cache uses less memory than continuous"""
        # Allocate only 10 pages instead of full 64
        num_pages_needed = 10
        layer_idx = 0
        
        for page in range(num_pages_needed):
            k = torch.randn(
                cache.config.num_heads,
                cache.config.page_size,
                cache.config.head_dim,
                dtype=cache.config.dtype,
                device=cache.device,
            )
            v = torch.randn(
                cache.config.num_heads,
                cache.config.page_size,
                cache.config.head_dim,
                dtype=cache.config.dtype,
                device=cache.device,
            )
            cache.write_kv(layer_idx, page, k, v)
        
        # Physical memory allocated for max_pages, but we only use 10
        stats = cache.get_stats()
        
        # In a real scenario, we'd compare against non-paged implementation
        # For now, just verify we tracked allocation correctly
        assert stats['pages_allocated'] == num_pages_needed
        print(f"Memory efficiency: using {num_pages_needed}/{cache.config.max_pages} pages")


class TestPagedKVCacheIntegration:
    """Integration tests with transformers"""
    
    def test_cache_dtype_compatibility(self):
        """Test cache works with different dtypes"""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            config = PagedKVCacheConfig(
                page_size=8,
                max_pages=32,
                num_layers=2,
                num_heads=4,
                head_dim=32,
                dtype=dtype,
                device="cpu",  # Use CPU for dtype tests
            )
            
            cache = PagedKVCache(config)
            
            # Write and read
            k = torch.randn(4, 8, 32, dtype=dtype)
            v = torch.randn(4, 8, 32, dtype=dtype)
            
            cache.write_kv(0, 0, k, v)
            k_read, v_read = cache.read_kv(0, [0])
            
            assert k_read.dtype == dtype
            assert v_read.dtype == dtype
    
    def test_long_context_simulation(self):
        """Simulate long context with 4k tokens"""
        config = PagedKVCacheConfig(
            page_size=16,
            max_pages=256,  # 4096 tokens
            num_layers=4,
            num_heads=8,
            head_dim=64,
            device="cpu",
        )
        
        cache = PagedKVCache(config)
        
        # Simulate processing 4k tokens (256 pages)
        layer_idx = 0
        num_pages = 200  # Use 3.2k tokens
        
        for page in range(num_pages):
            k = torch.randn(config.num_heads, config.page_size, config.head_dim)
            v = torch.randn(config.num_heads, config.page_size, config.head_dim)
            cache.write_kv(layer_idx, page, k, v)
        
        # Read entire context
        all_pages = list(range(num_pages))
        k_full, v_full = cache.read_kv(layer_idx, all_pages)
        
        # Verify shape
        expected_seq_len = num_pages * config.page_size
        assert k_full.shape == (config.num_heads, expected_seq_len, config.head_dim)
        assert v_full.shape == (config.num_heads, expected_seq_len, config.head_dim)
        
        print(f"Long context test: {expected_seq_len} tokens across {num_pages} pages")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

