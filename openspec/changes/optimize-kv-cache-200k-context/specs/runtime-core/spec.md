# Runtime Core - KV Cache Optimization

## MODIFIED Requirements

### Requirement: Paged KV Cache Memory Allocation

The inference runtime SHALL allocate KV cache memory incrementally on-demand rather than pre-allocating for maximum sequence length.

#### Scenario: Short sequence usage
- **WHEN** processing a 100-token sequence with max_seq_len=200k
- **THEN** only allocate pages for ~7 blocks (100/16 tokens per block)
- **AND** memory usage SHALL scale with actual tokens processed
- **AND** memory footprint SHALL be <50 MB for 100 tokens

#### Scenario: Long context prefill  
- **WHEN** processing a 200k-token input
- **THEN** allocate blocks incrementally during chunked prefill
- **AND** total memory SHALL not exceed 4 GB for full 200k context
- **AND** avoid VRAM spikes during prefill

---

## ADDED Requirements

### Requirement: KV Cache Precision Options

The runtime SHALL support configurable precision for KV cache tensors independent of model weights precision.

#### Scenario: FP8 quantized cache
- **WHEN** configured with `kv_dtype=FP8`
- **THEN** quantize K and V tensors to FP8 before storing in cache
- **AND** dequantize to model precision when reading for attention
- **AND** quality degradation SHALL be <2% vs FP16 baseline
- **AND** memory usage SHALL be ~50% of FP16

#### Scenario: INT8 quantized cache
- **WHEN** configured with `kv_dtype=INT8`  
- **THEN** quantize K and V tensors with scale/zero-point
- **AND** store quantized tensors + quantization parameters
- **AND** quality degradation SHALL be <2% vs FP16 baseline

#### Scenario: Fallback for unsupported hardware
- **WHEN** FP8 is configured but GPU lacks FP8 support
- **THEN** fall back to INT8 or FP16 with warning logged
- **AND** continue inference without failure

---

### Requirement: Group Query Attention Support

The runtime SHALL optimize KV cache for models using Grouped Query Attention (GQA) or Multi-Query Attention (MQA).

#### Scenario: GQA model detection
- **WHEN** loading a model with `num_key_value_heads < num_attention_heads`
- **THEN** configure KV cache with `num_kv_heads` from model config
- **AND** allocate KV tensors shaped `[batch, num_kv_heads, seq_len, head_dim]`
- **AND** memory savings SHALL reflect reduced head count

#### Scenario: MHA fallback
- **WHEN** model uses Multi-Head Attention (MHA)
- **THEN** set `num_kv_heads = num_attention_heads`
- **AND** behavior SHALL match traditional KV cache

---

### Requirement: Prefix Caching for Shared Prompts

The runtime SHALL cache and share KV states for common prompt prefixes across multiple inference sessions.

#### Scenario: Prefix cache hit
- **WHEN** two sessions use identical first N tokens (system prompt)
- **THEN** compute KV for prefix only once
- **AND** share read-only prefix KV across sessions
- **AND** each session maintains separate KV for unique tokens
- **AND** memory usage SHALL be prefix_size + sum(unique_tokens_per_session)

#### Scenario: Prefix identification
- **WHEN** receiving a new prompt
- **THEN** hash first N tokens to generate prefix key
- **AND** look up prefix in shared cache
- **AND** reuse cached prefix KV if available

#### Scenario: Prefix eviction
- **WHEN** prefix cache memory limit reached
- **THEN** evict least-recently-used prefixes
- **AND** preserve frequently-used prefixes (LRU with pinning)
- **AND** reference count SHALL prevent eviction of in-use prefixes

---

### Requirement: Chunked Prefill for Long Contexts

The runtime SHALL process long input sequences in chunks to prevent VRAM exhaustion during prefill.

#### Scenario: 200k token prefill
- **WHEN** prefilling a 200k-token input sequence
- **THEN** split input into chunks of 512-2048 tokens
- **AND** process each chunk sequentially
- **AND** commit KV to paged cache after each chunk
- **AND** peak VRAM SHALL not exceed configured limit

#### Scenario: Dynamic chunk sizing
- **WHEN** available VRAM headroom detected
- **THEN** adjust chunk size to maximize throughput
- **AND** reduce chunk size when VRAM pressure increases
- **AND** chunk size SHALL be between 512 and 2048 tokens

#### Scenario: Prefill progress tracking
- **WHEN** chunked prefill is in progress
- **THEN** expose progress callback with tokens_processed / total_tokens
- **AND** support cancellation mid-prefill
- **AND** allow resumable prefill from last committed chunk

---

### Requirement: Enhanced Page Eviction Policies

The runtime SHALL implement configurable eviction policies for paged KV cache management.

#### Scenario: Priority-based eviction
- **WHEN** pages need eviction and priority policy enabled
- **THEN** assign priority: prefix pages > recent pages > old pages
- **AND** evict lowest-priority pages first
- **AND** protect working set from eviction

#### Scenario: LRU eviction (default)
- **WHEN** pages need eviction and LRU policy enabled  
- **THEN** evict least-recently-accessed pages
- **AND** update access time on every read/write

#### Scenario: Adaptive eviction
- **WHEN** VRAM pressure detected
- **THEN** switch to aggressive eviction (larger batches)
- **WHEN** VRAM available
- **THEN** switch to conservative eviction (smaller batches)

---

### Requirement: Tensor Read/Write Operations

The paged KV cache SHALL implement complete tensor indexing and gathering operations for reading and writing K/V states.

#### Scenario: Write KV to pages
- **WHEN** writing K and V tensors for new tokens
- **THEN** slice physical page tensor at correct layer/page/slot indices
- **AND** write K tensor to designated page block
- **AND** write V tensor to designated page block  
- **AND** handle boundary cases for partial blocks

#### Scenario: Read KV from pages
- **WHEN** reading K and V for attention computation
- **THEN** gather K tensors from multiple pages for sequence
- **AND** gather V tensors from multiple pages for sequence
- **AND** concatenate in correct token order
- **AND** return contiguous tensors for attention layer

---

## ADDED Design Considerations

### Memory Budget Example

For Qwen3-0.6B with 200k context:

```
Without optimization:
  Mem(KV) = 2 × 24 layers × 200k × 2048 × 2 bytes = ~39 GB ❌

With optimization (GQA + FP8 + Paging):
  - GQA (8 heads → 2 kv_heads): 4× reduction
  - FP8 quantization: 2× reduction  
  - Incremental allocation: only actual tokens
  
  Mem(KV) = 2 × 24 × 200k × 2048 × 1 byte / 4 = ~4.8 GB ✓
```

### Quality vs Memory Tradeoffs

| Configuration | Memory | Quality Loss | Use Case |
|--------------|--------|--------------|----------|
| FP16 Full | 100% | 0% | Baseline |
| FP8 Full | 50% | <2% | Production (Ada/Hopper GPUs) |
| INT8 Full | 50% | <2% | Production (older GPUs) |
| FP16 + Sliding Window | Constant | 5-10% | Ultra-long context (>500k) |

### Implementation Phases

1. **Phase 1 (Critical)**: Tasks 1-4, 6 → Enable 200k support
2. **Phase 2 (Production)**: Tasks 5, 7 → Multi-session efficiency  
3. **Phase 3 (Advanced)**: Task 8 → Ultra-long context (optional)










