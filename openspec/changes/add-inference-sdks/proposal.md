# Proposal: Expert Inference SDKs

**Change ID**: `add-inference-sdks`  
**Status**: Draft  
**Created**: 2025-11-06  
**Author**: HiveLLM Team

## Overview

Create official SDKs for loading and running Expert inference in Python and Rust, providing a simple, ergonomic API for integrating trained experts into applications.

## Motivation

### Current State (Problems)

**Manual Integration Required**:
- Users must manually load base models, adapters, tokenizers
- Complex setup for PEFT models (LoRA/DoRA)
- Different code for each programming language
- No standardized way to load `.expert` packages
- Grammar validation and decoding params are manual

**Example Current Workflow (Python)**:
```python
# Manual, verbose setup
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, adapter_path)

# Manual message formatting
messages = [{"role": "system", "content": "..."}, ...]
text = tokenizer.apply_chat_template(...)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Manual generation
outputs = model.generate(**inputs, temperature=0.7, ...)
result = tokenizer.decode(outputs[0][...], skip_special_tokens=True)
```

**No Package Support**:
- `.expert` files must be manually extracted
- Manifest parsing is manual
- No validation of package integrity
- No automatic model discovery

**Inconsistent Experience**:
- Different APIs for Rust vs Python
- No shared conventions
- Difficult to port code between languages

### Desired State (Solution)

**Simple, Unified SDK**:

```python
# Python SDK - Simple and clean
from hivellm_expert import Expert

# Single expert
expert = Expert.load("expert-sql-qwen3-0-6b.v0.2.0.expert")
result = expert.generate("List all users who registered in 2024")
print(result.text)

# Multiple experts
from hivellm_expert import ExpertManager

manager = ExpertManager()
manager.load_expert("sql", "expert-sql.v0.2.0.expert")
manager.load_expert("cypher", "expert-neo4j.v0.1.0.expert")

# Route queries to appropriate expert
sql_result = manager.generate("sql", "List users from 2024")
cypher_result = manager.generate("cypher", "Find related movies")
```

```rust
// Rust SDK - Ergonomic and type-safe
use hivellm_expert::{Expert, ExpertManager};

// Single expert
let expert = Expert::load("expert-sql-qwen3-0-6b.v0.2.0.expert")?;
let result = expert.generate("List all users who registered in 2024")?;
println!("{}", result.text);

// Multiple experts with efficient memory management
let mut manager = ExpertManager::new();
manager.load_expert("sql", "expert-sql.v0.2.0.expert")?;
manager.load_expert("cypher", "expert-neo4j.v0.1.0.expert")?;

// Automatic base model reuse (both use Qwen3-0.6B)
let sql_result = manager.generate("sql", "List users")?;
let cypher_result = manager.generate("cypher", "Find movies")?;
```

**Key Features**:
- ✅ Automatic model + adapter + tokenizer loading
- ✅ `.expert` package support (extract, validate, load)
- ✅ **Multiple expert loading and management**
- ✅ **Base model sharing** (load once, use with multiple adapters)
- ✅ **Hot-swapping adapters** (switch experts without reloading base)
- ✅ Manifest-driven configuration (decoding params, grammar)
- ✅ ChatML template auto-application
- ✅ Grammar validation (when enabled)
- ✅ Streaming support
- ✅ Batch inference
- ✅ Error handling and validation
- ✅ Cross-platform (Windows, Linux, macOS)

## Impact Analysis

### Benefits

**Developer Experience**:
- 📈 **Faster Integration**: ~30 lines of code → 3 lines
- 📈 **Lower Barrier**: No need to understand PEFT, transformers internals
- 📈 **Consistency**: Same API across Python and Rust
- 📈 **Best Practices**: Built-in optimizations (device auto-detection, memory management)

**Enterprise Adoption**:
- 🎯 **Production Ready**: Validated package loading, error handling
- 🎯 **Type Safety**: Rust SDK provides compile-time guarantees
- 🎯 **Performance**: Optimized for inference (CUDA graphs, batching)
- 🎯 **Observability**: Built-in logging, metrics, tracing

**Ecosystem Growth**:
- 🚀 **More Applications**: Easier to integrate → more adoption
- 🚀 **Community**: Standardized SDK → easier to share examples
- 🚀 **Packages**: `.expert` becomes first-class citizen

### Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API changes breaking users | Medium | High | Semantic versioning, deprecation warnings |
| Performance overhead | Low | Medium | Benchmark against manual approach, optimize hot paths |
| Dependency bloat | Medium | Low | Keep dependencies minimal, make features opt-in |
| Cross-platform issues | Medium | Medium | CI testing on Windows/Linux/macOS |

## Technical Approach

### Python SDK (`hivellm-expert`)

**Architecture**:
```
hivellm_expert/
├── __init__.py           # Main API
├── expert.py             # Expert class
├── loader.py             # Package loading (.expert files)
├── manifest.py           # Manifest parsing/validation
├── inference.py          # Generation, streaming
├── grammar.py            # Grammar validation (optional)
└── utils.py              # Helpers
```

**Key Classes**:
- `Expert`: Main interface for loading and inference
- `ExpertManifest`: Manifest representation
- `InferenceResult`: Generation output with metadata
- `ExpertLoader`: Package extraction and validation

**Dependencies**:
- `transformers` (required)
- `peft` (required)
- `torch` (required)
- `jsonschema` (manifest validation)
- `llama-cpp-python` (optional, for grammar)

### Rust SDK (`hivellm-expert`)

**Architecture**:
```
src/
├── lib.rs                # Main API
├── expert.rs             # Expert struct
├── loader.rs             # Package loading
├── manifest.rs           # Manifest types (already exists)
├── inference.rs          # Generation logic
├── grammar.rs            # Grammar validation
└── error.rs              # Error types
```

**Key Structs**:
- `Expert`: Main interface
- `ExpertManifest`: Manifest representation (reuse from CLI)
- `InferenceConfig`: Generation parameters
- `InferenceResult`: Generation output

**Dependencies**:
- `candle-core` (inference engine)
- `tokenizers` (tokenization)
- `serde` / `serde_json` (manifest)
- `tar` / `flate2` (package extraction)

### Integration with Existing Code

**Reuse**:
- ✅ `expert/cli/src/manifest.rs` → Rust SDK manifest types
- ✅ `expert/schemas/expert-manifest.schema.json` → Python validation
- ✅ `expert/cli/expert_trainer.py` → Inference logic patterns
- ✅ `expert/cli/src/engines/qwen3_engine.rs` → Candle engine

**New**:
- Package extraction and validation
- Simple public API (currently CLI-focused)
- Streaming generators
- Batch inference helpers

## Success Criteria

### Functional

- [ ] Python SDK loads `.expert` packages
- [ ] Rust SDK loads `.expert` packages
- [ ] Both SDKs generate correct outputs (validate against manual approach)
- [ ] Grammar validation works (when enabled)
- [ ] Streaming inference works
- [ ] Batch inference works (multiple prompts)
- [ ] Error handling is comprehensive

### Non-Functional

- [ ] Performance within 5% of manual approach
- [ ] Documentation covers 90% of use cases
- [ ] Test coverage >80%
- [ ] Works on Windows, Linux, macOS
- [ ] Published to PyPI (Python) and crates.io (Rust)

### User Acceptance

- [ ] 3 lines of code for simple inference
- [ ] No manual model/tokenizer loading
- [ ] Clear error messages
- [ ] Migration guide from manual approach
- [ ] Example applications (REST API, CLI, notebook)

## Timeline

**Phase 1: Python SDK (Week 1-2)**
- Core API design
- Package loading
- Basic inference
- Unit tests
- Documentation

**Phase 2: Rust SDK (Week 2-3)**
- Core API design
- Package loading (reuse CLI code)
- Candle-based inference
- Unit tests
- Documentation

**Phase 3: Advanced Features (Week 3-4)**
- Streaming inference (both SDKs)
- Batch inference (both SDKs)
- Grammar validation (both SDKs)
- Performance optimization

**Phase 4: Release (Week 4)**
- Integration tests
- Example applications
- PyPI / crates.io publish
- Announcement & docs

## Multiple Experts Support

### Use Cases

**1. Multi-Domain Applications**:
- Web app supporting SQL + Neo4j + TypeScript queries
- Load all experts at startup
- Route queries based on user intent

**2. Memory-Efficient Deployment**:
- Multiple experts share same base model (Qwen3-0.6B)
- Base model loaded once (~1.2GB)
- Each adapter adds only ~25-50MB

**3. Dynamic Expert Loading**:
- Load experts on-demand based on request
- Unload inactive experts to save memory
- LRU cache for frequently used experts

### Technical Approach

**Base Model Sharing**:
```python
# Inefficient (loads base model 3 times)
sql_expert = Expert.load("sql.expert")      # 1.2GB
cypher_expert = Expert.load("cypher.expert") # 1.2GB
ts_expert = Expert.load("ts.expert")        # 1.2GB
# Total: ~3.6GB

# Efficient (loads base model once)
manager = ExpertManager()
manager.load_expert("sql", "sql.expert")      # 1.2GB + 25MB
manager.load_expert("cypher", "cypher.expert") # +25MB
manager.load_expert("ts", "ts.expert")        # +25MB
# Total: ~1.3GB (73% memory savings!)
```

**Hot-Swapping**:
```python
# Switch adapters without reloading base model
manager.load_expert("sql", "sql.expert")
result1 = manager.generate("sql", "SELECT * FROM users")

manager.load_expert("cypher", "cypher.expert")  # Swap adapter
result2 = manager.generate("cypher", "MATCH (u:User)")
# Base model stays in memory, only adapter swapped (~50ms)
```

**LRU Eviction** (Rust):
```rust
let mut manager = ExpertManager::builder()
    .max_loaded_experts(3)  // Keep max 3 adapters in memory
    .build();

manager.load_expert("sql", "sql.expert")?;
manager.load_expert("cypher", "cypher.expert")?;
manager.load_expert("ts", "ts.expert")?;
manager.load_expert("python", "python.expert")?;  // Evicts least-used (SQL)

// Automatically reloads if accessed
manager.generate("sql", "SELECT ...")?;  // Reloads SQL, evicts cypher
```

## Open Questions

1. **API Design**: Should we support both high-level (`expert.generate()`) and low-level APIs?
2. **Caching**: Should SDKs cache extracted `.expert` packages?
3. **Model Registry**: Should we support remote model loading (HuggingFace Hub)?
4. **Compatibility**: How to handle breaking changes in manifest format?
5. **Versioning**: Should SDK version match expert schema version?
6. **Multi-Expert Routing**: Should SDK provide automatic routing (keyword-based) or leave to user?
7. **Mixed Base Models**: How to handle experts with different base models (Qwen3 vs Llama)?

## Alternatives Considered

### Alternative 1: CLI-Only Approach
**Decision**: Rejected  
**Reason**: CLI is great for testing but not suitable for programmatic integration. Libraries are needed.

### Alternative 2: Wrapper Script Approach
**Decision**: Rejected  
**Reason**: Fragile, hard to maintain, no type safety, poor DX.

### Alternative 3: Extend Transformers/HuggingFace
**Decision**: Deferred  
**Reason**: Good long-term goal, but requires upstream collaboration. Start with standalone SDK first.

## References

- Expert manifest schema: `expert/schemas/expert-manifest.schema.json`
- CLI implementation: `expert/cli/`
- Qwen3 engine: `expert/cli/src/engines/qwen3_engine.rs`
- PEFT documentation: https://huggingface.co/docs/peft
- Candle docs: https://github.com/huggingface/candle

