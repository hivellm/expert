# Technical Design: Expert Inference SDKs

**Change ID**: `add-inference-sdks`  
**Last Updated**: 2025-11-06

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            Application Layer                     │
│  (User code: CLI, API, scripts)                 │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              SDK Public API                      │
│  Expert.load() / Expert.generate()              │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐  ┌─────────▼────────┐
│ Package Loader  │  │ Inference Engine │
│ - Extract .tar  │  │ - Load model     │
│ - Validate      │  │ - Tokenize       │
│ - Parse manifest│  │ - Generate       │
└────────┬────────┘  └─────────┬────────┘
         │                      │
    ┌────▼──────┐         ┌────▼────────┐
    │ Manifest  │         │ Transformers│
    │ Validator │         │ / Candle    │
    └───────────┘         └─────────────┘
```

---

## Python SDK Design

### Module Structure

```python
# hivellm_expert/
__init__.py          # Public exports
expert.py            # Expert class
loader.py            # ExpertLoader class
manifest.py          # ExpertManifest, ManifestValidator
inference.py         # InferenceEngine, InferenceResult
grammar.py           # GrammarValidator (optional)
errors.py            # Exception hierarchy
utils.py             # Helpers (device detection, etc.)
```

### Key Classes

**Expert (Main API)**:
```python
class Expert:
    """Main interface for loading and using experts."""
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ) -> "Expert":
        """Load expert from .expert package or directory."""
        
    def generate(
        self,
        prompt: str,
        schema: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate response for single prompt."""
        
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, InferenceResult]:
        """Stream tokens as they're generated."""
        
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[InferenceResult]:
        """Batch inference for multiple prompts."""
```

**ExpertLoader**:
```python
class ExpertLoader:
    """Handles loading .expert packages."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "hivellm-expert"
        
    def load_package(self, path: Path) -> LoadedExpert:
        """Extract and validate .expert package."""
        # 1. Extract tar.gz to temp dir
        # 2. Verify SHA256 checksum
        # 3. Parse and validate manifest.json
        # 4. Locate adapter, tokenizer files
        # 5. Return LoadedExpert with paths
        
    def _extract_tarball(self, path: Path) -> Path:
        """Extract .expert tar.gz to cache dir."""
        
    def _verify_checksum(self, extracted_dir: Path, expected: str) -> bool:
        """Verify package integrity."""
```

**InferenceEngine**:
```python
class InferenceEngine:
    """Handles model loading and inference."""
    
    def __init__(
        self,
        loaded_expert: LoadedExpert,
        device: Optional[str] = None
    ):
        self.manifest = loaded_expert.manifest
        self.device = device or self._auto_detect_device()
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
    def _load_model(self):
        """Load base model + PEFT adapter."""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.manifest.base_model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return PeftModel.from_pretrained(
            base_model,
            self.manifest.adapter_path
        )
        
    def generate(
        self,
        prompt: str,
        **params
    ) -> InferenceResult:
        """Core generation logic."""
        # 1. Apply ChatML template
        # 2. Tokenize
        # 3. Generate with model
        # 4. Decode and return result
```

### Error Hierarchy

```python
class ExpertError(Exception):
    """Base exception for all SDK errors."""

class LoadError(ExpertError):
    """Error loading expert package."""

class ManifestError(LoadError):
    """Invalid or missing manifest."""

class InferenceError(ExpertError):
    """Error during inference."""

class OutOfMemoryError(InferenceError):
    """Insufficient memory for inference."""

class GrammarError(InferenceError):
    """Grammar validation failed."""
```

---

## Rust SDK Design

### Module Structure

```rust
// src/
lib.rs               // Public exports
expert.rs            // Expert struct
loader.rs            // Package loading
manifest.rs          // Reuse from CLI
inference.rs         // Inference engine
grammar.rs           // Grammar validation
error.rs             // Error types
utils.rs             // Helpers
```

### Key Structs

**Expert (Main API)**:
```rust
pub struct Expert {
    engine: InferenceEngine,
    manifest: ExpertManifest,
}

impl Expert {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let loaded = ExpertLoader::new().load_package(path)?;
        let engine = InferenceEngine::new(loaded)?;
        Ok(Self {
            engine,
            manifest: loaded.manifest,
        })
    }
    
    pub fn generate(&self, prompt: impl AsRef<str>) -> Result<InferenceResult> {
        self.engine.generate(prompt.as_ref(), None)
    }
    
    pub async fn generate_stream(
        &self,
        prompt: impl AsRef<str>
    ) -> Result<impl Stream<Item = Result<String>>> {
        self.engine.generate_stream(prompt.as_ref())
    }
    
    pub fn generate_batch(
        &self,
        prompts: &[impl AsRef<str>]
    ) -> Result<Vec<InferenceResult>> {
        self.engine.generate_batch(prompts)
    }
}
```

**ExpertLoader**:
```rust
pub struct ExpertLoader {
    cache_dir: PathBuf,
}

impl ExpertLoader {
    pub fn new() -> Self {
        Self {
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("hivellm-expert"),
        }
    }
    
    pub fn load_package(&self, path: impl AsRef<Path>) -> Result<LoadedExpert> {
        // 1. Extract tar.gz
        // 2. Verify checksum
        // 3. Parse manifest
        // 4. Locate files
        // 5. Return LoadedExpert
    }
}
```

**InferenceEngine (Candle-based)**:
```rust
pub struct InferenceEngine {
    model: QwenEngine,  // Reuse from CLI
    tokenizer: Tokenizer,
    config: InferenceConfig,
}

impl InferenceEngine {
    pub fn new(loaded: LoadedExpert) -> Result<Self> {
        let device = Self::auto_detect_device()?;
        let model = QwenEngine::from_local(&loaded.adapter_path, device.is_cuda())?;
        let tokenizer = Tokenizer::from_file(&loaded.tokenizer_path)?;
        
        Ok(Self {
            model,
            tokenizer,
            config: InferenceConfig::from_manifest(&loaded.manifest),
        })
    }
    
    pub fn generate(
        &self,
        prompt: &str,
        params: Option<InferenceParams>
    ) -> Result<InferenceResult> {
        // 1. Apply ChatML template
        // 2. Tokenize
        // 3. Generate with QwenEngine
        // 4. Decode and return
    }
}
```

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ExpertError {
    #[error("Failed to load expert: {0}")]
    LoadError(String),
    
    #[error("Invalid manifest: {0}")]
    ManifestError(#[from] serde_json::Error),
    
    #[error("Inference failed: {0}")]
    InferenceError(String),
    
    #[error("Out of memory")]
    OutOfMemory,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ExpertError>;
```

---

## Design Decisions

### 1. Code Reuse Strategy

**Decision**: Maximize reuse of existing CLI code

**Rationale**:
- CLI already has working Qwen engine (Rust)
- CLI already has manifest parsing (Rust)
- Python inference logic from trainer (Python)
- Avoid duplicating complex logic

**Implementation**:
- Rust SDK: Import `QwenEngine` and `manifest` from CLI crate
- Python SDK: Extract inference patterns from `expert_trainer.py`
- Create shared utilities module if needed

### 2. Package Caching

**Decision**: Cache extracted packages by default

**Rationale**:
- `.expert` extraction is slow (~500ms for 25MB)
- Most use cases load same expert multiple times
- Disk space is cheap

**Implementation**:
- Cache location: `~/.cache/hivellm-expert/<expert-name>/`
- Cache key: SHA256 of original `.expert` file
- Auto-cleanup: Remove if original package deleted or modified

### 3. ChatML Template Handling

**Decision**: Automatically apply ChatML template, no option to disable

**Rationale**:
- All current experts use ChatML (Qwen3 native)
- Simplifies API (less configuration)
- Can add customization later if needed

**Implementation**:
- Python: Use `tokenizer.apply_chat_template()`
- Rust: Implement ChatML formatter manually (reuse from CLI)

### 4. Device Selection

**Decision**: Auto-detect device, allow manual override

**Rationale**:
- Most users want "just work" behavior
- Advanced users need control (multi-GPU, specific device)

**Implementation**:
```python
# Auto-detect (default)
expert = Expert.load("model.expert")

# Manual override
expert = Expert.load("model.expert", device="cuda:1")
expert = Expert.load("model.expert", device="cpu")  # Force CPU
```

### 5. Streaming API

**Decision**: Streaming returns generator/stream, not callback

**Rationale**:
- More Pythonic (generators)
- Easier to test
- More flexible (can convert to async easily)

**Implementation**:
```python
# Python - generator
for token in expert.generate_stream(prompt):
    print(token, end="", flush=True)

# Rust - async stream
let mut stream = expert.generate_stream(prompt).await?;
while let Some(token) = stream.next().await {
    print!("{}", token?);
}
```

### 6. Grammar Validation

**Decision**: Optional feature, only load if grammar.gbnf exists

**Rationale**:
- Not all experts have grammar
- llama-cpp-python is heavy dependency
- Grammar enforcement is advanced feature

**Implementation**:
- Check for `grammar.gbnf` in package
- If exists and llama-cpp available → enable
- If exists but llama-cpp missing → warning, skip
- If not exists → skip

### 7. Error Messages

**Decision**: Prioritize actionability over brevity

**Rationale**:
- Errors should guide users to solutions
- Include troubleshooting hints
- Reference documentation

**Example**:
```python
# Bad
LoadError: "Invalid manifest"

# Good
LoadError: 
  "Invalid manifest.json in expert package:
  - Missing required field 'base_models'
  - Line 15, column 3 in manifest.json
  
  Troubleshooting:
  1. Verify package is not corrupted (check SHA256)
  2. Update expert-cli to latest version
  3. Re-download expert package
  
  See: https://docs.hivellm.ai/experts/troubleshooting"
```

---

## Performance Considerations

### Python SDK

**Optimizations**:
1. **Lazy Loading**: Don't load model until first generate()
2. **Tokenizer Caching**: Reuse tokenizer across calls
3. **Batch Padding**: Efficient padding for batch inference
4. **Device Pinning**: Keep model on device, avoid transfers

**Benchmarks**:
- Target: <5% overhead vs manual approach
- Measure: Latency, throughput, memory
- Profile: cProfile for hot paths

### Rust SDK

**Optimizations**:
1. **Zero-Copy**: Avoid unnecessary allocations
2. **CUDA Graphs**: Enable for repeated inference patterns
3. **Async/Await**: Non-blocking streaming
4. **Smart Batching**: Auto-batch for multi-threaded contexts

**Benchmarks**:
- Target: Match or beat Python SDK
- Use criterion.rs for micro-benchmarks
- Profile with perf/valgrind

---

## Testing Strategy

### Unit Tests

**Python**:
- pytest for all modules
- Mock transformers/PEFT for fast tests
- Fixtures for test expert packages

**Rust**:
- Standard `#[cfg(test)]` tests
- Mock Candle for unit tests
- Property-based tests with proptest

### Integration Tests

- Use real `expert-sql.v0.2.0.expert` package
- Test full load → generate → validate cycle
- Compare outputs with known good results
- Test error paths (corrupted packages, OOM)

### Performance Tests

- Benchmark suite comparing SDK vs manual
- CI performance regression detection
- Memory profiling (memray for Python, valgrind for Rust)

### Cross-Platform Tests

- GitHub Actions matrix: Windows, Linux, macOS
- Test CUDA on Linux + Windows
- Test MPS on macOS
- Test CPU fallback on all platforms

---

## Security Considerations

1. **Package Integrity**: Always verify SHA256 before loading
2. **Untrusted Code**: Never eval/exec code from packages
3. **Path Traversal**: Sanitize paths during extraction
4. **DoS**: Limit package size (e.g., 1GB max)
5. **Dependencies**: Pin versions, audit regularly

---

## Migration from Manual Approach

### Before (Manual - ~30 lines)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "F:/Node/hivellm/expert/models/Qwen3-0.6B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "F:/Node/hivellm/expert/models/Qwen3-0.6B",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(
    base_model,
    "expert-sql/weights/checkpoint-1250"
)

schema = "CREATE TABLE users ..."
question = "List users from 2024"

messages = [
    {"role": "system", "content": f"Dialect: postgres\nSchema:\n{schema}"},
    {"role": "user", "content": question}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, temperature=0.7, top_p=0.8, top_k=20)

result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(result)
```

### After (SDK - 3 lines)

```python
from hivellm_expert import Expert

expert = Expert.load("expert-sql-qwen3-0-6b.v0.2.0.expert")
result = expert.generate("List users from 2024", schema="CREATE TABLE users ...")
print(result.text)
```

**Migration Benefits**:
- 90% less code
- No need to understand PEFT, transformers
- Automatic device management
- Built-in error handling
- Manifest-driven configuration

---

## Future Enhancements

### Phase 2 (Post-MVP)

1. **Model Registry**: Load from HuggingFace Hub directly
2. **Quantization**: Support 4-bit/8-bit quantization
3. **Multi-Modal**: Support vision/audio experts
4. **Fine-Tuning**: SDK for training, not just inference
5. **Serving**: Built-in HTTP server mode
6. **Metrics**: Prometheus/OpenTelemetry integration
7. **Caching**: KV-cache persistence across calls

### Integration Ideas

- **LangChain**: hivellm-expert as LangChain LLM
- **LlamaIndex**: Native integration
- **FastAPI**: Official FastAPI template
- **Gradio**: One-line UI for experts

---

## Open Issues for Discussion

1. **Async Python**: Should `generate()` be async or blocking?
2. **Context Managers**: Should Expert support `with` statement for cleanup?
3. **Logging**: Use standard logging or custom (for colorized output)?
4. **Versioning**: Should SDK version track manifest schema version?
5. **CLI Integration**: Should CLI depend on SDK or vice versa?

