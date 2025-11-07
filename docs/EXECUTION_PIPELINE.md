# Execution Pipeline

> Four-stage inference pipeline from prompt to output, with API design and parallelism model

## Overview

The Expert System processes each inference request through a well-defined four-stage pipeline. Each stage is optimized for its specific compute characteristics (CPU vs GPU, I/O vs compute) and can overlap with other stages for maximum throughput.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Reception                       │
│  - Parse request (prompt, template, params)                 │
│  - Validate inputs                                          │
│  - Normalize and sanitize                                   │
│  - Assign job ID                                            │
│  Latency: <1ms | Location: API Server (CPU)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: Reasoning (RG)                  │
│  - Fast classification (heuristics + embeddings)            │
│  - Query expert index (ANN search)                          │
│  - Select top-K experts (K ≤ 10)                            │
│  - Determine composition order                              │
│  - Tune generation params (temp, max_tokens, etc.)          │
│  Latency: 10-50ms | Location: Router Service (CPU/RAM)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Stage 3: Preparation                      │
│  - Check VRAM budget                                        │
│  - Load base model (if not already loaded)                  │
│  - Hot-load experts (SSD → RAM → VRAM)                      │
│  - Apply adapters to base model layers                      │
│  - Attach soft prompts (if any)                             │
│  - Initialize paged KV cache                                │
│  Latency: 1-200ms | Location: Inference Engine (GPU)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Stage 4: Inference                        │
│  - Forward pass with base model + experts                   │
│  - Paged KV cache management                                │
│  - Decode tokens (greedy/sampling/beam)                     │
│  - Apply generation constraints (temp, top-p, etc.)         │
│  - Optional: Speculative decoding                           │
│  Latency: 0.5-30s | Location: GPU                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Post-Processing                           │
│  - Validate output (JSON, schema, format)                   │
│  - Collect metrics (latency, tokens, VRAM)                  │
│  - Release experts (if not cached)                          │
│  - Clear KV cache                                           │
│  - Send feedback to Router (success/failure)                │
│  Latency: 1-10ms | Location: CPU                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                    ┌────────┐
                    │ Result │
                    └────────┘
```

---

## Stage 1: Reception

### Responsibilities

1. **Parse incoming request** (HTTP/gRPC/SDK)
2. **Validate inputs** (non-empty prompt, valid params)
3. **Normalize** (trim whitespace, standardize encoding)
4. **Assign job ID** (UUID for tracking)
5. **Enqueue job** (submit to orchestrator)

### Input Format

```typescript
interface InferenceRequest {
  prompt: string;                    // Required: user prompt
  template?: string;                 // Optional: prompt template
  body?: string;                     // Optional: document/data to process
  
  // User preferences (hints to router)
  temperature?: number;              // 0.0-2.0, default: router decides
  max_tokens?: number;               // Max output tokens, default: router decides
  top_p?: number;                    // Nucleus sampling, default: 0.95
  top_k?: number;                    // Top-k sampling, default: null
  min_p?: number;                    // Min-p sampling, default: null
  repetition_penalty?: number;       // 1.0-2.0, default: 1.1
  
  // Expert preferences (optional override)
  force_experts?: string[];          // Force specific experts
  exclude_experts?: string[];        // Exclude specific experts
  max_experts?: number;              // Limit expert count (default 10)
  
  // Context management
  context_size?: number;             // Override context window
  
  // Metadata
  user_id?: string;                  // For rate limiting/analytics
  priority?: number;                 // 0-9, default: 5
  timeout_ms?: number;               // Max execution time
}
```

### Validation Rules

```python
def validate_request(req: InferenceRequest) -> None:
    # Prompt validation
    if not req.prompt or len(req.prompt.strip()) == 0:
        raise ValidationError("Prompt cannot be empty")
    
    if len(req.prompt) > 500_000:  # ~100k tokens
        raise ValidationError("Prompt too long")
    
    # Parameter validation
    if req.temperature is not None:
        if not 0.0 <= req.temperature <= 2.0:
            raise ValidationError("Temperature must be in [0.0, 2.0]")
    
    if req.max_tokens is not None:
        if not 1 <= req.max_tokens <= 200_000:
            raise ValidationError("max_tokens must be in [1, 200000]")
    
    if req.max_experts is not None:
        if not 1 <= req.max_experts <= 10:
            raise ValidationError("max_experts must be in [1, 10]")
```

### Output

```typescript
interface JobSubmission {
  job_id: string;                    // UUID
  status: "queued" | "processing";
  submitted_at: string;              // ISO 8601
  estimated_wait_ms?: number;        // Queue estimate
}
```

---

## Stage 2: Reasoning (RG)

### Responsibilities

1. **Analyze prompt** (heuristics, embeddings, semantics)
2. **Query expert index** (ANN search for relevant experts)
3. **Score and rank experts** (relevance, VRAM cost, success rate)
4. **Select top-K** (K ≤ 10 or user-specified max)
5. **Determine composition order** (which expert applies first)
6. **Tune generation parameters** (temp, max_tokens based on task type)

### Feature Extraction

```python
def extract_features(prompt: str, body: str | None) -> Features:
    features = Features()
    
    # Heuristic features (fast)
    features.language = detect_language(prompt)  # langdetect, ~1ms
    features.has_json = "{" in prompt or "json" in prompt.lower()
    features.has_code = any(kw in prompt.lower() for kw in ["function", "class", "import"])
    features.task_type = classify_task(prompt)  # regex patterns, ~0.5ms
    
    # Embedding features (moderate)
    features.embedding = embed_model.encode(prompt)  # MiniLM, ~10ms
    
    # Semantic features (slow, optional)
    if len(prompt) < 2000:  # Only for short prompts
        features.semantic_intent = lightweight_llm_classify(prompt)  # ~50ms
    
    return features
```

### Expert Selection Algorithm

```python
def select_experts(
    features: Features,
    expert_index: ExpertIndex,
    max_experts: int = 10
) -> list[Expert]:
    
    # Step 1: Heuristic filtering (fast path)
    candidates = set()
    
    if features.language == "en":
        candidates.add("english-basic")
    if features.has_json:
        candidates.add("json-parser")
    # ... more heuristics
    
    # Step 2: Embedding-based retrieval (semantic similarity)
    similar_experts = expert_index.search(
        features.embedding,
        k=30,  # Retrieve more than needed
        threshold=0.6  # Min similarity
    )
    candidates.update(e.name for e in similar_experts)
    
    # Step 3: Score each candidate
    scored_experts = []
    for expert_name in candidates:
        expert = expert_index.get(expert_name)
        
        score = compute_score(expert, features)
        scored_experts.append((score, expert))
    
    # Step 4: Filter incompatibilities
    scored_experts = filter_incompatible(scored_experts)
    
    # Step 5: Select top-K
    scored_experts.sort(reverse=True)  # Highest score first
    selected = [exp for _, exp in scored_experts[:max_experts]]
    
    # Step 6: Order by composition sequence
    ordered = order_experts(selected, features)
    
    return ordered

def compute_score(expert: Expert, features: Features) -> float:
    # Weighted scoring
    semantic_sim = cosine_similarity(expert.embedding, features.embedding)
    vram_penalty = expert.vram_mb / 100.0  # Penalize large experts
    success_rate = expert.historical_success_rate  # From telemetry
    
    score = (
        0.50 * semantic_sim +
        0.30 * success_rate +
        0.20 * (1.0 - vram_penalty)
    )
    return score
```

### Parameter Tuning

```python
def tune_parameters(
    features: Features,
    user_prefs: dict
) -> GenerationParams:
    
    # Defaults based on task type
    if features.task_type == "json_parsing":
        params = GenerationParams(
            temperature=0.2,
            max_tokens=2048,
            top_p=0.95,
            repetition_penalty=1.0
        )
    elif features.task_type == "creative_writing":
        params = GenerationParams(
            temperature=0.8,
            max_tokens=4096,
            top_p=0.95,
            repetition_penalty=1.1
        )
    elif features.task_type == "code_generation":
        params = GenerationParams(
            temperature=0.4,
            max_tokens=8192,
            top_p=0.95,
            repetition_penalty=1.05
        )
    else:  # Default
        params = GenerationParams(
            temperature=0.6,
            max_tokens=2048,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    # Override with user preferences
    if user_prefs.get("temperature") is not None:
        params.temperature = user_prefs["temperature"]
    if user_prefs.get("max_tokens") is not None:
        params.max_tokens = user_prefs["max_tokens"]
    
    return params
```

### Output: Expert Plan

```typescript
interface ExpertPlan {
  experts: Expert[];                 // Ordered list of experts to load
  composition_order: string[];       // Expert names in application order
  generation_params: {
    temperature: number;
    max_tokens: number;
    top_p: number;
    top_k?: number;
    min_p?: number;
    repetition_penalty: number;
  };
  prompt_modified?: string;          // Router may rewrite prompt
  reasoning_trace?: string;          // Debug: why these experts?
}
```

---

## Stage 3: Preparation

### Responsibilities

1. **Check VRAM budget** (can we fit these experts?)
2. **Load base model** (if first inference)
3. **Load expert weights** (decompress .expert, read safetensors)
4. **Attach adapters** (apply LoRA/IA³/DoRA to base model layers)
5. **Apply soft prompts** (prepend learned embeddings)
6. **Initialize KV cache** (allocate paged memory)

### VRAM Budget Check

```python
def check_vram_budget(
    plan: ExpertPlan,
    budget: VRAMBudget
) -> bool:
    
    required_vram = 0
    
    # Base model (always loaded)
    required_vram += 500  # MB (Qwen3-0.6B INT4)
    
    # Experts
    for expert in plan.experts:
        required_vram += expert.vram_mb
    
    # KV cache estimate (depends on context size)
    required_vram += estimate_kv_cache_size(plan.generation_params.max_tokens)
    
    # Overhead (activation memory, buffers)
    required_vram += 200
    
    return budget.can_allocate(required_vram)

def estimate_kv_cache_size(max_tokens: int) -> int:
    # Formula: 2 * num_layers * hidden_size * max_tokens * bytes_per_element
    # Qwen3-0.6B: 28 layers, 896 hidden, FP16 = 2 bytes
    # Paged KV reduces fragmentation but doesn't reduce total size
    
    num_layers = 28
    hidden_size = 896
    bytes_per_element = 2  # FP16
    
    size_mb = (2 * num_layers * hidden_size * max_tokens * bytes_per_element) / (1024 * 1024)
    return int(size_mb * 1.2)  # Add 20% safety margin
```

### Expert Loading

```rust
// Conceptual Rust implementation
impl InferenceEngine {
    async fn load_experts(&mut self, plan: &ExpertPlan) -> Result<()> {
        for expert in &plan.experts {
            // Check if already in hot cache
            if self.expert_cache.contains(&expert.name) {
                log::debug!("Expert {} already loaded (hot cache)", expert.name);
                continue;
            }
            
            // Load from SSD
            let package_path = format!("~/.expert/store/{}.expert", expert.name);
            let package = ExpertPackage::load(&package_path).await?;
            
            // Verify signature (security)
            package.verify_signature()?;
            
            // Decompress and load weights
            let weights = package.load_weights().await?;
            
            // Attach to base model layers
            match expert.adapter_type {
                AdapterType::LoRA => {
                    self.attach_lora(&weights, &expert.config)?;
                },
                AdapterType::IA3 => {
                    self.attach_ia3(&weights, &expert.config)?;
                },
                AdapterType::SoftPrompt => {
                    self.attach_soft_prompt(&weights)?;
                },
            }
            
            // Add to hot cache (LRU)
            self.expert_cache.insert(expert.name.clone(), weights);
        }
        
        Ok(())
    }
    
    fn attach_lora(
        &mut self,
        weights: &SafeTensors,
        config: &LoRAConfig
    ) -> Result<()> {
        for layer_idx in 0..self.base_model.num_layers() {
            for module_name in &config.target_modules {
                let lora_a_key = format!(
                    "base_model.model.layers.{}.{}.lora_A",
                    layer_idx, module_name
                );
                let lora_b_key = format!(
                    "base_model.model.layers.{}.{}.lora_B",
                    layer_idx, module_name
                );
                
                let a = weights.tensor(&lora_a_key)?;
                let b = weights.tensor(&lora_b_key)?;
                
                // Attach to layer
                let layer = self.base_model.layer_mut(layer_idx);
                let module = layer.module_mut(module_name)?;
                module.attach_lora_adapter(a, b, config.alpha, config.r);
            }
        }
        Ok(())
    }
}
```

### KV Cache Initialization

```python
def initialize_kv_cache(
    session_id: str,
    context_size: int,
    num_layers: int = 28
) -> PagedKVCache:
    
    # Paged KV cache configuration
    block_size = 16  # Tokens per block
    num_blocks = (context_size + block_size - 1) // block_size
    
    cache = PagedKVCache(
        session_id=session_id,
        num_layers=num_layers,
        hidden_size=896,
        num_blocks=num_blocks,
        block_size=block_size,
        device="cuda"
    )
    
    return cache
```

---

## Stage 4: Inference

### Responsibilities

1. **Forward pass** through base model + experts
2. **KV cache management** (page allocation, eviction)
3. **Token decoding** (greedy, sampling, beam search)
4. **Apply generation constraints** (temp, top-p, repetition penalty)
5. **Stop condition checking** (EOS token, max_tokens reached)

### Generation Loop

```python
def generate(
    engine: InferenceEngine,
    session: Session,
    plan: ExpertPlan,
    prompt: str
) -> GenerationResult:
    
    # Tokenize prompt
    input_ids = engine.tokenizer.encode(prompt)
    
    # Prefill: Process input prompt
    kv_cache = session.kv_cache
    logits = engine.forward(input_ids, kv_cache, experts=plan.experts)
    
    # Decode loop
    generated_tokens = []
    current_token = sample_token(logits, plan.generation_params)
    
    while len(generated_tokens) < plan.generation_params.max_tokens:
        generated_tokens.append(current_token)
        
        # Check stop condition
        if current_token == engine.tokenizer.eos_token_id:
            break
        
        # Single token forward pass (decode)
        logits = engine.forward([current_token], kv_cache, experts=plan.experts)
        
        # Apply generation params
        logits = apply_temperature(logits, plan.generation_params.temperature)
        logits = apply_repetition_penalty(logits, generated_tokens, plan.generation_params.repetition_penalty)
        logits = apply_top_p(logits, plan.generation_params.top_p)
        
        # Sample next token
        current_token = sample_token(logits, plan.generation_params)
    
    # Decode tokens to text
    output_text = engine.tokenizer.decode(generated_tokens)
    
    return GenerationResult(
        text=output_text,
        tokens=generated_tokens,
        num_tokens=len(generated_tokens),
        finish_reason="stop" if current_token == engine.tokenizer.eos_token_id else "length"
    )
```

### Paged KV Cache Management

```python
class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size  # e.g., 16 tokens
        self.num_blocks = num_blocks
        
        # Physical blocks (contiguous GPU memory)
        self.physical_blocks = allocate_blocks(num_blocks, block_size)
        
        # Logical → Physical mapping
        self.block_mapping = {}  # {logical_block_id: physical_block_id}
        self.free_blocks = set(range(num_blocks))
    
    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise OutOfMemoryError("No free KV cache blocks")
        
        physical_id = self.free_blocks.pop()
        return physical_id
    
    def write_kv(self, token_position: int, k: Tensor, v: Tensor):
        logical_block = token_position // self.block_size
        offset = token_position % self.block_size
        
        # Allocate physical block if needed
        if logical_block not in self.block_mapping:
            self.block_mapping[logical_block] = self.allocate_block()
        
        physical_block = self.block_mapping[logical_block]
        
        # Write to physical memory
        self.physical_blocks[physical_block][offset] = (k, v)
    
    def read_kv(self, token_position: int) -> tuple[Tensor, Tensor]:
        logical_block = token_position // self.block_size
        offset = token_position % self.block_size
        
        physical_block = self.block_mapping[logical_block]
        return self.physical_blocks[physical_block][offset]
```

### Speculative Decoding (Optional, P6)

```python
def speculative_generate(
    engine: InferenceEngine,
    session: Session,
    plan: ExpertPlan,
    prompt: str,
    draft_length: int = 4
) -> GenerationResult:
    
    # Use base model alone as draft generator
    draft_plan = ExpertPlan(experts=[], ...)
    
    generated_tokens = []
    
    while len(generated_tokens) < plan.generation_params.max_tokens:
        # Draft: Generate N tokens speculatively (fast, no experts)
        draft_tokens = engine.generate_n_tokens(
            draft_length,
            plan=draft_plan,
            kv_cache=session.draft_cache
        )
        
        # Verify: Run base + experts on draft in parallel
        verification_logits = engine.forward_batch(
            draft_tokens,
            kv_cache=session.kv_cache,
            experts=plan.experts
        )
        
        # Accept/reject each draft token
        accepted_count = 0
        for i, draft_token in enumerate(draft_tokens):
            predicted_token = sample_token(verification_logits[i], plan.generation_params)
            
            if predicted_token == draft_token:
                accepted_count += 1
                generated_tokens.append(draft_token)
            else:
                # Reject: Use verified token and restart drafting
                generated_tokens.append(predicted_token)
                break
        
        # Speedup if acceptance rate > 70%
        if accepted_count == 0:
            # Fall back to normal generation
            return normal_generate(...)
    
    return GenerationResult(...)
```

---

## Post-Processing

### Validation

```python
def validate_output(output: str, task_type: str) -> ValidationResult:
    if task_type == "json_parsing":
        try:
            json.loads(output)
            return ValidationResult(valid=True)
        except json.JSONDecodeError as e:
            return ValidationResult(valid=False, error=str(e))
    
    elif task_type == "code_generation":
        # Check syntax (language-specific)
        return validate_code_syntax(output)
    
    else:
        # Generic validation (non-empty, reasonable length)
        return ValidationResult(valid=len(output.strip()) > 0)
```

### Metrics Collection

```python
@dataclass
class InferenceMetrics:
    job_id: str
    submitted_at: datetime
    router_latency_ms: float
    expert_load_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    
    experts_used: list[str]
    num_experts: int
    vram_peak_mb: int
    
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    
    success: bool
    error: str | None
    validation_passed: bool
```

### Cleanup

```python
def cleanup_session(session: Session, keep_hot: bool = True):
    # Clear KV cache (always)
    session.kv_cache.clear()
    
    # Release experts (unless keeping hot for next request)
    if not keep_hot:
        for expert in session.loaded_experts:
            session.expert_cache.evict(expert.name)
    
    # Free VRAM (if experts released)
    if not keep_hot:
        torch.cuda.empty_cache()
```

---

## API Design

### REST API

```http
POST /v1/infer HTTP/1.1
Content-Type: application/json

{
  "prompt": "Classify this JSON document",
  "body": "{\"type\": \"user\", \"name\": \"Alice\"}",
  "temperature": 0.3,
  "max_tokens": 512
}

---

HTTP/1.1 200 OK
Content-Type: application/json

{
  "job_id": "uuid-1234",
  "text": "Document type: User profile\nConfidence: 0.95",
  "tokens_generated": 12,
  "experts_used": ["json-parser", "english-basic", "classifier"],
  "latency_ms": 850,
  "finish_reason": "stop"
}
```

### Python SDK

```python
from expert_system import ExpertClient

client = ExpertClient(api_key="...")

# Simple inference
result = client.infer(
    prompt="Parse this JSON and extract fields",
    body='{"name": "Alice", "age": 30}',
    temperature=0.2
)

print(result.text)
print(f"Used experts: {result.experts_used}")
print(f"Latency: {result.latency_ms}ms")

# Streaming
for chunk in client.infer_stream(prompt="Write a story about..."):
    print(chunk.text, end="", flush=True)
```

### TypeScript SDK

```typescript
import { ExpertClient } from '@hivellm/expert-sdk';

const client = new ExpertClient({ apiKey: '...' });

// Async/await
const result = await client.infer({
  prompt: 'Generate Rust async function',
  temperature: 0.4,
  maxTokens: 2048
});

console.log(result.text);
console.log(`Experts: ${result.expertsUsed.join(', ')}`);

// Streaming
const stream = client.inferStream({ prompt: '...' });
for await (const chunk of stream) {
  process.stdout.write(chunk.text);
}
```

---

## Parallelism & Concurrency

### Multi-Job Execution

```
GPU has 16GB VRAM

Job 1: Base (0.5GB) + 3 experts (150MB) + KV (300MB) = 950MB
Job 2: Base (shared) + 4 experts (200MB) + KV (400MB) = 600MB
Job 3: Base (shared) + 2 experts (100MB) + KV (250MB) = 350MB

Total: 0.5 + 450 + 950 = 1.9GB → OK, run in parallel

Use separate CUDA streams:
stream_0: Job 1
stream_1: Job 2
stream_2: Job 3
```

### Expert Sharing

```python
# If two jobs use overlapping experts, share them
job1_experts = ["json-parser", "english", "neo4j"]
job2_experts = ["json-parser", "english", "postgres"]

shared = set(job1_experts) & set(job2_experts)
# shared = {"json-parser", "english"}

# Load once, use in both jobs
# VRAM savings: ~100MB
```

### Batching (Advanced)

```python
# If multiple jobs use SAME expert set, batch them
batch = [job1, job2, job3]  # All use same 3 experts

# Pad prompts to same length
padded_inputs = pad_sequences([j.input_ids for j in batch])

# Single forward pass
batch_logits = engine.forward_batch(
    padded_inputs,
    batch_kv_cache,
    experts=shared_experts
)

# Split outputs
for i, job in enumerate(batch):
    job.output = decode(batch_logits[i])
```

---

## Performance Optimization

### Pipeline Overlap

```
Timeline:
│
├─ Job 1: Reasoning (CPU)
│   └─ Job 1: Inference (GPU) ──────────────────┐
│       ├─ Job 2: Reasoning (CPU)                │
│       │   └─ Job 2: Inference (GPU) ────────┐  │
│       │       ├─ Job 3: Reasoning (CPU)     │  │
│       │       │                              │  │
│       │       │                              │  │
│       │       └─ Job 1: Done ◄───────────────┘  │
│       │           Job 3: Inference (GPU) ───────┘
│       │
│       └─ Job 2: Done ◄────────────────────────┘
```

Router runs on CPU while previous job uses GPU → hide latency

### Hot Cache Optimization

```python
# Keep top-N experts hot (LRU)
class ExpertCache:
    def __init__(self, max_hot: int = 20):
        self.max_hot = max_hot
        self.hot_experts = {}  # {name: weights}
        self.access_times = {}  # {name: timestamp}
    
    def get(self, name: str) -> Weights | None:
        if name in self.hot_experts:
            self.access_times[name] = time.time()  # Update LRU
            return self.hot_experts[name]
        return None
    
    def insert(self, name: str, weights: Weights):
        # Evict LRU if full
        if len(self.hot_experts) >= self.max_hot:
            lru_name = min(self.access_times, key=self.access_times.get)
            del self.hot_experts[lru_name]
            del self.access_times[lru_name]
        
        self.hot_experts[name] = weights
        self.access_times[name] = time.time()
```

---

## Error Handling

### Failure Scenarios

| Scenario | Stage | Recovery |
|----------|-------|----------|
| Invalid prompt | Reception | Return 400 error immediately |
| No suitable experts | Reasoning | Fall back to base model only |
| VRAM overflow | Preparation | Queue job or reduce expert count |
| Generation timeout | Inference | Return partial output + timeout flag |
| Validation failure | Post-process | Return output + validation error |

### Graceful Degradation

```python
try:
    plan = router.select_experts(prompt)
except NoExpertsFound:
    # Fall back to base model
    plan = ExpertPlan(experts=[], ...)

try:
    engine.load_experts(plan.experts)
except VRAMOverflow:
    # Reduce expert count
    plan.experts = plan.experts[:5]  # Keep top 5
    engine.load_experts(plan.experts)
```

---

## Future Enhancements

1. **Streaming output**: Return tokens as generated (SSE/WebSocket)
2. **Multi-turn conversations**: Maintain KV cache across turns
3. **Expert preloading**: Predict next experts based on conversation
4. **Dynamic batching**: Combine jobs on-the-fly
5. **Multi-GPU**: Split base model across GPUs
6. **Quantized KV cache**: INT8/INT4 KV cache for longer contexts

