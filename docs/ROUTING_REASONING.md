# Routing & Reasoning

> Router (RG) architecture, expert selection logic, and parameter tuning strategies

## Overview

The Router/Reasoning (RG) component is the "brain" of the Expert System. It analyzes incoming prompts, selects the optimal set of experts (up to 10), determines their composition order, and tunes generation parameters for the specific task at hand.

**Key responsibilities:**
1. Fast prompt analysis (heuristics + embeddings)
2. Expert selection from local registry
3. Composition ordering (which expert applies first)
4. Parameter tuning (temperature, max_tokens, etc.)
5. Continuous learning from outcomes

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Prompt                         │
│              (+ template, body, hints)                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │     Feature Extraction (parallel)  │
        │                                    │
        │  1. Heuristics (~1ms)              │
        │     - Language detection           │
        │     - Format detection (JSON, etc) │
        │     - Keyword matching             │
        │                                    │
        │  2. Embeddings (~10ms)             │
        │     - MiniLM semantic embedding    │
        │     - ANN search in expert index   │
        │                                    │
        │  3. Mini-policy (optional, ~50ms)  │
        │     - Base model classifies task   │
        │     - Ranks candidate experts      │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │       Expert Selection             │
        │                                    │
        │  - Score candidates                │
        │  - Filter incompatibilities        │
        │  - Select top-K (K ≤ 10)           │
        │  - Order by composition logic      │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │      Parameter Tuning              │
        │                                    │
        │  - Temperature (task-dependent)    │
        │  - Max tokens                      │
        │  - Top-p, top-k, min-p             │
        │  - Repetition penalty              │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │         Expert Plan                │
        │  {                                 │
        │    experts: [...],                 │
        │    order: [...],                   │
        │    params: {...}                   │
        │  }                                 │
        └────────────────────────────────────┘
```

---

## Feature Extraction

### 1. Heuristic Features (Fast Path)

Lightning-fast pattern matching for obvious cases.

```python
class HeuristicExtractor:
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.format_patterns = {
            "json": [r"\{.*\}", r'"[^"]+"\s*:', "parse json", "json"],
            "xml": [r"<[^>]+>", "parse xml", "xml"],
            "cypher": [r"MATCH\s*\(", r"CREATE\s*\(", "neo4j", "cypher"],
            "sql": [r"SELECT\s+", r"INSERT\s+INTO", "sql", "query"],
        }
        self.task_patterns = {
            "classification": ["classify", "categorize", "label"],
            "parsing": ["parse", "extract", "read"],
            "generation": ["generate", "create", "write"],
            "translation": ["translate", "convert"],
            "validation": ["validate", "check", "verify"],
        }
    
    def extract(self, prompt: str, body: str | None) -> dict:
        features = {}
        
        # Detect language
        full_text = f"{prompt} {body or ''}"
        features["language"] = self.language_detector.detect(full_text)
        
        # Detect formats
        features["formats"] = []
        for format_name, patterns in self.format_patterns.items():
            if any(self._matches(pattern, full_text) for pattern in patterns):
                features["formats"].append(format_name)
        
        # Detect task type
        features["tasks"] = []
        for task_name, keywords in self.task_patterns.items():
            if any(kw in prompt.lower() for kw in keywords):
                features["tasks"].append(task_name)
        
        # Detect technologies
        features["technologies"] = self._detect_technologies(full_text)
        
        return features
    
    def _detect_technologies(self, text: str) -> list[str]:
        tech_keywords = {
            "neo4j": ["neo4j", "graph database", "cypher"],
            "postgres": ["postgres", "postgresql", "psql"],
            "rust": ["rust", "cargo", ".rs"],
            "python": ["python", ".py", "import "],
        }
        
        detected = []
        text_lower = text.lower()
        
        for tech, keywords in tech_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(tech)
        
        return detected
    
    def _matches(self, pattern: str, text: str) -> bool:
        if pattern.startswith("r"):  # Regex
            return bool(re.search(pattern[1:], text, re.IGNORECASE))
        else:  # Keyword
            return pattern.lower() in text.lower()
```

**Example:**

```python
prompt = "Parse this JSON and extract Neo4j node types"
body = '{"type": "Person", "name": "Alice"}'

heuristics = extractor.extract(prompt, body)
# {
#   "language": "en",
#   "formats": ["json"],
#   "tasks": ["parsing"],
#   "technologies": ["neo4j"]
# }
```

### 2. Embedding-Based Search

Semantic similarity search using lightweight embeddings.

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingSearch:
    def __init__(self, expert_index_path: str):
        # Load lightweight embedding model (~50MB)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load expert index (FAISS for fast ANN search)
        self.index = faiss.read_index(f"{expert_index_path}/experts.faiss")
        self.expert_metadata = load_json(f"{expert_index_path}/metadata.json")
    
    def search(self, prompt: str, k: int = 20, threshold: float = 0.6) -> list[dict]:
        # Embed prompt (~10ms)
        query_embedding = self.model.encode([prompt])[0]
        
        # Search index (ANN, ~1-2ms)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )
        
        # Filter by threshold
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - (dist / 2)  # Convert L2 distance to cosine similarity
            
            if similarity >= threshold:
                expert = self.expert_metadata[idx]
                expert["similarity"] = similarity
                results.append(expert)
        
        return results
```

**Building the expert index:**

```python
def build_expert_index(experts: list[dict], output_path: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for all experts
    expert_texts = []
    for expert in experts:
        # Combine expert metadata for embedding
        text = f"{expert['name']} {expert['description']} {' '.join(expert['capabilities'])}"
        expert_texts.append(text)
    
    embeddings = model.encode(expert_texts)
    
    # Build FAISS index
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)  # Simple L2 index
    index.add(embeddings.astype(np.float32))
    
    # Save index and metadata
    faiss.write_index(index, f"{output_path}/experts.faiss")
    save_json(experts, f"{output_path}/metadata.json")
```

### 3. Mini-Policy (LLM-based Classification)

Use base model itself to classify complex tasks.

```python
class MiniPolicy:
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
    
    def classify_and_rank(
        self,
        prompt: str,
        candidate_experts: list[dict],
        max_experts: int = 10
    ) -> list[str]:
        
        # Format expert list for LLM
        expert_list = "\n".join([
            f"{i+1}. {exp['name']}: {exp['description']}"
            for i, exp in enumerate(candidate_experts)
        ])
        
        # Prompt for base model
        policy_prompt = f"""Given this task:
"{prompt}"

Available experts:
{expert_list}

Select up to {max_experts} most relevant experts and rank them by importance.
Output format: expert_name_1,expert_name_2,expert_name_3

Selected experts:"""
        
        # Generate ranking (~50ms)
        inputs = self.tokenizer(policy_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,  # Low temp for consistency
            do_sample=True
        )
        
        # Parse output
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        selected_names = [name.strip() for name in generated.split(",")]
        
        return selected_names[:max_experts]
```

---

## Expert Selection

### Scoring Function

Combine multiple signals to score each candidate expert.

```python
def compute_expert_score(
    expert: dict,
    heuristics: dict,
    similarity: float,
    vram_budget: int,
    historical_stats: dict
) -> float:
    
    # Weight factors
    W_SEMANTIC = 0.40
    W_HEURISTIC = 0.25
    W_SUCCESS_RATE = 0.20
    W_VRAM = 0.10
    W_POPULARITY = 0.05
    
    # 1. Semantic similarity (from embeddings)
    semantic_score = similarity  # 0-1
    
    # 2. Heuristic match
    heuristic_score = 0.0
    
    # Language match
    if expert.get("language") == heuristics.get("language"):
        heuristic_score += 0.3
    
    # Format match
    expert_formats = set(expert.get("capabilities", []))
    prompt_formats = set(f"format:{fmt}" for fmt in heuristics.get("formats", []))
    if expert_formats & prompt_formats:
        heuristic_score += 0.4
    
    # Technology match
    expert_techs = set(expert.get("technologies", []))
    prompt_techs = set(heuristics.get("technologies", []))
    if expert_techs & prompt_techs:
        heuristic_score += 0.3
    
    heuristic_score = min(1.0, heuristic_score)
    
    # 3. Historical success rate
    success_rate = historical_stats.get(expert["name"], {}).get("success_rate", 0.5)
    
    # 4. VRAM cost (penalty for large experts)
    vram_cost = expert.get("perf", {}).get("vram_mb_overhead", 50)
    vram_score = max(0, 1 - (vram_cost / vram_budget))
    
    # 5. Popularity (how often used successfully)
    usage_count = historical_stats.get(expert["name"], {}).get("usage_count", 0)
    popularity_score = min(1.0, usage_count / 1000)  # Cap at 1000 uses
    
    # Weighted sum
    total_score = (
        W_SEMANTIC * semantic_score +
        W_HEURISTIC * heuristic_score +
        W_SUCCESS_RATE * success_rate +
        W_VRAM * vram_score +
        W_POPULARITY * popularity_score
    )
    
    return total_score
```

### Incompatibility Filtering

```python
def filter_incompatible(candidates: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Remove experts that conflict with each other"""
    
    filtered = []
    selected_names = set()
    
    for score, expert in candidates:
        incompatible_with = expert.get("constraints", {}).get("incompatible_with", [])
        
        # Check if any already-selected expert is incompatible
        conflicts = False
        for pattern in incompatible_with:
            for selected in selected_names:
                if matches_version_pattern(selected, pattern):
                    conflicts = True
                    break
            if conflicts:
                break
        
        if not conflicts:
            filtered.append((score, expert))
            selected_names.add(expert["name"])
    
    return filtered

def matches_version_pattern(expert_name: str, pattern: str) -> bool:
    """Check if expert matches incompatibility pattern
    
    Examples:
      - "old-json@1.0.0" matches "old-json@1.0.0"
      - "old-json@1.5.0" matches "old-json@<=2.0.0"
      - "old-json@3.0.0" does NOT match "old-json@<3.0.0"
    """
    import semver
    
    if "@" in pattern:
        name, version_spec = pattern.split("@", 1)
        expert_base_name, expert_version = expert_name.split("@", 1)
        
        if name != expert_base_name:
            return False
        
        # Parse version constraints
        if version_spec == "*":
            return True
        elif version_spec.startswith("<="):
            return semver.compare(expert_version, version_spec[2:]) <= 0
        elif version_spec.startswith(">="):
            return semver.compare(expert_version, version_spec[2:]) >= 0
        elif version_spec.startswith("<"):
            return semver.compare(expert_version, version_spec[1:]) < 0
        elif version_spec.startswith(">"):
            return semver.compare(expert_version, version_spec[1:]) > 0
        else:
            return expert_version == version_spec
    else:
        return expert_name.startswith(pattern)
```

### Composition Ordering

Determine which expert applies first (order matters).

```python
def order_experts(experts: list[dict], heuristics: dict) -> list[dict]:
    """Order experts by logical dependency
    
    Example: JSON parsing should happen before domain-specific logic
    """
    
    # Priority tiers (lower = applied first)
    TIER_PRIORITY = {
        "format": 1,      # JSON, XML parsing first
        "language": 2,    # Language understanding next
        "tech": 3,        # Technology-specific logic
        "domain": 4,      # Domain knowledge last
        "task": 5,        # Task execution (classification, etc.)
    }
    
    def get_tier(expert: dict) -> int:
        capabilities = expert.get("capabilities", [])
        
        if any(cap.startswith("format:") for cap in capabilities):
            return TIER_PRIORITY["format"]
        elif any(cap.startswith("language:") for cap in capabilities):
            return TIER_PRIORITY["language"]
        elif any(cap.startswith("tech:") for cap in capabilities):
            return TIER_PRIORITY["tech"]
        elif any(cap.startswith("domain:") for cap in capabilities):
            return TIER_PRIORITY["domain"]
        else:
            return TIER_PRIORITY["task"]
    
    # Sort by tier, then by score (if available)
    experts_with_tier = [(get_tier(exp), exp) for exp in experts]
    experts_with_tier.sort(key=lambda x: x[0])
    
    return [exp for _, exp in experts_with_tier]
```

---

## Parameter Tuning

### Temperature Selection

```python
def select_temperature(task_type: str, user_hint: float | None) -> float:
    """Select appropriate temperature for task type"""
    
    if user_hint is not None:
        return user_hint
    
    # Default temperatures by task
    TASK_TEMPERATURES = {
        "json_parsing": 0.2,
        "xml_parsing": 0.2,
        "sql_generation": 0.3,
        "code_generation": 0.4,
        "classification": 0.3,
        "extraction": 0.3,
        "summarization": 0.6,
        "translation": 0.5,
        "creative_writing": 0.8,
        "brainstorming": 0.9,
    }
    
    return TASK_TEMPERATURES.get(task_type, 0.6)  # Default: 0.6
```

### Max Tokens Selection

```python
def select_max_tokens(task_type: str, input_length: int, user_hint: int | None) -> int:
    """Select max output tokens"""
    
    if user_hint is not None:
        return user_hint
    
    # Heuristics by task
    if task_type in ["classification", "extraction"]:
        # Short outputs
        return min(512, input_length // 2)
    
    elif task_type in ["json_parsing", "validation"]:
        # Roughly same length as input
        return min(2048, input_length)
    
    elif task_type in ["code_generation", "summarization"]:
        # Moderate length
        return 2048
    
    elif task_type == "creative_writing":
        # Long outputs
        return 8192
    
    else:
        # Default
        return 2048
```

### Other Parameters

```python
def tune_generation_params(
    task_type: str,
    heuristics: dict,
    user_prefs: dict
) -> dict:
    """Complete parameter tuning"""
    
    params = {
        "temperature": select_temperature(task_type, user_prefs.get("temperature")),
        "max_tokens": select_max_tokens(task_type, heuristics.get("input_length", 100), user_prefs.get("max_tokens")),
        "top_p": user_prefs.get("top_p", 0.95),
        "top_k": user_prefs.get("top_k", None),
        "min_p": user_prefs.get("min_p", None),
        "repetition_penalty": 1.0 if task_type == "json_parsing" else 1.1,
    }
    
    return params
```

---

## Caching & Learning

### Decision Cache

Cache routing decisions for similar prompts.

```python
class RouterCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}  # {prompt_hash: ExpertPlan}
        self.max_size = max_size
        self.access_times = {}  # LRU tracking
    
    def get(self, prompt: str) -> ExpertPlan | None:
        # Fuzzy hash (normalize prompt)
        prompt_hash = self._fuzzy_hash(prompt)
        
        if prompt_hash in self.cache:
            self.access_times[prompt_hash] = time.time()
            return self.cache[prompt_hash]
        
        return None
    
    def put(self, prompt: str, plan: ExpertPlan):
        prompt_hash = self._fuzzy_hash(prompt)
        
        # Evict LRU if full
        if len(self.cache) >= self.max_size:
            lru_hash = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_hash]
            del self.access_times[lru_hash]
        
        self.cache[prompt_hash] = plan
        self.access_times[prompt_hash] = time.time()
    
    def invalidate(self, prompt: str):
        """Remove from cache if failed"""
        prompt_hash = self._fuzzy_hash(prompt)
        if prompt_hash in self.cache:
            del self.cache[prompt_hash]
            del self.access_times[prompt_hash]
    
    def _fuzzy_hash(self, prompt: str) -> str:
        # Normalize: lowercase, remove extra whitespace, remove variables
        normalized = re.sub(r'\s+', ' ', prompt.lower().strip())
        # Remove likely variable parts (numbers, UUIDs, etc.)
        normalized = re.sub(r'\d+', 'N', normalized)
        normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', normalized)
        
        return hashlib.md5(normalized.encode()).hexdigest()
```

### Offline Learning

Update router based on historical outcomes.

```python
class RouterLearner:
    def __init__(self, telemetry_db):
        self.db = telemetry_db
    
    def update_statistics(self):
        """Compute success rates for each expert"""
        
        stats = {}
        
        # Query telemetry
        jobs = self.db.query("""
            SELECT expert_name, success, latency_ms
            FROM inference_jobs
            WHERE created_at > NOW() - INTERVAL '7 days'
        """)
        
        # Aggregate by expert
        for job in jobs:
            expert = job["expert_name"]
            
            if expert not in stats:
                stats[expert] = {
                    "total": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_latency": 0
                }
            
            stats[expert]["total"] += 1
            
            if job["success"]:
                stats[expert]["successes"] += 1
            else:
                stats[expert]["failures"] += 1
            
            stats[expert]["avg_latency"] += job["latency_ms"]
        
        # Compute success rates
        for expert, data in stats.items():
            if data["total"] > 0:
                data["success_rate"] = data["successes"] / data["total"]
                data["avg_latency"] /= data["total"]
        
        # Save updated stats
        save_json(stats, "expert_statistics.json")
        
        return stats
    
    def train_classifier(self, training_data: list[dict]):
        """Train lightweight classifier to predict expert usefulness
        
        Training data format:
        [
          {"prompt": "...", "expert": "json-parser", "success": True},
          ...
        ]
        """
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Feature extraction
        X = []  # Features
        y = []  # Labels (1=use expert, 0=don't use)
        
        for example in training_data:
            features = self._extract_features(example["prompt"])
            X.append(features)
            y.append(1 if example["success"] else 0)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)
        
        # Save model
        joblib.dump(clf, "router_classifier.pkl")
        
        return clf
```

---

## Complete Example

```python
class Router:
    def __init__(self):
        self.heuristic_extractor = HeuristicExtractor()
        self.embedding_search = EmbeddingSearch("expert_index/")
        self.mini_policy = MiniPolicy(base_model, tokenizer)
        self.cache = RouterCache()
        self.stats = load_json("expert_statistics.json")
    
    def plan(self, request: InferenceRequest) -> ExpertPlan:
        # Check cache first
        cached = self.cache.get(request.prompt)
        if cached and not request.force_experts:
            return cached
        
        # 1. Extract features
        heuristics = self.heuristic_extractor.extract(request.prompt, request.body)
        
        # 2. Embedding search
        candidates = self.embedding_search.search(request.prompt, k=30)
        
        # 3. Score candidates
        scored = []
        for candidate in candidates:
            score = compute_expert_score(
                candidate,
                heuristics,
                candidate["similarity"],
                vram_budget=8000,  # 8GB available
                historical_stats=self.stats
            )
            scored.append((score, candidate))
        
        # 4. Filter incompatibilities
        scored = filter_incompatible(scored)
        
        # 5. Select top-K
        scored.sort(reverse=True, key=lambda x: x[0])
        max_experts = request.max_experts or 10
        selected = [exp for _, exp in scored[:max_experts]]
        
        # 6. Order by composition logic
        ordered = order_experts(selected, heuristics)
        
        # 7. Tune parameters
        task_type = heuristics["tasks"][0] if heuristics["tasks"] else "default"
        params = tune_generation_params(task_type, heuristics, request.user_prefs)
        
        # 8. Create plan
        plan = ExpertPlan(
            experts=ordered,
            composition_order=[exp["name"] for exp in ordered],
            generation_params=params,
            reasoning_trace=f"Selected {len(ordered)} experts based on {task_type} task"
        )
        
        # Cache decision
        self.cache.put(request.prompt, plan)
        
        return plan
```

---

## Performance Optimization

### Latency Targets

| Component | Target | Typical |
|-----------|--------|---------|
| Heuristics | <1ms | 0.5ms |
| Embeddings | <15ms | 10ms |
| ANN search | <3ms | 1-2ms |
| Mini-policy | <100ms | 50ms |
| **Total (no policy)** | **<20ms** | **12ms** |
| **Total (with policy)** | **<120ms** | **60ms** |

### Parallel Execution

Run router while previous job is still inferring on GPU.

```python
async def overlap_routing_with_inference(jobs: list[Job]):
    async def process_job(job):
        # Route next job while current job is on GPU
        next_job = get_next_from_queue()
        
        # Parallel: routing (CPU) + inference (GPU)
        plan_future = asyncio.create_task(router.plan(next_job.request))
        inference_result = await gpu_engine.infer(job.plan)
        
        next_plan = await plan_future
        
        return inference_result, next_plan
```

---

## Best Practices

1. **Cache aggressively**: Most prompts are similar to previous ones
2. **Start with heuristics**: They're fast and often sufficient
3. **Use mini-policy sparingly**: Only for ambiguous cases
4. **Update statistics weekly**: Keep success rates current
5. **Monitor router accuracy**: Track how often selected experts produce valid output
6. **Provide debugging info**: Include reasoning_trace in plans for troubleshooting
7. **Balance exploration/exploitation**: Occasionally try new expert combinations

---

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for router's role in the system
- See [EXECUTION_PIPELINE.md](EXECUTION_PIPELINE.md) for integration with inference
- See [PERFORMANCE.md](PERFORMANCE.md) for optimization strategies

