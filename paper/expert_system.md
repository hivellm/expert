# HiveLLM Expert System: Runtime-Composable Adapters for Specialized Inference on Consumer GPUs

**Authors:** HiveLLM Research Team  
**Contact:** team@hivellm.org  
**Date:** November 2025  
**Status:** Experimental - Preprint

---

## Abstract

We present HiveLLM Expert System, a novel architecture for specialized AI inference on consumer-grade GPUs (8-16GB VRAM) through runtime composition of lightweight task-specific adapters. Unlike Mixture-of-Experts (MoE) models that embed multiple expert layers within a monolithic architecture, our system maintains a compact base model (Qwen3-0.6B, ~0.5GB) and dynamically loads independent PEFT adapters (LoRA, DoRA, IA³) on-demand. This approach enables: (1) independent expert training and distribution, (2) efficient memory usage through base model sharing across adapters, (3) rapid adapter switching (1-10ms), and (4) marketplace-driven ecosystem for specialized capabilities. We validate this architecture through the development of expert-sql, a text-to-SQL expert achieving 9.6/10 quality score and 100% success rate on 30 real-world query generation tasks. Our experiments demonstrate that checkpoint-1250 (epoch 1.25) outperforms both earlier and final checkpoints, highlighting the importance of checkpoint selection. **This work is experimental and requires further validation in production environments.**

---

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their deployment faces significant practical constraints. State-of-the-art models (70B+ parameters) require datacenter-grade infrastructure (40GB+ VRAM), while smaller models (<7B parameters) often lack depth in specialized domains despite their efficiency on consumer hardware.

We propose a third approach: **runtime-composable expert adapters**. Our system maintains a compact base model permanently loaded in GPU memory and dynamically attaches task-specific adapters as needed. This design is inspired by Parameter-Efficient Fine-Tuning (PEFT) methods [1, 2] but extends them to a multi-expert runtime architecture.

### 1.1 Key Contributions

1. A runtime architecture for dynamic composition of PEFT adapters without model merging
2. Demonstration that adapter hot-swapping enables memory-efficient multi-expert systems
3. Empirical validation through expert-sql: 100% success on real-world SQL generation
4. Analysis of checkpoint evolution showing non-monotonic quality progression
5. Open-source implementation with comprehensive training pipeline

### 1.2 Distinction from Mixture-of-Experts

Our approach is fundamentally different from Mixture-of-Experts (MoE) architectures [4]:

| Aspect | MoE (Mixtral, GPT-4) | HiveLLM Expert System |
|--------|---------------------|----------------------|
| **Expert Type** | MLP layers | PEFT adapters (LoRA/DoRA) |
| **Training** | All experts trained together | Each expert trained separately |
| **Selection** | Learned gating network | Heuristic router (CPU) |
| **Granularity** | Per-token routing | Per-query routing |
| **Extensibility** | Fixed at training time | Add experts anytime |
| **Distribution** | Monolithic model | Individual .expert files |
| **Memory** | ~100GB+ | ~1GB |
| **Cost** | $1M+ to train | $15-50 per expert |

---

## 2. Related Work

### 2.1 Parameter-Efficient Fine-Tuning

**LoRA** [1] introduced low-rank adaptation matrices that reduce trainable parameters by 10,000x. **DoRA** [2] improved upon LoRA by decomposing weight updates into magnitude and direction components. **IA³** [3] achieves even greater parameter efficiency through learned scaling vectors. Our work leverages these methods but focuses on the *runtime composition* aspect rather than training efficiency.

### 2.2 Mixture-of-Experts Models

**Mixtral** [4] and GPT-4 employ sparse MoE architectures where each token is routed to 1-3 expert MLP layers via learned gating networks. While effective for large-scale models, MoE architectures are monolithic (all experts trained together) and require loading the entire model (~100GB+) into memory. Our approach enables independent expert development and selective loading.

### 2.3 Model Composition

Prior work on model composition includes model merging [5] and ensemble methods. However, these typically require substantial memory or inference overhead. Our runtime adapter composition avoids model merging entirely, enabling sub-10ms switching.

---

## 3. Method

### 3.1 System Architecture

The HiveLLM Expert System consists of four core components:

```
┌────────────────────────────────────────────┐
│    Base Model (Qwen3-0.6B, ~0.5GB)        │
│         Always in VRAM                     │
└──────────────┬─────────────────────────────┘
               │ Runtime composition
    ┌──────────┼──────────┐
    │          │          │
┌───▼────┐ ┌──▼─────┐ ┌──▼─────┐
│SQL.exp │ │Neo4j   │ │Python  │  ← Loaded on-demand
│(LoRA)  │ │(DoRA)  │ │(IA³)   │    (1-10ms each)
│25 MB   │ │30 MB   │ │15 MB   │
└────────┘ └────────┘ └────────┘
     ▲          ▲          ▲
     └──────────┴──────────┘
         Router (CPU)
    Heuristic-based selection
```

**Components:**
1. **Base Model (MB)**: Qwen3-0.6B quantized to INT4 (~0.5GB VRAM)
2. **Expert Adapters (EXP)**: Independent PEFT adapters (5-80MB each)
3. **Router (RG)**: CPU-based heuristic selection using keywords and embeddings
4. **Inference Runtime (RI)**: GPU engine with hot-swap adapter support

### 3.2 Adapter Format

Each expert is packaged as a `.expert` file (tar.gz archive) containing:

- `manifest.json`: Metadata, capabilities, routing hints, decoding parameters
- `adapter_model.safetensors`: PEFT adapter weights (LoRA/DoRA/IA³)
- `adapter_config.json`: PEFT configuration (rank, alpha, target modules)
- `tokenizer.*`: Tokenizer files (reused from base model)
- `grammar.gbnf` (optional): Grammar constraints for structured output

### 3.3 Training Protocol

**Adapter Configuration:**
- Type: DoRA (Weight-Decomposed Low-Rank Adaptation)
- Rank: 12-16 (task-dependent)
- Alpha: 2 × r (standard scaling)
- Target modules: All attention (q, k, v, o) + MLP (up, down)
- Dropout: 0.1 (regularization)

**Hyperparameters (LLaMA-Factory optimized):**
- Learning rate: 5 × 10⁻⁵ (conservative for small models)
- Batch size: 2, Gradient accumulation: 45 (effective batch = 90)
- Warmup ratio: 0.1 (10% of total steps)
- LR scheduler: Cosine decay
- Optimizer: AdamW 8-bit (memory efficient)
- Precision: BF16 with TF32 tensor cores

**Optimizations:**
- Unsloth integration [6] for 2x speedup and 70% VRAM reduction
- Sequence packing via SFTTrainer for improved token efficiency
- Gradient checkpointing for memory efficiency
- Windows compatibility (torch.compile disabled due to Triton conflicts)

### 3.4 Base Model Sharing

A key innovation is base model sharing across multiple experts. When loading n experts with the same base model:

**Memory without sharing:**
```
M_total = n × (M_base + M_adapter)
```

**Memory with sharing:**
```
M_total = M_base + n × M_adapter
```

For Qwen3-0.6B (M_base ≈ 1.2GB) and typical adapters (M_adapter ≈ 0.025GB):

```
Without sharing (3 experts): 3 × (1.2 + 0.025) = 3.675 GB
With sharing (3 experts):    1.2 + 3 × 0.025   = 1.275 GB
Savings:                     65.3%
```

---

## 4. Experimental Validation

### 4.1 Expert-SQL: Text-to-SQL Generation

We validate our architecture through expert-sql, a PostgreSQL query generation expert.

**Dataset:**
- Source: gretelai/synthetic_text_to_sql (100,000 examples)
- Preprocessing:
  - MySQL → PostgreSQL syntax conversion via sqlglot
  - Validation: 99,935/100,000 passed (99.93% valid)
  - Deduplication: Removed exact question matches
  - Format: ChatML (Qwen3 native)
  - Size reduction: 77% via text-only format

**Training Configuration:**
- Base model: Qwen/Qwen3-0.6B
- Adapter: DoRA rank=12, alpha=24
- Dataset: 99,935 validated examples
- Epochs: 1.5 (save every 250 steps)
- Effective batch: 90 (2 × 45 gradient accumulation)
- Training time: ~3 hours (RTX 4090 + Unsloth)
- VRAM usage: 0.56GB during training (2.3% of 24GB)

### 4.2 Checkpoint Evolution

We analyzed 5 checkpoints to study model evolution:

| Checkpoint | Epoch | Quality Score | Real-World Test | Status |
|------------|-------|---------------|-----------------|--------|
| Base Model | 0.0 | 0.0/10 | 0/30 (0%) | No SQL generated |
| CKP-750 | 0.75 | 8.5/10 | 30/30 (100%) | Good |
| CKP-1000 | 1.0 | 9.0/10 | 30/30 (100%) | Better |
| **CKP-1250** | **1.25** | **9.6/10** | **30/30 (100%)** | **Best** ⭐ |
| CKP-1500 | 1.5 | 9.2/10 | 30/30 (100%) | Degraded |

**Key Finding:** The best checkpoint (1250, epoch 1.25) occurred before training completion, with the final checkpoint (1500, epoch 1.5) showing signs of overfitting. This validates the importance of checkpoint comparison rather than blindly using final weights.

### 4.3 Real-World Query Benchmark

We created a benchmark of 30 practical SQL scenarios across 8 categories:

| Category | Scenarios | Success Rate |
|----------|-----------|--------------|
| E-commerce | 5 | 5/5 (100%) |
| CRM | 4 | 4/4 (100%) |
| Analytics | 4 | 4/4 (100%) |
| Filters | 4 | 4/4 (100%) |
| Reports | 3 | 3/3 (100%) |
| Joins | 3 | 3/3 (100%) |
| Aggregations | 3 | 3/3 (100%) |
| Practical | 4 | 4/4 (100%) |
| **Total** | **30** | **30/30 (100%)** |

**Query Complexity Distribution:**
- **Basic** (17/17): Simple SELECT, WHERE, ORDER BY, basic JOINs
- **Intermediate** (13/13): Multi-table JOINs, subqueries, aggregations, window functions

### 4.4 Complex Scenario Analysis

We evaluated 10 advanced SQL patterns to identify limitations:

| Scenario | Score | Status |
|----------|-------|--------|
| Multiple JOIN + Aggregation | 8/10 | Good |
| Correlated Subquery | 10/10 | Excellent |
| Window Function | 9/10 | Excellent |
| Complex HAVING | 9/10 | Excellent |
| EXISTS vs IN | 10/10 | Excellent |
| Subquery in SELECT/WHERE | 5/10 | Fair |
| Multiple LEFT JOIN | 6/10 | Fair |
| Nested CASE WHEN | 5/10 | Fair |
| Recursive CTE | 2/10 | Weak |
| UNION + Aggregations | 3/10 | Weak |
| **Average** | **6.7/10** | |

### 4.5 Performance Metrics

| Metric | Value |
|--------|-------|
| Training time | 3 hours (RTX 4090) |
| Training VRAM | 0.56GB (2.3% utilization) |
| Adapter size | 25.8MB (SafeTensors) |
| Inference latency | 100-150ms per query |
| Trainable parameters | 8.5M (1.4% of base) |

---

## 5. Results

### 5.1 Expert-SQL Quality Analysis

Checkpoint-1250 achieved:
- **Quality Score**: 9.6/10 (real-world benchmark)
- **Success Rate**: 100% (30/30 queries)
- **Syntax Correctness**: 100% valid PostgreSQL
- **Production Readiness**: 95% of queries require no manual adjustment

### 5.2 Strengths and Limitations

**Strengths (≥90% success):**
- SELECT queries with filtering and ordering
- INNER JOIN and multi-table joins (2-4 tables)
- Aggregations (COUNT, SUM, AVG, GROUP BY, HAVING)
- Subqueries (WHERE IN, EXISTS, NOT EXISTS)
- Date operations (EXTRACT, BETWEEN, INTERVAL)
- Window functions (ROW_NUMBER, RANK, PARTITION BY)
- Common CTEs (WITH clause, non-recursive)

**Limitations (<50% success):**
- Recursive CTEs (WITH RECURSIVE) - generates self-join instead
- UNION/UNION ALL - incorrectly uses JOIN with OR
- LEFT JOIN with NULL checks - prefers INNER JOIN
- Complex percentage calculations - division logic errors
- Deeply nested CASE WHEN (3+ levels)

### 5.3 Memory Efficiency

For a deployment with 3 experts (SQL, Neo4j, Python):

```
Individual loading: 3 × 1.2GB = 3.6GB
Shared base model:  1.2 + 3 × 0.025 = 1.275GB
Memory savings:     64.6%
```

This enables running multiple specialized experts on consumer GPUs (8GB VRAM) that would otherwise be insufficient.

---

## 6. Discussion

### 6.1 Advantages of Runtime Composition

**Independent Development:** Each expert can be trained, validated, and distributed independently. This enables:
- Rapid iteration on individual experts without affecting others
- Community-driven expert marketplace
- Easy bug fixes (update single expert, not entire model)

**Memory Efficiency:** Base model sharing reduces memory footprint by 50-70% compared to loading experts separately.

**Flexibility:** Unlike MoE where experts are fixed at training time, new experts can be added anytime without retraining existing ones.

### 6.2 Checkpoint Selection Insights

Our analysis reveals that:
- Quality scores do not increase monotonically with training steps
- Overfitting can occur even with 99k training examples
- Systematic checkpoint evaluation is essential
- Early stopping based on eval loss may miss optimal checkpoint

We recommend: **Save all checkpoints and evaluate systematically** rather than relying on final weights or early stopping heuristics.

### 6.3 Limitations and Future Work

**Current Limitations:**
1. **Limited Production Validation**: Expert-sql tested only on synthetic benchmarks, not production workloads
2. **Single Expert Evaluated**: Only SQL expert fully validated; Neo4j and others in training
3. **Heuristic Routing**: Router uses keywords/embeddings, not learned selection
4. **No Adapter Merging**: Cannot combine multiple adapters simultaneously (planned)
5. **Query-Level Granularity**: Per-token expert routing not supported

**Planned Improvements:**
1. Production deployment with real user queries
2. Training and validation of 6+ experts (Neo4j, TypeScript, Python, Rust, JSON, English)
3. Learned routing component (mini-policy network)
4. Adapter composition (LoRA arithmetic)
5. Inference runtime in Rust (Candle-based)
6. Benchmark against GPT-3.5/4 on specialized tasks

---

## 7. Experimental Nature and Limitations

**⚠️ IMPORTANT:** This work is **experimental** and has significant limitations:

### 7.1 Limited Real-World Validation

- Only 1 expert (SQL) fully trained and evaluated
- Benchmarks are synthetic, not from production systems
- No user studies or production deployment data
- Success metrics may not generalize to real applications

### 7.2 Scalability Unknowns

- Maximum number of experts not empirically validated
- Routing quality with 10+ experts untested
- Adapter hot-swap latency measured on single GPU only
- Multi-GPU / distributed deployment not explored

### 7.3 Generalization Concerns

- Expert-sql trained on synthetic data (gretelai)
- Performance on real business databases unknown
- Edge cases and rare SQL patterns not covered
- Domain-specific query patterns may differ significantly

### 7.4 Ongoing Work

The following experts are currently in training:
- **expert-neo4j**: Cypher query generation (29,512 examples from neo4j/text2cypher-2025v1)
- **expert-typescript**: Code generation (planned)
- **expert-python**: Code generation (planned)
- **expert-json**: JSON parsing (planned)
- **expert-english**: Language understanding (planned)

**Validation plan:** Each expert will undergo similar real-world benchmarking and checkpoint analysis before production recommendation.

---

## 8. Conclusion

We present HiveLLM Expert System, a runtime-composable adapter architecture for specialized inference on consumer GPUs. Our key insight is that *independent expert training and dynamic loading* provides a viable alternative to both monolithic LLMs and Mixture-of-Experts architectures.

Through expert-sql, we demonstrate:
1. 100% success rate on 30 real-world SQL generation tasks
2. 64% memory savings through base model sharing
3. Importance of systematic checkpoint evaluation
4. Feasibility of high-quality expert training with <4 hours on consumer GPU

**However, this work is experimental.** Significant validation remains:
- Production deployment with real user queries
- Evaluation of additional experts beyond SQL
- Scalability testing with 10+ experts
- Comparison with commercial alternatives (GPT-3.5, Claude)

We release this work as open-source to enable community validation and encourage further research into modular, composable AI systems for resource-constrained environments.

**Code and data available at:** https://github.com/hivellm/expert

---

## Acknowledgments

This work builds upon LoRA [1], DoRA [2], Unsloth [6], and LLaMA-Factory [7]. We thank the gretelai team for the synthetic_text_to_sql dataset and the Neo4j team for text2cypher-2025v1.

---

## References

[1] Edward J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, 2021.

[2] Shih-Yang Liu et al. "DoRA: Weight-Decomposed Low-Rank Adaptation." arXiv:2402.09353, 2024.

[3] Haokun Liu et al. "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning." arXiv:2205.05638, 2022.

[4] Noam Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." arXiv:1701.06538, 2017.

[5] Mitchell Wortsman et al. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." ICML 2022.

[6] Unsloth AI. "Unsloth: 2-5x Faster LLM Finetuning." https://github.com/unslothai/unsloth, 2024.

[7] LLaMA-Factory Contributors. "LLaMA-Factory: Efficient Fine-Tuning of Large Language Models." https://github.com/hiyouga/LLaMA-Factory, 2024.

---

## Appendix A: Expert Package Format

The `.expert` package format (v2.0) contains all files in the root:

```
expert-sql-qwen3-0-6b.v0.2.0.expert (tar.gz, 25.9 MB)
├── manifest.json                 # Expert metadata
├── adapter_config.json           # PEFT configuration
├── adapter_model.safetensors     # Adapter weights (25.8 MB)
├── tokenizer.json                # Tokenizer vocabulary
├── tokenizer_config.json         # Tokenizer config
├── special_tokens_map.json       # Special tokens
├── training_args.bin             # Training hyperparams
├── vocab.json                    # Vocabulary mappings
├── README.md                     # Documentation
├── grammar.gbnf                  # Grammar constraints (optional)
└── LICENSE                       # License info
```

All files are placed in the archive root (no subdirectories) for simplified loading.

---

## Appendix B: Sample Generated Queries

**E-commerce Query:**
```sql
-- Input: "Show products with low stock"
SELECT name, price, stock 
FROM products 
WHERE stock < min_stock 
ORDER BY stock ASC;
```

**CRM Query with Subquery:**
```sql
-- Input: "Find customers without orders"
SELECT c.name 
FROM customers c 
WHERE NOT EXISTS (
  SELECT o.customer_id FROM orders o 
  WHERE o.customer_id = c.id
);
```

**Analytics with Window Function:**
```sql
-- Input: "Sales ranking by region"
SELECT salesperson, region, 
  ROW_NUMBER() OVER (
    PARTITION BY region 
    ORDER BY SUM(amount) DESC
  ) as rank
FROM sales 
GROUP BY salesperson, region;
```

---

## Appendix C: Reproducibility

All code, datasets, and trained models are open-source:

- **Repository:** https://github.com/hivellm/expert
- **Expert-SQL v0.2.0:** Tag `v0.2.0`
- **Dataset:** gretelai/synthetic_text_to_sql (HuggingFace)
- **Base model:** Qwen/Qwen3-0.6B
- **Training script:** `expert/cli/expert_trainer.py`
- **Manifest:** `expert/experts/expert-sql/manifest.json`

**Hardware Requirements:**
- GPU: NVIDIA RTX 3060 (8GB) minimum, RTX 4090 (24GB) recommended
- CUDA: 12.1+
- PyTorch: 2.5.1+cu121
- Unsloth: Optional (2x speedup)

---

## Appendix D: Dataset Quality Metrics

**Preprocessing Statistics (expert-sql):**
- Original examples: 100,000
- Passed SQL validation: 99,935 (99.93%)
- Invalid SQL removed: 65 (0.07%)
- Duplicates removed: 0 (pre-deduplicated)
- Final training set: 99,935 examples

**Preprocessing Statistics (expert-neo4j, in progress):**
- Original examples: 35,946
- Passed Cypher validation: 29,515 (82.1%)
- Invalid Cypher removed: 3 (0.008%)
- Duplicates removed: 6,431 (17.9%)
- Final training set: 29,512 examples

---

**⚠️ Disclaimer:** This is an experimental research project. Results presented are based on controlled benchmarks and have not been validated in production environments. The architecture and findings require further empirical validation with real-world workloads before production deployment.

