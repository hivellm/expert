# Train First Six Experts - Implementation Tasks

**Status**: ✅ **1/6 Complete**, ⏳ **3/6 In Progress** (2025-11-05)

## 1. Expert: expert-json (JSON Parser) - ✅ COMPLETE

- [x] 1.1 Create expert structure in /expert/experts/expert-json ✅
- [x] 1.2 Create manifest.json v0.0.2 ✅
- [x] 1.3 Complete dataset (5 categories: json_repair, json_style_strict, json_transform, schema_generate, text_to_json) ✅
- [x] 1.4 Dataset generated (~8000+ examples across all categories) ✅
- [x] 1.5 Trained with DoRA r=12 (weights/qwen3-06b/final/) ✅
- [x] 1.6 Test suite created (test_comparison.py, test_expert.py, test_hard.py, test_json_parsing.py) ✅
- [ ] 1.7 Run: expert-cli validate --expert expert-json ⏳
- [ ] 1.8 Run: expert-cli package --manifest manifest.json ⏳
- [ ] 1.9 Verify accuracy >95% on test cases ⏳

## 2. Expert: expert-sql (SQL Generator) - ⏳ 70% COMPLETE

- [x] 2.1 Create /expert/experts/expert-sql/ ✅
- [x] 2.2 Create manifest.json v0.0.1 ✅
  - DoRA r=12, alpha=24
  - Unsloth integration (2x faster, 70% less VRAM)
  - Capabilities: ["tech:sql", "language:postgres", "database"]
  - Grammar: grammar.gbnf for PostgreSQL syntax
- [x] 2.3 Create test cases (test_comparison.py, test_expert.py, test_hard.py, test_advanced.py, test_comprehensive.py) ✅
- [x] 2.4 Dataset processed (train.jsonl: 99,935 examples from gretelai/synthetic_text_to_sql) ✅
  - MySQL→PostgreSQL syntax conversion via sqlglot
  - Deduplication and validation
  - ChatML format
- [x] 2.5 Training pipeline ready (expert-cli train) ✅
- [ ] 2.6 Complete training (in progress) ⏳
- [ ] 2.7 Validate SQL quality >90% ⏳
- [ ] 2.8 Package as .expert ⏳

## 3. Expert: expert-neo4j (Neo4j Cypher) - ⏳ 40% COMPLETE

- [x] 3.1 Create /expert/experts/expert-neo4j/ ✅
- [x] 3.2 Create manifest.json v0.0.1 ✅
  - DoRA r=12, alpha=24
  - Capabilities: ["tech:neo4j", "language:cypher", "graph-database"]
  - Grammar: grammar.gbnf for Cypher syntax
- [x] 3.3 Create test cases (test_comparison.py, test_expert.py, test_hard.py) ✅
- [ ] 3.4 Generate dataset (need ~6k examples) ⏳
- [ ] 3.5 Train DoRA r=12 ⏳
- [ ] 3.6 Validate with real Cypher queries ⏳
- [ ] 3.7 Package as .expert ⏳

## 4. Expert: expert-typescript (TypeScript) - ⏳ 30% COMPLETE

- [x] 4.1 Create /expert/experts/expert-typescript/ ✅
- [x] 4.2 Create manifest.json v0.0.1 ✅
  - DoRA r=12, alpha=24
  - Capabilities: ["tech:typescript", "code-analysis", "language:typescript"]
- [x] 4.3 Create test cases (test_comparison.py, test_expert.py) ✅
- [ ] 4.4 Generate dataset (~8k examples) ⏳
- [ ] 4.5 Train DoRA r=12 ⏳
- [ ] 4.6 Validate on TypeScript code ⏳
- [ ] 4.7 Package as .expert ⏳

## 5. Expert: expert-python (Python Code) - ⏳ 0% COMPLETE

- [ ] 5.1 Create /expert/experts/expert-python/ ⏳
- [ ] 5.2 Create manifest.json with:
  - version: 0.0.1
  - DoRA r=12, alpha=24
  - capabilities: ["tech:python", "code-analysis", "language:python"]
- [ ] 5.3 Create test cases (code analysis, generation) ⏳
- [ ] 5.4 Generate dataset (~8k examples) ⏳
- [ ] 5.5 Train DoRA r=12 ⏳
- [ ] 5.6 Validate on Python code ⏳
- [ ] 5.7 Package as .expert ⏳

## 6. Expert: expert-rust (Rust Code) - ⏳ 0% COMPLETE

- [ ] 6.1 Create /expert/experts/expert-rust/ ⏳
- [ ] 6.2 Create manifest.json with:
  - version: 0.0.1
  - DoRA r=12, alpha=24
  - capabilities: ["tech:rust", "code-analysis", "language:rust"]
- [ ] 6.3 Create test cases (Rust patterns, async, ownership) ⏳
- [ ] 6.4 Generate dataset (~7k examples) ⏳
- [ ] 6.5 Train DoRA r=12 ⏳
- [ ] 6.6 Validate on Rust code ⏳
- [ ] 6.7 Package as .expert ⏳

## 7. Multi-Expert Runtime Testing

- [x] 7.1 ExpertManager implemented (hot-swap, LRU eviction, pre-loading) ✅
- [x] 7.2 KeywordRouter and EmbeddingRouter implemented ✅
- [x] 7.3 Test structure created (test_multi_expert.rs, test_hot_swap.rs, test_integration.rs) ✅
- [ ] 7.4 Test loading expert-json alone ⏳
- [ ] 7.5 Test loading multiple experts simultaneously ⏳
- [ ] 7.6 Verify dependency resolution works ⏳
- [ ] 7.7 Measure total VRAM with multiple experts loaded ⏳
- [ ] 7.8 Benchmark hot-swap latency ⏳

## 8. Documentation & Finalization

- [x] 8.1 Update ROADMAP.md with current status ✅
- [x] 8.2 Create CHANGELOG.md with training improvements ✅
- [x] 8.3 Update README.md with Unsloth integration ✅
- [x] 8.4 Document training parameters and optimizations ✅
- [ ] 8.5 Update /expert/experts/README.md (mark completed experts) ⏳
- [ ] 8.6 Document actual training times and VRAM usage ⏳
- [ ] 8.7 Add usage examples to each expert's README ⏳
- [ ] 8.8 Create performance comparison report ⏳

---

## Current Priorities (Next Steps)

1. **Complete expert-sql training** (in progress)
   - Monitor checkpoint quality
   - Validate SQL generation accuracy
   - Package and test inference

2. **Generate datasets for expert-neo4j and expert-typescript**
   - Use similar approach to expert-sql (HuggingFace datasets)
   - Implement validation and preprocessing

3. **Create expert-python and expert-rust**
   - Define manifest and capabilities
   - Generate code-specific datasets
   - Train with DoRA r=12

4. **Runtime integration testing**
   - Test LoRA adapter loading in Rust
   - Verify paged KV-cache integration
   - Benchmark multi-expert performance

---

## Training Infrastructure Achievements

- ✅ Complete Python training pipeline (expert-cli train)
- ✅ Unsloth integration (2x faster, 70% less VRAM)
- ✅ Checkpointing system (save_strategy, eval_strategy)
- ✅ Dataset preprocessing with validation (sqlglot for SQL)
- ✅ Windows compatibility (torch.compile disabled, Unicode fixes)
- ✅ Training parameter optimization (LR 5e-5, dropout 0.1, warmup_ratio 0.1)
- ✅ Multiple adapter types (LoRA, DoRA, IA³)

