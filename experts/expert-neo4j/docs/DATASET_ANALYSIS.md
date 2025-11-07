# Neo4j Cypher Dataset Analysis

Analysis of datasets integrated into the Neo4j expert training data and current dataset status.

## Current Dataset Status

**Final Dataset:** Multi-source integrated dataset
- **Total Examples:** 11,069 (after rebalancing)
- **Language:** English
- **Schema:** Yes (100% coverage)
- **Format:** ChatML
- **Rebalancing:** Applied (target MATCH ratio: 90%)

### Dataset Sources

1. **neo4j/text2cypher-2025v1**
   - Original: 35,946 examples
   - Processed: 29,512 examples
   - Duplicates removed: 6,431
   - Invalid removed: 3
   - **Status:** ✅ Integrated

2. **megagonlabs/cypherbench**
   - Original: 8,534 examples
   - Processed: 8,529 examples
   - Duplicates removed: 5
   - Invalid removed: 0
   - **Status:** ✅ Integrated
   - **Integration Date:** 2025-01-XX
   - **Notes:** Schema extracted from Cypher queries

3. **Neo4j Official Documentation**
   - Collected: 324 examples
   - Processed: 315 examples
   - Duplicates removed: 6
   - Invalid removed: 0
   - **Status:** ✅ Integrated
   - **Coverage:** CREATE, MERGE, DELETE, SET, REMOVE, MATCH, WITH, CALL, etc.

4. **Synthetic Fixes**
   - Examples: 200
   - **Status:** ✅ Integrated

### Command Type Distribution (After Rebalancing)

| Command | Count | Percentage |
|---------|-------|------------|
| MATCH   | 10,000 | 90.34%     |
| CALL    | 948    | 8.56%      |
| RETURN  | 27     | 0.24%      |
| MERGE   | 22     | 0.20%      |
| WITH    | 20     | 0.18%      |
| CREATE  | 19     | 0.17%      |
| OTHER   | 15     | 0.14%      |
| UNWIND  | 13     | 0.12%      |
| DELETE  | 3      | 0.03%      |
| REMOVE  | 1      | 0.01%      |
| SET     | 1      | 0.01%      |

**Total Non-MATCH:** 1,069 (9.66%)

### Complexity Distribution

| Complexity | Count | Percentage |
|------------|-------|------------|
| Simple     | 4,633 | 41.86%     |
| Medium     | 5,156 | 46.58%     |
| Complex    | 1,280 | 11.56%     |

### Rebalancing Results

- **Before:** MATCH = 17,697 (94.3%)
- **After:** MATCH = 10,000 (90.34%)
- **Reduction:** 7,697 MATCH examples removed
- **Final Size:** 11,069 examples (from 18,766 processed)

## Analyzed Datasets (Not Integrated)

### jiuyuan/train_cypher

**Status:** Available (Not Integrated)

**Total Examples:** 153

**Language:** English

**Schema Support:** No

**Schema Coverage:** 0.0%

**Fields:**
- Question: `input_text`
- Cypher: `output_text`
- Schema: `None`

**Cypher Patterns (sample of 100):**
- MATCH: 100
- RETURN: 100
- COUNT: 60
- ORDER BY: 44
- WITH: 29
- LIMIT: 22
- WHERE: 20
- AVG: 6

**Reason for Not Integrating:**
- ⚠️ Very small: Only 153 examples (~1.4% of current dataset)
- ⚠️ No schema: Missing graph schema context (critical limitation)
- ⚠️ Limited impact: Too small to significantly improve model
- ⚠️ All MATCH queries: Would not help balance dataset

---

### Doraemon-AI/text-to-neo4j-cypher-chinese

**Status:** ERROR (Known Issue)

**Error:** Dataset generation error (schema casting issue)

**Partial Info:** Dataset exists but has schema casting problems preventing full load. The dataset contains ~1,834 examples but fails during HuggingFace dataset generation due to schema type mismatches in the answer_list field.

**Language:** Chinese (based on dataset description)

**Known Issue:** The dataset has a schema casting error where the answer_list field contains variable structures that cannot be cast to a fixed schema. This is a known issue reported on the HuggingFace dataset page.

**Reason for Not Integrating:**
- ❌ Dataset has known generation errors preventing full load
- ❌ Language mismatch: Chinese (current expert is English-only)
- ❌ Cannot be loaded without manual extraction

---

### johanhan/neo4j-cypher-fixed

**Status:** Available (Not Integrated)

**Total Examples:** 850

**Language:** Chinese

**Schema Support:** No

**Schema Coverage:** 0.0%

**Fields:**
- Question: `question`
- Cypher: `cypher`
- Schema: `None`

**Cypher Patterns (sample of 100):**
- MATCH: 89
- RETURN: 83
- COUNT: 26
- CREATE: 11

**Reason for Not Integrating:**
- ⚠️ Language mismatch: Chinese language (current expert is English-only)
- ⚠️ Small size: 850 examples (~7.7% of current dataset)
- ⚠️ No schema: Missing graph schema context
- ⚠️ Encoding issues: Contains non-ASCII characters causing processing issues
- ⚠️ Would require translation and schema extraction

---

## Dataset Quality Assessment

### Strengths

✅ **Multi-source diversity:** Examples from official Neo4j dataset, academic benchmark, and official documentation
✅ **Schema coverage:** 100% of examples include graph schema context
✅ **Rebalanced distribution:** MATCH reduced from 99.6% to 90.34%
✅ **Complexity range:** Good mix of Simple (41.86%), Medium (46.58%), and Complex (11.56%) queries
✅ **Real-world schemas:** Production schemas from Neo4j official dataset
✅ **Domain coverage:** Movies, social networks, streaming, finance, biology, sports, art (Wikidata)

### Limitations

⚠️ **Write operations underrepresented:**
- CREATE: 19 examples (0.17%)
- MERGE: 22 examples (0.20%)
- DELETE: 3 examples (0.03%)
- SET: 1 example (0.01%)
- REMOVE: 1 example (0.01%)
- **Total write operations:** 46 examples (0.42%)

⚠️ **MATCH still dominant:** 90.34% of dataset (though improved from 99.6%)

⚠️ **CALL-heavy:** 8.56% CALL examples (mostly from cypherbench) - good for subqueries but may not reflect typical usage

## Future Improvements

### High Priority

1. **Generate synthetic write operation examples**
   - CREATE operations (nodes, relationships)
   - MERGE operations with ON CREATE/ON MATCH
   - DELETE operations (nodes, relationships, DETACH DELETE)
   - SET operations (properties, labels)
   - REMOVE operations (properties, labels)
   - **Target:** Add 1,000-2,000 write operation examples

2. **Collect more documentation examples**
   - Focus on write operation tutorials
   - Transaction examples
   - Bulk operations (UNWIND + CREATE/MERGE)
   - **Target:** Add 200-300 more write examples

3. **The Stack dataset filtering**
   - Extract Cypher mutation examples from `bigcode/the-stack`
   - Filter for CREATE, MERGE, DELETE, SET, REMOVE patterns
   - Limit to 10,000 random samples (quality control)
   - **Target:** Add 500-1,000 write examples

### Medium Priority

4. **Expand CALL subquery examples**
   - More diverse subquery patterns
   - Nested CALL examples
   - **Target:** Maintain current CALL ratio (~8-10%)

5. **Add more complex patterns**
   - Path queries (shortestPath, allShortestPaths)
   - Pattern comprehensions
   - List comprehensions
   - **Target:** Increase Complex examples to 15-20%

### Low Priority

6. **Domain-specific datasets**
   - If available, integrate domain-specific Cypher examples
   - Ensure schema support and English language

## Distribution Charts

Visual distribution charts are available in `/docs`:
- `dataset_distribution.png` / `.pdf` - Command type distribution (bar and pie charts)
- `complexity_distribution.png` / `.pdf` - Complexity distribution (bar and pie charts)

## Preprocessing Statistics

- **Total examples processed:** 18,974
- **Successfully processed:** 18,766
- **Duplicates removed:** 205 (across all sources)
- **Invalid Cypher removed:** 3
- **Final dataset size:** 11,069 (after rebalancing)

## Integration History

- **2025-01-XX:** Integrated `megagonlabs/cypherbench` (+8,529 examples)
- **2025-01-XX:** Collected Neo4j official documentation (+324 examples)
- **2025-01-XX:** Applied rebalancing (MATCH: 99.6% → 90.34%)
- **2025-01-XX:** Generated distribution charts
