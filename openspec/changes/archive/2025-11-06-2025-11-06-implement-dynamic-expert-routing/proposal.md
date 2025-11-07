# Proposal: Dynamic Expert Routing with Domain Detection

**Type**: Feature  
**Phase**: P1 - Routing & Composition  
**Impact**: High  
**Status**: Implemented  
**Date**: 2025-11-06

## Problem

Current expert system applies all loaded experts indiscriminately:
- Generic queries ("What is the capital of France?") get forced through ChatML with database dialect
- Experts override base model's generalist capabilities
- No intelligent selection between base model and specialists
- ChatML artifacts (`<|end|>`, `<|endoftext|>`) appear in output

**Result**: System loses generalist capabilities when experts are loaded.

## Solution

Implement intelligent router that:
1. Detects domain from prompt keywords
2. Selects appropriate expert or uses base model
3. Applies ChatML only for specialized queries
4. Cleans output artifacts from generation

**Key Innovation**: Manifest-driven routing using `exclude_keywords` to preserve generality.

## Changes Required

### 1. Manifest Schema (`cli/src/manifest.rs`)
- Add `exclude_keywords` field to `Routing` struct
- Supports filtering generic queries from expert routing

### 2. Expert Router Module (`cli/src/expert_router.rs`)
- New module with `ExpertRouter` struct
- Keyword-based domain detection
- Generic query detection (what is, explain, etc.)
- Scoring algorithm: keywords (+1), exclude_keywords (-2), priority multiplier

### 3. Output Cleaning (`cli/src/inference/qwen.rs`)
- Implement `clean_chatml_output()` function
- Removes everything after `<|end|>` or `<|endoftext|>`
- Returns clean response without ChatML artifacts

### 4. Intelligent Chat (`cli/src/commands/chat.rs`)
- Use router to select expert vs base model
- One-shot mode: auto-routing based on keywords
- Interactive mode: auto-routing unless user selects explicit expert
- Debug mode shows routing decision

### 5. Manifest Updates
- **expert-neo4j**: Add `exclude_keywords` for generic questions
- **expert-sql**: Add `exclude_keywords` for explanatory queries

## Constraints Affected

- VRAM budget: No change (routing is CPU-side)
- Latency: +0.5ms (keyword matching on CPU)
- Expert limit: Still 10 (no change)

## Dependencies

- Requires: Manifest schema v2.0 with routing config
- Requires: Adapter merging implemented (completed)
- Blocks: None

## Testing Plan

Create `test-routing.ps1` with scenarios:
- Generic queries → base model (no ChatML)
- Specialized queries → correct expert (with ChatML)
- Output validation → no ChatML artifacts
- Multi-expert → intelligent selection

## Documentation Updates

- CHANGELOG.md: Document dynamic routing feature
- README.md: Update with routing behavior
- CLI README: Document router selection logic

## Risks

- Over-aggressive filtering: Some valid queries might skip expert
- Keyword conflicts: Multiple experts match same keyword
- Performance: Router adds latency (mitigated by simple keyword matching)

## Expected Results

| Query Type | Current Behavior | New Behavior |
|------------|------------------|--------------|
| "What is SQL?" | SQL expert → generates SQL | Base model → explains SQL |
| "Find users" | SQL expert → generates query | SQL expert → generates query |
| "Capital of France" | Neo4j expert → generates Cypher | Base model → answers "Paris" |
| "MATCH (n) RETURN n" | Neo4j expert → completes query | Neo4j expert → completes query |

## Acceptance Criteria

- [ ] Generic queries use base model (no expert)
- [ ] Specialized queries select correct expert
- [ ] Output contains no `<|end|>` or `<|endoftext|>` tokens
- [ ] Router decision logged in debug mode
- [ ] 6/6 test scenarios pass in test-routing.ps1

