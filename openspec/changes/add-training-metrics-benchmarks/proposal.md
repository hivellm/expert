# Proposal: Training Metrics & Benchmark Workflow

**Change ID**: `add-training-metrics-benchmarks`  
**Status**: Draft  
**Created**: 2025-11-08  
**Author**: HiveLLM Team

## Overview

Establish an OpenSpec change that formalizes how we (a) collect training-time metrics, (b) execute benchmarks before packaging an `.expert`, and (c) compare consecutive versions. This ensures observable quality, prevents regressions, and provides trustworthy data for decision making.

## Motivation

### Current State (Problems)
- `expert-cli train` runs without consistent metric logging (loss, tool selection accuracy, argument validity, etc.).
- Packaging does not enforce any benchmark; a `.expert` can be shipped with performance regressions.
- There is no standardized workflow to compare versions (e.g., v0.1 vs v0.2) and validate gains or losses.

### Desired State (Solution)
- `expert-cli train` logs and exports standardized metrics (loss, tool_selection_accuracy, argument_validity, fallback_effectiveness, schema_compliance).
- A pre-packaging pipeline runs benchmarks (e.g., MCPToolBench, inference latency) and blocks publication when thresholds are violated.
- A comparison report between baseline and candidate versions is automatically generated (JSON + human-readable summary).

## Impact Analysis

### Benefits
- Continuous visibility into training and validation quality.
- Reduction of accidental regressions.
- Measurable history of expert evolution (quality and performance).
- Enables automated gates prior to shipping `.expert` packages.

### Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Increased pipeline duration | Run benchmarks in parallel and allow thresholds to be tuned per expert. |
| Excessive logging / storage usage | Retain only aggregated metrics and rotate historical files. |
| Local environment failures (Windows) | Provide a flag to skip heavy benchmarks where GPUs are unavailable. |

## Technical Approach

1. **Training Metrics Collection**  
   - Instrument `expert-cli train` to emit metrics every `eval_steps` (JSON/CSV + stdout).  
   - Export metrics to `training_metrics.jsonl` plus a summarized `training_report.md`.  
   - Align metric names with manifest configuration (`tool_selection_accuracy`, `argument_validity`, etc.).

2. **Pre-Packaging Benchmarks**  
   - Introduce `expert-cli bench` (or a packaging flag) that executes configurable suites (MCPToolBench, latency, memory).  
   - Read pass/fail thresholds from the manifest/CLI; fail the command when regressions exceed limits.  
  - Store results in `benchmarks/latest.json` including hardware/dataset metadata.

3. **Version Comparison Workflow**  
   - Implement `expert-cli compare --baseline <report>` to produce diffs across training/benchmark metrics.  
   - Persist results under `reports/comparisons/<timestamp>.json` plus a Markdown summary for PRs.  
   - Integrate with CI (GitHub Actions) to automatically attach comparisons to pull requests.

## Out of Scope
- Dataset or training refactors unrelated to metric capture.  
- External dashboards (Grafana/Superset); this change covers local/CI artifacts only.

