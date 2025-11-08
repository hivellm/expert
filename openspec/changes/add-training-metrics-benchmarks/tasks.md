# Implementation Tasks: Training Metrics & Benchmark Workflow

**Change ID**: `add-training-metrics-benchmarks`  
**Status**: Draft

---

## 1. Training Metrics Instrumentation

- [ ] Instrument `expert-cli train` to emit aggregated metrics (loss, tool_selection_accuracy, argument_validity, fallback_effectiveness, schema_compliance).
- [ ] Export per-step metrics (`training_metrics.jsonl`) and a final summary (`training_report.md`).
- [ ] Add CLI flags to control sampling frequency and output locations.
- [ ] Validate Windows/Linux compatibility (paths, encoding).

## 2. Benchmark Pré-Empacotamento

- [ ] Create command/flag `expert-cli bench` to run configured suites (e.g., MCPToolBench, latency, VRAM).
- [ ] Read thresholds from the manifest (`evaluation.metrics` / `failure_modes`).
- [ ] Persist results in `benchmarks/latest.json`; exit with non-zero status on regression.
- [ ] Provide optional flag (`--skip-bench`) for environments without GPUs.

## 3. Comparação entre Versões

- [ ] Implement `expert-cli compare --baseline <report>` to generate diffs between versions.
- [ ] Produce `reports/comparisons/<timestamp>.json` plus a human-readable Markdown summary.
- [ ] Integrate with CI (GitHub Actions) to attach comparisons to training/packaging PRs.
- [ ] Document the workflow in the project README (Release Checklist section).

---

## 4. Testes & Validação

- [ ] Add automated tests for metric parsing/serialization.
- [ ] Mock benchmarks in CI to validate thresholds and exit codes.
- [ ] Ensure `compare` fails gracefully when the baseline is missing or incompatible.

## 5. Documentação

- [ ] Update `expert-cli` docs with the new commands/flags and examples.
- [ ] Author a guide explaining `training_metrics.jsonl` and `benchmarks/latest.json`.
- [ ] Record the process in the handbook (expert release checklist). 

