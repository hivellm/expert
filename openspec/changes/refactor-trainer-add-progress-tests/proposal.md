# Proposal: Refactor Trainer and Add Progress Testing

**Change ID**: `refactor-trainer-add-progress-tests`  
**Status**: In Progress  
**Created**: 2025-11-08  
**Author**: HiveLLM Team

## Overview

Refactor `expert_trainer.py` (1791 lines) into modular components by responsibility and add a progress testing system that executes tests at each saved checkpoint, generating JSON+Markdown reports and comparing with previous versions.

## Motivation

### Current State (Problems)
- `expert_trainer.py` is monolithic (1791 lines), making maintenance difficult
- No automated testing during training to track progress
- Difficult to compare performance between checkpoints
- No standardized reports for training progress
- Hard to identify regressions or improvements during training

### Desired State (Solution)
- Modular trainer code split by responsibility (config, model loading, dataset loading, etc.)
- Automated progress tests executed at each checkpoint save
- JSON and Markdown reports generated automatically
- Comparison with previous checkpoints and versions
- Easy identification of improvements and regressions

## Impact Analysis

### Benefits
- **Maintainability**: Smaller, focused modules are easier to understand and modify
- **Observability**: Track training progress with automated tests
- **Quality Assurance**: Catch regressions early during training
- **Documentation**: Automatic reports provide training history
- **Comparison**: Easy to see if new version improves over previous

### Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Breaking existing Rust integration | Maintain backward compatibility, test thoroughly |
| Tests slow down training | Make tests optional, run asynchronously if possible |
| Storage overhead from reports | Keep only recent reports, archive old ones |
| Test failures block training | Tests are informational, don't block training |

## Technical Approach

### 1. Refactoring Structure

Split `expert_trainer.py` into:
- `config.py`: TrainingConfig dataclass and loading
- `model_loader.py`: Model and tokenizer loading (Unsloth + standard)
- `dataset_loader.py`: Dataset loading and preprocessing
- `adapter_setup.py`: Adapter configuration (LoRA/DoRA/IA³/LoKr)
- `callbacks.py`: Training callbacks (memory cleanup, overfitting monitor, progress tests)
- `progress_testing.py`: Progress test execution and reporting (NEW)
- `trainer.py`: Main orchestration function

### 2. Progress Testing System

**ProgressTestCallback**: Executes tests when checkpoints are saved
- Loads test cases from `tests/test_cases.json` (if exists)
- Runs tests against checkpoint model
- Generates JSON and Markdown reports
- Compares with previous checkpoints

**Report Structure**:
```
weights/training_reports/
├── checkpoint-250/
│   ├── report.json
│   └── report.md
├── checkpoint-500/
│   ├── report.json
│   └── report.md
└── comparison.json
```

**Report Format**:
- JSON: Structured data for programmatic analysis
- Markdown: Human-readable reports with tables and summaries
- Comparison: Track improvements/regressions between checkpoints

### 3. Version Comparison

- Read current version from `manifest.json`
- Find previous checkpoints in same version
- Find previous versions (if available)
- Compare metrics: success rate, test improvements/regressions

## Implementation Plan

### Phase 0: Documentation
- Create openspec proposal and tasks

### Phase 1: Refactoring
- Create `cli/train/` directory structure
- Extract modules by responsibility
- Update `expert_trainer.py` to import from modules
- Verify Rust compatibility

### Phase 2: Progress Testing
- Implement `ProgressTestRunner`
- Implement `ProgressTestCallback`
- Integrate callback into trainer
- Implement report generation (JSON + Markdown)
- Implement comparison logic

### Phase 3: Integration
- Test with existing expert
- Validate report generation
- Validate comparison functionality
- Update documentation

## Success Criteria

- [ ] Trainer code split into logical modules (<400 lines each)
- [ ] Progress tests execute at each checkpoint save
- [ ] Reports generated in JSON and Markdown formats
- [ ] Comparison with previous checkpoints works
- [ ] Rust integration continues to work
- [ ] Tests are optional (don't fail if test_cases.json missing)

## Related Changes

- Builds on existing training infrastructure
- Complements `add-training-metrics-benchmarks` proposal
- Uses test structure from `experts/tests/template/`

