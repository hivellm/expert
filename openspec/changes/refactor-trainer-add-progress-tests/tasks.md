# Tasks: Refactor Trainer and Add Progress Testing

**Status**: In Progress  
**Last Updated**: 2025-11-08

## Phase 0: Documentation

- [x] Create openspec proposal
- [x] Create tasks.md

## Phase 1: Refactoring (Base Modules)

- [x] Create `cli/train/` directory
- [x] Extract `config.py` (TrainingConfig, load_training_config)
- [x] Extract `model_loader.py` (model loading functions)
- [x] Extract `dataset_loader.py` (dataset loading functions)
- [x] Extract `adapter_setup.py` (adapter setup)
- [x] Extract `callbacks.py` (existing callbacks)
- [x] Create `trainer.py` (main function)
- [x] Update `expert_trainer.py` to import from `train/`

## Phase 2: Testing System

- [x] Create `progress_testing.py` with ProgressTestRunner
- [x] Implement `ProgressTestCallback`
- [x] Integrate callback into trainer
- [x] Implement JSON report generation
- [x] Implement Markdown report generation
- [x] Implement comparison with previous versions

## Phase 3: Testing and Organization

- [x] Create comprehensive test suite for all modules
  - [x] `test_config.py` - Configuration loading tests
  - [x] `test_model_loader.py` - Model loading tests
  - [x] `test_dataset_loader.py` - Dataset loading tests
  - [x] `test_adapter_setup.py` - Adapter setup tests
  - [x] `test_callbacks.py` - Callback tests
  - [x] `test_progress_testing.py` - Progress testing system tests
  - [x] `test_trainer.py` - Trainer integration tests
- [x] Organize tests: separate Python tests from Rust tests
  - [x] Create `cli/tests_python/` directory
  - [x] Move all Python tests to `tests_python/`
  - [x] Update imports in test files
  - [x] Create README and run script for Python tests
  - [x] Create README for Rust tests

## Phase 4: Integration and Validation

- [ ] Test refactoring with existing expert
- [ ] Validate that reports are generated correctly
- [ ] Validate comparison with previous versions
- [ ] Update documentation
- [ ] Update `tasks.md` with final status

## Notes

- Maintain compatibility with existing Rust code
- Tests should be optional (skip if test_cases.json doesn't exist)
- Reports should be saved even if tests fail
- Comparison should handle absence of previous versions gracefully

