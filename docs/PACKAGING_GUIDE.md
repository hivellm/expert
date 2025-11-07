# Expert Packaging Guide

## Recommended Directory Structure

```
expert-name/
├── manifest.json          # Expert configuration (required)
├── README.md              # Expert documentation (recommended)
├── LICENSE                # License file (recommended)
├── grammar.gbnf           # Grammar file for constrained decoding (optional)
├── weights/               # Trained weights directory
│   └── model-name/        # Model-specific weights
│       ├── adapter_model.safetensors  # Adapter weights (required)
│       ├── adapter_config.json        # PEFT config (required)
│       ├── special_tokens_map.json    # Tokenizer special tokens (required)
│       ├── tokenizer_config.json      # Tokenizer config (required)
│       ├── tokenizer.json             # Tokenizer vocabulary (required)
│       ├── training_args.bin          # Training arguments (required)
│       ├── vocab.json                 # Vocabulary file (required)
│       └── README.md                  # Adapter documentation (optional)
└── tests/                 # Test cases (optional, use --include-tests)
    ├── test_cases.json
    └── test_*.py
```

## Adapter Directory Structure

Essential files for adapters (all included in packages):
- `adapter_model.safetensors` - Adapter weights
- `adapter_config.json` - PEFT configuration
- `special_tokens_map.json` - Tokenizer special tokens
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Tokenizer vocabulary
- `training_args.bin` - Training arguments
- `vocab.json` - Vocabulary file
- `README.md` - Optional adapter documentation

Files excluded from packages (training artifacts):
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Scheduler state
- `trainer_state.json` - Trainer state
- `*.log` - Training logs
- `checkpoint-*/` - Intermediate checkpoints

## Package Contents

### Automatically Included
- `manifest.json` - Always included
- `adapter_model.safetensors` - Adapter weights
- `adapter_config.json` - PEFT config
- `special_tokens_map.json` - Tokenizer special tokens
- `tokenizer_config.json` - Tokenizer config
- `tokenizer.json` - Tokenizer vocabulary
- `training_args.bin` - Training arguments
- `vocab.json` - Vocabulary file
- `README.md` - If exists in expert root
- `grammar.gbnf` - If exists in expert root
- Custom grammar files from `manifest.training.decoding.grammar_file`
- `LICENSE` - If exists

### Optional (requires flags)
- `tests/` - Include with `--include-tests` flag

## Usage

### Basic Packaging

```bash
# Single model (schema v1.0)
expert-cli package --manifest manifest.json --weights weights

# Multi-model (schema v2.0)
expert-cli package --manifest manifest.json --weights weights --model Qwen3-0.6B
```

### Including Tests

```bash
expert-cli package --manifest manifest.json --weights weights --include-tests
```

### List Contents Without Creating Package

```bash
expert-cli package --manifest manifest.json --weights weights --list-contents
```

## Package Size Reduction

**Before** (including training artifacts):
- Typical expert: ~150-200 MB

**After** (selective inclusion):
- Optimized expert: ~50-80 MB
- Reduction: ~70% size reduction

## Validation

Validate an expert package:

```bash
expert-cli validate --expert expert-name.v1.0.0.expert
```

This will:
1. Extract package to temporary directory
2. Validate manifest.json schema
3. Check required files exist (adapter weights, config, tokenizer files)
4. Verify file integrity (SHA256 checksums)
5. Test manifest loading

## Best Practices

1. **Always include README.md** - Document expert capabilities and usage
2. **Include LICENSE** - Specify licensing terms
3. **Use grammar files** - For constrained decoding in production
4. **Test before packaging** - Use `expert-cli validate` on the directory
5. **Keep package size small** - Exclude unnecessary training artifacts
6. **Version properly** - Use semantic versioning in manifest

## Troubleshooting

### Package too large
- Ensure you're not including checkpoint directories
- Verify only essential adapter files are included
- Check if training logs are being included

### Missing files after extraction
- Check file paths in manifest.json
- Ensure paths are relative to expert root
- Verify weights directory structure matches manifest

### Grammar file not found
- Ensure grammar file path in manifest matches actual file
- Check file exists in expert root directory
- Verify file extension matches manifest declaration
