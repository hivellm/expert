# Improve Expert Package Completeness

**Status**: Proposal  
**Priority**: P1 (High - packages currently missing critical files)  
**Estimated Effort**: 2-3 hours  
**Date**: 2025-11-04

---

## Problem Statement

The `.expert` package currently includes:
- ✅ manifest.json
- ✅ Adapter weights (.safetensors)
- ✅ Soft prompts (.pt files)
- ✅ LICENSE (if exists)

**But is MISSING**:
- ❌ README.md (expert documentation)
- ❌ grammar.gbnf (GBNF grammar files for validation)
- ❌ adapter_config.json (PEFT config - may be included via append_dir_all)
- ❌ tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json)
- ❌ tests/ directory (test scripts and cases)

**Impact**:
- Users can't read expert docs without extracting full source
- Grammar validation files missing (decoding.grammar_file won't work)
- Testing infrastructure not portable
- Package not fully self-contained

---

## Current Packaging Behavior

### v1.0 Packaging (single model)
```rust
// Lines 121-180 in package.rs
tar.append_data(&mut header, "manifest.json", manifest_bytes)?;  // ✅
tar.append_dir_all(&adapter.path, &weight_path)?;                // ⚠️ May include extra files
tar.append_path_with_name(&prompt_path, &soft_prompt.path)?;     // ✅
tar.append_path_with_name(&license_path, "LICENSE")?;            // ✅
```

**Issue**: `append_dir_all` includes ALL files in adapter directory, which may include checkpoints, optimizer states, and other training artifacts we DON'T want.

### v2.0 Packaging (multi-model)
```rust
// Lines 340-395 in package.rs
// Same structure, filtered manifest
```

---

## Proposed Solution

### 1. Add Essential Expert Files

Include these at expert root level:
- `README.md` (expert documentation)
- `grammar.gbnf` or `grammars/` directory (if grammar_file specified in manifest)
- `tests/` directory (if exists - useful for validation)

### 2. Clean Adapter Directory Packaging

Instead of `append_dir_all` (includes everything), explicitly add:
- `adapter_model.safetensors` (weights)
- `adapter_config.json` (PEFT config)
- Skip: checkpoints, optimizer states, trainer state, etc.

### 3. Add Tokenizer Files (if customized)

If expert has custom tokenizer (rare), include:
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.json`, `merges.txt` (if present)

---

## Implementation Plan

### Phase 1: Add README and Grammar Files

**Files Modified**: `expert/cli/src/commands/package.rs`

```rust
// After LICENSE
let readme_path = weights_dir.parent().unwrap_or(weights_dir).join("README.md");
if readme_path.exists() {
    tar.append_path_with_name(&readme_path, "README.md")?;
    println!("    {} README.md", "✓".bright_green());
}

// Add grammar files if specified in manifest
if let Some(ref decoding) = manifest.training.decoding {
    if let Some(ref grammar_file) = decoding.grammar_file {
        let grammar_path = weights_dir.parent().unwrap_or(weights_dir).join(grammar_file);
        if grammar_path.exists() {
            tar.append_path_with_name(&grammar_path, grammar_file)?;
            println!("    {} {}", "✓".bright_green(), grammar_file);
        }
    }
}

// Check for root-level grammar.gbnf
let gbnf_path = weights_dir.parent().unwrap_or(weights_dir).join("grammar.gbnf");
if gbnf_path.exists() {
    tar.append_path_with_name(&gbnf_path, "grammar.gbnf")?;
    println!("    {} grammar.gbnf", "✓".bright_green());
}
```

### Phase 2: Selective Adapter File Inclusion

**Problem**: `append_dir_all` includes training artifacts (checkpoints, optimizer.pt, etc.)

**Solution**: Replace with selective file addition
```rust
// Instead of: tar.append_dir_all(&adapter.path, &weight_path)?;

// Required files only
let required_files = vec![
    "adapter_model.safetensors",  // Weights (required)
    "adapter_config.json",         // PEFT config (required)
];

for file_name in required_files {
    let file_path = weight_path.join(file_name);
    if file_path.exists() {
        let archive_path = format!("{}/{}", adapter.path, file_name);
        tar.append_path_with_name(&file_path, &archive_path)?;
    }
}
```

### Phase 3: Optional Test Directory

Add flag to include tests:
```rust
#[arg(long, help = "Include tests directory in package")]
include_tests: bool,

// In packaging code
if include_tests {
    let tests_dir = weights_dir.parent().unwrap_or(weights_dir).join("tests");
    if tests_dir.exists() {
        tar.append_dir_all("tests", &tests_dir)?;
        println!("    {} tests/ (directory)", "✓".bright_green());
    }
}
```

---

## Recommended Package Structure

```
expert-sql-0.0.1.expert (tar.gz)
├── manifest.json              ✅ Always included
├── README.md                  🆕 Expert documentation
├── LICENSE                    ✅ Already included
├── grammar.gbnf               🆕 GBNF grammar (if used)
├── grammars/                  🆕 Multiple grammars (future)
│   ├── json_strict.gbnf
│   └── sql_postgres.gbnf
├── weights/
│   └── qwen3-06b/
│       └── adapter/
│           ├── adapter_model.safetensors  ✅ Weights
│           └── adapter_config.json        🆕 PEFT config
├── soft_prompts/              ✅ Already included
│   ├── json_strict_32.pt
│   └── json_compact_64.pt
└── tests/                     🆕 Optional (--include-tests)
    ├── test_expert.py
    └── test_cases.json
```

---

## Benefits

1. **Self-Contained**: Package has everything needed for deployment
2. **Documentation**: README.md explains expert usage
3. **Grammar**: Validation files bundled with expert
4. **Cleaner**: No training artifacts (checkpoints, optimizer states)
5. **Testable**: Tests can be included for validation

---

## Backward Compatibility

- Existing packages still work (loader only requires manifest + adapters)
- New files are optional
- No breaking changes to format

---

## Testing

```bash
# Package expert
expert-cli package --manifest expert-sql/manifest.json --weights expert-sql/weights

# Extract and verify
tar -tzf expert-sql-0.0.1.expert
# Should see: manifest.json, README.md, grammar.gbnf, adapters, soft_prompts, LICENSE

# Validate package contents
expert-cli validate --expert expert-sql-0.0.1.expert
```

---

## Future: Package Manifest Schema

Add to manifest schema:
```json
"packaging": {
  "include": ["README.md", "grammar.gbnf", "tests/"],
  "exclude": ["checkpoints/", "*.tmp", "datasets/"],
  "require_license": true
}
```

