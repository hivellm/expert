# Improve Expert Package Completeness

## 1. Add Essential Expert Files

- [x] 1.1 Add README.md from expert root
- [x] 1.2 Add grammar.gbnf if exists
- [x] 1.3 Add grammar files from manifest.training.decoding.grammar_file
- [x] 1.4 Log each file being added

## 2. Selective Adapter File Inclusion

- [x] 2.1 Replace append_dir_all with selective files
- [x] 2.2 Include essential files: adapter_model.safetensors, adapter_config.json, tokenizer files, training_args.bin, vocab.json
- [x] 2.3 Test package size reduction (documented in PACKAGING_GUIDE.md)
- [x] 2.4 Document expected adapter directory structure

## 3. Add Optional Test Directory

- [x] 3.1 Add --include-tests flag to PackageArgs
- [x] 3.2 Check if tests/ directory exists
- [x] 3.3 Add tests/ to archive if flag enabled
- [x] 3.4 Log tests being included

## 4. Document Package Contents

- [x] 4.1 Log package size after creation
- [x] 4.2 Add --list-contents flag
- [x] 4.3 Document recommended expert directory structure (PACKAGING_GUIDE.md)

## 5. Add Package Verification

- [x] 5.1 Add validate support for .expert files
- [x] 5.2 Extract .expert to temp directory (using tempfile crate)
- [x] 5.3 Validate manifest (integrated in validate command)
- [x] 5.4 Check required files exist (validates all 7 essential adapter files)
- [x] 5.5 Verify file integrity (SHA256 verification in validate_adapters)
