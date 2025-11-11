use colored::Colorize;
use flate2::Compression;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use tar::Builder;

use crate::error::{Error, Result};
use crate::manifest::{Manifest, SchemaVersion};

pub fn package(
    manifest_path: PathBuf,
    weights_dir: PathBuf,
    output: Option<PathBuf>,
    model: Option<String>,
    include_tests: bool,
    list_contents: bool,
) -> Result<()> {
    println!("{}", "Packaging Expert".bright_cyan().bold());
    println!(
        "{}",
        "═══════════════════════════════════════".bright_cyan()
    );
    println!();

    // Load manifest
    println!(
        "  {} {}",
        "📄".bright_blue(),
        "Loading manifest...".bright_white()
    );
    let manifest = Manifest::load(&manifest_path)?;
    let schema_version = manifest.get_schema_version();

    println!(
        "  {} Expert: {}",
        "[>]".bright_blue(),
        manifest.name.bright_white()
    );
    println!(
        "  {} Version: {}",
        "[>]".bright_blue(),
        manifest.version.bright_white()
    );
    println!(
        "  {} Schema: {}",
        "[>]".bright_blue(),
        schema_version.as_str().bright_white()
    );
    println!();

    // Validate model parameter based on schema version
    match schema_version {
        SchemaVersion::V1_0 => {
            if model.is_some() {
                println!(
                    "{}",
                    "  [!] --model flag ignored for schema v1.0 (single model only)"
                        .bright_yellow()
                );
            }
            package_v1(
                &manifest,
                &weights_dir,
                output,
                include_tests,
                list_contents,
            )?;
        }
        SchemaVersion::V2_0 => {
            // Auto-detect model if only one is available
            let model_name = if let Some(model_arg) = model {
                model_arg
            } else {
                let available_models: Vec<String> = manifest
                    .base_models
                    .as_ref()
                    .map(|models| models.iter().map(|m| m.name.clone()).collect())
                    .unwrap_or_default();

                if available_models.len() == 1 {
                    let auto_model = available_models[0].clone();
                    println!(
                        "  {} Auto-detected model: {}",
                        "[>]".bright_blue(),
                        auto_model.bright_white()
                    );
                    auto_model
                } else if available_models.is_empty() {
                    return Err(Error::Packaging("No models found in manifest".to_string()));
                } else {
                    return Err(Error::Packaging(format!(
                        "Multiple models available. Specify --model flag:\n  {}",
                        available_models.join("\n  ")
                    )));
                }
            };
            package_v2(
                &manifest,
                &weights_dir,
                output,
                &model_name,
                include_tests,
                list_contents,
            )?;
        }
    }

    println!();
    println!("{}", "[OK] Packaging complete!".bright_green().bold());
    Ok(())
}

fn package_v1(
    manifest: &Manifest,
    weights_dir: &Path,
    output: Option<PathBuf>,
    include_tests: bool,
    _list_contents: bool,
) -> Result<()> {
    println!(
        "  {} {}",
        "[*]".bright_blue(),
        "Packaging schema v1.0 (single model)...".bright_white()
    );

    // Generate output filename if not provided - include model name
    let output_file = output.unwrap_or_else(|| {
        let model_suffix = if let Some(ref base_model) = manifest.base_model {
            // Extract model name: "Qwen3-0.6B" -> "qwen3-06b"
            let model_name = base_model
                .name
                .split('/')
                .last()
                .unwrap_or(&base_model.name)
                .to_lowercase()
                .replace(".", "")
                .replace("-", "");
            format!("-{}", model_name)
        } else {
            String::new()
        };

        PathBuf::from(format!(
            "{}{}.v{}.expert",
            manifest.name, model_suffix, manifest.version
        ))
    });

    println!(
        "  {} Output: {}",
        "[>]".bright_blue(),
        output_file.display().to_string().bright_white()
    );
    println!();

    println!(
        "  {} {}",
        "🔨".bright_blue(),
        "Creating package...".bright_white()
    );

    // Create tar.gz archive
    let tar_gz = File::create(&output_file)
        .map_err(|e| Error::Packaging(format!("Failed to create output file: {}", e)))?;

    let enc = GzEncoder::new(tar_gz, Compression::default());
    let mut tar = Builder::new(enc);

    // Serialize manifest to JSON
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    let manifest_bytes = manifest_json.as_bytes();

    let mut header = tar::Header::new_gnu();
    header.set_size(manifest_bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();

    tar.append_data(&mut header, "manifest.json", manifest_bytes)
        .map_err(|e| Error::Packaging(format!("Failed to add manifest to archive: {}", e)))?;

    println!("    {} manifest.json", "[OK]".bright_green());

    // Add adapter weights (automatically discovered in expert root)
    if let Some(ref adapters) = manifest.adapters {
        for adapter in adapters {
            // Adapters are in expert root directory (where manifest.json is)
            let expert_root = weights_dir.parent().unwrap_or(weights_dir);
            
            // Add essential adapter files (weights, config, tokenizer)
            let essential_files = vec![
                "adapter_model.safetensors", // Adapter weights
                "adapter_config.json",       // PEFT config
                "special_tokens_map.json",   // Tokenizer special tokens
                "tokenizer_config.json",     // Tokenizer config
                "tokenizer.json",            // Tokenizer vocabulary
                "training_args.bin",         // Training arguments
                "vocab.json",                // Vocabulary file
                "README.md",                 // Adapter docs (optional)
            ];

            let mut added_count = 0;
            for file_name in &essential_files {
                let file_path = expert_root.join(file_name);
                if file_path.exists() {
                    // Files go to root of package (same as expert root)
                    tar.append_path_with_name(&file_path, file_name)
                        .map_err(|e| {
                            Error::Packaging(format!("Failed to add {}: {}", file_name, e))
                        })?;
                    added_count += 1;
                }
            }

            if added_count > 0 {
                println!(
                    "    {} Adapter ({} files)",
                    "[OK]".bright_green(),
                    added_count
                );
            }
        }
    }

    // Add soft prompts if any
    for soft_prompt in &manifest.soft_prompts {
        let prompt_path = weights_dir
            .parent()
            .unwrap_or(weights_dir)
            .join(&soft_prompt.path);
        if prompt_path.exists() {
            tar.append_path_with_name(&prompt_path, &soft_prompt.path)
                .ok();
            println!(
                "    {} {}",
                    "[OK]".bright_green(),
                soft_prompt.path.bright_white()
            );
        } else {
            println!(
                "    {} {} (not found, skipping)",
                    "[!] ".bright_yellow(),
                soft_prompt.path.bright_white()
            );
        }
    }

    // Add expert documentation and resources
    let expert_root = weights_dir.parent().unwrap_or(weights_dir);

    // Add README.md
    let readme_path = expert_root.join("README.md");
    if readme_path.exists() {
        tar.append_path_with_name(&readme_path, "README.md").ok();
        println!("    {} README.md", "[OK]".bright_green());
    }

    // Add grammar.gbnf (common convention)
    let grammar_path = expert_root.join("grammar.gbnf");
    if grammar_path.exists() {
        tar.append_path_with_name(&grammar_path, "grammar.gbnf")
            .ok();
        println!("    {} grammar.gbnf", "[OK]".bright_green());
    }

    // Add grammar file from manifest (if specified)
    if let Some(ref training) = manifest.training.decoding {
        if let Some(ref grammar_file) = training.grammar_file {
            let grammar_custom_path = expert_root.join(grammar_file);
            if grammar_custom_path.exists() {
                tar.append_path_with_name(&grammar_custom_path, grammar_file)
                    .ok();
                println!(
                    "    {} {} (from manifest)",
                    "[OK]".bright_green(),
                    grammar_file
                );
            } else {
                println!(
                    "    {} {} (from manifest, not found)",
                    "[!] ".bright_yellow(),
                    grammar_file
                );
            }
        }
    }

    // Add tests/ directory if requested
    if include_tests {
        let tests_dir = expert_root.join("tests");
        if tests_dir.exists() && tests_dir.is_dir() {
            let mut test_count = 0;
            for entry in fs::read_dir(&tests_dir)
                .unwrap_or_else(|_| panic!("Failed to read tests directory"))
            {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() {
                        let archive_path =
                            format!("tests/{}", path.file_name().unwrap().to_string_lossy());
                        tar.append_path_with_name(&path, &archive_path).ok();
                        test_count += 1;
                    }
                }
            }
            if test_count > 0 {
                println!("    {} tests/ ({} files)", "[OK]".bright_green(), test_count);
            }
        } else {
            println!("    {} tests/ directory not found", "[!] ".bright_yellow());
        }
    }

    // Add LICENSE if exists
    let license_path = expert_root.join("LICENSE");
    if license_path.exists() {
        tar.append_path_with_name(&license_path, "LICENSE").ok();
        println!("    {} LICENSE", "[OK]".bright_green());
    }

    // Finalize archive
    tar.finish()
        .map_err(|e| Error::Packaging(format!("Failed to finalize archive: {}", e)))?;

    // Calculate package size
    let package_size = std::fs::metadata(&output_file)?.len();
    let size_mb = package_size as f64 / (1024.0 * 1024.0);

    println!();
    println!(
        "  {} Package created successfully!",
        "[OK]".bright_green().bold()
    );
    println!(
        "  {} File: {}",
        "[>]".bright_blue(),
        output_file.display().to_string().bright_white()
    );
    println!(
        "  {} Size: {:.2} MB",
        "[>]".bright_blue(),
        size_mb.to_string().bright_white()
    );

    // Calculate and display file size
    let file_size = fs::metadata(&output_file)?.len();
    let size_mb = file_size as f64 / 1_048_576.0;
    println!("  {} Size: {:.2} MB", "[>]".bright_blue(), size_mb);

    // Calculate SHA256 hash of the package
    let mut file = File::open(&output_file)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let hash = hasher.finalize();
    let hash_hex = format!("{:x}", hash);
    println!(
        "  {} SHA256: {}",
        "[>]".bright_blue(),
        hash_hex.bright_white()
    );

    // Write checksum file
    let checksum_file = output_file.with_extension("expert.sha256");
    let checksum_content = format!(
        "{}  {}\n",
        hash_hex,
        output_file.file_name().unwrap().to_string_lossy()
    );
    fs::write(&checksum_file, checksum_content)?;
    println!(
        "  {} Checksum: {}",
        "[>]".bright_blue(),
        checksum_file.display().to_string().bright_white()
    );

    Ok(())
}

fn package_v2(
    manifest: &Manifest,
    weights_dir: &Path,
    output: Option<PathBuf>,
    model_name: &str,
    include_tests: bool,
    _list_contents: bool,
) -> Result<()> {
    println!(
        "  {} {}",
        "[*]".bright_blue(),
        "Packaging schema v2.0 (multi-model)...".bright_white()
    );
    println!(
        "  {} Model: {}",
        "[>]".bright_blue(),
        model_name.bright_white()
    );

    // Check if a specific packaging checkpoint is specified
    let packaging_checkpoint = manifest.training.packaging_checkpoint.as_deref();
    if let Some(checkpoint) = packaging_checkpoint {
        println!(
            "  {} Packaging checkpoint: {}",
            "[>]".bright_blue(),
            checkpoint.bright_white()
        );
    }

    println!();

    // Find the requested model in base_models
    let base_models = manifest.base_models.as_ref().ok_or_else(|| {
        Error::Packaging("Schema v2.0 manifest missing base_models array".to_string())
    })?;

    let selected_model = base_models
        .iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| {
            let available: Vec<&str> = base_models.iter().map(|m| m.name.as_str()).collect();
            Error::Packaging(format!(
                "Model '{}' not found in manifest.\n  Available models: {}",
                model_name,
                available.join(", ")
            ))
        })?;

    println!("  {} Found model configuration", "[OK]".bright_green());
    println!(
        "    {} SHA256: {}",
        "[>]".bright_blue(),
        selected_model
            .sha256
            .as_deref()
            .unwrap_or("none")
            .bright_white()
    );
    println!(
        "    {} Quantization: {}",
        "[>]".bright_blue(),
        selected_model
            .quantization
            .as_deref()
            .unwrap_or("none")
            .bright_white()
    );
    println!(
        "    {} Adapters: {}",
        "[>]".bright_blue(),
        selected_model.adapters.len().to_string().bright_white()
    );

    // Show original adapter paths before potential replacement
    if packaging_checkpoint.is_some() {
        for (i, adapter) in selected_model.adapters.iter().enumerate() {
            println!(
                "    {} Adapter {}: {} (auto-discovered in root)",
                "[>]".bright_blue(),
                i + 1,
                adapter.adapter_type.bright_white()
            );
        }
    }

    println!();

    // Generate model-specific output filename
    let output_file = output.unwrap_or_else(|| {
        // Extract model base name from path (e.g., "F:/path/to/Qwen3-0.6B" -> "Qwen3-0.6B")
        let model_base_name = std::path::Path::new(model_name)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(model_name);

        // Normalize: "Qwen3-0.6B" -> "qwen3-0-6b" (lowercase, no dots)
        let normalized_model = model_base_name
            .to_lowercase()
            .replace('.', "-")
            .replace('_', "-");

        PathBuf::from(format!(
            "{}-{}.v{}.expert",
            manifest.name, normalized_model, manifest.version
        ))
    });

    // Create filtered manifest before listing contents
    // Replace adapter paths if packaging_checkpoint is specified
    let mut model_with_adjusted_paths = selected_model.clone();
    if let Some(checkpoint) = packaging_checkpoint {
        println!(
            "  {} Adjusting adapter paths to use {}",
            "[>]".bright_blue(),
            checkpoint.bright_white()
        );
        // Note: adapter.path field removed - adapters are automatically discovered in expert root
        // Checkpoint selection is now handled via checkpoint_step field in manifest
        // No path adjustment needed
    }

    println!(
        "  {} {}",
        "[*]".bright_blue(),
        "Package contents:".bright_white()
    );
    println!(
        "    {} Filtered manifest (model: {})",
        "[>]".bright_blue(),
        model_name.bright_white()
    );

    // List adapter weights (automatically discovered in expert root)
    for (i, adapter) in model_with_adjusted_paths.adapters.iter().enumerate() {
        // Adapters are in expert root directory
        let expert_root = weights_dir.parent().unwrap_or(weights_dir);
        let adapter_file = expert_root.join("adapter_model.safetensors");
        let exists = adapter_file.exists();
        let status = if exists {
                    "[OK]".bright_green()
        } else {
            "[X]".bright_red()
        };

        println!(
            "    {} Adapter {}: {} ({})",
            status,
            i + 1,
            adapter.adapter_type.bright_white(),
            if exists {
                "found in root".bright_green()
            } else {
                "missing".bright_red()
            }
        );

        if !exists {
            return Err(Error::Packaging(format!(
                "Adapter weight file not found: {}",
                adapter_file.display()
            )));
        }
    }

    // List shared resources
    if !manifest.soft_prompts.is_empty() {
        println!(
            "    {} {} soft prompt(s) (shared)",
            "[>]".bright_blue(),
            manifest.soft_prompts.len().to_string().bright_white()
        );
    }

    if manifest.license.is_some() {
        println!("    {} LICENSE", "[>]".bright_blue());
    }

    println!();
    println!(
        "  {} Output: {}",
        "[>]".bright_blue(),
        output_file.display().to_string().bright_white()
    );
    println!();

    // Create filtered manifest for this model (model_with_adjusted_paths already created above)
    let mut filtered_manifest = manifest.clone();
    filtered_manifest.base_models = Some(vec![model_with_adjusted_paths.clone()]);
    filtered_manifest.base_model = None; // Ensure v1.0 field is not present
    filtered_manifest.adapters = None; // Adapters are in BaseModelV2

    println!(
        "  {} {}",
        "🔨".bright_blue(),
        "Creating package...".bright_white()
    );

    // Create tar.gz archive
    let tar_gz = File::create(&output_file)
        .map_err(|e| Error::Packaging(format!("Failed to create output file: {}", e)))?;

    let enc = GzEncoder::new(tar_gz, Compression::default());
    let mut tar = Builder::new(enc);

    // Serialize filtered manifest to JSON
    let manifest_json = serde_json::to_string_pretty(&filtered_manifest)?;
    let manifest_bytes = manifest_json.as_bytes();

    let mut header = tar::Header::new_gnu();
    header.set_size(manifest_bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();

    tar.append_data(&mut header, "manifest.json", manifest_bytes)
        .map_err(|e| {
            Error::Packaging(format!("Failed to add filtered manifest to archive: {}", e))
        })?;

    println!(
        "    {} manifest.json (filtered for {})",
                    "[OK]".bright_green(),
        model_name.bright_white()
    );

    // Add model-specific adapter weights (automatically discovered in expert root)
    for adapter in &model_with_adjusted_paths.adapters {
        // Adapters are in expert root directory (where manifest.json is)
        let expert_root = weights_dir.parent().unwrap_or(weights_dir);
        
        // Add essential adapter files (weights, config, tokenizer) - DIRECTLY TO ROOT
        let essential_files = vec![
            "adapter_model.safetensors", // Adapter weights
            "adapter_config.json",       // PEFT config
            "special_tokens_map.json",   // Tokenizer special tokens
            "tokenizer_config.json",     // Tokenizer config
            "tokenizer.json",            // Tokenizer vocabulary
            "training_args.bin",         // Training arguments
            "vocab.json",                // Vocabulary file
            "README.md",                 // Adapter docs (optional)
        ];

        let mut added_count = 0;
        for file_name in &essential_files {
            let file_path = expert_root.join(file_name);
            if file_path.exists() {
                // Add directly to root of archive (no subdirectories)
                tar.append_path_with_name(&file_path, file_name)
                    .map_err(|e| {
                        Error::Packaging(format!("Failed to add {}: {}", file_name, e))
                    })?;
                added_count += 1;
            }
        }

        if added_count > 0 {
            println!(
                "    {} Adapter files ({} files added to root)",
                    "[OK]".bright_green(),
                added_count
            );
        } else {
            return Err(Error::Packaging(
                "Adapter files not found in expert root directory".to_string()
            ));
        }
    }

    // Add soft prompts (shared resources) - to root
    for soft_prompt in &manifest.soft_prompts {
        let prompt_path = weights_dir
            .parent()
            .unwrap_or(weights_dir)
            .join(&soft_prompt.path);
        if prompt_path.exists() {
            let file_name = std::path::Path::new(&soft_prompt.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("soft_prompt.pt");
            tar.append_path_with_name(&prompt_path, file_name).ok();
            println!(
                "    {} {} (added to root)",
                    "[OK]".bright_green(),
                file_name.bright_white()
            );
        }
    }

    // Add expert documentation and resources (shared)
    let expert_root = weights_dir.parent().unwrap_or(weights_dir);

    // README.md is already added from adapter's essential_files list above, so skip it here

    // Add grammar.gbnf (common convention)
    let grammar_path = expert_root.join("grammar.gbnf");
    if grammar_path.exists() {
        tar.append_path_with_name(&grammar_path, "grammar.gbnf")
            .ok();
        println!("    {} grammar.gbnf (shared)", "[OK]".bright_green());
    }

    // Add grammar file from manifest (if specified) - to root
    if let Some(ref training) = manifest.training.decoding {
        if let Some(ref grammar_file) = training.grammar_file {
            let grammar_custom_path = expert_root.join(grammar_file);
            if grammar_custom_path.exists() {
                let file_name = std::path::Path::new(grammar_file)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(grammar_file);
                tar.append_path_with_name(&grammar_custom_path, file_name)
                    .ok();
                println!(
                    "    {} {} (from manifest, added to root)",
                    "[OK]".bright_green(),
                    file_name
                );
            } else {
                println!(
                    "    {} {} (from manifest, not found)",
                    "[!] ".bright_yellow(),
                    grammar_file
                );
            }
        }
    }

    // Add tests/ directory if requested
    if include_tests {
        let tests_dir = expert_root.join("tests");
        if tests_dir.exists() && tests_dir.is_dir() {
            let mut test_count = 0;
            for entry in fs::read_dir(&tests_dir)
                .unwrap_or_else(|_| panic!("Failed to read tests directory"))
            {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() {
                        let archive_path =
                            format!("tests/{}", path.file_name().unwrap().to_string_lossy());
                        tar.append_path_with_name(&path, &archive_path).ok();
                        test_count += 1;
                    }
                }
            }
            if test_count > 0 {
                println!(
                    "    {} tests/ ({} files, shared)",
                    "[OK]".bright_green(),
                    test_count
                );
            }
        } else {
            println!("    {} tests/ directory not found", "[!] ".bright_yellow());
        }
    }

    // Add LICENSE if exists (shared resource)
    let license_path = expert_root.join("LICENSE");
    if license_path.exists() {
        tar.append_path_with_name(&license_path, "LICENSE").ok();
        println!("    {} LICENSE (shared)", "[OK]".bright_green());
    }

    // Finalize archive
    tar.finish()
        .map_err(|e| Error::Packaging(format!("Failed to finalize archive: {}", e)))?;

    println!();
    println!(
        "  {} Package created successfully!",
        "[OK]".bright_green().bold()
    );
    println!(
        "  {} File: {}",
        "[>]".bright_blue(),
        output_file.display().to_string().bright_white()
    );

    // Calculate and display file size
    let file_size = fs::metadata(&output_file)?.len();
    let size_mb = file_size as f64 / 1_048_576.0;
    println!("  {} Size: {:.2} MB", "[>]".bright_blue(), size_mb);

    // Calculate SHA256 hash of the package
    let mut file = File::open(&output_file)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let hash = hasher.finalize();
    let hash_hex = format!("{:x}", hash);
    println!(
        "  {} SHA256: {}",
        "[>]".bright_blue(),
        hash_hex.bright_white()
    );

    // Write checksum file
    let checksum_file = output_file.with_extension("expert.sha256");
    let checksum_content = format!(
        "{}  {}\n",
        hash_hex,
        output_file.file_name().unwrap().to_string_lossy()
    );
    fs::write(&checksum_file, checksum_content)?;
    println!(
        "  {} Checksum: {}",
        "[>]".bright_blue(),
        checksum_file.display().to_string().bright_white()
    );

    Ok(())
}
