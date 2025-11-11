use colored::Colorize;
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use tar::Archive;

use crate::error::{Error, Result};
use crate::manifest::{Manifest, SchemaVersion};

pub fn validate(expert_path: PathBuf, test_set: Option<PathBuf>, verbose: bool) -> Result<()> {
    println!("{}", "Expert Validation".bright_cyan().bold());
    println!(
        "{}",
        "═══════════════════════════════════════".bright_cyan()
    );
    println!();

    // Determine if it's a directory or .expert file
    let (manifest_path, is_packaged, temp_dir) = if expert_path.is_dir() {
        (expert_path.join("manifest.json"), false, None)
    } else if expert_path.extension().and_then(|s| s.to_str()) == Some("expert") {
        println!("  {} Extracting packaged expert...", "[>]".bright_blue());
        let (extracted_dir, temp_dir_handle) = extract_expert_package(&expert_path)?;
        let manifest_path = extracted_dir.join("manifest.json");
        println!(
            "  {} Package extracted to temporary directory",
            "[OK]".bright_green()
        );
        println!();
        (manifest_path, true, Some((extracted_dir, temp_dir_handle)))
    } else {
        return Err(Error::Validation(format!(
            "Invalid expert path: {}. Must be a directory or .expert file.",
            expert_path.display()
        )));
    };

    // Check manifest exists
    if !manifest_path.exists() {
        return Err(Error::Validation(format!(
            "manifest.json not found at: {}",
            manifest_path.display()
        )));
    }

    println!("  {} Loading manifest...", "[>]".bright_blue());
    let manifest = Manifest::load(&manifest_path)?;
    let schema_version = manifest.get_schema_version();

    println!(
        "  {} Expert: {}",
        "[OK]".bright_green(),
        manifest.name.bright_white()
    );
    println!(
        "  {} Version: {}",
        "[OK]".bright_green(),
        manifest.version.bright_white()
    );
    println!(
        "  {} Schema: {}",
        "[OK]".bright_green(),
        schema_version.as_str().bright_white()
    );
    println!();

    // Validate manifest structure
    println!("  {} Validating manifest structure...", "[>]".bright_blue());
    validate_manifest(&manifest, schema_version)?;
    println!("  {} Manifest structure is valid", "[OK]".bright_green());
    println!();

    // Validate adapters
    println!("  {} Validating adapters...", "[>]".bright_blue());
    let expert_dir = if let Some(ref temp) = temp_dir {
        &temp.0
    } else {
        expert_path.parent().unwrap_or(&expert_path)
    };
    validate_adapters(&manifest, expert_dir, verbose)?;

    // Validate required files if packaged
    if is_packaged {
        validate_packaged_files(&manifest, expert_dir, verbose)?;
    }

    println!("  {} All adapters validated", "[OK]".bright_green());
    println!();

    // Validate base model (if schema v1.0)
    if schema_version == SchemaVersion::V1_0 {
        if let Some(ref base_model) = manifest.base_model {
            println!("  {} Validating base model reference...", "[>]".bright_blue());
            println!("    Base Model: {}", base_model.name);
            if let Some(ref quant) = base_model.quantization {
                println!("    Quantization: {}", quant);
            }
            println!("  {} Base model reference is valid", "[OK]".bright_green());
            println!();
        }
    } else {
        // Schema v2.0
        if let Some(ref base_models) = manifest.base_models {
            println!(
                "  {} Validating base models ({} models)...",
                "[>]".bright_blue(),
                base_models.len()
            );
            for model in base_models {
                println!("    - {}", model.name);
                if let Some(ref quant) = model.quantization {
                    println!("      Quantization: {}", quant);
                }
            }
            println!("  {} Base models validated", "[OK]".bright_green());
            println!();
        }
    }

    // Validate capabilities
    if !manifest.capabilities.is_empty() {
        println!(
            "  {} Capabilities ({}):",
            "[>]".bright_blue(),
            manifest.capabilities.len()
        );
        for cap in &manifest.capabilities {
            println!("    - {}", cap.bright_white());
        }
        println!();
    }

    // Run test set if provided
    if let Some(test_set_path) = test_set {
        println!("  {} Running test set validation...", "[>]".bright_blue());
        validate_test_set(&manifest, &test_set_path, verbose)?;
        println!("  {} Test set validation passed", "[OK]".bright_green());
        println!();
    }

    // Final summary
    println!(
        "{}",
        "═══════════════════════════════════════".bright_cyan()
    );
    println!(
        "  {} {}",
        "[OK]".bright_green().bold(),
        "Expert validation passed!".bright_green().bold()
    );
    println!(
        "{}",
        "═══════════════════════════════════════".bright_cyan()
    );

    // Cleanup temp directory if needed
    if let Some((_dir, temp_handle)) = temp_dir {
        drop(temp_handle); // This will automatically cleanup the temp directory
    }

    Ok(())
}

/// Extract a .expert package to a temporary directory
fn extract_expert_package(package_path: &Path) -> Result<(PathBuf, tempfile::TempDir)> {
    // Create temporary directory
    let temp_dir = tempfile::tempdir()
        .map_err(|e| Error::Validation(format!("Failed to create temp directory: {}", e)))?;

    // Open the .expert file (tar.gz)
    let file = File::open(package_path)
        .map_err(|e| Error::Validation(format!("Failed to open package: {}", e)))?;

    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    // Extract all files
    archive
        .unpack(temp_dir.path())
        .map_err(|e| Error::Validation(format!("Failed to extract package: {}", e)))?;

    Ok((temp_dir.path().to_path_buf(), temp_dir))
}

/// Validate required files in packaged expert
fn validate_packaged_files(manifest: &Manifest, expert_dir: &Path, verbose: bool) -> Result<()> {
    if verbose {
        println!("    Validating packaged files...");
    }

    // Required files
    let required_files = vec!["manifest.json"];

    for file in &required_files {
        let path = expert_dir.join(file);
        if !path.exists() {
            return Err(Error::Validation(format!(
                "Required file missing: {}",
                file
            )));
        }
        if verbose {
            println!("      {} {}", "[OK]".bright_green(), file);
        }
    }

    // Validate adapter files
    let adapters = if let Some(ref adapters) = manifest.adapters {
        adapters
    } else if let Some(ref models) = manifest.base_models {
        if let Some(first_model) = models.first() {
            &first_model.adapters
        } else {
            return Err(Error::Validation(
                "No base models defined in manifest".to_string(),
            ));
        }
    } else {
        return Err(Error::Validation(
            "No adapters found in manifest".to_string(),
        ));
    };

    for adapter in adapters {
        // Adapters are automatically discovered in expert root directory
        let adapter_path = &expert_dir;

        // Essential adapter files that must be present
        let essential_files = vec![
            "adapter_model.safetensors",
            "adapter_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "training_args.bin",
            "vocab.json",
        ];

        let mut found_count = 0;
        for file in &essential_files {
            let file_path = adapter_path.join(file);
            if file_path.exists() {
                found_count += 1;
                if verbose {
                    println!("      {} {}", "[OK]".bright_green(), file);
                }
            } else if file == &"adapter_model.safetensors" || file == &"adapter_config.json" {
                // Critical files
                return Err(Error::Validation(format!(
                    "Critical adapter file missing: {}/{}",
                    adapter_path.display(), file
                )));
            } else if verbose {
                println!(
                    "      {} {} (optional, not found)",
                    "[!] ".bright_yellow(),
                    file
                );
            }
        }

        if verbose {
            println!(
                "      Found {}/{} essential files",
                found_count,
                essential_files.len()
            );
        }
    }

    Ok(())
}

fn validate_manifest(manifest: &Manifest, schema_version: SchemaVersion) -> Result<()> {
    // Check required fields
    if manifest.name.is_empty() {
        return Err(Error::Validation("Expert name is empty".to_string()));
    }

    if manifest.version.is_empty() {
        return Err(Error::Validation("Expert version is empty".to_string()));
    }

    // Validate base model(s) based on schema
    match schema_version {
        SchemaVersion::V1_0 => {
            if manifest.base_model.is_none() {
                return Err(Error::Validation(
                    "Schema v1.0 requires 'base_model' field".to_string(),
                ));
            }
        }
        SchemaVersion::V2_0 => {
            if manifest.base_models.is_none() || manifest.base_models.as_ref().unwrap().is_empty() {
                return Err(Error::Validation(
                    "Schema v2.0 requires 'base_models' array with at least one model".to_string(),
                ));
            }
        }
    }

    // Get adapters based on schema version
    let adapters = if let Some(ref adapters) = manifest.adapters {
        // Schema v1.0
        adapters
    } else if let Some(ref models) = manifest.base_models {
        // Schema v2.0
        if let Some(first_model) = models.first() {
            &first_model.adapters
        } else {
            return Err(Error::Validation(
                "No base models defined in manifest".to_string(),
            ));
        }
    } else {
        return Err(Error::Validation(
            "No adapters defined in manifest".to_string(),
        ));
    };

    if adapters.is_empty() {
        return Err(Error::Validation("Adapters array is empty".to_string()));
    }

    // Validate training config
    if manifest.training.dataset.path.is_none()
        && manifest.training.dataset.generation.is_none()
        && manifest.training.dataset.tasks.is_none()
    {
        return Err(Error::Validation(
            "Training dataset must specify either 'path', 'generation', or 'tasks'".to_string(),
        ));
    }

    Ok(())
}

fn validate_adapters(manifest: &Manifest, expert_dir: &Path, verbose: bool) -> Result<()> {
    // Get adapters based on schema version
    let adapters = if let Some(ref adapters) = manifest.adapters {
        // Schema v1.0
        adapters
    } else if let Some(ref models) = manifest.base_models {
        // Schema v2.0
        if let Some(first_model) = models.first() {
            &first_model.adapters
        } else {
            return Err(Error::Validation(
                "No base models defined in manifest".to_string(),
            ));
        }
    } else {
        return Err(Error::Validation(
            "No adapters found in manifest".to_string(),
        ));
    };

    for (idx, adapter) in adapters.iter().enumerate() {
        if verbose {
            println!("    Adapter {} ({}):", idx + 1, adapter.adapter_type);
        }

        // Adapters are automatically discovered in expert root directory
        let adapter_path = &expert_dir;

        // For LoRA, check for adapter files
        if adapter.adapter_type == "lora" {
            // Check for adapter_model.safetensors or adapter_model.bin
            let safetensors_path = adapter_path.join("adapter_model.safetensors");
            let bin_path = adapter_path.join("adapter_model.bin");

            if !safetensors_path.exists() && !bin_path.exists() {
                return Err(Error::Validation(format!(
                    "Adapter files not found at: {}",
                    adapter_path.display()
                )));
            }

            let adapter_file = if safetensors_path.exists() {
                safetensors_path
            } else {
                bin_path
            };

            if verbose {
                println!("      Path: {}", adapter_file.display());
            }

            // Validate file size if specified
            if let Some(expected_size) = adapter.size_bytes {
                if expected_size > 0 {
                    let actual_size = fs::metadata(&adapter_file)?.len();
                    if actual_size != expected_size {
                        println!(
                            "      {} Size mismatch: expected {}, got {}",
                            "[WARN]".bright_yellow(),
                            expected_size,
                            actual_size
                        );
                    } else if verbose {
                        println!("      Size: {} bytes", actual_size);
                    }
                }
            }

            // Validate SHA256 if specified
            if let Some(ref expected_sha) = adapter.sha256 {
                if !expected_sha.is_empty() {
                    if verbose {
                        println!("      Verifying SHA256 checksum...");
                    }

                    let actual_hash = calculate_sha256(&adapter_file)?;
                    if &actual_hash != expected_sha {
                        return Err(Error::Validation(format!(
                            "SHA256 mismatch for adapter at {}. \
                                     Expected: {}, Got: {}",
                            adapter_file.display(),
                            expected_sha,
                            actual_hash
                        )));
                    } else if verbose {
                        println!("      {} SHA256 verified", "[OK]".bright_green());
                    }
                }
            }

            // Check for adapter_config.json
            let config_path = adapter_path.join("adapter_config.json");
            if !config_path.exists() {
                println!(
                    "      {} adapter_config.json not found (optional)",
                    "[WARN]".bright_yellow()
                );
            } else if verbose {
                println!("      {} adapter_config.json found", "[OK]".bright_green());
            }
        }

        if verbose {
            println!();
        }
    }

    Ok(())
}

fn validate_test_set(_manifest: &Manifest, test_set_path: &Path, verbose: bool) -> Result<()> {
    // Check test set exists
    if !test_set_path.exists() {
        return Err(Error::Validation(format!(
            "Test set not found: {}",
            test_set_path.display()
        )));
    }

    // Read and validate JSONL format
    let content = fs::read_to_string(test_set_path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

    if lines.is_empty() {
        return Err(Error::Validation("Test set is empty".to_string()));
    }

    println!("    Test examples: {}", lines.len());

    // Validate each line is valid JSON
    for (idx, line) in lines.iter().enumerate() {
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(json) => {
                if verbose && idx < 3 {
                    println!(
                        "    Example {}: {}",
                        idx + 1,
                        serde_json::to_string(&json).unwrap_or_default()
                    );
                }
            }
            Err(e) => {
                return Err(Error::Validation(format!(
                    "Invalid JSON on line {}: {}",
                    idx + 1,
                    e
                )));
            }
        }
    }

    // TODO: Run actual inference with the expert
    // This would require loading the model and running predictions
    println!(
        "    {} Inference testing not yet implemented",
        "[!]".bright_yellow()
    );

    Ok(())
}

fn calculate_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}
