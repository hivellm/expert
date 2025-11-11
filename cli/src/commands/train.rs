use colored::Colorize;
use std::path::PathBuf;
use std::time::Instant;

use crate::error::Result;
use crate::manifest::Manifest;
use crate::python_bridge::PythonTrainer;

pub fn train(
    manifest_path: Option<PathBuf>,
    dataset_path_arg: Option<PathBuf>,
    output_dir: PathBuf,
    epochs_override: Option<f32>,
    device: String,
    resume_checkpoint: Option<PathBuf>,
) -> Result<()> {
    println!("{}", "📚 Training Expert".bright_cyan().bold());
    println!();

    // Default to ./manifest.json if not specified
    let manifest_path = manifest_path.unwrap_or_else(|| PathBuf::from("manifest.json"));

    // Load manifest
    println!("Loading manifest from: {}", manifest_path.display());
    let mut manifest = Manifest::load(&manifest_path)?;

    // Determine dataset path: CLI arg takes precedence, fallback to manifest
    let dataset_path = if let Some(path) = dataset_path_arg {
        path
    } else if let Some(ref dataset_type) = manifest.training.dataset.dataset_type {
        // Multi-task dataset - no single path needed
        if dataset_type == "multi_task" {
            // Use empty path - Python will load from multi_task config
            PathBuf::from("")
        } else if let Some(ref dataset_path) = manifest.training.dataset.path {
            resolve_dataset_path(dataset_path, &manifest_path)?
        } else {
            return Err(crate::error::Error::Training(
                "No dataset specified for non-multi-task training".to_string(),
            ));
        }
    } else if let Some(ref dataset_path) = manifest.training.dataset.path {
        resolve_dataset_path(dataset_path, &manifest_path)?
    } else {
        return Err(crate::error::Error::Training(
            "No dataset specified. Provide --dataset or define training.dataset.path in manifest.json".to_string()
        ));
    };

    fn resolve_dataset_path(dataset_path: &str, manifest_path: &PathBuf) -> Result<PathBuf> {
        // Check if it's a HuggingFace dataset (no local path needed)
        if dataset_path.contains('/')
            && !dataset_path.starts_with('.')
            && !dataset_path.starts_with('/')
        {
            // Likely a HuggingFace dataset ID (e.g., "neo4j/text2cypher-2025v1")
            Ok(PathBuf::from(dataset_path))
        } else {
            // Local path relative to manifest directory
            let manifest_dir = manifest_path.parent().ok_or_else(|| {
                crate::error::Error::Manifest("Invalid manifest path".to_string())
            })?;
            Ok(manifest_dir.join(dataset_path))
        }
    }

    // Override epochs if specified
    if let Some(epochs) = epochs_override {
        println!(
            "Overriding epochs: {} -> {}",
            manifest.training.config.epochs, epochs
        );
        manifest.training.config.epochs = epochs;
    }

    // Validate dataset exists (only for local paths, skip for multi-task)
    let dataset_str = dataset_path.to_string_lossy();
    let is_multi_task = dataset_str.is_empty();
    let is_huggingface = !is_multi_task
        && dataset_str.contains('/')
        && !dataset_str.starts_with('.')
        && !dataset_str.starts_with('/');

    if !is_multi_task && !is_huggingface && !dataset_path.exists() {
        return Err(crate::error::Error::Training(format!(
            "Dataset not found: {}",
            dataset_path.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Initialize Python trainer (subprocess-based, no PyO3 initialization)
    println!("\n{}", "Preparing training environment...".bright_yellow());
    let trainer = PythonTrainer::new()?;

    // For subprocess mode, skip CUDA check (Python script will handle detection)
    // check_cuda_available() uses PyO3 which causes DLL conflicts on Windows
    let final_device = if device == "auto" {
        // Let Python script detect CUDA (safer for Windows/PyTorch DLLs)
        "auto".to_string()
    } else {
        device
    };

    println!();
    println!(
        "{}",
        "═══════════════════════════════════════".bright_blue()
    );
    println!("{}", "Training Configuration".bright_blue().bold());
    println!(
        "{}",
        "═══════════════════════════════════════".bright_blue()
    );
    println!("  Expert: {}", manifest.name.bright_white());
    println!("  Version: {}", manifest.version.bright_white());
    println!("  Schema: {}", manifest.schema_version.bright_white());

    // Display base model info based on schema version
    let base_models = manifest.get_base_models();
    if base_models.is_empty() {
        println!("  Base Model: {}", "none".bright_white());
    } else if base_models.len() == 1 {
        let model_name = &base_models[0];
        println!("  Base Model: {}", model_name.bright_white());
        if let Some(ref base) = manifest.base_model {
            println!(
                "  Quantization: {}",
                base.quantization
                    .as_deref()
                    .unwrap_or("none")
                    .bright_white()
            );
        } else if let Some(ref models) = manifest.base_models {
            if let Some(model) = models.first() {
                println!(
                    "  Quantization: {}",
                    model
                        .quantization
                        .as_deref()
                        .unwrap_or("none")
                        .bright_white()
                );
            }
        }
    } else {
        println!("  Base Models: {}", base_models.join(", ").bright_white());
    }
    println!();
    println!(
        "  Adapter Type: {}",
        manifest.training.config.adapter_type.bright_white()
    );
    if let Some(rank) = manifest.training.config.rank {
        println!("  Rank: {}", rank.to_string().bright_white());
    }
    if let Some(alpha) = manifest.training.config.alpha {
        println!("  Alpha: {}", alpha.to_string().bright_white());
    }
    println!(
        "  Target Modules: {}",
        manifest
            .training
            .config
            .target_modules
            .join(", ")
            .bright_white()
    );
    println!();
    println!(
        "  Epochs: {}",
        manifest.training.config.epochs.to_string().bright_white()
    );
    println!(
        "  Learning Rate: {}",
        manifest
            .training
            .config
            .learning_rate
            .to_string()
            .bright_white()
    );
    println!(
        "  Batch Size: {}",
        manifest
            .training
            .config
            .batch_size
            .to_string()
            .bright_white()
    );
    println!(
        "  Grad Accumulation: {}",
        manifest
            .training
            .config
            .gradient_accumulation_steps
            .to_string()
            .bright_white()
    );
    println!();
    println!(
        "  Dataset: {}",
        dataset_path.display().to_string().bright_white()
    );
    println!(
        "  Output: {}",
        output_dir.display().to_string().bright_white()
    );
    println!("  Device: {}", final_device.bright_white());

    // Skip GPU info check (requires PyO3, causes DLL issues on Windows)
    println!(
        "  {}",
        "Note: CUDA detection handled by Python training script".bright_blue()
    );

    println!(
        "{}",
        "═══════════════════════════════════════".bright_blue()
    );
    println!();

    // Start training
    let start_time = Instant::now();

    println!("{}", "[*] Starting training...".bright_green().bold());
    println!();

    // Disable spinner - it causes conflicts with Python progress bars
    // let spinner = ProgressBar::new_spinner();
    // spinner.set_style(
    //     ProgressStyle::default_spinner()
    //         .template("{spinner:.green} {msg}")
    //         .unwrap(),
    // );
    // spinner.set_message("Training in progress...");
    // spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Call Python training via subprocess (avoids PyO3 conflicts)
    let result = trainer.train_via_subprocess(
        &manifest,
        &dataset_path,
        &output_dir,
        &final_device,
        resume_checkpoint.as_deref(),
    );

    // spinner.finish_and_clear();

    match result {
        Ok(_) => {
            let duration = start_time.elapsed();
            println!();
            println!(
                "{}",
                "[OK] Training completed successfully!".bright_green().bold()
            );
            println!();
            println!("  Duration: {:.2} hours", duration.as_secs_f32() / 3600.0);
            println!("  Weights saved to: {}", output_dir.display());
            println!();
            println!("{}", "Next steps:".bright_cyan());
            println!(
                "  1. Validate: expert-cli validate --expert {}",
                output_dir.display()
            );
            println!(
                "  2. Package:  expert-cli package --manifest {} --weights {}",
                manifest_path.display(),
                output_dir.display()
            );
            println!();
            Ok(())
        }
        Err(e) => {
            println!();
            println!("{}", "[X] Training failed!".bright_red().bold());
            println!("  Error: {}", e);
            println!();
            Err(e)
        }
    }
}
