use colored::Colorize;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::manifest::Manifest;

pub fn generate(
    manifest_path: PathBuf,
    output: Option<PathBuf>,
    count: Option<usize>,
    provider: Option<String>,
) -> Result<()> {
    println!("{}", "📊 Dataset Generation".bright_cyan().bold());
    println!(
        "{}",
        "═══════════════════════════════════════".bright_blue()
    );
    println!();

    // Load manifest
    let manifest = Manifest::load(&manifest_path)?;
    println!(
        "  {} Loading manifest: {}",
        "✓".bright_green(),
        manifest.name
    );

    // Get dataset generation config
    let dataset_gen = manifest
        .training
        .dataset
        .generation
        .as_ref()
        .ok_or_else(|| Error::Config("No dataset.generation config in manifest".to_string()))?;

    println!("  {} Domain: {}", "→".bright_blue(), dataset_gen.domain);
    println!("  {} Task: {}", "→".bright_blue(), dataset_gen.task);

    let total_count = count.unwrap_or(dataset_gen.count);
    let llm_provider = provider.unwrap_or_else(|| dataset_gen.provider.clone());

    println!("  {} Provider: {}", "→".bright_blue(), llm_provider);
    println!("  {} Count: {}", "→".bright_blue(), total_count);
    println!();

    // Determine output path
    let output_path =
        output.unwrap_or_else(|| PathBuf::from(format!("dataset_{}.jsonl", manifest.name)));

    println!(
        "{}",
        "⚠️  Dataset generation requires LLM API integration".bright_yellow()
    );
    println!("   Please use Python scripts for now:");
    println!("   → cd experts/{}", manifest.name);
    println!("   → python generate_dataset.py");
    println!();
    println!("   Future: Will integrate with DeepSeek, Claude, and Cursor APIs");
    println!("   Output will be saved to: {}", output_path.display());

    Ok(())
}

pub fn validate(dataset: PathBuf) -> Result<()> {
    println!("{}", "✓ Dataset Validation".bright_cyan().bold());
    println!(
        "{}",
        "═══════════════════════════════════════".bright_blue()
    );
    println!();

    println!("  {} Validating: {}", "→".bright_blue(), dataset.display());

    // Check file exists
    if !dataset.exists() {
        return Err(Error::Validation(format!(
            "Dataset file not found: {}",
            dataset.display()
        )));
    }

    // Check extension
    let ext = dataset.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "jsonl" && ext != "json" {
        println!(
            "  {} Warning: Expected .jsonl or .json extension",
            "⚠".bright_yellow()
        );
    }

    // Read and validate each line
    let file = File::open(&dataset)?;
    let reader = BufReader::new(file);

    let mut line_count = 0;
    let mut valid_count = 0;
    let mut error_count = 0;
    let mut has_instruction = 0;
    let mut has_response = 0;

    println!("  {} Checking JSONL format...", "→".bright_blue());

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        line_count += 1;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as JSON
        match serde_json::from_str::<Value>(&line) {
            Ok(json) => {
                valid_count += 1;

                // Check for common fields
                if json.get("instruction").is_some() || json.get("question").is_some() {
                    has_instruction += 1;
                }
                if json.get("response").is_some()
                    || json.get("output").is_some()
                    || json.get("answer").is_some()
                {
                    has_response += 1;
                }
            }
            Err(e) => {
                error_count += 1;
                if error_count <= 5 {
                    println!(
                        "  {} Line {}: Invalid JSON - {}",
                        "✗".bright_red(),
                        idx + 1,
                        e
                    );
                }
            }
        }
    }

    println!();
    println!("{}", "Results:".bright_cyan().bold());
    println!("  Total lines: {}", line_count);
    println!("  Valid JSON: {} {}", valid_count, "✓".bright_green());

    if error_count > 0 {
        println!("  Invalid JSON: {} {}", error_count, "✗".bright_red());
    }

    println!();
    println!("{}", "Field Analysis:".bright_cyan().bold());
    println!("  Has instruction/question: {}", has_instruction);
    println!("  Has response/output/answer: {}", has_response);

    if has_instruction > 0 && has_response > 0 {
        println!();
        println!(
            "  {} Dataset appears valid for training",
            "✓".bright_green()
        );
    } else {
        println!();
        println!(
            "  {} Warning: Dataset may be missing required fields",
            "⚠".bright_yellow()
        );
        println!("    Expected: instruction/question + response/output/answer");
    }

    if error_count > 0 {
        return Err(Error::Validation(format!(
            "{} invalid JSON lines found",
            error_count
        )));
    }

    Ok(())
}
