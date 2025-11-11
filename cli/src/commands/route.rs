use crate::error::Result;
use crate::manifest::Manifest;
use crate::routing::KeywordRouter;
use colored::Colorize;
use std::path::PathBuf;

pub fn run(query: &str, top_k: usize, verbose: bool, experts_dir: &PathBuf) -> Result<()> {
    println!(
        "{}",
        "╔═══════════════════════════════════════╗".bright_cyan()
    );
    println!(
        "{}",
        "║   HiveLLM Expert Routing System     ║".bright_cyan()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════╝".bright_cyan()
    );
    println!();

    let mut router = KeywordRouter::new();
    let mut loaded_count = 0;

    if verbose {
        println!("📂 Scanning: {}", experts_dir.display());
    }

    // Scan for expert manifests
    if let Ok(entries) = std::fs::read_dir(experts_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let manifest_path = path.join("manifest.json");
                if manifest_path.exists() {
                    match Manifest::load(&manifest_path) {
                        Ok(manifest) => {
                            if verbose {
                                println!("  [OK] Loaded: {}", manifest.name);
                            }
                            router.add_expert(&manifest);
                            loaded_count += 1;
                        }
                        Err(e) => {
                            if verbose {
                                eprintln!(
                                    "  [!] Skipped {}: {}",
                                    path.file_name().unwrap().to_string_lossy(),
                                    e
                                );
                            }
                        }
                    }
                } else if verbose {
                    println!("  [!] No manifest: {}", path.display());
                }
            }
        }
    }

    println!(
        "📋 Loaded {} experts",
        loaded_count.to_string().bright_white()
    );
    println!();

    // Route query
    println!("[*] Query: {}", query.bright_white());
    println!();

    let results = router.route(&query, top_k);

    if results.is_empty() {
        println!("{}", "[X] No experts matched the query".bright_red());
        println!();
        println!("[*] Try:");
        println!("  - Using different keywords");
        println!("  - Being more specific");
        println!("  - Using --verbose to see expert keywords");
        return Ok(());
    }

    println!("[*] Top {} matches:", top_k.to_string().bright_white());
    println!();

    for (i, result) in results.iter().enumerate() {
        let rank = format!("{}.", i + 1);
        let score_pct = (result.score * 100.0) as u32;

        println!(
            "  {} {} ({}%)",
            rank.bright_blue(),
            result.expert_name.bright_white(),
            score_pct.to_string().bright_green()
        );

        if verbose && !result.matched_keywords.is_empty() {
            println!(
                "     Matched: {}",
                result.matched_keywords.join(", ").bright_yellow()
            );
        }
    }

    println!();
    println!("[*] Use this expert:");
    println!(
        "   {}",
        format!("expert-cli chat --experts {}", results[0].expert_name).bright_cyan()
    );

    Ok(())
}
