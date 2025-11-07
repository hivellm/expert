#![allow(dead_code)]
#![allow(unused_imports)]

use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::PathBuf;

mod commands;
mod config;
mod error;
mod inference;
mod manifest;
mod python_bridge;
mod registry;
mod model_detection;
mod expert_router;
mod routing;

use config::AppConfig;
use error::Result;

#[derive(Parser)]
#[command(name = "expert-cli")]
#[command(about = "HiveLLM Expert System CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
    
    /// Project root directory (defaults to auto-detect)
    #[arg(long, global = true)]
    project_root: Option<PathBuf>,
    
    /// Models directory (defaults to <project_root>/models)
    #[arg(long, global = true)]
    models_dir: Option<PathBuf>,
    
    /// Experts directory (defaults to <project_root>/experts)
    #[arg(long, global = true)]
    global_experts_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate training dataset
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },
    
    /// Train expert from manifest
    Train {
        /// Path to manifest.json (defaults to ./manifest.json if not specified)
        #[arg(short, long)]
        manifest: Option<PathBuf>,
        
        /// Path to training dataset (JSONL). If not provided, uses dataset.path from manifest
        #[arg(short, long)]
        dataset: Option<PathBuf>,
        
        /// Output directory for weights
        #[arg(short, long, default_value = "weights")]
        output: PathBuf,
        
        /// Number of epochs (overrides manifest, supports fractional values like 2.5)
        #[arg(long)]
        epochs: Option<f32>,
        
        /// Device to use (cuda, cpu, auto)
        #[arg(long, default_value = "auto")]
        device: String,
        
        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    
    /// Validate trained expert
    Validate {
        /// Path to expert directory or .expert file
        #[arg(short, long)]
        expert: PathBuf,
        
        /// Custom test set (JSONL)
        #[arg(short, long)]
        test_set: Option<PathBuf>,
        
        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Package expert into .expert file
    Package {
        /// Path to manifest.json
        #[arg(short, long, default_value = "manifest.json")]
        manifest: PathBuf,
        
        /// Path to weights directory (defaults to ./weights)
        #[arg(short, long, default_value = "weights")]
        weights: PathBuf,
        
        /// Output .expert file (optional, auto-generated if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Model name (required for schema v2.0 multi-model experts)
        #[arg(long)]
        model: Option<String>,
        
        /// Include tests/ directory in package
        #[arg(long)]
        include_tests: bool,
        
        /// List package contents without creating archive
        #[arg(long)]
        list_contents: bool,
    },
    
    /// Sign expert package
    Sign {
        /// Path to .expert file
        #[arg(short, long)]
        expert: PathBuf,
        
        /// Path to private key
        #[arg(short, long)]
        key: PathBuf,
    },
    
    /// Generate signing key pair
    Keygen {
        /// Output path for private key
        #[arg(short, long, default_value = "publisher.pem")]
        output: PathBuf,
        
        /// Key owner name (optional)
        #[arg(short, long)]
        name: Option<String>,
    },
    
    /// Install expert from Git or local path
    Install {
        /// Source URL (git+https://..., file://..., or local path)
        source: String,
        
        /// Development mode (symlink instead of copy)
        #[arg(long)]
        dev: bool,
    },
    
    /// List installed experts
    List {
        /// Show verbose information
        #[arg(short, long)]
        verbose: bool,
        
        /// Filter by base model
        #[arg(long)]
        base_model: Option<String>,
        
        /// Show installed models instead of experts
        #[arg(long)]
        models: bool,
    },
    
    /// Uninstall expert
    Uninstall {
        /// Expert name
        name: String,
        
        /// Cleanup unused base models
        #[arg(long)]
        cleanup: bool,
    },
    
    /// Update expert from Git
    Update {
        /// Expert name
        name: String,
        
        /// Force reinstall if not from Git
        #[arg(long)]
        force: bool,
    },
    
    /// Interactive chat with multiple experts
    Chat {
        /// Comma-separated list of expert names (e.g., neo4j,json,sql)
        #[arg(long, value_delimiter = ',')]
        experts: Vec<String>,

        /// Path to base model (defaults to Qwen3-0.6B)
        #[arg(long)]
        base_model: Option<PathBuf>,

        /// Device to use (cuda, cpu, auto)
        #[arg(long, default_value = "auto")]
        device: String,

        /// Initial prompt to send (one-shot mode, exits after response)
        #[arg(long)]
        prompt: Option<String>,
        
        /// Enable debug mode (show all loading details)
        #[arg(long)]
        debug: bool,
        
        /// Sampling temperature (overrides expert manifest default)
        #[arg(long, help = "Temperature (0.0=greedy, 2.0=creative). Overrides manifest.")]
        temperature: Option<f64>,
        
        /// Top-p nucleus sampling (overrides expert manifest default)
        #[arg(long, help = "Top-p nucleus sampling (0.0-1.0). Overrides manifest.")]
        top_p: Option<f64>,
        
        /// Top-k sampling (overrides expert manifest default)
        #[arg(long, help = "Top-k sampling limit. Overrides manifest.")]
        top_k: Option<usize>,
        
        /// Maximum tokens to generate
        #[arg(long, help = "Maximum tokens to generate (default: 50)")]
        max_tokens: Option<usize>,
    },
    
    /// Route a query to the best matching expert(s)
    Route {
        /// Query to route
        query: String,
        
        /// Number of experts to return
        #[arg(long, short = 'k', default_value = "3")]
        top_k: usize,
        
        /// Show detailed routing information
        #[arg(long, short = 'v')]
        verbose: bool,
        
        /// Path to experts directory (defaults to auto-detect)
        #[arg(long)]
        experts_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum DatasetAction {
    /// Generate synthetic dataset
    Generate {
        /// Path to manifest.json
        #[arg(short, long, default_value = "manifest.json")]
        manifest: PathBuf,
        
        /// Output path
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Number of examples (overrides manifest)
        #[arg(short, long)]
        count: Option<usize>,
        
        /// LLM provider (overrides manifest)
        #[arg(short, long)]
        provider: Option<String>,
    },
    
    /// Validate dataset format
    Validate {
        /// Path to dataset file
        dataset: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize app configuration
    let app_config = AppConfig::new(
        cli.project_root.clone(),
        cli.models_dir.clone(),
        cli.global_experts_dir.clone(),
    );

    // Initialize logging
    let log_level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    // Print banner (skip for one-shot chat without debug)
    let skip_banner = match &cli.command {
        Commands::Chat { prompt, debug, .. } => prompt.is_some() && !debug,
        _ => false,
    };
    
    if !skip_banner {
        print_banner();
    }

    // Execute command
    match cli.command {
        Commands::Dataset { action } => match action {
            DatasetAction::Generate {
                manifest,
                output,
                count,
                provider,
            } => commands::dataset::generate(manifest, output, count, provider),
            DatasetAction::Validate { dataset } => commands::dataset::validate(dataset),
        },
        Commands::Train {
            manifest,
            dataset,
            output,
            epochs,
            device,
            resume,
        } => commands::train::train(manifest, dataset, output, epochs, device, resume),
        Commands::Validate {
            expert,
            test_set,
            verbose,
        } => commands::validate::validate(expert, test_set, verbose),
        Commands::Package {
            manifest,
            weights,
            output,
            model,
            include_tests,
            list_contents,
        } => commands::package::package(manifest, weights, output, model, include_tests, list_contents),
        Commands::Sign { expert, key } => commands::sign::sign(expert, key),
        Commands::Keygen { output, name } => commands::sign::keygen(output, name),
        Commands::Install { source, dev } => {
            commands::install::install(&source, dev)
        },
        Commands::List { verbose, base_model, models } => {
            commands::list::list(verbose, base_model, models)
        },
        Commands::Uninstall { name, cleanup } => {
            commands::uninstall::uninstall(&name, cleanup)
        },
        Commands::Update { name, force } => {
            commands::update::update(&name, force)
        },
        Commands::Chat {
            experts,
            base_model,
            device,
            prompt,
            debug,
            temperature,
            top_p,
            top_k,
            max_tokens,
        } => commands::chat::chat(experts, base_model, device, prompt, debug, temperature, top_p, top_k, max_tokens, &app_config),
        
        Commands::Route { query, top_k, verbose, experts_dir } => {
            let route_experts_dir = experts_dir.unwrap_or_else(|| app_config.experts_dir.clone());
            commands::route::run(&query, top_k, verbose, &route_experts_dir)
        }
    }
}

fn print_banner() {
    println!("\n{}", "╔═══════════════════════════════════════╗".bright_blue());
    println!("{}", "║     HiveLLM Expert System CLI        ║".bright_blue());
    println!("{}", "║     Train • Package • Deploy         ║".bright_blue());
    println!("{}", "╚═══════════════════════════════════════╝".bright_blue());
    println!();
}
