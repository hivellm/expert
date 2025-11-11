use colored::Colorize;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::commands::spec::{display_name, parse_expert_spec};
use crate::config::AppConfig;
use crate::error::Result;
use crate::expert_router::{ExpertRouter, LoadedExpert};
use crate::inference::QwenEngine;
use crate::inference::generation::GenerationConfig;
use crate::manifest::Manifest;

/// Format prompt according to expert's template
fn format_expert_prompt(prompt: &str, expert: &LoadedExpert) -> String {
    // Check if expert specifies a prompt template
    let template = if let Some(ref base_models) = expert.manifest.base_models {
        base_models
            .first()
            .and_then(|m| m.prompt_template.as_deref())
    } else {
        None
    };

    match template {
        Some("qwen3") => {
            let dialect = detect_dialect_from_capabilities(&expert.manifest.capabilities);
            
            // Generic prompt splitting: if prompt contains double newline, treat as context/query split
            // This works for any expert type (Cypher, SQL, JSON, etc.)
            let (system_msg, user_prompt) = if let Some(split_pos) = prompt.find("\n\n") {
                let context_part = prompt[..split_pos].trim();
                let query_part = prompt[split_pos + 2..].trim();
                
                // Build system message: add dialect if not already present and dialect is not "text"
                let system_msg = if context_part.starts_with("Dialect:") {
                    context_part.to_string()
                } else if dialect != "text" {
                    format!("Dialect: {}\n{}", dialect, context_part)
                } else if !context_part.is_empty() {
                    context_part.to_string()
                } else {
                    "You are a helpful AI assistant.".to_string()
                };
                
                (system_msg, query_part.to_string())
            } else {
                // No double newline: check if prompt starts with context-like patterns
                // If it starts with "Dialect:" or looks like context, put in system, otherwise user
                let system_msg = if prompt.starts_with("Dialect:") {
                    if dialect != "text" {
                        prompt.to_string()
                    } else {
                        format!("Dialect: {}", prompt)
                    }
                } else if dialect != "text" {
                    format!("Dialect: {}", dialect)
                } else {
                    "You are a helpful AI assistant.".to_string()
                };
                (system_msg, prompt.to_string())
            };

            format!(
                "<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
                system_msg = system_msg,
                user_prompt = user_prompt
            )
        }
        Some("chatml") | Some("qwen") => {
            // ChatML format for Qwen models
            // Detect dialect from capabilities
            let dialect = detect_dialect_from_capabilities(&expert.manifest.capabilities);

            // Build system message
            let system_msg = if dialect != "text" {
                format!("Dialect: {}", dialect)
            } else {
                "You are a helpful AI assistant.".to_string()
            };

            format!(
                "<|system|>\n{}\n<|end|>\n<|user|>\n{}\n<|end|>\n<|assistant|>\n",
                system_msg, prompt
            )
        }
        Some("llama") | Some("llama2") => {
            // Llama format
            format!("[INST] {} [/INST]", prompt)
        }
        Some("alpaca") => {
            // Alpaca format
            format!("### Instruction:\n{}\n\n### Response:\n", prompt)
        }
        _ => {
            // No special formatting
            prompt.to_string()
        }
    }
}

/// Detect output dialect from expert capabilities
fn detect_dialect_from_capabilities(capabilities: &[String]) -> &'static str {
    // Check query types
    if capabilities
        .iter()
        .any(|c| c.contains("cypher") || c.contains("neo4j"))
    {
        "cypher"
    } else if capabilities
        .iter()
        .any(|c| c.contains("sql") || c.contains("postgres") || c.contains("mysql"))
    {
        "sql"
    } else if capabilities.iter().any(|c| c.contains("json")) {
        "json"
    } else if capabilities.iter().any(|c| c.contains("python")) {
        "python"
    } else if capabilities.iter().any(|c| c.contains("rust")) {
        "rust"
    } else if capabilities
        .iter()
        .any(|c| c.contains("typescript") || c.contains("javascript"))
    {
        "typescript"
    } else {
        "text"
    }
}

/// Sanitize responses by removing duplicated code fences or concatenated payloads
fn sanitize_response(raw: &str) -> String {
    let trimmed = raw.trim();

    // Prefer explicit ```json fenced block
    if let Some(start) = trimmed.find("```json") {
        let after_fence = &trimmed[start + "```json".len()..];
        if let Some(end) = after_fence.find("```") {
            return after_fence[..end].trim().to_string();
        }
    }

    // Fallback: extract first balanced JSON object
    if let Some(start) = trimmed.find('{') {
        let mut depth = 0i32;
        for (idx, ch) in trimmed[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return trimmed[start..start + idx + 1].trim().to_string();
                    }
                }
                _ => {}
            }
        }
    }

    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Adapter, BaseModelV2, Manifest};
    use std::path::PathBuf;

    fn build_loaded_expert(prompt_template: &str, capabilities: Vec<&str>) -> LoadedExpert {
        let mut manifest = Manifest::default();
        manifest.capabilities = capabilities.into_iter().map(|c| c.to_string()).collect();
        manifest.base_models = Some(vec![BaseModelV2 {
            name: "base".to_string(),
            sha256: None,
            quantization: None,
            rope_scaling: None,
            prompt_template: Some(prompt_template.to_string()),
            adapters: vec![Adapter {
                adapter_type: "dora".to_string(),
                target_modules: vec![],
                r: None,
                alpha: None,
                scaling: None,
                dropout: None,
                use_dora: None,
                path: "adapter".to_string(),
                size_bytes: None,
                sha256: None,
            }],
        }]);

        LoadedExpert {
            name: "expert-test@0.1.0".to_string(),
            manifest,
            adapter_path: PathBuf::from("adapter"),
            manifest_path: PathBuf::from("manifest.json"),
        }
    }

    #[test]
    fn formats_qwen3_prompt_with_im_tokens() {
        let expert =
            build_loaded_expert("qwen3", vec!["query:cypher", "feature:pattern_matching"]);
        let formatted = format_expert_prompt("MATCH (n) RETURN n", &expert);
        let expected = "<|im_start|>system\nDialect: cypher<|im_end|>\n<|im_start|>user\nMATCH (n) RETURN n<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn formats_chatml_prompt_as_before() {
        let expert = build_loaded_expert("chatml", vec!["feature:json_output"]);
        let formatted = format_expert_prompt("Generate JSON", &expert);
        let expected = "<|system|>\nYou are a helpful AI assistant.\n<|end|>\n<|user|>\nGenerate JSON\n<|end|>\n<|assistant|>\n";
        assert_eq!(formatted, expected);
    }
}

pub fn chat(
    experts: Vec<String>,
    base_model: Option<PathBuf>,
    device: String,
    prompt: Option<String>,
    debug: bool,
    temperature_override: Option<f64>,
    top_p_override: Option<f64>,
    top_k_override: Option<usize>,
    max_tokens_override: Option<usize>,
    show_reasoning: bool,
    app_config: &AppConfig,
) -> Result<()> {
    // One-shot mode: if prompt is provided, skip banner unless debug is enabled
    let is_oneshot = prompt.is_some();

    if !is_oneshot || debug {
        println!("{}", "Expert Chat".bright_cyan().bold());
        println!(
            "{}",
            "═══════════════════════════════════════".bright_cyan()
        );
        println!();
    }

    // Print configuration (only in debug or interactive mode)
    if !is_oneshot || debug {
        if !experts.is_empty() {
            println!(
                "  {} Experts to load ({}):",
                "[>]".bright_blue(),
                experts.len()
            );
            for expert in &experts {
                println!("    - {}", expert.bright_white());
            }
        } else {
            println!("  {} Mode: Base model only", "[>]".bright_blue());
        }
        println!();
    }

    // Determine base model path
    let base_model_path = base_model.unwrap_or_else(|| app_config.default_base_model());

    if !is_oneshot || debug {
        println!(
            "  {} Base Model: {}",
            "[>]".bright_blue(),
            base_model_path.display().to_string().bright_white()
        );
        println!("  {} Device: {}", "[>]".bright_blue(), device.bright_white());
        println!();
    }

    // Find and load expert packages (if specified)
    let mut loaded_experts = Vec::new();

    if !experts.is_empty() {
        if !is_oneshot || debug {
            println!("  {} Locating expert packages...", "[*]".bright_blue());
        }

        // Try to load from registry first, fallback to local experts/ directory
        let registry = crate::registry::ExpertRegistry::load().ok();

        for expert_spec in &experts {
            let (registry_name, requested_version) = parse_expert_spec(expert_spec);
            let pretty_label = if let Some(version) = &requested_version {
                display_name(&registry_name, version)
            } else {
                registry_name.clone()
            };

            // Try to find expert in registry first
            let expert_dir = if let Some(ref reg) = registry {
                // Find expert entry in registry
                let entry_opt = if let Some(version) = &requested_version {
                    reg.get_expert_version(&registry_name, version)
                } else {
                    reg.get_expert(&registry_name)
                };

                if let Some(entry) = entry_opt {
                    entry.path.clone()
                } else {
                    // Not in registry, try local
                    if let Some(version) = &requested_version {
                        app_config.experts_dir.join(&registry_name).join(version)
                    } else {
                        app_config.experts_dir.join(&registry_name)
                    }
                }
            } else {
                // No registry, use local
                if let Some(version) = &requested_version {
                    app_config.experts_dir.join(&registry_name).join(version)
                } else {
                    app_config.experts_dir.join(&registry_name)
                }
            };

            if !expert_dir.exists() {
                if !is_oneshot || debug {
                    println!(
                        "    {} Expert directory not found: {} ({})",
                        "[!]".bright_yellow(),
                        expert_dir.display(),
                        pretty_label
                    );
                }
                continue;
            }

            // Load manifest
            let manifest_path = expert_dir.join("manifest.json");
            if !manifest_path.exists() {
                if !is_oneshot || debug {
                    println!(
                        "    {} No manifest.json found in {}",
                        "[!]".bright_yellow(),
                        expert_dir.display()
                    );
                }
                continue;
            }

            let manifest = Manifest::load(&manifest_path)?;

            // Find adapter path automatically - standard location is root of expert directory
            // The adapter_model.safetensors should be in the root of the expert directory
            let adapter_path = expert_dir.clone();

            // Verify adapter files exist in root - if not found, skip loading this expert
            // Standard location: adapter_model.safetensors must be in the root of expert directory
            if !adapter_path.join("adapter_model.safetensors").exists() {
                if !is_oneshot || debug {
                    println!(
                        "    {} Adapter not found in root: {}/adapter_model.safetensors",
                        "[X]".bright_red(),
                        adapter_path.display()
                    );
                    println!(
                        "       Expert will not be loaded. Adapter must be in root directory.",
                    );
                }
                continue;
            }

            if !is_oneshot || debug {
                let version_label = &manifest.version;
                println!(
                    "    {} {} v{}",
                    "[OK]".bright_green(),
                    manifest.name,
                    version_label
                );
            }

            loaded_experts.push(LoadedExpert {
                name: pretty_label.clone(),
                manifest,
                adapter_path,
                manifest_path: manifest_path.clone(),
            });
        }

        if loaded_experts.is_empty() {
            if !is_oneshot || debug {
                println!(
                    "    {} No expert packages found, using base model only",
                    "[!]".bright_yellow()
                );
                println!();
            }
        }
    }

    if !is_oneshot || debug {
        println!();
        println!("  {} Loading base model...", "[*]".bright_green());
    }

    // Determine if we should load with adapter
    let use_cuda = match device.to_lowercase().as_str() {
        "cuda" => true,
        "cpu" => false,
        "auto" => {
            // Auto-detect CUDA availability
            #[cfg(feature = "cuda")]
            {
                candle_core::utils::cuda_is_available()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        }
        _ => false,
    };

    // Try to load first expert with adapter
    let mut engine = if let Some(first_expert) = loaded_experts.first() {
        if !is_oneshot || debug {
            println!();
            println!(
                "  {} Attempting to load adapter from first expert: {}",
                "[*]".bright_blue(),
                first_expert.name
            );
        }

        // Get adapter type from manifest
        let adapter_type = if let Some(ref base_models) = first_expert.manifest.base_models {
            if let Some(first_model) = base_models.first() {
                if let Some(first_adapter) = first_model.adapters.first() {
                    first_adapter.adapter_type.as_str()
                } else {
                    "dora"
                }
            } else {
                "dora"
            }
        } else {
            "dora"
        };

        QwenEngine::from_local_with_adapter(
            &base_model_path,
            &first_expert.adapter_path,
            adapter_type,
            use_cuda,
            !is_oneshot || debug,
        )
        .map_err(|e| {
            crate::error::Error::Training(format!("Failed to load model with adapter: {}", e))
        })?
    } else {
        QwenEngine::from_local(&base_model_path, use_cuda)
            .map_err(|e| crate::error::Error::Training(format!("Failed to load model: {}", e)))?
    };

    // Load soft prompts from experts
    if !loaded_experts.is_empty() {
        if !is_oneshot || debug {
            println!();
            println!("  {} Loading soft prompts...", "[*]".bright_blue());
        }

        for expert in &loaded_experts {
            if let Err(e) =
                engine.load_soft_prompts_from_manifest(&expert.manifest, &expert.adapter_path)
            {
                if !is_oneshot || debug {
                    println!(
                        "    {} Failed to load soft prompts for {}: {}",
                        "[!]".bright_yellow(),
                        expert.name,
                        e
                    );
                }
            }
        }
    }

    // Skip "Chat Ready" message in one-shot mode without debug
    if !is_oneshot || debug {
        println!();
        println!(
            "{}",
            "═══════════════════════════════════════".bright_cyan()
        );
        println!(
            "{}",
            "Chat Ready! Type 'exit' or 'quit' to end".bright_green()
        );

        if !loaded_experts.is_empty() {
            println!("{}", "Commands:".bright_yellow());
            println!("  {} - Switch to expert", "/expert <name>".bright_white());
            println!(
                "  {} - Activate soft prompt",
                "/soft <name> or /soft none".bright_white()
            );
            println!("  {} - List soft prompts", "/soft list".bright_white());
            println!("  {} - List loaded experts", "/list".bright_white());
            println!("  {} - Exit chat", "/exit or /quit".bright_white());
        } else {
            println!("{}", "Commands:".bright_yellow());
            println!("  {} - Exit chat", "/exit or /quit".bright_white());
            println!();
            println!(
                "{}",
                "Tip: Load experts with --experts=neo4j,sql".bright_blue()
            );
        }

        println!(
            "{}",
            "═══════════════════════════════════════".bright_cyan()
        );
        println!();
    }

    // Chat loop
    let mut current_expert_idx: Option<usize> = None;

    // Load decoding config from expert manifest (if experts loaded)
    let decoding_defaults = if !loaded_experts.is_empty() {
        // Use first expert's decoding config as default
        loaded_experts[0].manifest.training.decoding.clone()
    } else {
        None
    };
    
    // Load grammar file from first expert if available (completely generic - works with any GBNF)
    let grammar_file_path = if !loaded_experts.is_empty() {
        let expert_dir = loaded_experts[0].manifest_path.parent().unwrap_or(&std::path::Path::new(""));
        
        // Priority 1: grammar_file from manifest decoding config
        if let Some(ref decoding) = decoding_defaults {
            if let Some(ref grammar_file) = decoding.grammar_file {
                let grammar_path = expert_dir.join(grammar_file);
                if grammar_path.exists() {
                    Some(grammar_path)
                } else {
                    // Fallback to grammar.gbnf
                    let fallback_path = expert_dir.join("grammar.gbnf");
                    if fallback_path.exists() {
                        Some(fallback_path)
                    } else {
                        None
                    }
                }
            } else {
                // Try grammar.gbnf in expert root
                let fallback_path = expert_dir.join("grammar.gbnf");
                if fallback_path.exists() {
                    Some(fallback_path)
                } else {
                    None
                }
            }
        } else {
            // Try grammar.gbnf in expert root
            let fallback_path = expert_dir.join("grammar.gbnf");
            if fallback_path.exists() {
                Some(fallback_path)
            } else {
                None
            }
        }
    } else {
        None
    };

    // Build generation config with 3-level priority:
    // 1. CLI overrides (highest priority)
    // 2. Expert manifest decoding config
    // 3. Hardcoded defaults (fallback)

    let manifest_temp = decoding_defaults.as_ref().and_then(|d| d.temperature);
    let manifest_top_p = decoding_defaults.as_ref().and_then(|d| d.top_p);
    let manifest_top_k = decoding_defaults.as_ref().and_then(|d| d.top_k);

    let final_temp = temperature_override.or(manifest_temp).unwrap_or(0.7);
    let final_top_p = top_p_override.or(manifest_top_p).or(Some(0.9));
    let final_top_k = top_k_override.or(manifest_top_k).or(Some(50));
    let final_max_tokens = max_tokens_override.unwrap_or(50);
    
    // Load grammar validator from expert if available (completely generic - works with any GBNF)
    if let Some(ref grammar_path) = grammar_file_path {
        if !is_oneshot || debug {
            println!();
            println!("  {} Loading grammar validator...", "📐".bright_blue());
        }
        
        match engine.load_grammar_validator(grammar_path) {
            Ok(_) => {
                if !is_oneshot || debug {
                    println!(
                        "    {} Grammar loaded from: {}",
                        "[OK]".bright_green(),
                        grammar_path.display()
                    );
                }
            }
            Err(e) => {
                if !is_oneshot || debug {
                    println!(
                        "    {} Failed to load grammar: {}",
                        "[!]".bright_yellow(),
                        e
                    );
                }
            }
        }
    }

    // Log parameter sources (only in debug or interactive mode)
    if (!is_oneshot || debug)
        && (decoding_defaults.is_some()
            || temperature_override.is_some()
            || top_p_override.is_some()
            || top_k_override.is_some())
    {
        println!("  {} Generation parameters:", "[>]".bright_blue());

        let temp_source = if temperature_override.is_some() {
            "CLI override"
        } else if manifest_temp.is_some() {
            "expert manifest"
        } else {
            "default"
        };
        println!(
            "    Temperature: {:.2} ({})",
            final_temp,
            temp_source.bright_white()
        );

        if let Some(tp) = final_top_p {
            let tp_source = if top_p_override.is_some() {
                "CLI override"
            } else if manifest_top_p.is_some() {
                "expert manifest"
            } else {
                "default"
            };
            println!("    Top-p: {:.2} ({})", tp, tp_source.bright_white());
        }

        if let Some(tk) = final_top_k {
            let tk_source = if top_k_override.is_some() {
                "CLI override"
            } else if manifest_top_k.is_some() {
                "expert manifest"
            } else {
                "default"
            };
            println!("    Top-k: {} ({})", tk, tk_source.bright_white());
        }

        println!("    Max tokens: {}", final_max_tokens);
        println!();
    }

    let gen_config = GenerationConfig {
        max_tokens: final_max_tokens,
        temperature: final_temp,
        top_p: final_top_p,
        top_k: final_top_k,
        repetition_penalty: Some(1.1),
    };

    // If prompt is provided, run in one-shot mode
    if let Some(ref prompt_text) = prompt {
        // Use router to select best expert for this prompt
        let (selected_expert, formatted_prompt) = if !loaded_experts.is_empty() {
            let router = ExpertRouter::new(loaded_experts.clone());

            if let Some(expert) = router.select_expert(prompt_text) {
                // Domain-specific query - use expert with ChatML
                if debug {
                    println!(
                        "  {} Using expert: {}",
                        "[>]".bright_blue(),
                        expert.name.bright_cyan()
                    );
                }
                (
                    Some(expert.name.clone()),
                    format_expert_prompt(prompt_text, expert),
                )
            } else {
                // Generic query - use base model without ChatML
                if debug {
                    println!("  {} Using base model (generic query)", "[>]".bright_blue());
                }
                (None, prompt_text.clone())
            }
        } else {
            (None, prompt_text.clone())
        };

        if debug {
            println!(
                "{}> {}",
                "base".bright_blue(),
                formatted_prompt.bright_white()
            );
            println!();
            print!("{}", "Thinking... ".bright_blue());
            io::stdout().flush().unwrap();
        }

        match engine.generate_verbose(
            &formatted_prompt,
            gen_config.max_tokens,
            gen_config.temperature,
            gen_config.top_p,
            debug,
            show_reasoning,
        ) {
            Ok(response) => {
                // Response is already cleaned by generate_verbose if grammar validator is available
                // Only apply sanitize_response if no grammar validator was used (fallback for JSON)
                let cleaned = if grammar_file_path.is_some() {
                    // Grammar validator already extracted/cleaned the query
                    response
                } else {
                    // No grammar validator, use sanitize_response for JSON extraction
                    sanitize_response(&response)
                };
                if debug {
                    print!("\r");
                    println!(
                        "{} {}",
                        "Assistant:".bright_green().bold(),
                        cleaned.as_str()
                    );
                } else {
                    // One-shot mode: just print the response
                    println!("{}", cleaned);
                }
            }
            Err(e) => {
                if debug {
                    print!("\r");
                    println!("{} Error: {}", "[X]".bright_red(), e);
                } else {
                    eprintln!("Error: {}", e);
                }
                return Err(crate::error::Error::Training(format!(
                    "Generation failed: {}",
                    e
                )));
            }
        }

        return Ok(());
    }

    // Interactive chat loop
    loop {
        // Show prompt
        let prompt_text = if let Some(idx) = current_expert_idx {
            format!("{}> ", loaded_experts[idx].name.bright_cyan())
        } else {
            format!("{}> ", "base".bright_blue())
        };

        print!("{}", prompt_text);
        io::stdout().flush().unwrap();

        // Read input
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input == "exit" || input == "quit" || input == "/exit" || input == "/quit" {
            println!("{}", "Goodbye!".bright_green());
            break;
        }

        if input == "/list" {
            println!();
            if loaded_experts.is_empty() {
                println!("No experts loaded. Using base model only.");
                println!("Load experts with: expert-cli chat --experts=neo4j,sql");
            } else {
                println!("Loaded experts:");
                for (idx, expert) in loaded_experts.iter().enumerate() {
                    let marker = if Some(idx) == current_expert_idx {
                        "[>]"
                    } else {
                        " "
                    };
                    println!("  {} {}: {}", marker, idx, expert.name.bright_cyan());
                }
            }
            println!();
            continue;
        }

        if input.starts_with("/expert ") {
            if loaded_experts.is_empty() {
                println!(
                    "{} No experts loaded. Using base model only.",
                    "[!]".bright_yellow()
                );
                println!();
                continue;
            }

            let expert_name = input.trim_start_matches("/expert ").trim();
            if let Some(idx) = loaded_experts.iter().position(|e| e.name == expert_name) {
                current_expert_idx = Some(idx);
                println!(
                    "{} Switched to expert: {}",
                    "[OK]".bright_green(),
                    expert_name.bright_cyan()
                );

                // TODO: Load adapter for this expert
                println!(
                    "{} Note: Adapter hot-swap not yet implemented",
                    "[!]".bright_yellow()
                );
            } else {
                println!("{} Expert not found: {}", "[X]".bright_red(), expert_name);
            }
            println!();
            continue;
        }

        if input.starts_with("/soft ") {
            let soft_cmd = input.trim_start_matches("/soft ").trim();

            if soft_cmd == "list" {
                println!();
                println!("Available soft prompts:");
                let soft_prompts = engine.get_soft_prompts();
                if soft_prompts.is_empty() {
                    println!("  None loaded");
                } else {
                    for name in soft_prompts {
                        let active = if engine
                            .get_active_soft_prompt_name()
                            .map(|active| active == name.as_str())
                            .unwrap_or(false)
                        {
                            "[>]"
                        } else {
                            " "
                        };
                        println!("  {} {}", active, name.bright_cyan());
                    }
                }
                println!();
                continue;
            } else if soft_cmd == "none" {
                engine.activate_soft_prompt(None);
                println!("{} Soft prompt deactivated", "[OK]".bright_green());
                println!();
                continue;
            } else {
                // Try to activate the specified soft prompt
                let soft_prompts = engine.get_soft_prompts();
                if soft_prompts.contains(&soft_cmd.to_string()) {
                    engine.activate_soft_prompt(Some(soft_cmd));
                    println!(
                        "{} Activated soft prompt: {}",
                        "[OK]".bright_green(),
                        soft_cmd.bright_cyan()
                    );
                } else {
                    println!("{} Soft prompt not found: {}", "[X]".bright_red(), soft_cmd);
                }
                println!();
                continue;
            }
        }

        // Use router to select best expert or use explicitly selected expert
        let formatted_input = if !loaded_experts.is_empty() {
            if let Some(idx) = current_expert_idx {
                // User explicitly selected an expert - always use it
                if let Some(expert) = loaded_experts.get(idx) {
                    format_expert_prompt(input.trim(), expert)
                } else {
                    input.trim().to_string()
                }
            } else {
                // No explicit selection - use router
                let router = ExpertRouter::new(loaded_experts.clone());

                if let Some(expert) = router.select_expert(input.trim()) {
                    format_expert_prompt(input.trim(), expert)
                } else {
                    // Generic query - no ChatML
                    input.trim().to_string()
                }
            }
        } else {
            input.trim().to_string()
        };

        // Generate response
        println!();
        print!("{}", "Thinking... ".bright_blue());
        io::stdout().flush().unwrap();

        match engine.generate(
            &formatted_input,
            gen_config.max_tokens,
            gen_config.temperature,
            gen_config.top_p,
            show_reasoning,
        ) {
            Ok(response) => {
                print!("\r");
                println!("{} {}", "Assistant:".bright_green().bold(), response.trim());
            }
            Err(e) => {
                print!("\r");
                println!("{} Error: {}", "[X]".bright_red(), e);
            }
        }

        println!();
    }

    Ok(())
}
