use crate::error::Error;
use crate::registry::ExpertRegistry;

/// List installed experts
pub fn list(verbose: bool, base_model: Option<String>, show_models: bool) -> Result<(), Error> {
    let registry = ExpertRegistry::load()?;
    
    if show_models {
        list_models(&registry, verbose)?;
    } else {
        list_experts(&registry, verbose, base_model.as_deref())?;
    }
    
    Ok(())
}

fn list_experts(registry: &ExpertRegistry, verbose: bool, base_model_filter: Option<&str>) -> Result<(), Error> {
    let experts = registry.list_experts();
    
    let filtered: Vec<_> = if let Some(filter) = base_model_filter {
        experts.iter().filter(|e| e.base_model == filter).collect()
    } else {
        experts.iter().collect()
    };
    
    if filtered.is_empty() {
        println!("No experts installed");
        return Ok(());
    }
    
    println!("Installed Experts ({}):", filtered.len());
    println!();
    
    for expert in filtered {
        println!("  {} v{}", expert.name, expert.version);
        
        if verbose {
            println!("    Base Model: {}", expert.base_model);
            println!("    Path:       {}", expert.path.display());
            println!("    Source:     {}", expert.source);
            println!("    Installed:  {}", expert.installed_at.format("%Y-%m-%d %H:%M:%S"));
            
            if !expert.adapters.is_empty() {
                println!("    Adapters:   {} adapter(s)", expert.adapters.len());
                for adapter in &expert.adapters {
                    let size_mb = adapter.size_bytes as f64 / 1_048_576.0;
                    println!("      - {} ({:.2} MB)", adapter.adapter_type, size_mb);
                }
            }
            
            if !expert.capabilities.is_empty() {
                println!("    Capabilities:");
                for cap in &expert.capabilities {
                    println!("      - {}", cap);
                }
            }
            
            println!();
        }
    }
    
    if !verbose {
        println!();
        println!("Use --verbose for more details");
    }
    
    Ok(())
}

fn list_models(registry: &ExpertRegistry, verbose: bool) -> Result<(), Error> {
    let models = registry.list_base_models();
    
    if models.is_empty() {
        println!("No base models installed");
        return Ok(());
    }
    
    println!("Installed Base Models ({}):", models.len());
    println!();
    
    for model in models {
        println!("  {}", model.name);
        
        if verbose {
            let size_gb = model.size_bytes as f64 / 1_073_741_824.0;
            println!("    Path:         {}", model.path.display());
            println!("    Size:         {:.2} GB", size_gb);
            
            if let Some(ref quant) = model.quantization {
                println!("    Quantization: {}", quant);
            }
            
            if let Some(ref sha) = model.sha256 {
                println!("    SHA256:       {}...", &sha[..16]);
            }
            
            println!("    Source:       {}", model.source);
            println!("    Installed:    {}", model.installed_at.format("%Y-%m-%d %H:%M:%S"));
            println!();
        }
    }
    
    if !verbose {
        println!();
        println!("Use --verbose for more details");
    }
    
    Ok(())
}

