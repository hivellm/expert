use crate::error::Error;
use crate::registry::ExpertRegistry;

/// Uninstall an expert
pub fn uninstall(name: &str, cleanup: bool) -> Result<(), Error> {
    let mut registry = ExpertRegistry::load()?;
    
    // Get expert info before removing
    let expert = registry.get_expert(name)
        .ok_or_else(|| Error::Installation(format!("Expert '{}' is not installed", name)))?
        .clone();
    
    println!("Uninstalling expert: {}", name);
    
    // Remove expert files
    if expert.path.exists() {
        std::fs::remove_dir_all(&expert.path)
            .map_err(|e| Error::Io(e))?;
        println!("[OK] Removed expert files from: {}", expert.path.display());
    }
    
    // Remove from registry
    registry.remove_expert(name);
    registry.save()?;
    
    println!("[OK] Expert '{}' uninstalled", name);
    
    // Cleanup unused base models if requested
    if cleanup {
        cleanup_unused_models(&mut registry)?;
    }
    
    Ok(())
}

/// Remove base models that are no longer used by any expert
fn cleanup_unused_models(registry: &mut ExpertRegistry) -> Result<(), Error> {
    let experts = registry.list_experts();
    let models = registry.list_base_models();
    
    // Find models used by experts
    let used_models: std::collections::HashSet<String> = experts
        .iter()
        .map(|e| e.base_model.clone())
        .collect();
    
    // Find unused models
    let unused: Vec<String> = models
        .iter()
        .filter(|m| !used_models.contains(&m.name))
        .map(|m| m.name.clone())
        .collect();
    
    if unused.is_empty() {
        println!("[INFO] No unused base models to cleanup");
        return Ok(());
    }
    
    println!("Found {} unused base model(s):", unused.len());
    for model_name in &unused {
        println!("  - {}", model_name);
    }
    
    // Ask for confirmation
    print!("Remove these models? [y/N]: ");
    use std::io::Write;
    std::io::stdout().flush()
        .map_err(|e| Error::Io(e))?;
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)
        .map_err(|e| Error::Io(e))?;
    
    if input.trim().to_lowercase() != "y" {
        println!("Cleanup cancelled");
        return Ok(());
    }
    
    // Remove unused models
    for model_name in unused {
        if let Some(model) = registry.get_base_model(&model_name) {
            if model.path.exists() {
                std::fs::remove_dir_all(&model.path)
                    .map_err(|e| Error::Io(e))?;
                println!("[OK] Removed: {}", model_name);
            }
        }
        
        registry.remove_base_model(&model_name);
    }
    
    registry.save()?;
    println!("[OK] Cleanup complete");
    
    Ok(())
}

