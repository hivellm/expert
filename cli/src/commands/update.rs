use crate::error::Error;
use crate::registry::ExpertRegistry;
use std::process::Command;

/// Update expert from Git source
pub fn update(name: &str, force: bool) -> Result<(), Error> {
    println!("Updating expert: {}", name);
    
    let mut registry = ExpertRegistry::load()?;
    
    // Get expert info
    let expert = registry.get_expert(name)
        .ok_or_else(|| Error::Installation(format!("Expert '{}' is not installed", name)))?
        .clone();
    
    // Check if source is a Git URL
    if !expert.source.starts_with("git+") {
        if force {
            println!("[WARN] Expert was not installed from Git, reinstalling from source...");
            return crate::commands::install::install(&expert.source, false);
        } else {
            return Err(Error::Installation(
                "Expert was not installed from Git. Use --force to reinstall".to_string()
            ));
        }
    }
    
    // Parse Git URL
    let git_url = expert.source.strip_prefix("git+").unwrap();
    let (repo_url, ref_spec) = if let Some(pos) = git_url.find('#') {
        let (url, ref_part) = git_url.split_at(pos);
        (url, Some(ref_part.strip_prefix('#').unwrap()))
    } else {
        (git_url, None)
    };
    
    println!("  Repository: {}", repo_url);
    if let Some(ref_spec) = ref_spec {
        println!("  Ref: {}", ref_spec);
    }
    
    // Navigate to expert directory and git pull
    let expert_path = &expert.path;
    
    if !expert_path.join(".git").exists() {
        println!("[WARN] Expert directory is not a Git repository, reinstalling...");
        return crate::commands::install::install(&expert.source, false);
    }
    
    println!("  {} Pulling latest changes...", "[→]");
    
    let output = Command::new("git")
        .arg("-C")
        .arg(expert_path)
        .arg("pull")
        .arg("origin")
        .arg(ref_spec.unwrap_or("main"))
        .output()
        .map_err(|e| Error::Installation(format!("Failed to run git pull: {}", e)))?;
    
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Installation(format!("Git pull failed: {}", error)));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);
    
    // Update registry timestamp
    registry.remove_expert(name);
    let updated_expert = expert;
    registry.add_expert(updated_expert);
    registry.save()?;
    
    println!("[OK] Expert '{}' updated successfully!", name);
    
    Ok(())
}

