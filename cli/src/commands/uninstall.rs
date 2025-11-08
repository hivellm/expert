use crate::commands::spec::{display_name, parse_expert_spec};
use crate::error::Error;
use crate::registry::ExpertRegistry;
use std::path::Path;

/// Uninstall an expert
pub fn uninstall(name: &str, cleanup: bool) -> Result<(), Error> {
    let mut registry = ExpertRegistry::load()?;
    let (registry_name, version) = parse_expert_spec(name);

    match version {
        Some(ver) => {
            let entry = registry
                .get_expert_version(&registry_name, &ver)
                .ok_or_else(|| {
                    Error::Installation(format!(
                        "Expert '{}' v{} is not installed",
                        registry_name, ver
                    ))
                })?
                .clone();

            let label = display_name(&registry_name, &ver);
            println!("Uninstalling expert: {}", label);

            remove_expert_files(&entry.path, &registry.install_dir)?;

            registry.remove_expert_version(&registry_name, &ver);
            registry.save()?;

            println!("[OK] Expert '{}' uninstalled", label);
        }
        None => {
            let existing: Vec<_> = registry
                .list_expert_versions(&registry_name)
                .into_iter()
                .cloned()
                .collect();

            if existing.is_empty() {
                return Err(Error::Installation(format!(
                    "Expert '{}' is not installed",
                    registry_name
                )));
            }

            println!("Uninstalling expert: {} (all versions)", registry_name);

            for entry in &existing {
                remove_expert_files(&entry.path, &registry.install_dir)?;
            }

            registry.remove_all_versions(&registry_name);
            registry.save()?;

            println!(
                "[OK] Removed {} version(s) of '{}'",
                existing.len(),
                registry_name
            );
        }
    }

    // Cleanup unused base models if requested
    if cleanup {
        cleanup_unused_models(&mut registry)?;
    }

    Ok(())
}

/// Remove base models that are no longer used by any expert
fn cleanup_unused_models(registry: &mut ExpertRegistry) -> Result<(), Error> {
    let models = registry.list_base_models();

    // Find models used by experts
    let used_models: std::collections::HashSet<String> = registry
        .iter_versions()
        .map(|(_, version)| version.base_model.clone())
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
    std::io::stdout().flush().map_err(|e| Error::Io(e))?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| Error::Io(e))?;

    if input.trim().to_lowercase() != "y" {
        println!("Cleanup cancelled");
        return Ok(());
    }

    // Remove unused models
    for model_name in unused {
        if let Some(model) = registry.get_base_model(&model_name) {
            if model.path.exists() {
                std::fs::remove_dir_all(&model.path).map_err(|e| Error::Io(e))?;
                println!("[OK] Removed: {}", model_name);
            }
        }

        registry.remove_base_model(&model_name);
    }

    registry.save()?;
    println!("[OK] Cleanup complete");

    Ok(())
}

fn remove_expert_files(path: &Path, install_root: &Path) -> Result<(), Error> {
    if !path.exists() {
        return Ok(());
    }

    // Normalize paths - strip \\?\ prefix if present
    let path_str = path.to_string_lossy();
    let root_str = install_root.to_string_lossy();
    
    let normalized_path_str = path_str.strip_prefix(r"\\?\").unwrap_or(&path_str);
    let normalized_root_str = root_str.strip_prefix(r"\\?\").unwrap_or(&root_str);
    
    // Check if path is within install root using string comparison
    if normalized_path_str.starts_with(normalized_root_str) {
        std::fs::remove_dir_all(path).map_err(|e| Error::Io(e))?;
        println!("[OK] Removed expert files from: {}", path.display());

        if let Some(parent) = path.parent() {
            if parent != install_root && parent.exists() {
                let is_empty = std::fs::read_dir(parent)
                    .map_err(|e| Error::Io(e))?
                    .next()
                    .is_none();

                if is_empty {
                    std::fs::remove_dir_all(parent).map_err(|e| Error::Io(e))?;
                }
            }
        }
    } else {
        println!(
            "[INFO] Skipping file removal outside install root: {}",
            path.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ExpertRegistry, ExpertVersionEntry};
    use chrono::Utc;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_registry() -> (ExpertRegistry, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let experts_dir = temp_dir.path().join("experts");
        let models_dir = temp_dir.path().join("models");
        std::fs::create_dir_all(&experts_dir).unwrap();
        std::fs::create_dir_all(&models_dir).unwrap();

        let mut registry = ExpertRegistry::new(experts_dir.clone(), models_dir);

        // Add expert-sql with multiple versions
        let sql_v021_path = experts_dir.join("expert-sql").join("0.2.1");
        std::fs::create_dir_all(&sql_v021_path).unwrap();

        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.2.1".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: sql_v021_path.clone(),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: Vec::new(),
                dependencies: Vec::new(),
            },
        );

        let sql_v030_path = experts_dir.join("expert-sql").join("0.3.0");
        std::fs::create_dir_all(&sql_v030_path).unwrap();

        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.3.0".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: sql_v030_path.clone(),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: Vec::new(),
                dependencies: Vec::new(),
            },
        );

        (registry, temp_dir)
    }

    #[test]
    fn test_parse_expert_spec_in_uninstall() {
        let (name, version) = parse_expert_spec("sql@0.3.0");
        assert_eq!(name, "expert-sql");
        assert_eq!(version, Some("0.3.0".to_string()));

        let (name, version) = parse_expert_spec("sql");
        assert_eq!(name, "expert-sql");
        assert_eq!(version, None);
    }

    #[test]
    fn test_uninstall_specific_version() {
        let (mut registry, _temp_dir) = create_test_registry();

        assert!(registry.has_expert_version("expert-sql", "0.2.1"));
        assert!(registry.has_expert_version("expert-sql", "0.3.0"));

        // Remove specific version
        let removed = registry.remove_expert_version("expert-sql", "0.2.1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().version, "0.2.1");

        assert!(!registry.has_expert_version("expert-sql", "0.2.1"));
        assert!(registry.has_expert_version("expert-sql", "0.3.0"));
        assert!(registry.has_expert("expert-sql"));
    }

    #[test]
    fn test_uninstall_all_versions() {
        let (mut registry, _temp_dir) = create_test_registry();

        assert!(registry.has_expert("expert-sql"));

        let removed = registry.remove_all_versions("expert-sql");
        assert_eq!(removed.len(), 2);

        assert!(!registry.has_expert("expert-sql"));
        assert!(!registry.has_expert_version("expert-sql", "0.2.1"));
        assert!(!registry.has_expert_version("expert-sql", "0.3.0"));
    }

    #[test]
    fn test_remove_expert_files() {
        let temp_dir = TempDir::new().unwrap();
        let experts_dir = temp_dir.path().join("experts");
        std::fs::create_dir_all(&experts_dir).unwrap();

        let expert_path = experts_dir.join("expert-sql").join("0.3.0");
        std::fs::create_dir_all(&expert_path).unwrap();
        std::fs::write(expert_path.join("test.txt"), "test").unwrap();

        assert!(expert_path.exists());

        let result = remove_expert_files(&expert_path, &experts_dir);
        assert!(result.is_ok());
        assert!(!expert_path.exists());
        
        // Verify parent directory cleanup
        let parent = expert_path.parent().unwrap();
        if parent != experts_dir {
            // Parent should be removed if empty
            assert!(!parent.exists());
        }
    }
}
