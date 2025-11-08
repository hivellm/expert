use crate::commands::spec::display_name;
use crate::error::Error;
use crate::registry::ExpertRegistry;
use semver::Version;
use std::cmp::Ordering;

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

fn list_experts(
    registry: &ExpertRegistry,
    verbose: bool,
    base_model_filter: Option<&str>,
) -> Result<(), Error> {
    let mut entries: Vec<(String, &crate::registry::ExpertVersionEntry)> = registry
        .iter_versions()
        .map(|(name, version)| (name.to_string(), version))
        .collect();

    if let Some(filter) = base_model_filter {
        entries.retain(|(_, version)| version.base_model == filter);
    }

    entries.sort_by(
        |(name_a, version_a), (name_b, version_b)| match name_a.cmp(name_b) {
            Ordering::Equal => compare_versions(&version_b.version, &version_a.version),
            other => other,
        },
    );

    if entries.is_empty() {
        println!("No experts installed");
        return Ok(());
    }

    println!("Installed Experts ({}):", entries.len());
    println!();

    for (name, expert) in entries {
        println!("  {}", display_name(&name, &expert.version));

        if verbose {
            println!("    Base Model: {}", expert.base_model);
            println!("    Path:       {}", expert.path.display());
            println!("    Source:     {}", expert.source);
            println!(
                "    Installed:  {}",
                expert.installed_at.format("%Y-%m-%d %H:%M:%S")
            );

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
            println!(
                "    Installed:    {}",
                model.installed_at.format("%Y-%m-%d %H:%M:%S")
            );
            println!();
        }
    }

    if !verbose {
        println!();
        println!("Use --verbose for more details");
    }

    Ok(())
}

fn compare_versions(a: &str, b: &str) -> Ordering {
    match (Version::parse(a), Version::parse(b)) {
        (Ok(va), Ok(vb)) => va.cmp(&vb),
        _ => a.cmp(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ExpertRegistry, ExpertVersionEntry};
    use chrono::Utc;
    use std::path::PathBuf;

    fn create_test_registry() -> ExpertRegistry {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        // Add expert-sql with multiple versions
        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.2.1".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/experts/expert-sql/0.2.1"),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: vec!["database:sql".to_string()],
                dependencies: Vec::new(),
            },
        );

        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.3.0".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/experts/expert-sql/0.3.0"),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: vec!["database:sql".to_string()],
                dependencies: Vec::new(),
            },
        );

        // Add expert-json
        registry.add_expert_version(
            "expert-json",
            ExpertVersionEntry {
                version: "0.0.2".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/experts/expert-json/0.0.2"),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: vec!["format:json".to_string()],
                dependencies: Vec::new(),
            },
        );

        registry
    }

    #[test]
    fn test_list_experts_shows_all_versions() {
        let registry = create_test_registry();
        let result = list_experts(&registry, false, None);
        assert!(result.is_ok());

        // Should list all versions
        let entries: Vec<_> = registry.iter_versions().collect();
        assert_eq!(entries.len(), 3); // 2 sql + 1 json
        
        // Verify versions are present
        assert!(registry.has_expert_version("expert-sql", "0.2.1"));
        assert!(registry.has_expert_version("expert-sql", "0.3.0"));
        assert!(registry.has_expert_version("expert-json", "0.0.2"));
    }

    #[test]
    fn test_list_experts_filter_by_base_model() {
        let registry = create_test_registry();
        let result = list_experts(&registry, false, Some("Qwen3-0.6B"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_experts_empty_registry() {
        let registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));
        let result = list_experts(&registry, false, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compare_versions() {
        assert_eq!(
            compare_versions("0.3.0", "0.2.1"),
            Ordering::Greater
        );
        assert_eq!(
            compare_versions("0.2.1", "0.3.0"),
            Ordering::Less
        );
        assert_eq!(
            compare_versions("0.3.0", "0.3.0"),
            Ordering::Equal
        );
    }
}
