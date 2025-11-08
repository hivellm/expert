use chrono::{DateTime, Utc};
use semver::Version;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(not(test))]
use crate::error::Error;

#[cfg(test)]
pub use crate::error::Error;

/// Expert Registry - tracks installed experts and base models
#[derive(Debug, Clone, Serialize)]
pub struct ExpertRegistry {
    pub version: String,
    pub last_updated: DateTime<Utc>,
    pub install_dir: PathBuf,
    pub models_dir: PathBuf,
    pub base_models: Vec<BaseModelEntry>,
    pub experts: Vec<ExpertRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelEntry {
    pub name: String,
    pub path: PathBuf,
    pub sha256: Option<String>,
    pub quantization: Option<String>,
    pub size_bytes: u64,
    pub installed_at: DateTime<Utc>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRecord {
    pub name: String,
    pub versions: Vec<ExpertVersionEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertVersionEntry {
    pub version: String,
    pub base_model: String,
    pub path: PathBuf,
    pub source: String,
    pub installed_at: DateTime<Utc>,
    pub adapters: Vec<AdapterEntry>,
    pub capabilities: Vec<String>,
    pub dependencies: Vec<DependencyEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterEntry {
    #[serde(rename = "type")]
    pub adapter_type: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEntry {
    pub name: String,
    pub version: String,
    pub optional: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct LegacyExpertEntry {
    pub name: String,
    pub version: String,
    pub base_model: String,
    pub path: PathBuf,
    pub source: String,
    pub installed_at: DateTime<Utc>,
    pub adapters: Vec<AdapterEntry>,
    pub capabilities: Vec<String>,
    pub dependencies: Vec<DependencyEntry>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ExpertRegistryFormat {
    New {
        version: String,
        last_updated: DateTime<Utc>,
        install_dir: PathBuf,
        models_dir: PathBuf,
        base_models: Vec<BaseModelEntry>,
        experts: Vec<ExpertRecord>,
    },
    Legacy {
        version: String,
        last_updated: DateTime<Utc>,
        install_dir: PathBuf,
        models_dir: PathBuf,
        base_models: Vec<BaseModelEntry>,
        experts: Vec<LegacyExpertEntry>,
    },
}

impl<'de> Deserialize<'de> for ExpertRegistry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match ExpertRegistryFormat::deserialize(deserializer)? {
            ExpertRegistryFormat::New {
                version,
                last_updated,
                install_dir,
                models_dir,
                base_models,
                experts,
            } => Ok(ExpertRegistry {
                version,
                last_updated,
                install_dir,
                models_dir,
                base_models,
                experts,
            }),
            ExpertRegistryFormat::Legacy {
                version,
                last_updated,
                install_dir,
                models_dir,
                base_models,
                experts,
            } => {
                let mut registry = ExpertRegistry {
                    version,
                    last_updated,
                    install_dir,
                    models_dir,
                    base_models,
                    experts: Vec::new(),
                };

                for legacy in experts {
                    registry.add_expert_version(
                        &legacy.name,
                        ExpertVersionEntry {
                            version: legacy.version,
                            base_model: legacy.base_model,
                            path: legacy.path,
                            source: legacy.source,
                            installed_at: legacy.installed_at,
                            adapters: legacy.adapters,
                            capabilities: legacy.capabilities,
                            dependencies: legacy.dependencies,
                        },
                    );
                }

                Ok(registry)
            }
        }
    }
}

impl ExpertRegistry {
    /// Create a new empty registry
    pub fn new(install_dir: PathBuf, models_dir: PathBuf) -> Self {
        Self {
            version: "1.0".to_string(),
            last_updated: Utc::now(),
            install_dir,
            models_dir,
            base_models: Vec::new(),
            experts: Vec::new(),
        }
    }

    /// Load registry from default location
    pub fn load() -> Result<Self, Error> {
        let registry_path = Self::default_path()?;

        if !registry_path.exists() {
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(&registry_path).map_err(|e| Error::Io(e))?;

        let registry: ExpertRegistry = serde_json::from_str(&content)
            .map_err(|e| Error::Parse(format!("Failed to parse registry: {}", e)))?;

        Ok(registry)
    }

    /// Save registry to default location
    pub fn save(&self) -> Result<(), Error> {
        let registry_path = Self::default_path()?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = registry_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::Io(e))?;
        }

        let mut updated = self.clone();
        updated.last_updated = Utc::now();

        let content = serde_json::to_string_pretty(&updated)
            .map_err(|e| Error::Parse(format!("Failed to serialize registry: {}", e)))?;

        std::fs::write(&registry_path, content).map_err(|e| Error::Io(e))?;

        Ok(())
    }

    /// Get default registry path
    pub fn default_path() -> Result<PathBuf, Error> {
        let home = dirs::home_dir()
            .ok_or_else(|| Error::Config("Could not determine home directory".to_string()))?;

        Ok(home.join(".expert").join("expert-registry.json"))
    }

    /// Get default registry
    pub fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let expert_dir = home.join(".expert");

        Self::new(expert_dir.join("experts"), expert_dir.join("models"))
    }

    /// Add or update an expert version entry
    pub fn add_expert_version(&mut self, name: &str, version: ExpertVersionEntry) {
        if let Some(record) = self.experts.iter_mut().find(|r| r.name == name) {
            record.versions.retain(|v| v.version != version.version);
            record.versions.push(version);
            record
                .versions
                .sort_by(|a, b| compare_versions(&b.version, &a.version));
        } else {
            self.experts.push(ExpertRecord {
                name: name.to_string(),
                versions: vec![version],
            });
            self.experts.sort_by(|a, b| a.name.cmp(&b.name));
        }
    }

    /// Remove all versions for an expert, returning the removed record
    pub fn remove_expert(&mut self, name: &str) -> Option<ExpertRecord> {
        let index = self.experts.iter().position(|e| e.name == name)?;
        Some(self.experts.remove(index))
    }

    /// Remove a specific expert version entry
    pub fn remove_expert_version(
        &mut self,
        name: &str,
        version: &str,
    ) -> Option<ExpertVersionEntry> {
        if let Some(record) = self.experts.iter_mut().find(|r| r.name == name) {
            let index = record.versions.iter().position(|v| v.version == version)?;
            let removed = record.versions.remove(index);
            if record.versions.is_empty() {
                self.remove_expert(name);
            }
            Some(removed)
        } else {
            None
        }
    }

    /// Remove all versions for an expert, returning removed entries
    pub fn remove_all_versions(&mut self, name: &str) -> Vec<ExpertVersionEntry> {
        self.remove_expert(name)
            .map(|record| record.versions)
            .unwrap_or_default()
    }

    /// Add or update a base model entry
    pub fn add_base_model(&mut self, model: BaseModelEntry) {
        // Remove existing entry if present
        self.base_models.retain(|m| m.name != model.name);
        self.base_models.push(model);
    }

    /// Remove a base model entry
    pub fn remove_base_model(&mut self, name: &str) -> Option<BaseModelEntry> {
        let index = self.base_models.iter().position(|m| m.name == name)?;
        Some(self.base_models.remove(index))
    }

    /// Get expert by name (latest version)
    pub fn get_expert(&self, name: &str) -> Option<&ExpertVersionEntry> {
        self.experts
            .iter()
            .find(|record| record.name == name)
            .and_then(|record| record.versions.first())
    }

    /// Get expert by name and version
    pub fn get_expert_version(&self, name: &str, version: &str) -> Option<&ExpertVersionEntry> {
        self.experts
            .iter()
            .find(|record| record.name == name)
            .and_then(|record| record.versions.iter().find(|v| v.version == version))
    }

    /// Get expert record (all versions)
    pub fn get_expert_record(&self, name: &str) -> Option<&ExpertRecord> {
        self.experts.iter().find(|record| record.name == name)
    }

    /// Get base model by name
    pub fn get_base_model(&self, name: &str) -> Option<&BaseModelEntry> {
        self.base_models.iter().find(|m| m.name == name)
    }

    /// Expose expert records
    pub fn expert_records(&self) -> &[ExpertRecord] {
        &self.experts
    }

    /// Iterate over all expert/version pairs
    pub fn iter_versions(&self) -> impl Iterator<Item = (&str, &ExpertVersionEntry)> {
        self.experts.iter().flat_map(|record| {
            record
                .versions
                .iter()
                .map(move |version| (record.name.as_str(), version))
        })
    }

    /// List expert entries filtered by name
    pub fn list_expert_versions(&self, name: &str) -> Vec<&ExpertVersionEntry> {
        self.experts
            .iter()
            .find(|record| record.name == name)
            .map(|record| record.versions.iter().collect())
            .unwrap_or_else(Vec::new)
    }

    /// List all installed base models
    pub fn list_base_models(&self) -> &[BaseModelEntry] {
        &self.base_models
    }

    /// Check if expert is installed
    pub fn has_expert(&self, name: &str) -> bool {
        self.experts.iter().any(|record| record.name == name)
    }

    /// Check if specific expert version is installed
    pub fn has_expert_version(&self, name: &str, version: &str) -> bool {
        self.experts
            .iter()
            .any(|record| record.versions.iter().any(|entry| entry.version == version))
    }

    /// Check if base model is installed
    pub fn has_base_model(&self, name: &str) -> bool {
        self.base_models.iter().any(|m| m.name == name)
    }

    /// Validate registry integrity
    pub fn validate(&self) -> Result<(), Error> {
        // Check that install directories exist
        if !self.install_dir.exists() {
            return Err(Error::Config(format!(
                "Install directory does not exist: {}",
                self.install_dir.display()
            )));
        }

        if !self.models_dir.exists() {
            return Err(Error::Config(format!(
                "Models directory does not exist: {}",
                self.models_dir.display()
            )));
        }

        // Validate each expert version
        for record in &self.experts {
            for version in &record.versions {
                if !version.path.exists() {
                    return Err(Error::Config(format!(
                        "Expert path does not exist: {} ({})",
                        record.name,
                        version.path.display()
                    )));
                }

                if !self.has_base_model(&version.base_model) {
                    return Err(Error::Config(format!(
                        "Expert {} requires base model {} which is not installed",
                        record.name, version.base_model
                    )));
                }
            }
        }

        // Validate each base model
        for model in &self.base_models {
            if !model.path.exists() {
                return Err(Error::Config(format!(
                    "Base model path does not exist: {} ({})",
                    model.name,
                    model.path.display()
                )));
            }
        }

        Ok(())
    }

    /// Rebuild registry by scanning directories
    pub fn rebuild(&mut self) -> Result<(), Error> {
        self.experts.clear();
        self.base_models.clear();

        // Scan experts directory
        if self.install_dir.exists() {
            self.scan_experts_dir()?;
        }

        // Scan models directory
        if self.models_dir.exists() {
            self.scan_models_dir()?;
        }

        Ok(())
    }

    fn scan_experts_dir(&mut self) -> Result<(), Error> {
        for entry in std::fs::read_dir(&self.install_dir).map_err(|e| Error::Io(e))? {
            let entry = entry.map_err(|e| Error::Io(e))?;
            let path = entry.path();

            if path.is_dir() {
                let manifest_path = path.join("manifest.json");
                if manifest_path.exists() {
                    // TODO: Parse manifest and create ExpertRecord
                    // This requires importing manifest.rs types
                }
            }
        }

        Ok(())
    }

    fn scan_models_dir(&mut self) -> Result<(), Error> {
        for entry in std::fs::read_dir(&self.models_dir).map_err(|e| Error::Io(e))? {
            let entry = entry.map_err(|e| Error::Io(e))?;
            let path = entry.path();

            if path.is_dir() {
                // Check for HuggingFace model structure
                let config_path = path.join("config.json");
                if config_path.exists() {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let size = Self::dir_size(&path)?;

                    self.add_base_model(BaseModelEntry {
                        name,
                        path,
                        sha256: None,
                        quantization: None,
                        size_bytes: size,
                        installed_at: Utc::now(),
                        source: "local".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    fn dir_size(path: &PathBuf) -> Result<u64, Error> {
        let mut size = 0u64;

        for entry in std::fs::read_dir(path).map_err(|e| Error::Io(e))? {
            let entry = entry.map_err(|e| Error::Io(e))?;
            let metadata = entry.metadata().map_err(|e| Error::Io(e))?;

            if metadata.is_file() {
                size += metadata.len();
            } else if metadata.is_dir() {
                size += Self::dir_size(&entry.path())?;
            }
        }

        Ok(size)
    }
}

fn compare_versions(a: &str, b: &str) -> std::cmp::Ordering {
    match (Version::parse(a), Version::parse(b)) {
        (Ok(va), Ok(vb)) => va.cmp(&vb),
        _ => a.cmp(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_registry() {
        let registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        assert_eq!(registry.version, "1.0");
        assert_eq!(registry.experts.len(), 0);
        assert_eq!(registry.base_models.len(), 0);
    }

    #[test]
    fn test_add_remove_expert() {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        let expert = ExpertVersionEntry {
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-test/0.0.1"),
            source: "git+https://example.com".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        registry.add_expert_version("expert-test", expert);
        assert_eq!(registry.experts.len(), 1);
        assert!(registry.has_expert("expert-test"));

        let removed = registry.remove_expert("expert-test");
        assert!(removed.is_some());
        assert_eq!(registry.experts.len(), 0);
    }

    #[test]
    fn test_multiple_versions() {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        let v1 = ExpertVersionEntry {
            version: "0.2.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.2.1"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        let v2 = ExpertVersionEntry {
            version: "0.3.0".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.3.0"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        registry.add_expert_version("expert-sql", v1);
        registry.add_expert_version("expert-sql", v2);

        assert_eq!(registry.experts.len(), 1);
        assert!(registry.has_expert("expert-sql"));
        assert!(registry.has_expert_version("expert-sql", "0.2.1"));
        assert!(registry.has_expert_version("expert-sql", "0.3.0"));

        let versions = registry.list_expert_versions("expert-sql");
        assert_eq!(versions.len(), 2);

        // Latest version should be 0.3.0
        let latest = registry.get_expert("expert-sql").unwrap();
        assert_eq!(latest.version, "0.3.0");
    }

    #[test]
    fn test_get_expert_version() {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        let v1 = ExpertVersionEntry {
            version: "0.2.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.2.1"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        let v2 = ExpertVersionEntry {
            version: "0.3.0".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.3.0"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        registry.add_expert_version("expert-sql", v1);
        registry.add_expert_version("expert-sql", v2);

        let v021 = registry.get_expert_version("expert-sql", "0.2.1");
        assert!(v021.is_some());
        assert_eq!(v021.unwrap().version, "0.2.1");

        let v030 = registry.get_expert_version("expert-sql", "0.3.0");
        assert!(v030.is_some());
        assert_eq!(v030.unwrap().version, "0.3.0");

        let v999 = registry.get_expert_version("expert-sql", "0.9.9");
        assert!(v999.is_none());
    }

    #[test]
    fn test_remove_expert_version() {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        let v1 = ExpertVersionEntry {
            version: "0.2.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.2.1"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        let v2 = ExpertVersionEntry {
            version: "0.3.0".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.3.0"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        registry.add_expert_version("expert-sql", v1);
        registry.add_expert_version("expert-sql", v2);

        assert_eq!(registry.experts.len(), 1);
        let versions = registry.list_expert_versions("expert-sql");
        assert_eq!(versions.len(), 2);

        // Remove one version
        let removed = registry.remove_expert_version("expert-sql", "0.2.1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().version, "0.2.1");

        assert!(registry.has_expert("expert-sql"));
        assert!(!registry.has_expert_version("expert-sql", "0.2.1"));
        assert!(registry.has_expert_version("expert-sql", "0.3.0"));

        let versions = registry.list_expert_versions("expert-sql");
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].version, "0.3.0");
    }

    #[test]
    fn test_remove_all_versions() {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        let v1 = ExpertVersionEntry {
            version: "0.2.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.2.1"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        let v2 = ExpertVersionEntry {
            version: "0.3.0".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-sql/0.3.0"),
            source: "package".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };

        registry.add_expert_version("expert-sql", v1);
        registry.add_expert_version("expert-sql", v2);

        let removed = registry.remove_all_versions("expert-sql");
        assert_eq!(removed.len(), 2);
        assert_eq!(registry.experts.len(), 0);
        assert!(!registry.has_expert("expert-sql"));
    }

    #[test]
    fn test_version_sorting() {
        let mut registry =
            ExpertRegistry::new(PathBuf::from("/tmp/experts"), PathBuf::from("/tmp/models"));

        // Add versions in non-sequential order
        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.3.0".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/experts/expert-sql/0.3.0"),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: Vec::new(),
                dependencies: Vec::new(),
            },
        );

        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.2.1".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/experts/expert-sql/0.2.1"),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: Vec::new(),
                dependencies: Vec::new(),
            },
        );

        registry.add_expert_version(
            "expert-sql",
            ExpertVersionEntry {
                version: "0.1.0".to_string(),
                base_model: "Qwen3-0.6B".to_string(),
                path: PathBuf::from("/tmp/experts/expert-sql/0.1.0"),
                source: "package".to_string(),
                installed_at: Utc::now(),
                adapters: Vec::new(),
                capabilities: Vec::new(),
                dependencies: Vec::new(),
            },
        );

        // get_expert should return latest version
        let latest = registry.get_expert("expert-sql").unwrap();
        assert_eq!(latest.version, "0.3.0");

        // list_expert_versions should be sorted descending
        let versions = registry.list_expert_versions("expert-sql");
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].version, "0.3.0");
        assert_eq!(versions[1].version, "0.2.1");
        assert_eq!(versions[2].version, "0.1.0");
    }

    #[test]
    fn test_legacy_migration() {
        // Create legacy registry JSON
        let legacy_json = r#"
        {
            "version": "1.0",
            "last_updated": "2025-01-01T00:00:00Z",
            "install_dir": "/tmp/experts",
            "models_dir": "/tmp/models",
            "base_models": [],
            "experts": [
                {
                    "name": "expert-sql",
                    "version": "0.2.1",
                    "base_model": "Qwen3-0.6B",
                    "path": "/tmp/experts/expert-sql",
                    "source": "package",
                    "installed_at": "2025-01-01T00:00:00Z",
                    "adapters": [],
                    "capabilities": [],
                    "dependencies": []
                }
            ]
        }
        "#;

        // Parse directly without file I/O
        let registry: ExpertRegistry = serde_json::from_str(legacy_json).unwrap();

        assert_eq!(registry.experts.len(), 1);
        assert!(registry.has_expert("expert-sql"));
        assert!(registry.has_expert_version("expert-sql", "0.2.1"));
    }
}
