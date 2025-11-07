use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use chrono::{DateTime, Utc};

#[cfg(not(test))]
use crate::error::Error;

#[cfg(test)]
pub use crate::error::Error;

/// Expert Registry - tracks installed experts and base models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRegistry {
    pub version: String,
    pub last_updated: DateTime<Utc>,
    pub install_dir: PathBuf,
    pub models_dir: PathBuf,
    pub base_models: Vec<BaseModelEntry>,
    pub experts: Vec<ExpertEntry>,
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
pub struct ExpertEntry {
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
        
        let content = std::fs::read_to_string(&registry_path)
            .map_err(|e| Error::Io(e))?;
        
        let registry: ExpertRegistry = serde_json::from_str(&content)
            .map_err(|e| Error::Parse(format!("Failed to parse registry: {}", e)))?;
        
        Ok(registry)
    }

    /// Save registry to default location
    pub fn save(&self) -> Result<(), Error> {
        let registry_path = Self::default_path()?;
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = registry_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Io(e))?;
        }
        
        let mut updated = self.clone();
        updated.last_updated = Utc::now();
        
        let content = serde_json::to_string_pretty(&updated)
            .map_err(|e| Error::Parse(format!("Failed to serialize registry: {}", e)))?;
        
        std::fs::write(&registry_path, content)
            .map_err(|e| Error::Io(e))?;
        
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
        
        Self::new(
            expert_dir.join("experts"),
            expert_dir.join("models"),
        )
    }

    /// Add or update an expert entry
    pub fn add_expert(&mut self, expert: ExpertEntry) {
        // Remove existing entry if present
        self.experts.retain(|e| e.name != expert.name);
        self.experts.push(expert);
    }

    /// Remove an expert entry
    pub fn remove_expert(&mut self, name: &str) -> Option<ExpertEntry> {
        let index = self.experts.iter().position(|e| e.name == name)?;
        Some(self.experts.remove(index))
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

    /// Get expert by name
    pub fn get_expert(&self, name: &str) -> Option<&ExpertEntry> {
        self.experts.iter().find(|e| e.name == name)
    }

    /// Get base model by name
    pub fn get_base_model(&self, name: &str) -> Option<&BaseModelEntry> {
        self.base_models.iter().find(|m| m.name == name)
    }

    /// List all installed experts
    pub fn list_experts(&self) -> &[ExpertEntry] {
        &self.experts
    }

    /// List all installed base models
    pub fn list_base_models(&self) -> &[BaseModelEntry] {
        &self.base_models
    }

    /// Check if expert is installed
    pub fn has_expert(&self, name: &str) -> bool {
        self.experts.iter().any(|e| e.name == name)
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
        
        // Validate each expert
        for expert in &self.experts {
            if !expert.path.exists() {
                return Err(Error::Config(format!(
                    "Expert path does not exist: {} ({})",
                    expert.name,
                    expert.path.display()
                )));
            }
            
            // Check base model exists
            if !self.has_base_model(&expert.base_model) {
                return Err(Error::Config(format!(
                    "Expert {} requires base model {} which is not installed",
                    expert.name,
                    expert.base_model
                )));
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
                    // TODO: Parse manifest and create ExpertEntry
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
                    let name = path.file_name()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_registry() {
        let registry = ExpertRegistry::new(
            PathBuf::from("/tmp/experts"),
            PathBuf::from("/tmp/models"),
        );
        
        assert_eq!(registry.version, "1.0");
        assert_eq!(registry.experts.len(), 0);
        assert_eq!(registry.base_models.len(), 0);
    }

    #[test]
    fn test_add_remove_expert() {
        let mut registry = ExpertRegistry::new(
            PathBuf::from("/tmp/experts"),
            PathBuf::from("/tmp/models"),
        );
        
        let expert = ExpertEntry {
            name: "expert-test".to_string(),
            version: "0.0.1".to_string(),
            base_model: "Qwen3-0.6B".to_string(),
            path: PathBuf::from("/tmp/experts/expert-test"),
            source: "git+https://example.com".to_string(),
            installed_at: Utc::now(),
            adapters: Vec::new(),
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        };
        
        registry.add_expert(expert.clone());
        assert_eq!(registry.experts.len(), 1);
        assert!(registry.has_expert("expert-test"));
        
        let removed = registry.remove_expert("expert-test");
        assert!(removed.is_some());
        assert_eq!(registry.experts.len(), 0);
    }
}

