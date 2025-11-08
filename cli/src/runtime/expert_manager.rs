// Multi-expert manager with hot-swap and pre-loading

use crate::inference::QwenEngine;
use crate::manifest::Manifest;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ExpertState {
    pub name: String,
    pub manifest: Manifest,
    pub path: PathBuf,
    pub loaded: bool,
    pub engine: Option<Arc<Mutex<QwenEngine>>>,
    pub load_time: Option<std::time::Instant>,
    pub last_used: Option<std::time::Instant>,
}

// Manual Debug implementation since QwenEngine doesn't implement Debug
impl std::fmt::Debug for ExpertState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertState")
            .field("name", &self.name)
            .field("manifest", &self.manifest)
            .field("path", &self.path)
            .field("loaded", &self.loaded)
            .field("engine", &self.engine.is_some())
            .field("load_time", &self.load_time)
            .field("last_used", &self.last_used)
            .finish()
    }
}

#[derive(Debug)]
pub struct ExpertManager {
    experts: HashMap<String, ExpertState>,
    max_loaded: usize,
    preload_queue: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LoadStats {
    pub total_experts: usize,
    pub loaded_experts: usize,
    pub memory_mb: f64,
    pub avg_load_time_ms: f64,
}

impl ExpertManager {
    /// Create new expert manager
    pub fn new(max_loaded: usize) -> Self {
        Self {
            experts: HashMap::new(),
            max_loaded,
            preload_queue: Vec::new(),
        }
    }

    /// Register an expert (without loading)
    pub fn register_expert(&mut self, name: String, manifest: Manifest, path: PathBuf) {
        self.experts.insert(
            name.clone(),
            ExpertState {
                name: name.clone(),
                manifest,
                path,
                loaded: false,
                engine: None,
                load_time: None,
                last_used: None,
            },
        );
    }

    /// Load an expert (hot-swap)
    pub fn load_expert(&mut self, name: &str, use_cuda: bool) -> anyhow::Result<()> {
        // First check if already loaded (without holding mutable borrow)
        {
            let expert = self
                .experts
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Expert {} not found", name))?;

            if expert.loaded {
                // Already loaded, just update last_used
                self.experts.get_mut(name).unwrap().last_used = Some(std::time::Instant::now());
                return Ok(());
            }
        }

        // Check if we need to unload another expert
        if self.loaded_count() >= self.max_loaded {
            self.unload_lru()?;
        }

        // Get expert path for loading (before taking mutable borrow)
        let expert_path = self
            .experts
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Expert {} not found", name))?
            .path
            .clone();

        // Load the expert
        let start = std::time::Instant::now();
        let engine = QwenEngine::from_local(&expert_path, use_cuda)?;
        let load_time = start.elapsed();

        // Update expert state
        let expert = self.experts.get_mut(name).unwrap();
        expert.engine = Some(Arc::new(Mutex::new(engine)));
        expert.loaded = true;
        expert.load_time = Some(start);
        expert.last_used = Some(std::time::Instant::now());

        println!(
            "✅ Loaded expert '{}' in {:.2}ms",
            name,
            load_time.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Unload an expert (hot-swap)
    pub fn unload_expert(&mut self, name: &str) -> anyhow::Result<()> {
        let expert = self
            .experts
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("Expert {} not found", name))?;

        if !expert.loaded {
            return Ok(()); // Already unloaded
        }

        expert.engine = None;
        expert.loaded = false;
        expert.load_time = None;

        println!("✅ Unloaded expert '{}'", name);

        Ok(())
    }

    /// Unload least-recently-used expert
    fn unload_lru(&mut self) -> anyhow::Result<()> {
        let mut lru_name: Option<String> = None;
        let mut lru_time: Option<std::time::Instant> = None;

        for (name, expert) in &self.experts {
            if expert.loaded {
                if let Some(last_used) = expert.last_used {
                    if lru_time.is_none() || last_used < lru_time.unwrap() {
                        lru_name = Some(name.clone());
                        lru_time = Some(last_used);
                    }
                }
            }
        }

        if let Some(name) = lru_name {
            self.unload_expert(&name)?;
        }

        Ok(())
    }

    /// Get loaded expert count
    fn loaded_count(&self) -> usize {
        self.experts.values().filter(|e| e.loaded).count()
    }

    /// Pre-load experts based on priority
    pub fn preload_priority(&mut self, use_cuda: bool) -> anyhow::Result<()> {
        // Sort experts by priority (from manifest routing.priority)
        let mut experts_to_load: Vec<(String, f32)> = self
            .experts
            .iter()
            .filter(|(_, e)| !e.loaded)
            .map(|(name, e)| {
                let priority = e
                    .manifest
                    .routing
                    .as_ref()
                    .and_then(|r| r.priority)
                    .unwrap_or(0.5) as f32;
                (name.clone(), priority)
            })
            .collect();

        experts_to_load.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Pre-load top N
        for (name, _) in experts_to_load.iter().take(self.max_loaded) {
            if let Err(e) = self.load_expert(name, use_cuda) {
                eprintln!("⚠️ Failed to pre-load expert '{}': {}", name, e);
            }
        }

        Ok(())
    }

    /// Get expert engine (loads if needed)
    pub fn get_expert(
        &mut self,
        name: &str,
        use_cuda: bool,
    ) -> anyhow::Result<Arc<Mutex<QwenEngine>>> {
        if !self.experts.contains_key(name) {
            return Err(anyhow::anyhow!("Expert {} not registered", name));
        }

        // Load if not loaded
        if !self.experts.get(name).unwrap().loaded {
            self.load_expert(name, use_cuda)?;
        }

        // Update last_used
        if let Some(expert) = self.experts.get_mut(name) {
            expert.last_used = Some(std::time::Instant::now());
            if let Some(engine) = &expert.engine {
                return Ok(engine.clone());
            }
        }

        Err(anyhow::anyhow!("Expert {} not loaded", name))
    }

    /// Get statistics
    pub fn stats(&self) -> LoadStats {
        let loaded: Vec<_> = self.experts.values().filter(|e| e.loaded).collect();
        let total_load_time: f64 = loaded
            .iter()
            .filter_map(|e| e.load_time.map(|t| t.elapsed().as_secs_f64() * 1000.0))
            .sum();

        let avg_load_time = if !loaded.is_empty() {
            total_load_time / loaded.len() as f64
        } else {
            0.0
        };

        LoadStats {
            total_experts: self.experts.len(),
            loaded_experts: loaded.len(),
            memory_mb: self.estimate_memory_mb(),
            avg_load_time_ms: avg_load_time,
        }
    }

    /// Estimate memory usage (simplified)
    fn estimate_memory_mb(&self) -> f64 {
        // Rough estimate: ~500MB per loaded expert
        self.loaded_count() as f64 * 500.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_manager() {
        let mut manager = ExpertManager::new(2);

        // Register expert
        let mut manifest = Manifest::default();
        manifest.name = "test-expert".to_string();
        manager.register_expert("test-expert".to_string(), manifest, PathBuf::from("test"));

        // Stats
        let stats = manager.stats();
        assert_eq!(stats.total_experts, 1);
        assert_eq!(stats.loaded_experts, 0);
    }
}
