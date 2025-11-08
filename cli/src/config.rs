use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub project_root: PathBuf,
    pub models_dir: PathBuf,
    pub experts_dir: PathBuf,
    pub venv_dir: PathBuf,
}

impl AppConfig {
    pub fn new(
        project_root: Option<PathBuf>,
        models_dir: Option<PathBuf>,
        experts_dir: Option<PathBuf>,
    ) -> Self {
        let root = project_root.unwrap_or_else(|| Self::detect_project_root());

        let models = models_dir.unwrap_or_else(|| root.join("models"));
        let experts = experts_dir.unwrap_or_else(|| root.join("experts"));
        let venv = root.join("cli").join("venv_windows");

        Self {
            project_root: root,
            models_dir: models,
            experts_dir: experts,
            venv_dir: venv,
        }
    }

    fn detect_project_root() -> PathBuf {
        // Try to detect from current exe location
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                // If running from target/release, go up to cli, then to project root
                if parent.ends_with("release") {
                    if let Some(cli_dir) = parent.parent().and_then(|p| p.parent()) {
                        if let Some(project_root) = cli_dir.parent() {
                            return project_root.to_path_buf();
                        }
                    }
                }
            }
        }

        // Fallback: assume current directory is project root or subdirectory
        if let Ok(current_dir) = std::env::current_dir() {
            if current_dir.ends_with("cli") {
                if let Some(parent) = current_dir.parent() {
                    return parent.to_path_buf();
                }
            }
            return current_dir;
        }

        // Last resort fallback
        PathBuf::from(".")
    }

    pub fn default_base_model(&self) -> PathBuf {
        self.models_dir.join("Qwen3-0.6B")
    }
}
