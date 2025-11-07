use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::Result;
use crate::manifest::{Manifest, TrainingConfig};

pub struct PythonTrainer;

impl PythonTrainer {
    pub fn new() -> Result<Self> {
        // Setup venv environment variables (but don't initialize Python GIL)
        // This only sets environment variables, PyO3 won't initialize until GIL is requested
        Self::setup_venv_environment_variables_only()?;
        Ok(Self)
    }
    
    /// Setup venv environment variables ONLY (no Python initialization)
    fn setup_venv_environment_variables_only() -> Result<()> {
        use std::env;
        
        // Only setup if VIRTUAL_ENV is not already set
        if env::var("VIRTUAL_ENV").is_ok() {
            return Ok(());
        }
        
        // Find CLI directory (where venv_windows is located)
        let cli_dir = if let Ok(exe_path) = env::current_exe() {
            exe_path
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
        } else {
            None
        };
        
        if let Some(cli_dir) = cli_dir {
            // Check for venv_windows
            let venv_windows = cli_dir.join("venv_windows");
            let venv_unix = cli_dir.join("venv");
            
            let venv_path = if venv_windows.exists() {
                Some(venv_windows)
            } else if venv_unix.exists() {
                Some(venv_unix)
            } else {
                None
            };
            
            if let Some(venv) = venv_path {
                // Set VIRTUAL_ENV to activate the venv
                // SAFETY: We're setting environment variables before Python initialization
                // This is safe because we're the only thread at this point
                unsafe {
                    env::set_var("VIRTUAL_ENV", &venv);
                }
                println!("✅ Auto-detected venv: {}", venv.display());
                
                // Also update PATH to include venv Scripts/bin
                #[cfg(windows)]
                let venv_bin = venv.join("Scripts");
                #[cfg(not(windows))]
                let venv_bin = venv.join("bin");
                
                if venv_bin.exists() {
                    if let Ok(current_path) = env::var("PATH") {
                        let new_path = format!("{};{}", venv_bin.display(), current_path);
                        // SAFETY: Setting PATH before Python initialization
                        unsafe {
                            env::set_var("PATH", new_path);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Setup Python sys.path to include venv site-packages (DEPRECATED - not used for subprocess)
    #[allow(dead_code)]
    fn setup_python_path(py: Python) -> Result<()> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        let path_list: &Bound<PyList> = path.downcast()?;
        
        // Try to detect active virtual environment from VIRTUAL_ENV env var
        let venv_site_packages = if let Ok(venv_path) = std::env::var("VIRTUAL_ENV") {
            let venv_root = PathBuf::from(venv_path);
            println!("🔍 Detected active venv: {}", venv_root.display());
            
            // Windows: venv\Lib\site-packages
            // Unix: venv/lib/python3.x/site-packages
            #[cfg(windows)]
            let site_packages = venv_root.join("Lib").join("site-packages");
            
            #[cfg(not(windows))]
            let site_packages = {
                let lib_dir = venv_root.join("lib");
                if let Ok(entries) = std::fs::read_dir(&lib_dir) {
                    entries
                        .filter_map(|e| e.ok())
                        .find(|e| e.file_name().to_string_lossy().starts_with("python"))
                        .map(|e| e.path().join("site-packages"))
                        .unwrap_or_else(|| lib_dir.join("site-packages"))
                } else {
                    lib_dir.join("site-packages")
                }
            };
            
            Some(site_packages)
        } else {
            // Fallback: try common venv locations in current directory
            let current_dir = std::env::current_dir()?;
            let candidates = vec![
                current_dir.join("venv_windows").join("Lib").join("site-packages"),
                current_dir.join("venv").join("Lib").join("site-packages"),
                current_dir.join(".venv").join("Lib").join("site-packages"),
            ];
            
            println!("🔍 No active venv detected, searching in current directory...");
            candidates.into_iter().find(|p| p.exists())
        };
        
        if let Some(sp_path) = venv_site_packages {
            if sp_path.exists() {
                println!("✅ Found venv at: {}", sp_path.display());
                let venv_str = sp_path.to_str().ok_or_else(|| {
                    crate::error::Error::Training("Invalid venv path".to_string())
                })?;
                path_list.insert(0, venv_str)?;
                println!("✅ Added venv site-packages to sys.path");
            } else {
                println!("⚠️  Warning: venv path does not exist: {}", sp_path.display());
            }
        } else {
            println!("⚠️  Warning: No Python virtual environment found");
            println!("   Activate a venv or ensure torch is installed globally");
        }
        
        Ok(())
    }

    pub fn train_via_subprocess(
        &self,
        manifest: &Manifest,
        dataset_path: &Path,
        output_dir: &Path,
        device: &str,
        _resume_checkpoint: Option<&Path>,
    ) -> Result<()> {
        // Find venv Python
        let venv_python = if cfg!(windows) {
            PathBuf::from("F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe")
        } else {
            PathBuf::from("venv/bin/python")
        };
        
        if !venv_python.exists() {
            return Err(crate::error::Error::Training(
                format!("Python venv not found at {:?}", venv_python)
            ));
        }
        
        // Build config JSON
        let base_models = manifest.get_base_models();
        let base_model_name = base_models.first()
            .ok_or_else(|| crate::error::Error::Training("No base model found".to_string()))?;
        
        let quantization = if let Some(ref base) = manifest.base_model {
            base.quantization.as_deref().unwrap_or("int4")
        } else if let Some(ref models) = manifest.base_models {
            models.first()
                .and_then(|m| m.quantization.as_deref())
                .unwrap_or("int4")
        } else {
            "int4"
        };
        
        let config = serde_json::json!({
            "base_model_name": base_model_name,
            "quantization": quantization,
            "dataset_path": dataset_path.to_str().unwrap(),
            "output_dir": output_dir.to_str().unwrap(),
            "device": device,
            "training": manifest.training.config,
            "dataset": manifest.training.dataset,
        });
        
        // Write config to temp file
        let temp_config = std::env::temp_dir().join("expert_train_config.json");
        let config_json = serde_json::to_string_pretty(&config)?;
        std::fs::write(&temp_config, &config_json)?;
        
        // Debug: Verify file was written
        if !temp_config.exists() {
            return Err(crate::error::Error::Training(
                format!("Failed to create temp config at: {}", temp_config.display())
            ));
        }
        println!("   [DEBUG] Config written to: {}", temp_config.display());
        
        // Get expert_trainer.py path (in CLI root, not scripts/)
        let train_script = if let Ok(exe_path) = std::env::current_exe() {
            exe_path.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|p| p.join("expert_trainer.py"))
                .unwrap_or_else(|| PathBuf::from("expert_trainer.py"))
        } else {
            PathBuf::from("expert_trainer.py")
        };
        
        // Execute Python script with config JSON file
        let mut cmd = Command::new(&venv_python);
        cmd.arg(train_script)
           .arg(temp_config.to_str().unwrap())  // expert_trainer.py expects config.json as first arg
           .stdout(Stdio::inherit())
           .stderr(Stdio::inherit());
        
        // Set environment variables for PyTorch/CUDA
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            cmd.env("CUDA_PATH", cuda_path);
        }
        if let Ok(cudnn_path) = std::env::var("CUDNN_PATH") {
            cmd.env("CUDNN_PATH", cudnn_path);
        }
        
        // Disable torch.compile (Triton incompatible with Windows)
        cmd.env("PYTORCH_DISABLE_COMPILE", "1");
        cmd.env("TORCH_COMPILE_DISABLE", "1");
        
        // Ensure venv activation
        cmd.env("VIRTUAL_ENV", venv_python.parent().unwrap().parent().unwrap());
        
        let status = cmd.status()
            .map_err(|e| crate::error::Error::Training(format!("Failed to execute Python: {}", e)))?;
        
        // Cleanup temp file
        let _ = std::fs::remove_file(&temp_config);
        
        if status.success() {
            Ok(())
        } else {
            Err(crate::error::Error::Training(
                format!("Training failed with exit code: {:?}", status.code())
            ))
        }
    }

    pub fn run_chat(
        &self,
        expert_packages: Vec<PathBuf>,
        base_model_path: PathBuf,
        device: String,
    ) -> Result<()> {
        // Find venv Python
        let venv_python = if cfg!(windows) {
            PathBuf::from("F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe")
        } else {
            PathBuf::from("venv/bin/python")
        };
        
        if !venv_python.exists() {
            return Err(crate::error::Error::Training(
                format!("Python venv not found at {:?}", venv_python)
            ));
        }
        
        // Get chat script path
        let chat_script = if let Ok(exe_path) = std::env::current_exe() {
            exe_path.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|p| p.join("scripts").join("expert_chat.py"))
                .unwrap_or_else(|| PathBuf::from("scripts/expert_chat.py"))
        } else {
            PathBuf::from("scripts/expert_chat.py")
        };
        
        // Build command args
        let mut cmd = std::process::Command::new(&venv_python);
        cmd.arg(chat_script)
           .arg("--base-model")
           .arg(base_model_path.to_str().unwrap())
           .arg("--device")
           .arg(device);
        
        // Add expert packages
        for package in expert_packages {
            cmd.arg("--expert").arg(package.to_str().unwrap());
        }
        
        cmd.stdout(std::process::Stdio::inherit())
           .stderr(std::process::Stdio::inherit())
           .stdin(std::process::Stdio::inherit());
        
        let status = cmd.status()
            .map_err(|e| crate::error::Error::Training(format!("Failed to execute chat: {}", e)))?;
        
        if status.success() {
            Ok(())
        } else {
            Err(crate::error::Error::Training(
                format!("Chat exited with code: {:?}", status.code())
            ))
        }
    }

    pub fn check_cuda_available(&self) -> Result<bool> {
        Python::with_gil(|py| {
            // Setup Python path first
            Self::setup_python_path(py)?;
            
            let torch = py.import_bound("torch")?;
            let cuda_available: bool = torch.getattr("cuda")?.getattr("is_available")?.call0()?.extract()?;
            Ok(cuda_available)
        })
    }

    pub fn train(
        &self,
        manifest: &Manifest,
        dataset_path: &Path,
        output_dir: &Path,
        device: &str,
        resume_checkpoint: Option<&Path>,
    ) -> Result<()> {
        Python::with_gil(|py| {
            // Setup Python path first (including venv)
            Self::setup_python_path(py)?;
            
            // Import training module
            let sys = py.import_bound("sys")?;
            let path = sys.getattr("path")?;
            let path_list: &Bound<PyList> = path.downcast()?;
            
            // Get CLI directory where expert_trainer.py is located
            // If exe is at expert/cli/target/release/expert-cli.exe, we need expert/cli/
            let cli_dir = if let Ok(exe_path) = std::env::current_exe() {
                // Go up 2 levels: target/release -> target -> cli
                exe_path
                    .parent() // remove expert-cli.exe -> target/release
                    .and_then(|p| p.parent()) // remove release -> target
                    .and_then(|p| p.parent()) // remove target -> cli
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            } else {
                PathBuf::from(".")
            };
            
            // Add CLI directory (where expert_trainer.py is)
            let cli_dir_str = cli_dir.to_str().ok_or_else(|| {
                crate::error::Error::Training("Invalid CLI directory path".to_string())
            })?;
            path_list.insert(0, cli_dir_str)?;
            
            // Add current directory (where manifest.json is)
            let current_dir = std::env::current_dir()?;
            let current_dir_str = current_dir.to_str().ok_or_else(|| {
                crate::error::Error::Training("Invalid current directory path".to_string())
            })?;
            if current_dir_str != cli_dir_str {
                path_list.insert(0, current_dir_str)?;
            }

            // Import our training script
            let trainer_module = py.import_bound("expert_trainer")
                .map_err(|e| {
                    let paths: Vec<String> = (0..path_list.len())
                        .filter_map(|i| path_list.get_item(i).ok())
                        .filter_map(|item| item.extract::<String>().ok())
                        .collect();
                    crate::error::Error::Training(format!(
                        "Failed to import expert_trainer module. Python path: {:?}, Error: {}",
                        paths,
                        e
                    ))
                })?;
            let train_fn = trainer_module.getattr("train_expert")
                .map_err(|e| crate::error::Error::Training(format!(
                    "Failed to get train_expert function: {}",
                    e
                )))?;

            // Prepare configuration
            let config = PyDict::new_bound(py);
            
            // Base model config - handle both v1.0 and v2.0
            let base_models = manifest.get_base_models();
            let default_model = String::from("unknown");
            let base_model_name = base_models.first().unwrap_or(&default_model);
            config.set_item("base_model_name", base_model_name)?;
            
            let quantization = if let Some(ref base) = manifest.base_model {
                base.quantization.as_deref().unwrap_or("none")
            } else if let Some(ref models) = manifest.base_models {
                models.first()
                    .and_then(|m| m.quantization.as_deref())
                    .unwrap_or("none")
            } else {
                "none"
            };
            config.set_item("quantization", quantization)?;
            
            // Pass base_model dict for prompt_template extraction
            let base_model_dict = PyDict::new_bound(py);
            if let Some(ref models) = manifest.base_models {
                if let Some(first_model) = models.first() {
                    if let Some(ref template) = first_model.prompt_template {
                        base_model_dict.set_item("prompt_template", template)?;
                    }
                }
            }
            config.set_item("base_model", base_model_dict)?;
            
            // Training config
            let training_config = self.training_config_to_dict(py, &manifest.training.config)?;
            config.set_item("training", training_config)?;
            
            // Dataset config (for multi-task support)
            let dataset_json = serde_json::to_string(&manifest.training.dataset)
                .map_err(|e| crate::error::Error::Training(format!("Failed to serialize dataset config: {}", e)))?;
            let dataset_dict: Bound<PyDict> = py.eval_bound(&format!("__import__('json').loads(r'''{}''')", dataset_json), None, None)?
                .extract()?;
            config.set_item("dataset", dataset_dict)?;
            
            // Paths
            config.set_item("dataset_path", dataset_path.to_str().unwrap())?;
            
            // Optional validation and test paths from manifest
            if let Some(ref validation_path) = manifest.training.dataset.validation_path {
                config.set_item("validation_path", validation_path)?;
            }
            if let Some(ref test_path) = manifest.training.dataset.test_path {
                config.set_item("test_path", test_path)?;
            }
            
            config.set_item("output_dir", output_dir.to_str().unwrap())?;
            config.set_item("device", device)?;
            
            if let Some(checkpoint) = resume_checkpoint {
                config.set_item("resume_checkpoint", checkpoint.to_str().unwrap())?;
            }

            // Call training function
            println!("Starting training...");
            train_fn.call1((config,))?;
            
            Ok(())
        })
    }

    fn training_config_to_dict<'py>(&self, py: Python<'py>, config: &TrainingConfig) -> Result<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        
        // Required fields
        dict.set_item("method", &config.method)?;
        dict.set_item("adapter_type", &config.adapter_type)?;
        // Rank and alpha are optional (not needed for IA³)
        if let Some(rank) = config.rank {
            dict.set_item("rank", rank)?;
        }
        if let Some(alpha) = config.alpha {
            dict.set_item("alpha", alpha)?;
        }
        if let Some(ref feedforward_modules) = config.feedforward_modules {
            let ff_modules = PyList::new_bound(py, feedforward_modules);
            dict.set_item("feedforward_modules", ff_modules)?;
        }
        
        let target_modules = PyList::new_bound(py, &config.target_modules);
        dict.set_item("target_modules", target_modules)?;
        
        dict.set_item("epochs", config.epochs)?;
        dict.set_item("learning_rate", config.learning_rate)?;
        dict.set_item("batch_size", config.batch_size)?;
        dict.set_item("gradient_accumulation_steps", config.gradient_accumulation_steps)?;
        dict.set_item("warmup_steps", config.warmup_steps)?;
        dict.set_item("lr_scheduler", &config.lr_scheduler)?;
        
        // Optional advanced optimization parameters
        if let Some(val) = config.max_seq_length {
            dict.set_item("max_seq_length", val)?;
        }
        if let Some(val) = config.dataloader_num_workers {
            dict.set_item("dataloader_num_workers", val)?;
        }
        if let Some(val) = config.dataloader_pin_memory {
            dict.set_item("dataloader_pin_memory", val)?;
        }
        if let Some(val) = config.dataloader_prefetch_factor {
            dict.set_item("dataloader_prefetch_factor", val)?;
        }
        if let Some(val) = config.dataloader_persistent_workers {
            dict.set_item("dataloader_persistent_workers", val)?;
        }
        if let Some(val) = config.fp16 {
            dict.set_item("fp16", val)?;
        }
        if let Some(val) = config.bf16 {
            dict.set_item("bf16", val)?;
        }
        if let Some(val) = config.use_tf32 {
            dict.set_item("use_tf32", val)?;
        }
        if let Some(val) = config.use_sdpa {
            dict.set_item("use_sdpa", val)?;
        }
        if let Some(ref val) = config.optim {
            dict.set_item("optim", val)?;
        }
        if let Some(val) = config.group_by_length {
            dict.set_item("group_by_length", val)?;
        }
        if let Some(val) = config.save_steps {
            dict.set_item("save_steps", val)?;
        }
        if let Some(val) = config.logging_steps {
            dict.set_item("logging_steps", val)?;
        }
        if let Some(ref val) = config.gradient_checkpointing {
            // Pass JSON value as string (Python will parse)
            let json_str = serde_json::to_string(val)?;
            dict.set_item("gradient_checkpointing_json", json_str)?;
        }
        if let Some(ref val) = config.gradient_checkpointing_kwargs {
            let json_str = serde_json::to_string(val)?;
            dict.set_item("gradient_checkpointing_kwargs_json", json_str)?;
        }
        if let Some(ref val) = config.lr_scheduler_kwargs {
            let json_str = serde_json::to_string(val)?;
            dict.set_item("lr_scheduler_kwargs_json", json_str)?;
        }
        if let Some(val) = config.flash_attention_2 {
            dict.set_item("flash_attention_2", val)?;
        }
        if let Some(val) = config.memory_efficient_attention {
            dict.set_item("memory_efficient_attention", val)?;
        }
        if let Some(ref val) = config.activation_checkpointing {
            dict.set_item("activation_checkpointing", val)?;
        }
        if let Some(val) = config.packing {
            dict.set_item("packing", val)?;
        }
        if let Some(val) = config.torch_compile {
            dict.set_item("torch_compile", val)?;
        }
        if let Some(ref val) = config.torch_compile_backend {
            dict.set_item("torch_compile_backend", val)?;
        }
        if let Some(ref val) = config.torch_compile_mode {
            dict.set_item("torch_compile_mode", val)?;
        }
        if let Some(ref val) = config.pretokenized_cache {
            dict.set_item("pretokenized_cache", val)?;
        }
        
        Ok(dict)
    }

    pub fn get_gpu_info(&self) -> Result<String> {
        Python::with_gil(|py| {
            // Setup Python path first
            Self::setup_python_path(py)?;
            
            let torch = py.import_bound("torch")?;
            
            if !torch.getattr("cuda")?.getattr("is_available")?.call0()?.extract::<bool>()? {
                return Ok("No CUDA devices available".to_string());
            }

            let device_count: i64 = torch.getattr("cuda")?.getattr("device_count")?.call0()?.extract()?;
            let device_name: String = torch.getattr("cuda")?.getattr("get_device_name")?.call1((0,))?.extract()?;
            
            Ok(format!("GPU: {} (count: {})", device_name, device_count))
        })
    }

    /// Generate text using Python transformers
    pub fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<String> {
        // Find venv Python
        let venv_python = if cfg!(windows) {
            PathBuf::from("F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe")
        } else {
            PathBuf::from("venv/bin/python")
        };

        if !venv_python.exists() {
            return Err(crate::error::Error::Training(
                format!("Python venv not found at {:?}", venv_python)
            ));
        }

        // Get generation script path
        let gen_script = if let Ok(exe_path) = std::env::current_exe() {
            exe_path.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|p| p.join("scripts").join("test_simple_gen.py"))
                .unwrap_or_else(|| PathBuf::from("scripts/test_simple_gen.py"))
        } else {
            PathBuf::from("scripts/test_simple_gen.py")
        };

        // Execute Python script with arguments
        let top_p_str = top_p.map(|p| p.to_string()).unwrap_or_else(|| "1.0".to_string());
        let mut cmd = Command::new(&venv_python);
        cmd.arg(gen_script)
           .arg("F:/Node/hivellm/expert/models/Qwen3-0.6B")  // model_path
           .arg(prompt)  // prompt
           .arg(max_tokens.to_string())  // max_tokens
           .arg(temperature.to_string())  // temperature
           .arg(top_p_str)  // top_p
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let output = cmd.output()
            .map_err(|e| crate::error::Error::Training(format!("Failed to execute Python: {}", e)))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let _stderr = String::from_utf8_lossy(&output.stderr);

            // Parse the output - the script prints to stdout
            // Look for the generated text after "Generated: "
            if let Some(line) = stdout.lines().find(|l| l.starts_with("Generated:")) {
                if let Some(text) = line.split(": ").nth(1) {
                    return Ok(text.trim_matches('"').to_string());
                }
            }

            // Fallback: return full stdout
            Ok(stdout.trim().to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(crate::error::Error::Training(format!("Generation failed: {}", stderr)))
        }
    }
}

impl Default for PythonTrainer {
    fn default() -> Self {
        Self::new().expect("Failed to initialize Python")
    }
}

