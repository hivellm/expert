use crate::error::Error;
use crate::manifest::Manifest;
use crate::registry::{AdapterEntry, ExpertRegistry, ExpertVersionEntry};
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Install expert from Git repository or local path
pub fn install(source: &str, dev_mode: bool) -> Result<(), Error> {
    install_with_deps(source, dev_mode, true, 0)
}

fn install_with_deps(
    source: &str,
    dev_mode: bool,
    check_deps: bool,
    depth: usize,
) -> Result<(), Error> {
    if depth > 10 {
        return Err(Error::Installation(
            "Dependency depth exceeded 10 (circular dependency?)".to_string(),
        ));
    }

    let indent = "  ".repeat(depth);
    println!("{}Installing expert from: {}", indent, source);

    let mut registry = ExpertRegistry::load().unwrap_or_else(|_| {
        if depth == 0 {
            println!("{}Creating new registry...", indent);
        }
        ExpertRegistry::default()
    });

    // Parse source
    let install_source = InstallSource::parse(source)?;

    // Get expert files
    let temp_dir = install_source.fetch()?;

    // Read manifest
    let manifest_path = temp_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(Error::Manifest(
            "manifest.json not found in source".to_string(),
        ));
    }

    let manifest = Manifest::load(&manifest_path)?;

    // Check if same version already installed
    if registry.has_expert_version(&manifest.name, &manifest.version) && !dev_mode {
        println!(
            "{}[WARN] Expert '{}' v{} already installed. Replacing with new files...",
            indent, manifest.name, manifest.version
        );
    }

    // Verify signature if package is signed
    verify_signature(&temp_dir, &manifest)?;

    // Check base model compatibility
    check_base_model_compatibility(&manifest, &registry)?;

    // Resolve and install dependencies first
    if check_deps {
        resolve_and_install_dependencies(&manifest, &registry, dev_mode, depth)?;
    }

    // Install expert
    let install_path = if dev_mode {
        // Dev mode: use local path directly (no copy). Supports editing.
        install_source.path()
    } else {
        // Copy to install directory with versioned layout
        copy_expert(&temp_dir, &manifest, &mut registry)?
    };

    // Create registry entry
    let expert_entry = create_expert_entry(&manifest, &install_path, source)?;

    // Update registry
    registry.add_expert_version(&manifest.name, expert_entry);
    registry.save()?;

    println!(
        "{}[OK] Expert '{}' installed successfully!",
        indent, manifest.name
    );
    println!("{}     Location: {}", indent, install_path.display());

    // Cleanup temp directory if not local
    if !matches!(install_source, InstallSource::Local(_)) {
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    Ok(())
}

/// Install source type
enum InstallSource {
    Git {
        url: String,
        ref_spec: Option<String>,
    },
    Local(PathBuf),
    Package(PathBuf),
}

impl InstallSource {
    fn parse(source: &str) -> Result<Self, Error> {
        if source.starts_with("git+") {
            // Git URL: git+https://github.com/user/repo.git or git+https://...#tag
            let url = source.strip_prefix("git+").unwrap();

            if let Some(pos) = url.find('#') {
                let (base_url, ref_spec) = url.split_at(pos);
                let ref_spec = ref_spec.strip_prefix('#').unwrap();
                Ok(InstallSource::Git {
                    url: base_url.to_string(),
                    ref_spec: Some(ref_spec.to_string()),
                })
            } else {
                Ok(InstallSource::Git {
                    url: url.to_string(),
                    ref_spec: None,
                })
            }
        } else if source.starts_with("file://") {
            // Local path: file://./path/to/expert
            let path = source.strip_prefix("file://").unwrap();
            Ok(InstallSource::Local(PathBuf::from(path)))
        } else if source.ends_with(".expert") {
            // Expert package file
            Ok(InstallSource::Package(PathBuf::from(source)))
        } else {
            // Assume local path
            Ok(InstallSource::Local(PathBuf::from(source)))
        }
    }

    fn fetch(&self) -> Result<PathBuf, Error> {
        match self {
            InstallSource::Git { url, ref_spec } => clone_git_repo(url, ref_spec.as_deref()),
            InstallSource::Local(path) => {
                if !path.exists() {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Path does not exist: {}", path.display()),
                    )));
                }
                Ok(path.clone())
            }
            InstallSource::Package(path) => extract_expert_package(path),
        }
    }

    fn path(&self) -> PathBuf {
        match self {
            InstallSource::Local(path) => path.clone(),
            _ => PathBuf::new(),
        }
    }
}

/// Clone Git repository
fn clone_git_repo(url: &str, ref_spec: Option<&str>) -> Result<PathBuf, Error> {
    let temp_dir = std::env::temp_dir().join(format!("expert-{}", uuid::Uuid::new_v4()));

    println!("Cloning repository...");

    let mut cmd = Command::new("git");
    cmd.arg("clone").arg("--depth").arg("1");

    if let Some(ref_spec) = ref_spec {
        cmd.arg("--branch").arg(ref_spec);
    }

    cmd.arg(url).arg(&temp_dir);

    let output = cmd
        .output()
        .map_err(|e| Error::Installation(format!("Failed to run git: {}", e)))?;

    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Installation(format!("Git clone failed: {}", error)));
    }

    println!("[OK] Repository cloned");

    Ok(temp_dir)
}

/// Extract .expert package
fn extract_expert_package(package_path: &Path) -> Result<PathBuf, Error> {
    use flate2::read::GzDecoder;
    use std::fs::File;
    use tar::Archive;

    if !package_path.exists() {
        return Err(Error::Installation(format!(
            "Package not found: {}",
            package_path.display()
        )));
    }

    let temp_dir = std::env::temp_dir().join(format!("expert-pkg-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir).map_err(|e| Error::Io(e))?;

    println!("Extracting package...");

    // Open .expert file (tar.gz format)
    let file = File::open(package_path)
        .map_err(|e| Error::Installation(format!("Failed to open package: {}", e)))?;

    let gz = GzDecoder::new(file);
    let mut archive = Archive::new(gz);

    // Extract all files
    archive
        .unpack(&temp_dir)
        .map_err(|e| Error::Installation(format!("Failed to extract package: {}", e)))?;

    println!("[OK] Package extracted");

    Ok(temp_dir)
}

/// Check if required base model is available
fn check_base_model_compatibility(
    manifest: &Manifest,
    registry: &ExpertRegistry,
) -> Result<(), Error> {
    let required_models = manifest.get_base_models();

    if required_models.is_empty() {
        return Err(Error::Manifest(
            "No base models specified in manifest".to_string(),
        ));
    }

    for model_name in &required_models {
        if !registry.has_base_model(model_name) {
            println!(
                "[WARN] Required base model '{}' not found in registry",
                model_name
            );
            println!("       You may need to install it manually");
            println!("       Hint: expert-cli install-model {}", model_name);
        }
    }

    Ok(())
}

/// Copy expert to install directory
fn copy_expert(
    source: &Path,
    manifest: &Manifest,
    registry: &mut ExpertRegistry,
) -> Result<PathBuf, Error> {
    // Ensure install root exists
    if !registry.install_dir.exists() {
        fs::create_dir_all(&registry.install_dir).map_err(|e| Error::Io(e))?;
    }

    let base_dir = registry.install_dir.join(&manifest.name);
    if !base_dir.exists() {
        fs::create_dir_all(&base_dir).map_err(|e| Error::Io(e))?;
    }

    let install_path = base_dir.join(&manifest.version);

    if install_path.exists() {
        println!("[WARN] Expert version already exists, removing old files...");
        std::fs::remove_dir_all(&install_path).map_err(|e| Error::Io(e))?;
    }

    println!("Copying expert files...");

    copy_dir_all(source, &install_path)?;

    // Track latest pointer
    let current_marker = base_dir.join("current.txt");
    std::fs::write(&current_marker, &manifest.version).map_err(|e| Error::Io(e))?;

    println!("[OK] Files copied");

    Ok(install_path)
}

/// Recursively copy directory
fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(dst).map_err(|e| Error::Io(e))?;

    for entry in std::fs::read_dir(src).map_err(|e| Error::Io(e))? {
        let entry = entry.map_err(|e| Error::Io(e))?;
        let path = entry.path();
        let file_name = entry.file_name();

        // Skip .git directory
        if file_name == ".git" {
            continue;
        }

        let dst_path = dst.join(&file_name);

        if path.is_dir() {
            copy_dir_all(&path, &dst_path)?;
        } else {
            std::fs::copy(&path, &dst_path).map_err(|e| Error::Io(e))?;
        }
    }

    Ok(())
}

/// Verify package signature if present
fn verify_signature(expert_dir: &Path, manifest: &Manifest) -> Result<(), Error> {
    let signature_path = expert_dir.join("signature.sig");

    // If no signature file, skip verification
    if !signature_path.exists() {
        println!("  [INFO] No signature found, skipping verification");
        return Ok(());
    }

    println!("  [>] Verifying signature...");

    // Check if manifest has integrity section
    if manifest.integrity.is_none() {
        println!("  [WARN] signature.sig found but no integrity section in manifest");
        return Ok(());
    }

    let integrity = manifest.integrity.as_ref().unwrap();

    // Read signature file
    let signature_hex = std::fs::read_to_string(&signature_path)
        .map_err(|e| Error::Signing(format!("Failed to read signature: {}", e)))?;

    let signature_bytes = hex::decode(signature_hex.trim())
        .map_err(|e| Error::Signing(format!("Invalid signature format: {}", e)))?;

    // Read public key from integrity section
    let pubkey_hex = integrity
        .pubkey
        .as_ref()
        .ok_or_else(|| Error::Signing("No public key in integrity section".to_string()))?;

    let pubkey_bytes = hex::decode(pubkey_hex)
        .map_err(|e| Error::Signing(format!("Invalid public key format: {}", e)))?;

    // Verify signature
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let verifying_key = VerifyingKey::from_bytes(
        pubkey_bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::Signing("Invalid public key length".to_string()))?,
    )
    .map_err(|e| Error::Signing(format!("Invalid public key: {}", e)))?;

    let signature = Signature::from_bytes(
        signature_bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::Signing("Invalid signature length".to_string()))?,
    );

    // Reconstruct canonical message from file hashes
    let mut file_hashes = Vec::new();

    // Hash manifest
    let manifest_content = std::fs::read_to_string(expert_dir.join("manifest.json"))?;
    file_hashes.push((
        "manifest.json".to_string(),
        compute_sha256_string(&manifest_content),
    ));

    // Hash other files (TODO: scan directory for all files)
    file_hashes.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_message: String = file_hashes
        .iter()
        .map(|(file, hash)| format!("{}:{}", file, hash))
        .collect::<Vec<_>>()
        .join("\n");

    // Verify
    if verifying_key
        .verify(canonical_message.as_bytes(), &signature)
        .is_ok()
    {
        println!("  [OK] Signature verified!");
        let key_preview = if pubkey_hex.len() > 32 {
            format!(
                "{}...{}",
                &pubkey_hex[..16],
                &pubkey_hex[pubkey_hex.len() - 16..]
            )
        } else {
            pubkey_hex.clone()
        };
        println!("       Signed by: {}", key_preview);
    } else {
        return Err(Error::Signing(
            "Signature verification failed! Package may be tampered.".to_string(),
        ));
    }

    Ok(())
}

/// Compute SHA256 of string content
fn compute_sha256_string(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Resolve and install dependencies
fn resolve_and_install_dependencies(
    manifest: &Manifest,
    _registry: &ExpertRegistry,
    dev_mode: bool,
    depth: usize,
) -> Result<(), Error> {
    // Check if manifest has dependencies
    let dependencies = manifest.constraints.requires.clone();

    if dependencies.is_empty() {
        return Ok(());
    }

    println!("  [>] Found {} dependencies", dependencies.len());

    for dep in dependencies {
        println!("  [>] Checking dependency: {}", dep);

        // Parse dependency (format: "expert-name@version" or "expert-name@>=version")
        let parts: Vec<&str> = dep.split('@').collect();
        let dep_name = parts[0];
        let dep_version = parts.get(1).copied();

        // Check if already installed
        let current_registry = ExpertRegistry::load().unwrap_or_else(|_| ExpertRegistry::default());
        if let Some(installed) = current_registry.get_expert(dep_name) {
            if let Some(_version_req) = dep_version {
                // TODO: Implement semver checking
                println!(
                    "  [OK] Dependency '{}' already installed (v{})",
                    dep_name, installed.version
                );
            } else {
                println!("  [OK] Dependency '{}' already installed", dep_name);
            }
            continue;
        }

        // Dependency not installed, try to install
        println!("  [>] Installing dependency: {}", dep_name);

        // Construct Git URL (assume GitHub hivellm org)
        let dep_source = format!("git+https://github.com/hivellm/{}.git", dep_name);

        // Recursive install
        install_with_deps(&dep_source, dev_mode, false, depth + 1)?;
    }

    Ok(())
}

/// Create expert registry entry from manifest
fn create_expert_entry(
    manifest: &Manifest,
    install_path: &Path,
    source: &str,
) -> Result<ExpertVersionEntry, Error> {
    let base_model = manifest
        .get_base_models()
        .first()
        .ok_or_else(|| Error::Manifest("No base models in manifest".to_string()))?
        .clone();

    let resolved_path = install_path
        .canonicalize()
        .unwrap_or_else(|_| install_path.to_path_buf());

    // Collect adapters (path is automatically resolved to expert root)
    let adapters = if let Some(ref adapters) = manifest.adapters {
        adapters
            .iter()
            .map(|a| AdapterEntry {
                adapter_type: a.adapter_type.clone(),
                path: resolved_path.clone(), // Adapters are in expert root
                size_bytes: a.size_bytes.unwrap_or(0),
                sha256: a.sha256.clone(),
            })
            .collect()
    } else if let Some(ref models) = manifest.base_models {
        models
            .first()
            .map(|m| {
                m.adapters
                    .iter()
                    .map(|a| AdapterEntry {
                        adapter_type: a.adapter_type.clone(),
                        path: resolved_path.clone(), // Adapters are in expert root
                        size_bytes: a.size_bytes.unwrap_or(0),
                        sha256: a.sha256.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    // Collect capabilities
    let capabilities = manifest.capabilities.clone();

    // Collect dependencies
    let dependencies: Vec<_> = manifest
        .constraints
        .requires
        .iter()
        .map(|dep: &String| {
            let parts: Vec<&str> = dep.split('@').collect();
            crate::registry::DependencyEntry {
                name: parts[0].to_string(),
                version: parts.get(1).unwrap_or(&"*").to_string(),
                optional: false,
            }
        })
        .collect();

    Ok(ExpertVersionEntry {
        version: manifest.version.clone(),
        base_model,
        path: resolved_path,
        source: source.to_string(),
        installed_at: Utc::now(),
        adapters,
        capabilities,
        dependencies,
    })
}
