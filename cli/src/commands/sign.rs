use colored::Colorize;
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use std::collections::HashMap;

use sha2::{Sha256, Digest};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use tar::{Archive, Builder};

use crate::error::{Error, Result};
use crate::manifest::Manifest;

pub fn sign(
    expert_path: PathBuf,
    key_path: PathBuf,
) -> Result<()> {
    println!("{}", "✍️  Signing Expert Package".bright_cyan().bold());
    println!("{}", "═══════════════════════════════════════".bright_cyan());
    println!();

    // Verify expert package exists
    if !expert_path.exists() {
        return Err(Error::Packaging(format!(
            "Expert package not found: {}",
            expert_path.display()
        )));
    }

    // Load private key
    println!("  {} {}", "🔑".bright_blue(), "Loading signing key...".bright_white());
    let key_data = fs::read(&key_path).map_err(|e| {
        Error::Packaging(format!("Failed to read private key: {}", e))
    })?;

    if key_data.len() != 32 {
        return Err(Error::Packaging(
            "Invalid private key format (expected 32 bytes for Ed25519)".to_string()
        ));
    }

    let signing_key = SigningKey::from_bytes(&key_data.try_into().unwrap());
    let verifying_key: VerifyingKey = (&signing_key).into();
    
    println!("  {} Public key: {}", "→".bright_blue(), 
        hex::encode(verifying_key.to_bytes()).bright_white());
    println!();

    // Extract package to temporary directory
    println!("  {} {}", "📦".bright_blue(), "Extracting package...".bright_white());
    let temp_dir = std::env::temp_dir().join(format!("expert-sign-{}", 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()));
    
    fs::create_dir_all(&temp_dir)?;

    // Extract tar.gz
    let tar_gz = File::open(&expert_path)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);
    archive.unpack(&temp_dir)?;

    // Load manifest
    let manifest_path = temp_dir.join("manifest.json");
    let mut manifest = Manifest::load(&manifest_path)?;

    println!("  {} Expert: {}", "→".bright_blue(), manifest.name.bright_white());
    println!("  {} Version: {}", "→".bright_blue(), manifest.version.bright_white());
    println!("  {} Schema: {}", "→".bright_blue(), manifest.schema_version.bright_white());
    println!();

    // Calculate hashes of all files in package
    println!("  {} {}", "🔐".bright_blue(), "Computing file hashes...".bright_white());
    let mut file_hashes: HashMap<String, String> = HashMap::new();

    for entry in fs::read_dir(&temp_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
            let mut file = File::open(&path)?;
            let mut hasher = Sha256::new();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            hasher.update(&buffer);
            let hash = hasher.finalize();
            let hash_hex = format!("{:x}", hash);
            
            file_hashes.insert(file_name.clone(), hash_hex.clone());
            println!("    {} {} - sha256:{}", "✓".bright_green(), 
                file_name.bright_white(), 
                &hash_hex[..16].bright_yellow());
        }
    }

    // Also hash files in subdirectories (weights, soft_prompts, etc.)
    for entry in walkdir::WalkDir::new(&temp_dir).min_depth(1) {
        let entry = entry.map_err(|e| Error::Packaging(format!("Walk error: {}", e)))?;
        let path = entry.path();
        
        if path.is_file() {
            let relative_path = path.strip_prefix(&temp_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            
            if !file_hashes.contains_key(&relative_path) {
                let mut file = File::open(&path)?;
                let mut hasher = Sha256::new();
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                hasher.update(&buffer);
                let hash = hasher.finalize();
                let hash_hex = format!("{:x}", hash);
                
                file_hashes.insert(relative_path.clone(), hash_hex.clone());
                println!("    {} {} - sha256:{}", "✓".bright_green(), 
                    relative_path.bright_white(), 
                    &hash_hex[..16].bright_yellow());
            }
        }
    }

    println!();
    println!("  {} Files hashed: {}", "→".bright_blue(), file_hashes.len().to_string().bright_white());
    println!();

    // Create canonical message to sign
    let mut sorted_files: Vec<_> = file_hashes.iter().collect();
    sorted_files.sort_by_key(|(k, _)| *k);
    
    let message = sorted_files
        .iter()
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect::<Vec<_>>()
        .join("\n");

    // Sign the message
    println!("  {} {}", "✍️".bright_blue(), "Generating signature...".bright_white());
    let signature = signing_key.sign(message.as_bytes());
    let signature_hex = hex::encode(signature.to_bytes());

    println!("  {} Signature: {}", "✓".bright_green(), 
        &signature_hex[..32].bright_white());
    println!();

    // Update manifest with integrity information
    let timestamp = chrono::Utc::now().to_rfc3339();
    let pubkey_hex = hex::encode(verifying_key.to_bytes());

    manifest.integrity = Some(crate::manifest::Integrity {
        created_at: Some(timestamp),
        publisher: "expert-cli".to_string(),
        pubkey: Some(format!("ed25519:{}", pubkey_hex)),
        signature_algorithm: "ed25519".to_string(),
        signature: Some(signature_hex.clone()),
        files: file_hashes.clone(),
    });

    // Save updated manifest
    manifest.save(&manifest_path)?;
    println!("  {} Manifest updated with signature", "✓".bright_green());

    // Save signature to separate file
    let sig_path = temp_dir.join("signature.sig");
    fs::write(&sig_path, &signature_hex)?;
    println!("  {} signature.sig created", "✓".bright_green());
    println!();

    // Re-package with signature
    println!("  {} {}", "📦".bright_blue(), "Re-packaging with signature...".bright_white());
    
    // Create new tar.gz with signature
    let signed_output = expert_path.with_extension("signed.expert");
    let tar_gz = File::create(&signed_output)?;
    let enc = GzEncoder::new(tar_gz, Compression::default());
    let mut tar = Builder::new(enc);

    // Add all files from temp directory
    for entry in walkdir::WalkDir::new(&temp_dir).min_depth(1) {
        let entry = entry.map_err(|e| Error::Packaging(format!("Walk error: {}", e)))?;
        let path = entry.path();
        
        if path.is_file() {
            let relative_path = path.strip_prefix(&temp_dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            
            tar.append_path_with_name(&path, &relative_path)?;
        }
    }

    tar.finish()?;

    // Cleanup temp directory
    fs::remove_dir_all(&temp_dir)?;

    // Replace original with signed version
    fs::rename(&signed_output, &expert_path)?;

    println!();
    println!("{}", "✅ Package signed successfully!".bright_green().bold());
    println!("  {} File: {}", "→".bright_blue(), expert_path.display().to_string().bright_white());
    println!("  {} Algorithm: Ed25519", "→".bright_blue());
    println!("  {} Signature: {}...", "→".bright_blue(), &signature_hex[..32].bright_white());
    println!("  {} Public Key: {}...", "→".bright_blue(), &pubkey_hex[..32].bright_white());

    Ok(())
}

pub fn keygen(
    output_path: PathBuf,
    name: Option<String>,
) -> Result<()> {
    println!("{}", "🔑 Generating Signing Key".bright_cyan().bold());
    println!("{}", "═══════════════════════════════════════".bright_cyan());
    println!();

    // Generate new Ed25519 key pair
    let mut csprng = rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key: VerifyingKey = (&signing_key).into();

    // Save private key
    fs::write(&output_path, signing_key.to_bytes())?;
    println!("  {} Private key saved: {}", "✓".bright_green(), 
        output_path.display().to_string().bright_white());

    // Save public key
    let pubkey_path = output_path.with_extension("pub");
    let pubkey_hex = hex::encode(verifying_key.to_bytes());
    fs::write(&pubkey_path, format!("ed25519:{}", pubkey_hex))?;
    println!("  {} Public key saved: {}", "✓".bright_green(), 
        pubkey_path.display().to_string().bright_white());

    println!();
    println!("{}", "Key Information:".bright_blue().bold());
    println!("  {} Algorithm: Ed25519", "→".bright_blue());
    println!("  {} Public Key: {}", "→".bright_blue(), pubkey_hex.bright_white());
    
    if let Some(ref owner_name) = name {
        println!("  {} Owner: {}", "→".bright_blue(), owner_name.bright_white());
    }

    println!();
    println!("{}", "⚠️  IMPORTANT:".bright_yellow().bold());
    println!("  {} Keep the private key (.pem) SECRET", "→".bright_yellow());
    println!("  {} Share the public key (.pub) with users", "→".bright_yellow());
    println!("  {} Backup the private key securely", "→".bright_yellow());

    Ok(())
}

