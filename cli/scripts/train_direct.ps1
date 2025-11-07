# Train expert directly via Python (bypass Rust CLI)
param(
    [string]$Manifest = "manifest.json",
    [string]$Output = "weights",
    [string]$Device = "cuda"
)

# Activate venv
& "F:\Node\hivellm\expert\cli\venv_windows\Scripts\Activate.ps1"

# Run training
python "F:\Node\hivellm\expert\cli\scripts\train_direct.py" --manifest $Manifest --output $Output --device $Device

