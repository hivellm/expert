# Expert Training Script for PowerShell (with CUDA support)

$CliPath = "F:\Node\hivellm\expert\cli"
$VenvActivate = Join-Path $CliPath "venv\Scripts\Activate.ps1"
$ExpertCli = Join-Path $CliPath "target\release\expert-cli.exe"
$Manifest = "F:\Node\hivellm\expert\experts\expert-json-parser\manifest.json"
$Dataset = "F:\Node\hivellm\expert\experts\expert-json-parser\datasets\json_8k.jsonl"
$Output = "F:\Node\hivellm\expert\experts\expert-json-parser\weights"

Write-Host "Checking environment..." -ForegroundColor Cyan

if (-not (Test-Path $VenvActivate)) {
    Write-Host "Virtual environment not found at: $VenvActivate" -ForegroundColor Yellow
    Write-Host "Using WSL venv instead..." -ForegroundColor Yellow
    wsl -d Ubuntu-24.04 -- bash -l -c "cd /mnt/f/Node/hivellm/expert/cli && source venv/bin/activate && export PYTHONPATH=/mnt/f/Node/hivellm/expert/cli && ./target/release/expert-cli train --manifest ../experts/expert-json-parser/manifest.json --dataset ../experts/expert-json-parser/datasets/json_8k.jsonl --output ../experts/expert-json-parser/weights --epochs 1 --device auto"
    exit
}

Write-Host "Activating Python virtual environment..." -ForegroundColor Cyan
& $VenvActivate

Write-Host "Setting PYTHONPATH..." -ForegroundColor Cyan
$env:PYTHONPATH = $CliPath

Write-Host "`nStarting training with CUDA support..." -ForegroundColor Green
& $ExpertCli train `
    --manifest $Manifest `
    --dataset $Dataset `
    --output $Output `
    --epochs 1 `
    --device auto

Write-Host "`nTraining complete!" -ForegroundColor Green

