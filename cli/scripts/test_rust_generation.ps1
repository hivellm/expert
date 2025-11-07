# Test Rust generation vs Python generation
# This compares outputs to find where they diverge

$prompt = "Hello"
$max_tokens = 5

Write-Host "Testing Rust generation..." -ForegroundColor Yellow
.\target\release\expert-cli.exe chat --once --prompt $prompt --max-tokens $max_tokens

Write-Host "`nTesting Python generation..." -ForegroundColor Yellow
.\venv_windows\Scripts\python.exe scripts\test_generation.py "F:/Node/hivellm/expert/models/Qwen3-0.6B" $prompt $max_tokens

