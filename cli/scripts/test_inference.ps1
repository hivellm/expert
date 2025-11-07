# Test script for Qwen3 inference after fixing implementation
# Tests both Portuguese and English prompts

Write-Host "🧪 Testing Qwen3 Inference Fixes" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$expert_cli = ".\target\release\expert-cli.exe"

if (-not (Test-Path $expert_cli)) {
    Write-Host "❌ expert-cli not found. Building..." -ForegroundColor Red
    .\build-cuda.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed" -ForegroundColor Red
        exit 1
    }
}

# Test 1: Portuguese prompt (short)
Write-Host "📝 Test 1: Portuguese prompt" -ForegroundColor Yellow
Write-Host "Prompt: 'Olá, como você está?'" -ForegroundColor Gray
Write-Host ""
& $expert_cli chat --prompt "Olá, como você está?"
Write-Host ""
Write-Host "---" -ForegroundColor DarkGray
Write-Host ""

# Test 2: English prompt (simple question)
Write-Host "📝 Test 2: English prompt" -ForegroundColor Yellow
Write-Host "Prompt: 'What is the capital of Brazil?'" -ForegroundColor Gray
Write-Host ""
& $expert_cli chat --prompt "What is the capital of Brazil?"
Write-Host ""
Write-Host "---" -ForegroundColor DarkGray
Write-Host ""

# Test 3: Code-related prompt
Write-Host "📝 Test 3: Code-related prompt" -ForegroundColor Yellow
Write-Host "Prompt: 'Write a hello world in Python'" -ForegroundColor Gray
Write-Host ""
& $expert_cli chat --prompt "Write a hello world in Python"
Write-Host ""
Write-Host "---" -ForegroundColor DarkGray
Write-Host ""

Write-Host "✅ Tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Evaluation criteria:" -ForegroundColor Cyan
Write-Host "  ✅ Output is coherent (not 'vecunovecuno...' or gibberish)" -ForegroundColor White
Write-Host "  ✅ Contextually relevant to the prompt" -ForegroundColor White
Write-Host "  ✅ No repetition loops" -ForegroundColor White
Write-Host "  ✅ Proper Portuguese/English grammar" -ForegroundColor White

