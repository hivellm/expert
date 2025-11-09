# Run all Rust tests organized by category

Write-Host "Running Rust Tests" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host ""

$testCategories = @(
    @{ Name = "Commands"; Path = "commands"; Tests = @("chat_test", "dataset_command_tests", "install_tests", "list_command_tests", "package_command_tests", "sign_command_tests", "train_command_tests", "update_command_tests") },
    @{ Name = "Inference"; Path = "inference"; Tests = @("grammar_validator_test", "test_generation", "test_hot_swap", "test_lora", "test_qwen") },
    @{ Name = "Core"; Path = "core"; Tests = @("manifest_tests", "manifest_feature_tests", "model_detection_tests", "registry_tests", "router_tests", "test_keyword_routing", "test_router") },
    @{ Name = "Integration"; Path = "integration"; Tests = @("dependency_resolution_tests", "error_message_tests", "package_integration_tests", "test_integration", "test_multi_expert", "validation_integration_tests") },
    @{ Name = "Benchmarks"; Path = "benchmarks"; Tests = @("test_latency_benchmarks", "test_vram_profiling") }
)

$totalPassed = 0
$totalFailed = 0

foreach ($category in $testCategories) {
    Write-Host "`n[$($category.Name)]" -ForegroundColor Yellow
    
    foreach ($test in $category.Tests) {
        $testFile = "$($category.Path)\$test.rs"
        if (Test-Path $testFile) {
            Write-Host "  Running: $test..." -ForegroundColor Gray -NoNewline
            $result = cargo test --test $test 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host " PASSED" -ForegroundColor Green
                $totalPassed++
            } else {
                Write-Host " FAILED" -ForegroundColor Red
                $totalFailed++
            }
        } else {
            Write-Host "  Skipping: $test (file not found)" -ForegroundColor DarkGray
        }
    }
}

Write-Host "`n==================" -ForegroundColor Cyan
Write-Host "Summary: $totalPassed passed, $totalFailed failed" -ForegroundColor $(if ($totalFailed -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($totalFailed -gt 0) {
    exit 1
}
