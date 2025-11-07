#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Clean up old Unsloth compiled caches from individual expert directories.

.DESCRIPTION
    This script removes the deprecated unsloth_compiled_cache directories
    from expert directories. As of now, all Unsloth compiled caches are
    centralized in expert/cache/unsloth_compiled/ to avoid duplication.

.NOTES
    Author: HiveLLM Expert System
    Date: 2025-11-06
    
    This cache is safe to delete - it will be regenerated automatically
    by Unsloth during the next training run.
#>

param(
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$expertRoot = Split-Path -Parent $scriptDir
$expertsDir = Join-Path $expertRoot "experts"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Unsloth Cache Cleanup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN MODE] No files will be deleted" -ForegroundColor Yellow
    Write-Host ""
}

# Find all unsloth_compiled_cache directories
$cacheDirectories = Get-ChildItem -Path $expertsDir -Recurse -Directory -Filter "unsloth_compiled_cache" -ErrorAction SilentlyContinue

if ($cacheDirectories.Count -eq 0) {
    Write-Host "[OK] No old cache directories found" -ForegroundColor Green
    Write-Host ""
    exit 0
}

Write-Host "Found $($cacheDirectories.Count) old cache director(ies):" -ForegroundColor Yellow
Write-Host ""

$totalSize = 0

foreach ($cacheDir in $cacheDirectories) {
    $relativePath = $cacheDir.FullName.Replace($expertRoot, "").TrimStart('\', '/')
    
    # Calculate size
    $size = (Get-ChildItem -Path $cacheDir.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 2)
    $totalSize += $size
    
    Write-Host "  - $relativePath" -ForegroundColor White
    Write-Host "    Size: $sizeMB MB" -ForegroundColor Gray
    
    if ($Verbose) {
        # Show file count
        $fileCount = (Get-ChildItem -Path $cacheDir.FullName -Recurse -File).Count
        Write-Host "    Files: $fileCount" -ForegroundColor Gray
    }
    
    if (-not $DryRun) {
        try {
            Remove-Item -Path $cacheDir.FullName -Recurse -Force
            Write-Host "    [DELETED]" -ForegroundColor Red
        }
        catch {
            Write-Host "    [ERROR] Failed to delete: $_" -ForegroundColor Red
        }
    }
    else {
        Write-Host "    [WILL BE DELETED]" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Total size: $totalSizeMB MB" -ForegroundColor White

if ($DryRun) {
    Write-Host "  Status: Dry run - no files deleted" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run without -DryRun to actually delete the caches" -ForegroundColor Yellow
}
else {
    Write-Host "  Status: Cleanup complete" -ForegroundColor Green
    Write-Host ""
    Write-Host "All Unsloth compiled caches are now centralized in:" -ForegroundColor Green
    Write-Host "  expert/cache/unsloth_compiled/" -ForegroundColor White
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""


