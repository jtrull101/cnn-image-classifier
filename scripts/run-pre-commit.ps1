#!/usr/bin/env pwsh
# Run pre-commit checks
# This script replicates the behavior of the Git pre-commit hook

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Running pre-commit checks..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "" 

$ErrorCount = 0

# Format code
Write-Host "[1/4] Formatting code with ruff..." -ForegroundColor Yellow
uv run ruff format apps packages scripts tests
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Formatting failed" -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "[OK] Formatting complete" -ForegroundColor Green
}
Write-Host ""

# Lint code
Write-Host "[2/4] Linting code with ruff..." -ForegroundColor Yellow
uv run ruff check apps packages scripts tests
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Linting failed" -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "[OK] Linting passed" -ForegroundColor Green
}
Write-Host ""

# Type check
Write-Host "[3/4] Running type checks with pyright..." -ForegroundColor Yellow
uv run pyright
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Type checking failed" -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "[OK] Type checking passed" -ForegroundColor Green
}
Write-Host ""

# Run tests
Write-Host "[4/4] Running tests..." -ForegroundColor Yellow
uv run python -m pytest -c pyproject.toml --rootdir . -v -n auto --maxfail=3 --cov=packages --cov=apps --cov-report=term-missing
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Tests failed" -ForegroundColor Red
    $ErrorCount++
} else {
    Write-Host "[OK] Tests passed" -ForegroundColor Green
}
Write-Host ""

# Summary
Write-Host "=========================================" -ForegroundColor Cyan
if ($ErrorCount -eq 0) {
    Write-Host "[OK] All checks passed!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "[ERROR] $ErrorCount check(s) failed!" -ForegroundColor Red
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Please fix the issues above before committing." -ForegroundColor Yellow
    exit 1
}

