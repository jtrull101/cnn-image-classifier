#!/usr/bin/env pwsh
# Auto-fix pre-commit issues
# This script attempts to automatically fix formatting and linting issues

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Attempting to auto-fix issues..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Auto-fix format issues
Write-Host "[1/3] Auto-fixing format issues..." -ForegroundColor Yellow
npx nx run-many --target=format --all
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Formatting applied" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some formatting issues remain" -ForegroundColor Yellow
}
Write-Host ""

# Auto-fix linting issues
Write-Host "[2/3] Auto-fixing linting issues with --fix..." -ForegroundColor Yellow
uv run ruff check --fix .
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Linting issues fixed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some linting issues require manual fixing" -ForegroundColor Yellow
}
Write-Host ""

# Re-run all checks
Write-Host "[3/3] Re-running all checks..." -ForegroundColor Yellow
Write-Host ""
& "$PSScriptRoot\run-pre-commit.ps1"

