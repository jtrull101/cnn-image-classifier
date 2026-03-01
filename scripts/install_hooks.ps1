# install_hooks.ps1
# Install Git pre-commit hooks

$repoRoot = Split-Path -Parent $PSScriptRoot
$hooksDir = Join-Path $repoRoot ".git\hooks"

Write-Host ""
Write-Host "Installing Git pre-commit hooks..." -ForegroundColor Cyan
Write-Host ""

# Check if .git directory exists
if (-not (Test-Path "$repoRoot\.git")) {
    Write-Host "[ERROR] .git directory not found" -ForegroundColor Red
    Write-Host "This doesn't appear to be a Git repository" -ForegroundColor Yellow
    exit 1
}

# Create hooks directory if it doesn't exist
if (-not (Test-Path $hooksDir)) {
    New-Item -ItemType Directory -Path $hooksDir -Force | Out-Null
}

# Copy pre-commit hook (bash version)
$preCommitSrc = Join-Path $PSScriptRoot "pre-commit"
$preCommitDst = Join-Path $hooksDir "pre-commit"

if (Test-Path $preCommitSrc) {
    Copy-Item $preCommitSrc $preCommitDst -Force
    Write-Host "[OK] Installed pre-commit hook (bash)" -ForegroundColor Green
}

# Copy pre-commit hook (PowerShell version)
$preCommitPsSrc = Join-Path $PSScriptRoot "pre-commit.ps1"
$preCommitPsDst = Join-Path $hooksDir "pre-commit.ps1"

if (Test-Path $preCommitPsSrc) {
    Copy-Item $preCommitPsSrc $preCommitPsDst -Force
    Write-Host "[OK] Installed pre-commit.ps1 hook (PowerShell)" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Git hooks installed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "The pre-commit hook will now run automatically before each commit." -ForegroundColor Cyan
Write-Host ""
Write-Host "It will check:" -ForegroundColor Yellow
Write-Host "  1. Code formatting (ruff)" -ForegroundColor White
Write-Host "  2. Linting (ruff)" -ForegroundColor White
Write-Host "  3. Type checking (pyright)" -ForegroundColor White
Write-Host "  4. Tests (pytest)" -ForegroundColor White
Write-Host ""
Write-Host "To skip the hook (not recommended):" -ForegroundColor Yellow
Write-Host "  git commit --no-verify" -ForegroundColor DarkGray
Write-Host ""

