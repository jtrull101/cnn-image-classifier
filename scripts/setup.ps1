# setup.ps1
# Initial setup script

# Change to repository root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptPath
Set-Location $repoRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "alz-mri-cnn Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if UV is installed
Write-Host ""
Write-Host "Checking for UV installation..." -ForegroundColor Yellow

# Function to get UV command
function Get-UvCommand {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        return "uv"
    }

    $uvPath = "$env:USERPROFILE\.local\bin\uv.exe"
    if (Test-Path $uvPath) {
        return $uvPath
    }

    return $null
}

$uvCommand = Get-UvCommand

if ($uvCommand) {
    $uvVersion = & $uvCommand --version
    Write-Host "[OK] UV is installed: $uvVersion" -ForegroundColor Green
} else {
    Write-Host "[WARN] UV is not installed" -ForegroundColor Yellow
    Write-Host "Installing UV..." -ForegroundColor Cyan

    try {
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] UV installed successfully" -ForegroundColor Green

            # Check if installation succeeded
            $uvCommand = Get-UvCommand
            if ($uvCommand) {
                Write-Host "[OK] UV is now available" -ForegroundColor Green
            } else {
                Write-Host "[ERROR] UV installation completed but command not found" -ForegroundColor Red
                Write-Host "Please restart your terminal and run this script again" -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host "[ERROR] UV installation failed" -ForegroundColor Red
            Write-Host "Please install UV manually from: https://github.com/astral-sh/uv" -ForegroundColor Yellow
            exit 1
        }
    }
    catch {
        Write-Host "[ERROR] Error installing UV: $_" -ForegroundColor Red
        Write-Host "Please install UV manually from: https://github.com/astral-sh/uv" -ForegroundColor Yellow
        exit 1
    }
}

# Check Python version
Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from: https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Check Node.js (for NX)
Write-Host ""
Write-Host "Checking Node.js installation..." -ForegroundColor Yellow
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue

if ($nodeCmd) {
    $nodeVersion = node --version
    Write-Host "[OK] Node.js $nodeVersion installed" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Node.js not found" -ForegroundColor Red
    Write-Host "Please install Node.js 18+ from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Install NX and Node dependencies
Write-Host ""
Write-Host "Installing NX and Node dependencies..." -ForegroundColor Yellow
npm install

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] NX installed successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to install NX" -ForegroundColor Red
    exit 1
}

# Sync UV workspace dependencies
Write-Host ""
Write-Host "Installing Python workspace dependencies..." -ForegroundColor Yellow
& $uvCommand sync

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Run initial checks
Write-Host ""
Write-Host "Running initial code quality checks..." -ForegroundColor Yellow

Write-Host "  [1/5] Installing Git hooks..." -ForegroundColor Cyan
& "$PSScriptRoot\install_hooks.ps1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Git hooks installed" -ForegroundColor Green
}

Write-Host "  [2/5] Formatting code..." -ForegroundColor Cyan
npx nx run-many --target=format --all
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Formatting complete" -ForegroundColor Green
}

Write-Host "  [3/5] Linting code..." -ForegroundColor Cyan
npx nx run-many --target=lint --all
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Linting complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Some linting issues found" -ForegroundColor Yellow
}

Write-Host "  [4/5] Type checking..." -ForegroundColor Cyan
& $uvCommand run pyright
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Type checking complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Some type issues found" -ForegroundColor Yellow
}

Write-Host "  [5/5] Building packages..." -ForegroundColor Cyan
npx nx run-many --target=build --all
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Build complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Some packages failed to build" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your monorepo structure:" -ForegroundColor Cyan
Write-Host "  apps/api/              - Flask REST API" -ForegroundColor White
Write-Host "  packages/config/       - Configuration (Pydantic)" -ForegroundColor White
Write-Host "  packages/utils/        - Common utilities" -ForegroundColor White
Write-Host "  packages/data/         - Data loading" -ForegroundColor White
Write-Host "  packages/models/       - Neural networks" -ForegroundColor White
Write-Host "  packages/training/     - Training pipeline" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review README.md for usage and documentation" -ForegroundColor White
Write-Host "  2. Run 'make help' to see available commands" -ForegroundColor White
Write-Host "  3. Run 'make graph' to view dependency graph" -ForegroundColor White
Write-Host "  4. Run 'make serve-api' to start the Flask API" -ForegroundColor White
Write-Host ""
Write-Host "Common commands:" -ForegroundColor Cyan
Write-Host "  make build       - Build all packages" -ForegroundColor White
Write-Host "  make test        - Run all tests" -ForegroundColor White
Write-Host "  make lint        - Lint all code" -ForegroundColor White
Write-Host "  make format      - Format all code" -ForegroundColor White
Write-Host "  make serve-api   - Start Flask API" -ForegroundColor White
Write-Host "  make graph       - View dependency graph" -ForegroundColor White
Write-Host ""
Write-Host "Git hooks installed:" -ForegroundColor Cyan
Write-Host "  Pre-commit checks will run automatically before each commit" -ForegroundColor White
Write-Host "  To skip: git commit --no-verify (not recommended)" -ForegroundColor DarkGray
Write-Host ""

