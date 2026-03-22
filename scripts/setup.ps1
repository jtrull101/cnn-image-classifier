# setup.ps1
# Initial setup script for the Python/UV monorepo

# Change to repository root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptPath
Set-Location $repoRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "alz-mri-cnn Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

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

# Ensure UV is available
Write-Host "Checking for UV installation..." -ForegroundColor Yellow
$uvCommand = Get-UvCommand

if (-not $uvCommand) {
    Write-Host "[WARN] UV is not installed" -ForegroundColor Yellow
    Write-Host "Installing UV..." -ForegroundColor Cyan

    try {
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] UV installed successfully" -ForegroundColor Green

            # Refresh PATH in current session
            $uvCommand = Get-UvCommand
            if (-not $uvCommand) {
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
} else {
    $uvVersion = & $uvCommand --version
    Write-Host "[OK] UV is installed: $uvVersion" -ForegroundColor Green
}

# Check Python version
Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ from: https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Sync UV workspace dependencies
Write-Host ""
Write-Host "Installing Python workspace dependencies..." -ForegroundColor Yellow
& $uvCommand sync --dev

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Run initial checks
Write-Host ""
Write-Host "Running initial code quality checks..." -ForegroundColor Yellow

Write-Host "  [1/6] Installing Git hooks..." -ForegroundColor Cyan
& "$PSScriptRoot\install_hooks.ps1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Git hooks installed" -ForegroundColor Green
}

Write-Host "  [2/6] Formatting code..." -ForegroundColor Cyan
& $uvCommand run ruff format apps packages scripts tests
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Formatting complete" -ForegroundColor Green
}

Write-Host "  [3/6] Linting code..." -ForegroundColor Cyan
& $uvCommand run ruff check apps packages scripts tests
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Linting complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Some linting issues found" -ForegroundColor Yellow
}

Write-Host "  [4/6] Type checking..." -ForegroundColor Cyan
& $uvCommand run pyright
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Type checking complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Some type issues found" -ForegroundColor Yellow
}

Write-Host "  [5/6] Running tests..." -ForegroundColor Cyan
& $uvCommand run python -m pytest -c pyproject.toml --rootdir . -v -n auto --maxfail=3 --cov=packages --cov=apps --cov-report=term-missing
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Tests complete" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Some tests failed" -ForegroundColor Yellow
}

Write-Host "  [6/6] Building packages..." -ForegroundColor Cyan
foreach ($pkg in @("packages/config","packages/utils","packages/data","packages/models","packages/training","packages/cli","apps/api")) {
    Write-Host "    -> Building $pkg" -ForegroundColor White
    Push-Location $pkg
    & $uvCommand build
    $buildExit = $LASTEXITCODE
    Pop-Location

    if ($buildExit -ne 0) {
        Write-Host "    [WARN] Build failed for $pkg" -ForegroundColor Yellow
    }
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Workspace layout:" -ForegroundColor Cyan
Write-Host "  apps/api/              - FastAPI application" -ForegroundColor White
Write-Host "  packages/config/       - Configuration (Pydantic)" -ForegroundColor White
Write-Host "  packages/utils/        - Common utilities" -ForegroundColor White
Write-Host "  packages/data/         - Data loading" -ForegroundColor White
Write-Host "  packages/models/       - Neural networks" -ForegroundColor White
Write-Host "  packages/training/     - Training pipeline" -ForegroundColor White
Write-Host "  packages/cli/          - CLI interface" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review README.md for usage and documentation" -ForegroundColor White
Write-Host "  2. Run 'make help' to see available commands" -ForegroundColor White
Write-Host "  3. Run 'make serve-api' to start the FastAPI server" -ForegroundColor White
Write-Host "  4. Run 'make pre-commit' before committing changes" -ForegroundColor White
Write-Host ""
Write-Host "Git hooks installed:" -ForegroundColor Cyan
Write-Host "  Pre-commit checks will run automatically before each commit" -ForegroundColor White
Write-Host "  To skip: git commit --no-verify (not recommended)" -ForegroundColor DarkGray
Write-Host ""
