# Script to check and install UV if needed
$ErrorActionPreference = "Stop"

try {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue

    if (-not $uvCmd) {
        $uvPath = Join-Path $env:USERPROFILE '.local\bin\uv.exe'

        if (-not (Test-Path $uvPath)) {
            Write-Host 'UV not found. Installing UV...' -ForegroundColor Yellow
            irm https://astral.sh/uv/install.ps1 | iex
            Write-Host 'UV installation complete. You may need to restart your terminal or add it to PATH.' -ForegroundColor Green

            # Try to refresh PATH in current session
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        } else {
            Write-Host "UV found at: $uvPath" -ForegroundColor Green
            Write-Host "Adding UV to PATH for this session..." -ForegroundColor Yellow
            $uvDir = Split-Path $uvPath
            if ($env:Path -notlike "*$uvDir*") {
                $env:Path += ";$uvDir"
            }
        }
    } else {
        Write-Host "UV is already installed: $(uv --version)" -ForegroundColor Green
    }

    exit 0
} catch {
    Write-Host "Error during UV installation: $_" -ForegroundColor Red
    exit 1
}

