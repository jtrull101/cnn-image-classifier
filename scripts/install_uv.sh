#!/usr/bin/env bash
# Script to check and install UV if needed

set -e

if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "UV installation complete."

    # Try to add UV to PATH for current session
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    # Add common UV installation paths
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if [ -d "$HOME/.cargo/bin" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi

    echo "You may need to restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
else
    echo "UV is already installed: $(uv --version)"
fi
