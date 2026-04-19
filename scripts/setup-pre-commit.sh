#!/bin/bash

# Setup script for pre-commit hooks in libhmm.
# Run this once after cloning to activate the git hooks.
#
# Requires: Python 3.7+, pip
# Optional: clang-format (for formatting checks)
#
# Usage: bash scripts/setup-pre-commit.sh

set -e

# ─── helpers ─────────────────────────────────────────────────────────────────
info()    { echo "[INFO]  $1"; }
warn()    { echo "[WARN]  $1"; }
error()   { echo "[ERROR] $1"; }
step()    { echo ""; echo "──── $1"; }

# ─── check Python ────────────────────────────────────────────────────────────
step "Checking prerequisites"

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    error "Python is not installed. Install Python 3.7+ and try again."
    exit 1
fi
info "Python: $($PYTHON --version)"

# ─── install pre-commit ───────────────────────────────────────────────────────
step "Installing pre-commit"

if command -v pre-commit &>/dev/null; then
    info "pre-commit already installed: $(pre-commit --version)"
else
    $PYTHON -m pip install --user pre-commit
    if ! command -v pre-commit &>/dev/null; then
        warn "pre-commit installed but not in PATH."
        warn "Add ~/.local/bin to your PATH and re-run this script."
        exit 1
    fi
    info "Installed: $(pre-commit --version)"
fi

# ─── check optional tools ─────────────────────────────────────────────────────
step "Checking optional tools"

if command -v clang-format &>/dev/null; then
    info "clang-format: $(clang-format --version | head -1)"
else
    warn "clang-format not found. Install it for formatting support:"
    warn "  macOS:  brew install clang-format"
    warn "  Ubuntu: apt-get install clang-format"
fi

# ─── install git hooks ────────────────────────────────────────────────────────
step "Installing git hooks"

pre-commit install
info "Git hooks installed. They will run automatically on 'git commit'."

# ─── done ─────────────────────────────────────────────────────────────────────
step "Setup complete"
echo ""
echo "Useful commands:"
echo "  pre-commit run --all-files     # Run all hooks on all files"
echo "  pre-commit run --files FILE    # Run hooks on specific files"
echo "  pre-commit autoupdate          # Update hook versions"
echo "  git commit --no-verify         # Skip hooks (use sparingly)"
echo ""
info "See .pre-commit-config.yaml for the active hook set."
