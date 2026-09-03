#!/bin/bash
# One-command development launch for VapourBox on macOS.
#
# Checks the toolchain (Xcode Command Line Tools, Rust, Flutter) and the
# VapourSynth dependency bundle, installing or explaining what is missing, then
# hands over to run-debug-macos.sh to build and launch the debug app.
#
# Usage: ./Scripts/dev-macos.sh [--skip-worker] [--skip-app] [--run-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: this script is macOS only. On Linux use ./Scripts/run-debug-linux.sh" >&2
  exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  PLATFORM_DIR="macos-arm64"
else
  PLATFORM_DIR="macos-x64"
fi

# Homebrew is how the missing tools get installed; without it, report and stop.
if ! command -v brew &> /dev/null; then
  for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    [ -x "$candidate" ] && eval "$($candidate shellenv)" && break
  done
fi

require_brew() {
  if ! command -v brew &> /dev/null; then
    echo "ERROR: $1 is missing and Homebrew is not installed, so it cannot be installed automatically." >&2
    echo "       Install Homebrew from https://brew.sh and re-run this script." >&2
    exit 1
  fi
}

echo "==> Checking toolchain..."

# Xcode Command Line Tools: needed by both cargo and the Flutter macOS build.
if ! xcode-select -p &> /dev/null; then
  echo "    Xcode Command Line Tools missing — opening the installer."
  echo "    Complete it, then re-run this script."
  xcode-select --install || true
  exit 1
fi
echo "    Xcode Command Line Tools: $(xcode-select -p)"

# Rust (the worker crate).
if ! command -v cargo &> /dev/null; then
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
fi
if ! command -v cargo &> /dev/null; then
  echo "    Rust missing — installing via rustup."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
  . "$HOME/.cargo/env"
fi
echo "    Rust: $(cargo --version)"

# Flutter (the app). CocoaPods too — the macOS build resolves pods.
if ! command -v flutter &> /dev/null; then
  require_brew Flutter
  echo "    Flutter missing — installing via Homebrew (this takes a few minutes)."
  brew install --cask flutter
fi
echo "    Flutter: $(flutter --version 2>/dev/null | head -1)"

if ! command -v pod &> /dev/null; then
  require_brew CocoaPods
  echo "    CocoaPods missing — installing via Homebrew."
  brew install cocoapods
fi
echo "    CocoaPods: $(pod --version)"

# Dependency bundle. The debug worker searches upward from the executable for
# deps/, so a symlink at the repo root pointing at an existing install is enough
# and avoids downloading a second copy.
DEPS_DIR="$PROJECT_ROOT/deps/$PLATFORM_DIR"
INSTALLED_DEPS="$HOME/Library/Application Support/VapourBox/deps"

if [ ! -d "$DEPS_DIR" ]; then
  if [ -d "$INSTALLED_DEPS/$PLATFORM_DIR" ]; then
    echo "==> Linking deps/ to the installed bundle in Application Support..."
    ln -sfn "$INSTALLED_DEPS" "$PROJECT_ROOT/deps"
  else
    echo "==> No dependency bundle found — building it (one-off, takes a while)..."
    "$SCRIPT_DIR/download-deps-macos.sh"
  fi
fi
echo "    Deps: $DEPS_DIR"

exec "$SCRIPT_DIR/run-debug-macos.sh" "$@"
