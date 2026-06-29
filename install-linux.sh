#!/usr/bin/env bash
set -euo pipefail

TEXT_MODEL="${TEXT_MODEL:-qwen2.5:7b}"
VISION_MODEL="${VISION_MODEL:-qwen2.5vl:7b}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log() {
  printf '\n==> %s\n' "$1"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_system_packages() {
  log "Installing system packages"

  if has_cmd apt-get; then
    sudo apt-get update
    sudo apt-get install -y curl ca-certificates git make poppler-utils
  elif has_cmd dnf; then
    sudo dnf install -y curl ca-certificates git make poppler-utils
  elif has_cmd yum; then
    sudo yum install -y curl ca-certificates git make poppler-utils
  elif has_cmd pacman; then
    sudo pacman -Sy --needed --noconfirm curl ca-certificates git make poppler
  else
    echo "Unsupported package manager. Install manually: curl, git, make, poppler-utils."
  fi
}

install_uv() {
  if has_cmd uv; then
    log "uv already installed"
    return
  fi

  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

install_ollama() {
  if has_cmd ollama; then
    log "Ollama already installed"
    return
  fi

  log "Installing Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
}

start_ollama() {
  log "Starting Ollama"

  if pgrep -x ollama >/dev/null 2>&1; then
    return
  fi

  if has_cmd systemctl; then
    sudo systemctl enable --now ollama >/dev/null 2>&1 || true
  fi

  if ! pgrep -x ollama >/dev/null 2>&1; then
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 5
  fi
}

install_python_deps() {
  log "Installing Python ${PYTHON_VERSION} and project dependencies"
  uv python install "$PYTHON_VERSION"
  uv sync

  log "Installing spaCy French model"
  uv run python -m spacy download fr_core_news_sm
}

pull_models() {
  log "Pulling Ollama text model: ${TEXT_MODEL}"
  ollama pull "$TEXT_MODEL"

  log "Pulling Ollama vision model: ${VISION_MODEL}"
  ollama pull "$VISION_MODEL"
}

main() {
  cd "$(dirname "$0")"

  install_system_packages
  install_uv
  install_ollama
  start_ollama
  install_python_deps
  pull_models

  log "Installation complete"
  echo "Run: make run"
}

main "$@"
