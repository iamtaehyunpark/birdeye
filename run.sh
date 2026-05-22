#!/usr/bin/env bash
# run.sh — start the BirdEye backend server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

# ── Copy .env if it doesn't exist ────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo "⚡ Creating .env from .env.example"
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
fi

# ── Python virtual environment ────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "⚡ Creating virtual environment…"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── Install dependencies ──────────────────────────────────────────────────────
echo "⚡ Installing Python dependencies…"
pip install -q --upgrade pip
pip install -q -r "$BACKEND_DIR/requirements.txt"

# ── Optionally install LightGlue (recommended for GPU machines) ───────────────
if python3 -c "import lightglue" 2>/dev/null; then
  echo "✓ LightGlue already installed"
else
  echo ""
  echo "⚠ LightGlue not found. Install it for best auto-initialization accuracy:"
  echo "  pip install git+https://github.com/cvg/LightGlue.git"
  echo ""
  echo "  Without it, ORB fallback will be used (CPU-only, less cross-view capable)"
  echo ""
fi

# ── Load .env ─────────────────────────────────────────────────────────────────
set -o allexport
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +o allexport

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                  BirdEye Server                      ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  API  → http://${HOST}:${PORT}/api/docs              ║"
echo "║  App  → http://localhost:${PORT}                     ║"
echo "║  Model: ${MODEL_PATH:-yolov8n.pt}                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

cd "$BACKEND_DIR"
python3 -m uvicorn main:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload \
  --reload-dir "$BACKEND_DIR"
