#!/usr/bin/env bash
# The definition of done. Keep it fast (< 2 min) or split a slow suite behind `--full`.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
if [ "${1:-}" = "--full" ]; then
  uv run pytest -q
else
  uv run pytest -q -m "not slow"
fi
