#!/usr/bin/env bash
# Format check, lint and tests. `--full` also runs the slow end-to-end pipeline test.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff format --check .
uv run ruff check .
if [ "${1:-}" = "--full" ]; then
  uv run pytest -q
else
  uv run pytest -q -m "not slow"
fi
