#!/usr/bin/env bash
# Launch CrowdCanvas. Creates a local venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
fi

# Also update dependencies after an upgrade; an existing venv is not necessarily
# complete. pip skips packages that already satisfy these requirements.
./.venv/bin/python -m pip install -r requirements.txt
exec ./.venv/bin/streamlit run app.py
