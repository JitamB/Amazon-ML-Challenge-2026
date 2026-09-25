#!/usr/bin/env bash
# Create the project's Python environment in .venv and check it works.
#   scripts/setup_env.sh            # uses python3.12 / python3.11 / python3, whichever is found first
#   PYTHON=/path/to/python scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3; do command -v "$c" >/dev/null && { PY="$c"; break; }; done
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), f"Python >= 3.11 required, found {sys.version.split()[0]}"'

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python -c "import pandas, numpy, pyarrow, rapidfuzz, lightgbm, yaml, unidecode, indic_transliteration; print('environment OK:', 'python', __import__('sys').version.split()[0])"
echo "Activate with: source .venv/bin/activate"
