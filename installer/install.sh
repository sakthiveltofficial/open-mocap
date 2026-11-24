#!/usr/bin/env bash
# Open Mocap installer launcher (Linux / macOS).
# Fetches uv (a standalone Python manager) into this folder if needed, then
# runs install.py on a managed Python 3.12. Arguments pass through, e.g.
#   bash install.sh --profile both --hf-token hf_xxx
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

UV=""
if [ -x "$DIR/.uv/uv" ]; then
  UV="$DIR/.uv/uv"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  echo "[Open Mocap] Fetching uv (Python environment manager)..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DIR/.uv" UV_NO_MODIFY_PATH=1 sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DIR/.uv" UV_NO_MODIFY_PATH=1 sh
  else
    echo "[ERROR] Need curl or wget to fetch uv." >&2
    exit 1
  fi
  UV="$DIR/.uv/uv"
fi

export OPENMOCAP_UV="$UV"
"$UV" run --no-project --python 3.12 "$DIR/install.py" "$@" 2>&1 | tee "$DIR/install.log"
exit "${PIPESTATUS[0]}"
