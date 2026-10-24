#!/usr/bin/env bash
set -euo pipefail
SOURCE=$(cd -- "${1:?Pass the mounted bundle directory}" && pwd)
RUNTIME="${XDG_DATA_HOME:-$HOME/.local/share}/thesis-run-cpu-v1"
mkdir -p "$RUNTIME/bin"
if [[ ! -x "$RUNTIME/bin/uv" ]]; then
  INSTALLER=$(mktemp)
  trap 'rm -f "$INSTALLER"' EXIT
  curl --fail --location --retry 3 https://astral.sh/uv/install.sh --output "$INSTALLER"
  UV_INSTALL_DIR="$RUNTIME/bin" UV_NO_MODIFY_PATH=1 sh "$INSTALLER"
  rm -f "$INSTALLER"
  trap - EXIT
fi
export UV_PYTHON_INSTALL_DIR="$RUNTIME/python"
export UV_PROJECT_ENVIRONMENT="$RUNTIME/env"
export UV_CACHE_DIR="$RUNTIME/cache"
"$RUNTIME/bin/uv" python install 3.10.20
"$RUNTIME/bin/uv" sync --project "$SOURCE/wsl" --locked --no-dev --python 3.10.20
"$RUNTIME/env/bin/python" -B "$SOURCE/wsl/stage.py" "$SOURCE" "$RUNTIME/workspace"
bash "$SOURCE/wsl/run.sh" "$SOURCE" check
