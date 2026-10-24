#!/usr/bin/env bash
set -euo pipefail
SOURCE=$(cd -- "${1:?Pass bundle directory}" && pwd)
shift
RUNTIME="${XDG_DATA_HOME:-$HOME/.local/share}/thesis-run-cpu-v1"
PYTHON="$RUNTIME/env/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Run setup-wsl.ps1 first.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export THESIS_DATA_ROOT="$(dirname "$SOURCE")/論文資料/ERA5/global_predictors_1976_2025"
"$PYTHON" -B "$SOURCE/wsl/stage.py" "$SOURCE" "$RUNTIME/workspace"
cd "$RUNTIME/workspace"
mkdir -p "$SOURCE/outputs_wsl/logs"
LOG="$SOURCE/outputs_wsl/logs/$(date +%Y%m%d_%H%M%S_%N)_${1:-check}.log"
"$PYTHON" -B -u run.py "$@" 2>&1 | tee "$LOG"
