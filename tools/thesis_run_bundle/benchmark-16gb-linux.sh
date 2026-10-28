#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME="${XDG_DATA_HOME:-$HOME/.local/share}/thesis-run-cpu-v1"
PYTHON="$RUNTIME/env/bin/python"

# This wrapper alone consumes --plan-only. Preserve all benchmark overrides in
# their original order, after defaults, for argparse's last-value-wins rules.
PLAN_ONLY=false
EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == --plan-only ]]; then PLAN_ONLY=true; else EXTRA+=("$arg"); fi
done
OUTPUT="outputs/benchmarks/gp_ram16_layout_$(date +%Y%m%d_%H%M%S_%N).json"
DEFAULTS=(--profiles 2x4,2x10,1x20 --train-grids 2000 --tasks 4 --repeats 2
  --cpu-budget 20 --memory-budget-gib 8 --output "$OUTPUT")

if [[ "$PLAN_ONLY" == true ]]; then
  printf '%s\n' \
    'PLAN ONLY: no setup, staging, directory creation or computation will run.' \
    'Default comparison: 2x4, 2x10, 1x20; 2000 training grids, 4 tasks, 2 repeats.' \
    'CPU budget: 20; application RAM budget: 8 GiB; benchmark RAM reserve: 2 GiB.' \
    'Native library initialization: 20 threads before imports; benchmark overrides may raise it.' \
    'The benchmark checks actual CPU/RAM and may skip unsafe profiles. No winning layout is applied.' \
    'Synthetic exact-GP fits only, not a complete EAS/Spatial-CV benchmark. This workload can take minutes.' \
    'Commands below are in order; later CLI arguments override benchmark defaults.'
  printf '  '
  printf '%q ' bash "$ROOT/wsl/setup.sh" "$ROOT" --skip-check
  printf '\n  cd -- '
  printf '%q' "$RUNTIME/workspace"
  printf '\n  '
  printf '%q ' "$PYTHON" -B -u benchmark_gp_execution.py "${DEFAULTS[@]}" "${EXTRA[@]}"
  printf '\n'
  exit 0
fi

# The existing setup uses the locked CPU-only environment and verified staging.
# Initialize at least the largest default native count before any Python import.
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20
bash "$ROOT/wsl/setup.sh" "$ROOT" --skip-check
cd -- "$RUNTIME/workspace"
mkdir -p outputs/benchmarks
exec "$PYTHON" -B -u benchmark_gp_execution.py "${DEFAULTS[@]}" "${EXTRA[@]}"
