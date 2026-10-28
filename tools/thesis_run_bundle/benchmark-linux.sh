#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$ROOT/wsl/setup.sh" "$ROOT" --skip-check
RUNTIME="${XDG_DATA_HOME:-$HOME/.local/share}/thesis-run-cpu-v1"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd "$RUNTIME/workspace"
mkdir -p outputs/benchmarks
"$RUNTIME/env/bin/python" -B -u benchmark_gp_execution.py \
  --profiles 4x4,6x2,8x2 --train-grids 512 --tasks 8 --repeats 2 \
  --cpu-budget 18 --memory-budget-gib 20 \
  --output "outputs/benchmarks/gp_$(date +%Y%m%d_%H%M%S_%N).json" "$@"
