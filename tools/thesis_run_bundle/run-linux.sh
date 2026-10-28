#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT="u265_rolling_$(date +%Y%m%d_%H%M%S_%N)"
# Native Linux or WSL. Dependencies and prepared inputs stay on Linux filesystem.
bash "$ROOT/wsl/setup.sh" "$ROOT" --skip-check
bash "$ROOT/wsl/run.sh" "$ROOT" compare-mle --parallel-mode adaptive \
  --memory-budget-gib 20 --memory-reserve-gib 2 --cpu-budget 18 --jobs 4 --mle-jobs 16 \
  --worker-threads-max 4 --refit-threads 8 --threads 8 --method-order mle-first \
  --experiment "$EXPERIMENT" --replicates 1 "$@"
