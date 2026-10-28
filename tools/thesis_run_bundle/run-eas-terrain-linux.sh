#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-eas}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in prepare|eas|compare) ;; *) echo 'Use prepare, eas or compare.' >&2; exit 2 ;; esac
EXPERIMENT="terrain_v2_$(date +%Y%m%d_%H%M%S_%N)"
COMMON=(--terrain-profile expanded_v2 --parallel-mode adaptive
  --memory-budget-gib 20 --memory-reserve-gib 2 --cpu-budget 18
  --jobs 4 --worker-threads-max 4 --refit-threads 8 --threads 8 --mle-jobs 16)
bash "$ROOT/wsl/setup.sh" "$ROOT" --skip-check
if [[ "$MODE" == prepare ]]; then
  bash "$ROOT/wsl/run.sh" "$ROOT" prepare-terrain "${COMMON[@]}" "$@"
else
  # The new candidate profile needs its own real-data selection/calibration.
  # A verified completed run with the same protocol is reused. Old EAS is untouched.
  bash "$ROOT/wsl/run.sh" "$ROOT" eas "${COMMON[@]}" "$@"
  if [[ "$MODE" == compare ]]; then
    bash "$ROOT/wsl/run.sh" "$ROOT" compare-mle "${COMMON[@]}" \
      --method-order mle-first --experiment "$EXPERIMENT" --replicates 1 "$@"
  fi
fi
