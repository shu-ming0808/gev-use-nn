#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-eas}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  prepare|eas|compare|compare-existing|check) ;;
  *) echo 'Use prepare, eas, compare, compare-existing or check; optionally add --plan-only after the mode.' >&2; exit 2 ;;
esac

# Only this launcher consumes --plan-only. All scientific/runtime overrides are
# forwarded unchanged, after the defaults, to the existing validated runner.
PLAN_ONLY=false
EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == --plan-only ]]; then PLAN_ONLY=true; else EXTRA+=("$arg"); fi
done
EXPERIMENT="latitude_v3_ram16_mem8_cpu20_$(date +%Y%m%d_%H%M%S_%N)"
# Latitude is a new scientific candidate: never reuse expanded_v2 real results.
# RAM admission still limits GP workers and MLE concurrency; these CPU values
# are ceilings, not a guarantee of full utilization or a physical-memory cap.
COMMON=(--terrain-profile expanded_latitude_v3 --run-label ram16_mem8_v1 --parallel-mode adaptive
  --memory-budget-gib 8 --memory-reserve-gib 1.5 --cpu-budget 20
  --jobs 2 --worker-threads-max 20 --refit-threads 20 --threads 4 --mle-jobs 20)

run_command() {
  if [[ "$PLAN_ONLY" == true ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$PLAN_ONLY" == true ]]; then
  printf 'PLAN ONLY: mode=%s; no setup, staging or computation will run.\n' "$MODE"
  printf '%s\n' \
    'Commands below are in execution order; later CLI options override defaults.' \
    'RAM values are adaptive scheduling budgets, not OS caps or an OOM guarantee.' \
    'New latitude profile: compare runs real EAS before calibrated MLE/NN comparison.' \
    'NN uses min(4, effective CPU budget) threads; --threads is not an independent NN limit.' \
    'check validates the bundle and 8 NN rows, preparing/reusing terrain; it does not run GP.' \
    'compare-existing requires a verified completed latitude-profile EAS source; old terrain results cannot substitute.'
fi

run_command bash "$ROOT/wsl/setup.sh" "$ROOT" --skip-check
case "$MODE" in
  prepare)
    run_command bash "$ROOT/wsl/run.sh" "$ROOT" prepare-terrain "${COMMON[@]}" "${EXTRA[@]}"
    ;;
  check)
    run_command bash "$ROOT/wsl/run.sh" "$ROOT" check "${COMMON[@]}" "${EXTRA[@]}"
    ;;
  eas|compare|compare-existing)
    # Isolate latitude science and 8 GiB budget from older runs while keeping
    # strict input/cache identity and completed-protocol checks.
    # No --new-run: matching completed EAS may be reused; old attempts stay intact.
    if [[ "$MODE" != compare-existing ]]; then
      run_command bash "$ROOT/wsl/run.sh" "$ROOT" eas "${COMMON[@]}" "${EXTRA[@]}"
    fi
    # compare-mle checks the completed source's scientific/data compatibility.
    # A missing or incompatible source fails; do not silently start real EAS.
    if [[ "$MODE" != eas ]]; then
      run_command bash "$ROOT/wsl/run.sh" "$ROOT" compare-mle "${COMMON[@]}" \
        --method-order mle-first --experiment "$EXPERIMENT" --replicates 1 "${EXTRA[@]}"
    fi
    ;;
esac
