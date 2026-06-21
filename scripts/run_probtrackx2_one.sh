#!/usr/bin/env bash
# Single-shot probtrackx2 run with a given random seed and output dir.
# Called by evaluate_reproducibility.py to produce N runs.
#
# Usage: run_probtrackx2_one.sh <random_seed_int> <output_dir>

set -euo pipefail

SEED="${1:?seed required}"
OUT_DIR="${2:?output dir required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PROJECT_DIR")"
CFG="$SCRIPT_DIR/fma_config.json"

read_json() {
  local key="$1"
  if command -v jq >/dev/null 2>&1; then
    jq -r "$key" "$CFG"
  else
    python - "$CFG" "$key" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
v = cfg
for k in sys.argv[2].lstrip(".").split("."):
    v = v[k]
print(v)
PY
  fi
}

BEDPOSTX_DIR="$(read_json '.paths.bedpostx_dir')"
MASKS_DIR_REL="$(read_json '.paths.masks_dir')"
MASKS_DIR="$REPO_ROOT/$MASKS_DIR_REL"
FMA_SEED="$MASKS_DIR/$(read_json '.paths.fma_seed')"
FMA_TARGET="$MASKS_DIR/$(read_json '.paths.fma_target')"
FMA_EXCLUDE="$MASKS_DIR/$(read_json '.paths.fma_exclude')"
BRAIN_MASK="$MASKS_DIR/$(read_json '.paths.brain_mask')"
NSAMPLES="$(read_json '.probtrackx2.nsamples')"
NSTEPS="$(read_json '.probtrackx2.nsteps')"
STEPLEN="$(read_json '.probtrackx2.steplength')"
CURV="$(read_json '.probtrackx2.curvature_thresh')"

if [[ -z "${FSLDIR:-}" ]]; then
  for c in "$HOME/fsl" "/usr/local/fsl" "/opt/fsl"; do
    [[ -d "$c" ]] && export FSLDIR="$c" && break
  done
fi
if [[ -n "${FSLDIR:-}" ]] && [[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]]; then
  # shellcheck disable=SC1090
  source "$FSLDIR/etc/fslconf/fsl.sh"
  export PATH="$FSLDIR/bin:$PATH"
fi
command -v probtrackx2 >/dev/null || { echo "probtrackx2 not on PATH" >&2; exit 1; }

mkdir -p "$OUT_DIR"

probtrackx2 \
  -x "$FMA_SEED" \
  --waypoints="$FMA_TARGET" \
  --avoid="$FMA_EXCLUDE" \
  -s "$BEDPOSTX_DIR/merged" \
  -m "$BRAIN_MASK" \
  --dir="$OUT_DIR" \
  --forcedir \
  --nsamples="$NSAMPLES" \
  --nsteps="$NSTEPS" \
  --steplength="$STEPLEN" \
  --cthr="$CURV" \
  --rseed="$SEED" \
  --loopcheck \
  --onewaycondition \
  --opd \
  2>&1 | tail -20
