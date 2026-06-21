#!/usr/bin/env bash
# probtrackx2 baseline for the FMA tract.
#
# Reads paths/parameters from scripts/fma_config.json (via jq if available, else
# python). All three conditions (probtrackx2, BedpostX-only, PSOCT-constrained)
# read the SAME config so we cannot drift.
#
# Outputs (under <repo>/TractographyAlgo(withmask)/results_for_report/real_data/FMA_probtrackx2/):
#   fdt_paths.nii.gz            -- streamline density volume
#   probtrackx.log              -- run log
#   waytotal                    -- streamlines reaching waypoint
#   particle_paths              -- per-particle streamlines  (if --savepaths set)
#   FMA_probtrackx2.trk         -- converted TRK via convert_fsl_to_trk.py
#
# Requirements: FSL 6.x with probtrackx2 on PATH (or FSLDIR set).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$PROJECT_DIR")"
CFG="$SCRIPT_DIR/fma_config.json"

# ---- config readers -----------------------------------------------------------
# Use jq if available; else fall back to python.
read_json() {
  local key="$1"
  if command -v jq >/dev/null 2>&1; then
    jq -r "$key" "$CFG"
  else
    python - "$CFG" "$key" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
keys = sys.argv[2].lstrip(".").split(".")
v = cfg
for k in keys:
    v = v[k]
print(v)
PY
  fi
}

BEDPOSTX_DIR="$(read_json '.paths.bedpostx_dir')"
FMA_SEED="$REPO_ROOT/$(read_json '.paths.fma_seed')"
FMA_TARGET="$REPO_ROOT/$(read_json '.paths.fma_target')"
FMA_EXCLUDE="$REPO_ROOT/$(read_json '.paths.fma_exclude')"
BRAIN_MASK="$(read_json '.paths.brain_mask')"

NSAMPLES="$(read_json '.probtrackx2.nsamples')"
NSTEPS="$(read_json '.probtrackx2.nsteps')"
STEPLEN="$(read_json '.probtrackx2.steplength')"
CURV="$(read_json '.probtrackx2.curvature_thresh')"

RESULTS_DIR="$(read_json '.paths.results_dir')"
OUT_DIR="$RESULTS_DIR/FMA_probtrackx2"

# ---- preflight ----------------------------------------------------------------
if [[ -z "${FSLDIR:-}" ]]; then
  for candidate in "$HOME/fsl" "/usr/local/fsl" "/opt/fsl"; do
    if [[ -d "$candidate" ]]; then
      export FSLDIR="$candidate"
      break
    fi
  done
fi
if [[ -n "${FSLDIR:-}" ]] && [[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]]; then
  # shellcheck disable=SC1090
  source "$FSLDIR/etc/fslconf/fsl.sh"
  export PATH="$FSLDIR/bin:$PATH"
fi
if ! command -v probtrackx2 >/dev/null; then
  echo "ERROR: probtrackx2 not on PATH and FSLDIR is unset/invalid." >&2
  exit 1
fi
if [[ ! -d "$BEDPOSTX_DIR" ]]; then
  echo "ERROR: BedpostX directory not found: $BEDPOSTX_DIR" >&2
  echo "Connect the HDD or edit fma_config.json -> paths.bedpostx_dir" >&2
  exit 1
fi
for f in "$FMA_SEED" "$FMA_TARGET" "$FMA_EXCLUDE"; do
  [[ -f "$f" ]] || { echo "ERROR: mask missing: $f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/probtrackx.log"

# ---- run ----------------------------------------------------------------------
echo "probtrackx2 FMA baseline"
echo "  bedpostX: $BEDPOSTX_DIR"
echo "  seed:     $FMA_SEED"
echo "  waypoint: $FMA_TARGET"
echo "  exclude:  $FMA_EXCLUDE"
echo "  nsamples=$NSAMPLES nsteps=$NSTEPS steplength=$STEPLEN curvature_thresh=$CURV"
echo "  out:      $OUT_DIR"
echo

# --savepaths writes per-particle streamlines (text). Required so we can build
# a TRK that visualises identically to our Python tracker outputs.
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
  --loopcheck \
  --onewaycondition \
  --opd \
  --savepaths="$OUT_DIR/particle_paths" \
  2>&1 | tee "$LOG"

# ---- convert paths -> TRK for matched visualisation --------------------------
PATHS_FILE="$OUT_DIR/particle_paths"
if [[ -f "$PATHS_FILE" ]]; then
  echo
  echo "Converting particle_paths -> TRK ..."
  PY_BIN="$(command -v python || echo /Users/ommahajan/anaconda3/envs/tractography/bin/python)"
  "$PY_BIN" "$SCRIPT_DIR/convert_probtrackx2_to_trk.py" \
    --paths "$PATHS_FILE" \
    --ref   "$BRAIN_MASK" \
    --out   "$OUT_DIR/FMA_probtrackx2.trk"
fi

echo
echo "Done. Outputs:"
ls -lh "$OUT_DIR"
