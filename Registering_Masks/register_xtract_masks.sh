#!/bin/bash
# register_xtract_masks.sh
#
# Register an XTRACT macaque tract's masks from F99 template space
# into Moe subject (BedpostX) space, using the existing F99 -> Moe
# FNIRT warp that was built for the FMA pipeline.
#
# Mirrors the existing FMA layout, under Registering_Masks/:
#   Registering_Masks/<TRACT>/original_in_F99/   (copies from XTRACT)
#   Registering_Masks/<TRACT>/Moe_registered/    (after applywarp)
#   Registering_Masks/<TRACT>/Moe_rebinarised/   (after fslmaths)
#
# Usage:
#   # FSLDIR is typically set by ~/.zprofile; otherwise source it.
#   ./register_xtract_masks.sh CST_l
#   ./register_xtract_masks.sh CST_r
#   ls "$FSLDIR/data/xtract_data/MACAQUE/"   # to see available tracts

set -euo pipefail

TRACT="${1:-}"
if [[ -z "$TRACT" ]]; then
    echo "Usage: $0 <tract_name>   (e.g. CST_l)"
    echo "Available tracts: ls \$FSLDIR/data/xtract_data/MACAQUE/"
    exit 1
fi

# XTRACT folder names are lowercase (cst_l), our output convention is UPPER (CST_l).
TRACT_LC="$(echo "$TRACT" | tr '[:upper:]' '[:lower:]')"

PROJECT_ROOT="/Users/ommahajan/Desktop/Year_4/MEng Individual project"
REG_BASE="${PROJECT_ROOT}/TractographyAlgo/Registering_Masks"
WARP_FIELD="${REG_BASE}/F99_2_Moe_field.nii.gz"
# Reference volume: prefer the uncompressed BedpostX dir to match
# run_tractography.py, but accept either .nii or .nii.gz.
REF_VOLUME=""
for candidate in \
    "${PROJECT_ROOT}/DATA/data.bedpostX_SSFP_uncompressed/nodif_brain_mask.nii" \
    "${PROJECT_ROOT}/DATA/data.bedpostX_SSFP_uncompressed/nodif_brain_mask.nii.gz" \
    "${PROJECT_ROOT}/DATA/data.bedpostX_SSFP/nodif_brain_mask.nii.gz" \
    "${PROJECT_ROOT}/DATA/data.bedpostX_SSFP/nodif_brain_mask.nii"
do
    if [[ -f "$candidate" ]]; then REF_VOLUME="$candidate"; break; fi
done
OUT_BASE="${REG_BASE}/${TRACT}"

# ---- preflight ----
if [[ -z "${FSLDIR:-}" ]]; then
    echo "ERROR: FSLDIR not set. Source your FSL config first:"
    echo "  source ~/.zprofile        # or"
    echo "  source \$FSLDIR/etc/fslconf/fsl.sh"
    exit 1
fi
XTRACT_DIR="${FSLDIR}/data/xtract_data/MACAQUE/${TRACT_LC}"
if [[ ! -d "$XTRACT_DIR" ]]; then
    echo "ERROR: XTRACT macaque tract dir not found:"
    echo "  $XTRACT_DIR"
    echo "Tracts available:"
    ls "${FSLDIR}/data/xtract_data/MACAQUE/" 2>/dev/null || true
    exit 1
fi
[[ -f "$WARP_FIELD" ]] || { echo "ERROR: warp not found: $WARP_FIELD"; exit 1; }
[[ -f "$REF_VOLUME" ]] || { echo "ERROR: ref not found:  $REF_VOLUME"; exit 1; }
command -v applywarp >/dev/null || { echo "ERROR: applywarp not on PATH"; exit 1; }
command -v fslmaths  >/dev/null || { echo "ERROR: fslmaths not on PATH";  exit 1; }

mkdir -p "${OUT_BASE}/original_in_F99" \
         "${OUT_BASE}/Moe_registered"  \
         "${OUT_BASE}/Moe_rebinarised"

echo "===================="
echo "Tract:   $TRACT"
echo "Source:  $XTRACT_DIR"
echo "Warp:    $WARP_FIELD"
echo "Ref:     $REF_VOLUME"
echo "Output:  $OUT_BASE"
echo "===================="
echo "Files in XTRACT source:"
ls "$XTRACT_DIR"
echo

shopt -s nullglob
for SRC in "${XTRACT_DIR}"/*.nii.gz; do
    MASK_FILE=$(basename "$SRC")
    MASK_NAME="${MASK_FILE%.nii.gz}"

    F99_COPY="${OUT_BASE}/original_in_F99/${MASK_FILE}"
    REG="${OUT_BASE}/Moe_registered/${TRACT}_${MASK_NAME}_Moe.nii.gz"
    BIN="${OUT_BASE}/Moe_rebinarised/${TRACT}_${MASK_NAME}_Moe_bin.nii.gz"

    echo "[$MASK_NAME]"
    cp "$SRC" "$F99_COPY"
    applywarp -i "$F99_COPY" -r "$REF_VOLUME" -w "$WARP_FIELD" -o "$REG" --interp=trilinear
    fslmaths "$REG" -thr 0.5 -bin "$BIN"
    echo "  -> $BIN"
    echo
done

echo "==== Done. Suggested run_tractography.py CONFIG entries: ===="
# XTRACT file -> CONFIG key mapping
declare -a XTRACT_NAMES=(seed   target  stop  exclude)
declare -a CFG_KEYS=(    seed   target  stop  exclude)
for idx in "${!XTRACT_NAMES[@]}"; do
    xname="${XTRACT_NAMES[$idx]}"
    ckey="${CFG_KEYS[$idx]}"
    BIN="${OUT_BASE}/Moe_rebinarised/${TRACT}_${xname}_Moe_bin.nii.gz"
    if [[ -f "$BIN" ]]; then
        echo "  '${ckey}_mask': \"${BIN}\","
    fi
done
echo
echo "Sanity-check before tracking:"
echo "  fsleyes $REF_VOLUME ${OUT_BASE}/Moe_rebinarised/*.nii.gz"
echo "  - seed should sit in left M1"
echo "  - stop/target should sit at the cerebral peduncle / pons"
echo "  - exclusion should cover the contralateral hemisphere & non-WM regions"
