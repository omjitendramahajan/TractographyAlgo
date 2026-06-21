"""
reference_overlap.py -- Anatomical reference overlap.
=====================================================

Self-contained. Edit the CONFIG block with the EXACT path to every file, run:

    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    $PY TractographyAlgo/scripts/results/reference_overlap.py

Dice/Jaccard of the representative hybrid and baseline tracts against an
anatomical reference mask (e.g. the XTRACT standardised FMA, warped into native
dMRI space). Set REFERENCE_MASK to enable; otherwise the test prints "skipped".

Writes reference_overlap.json and reference_overlap.md to OUTPUT_DIR (no figure).
"""

import os
import json

import numpy as np
import nibabel as nib

# =============================================================================
# CONFIG  -- put the EXACT path to every file.
# =============================================================================
OUTPUT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/TRK_outputs/_postprocessing"

DENSITY_THRESHOLD = 1

# Anatomical reference mask warped to native dMRI space. None -> skip.
REFERENCE_MASK = None

# Representative hybrid and baseline tract densities to compare to the reference.
HYBRID_VISIT_MAP = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/TRK_outputs/hybrid_run_01/FMA_tract_seed_10seeds_PSOCT_visit_map.nii.gz"
BASELINE_VISIT_MAP = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/TRK_outputs/baseline_run_01/FMA_tract_seed_10seeds_BedpostX_visit_map.nii.gz"
# =============================================================================


# ---------------------------------------------------------------- helpers ----
def load_density(path):
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        if "visit_map" in d.files and d["visit_map"] is not None:
            return np.asarray(d["visit_map"], dtype=np.float64)
        return None
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float64)


def binarise(arr, thresh):
    return arr >= thresh


def dice(a_bin, b_bin):
    a = a_bin.astype(bool); b = b_bin.astype(bool)
    denom = a.sum() + b.sum()
    return float("nan") if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def jaccard(a_bin, b_bin):
    a = a_bin.astype(bool); b = b_bin.astype(bool)
    union = np.logical_or(a, b).sum()
    return float("nan") if union == 0 else float(np.logical_and(a, b).sum() / union)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "[n/a]"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# --------------------------------------------------------------- analysis ----
def analyse():
    if not REFERENCE_MASK or not os.path.exists(REFERENCE_MASK):
        return {"skipped": "REFERENCE_MASK not set / not found"}
    ref_bin = np.asarray(nib.load(REFERENCE_MASK).get_fdata(), dtype=np.float64) > 0
    print(f"  reference: {int(ref_bin.sum())} voxels")
    out = {}
    for label, path in (("hybrid", HYBRID_VISIT_MAP), ("baseline", BASELINE_VISIT_MAP)):
        arr = load_density(path)
        if arr is None:
            continue
        a_bin = binarise(arr, DENSITY_THRESHOLD)
        if a_bin.shape != ref_bin.shape:
            out[label] = {"error": f"shape {a_bin.shape} != reference {ref_bin.shape}"}; continue
        out[label] = {"dice": dice(a_bin, ref_bin), "jaccard": jaccard(a_bin, ref_bin)}
    return out


def markdown(result):
    L = ["## Anatomical reference overlap", ""]
    if "skipped" in result:
        L.append(f"- (skipped: {result['skipped']})")
    else:
        if "hybrid" in result:
            L.append(f"- HYBRID REF DICE = {fmt(result['hybrid'].get('dice'))}")
        if "baseline" in result:
            L.append(f"- BASELINE REF DICE = {fmt(result['baseline'].get('dice'))}")
    L.append("")
    return L


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== reference overlap ===")
    result = analyse()
    with open(os.path.join(OUTPUT_DIR, "reference_overlap.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "reference_overlap.md"), "w") as f:
        f.write("\n".join(markdown(result)) + "\n")
    print(f"  wrote reference_overlap.json/.md to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
