"""
reproducibility.py -- Reproducibility across repeats (FTC).
===========================================================

Self-contained. Edit the CONFIG block with the EXACT path to every visit map,
run:

    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    $PY TractographyAlgo/scripts/results/reproducibility.py

For the hybrid and baseline repeat groups, computes the pairwise Fractional
Tanimoto Coefficient between every pair of runs using TWO thresholding modes:

  1. **Absolute thresholding** -- binarise at >= T raw visit counts.
  2. **Normalised relative thresholding** -- binarise at >= (pct/100)*N_streamlines,
     where N_streamlines is the total valid streamlines for that specific run.
     This compensates for pipelines generating different numbers of streamlines.

Also computes the continuous (fuzzy) FTC on normalised densities.

Outputs written to OUTPUT_DIR:
  - reproducibility.json                   (full results)
  - reproducibility_absolute.csv           (absolute thresholding)
  - reproducibility_relative.csv           (relative thresholding)
  - fig_reproducibility_ftc_absolute.png   (FTC vs absolute T)
  - fig_reproducibility_ftc_relative.png   (FTC vs relative pct)
  - reproducibility.md                     (summary)
"""

import os
import csv
import json
import glob
from itertools import combinations

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG  -- put the EXACT path to every visit map.
# =============================================================================
OUTPUT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/results/reproducibility"

# Absolute thresholds: binarise at >= T raw visits.
FTC_THRESHOLDS_ABSOLUTE = [1, 2, 3, 5, 10, 20]

# Relative thresholds: percentage of total valid streamlines for each run.
# e.g. 0.01 means "voxels visited by >= 0.01% of that run's streamlines".
FTC_THRESHOLDS_RELATIVE_PCT = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0, 15.0]

HYBRID_VISIT_MAPS = [
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_01/hybrid1_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_02/hybrid2_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_04/hybrid4_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_05/hybrid5_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_06/hybrid6_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_07/hybrid7_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_08/hybrid8_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_09/hybrid9_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_10/hybrid10_visit_map.nii.gz",
]

BASELINE_VISIT_MAPS = [
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_01/FMA_tract_seed_10seeds_BedpostX_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_02/FMA_tract_seed_10seeds_Bedpostx_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_03/baseline_3_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_04/baseline4_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_05/baseline_run_5_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_06/baseline_run_6_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_07/baseline_run_7_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_08/baseline_run_8_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_09/baseline_run_9_visit_map.nii.gz",
    "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_10/baseline_run_10_visit_map.nii.gz",
]
# =============================================================================


# ---------------------------------------------------------------- helpers ----
def load_density(path):
    """Load a visit-map volume as a float64 array."""
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        if "visit_map" in d.files and d["visit_map"] is not None:
            return np.asarray(d["visit_map"], dtype=np.float64)
        return None
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float64)


def read_total_streamlines(visit_map_path):
    """Auto-read total_streamlines from the *_params.json sitting next to the visit map.

    Each run directory contains exactly one *_params.json whose
    results.total_streamlines gives the number of valid streamlines generated.
    """
    run_dir = os.path.dirname(visit_map_path)
    param_files = glob.glob(os.path.join(run_dir, "*_params.json"))
    if not param_files:
        return None
    with open(param_files[0]) as f:
        params = json.load(f)
    return params.get("results", {}).get("total_streamlines", None)


def binarise(arr, thresh):
    """Binarise: voxels with value >= thresh become True."""
    return arr >= thresh


def binarise_relative(tdi, total_streamlines, threshold_pct):
    """Binarise a TDI using a *relative* threshold.

    Parameters
    ----------
    tdi : np.ndarray
        Raw visit-map / tract-density image (voxel values = raw streamline
        visit counts).
    total_streamlines : int
        Total number of valid streamlines in this specific tractogram run.
    threshold_pct : float
        Relative threshold expressed as a **percentage** (e.g. 0.01 means
        0.01% of total_streamlines).

    Returns
    -------
    np.ndarray[bool]
        Binary mask where voxels with visit count >= dynamic cutoff are True.

    Notes
    -----
    The dynamic absolute cutoff is:
        cutoff = (threshold_pct / 100) * total_streamlines
    This ensures the threshold scales proportionally with each run's
    streamline budget, making cross-pipeline comparison fair.
    """
    absolute_cutoff = (threshold_pct / 100.0) * total_streamlines
    return tdi >= absolute_cutoff


def jaccard(a_bin, b_bin):
    """Jaccard / Tanimoto similarity between two binary masks."""
    a = a_bin.astype(bool); b = b_bin.astype(bool)
    union = np.logical_or(a, b).sum()
    return float("nan") if union == 0 else float(np.logical_and(a, b).sum() / union)


def ftc_threshold(A, B, T):
    """FTC at absolute threshold T."""
    return jaccard(binarise(A, T), binarise(B, T))


def ftc_relative(tdi_a, n_a, tdi_b, n_b, threshold_pct):
    """FTC using normalised relative thresholding.

    Each run is binarised at its own per-run dynamic cutoff:
        cutoff_a = (threshold_pct / 100) * n_a
        cutoff_b = (threshold_pct / 100) * n_b
    Then Jaccard is computed on the two binary masks.

    Parameters
    ----------
    tdi_a, tdi_b : np.ndarray
        Raw visit-map arrays for the two runs.
    n_a, n_b : int
        Total valid streamlines for run A and run B respectively.
    threshold_pct : float
        Relative threshold as a percentage (e.g. 0.01 = 0.01%).

    Returns
    -------
    float
        Tanimoto / Jaccard similarity in [0, 1], or NaN if both masks empty.
    """
    mask_a = binarise_relative(tdi_a, n_a, threshold_pct)
    mask_b = binarise_relative(tdi_b, n_b, threshold_pct)
    return jaccard(mask_a, mask_b)


def fractional_tanimoto_continuous(A, B):
    """Fuzzy / continuous FTC on max-normalised densities."""
    A = A / A.max() if A.max() > 0 else A
    B = B / B.max() if B.max() > 0 else B
    mx = np.maximum(A, B).sum()
    return float("nan") if mx <= 0 else float(np.minimum(A, B).sum() / mx)


def summ(vals):
    """Summary statistics for a list of values."""
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "[n/a]"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# --------------------------------------------------------------- analysis ----
def analyse():
    """Run both absolute and relative FTC analysis for all pipeline groups."""
    out = {}
    for label, paths in (("hybrid", HYBRID_VISIT_MAPS), ("baseline", BASELINE_VISIT_MAPS)):
        # Load densities and their corresponding streamline counts.
        dens, n_streams = [], []
        for p in paths:
            d = load_density(p)
            if d is not None:
                n = read_total_streamlines(p)
                if n is None:
                    print(f"    WARNING: no total_streamlines for {p}, skipping")
                    continue
                dens.append(d)
                n_streams.append(n)
        print(f"  {label}: {len(dens)}/{len(paths)} densities loaded"
              f"  (streamlines: {n_streams})")
        if len(dens) < 2:
            out[label] = {"n_runs": len(dens), "note": "need >=2 runs"}; continue

        pairs = list(combinations(range(len(dens)), 2))
        entry = {
            "n_runs": len(dens),
            "n_pairs": len(pairs),
            "streamline_counts": n_streams,
        }

        # --- Absolute thresholding ---
        for T in FTC_THRESHOLDS_ABSOLUTE:
            vals = [ftc_threshold(dens[i], dens[j], T) for i, j in pairs]
            entry[f"ftc_abs_T{T}"] = summ(vals)
            entry[f"ftc_abs_T{T}_pairs"] = vals

        # --- Normalised relative thresholding ---
        for pct in FTC_THRESHOLDS_RELATIVE_PCT:
            vals = [ftc_relative(dens[i], n_streams[i],
                                 dens[j], n_streams[j], pct)
                    for i, j in pairs]
            # Use a string key that encodes the percentage unambiguously.
            pct_key = f"{pct:g}"
            entry[f"ftc_rel_{pct_key}pct"] = summ(vals)
            entry[f"ftc_rel_{pct_key}pct_pairs"] = vals

        # --- Continuous FTC ---
        entry["ftc_continuous"] = summ(
            [fractional_tanimoto_continuous(dens[i], dens[j]) for i, j in pairs])

        out[label] = entry
    return out


# --------------------------------------------------------- CSV / figures ----
def csv_absolute(result, path):
    """CSV for absolute thresholding results."""
    fieldnames = ["pipeline", "threshold_T", "mean_ftc", "std_ftc",
                  "min_ftc", "max_ftc", "n_pairs"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond in ("hybrid", "baseline"):
            e = result.get(cond, {})
            for T in FTC_THRESHOLDS_ABSOLUTE:
                s = e.get(f"ftc_abs_T{T}", {})
                writer.writerow({
                    "pipeline": cond,
                    "threshold_T": T,
                    "mean_ftc": fmt(s.get("mean")),
                    "std_ftc": fmt(s.get("std")),
                    "min_ftc": fmt(s.get("min")),
                    "max_ftc": fmt(s.get("max")),
                    "n_pairs": s.get("n", 0),
                })


def csv_relative(result, path):
    """CSV for normalised relative thresholding results."""
    fieldnames = ["pipeline", "threshold_pct", "mean_ftc", "std_ftc",
                  "min_ftc", "max_ftc", "n_pairs"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond in ("hybrid", "baseline"):
            e = result.get(cond, {})
            for pct in FTC_THRESHOLDS_RELATIVE_PCT:
                pct_key = f"{pct:g}"
                s = e.get(f"ftc_rel_{pct_key}pct", {})
                writer.writerow({
                    "pipeline": cond,
                    "threshold_pct": pct,
                    "mean_ftc": fmt(s.get("mean")),
                    "std_ftc": fmt(s.get("std")),
                    "min_ftc": fmt(s.get("min")),
                    "max_ftc": fmt(s.get("max")),
                    "n_pairs": s.get("n", 0),
                })


def figure_absolute(result, path):
    """Line plot: mean FTC (±1 std) vs. absolute threshold T."""
    palette = {"hybrid": "mediumseagreen", "baseline": "salmon"}
    marker = {"hybrid": "o", "baseline": "s"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ("hybrid", "baseline"):
        e = result.get(cond, {})
        means, stds, thresholds = [], [], []
        for T in FTC_THRESHOLDS_ABSOLUTE:
            s = e.get(f"ftc_abs_T{T}")
            if s and s.get("mean") is not None:
                thresholds.append(T)
                means.append(s["mean"])
                stds.append(s["std"])
        if not means:
            continue
        means, stds = np.asarray(means), np.asarray(stds)
        ax.plot(thresholds, means, marker=marker[cond], color=palette[cond],
                label=cond.capitalize(), linewidth=2, markersize=7)
        ax.fill_between(thresholds, means - stds, np.minimum(means + stds, 1.0),
                        color=palette[cond], alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("Threshold T (absolute visit count)", fontsize=12)
    ax.set_ylabel("Mean Fractional Tanimoto Coefficient", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(FTC_THRESHOLDS_ABSOLUTE)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title("Reproducibility: FTC vs. Absolute Threshold",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt

def figure_relative(result, path):
    """Line plot: mean FTC (±1 std) vs. relative threshold percentage.

    X-axis uses a log scale since the percentage thresholds span several
    orders of magnitude (0.001% to 5%).
    """
    palette = {"hybrid": "mediumseagreen", "baseline": "salmon"}
    marker = {"hybrid": "o", "baseline": "s"}
    
    # Map internal dictionary keys to their final legend labels
    display_name = {"hybrid": "Hybrid", "baseline": "Diffusion-only"}
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for cond in ("hybrid", "baseline"):
        e = result.get(cond, {})
        means, stds, pcts = [], [], []
        
        # Note: make sure FTC_THRESHOLDS_RELATIVE_PCT is defined in your broader script!
        for pct in FTC_THRESHOLDS_RELATIVE_PCT:
            pct_key = f"{pct:g}"
            s = e.get(f"ftc_rel_{pct_key}pct")
            if s and s.get("mean") is not None:
                pcts.append(pct)
                means.append(s["mean"])
                stds.append(s["std"])
                
        if not means:
            continue
            
        means, stds = np.asarray(means), np.asarray(stds)
        
        # Swap out cond.capitalize() for the mapped display_name
        ax.plot(pcts, means, marker=marker[cond], color=palette[cond],
                label=display_name[cond], linewidth=2, markersize=7)
                
        ax.fill_between(pcts, means - stds, np.minimum(means + stds, 1.0),
                        color=palette[cond], alpha=0.18)
                        
    ax.set_xlabel("Relative Threshold (% of total streamlines)", fontsize=12)
    ax.set_ylabel("Mean Fractional Tanimoto Coefficient", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    ax.set_xticks(FTC_THRESHOLDS_RELATIVE_PCT)
    ax.set_xticklabels([f"{p:g}%" for p in FTC_THRESHOLDS_RELATIVE_PCT],
                       fontsize=9, ha="right")
                       
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)
    
    ax.set_title("Reproducibility: FTC vs. Normalised Relative Threshold",
                 fontweight="bold", fontsize=13)
                 
    fig.tight_layout()
    fig.savefig(path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def markdown(result):
    """Generate a markdown summary covering both thresholding modes."""
    L = ["## Reproducibility (FTC)", ""]

    L.append("### Absolute Thresholding")
    for cond in ("hybrid", "baseline"):
        e = result.get(cond, {})
        n_str = e.get("streamline_counts", [])
        if n_str:
            L.append(f"- **{cond.upper()}** streamline counts: "
                     f"mean={np.mean(n_str):.0f}, range=[{min(n_str)}, {max(n_str)}]")
        for T in FTC_THRESHOLDS_ABSOLUTE:
            k = f"ftc_abs_T{T}"
            if k in e:
                L.append(f"  - FTC T={T}: {fmt(e[k]['mean'])} "
                         f"+/- {fmt(e[k]['std'])} (n_pairs={e.get('n_pairs')})")
    L.append("")

    L.append("### Normalised Relative Thresholding")
    for cond in ("hybrid", "baseline"):
        e = result.get(cond, {})
        for pct in FTC_THRESHOLDS_RELATIVE_PCT:
            pct_key = f"{pct:g}"
            k = f"ftc_rel_{pct_key}pct"
            if k in e:
                L.append(f"  - FTC {cond.upper()} {pct}%: {fmt(e[k]['mean'])} "
                         f"+/- {fmt(e[k]['std'])} (n_pairs={e.get('n_pairs')})")
    L.append("")
    return L


# ------------------------------------------------------------------ main ----
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== reproducibility (FTC) ===")
    result = analyse()

    # JSON (full dump)
    with open(os.path.join(OUTPUT_DIR, "reproducibility.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Markdown summary
    with open(os.path.join(OUTPUT_DIR, "reproducibility.md"), "w") as f:
        f.write("\n".join(markdown(result)) + "\n")

    # CSVs
    csv_absolute(result, os.path.join(OUTPUT_DIR, "reproducibility_absolute.csv"))
    csv_relative(result, os.path.join(OUTPUT_DIR, "reproducibility_relative.csv"))

    # Figures
    figure_relative(result, os.path.join(OUTPUT_DIR, "fig_reproducibility_ftc_relative.png"))

    print(f"  outputs written to {OUTPUT_DIR}")
    print(f"    - reproducibility.json / .md")
    print(f"    - reproducibility_absolute.csv / reproducibility_relative.csv")
    print(f"    - fig_reproducibility_ftc_relative.png")


if __name__ == "__main__":
    main()
