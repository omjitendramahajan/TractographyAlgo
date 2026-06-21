"""
volume_overlap.py -- Hybrid vs baseline: volumes, streamline counts, overlap.
=============================================================================

Self-contained. Nothing is imported from the other result scripts -- edit the
CONFIG block below with the EXACT path to every file this test needs, then run:

    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    $PY TractographyAlgo/scripts/results/volume_overlap.py

Computes, over the hybrid and baseline repeat groups:
  * tract volume (mm^3) and retained streamline count (mean +/- sd),
  * headline Dice/Jaccard between the representative hybrid and baseline runs,
  * mean pairwise Dice over all hybrid x baseline pairs.

It ALSO retains the raw per-run values and the full pairwise Dice matrix so the
figures below can show the actual distributions rather than just mean +/- sd.

Writes volume_overlap.json, volume_overlap.md and these figures to OUTPUT_DIR:
  * fig_volume_count.png   -- boxplots (+ swarm of individual runs)
  * fig_dice_heatmap.png   -- 10x10 pairwise Dice matrix + value distribution
"""

import os
import json

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =============================================================================
# CONFIG  -- put the EXACT path to every file. (.nii.gz visit maps preferred;
# a *_stats.npz also works for the density, but then volume uses VOXEL_VOLUME_MM3
# because the npz carries no affine.)
# =============================================================================

# Where the outputs of THIS script go.
OUTPUT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/results/volume_overlap/"

# Binarise visit maps at >= this many visits.
DENSITY_THRESHOLD = 1

# Fallback voxel volume (mm^3) used only when a density is read from a *_stats.npz
# (no affine). When you point at *_visit_map.nii.gz the affine is used instead.
VOXEL_VOLUME_MM3 = 0.064

# Consistent colours across every figure.
HYB_COLOR = "mediumseagreen"
BAS_COLOR = "salmon"

# --- Hybrid runs: one entry per repeat. visit_map = density; trk = streamlines.
HYBRID_RUNS = [
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_01/hybrid1_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_01/hybrid1.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_02/hybrid2_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_02/hybrid2.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_04/hybrid4_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_04/hybrid4.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_05/hybrid5_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_05/hybrid5.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_06/hybrid6_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_06/hybrid6.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_07/hybrid7_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_07/hybrid7.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_08/hybrid8_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_08/hybrid8.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_09/hybrid9_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_09/hybrid9.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_10/hybrid10_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_10/hybrid10.trk"},
]

# --- Baseline runs: one entry per repeat.
BASELINE_RUNS = [
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_01/FMA_tract_seed_10seeds_BedpostX_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_01/FMA_tract_seed_10seeds_BedpostX.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_02/FMA_tract_seed_10seeds_Bedpostx_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_02/FMA_tract_seed_10seeds_Bedpostx.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_03/baseline_3_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_03/baseline_3.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_04/baseline4_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_04/baseline4.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_05/baseline_run_5_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_05/baseline_run_5.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_06/baseline_run_6_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_06/baseline_run_6.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_07/baseline_run_7_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_07/baseline_run_7.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_08/baseline_run_8_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_08/baseline_run_8.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_09/baseline_run_9_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_09/baseline_run_9.trk"},
    {"visit_map": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_10/baseline_run_10_visit_map.nii.gz",
     "trk":       "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_10/baseline_run_10.trk"},
]

# Representative runs for the headline Dice/Jaccard (visit maps only).
HYBRID_DEFAULT_VISIT_MAP = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz"
BASELINE_DEFAULT_VISIT_MAP = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_06/baseline_run_6_visit_map.nii.gz"

# =============================================================================


# ---------------------------------------------------------------- helpers ----
def load_density(path):
    """(array, affine) from a *_visit_map.nii.gz, or (array, None) from a *.npz."""
    if not path or not os.path.exists(path):
        return None, None
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        if "visit_map" in d.files and d["visit_map"] is not None:
            return np.asarray(d["visit_map"], dtype=np.float64), None
        return None, None
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float64), img.affine


def voxel_volume_mm3(affine):
    return float(abs(np.linalg.det(affine[:3, :3])))


def streamline_count(entry):
    """Retained streamline count from the .trk header, else *_params.json."""
    trk = entry.get("trk")
    if trk and os.path.exists(trk):
        try:
            tg = nib.streamlines.load(trk, lazy_load=True)
            n = tg.header.get("nb_streamlines")
            if n:
                return int(n)
        except Exception:
            pass
    params = entry.get("params")
    if params and os.path.exists(params):
        with open(params) as f:
            res = json.load(f).get("results", {})
        return int(res.get("total_streamlines", res.get("valid_streamlines", 0)) or 0)
    return 0


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


def summ(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def cv(summary):
    """Coefficient of variation (sd / mean) from a summ() dict, as a fraction."""
    m, s = summary.get("mean"), summary.get("std")
    if not m or m == 0 or s is None:
        return None
    return float(s / m)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "[n/a]"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# --------------------------------------------------------------- analysis ----
def analyse():
    out = {}
    raw = {}
    T = DENSITY_THRESHOLD

    def per_group(entries):
        vols, counts, dens = [], [], []
        for e in entries:
            arr, aff = load_density(e["visit_map"])
            if arr is None:
                continue
            vv = voxel_volume_mm3(aff) if aff is not None else VOXEL_VOLUME_MM3
            vols.append(binarise(arr, T).sum() * vv)
            counts.append(streamline_count(e))
            dens.append(arr)
        return vols, counts, dens

    hyb_vols, hyb_counts, hyb_dens = per_group(HYBRID_RUNS)
    bas_vols, bas_counts, bas_dens = per_group(BASELINE_RUNS)
    print(f"  hybrid: {len(hyb_dens)}/{len(HYBRID_RUNS)} densities loaded; "
          f"baseline: {len(bas_dens)}/{len(BASELINE_RUNS)}")

    out["hybrid"] = {"volume_mm3": summ(hyb_vols), "streamline_count": summ(hyb_counts)}
    out["baseline"] = {"volume_mm3": summ(bas_vols), "streamline_count": summ(bas_counts)}

    # Reproducibility: relative spread of each metric.
    out["coefficient_of_variation"] = {
        "hybrid": {"volume_mm3": cv(out["hybrid"]["volume_mm3"]),
                   "streamline_count": cv(out["hybrid"]["streamline_count"])},
        "baseline": {"volume_mm3": cv(out["baseline"]["volume_mm3"]),
                     "streamline_count": cv(out["baseline"]["streamline_count"])},
    }

    # Keep the per-run values so the figures (and later re-plots) have the data.
    raw["hybrid_volume_mm3"] = [float(v) for v in hyb_vols]
    raw["hybrid_streamline_count"] = [int(c) for c in hyb_counts]
    raw["baseline_volume_mm3"] = [float(v) for v in bas_vols]
    raw["baseline_streamline_count"] = [int(c) for c in bas_counts]

    hd, _ = load_density(HYBRID_DEFAULT_VISIT_MAP)
    bd, _ = load_density(BASELINE_DEFAULT_VISIT_MAP)
    if hd is not None and bd is not None:
        out["overlap_default"] = {"dice": dice(binarise(hd, T), binarise(bd, T)),
                                  "jaccard": jaccard(binarise(hd, T), binarise(bd, T))}
        raw["default_dice"] = out["overlap_default"]["dice"]

    # Full pairwise Dice matrix (rows = hybrid run, cols = baseline run).
    if hyb_dens and bas_dens:
        mat = np.array([[dice(binarise(h, T), binarise(b, T)) for b in bas_dens]
                        for h in hyb_dens], dtype=float)
        out["overlap_all_pairs"] = {"dice": summ(mat[np.isfinite(mat)].tolist())}
        raw["pairwise_dice_matrix"] = mat.tolist()

    return out, raw


# ----------------------------------------------------------------- figures ----
# def figure_volume_count(raw, path):
#     """Boxplots overlaid with a swarmplot to show the true per-run distributions."""
#     rows = []
#     for group, label in (("hybrid", "Hybrid"), ("baseline", "Baseline")):
#         for metric in ("volume_mm3", "streamline_count"):
#             for v in raw.get(f"{group}_{metric}", []):
#                 rows.append({"Group": label, "Metric": metric, "Value": v})
#     df = pd.DataFrame(rows)
#     if df.empty:
#         return

#     palette = {"Hybrid": HYB_COLOR, "Baseline": BAS_COLOR}
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
#     panels = [
#         (axes[0, 0], "Hybrid", "volume_mm3", "Hybrid Tract volume", "mm$^3$"),
#         (axes[0, 1], "Baseline", "volume_mm3", "Diffusion-only Tract volume", "mm$^3$"),
#         (axes[1, 0], "Hybrid", "streamline_count", "Hybrid Retained streamlines", "count"),
#         (axes[1, 1], "Baseline", "streamline_count", "Diffusion-only Retained streamlines", "count"),
#     ]
    
#     for ax, group, metric, title, unit in panels:
#         sub = df[(df["Metric"] == metric) & (df["Group"] == group)]
#         if not sub.empty:
#             sns.boxplot(data=sub, x="Group", y="Value", hue="Group", ax=ax,
#                         palette=palette, width=0.5, legend=False,
#                         boxprops={"alpha": 0.4}, showfliers=False)
#             sns.swarmplot(data=sub, x="Group", y="Value", ax=ax,
#                           color="darkblue", alpha=1, size=5)
#         ax.set_title(title, fontweight="bold")
#         ax.set_ylabel(unit,fontweight="bold")
#         ax.set_xlabel("",fontweight="bold")
#         ax.set_xticks([],fontweight="bold")  # Redundant with titles
    
#     fig.suptitle("Hybrid vs Diffusion-only distributions", fontweight="bold", y=1.02)
#     fig.tight_layout()
#     fig.savefig(path, dpi=800, bbox_inches="tight")
#     plt.close(fig)


def figure_dice_heatmap(raw, path):
    """10x10 pairwise Dice matrix."""
    mat = raw.get("pairwise_dice_matrix")
    if not mat:
        return
    M = np.asarray(mat, dtype=float)
    nh, nb = M.shape

    # Adjusted figsize for a single plot (width reduced from 10 to 6.5)
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)

    # Heatmap
    sns.heatmap(
        M, ax=ax, cmap="viridis",
        annot=(nh <= 12 and nb <= 12), fmt=".2f", annot_kws={"size": 6.5},
        cbar_kws={"label": "Dice"},
        xticklabels=range(1, nb + 1), yticklabels=range(1, nh + 1),
    )
    
    ax.set_xlabel("Diffusion-only run")
    ax.set_ylabel("Hybrid run")
    ax.set_title("Pairwise Dice (Hybrid $\\times$ Diffusion-only)",fontweight="bold")
    ax.tick_params(labelsize=8)

    # fig.suptitle("Reproducibility of hybrid-vs-baseline overlap", fontweight="bold")
    fig.savefig(path, dpi=800, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------------- reporting --
def markdown(result):
    L = ["## Hybrid vs baseline FMA tractography", ""]
    h, b, od = result.get("hybrid", {}), result.get("baseline", {}), result.get("overlap_default", {})
    cvd = result.get("coefficient_of_variation", {})
    if h:
        L.append(f"- HYBRID VOLUME = {fmt(h['volume_mm3']['mean'], 0)} mm^3 "
                 f"(mean over {h['volume_mm3']['n']} runs, sd {fmt(h['volume_mm3']['std'], 0)}"
                 f", CV {fmt((cvd.get('hybrid', {}).get('volume_mm3') or 0) * 100, 1)}%)")
        L.append(f"- HYBRID STREAMLINE COUNT = {fmt(h['streamline_count']['mean'], 0)} "
                 f"(sd {fmt(h['streamline_count']['std'], 0)}"
                 f", CV {fmt((cvd.get('hybrid', {}).get('streamline_count') or 0) * 100, 1)}%)")
    if b:
        L.append(f"- BASELINE VOLUME = {fmt(b['volume_mm3']['mean'], 0)} mm^3 "
                 f"(sd {fmt(b['volume_mm3']['std'], 0)}"
                 f", CV {fmt((cvd.get('baseline', {}).get('volume_mm3') or 0) * 100, 1)}%)")
        L.append(f"- BASELINE STREAMLINE COUNT = {fmt(b['streamline_count']['mean'], 0)} "
                 f"(sd {fmt(b['streamline_count']['std'], 0)}"
                 f", CV {fmt((cvd.get('baseline', {}).get('streamline_count') or 0) * 100, 1)}%)")
    if od:
        L.append(f"- DICE (hybrid vs baseline, default run) = {fmt(od.get('dice'))}")
        L.append(f"- JACCARD = {fmt(od.get('jaccard'))}")
    if "overlap_all_pairs" in result:
        ap = result["overlap_all_pairs"]["dice"]
        L.append(f"  - (mean Dice over all {ap['n']} hybrid x baseline pairs = "
                 f"{fmt(ap['mean'])} +/- {fmt(ap['std'])})")
    L.append("")
    return L


FIGURES = [
    ("fig_volume_count.png",    lambda res, raw, p: figure_volume_count(raw, p)),
    ("fig_dice_heatmap.png",    lambda res, raw, p: figure_dice_heatmap(raw, p)),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    print("=== volume + overlap ===")
    result, raw = analyse()

    out_json = dict(result)
    out_json["raw"] = raw
    with open(os.path.join(OUTPUT_DIR, "volume_overlap.json"), "w") as f:
        json.dump(out_json, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "volume_overlap.md"), "w") as f:
        f.write("\n".join(markdown(result)) + "\n")

    for name, fn in FIGURES:
        try:
            fn(result, raw, os.path.join(OUTPUT_DIR, name))
            print(f"  wrote {name}")
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")

    print(f"  wrote volume_overlap.json/.md to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()