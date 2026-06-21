"""
divergence.py -- Population-selection divergence.
=================================================

Self-contained. Edit the CONFIG block with the EXACT path to every file, run:

    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    $PY TractographyAlgo/scripts/results/divergence.py

Quantifies how often the hybrid PS-OCT scorer picks a different fibre
population than the trajectory-only rule:
  * aggregate divergence fraction (pooled + per-run), and
  * a divergence-rate TREND binned by fibre fraction -- separately for the
    second (f2) and third (f3) bedpostX fibre populations -- using the
    outlier-robust POOLED rate per bin, sum(divergence)/sum(visits), so a few
    low-traffic voxels can't dominate (the failure mode of a per-voxel mean).
  * a pooled crossing- vs single-fibre summary at F_MIN, with a Mann-Whitney U
    test, kept for the existing results sentence.

Writes divergence.json, divergence.md, fig_divergence_trend.png to OUTPUT_DIR.
"""

import os
import json

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG  -- put the EXACT path to every file.
# =============================================================================
OUTPUT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/results/divergence"

# crossing-fibre criterion for the (secondary) pooled crossing-vs-single
# summary: a voxel is "crossing" if mean f2 >= F_MIN.
F_MIN = 0.3

# Bin edges for the divergence-rate trend (used for BOTH f2 and f3). Voxels are
# grouped by their mean fibre fraction into [edge_i, edge_{i+1}); the last bin is
# closed. f3 is typically small, so its upper bins will be sparse -- that's fine.
BIN_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

# Bins with fewer than this many visited voxels are dropped from the FIGURE line
# (too noisy to plot). They are still written to the JSON/markdown, flagged
# "(sparse)". Set to 0 to plot every bin.
MIN_VOXELS_PER_BIN = 20

# --- Hybrid runs for the aggregate divergence fraction. Each needs:
#       divergence -> *_divergence_counts.nii.gz  (or the *_stats.npz holding
#                     a "divergence_map"), and
#       params     -> *_params.json  (for results.psoct_constrained_steps).
HYBRID_RUNS = [
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_01/hybrid1_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_01/hybrid1_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_02/hybrid2_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_02/hybrid2_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_04/hybrid4_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_04/hybrid4_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_05/hybrid5_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_05/hybrid5_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_06/hybrid6_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_06/hybrid6_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_07/hybrid7_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_07/hybrid7_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_08/hybrid8_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_08/hybrid8_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_09/hybrid9_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_09/hybrid9_params.json"},
    {"divergence": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_10/hybrid10_divergence_counts.nii.gz",
     "params":     "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_10/hybrid10_params.json"},
]

# --- Representative run for the crossing-vs-single split.
REPRESENTATIVE_DIVERGENCE = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_divergence_counts.nii.gz"
REPRESENTATIVE_VISIT_MAP  = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz"

# bedpostX mean f2 / f3 (second / third fibre fraction) -- the trend x-axis.
MEAN_F2SAMPLES = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP/mean_f2samples.nii.gz"
MEAN_F3SAMPLES = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP/mean_f3samples.nii.gz"
# =============================================================================


# ---------------------------------------------------------------- helpers ----
def load_map(path, npz_key):
    """Load a 3D array from a NIfTI, or from `npz_key` inside a *_stats.npz."""
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        if npz_key in d.files and d[npz_key] is not None:
            return np.asarray(d[npz_key], dtype=np.float64)
        return None
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float64)


def psoct_constrained_steps(params_path):
    if not params_path or not os.path.exists(params_path):
        return None
    with open(params_path) as f:
        return json.load(f).get("results", {}).get("psoct_constrained_steps")


def summ(vals):
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


# ----------------------------------------------------------- trend helpers ----
def pooled_rate(dm, vis, mask):
    """sum(divergence) / sum(visits) over the masked voxels (NaN if no visits)."""
    v = float(vis[mask].sum())
    return float(dm[mask].sum() / v) if v > 0 else float("nan")


def binned_trend(dm, vis, frac, visited):
    """Per-bin pooled + median divergence rate, grouped by fibre fraction `frac`.

    Pooled rate = sum(divergence)/sum(visits) in the bin (outlier-robust). The
    per-voxel median is reported alongside as a distribution summary.
    """
    edges = BIN_EDGES
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = (i == len(edges) - 2)
        m = visited & (frac >= lo) & ((frac <= hi) if last else (frac < hi))
        n = int(m.sum())
        if n > 0 and vis[m].sum() > 0:
            per_vox = dm[m] / vis[m]
            median = float(np.median(per_vox))
            pr = pooled_rate(dm, vis, m)
        else:
            median, pr = float("nan"), float("nan")
        rows.append({"lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                     "n_voxels": n, "pooled_rate": pr, "median_rate": median})
    return rows


# --------------------------------------------------------------- analysis ----
def analyse():
    """Returns (result_dict, plotdata) where plotdata feeds the trend figure."""
    from scipy.stats import mannwhitneyu
    out = {}

    # 1) Aggregate fraction of PS-OCT-constrained steps that diverged (solid).
    div_sums, psoct_steps, fracs = [], [], []
    for e in HYBRID_RUNS:
        dm = load_map(e["divergence"], "divergence_map")
        ps = psoct_constrained_steps(e["params"])
        if dm is None or not ps:
            continue
        ds = float(dm.sum())
        div_sums.append(ds); psoct_steps.append(ps); fracs.append(100.0 * ds / ps)
    print(f"  aggregate fraction: {len(div_sums)}/{len(HYBRID_RUNS)} runs usable")
    if div_sums:
        out["divergence_fraction_pct"] = {
            "pooled": 100.0 * sum(div_sums) / sum(psoct_steps),
            "per_run": summ(fracs),
            "definition": "100 * sum(divergence_map) / psoct_constrained_steps",
        }

    # 2) Divergence-rate trend binned by f2 and f3 on the representative run.
    plotdata = None
    dm = load_map(REPRESENTATIVE_DIVERGENCE, "divergence_map")
    vis = load_map(REPRESENTATIVE_VISIT_MAP, "visit_map")
    f2 = load_map(MEAN_F2SAMPLES, None)
    f3 = load_map(MEAN_F3SAMPLES, None)

    if dm is None or vis is None or f2 is None or dm.shape != f2.shape:
        out["trend"] = {"skipped":
            "needs representative divergence map + visit map + mean_f2 on a matching grid"}
        return out, plotdata

    visited = vis > 0
    trend = {
        "definition": "pooled rate = sum(divergence_counts)/sum(visit_count) over "
                      "visited voxels in each fibre-fraction bin (outlier-robust); "
                      "median = median per-voxel divergence_counts/visit_count",
        "bin_edges": list(BIN_EDGES),
        "f2": binned_trend(dm, vis, f2, visited),
    }
    if f3 is not None and f3.shape == dm.shape:
        trend["f3"] = binned_trend(dm, vis, f3, visited)
    else:
        trend["f3"] = None
        print("  note: mean_f3 missing/mismatched -- f3 trend skipped")
    out["trend"] = trend
    plotdata = trend

    # 3) Pooled crossing-vs-single summary at F_MIN (robust; for the sentence).
    crossing = visited & (f2 >= F_MIN)
    single = visited & (f2 < F_MIN)
    cs = {
        "definition": "pooled rate = sum(divergence)/sum(visits) within group; "
                      "Mann-Whitney U on the per-voxel rates",
        "f_min": F_MIN,
        "crossing": {"n_voxels": int(crossing.sum()),
                     "pooled_rate": pooled_rate(dm, vis, crossing),
                     "median_rate": float(np.median(dm[crossing] / vis[crossing])) if crossing.sum() else float("nan")},
        "single": {"n_voxels": int(single.sum()),
                   "pooled_rate": pooled_rate(dm, vis, single),
                   "median_rate": float(np.median(dm[single] / vis[single])) if single.sum() else float("nan")},
    }
    if crossing.sum() and single.sum():
        u, p = mannwhitneyu(dm[crossing] / vis[crossing], dm[single] / vis[single],
                            alternative="two-sided")
        cs["mannwhitney_u"] = float(u); cs["p_value"] = float(p)
    out["crossing_vs_single"] = cs
    return out, plotdata


def figure(plotdata, path):
    """Pooled divergence rate vs fibre fraction, overlaying f2 and f3.

    Marker area is scaled by the voxel count in each bin, so sparse bins are
    visually de-emphasised. Bins with no visited voxels are dropped.
    """
    if not plotdata or not plotdata.get("f2"):
        return

    def series(rows):
        rows = [r for r in (rows or []) if r["n_voxels"] >= MIN_VOXELS_PER_BIN
                and not np.isnan(r["pooled_rate"])]
        x = np.array([r["center"] for r in rows])
        y = np.array([r["pooled_rate"] for r in rows])
        n = np.array([r["n_voxels"] for r in rows], dtype=float)
        return x, y, n

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    nmax = 1.0
    for rows in (plotdata.get("f2"), plotdata.get("f3")):
        _, _, n = series(rows)
        if n.size:
            nmax = max(nmax, n.max())

    def add(rows, color, label, marker):
        x, y, n = series(rows)
        if not x.size:
            return
        ax.plot(x, y, "-", color=color, lw=1.6, alpha=0.9, zorder=1)
        sizes = 40 + 360 * (n / nmax)
        ax.scatter(x, y, s=sizes, color=color, marker=marker, edgecolor="black",
                   linewidth=0.6, label=label, zorder=2)

    add(plotdata.get("f2"), "#3b6fb0", r"$\bar f_2$ (2nd fibre)", "o")
    add(plotdata.get("f3"), "#d9822b", r"$\bar f_3$ (3rd fibre)", "s")

    ax.set_xlabel(r"mean fibre fraction $\bar f_k$")
    ax.set_ylabel("pooled divergence rate\n(divergence steps per streamline-pass)")
    ax.grid(alpha=0.3)
    ax.legend(title="marker area $\\propto$ #voxels", fontsize=9)
    ax.set_title("Population-selection divergence vs fibre fraction", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def markdown(result):
    L = ["## Population-selection divergence", ""]
    if "divergence_fraction_pct" in result:
        d = result["divergence_fraction_pct"]
        L.append(f"- DIVERGENCE FRACTION = {fmt(d['pooled'], 2)} % (pooled; {d['definition']})")

    tr = result.get("trend", {})
    if "skipped" in tr:
        L.append(f"- trend: skipped ({tr['skipped']})")
    else:
        for key, name in (("f2", "f2"), ("f3", "f3")):
            rows = tr.get(key)
            if not rows:
                continue
            L.append(f"- Divergence-rate trend vs {name} "
                     "(bin: pooled_rate [median], n voxels):")
            for r in rows:
                if r["n_voxels"] > 0 and not (isinstance(r["pooled_rate"], float) and np.isnan(r["pooled_rate"])):
                    sparse = " (sparse)" if r["n_voxels"] < MIN_VOXELS_PER_BIN else ""
                    L.append(f"  - [{r['lo']:.2f},{r['hi']:.2f}): "
                             f"{fmt(r['pooled_rate'])} [{fmt(r['median_rate'])}], "
                             f"n={r['n_voxels']}{sparse}")

    cs = result.get("crossing_vs_single", {})
    if "crossing" in cs:
        L.append(f"- CROSSING DIVERGENCE (pooled, f2>={cs['f_min']}) = "
                 f"{fmt(cs['crossing']['pooled_rate'])} "
                 f"(n={cs['crossing']['n_voxels']} voxels)")
        L.append(f"- SINGLE-FIBRE DIVERGENCE (pooled) = "
                 f"{fmt(cs['single']['pooled_rate'])} "
                 f"(n={cs['single']['n_voxels']} voxels)")
        if "p_value" in cs:
            L.append(f"- STATISTIC = Mann-Whitney U={fmt(cs['mannwhitney_u'], 0)}, "
                     f"p={cs['p_value']:.2e}")
    L.append("")
    return L


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== divergence ===")
    result, plotdata = analyse()
    with open(os.path.join(OUTPUT_DIR, "divergence.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "divergence.md"), "w") as f:
        f.write("\n".join(markdown(result)) + "\n")
    figure(plotdata, os.path.join(OUTPUT_DIR, "fig_divergence_trend.png"))
    print(f"  wrote divergence.json/.md and fig_divergence_trend.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
