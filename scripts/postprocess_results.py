"""
Post-processing for the hybrid-vs-baseline FMA tractography results.
=====================================================================

Consumes the per-run output folders written by run_tractography.py
(``*_visit_map.nii.gz``, ``*_divergence_counts.nii.gz``, ``*_params.json``,
``*_stats.npz``, ``*.trk``) and computes every *numeric* result in the
methodology, plus the statistics-based figures:

  * Hybrid vs baseline: volumes, streamline counts, Dice/Jaccard, probtrackx2.
  * Population-selection divergence: aggregate fraction + crossing/single split.
  * Sensitivity: alpha (blend) sweep, theta sweep, in-plane-gate sweep, and the
    PS-OCT density-thinning dose-response.
  * Reproducibility: pairwise FTC at T>=1 and T>=5 (hybrid and baseline).
  * Anatomical reference overlap (Dice/Jaccard) -- needs --reference.
  * Held-out interleaved validation: median acute in-plane angle between
    streamline tangents and the held-out odd-slice optic axis -- needs
    --held-out-slides (re-reads PS-OCT through PSOCTData; heavy deps imported
    lazily so the rest of the script runs without fsl/cmc_hybrid).

It does NOT make the spatial/glass-brain overlays (those are done in FSLeyes).
Missing folders / unset reference / unset held-out slides are skipped with a
note, never fatal, so it is safe to run mid-way through generating the runs.

Usage
-----
    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    PYTHONPATH=TractographyAlgo/cmc_hybrid:TractographyAlgo \
        $PY TractographyAlgo/scripts/postprocess_results.py \
            [--reference /path/to/FMA_reference_native.nii.gz] \
            [--held-out-slides /path/to/odd_slices_dir] \
            [--trk-outputs /path/to/TRK_outputs]

Edit the CONFIG dict below to point at your folders / set the sweep run names.

Outputs (under <trk_outputs>/_postprocessing/):
    results.json            -- every computed number, machine-readable
    results_filled.md       -- the methodology placeholders, filled in
    fig_volume_count.png       fig_reproducibility_ftc.png
    fig_sensitivity_alpha.png  fig_sensitivity_theta.png
    fig_sensitivity_density.png  fig_divergence.png  fig_held_out.png
"""

import os
import sys
import json
import argparse
from glob import glob
from itertools import combinations
from datetime import datetime

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.dirname(SCRIPT_DIR)                       # .../TractographyAlgo
PROJECT_DIR = os.path.dirname(ALGO_DIR)                      # .../MEng Individual project

# =============================================================================
# CONFIG  -- edit these, then run.
# =============================================================================
CONFIG = {
    "trk_outputs": os.path.join(ALGO_DIR, "tractography", "TRK_outputs"),
    "bedpostx_dir": os.path.join(PROJECT_DIR, "DATA", "data.bedpostX_SSFP_uncompressed"),

    # XTRACT standardised FMA reference, already warped into native dMRI space.
    # Leave None to skip every "overlap with reference" number (run it later
    # once the reference is warped).
    "reference_mask": None,

    "f_min": 0.05,            # crossing-fibre criterion: mean f2 >= f_min
    "density_threshold": 1,   # binarise visit maps at >= this many visits
    "ftc_thresholds": [1, 5], # FTC(T): Tanimoto of maps thresholded at >= T visits

    # Logical run groups -> folder name(s) under trk_outputs/.
    # Phase 1 (hybrid) and Phase 2 (baseline) repeats:
    "hybrid_repeats":   [f"hybrid_run_{i:02d}"   for i in range(1, 11)],
    "baseline_repeats": [f"baseline_run_{i:02d}" for i in range(1, 11)],

    # Representative single runs used as the "default" point of each sweep
    # (theta=60, thresh=0.3, full density). Override if you prefer another.
    "hybrid_default":   "hybrid_run_01",
    "baseline_default": "baseline_run_01",

    # Phase 3 sweeps: parameter value -> folder name.
    "theta_sweep":  {45: "sweep_theta_45", 60: "hybrid_run_01", 80: "sweep_theta_80"},
    "thresh_sweep": {1.0: "sweep_thresh_100", 0.5: "sweep_thresh_050",
                     0.3: "hybrid_run_01", 0.1: "sweep_thresh_010"},
    "density_sweep": {"full (every slice)": "hybrid_run_01",
                      "every 2nd": "density_every_2nd",
                      "every 4th": "density_every_4th"},

    # Alpha (blend coefficient) sweep -- value -> folder. Not in the original
    # 29-run plan; add ALPHA=0,0.25,...,1.0 runs (alpha is now a parameter) and
    # name the folders to match. alpha=1 should reproduce the baseline.
    "alpha_sweep": {0.0: "sweep_alpha_00", 0.25: "sweep_alpha_025",
                    0.5: "hybrid_run_01", 0.75: "sweep_alpha_075",
                    1.0: "sweep_alpha_10"},

    # Phase 4 sanity: probtrackx2 fdt_paths.nii.gz (run 29), optional.
    "probtrackx2_density": None,

    # Phase 4 held-out interleaved validation (run 28): tract tracked on EVEN
    # slices, validated against the held-out ODD optic-axis orientations.
    "held_out_run": "held_out_even",
    "held_out_psoct_slides": None,    # dir of held-out (odd) PS-OCT slides; set to enable
    "held_out_slice_parity": None,    # None | 'odd' | 'even': subsample a full dir by index
    "held_out_vertex_stride": 1,      # sample every Nth streamline vertex (speed knob)
    "held_out_max_streamlines": None, # cap streamlines processed (None = all)

    "output_subdir": "_postprocessing",
}


# =============================================================================
# Small IO / geometry helpers
# =============================================================================
def _resolve_folder(trk_outputs, name):
    """Return the absolute path to a run folder, or None if it isn't there yet."""
    if name is None:
        return None
    path = os.path.join(trk_outputs, name)
    return path if os.path.isdir(path) else None


def _find(folder, suffix):
    """First file in `folder` ending with `suffix`, or None."""
    hits = sorted(glob(os.path.join(folder, f"*{suffix}")))
    return hits[0] if hits else None


def load_density(folder):
    """Load a run's per-voxel visit map (un-normalised counts) + affine.

    Prefers the saved ``*_visit_map.nii.gz``; falls back to the ``visit_map``
    array inside ``*_stats.npz`` (which needs the bedpostx affine, loaded by
    the caller). Returns (array_float, affine) or (None, None).
    """
    nii = _find(folder, "_visit_map.nii.gz")
    if nii is not None:
        img = nib.load(nii)
        return np.asarray(img.get_fdata(), dtype=np.float64), img.affine
    npz = _find(folder, "_stats.npz")
    if npz is not None:
        d = np.load(npz, allow_pickle=True)
        if "visit_map" in d.files and d["visit_map"] is not None:
            return np.asarray(d["visit_map"], dtype=np.float64), None
    return None, None


def load_divergence(folder):
    """Per-voxel divergence-count map (counts), from NIfTI or stats.npz."""
    nii = _find(folder, "_divergence_counts.nii.gz")
    if nii is not None:
        return np.asarray(nib.load(nii).get_fdata(), dtype=np.float64)
    npz = _find(folder, "_stats.npz")
    if npz is not None:
        d = np.load(npz, allow_pickle=True)
        if "divergence_map" in d.files and d["divergence_map"] is not None:
            return np.asarray(d["divergence_map"], dtype=np.float64)
    return None


def load_params(folder):
    p = _find(folder, "_params.json")
    if p is None:
        return {}
    with open(p) as f:
        return json.load(f)


def streamline_count(folder):
    """Retained streamline count: prefer the .trk header, fall back to params."""
    trk = _find(folder, ".trk")
    if trk is not None:
        try:
            tg = nib.streamlines.load(trk, lazy_load=True)
            n = tg.header.get("nb_streamlines")
            if n:
                return int(n)
        except Exception:
            pass
    res = load_params(folder).get("results", {})
    # In run_tractography.py the retained streamlines are saved as
    # results.total_streamlines (= len(streamlines)).
    return int(res.get("total_streamlines", res.get("valid_streamlines", 0)) or 0)


def voxel_volume_mm3(affine):
    return float(abs(np.linalg.det(affine[:3, :3])))


# =============================================================================
# Overlap metrics
# =============================================================================
def binarise(arr, thresh):
    return arr >= thresh


def dice(a_bin, b_bin):
    a = a_bin.astype(bool); b = b_bin.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def jaccard(a_bin, b_bin):
    a = a_bin.astype(bool); b = b_bin.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def fractional_tanimoto_continuous(A, B):
    """Crum 2006 fuzzy Tanimoto on densities normalised to [0, 1]."""
    A = A / A.max() if A.max() > 0 else A
    B = B / B.max() if B.max() > 0 else B
    mx = np.maximum(A, B).sum()
    if mx <= 0:
        return float("nan")
    return float(np.minimum(A, B).sum() / mx)


def ftc_threshold(A, B, T):
    """FTC at density threshold T: Tanimoto of binarised maps = Jaccard(>=T)."""
    return jaccard(binarise(A, T), binarise(B, T))


def _summ(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


# =============================================================================
# Analyses
# =============================================================================
def analyse_volume_and_overlap(ctx):
    """Volumes, streamline counts, and hybrid-vs-baseline Dice/Jaccard."""
    out = {}
    T = ctx["density_threshold"]

    hyb = ctx["resolve_group"]("hybrid_repeats")
    bas = ctx["resolve_group"]("baseline_repeats")

    def per_group(folders):
        vols, counts, dens = [], [], []
        for f in folders:
            arr, aff = load_density(f)
            if arr is None:
                continue
            vv = voxel_volume_mm3(aff) if aff is not None else ctx["voxel_vol"]
            vols.append(binarise(arr, T).sum() * vv)
            counts.append(streamline_count(f))
            dens.append(arr)
        return vols, counts, dens

    hyb_vols, hyb_counts, hyb_dens = per_group(hyb)
    bas_vols, bas_counts, bas_dens = per_group(bas)

    out["hybrid"] = {"volume_mm3": _summ(hyb_vols),
                     "streamline_count": _summ(hyb_counts)}
    out["baseline"] = {"volume_mm3": _summ(bas_vols),
                       "streamline_count": _summ(bas_counts)}

    # Headline overlap: representative run01 vs run01.
    hd, _ = load_density(ctx["hybrid_default_path"]) if ctx["hybrid_default_path"] else (None, None)
    bd, _ = load_density(ctx["baseline_default_path"]) if ctx["baseline_default_path"] else (None, None)
    if hd is not None and bd is not None:
        out["overlap_default"] = {
            "dice": dice(binarise(hd, T), binarise(bd, T)),
            "jaccard": jaccard(binarise(hd, T), binarise(bd, T)),
        }

    # Robustness: mean pairwise Dice over all hybrid x baseline pairs.
    if hyb_dens and bas_dens:
        pair_dice = [dice(binarise(h, T), binarise(b, T))
                     for h in hyb_dens for b in bas_dens]
        out["overlap_all_pairs"] = {"dice": _summ(pair_dice)}

    # probtrackx2 sanity vs baseline.
    if ctx["probtrackx2_density"] and os.path.exists(ctx["probtrackx2_density"]) and bd is not None:
        pd = np.asarray(nib.load(ctx["probtrackx2_density"]).get_fdata(), dtype=np.float64)
        if pd.shape == bd.shape:
            out["probtrackx2_vs_baseline"] = {"dice": dice(binarise(pd, T), binarise(bd, T))}
        else:
            out["probtrackx2_vs_baseline"] = {"error": f"shape {pd.shape} != {bd.shape}"}

    ctx["_hyb_dens"] = hyb_dens
    ctx["_bas_dens"] = bas_dens
    return out


def analyse_divergence(ctx):
    """Population-selection divergence: aggregate fraction + crossing/single split."""
    from scipy.stats import mannwhitneyu

    out = {}
    folders = ctx["resolve_group"]("hybrid_repeats")

    # Aggregate fraction of PS-OCT-constrained steps where the hybrid scorer
    # picked a different population than the trajectory-only rule.
    div_sums, psoct_steps, fracs = [], [], []
    for f in folders:
        dm = load_divergence(f)
        res = load_params(f).get("results", {})
        ps = res.get("psoct_constrained_steps")
        if dm is None or not ps:
            continue
        ds = float(dm.sum())
        div_sums.append(ds); psoct_steps.append(ps)
        fracs.append(100.0 * ds / ps)
    if div_sums:
        out["divergence_fraction_pct"] = {
            "pooled": 100.0 * sum(div_sums) / sum(psoct_steps),
            "per_run": _summ(fracs),
            "definition": "100 * sum(divergence_map) / psoct_constrained_steps",
        }

    # Crossing- vs single-fibre split on the representative run.
    rep = ctx["hybrid_default_path"]
    dm = load_divergence(rep) if rep else None
    vis, _ = load_density(rep) if rep else (None, None)
    f2 = ctx["mean_f2"]
    if dm is not None and vis is not None and f2 is not None and dm.shape == f2.shape:
        visited = vis > 0
        rate = np.zeros_like(vis)
        rate[visited] = dm[visited] / vis[visited]   # divergence steps per streamline-pass
        crossing = visited & (f2 >= ctx["f_min"])
        single = visited & (f2 < ctx["f_min"])
        cr = rate[crossing] * 100.0
        sr = rate[single] * 100.0
        entry = {
            "definition": "per-voxel rate = divergence_counts / visit_count, over visited voxels",
            "crossing_pct": _summ(cr.tolist()),
            "single_pct": _summ(sr.tolist()),
            "f_min": ctx["f_min"],
        }
        if cr.size > 0 and sr.size > 0:
            u, p = mannwhitneyu(cr, sr, alternative="two-sided")
            entry["mannwhitney_u"] = float(u)
            entry["p_value"] = float(p)
        out["crossing_vs_single"] = entry
        ctx["_div_rates"] = {"crossing": cr, "single": sr}
    else:
        out["crossing_vs_single"] = {"skipped":
            "needs representative divergence map + visit map + mean_f2 on matching grid"}
    return out


def _sweep_table(ctx, sweep, ref_bin):
    """For a {value: folder} sweep, compute volume, Dice-vs-baseline, Dice-vs-ref."""
    T = ctx["density_threshold"]
    bd, _ = load_density(ctx["baseline_default_path"]) if ctx["baseline_default_path"] else (None, None)
    bd_bin = binarise(bd, T) if bd is not None else None
    rows = []
    for val, name in sweep.items():
        folder = _resolve_folder(ctx["trk_outputs"], name)
        if folder is None:
            rows.append({"value": val, "folder": name, "missing": True})
            continue
        arr, aff = load_density(folder)
        if arr is None:
            rows.append({"value": val, "folder": name, "missing": True})
            continue
        vv = voxel_volume_mm3(aff) if aff is not None else ctx["voxel_vol"]
        a_bin = binarise(arr, T)
        row = {"value": val, "folder": name, "volume_mm3": float(a_bin.sum() * vv)}
        if bd_bin is not None:
            row["dice_vs_baseline"] = dice(a_bin, bd_bin)
        if ref_bin is not None and ref_bin.shape == a_bin.shape:
            row["dice_vs_reference"] = dice(a_bin, ref_bin)
        rows.append(row)
    return rows


def _range_pct(rows, key):
    vals = [r[key] for r in rows if key in r]
    if len(vals) < 2 or min(vals) == 0:
        return None
    return 100.0 * (max(vals) - min(vals)) / min(vals)


def _range_of(rows, key):
    """(min, max) of `key` across sweep rows that have it, ignoring None/NaN."""
    vals = [r[key] for r in rows if key in r and r[key] is not None
            and not (isinstance(r[key], float) and np.isnan(r[key]))]
    if not vals:
        return None, None
    return min(vals), max(vals)


def analyse_sensitivity(ctx, ref_bin):
    out = {}
    T = ctx["density_threshold"]

    # --- alpha (blend coefficient) sweep ---
    alpha = _sweep_table(ctx, ctx["alpha_sweep"], ref_bin)
    dice_lo, dice_hi = _range_of(alpha, "dice_vs_baseline")
    a_entry = {"rows": alpha,
               "volume_range_pct": _range_pct(alpha, "volume_mm3"),
               "dice_vs_baseline_lo": dice_lo, "dice_vs_baseline_hi": dice_hi}
    # alpha == 1 should reproduce the baseline (microscopy term vanishes).
    bd, baff = (load_density(ctx["baseline_default_path"])
                if ctx["baseline_default_path"] else (None, None))
    base_vol = None
    if bd is not None:
        vv = voxel_volume_mm3(baff) if baff is not None else ctx["voxel_vol"]
        base_vol = float(binarise(bd, T).sum() * vv)
    for r in alpha:
        if r.get("value") == 1.0 and "volume_mm3" in r:
            a_entry["alpha1_dice_vs_baseline"] = r.get("dice_vs_baseline")
            if base_vol:
                a_entry["alpha1_volume_diff_pct"] = \
                    100.0 * abs(r["volume_mm3"] - base_vol) / base_vol
    out["alpha"] = a_entry

    # --- angle threshold theta sweep ---
    theta = _sweep_table(ctx, ctx["theta_sweep"], ref_bin)
    tref_lo, tref_hi = _range_of(theta, "dice_vs_reference")
    tbase_lo, tbase_hi = _range_of(theta, "dice_vs_baseline")
    out["theta"] = {"rows": theta,
                    "volume_range_pct": _range_pct(theta, "volume_mm3"),
                    "dice_vs_reference_lo": tref_lo, "dice_vs_reference_hi": tref_hi,
                    "dice_vs_baseline_lo": tbase_lo, "dice_vs_baseline_hi": tbase_hi}

    # --- psoct in-plane gate sweep (extra; from the run plan) ---
    thresh = _sweep_table(ctx, ctx["thresh_sweep"], ref_bin)
    out["psoct_inplane_threshold"] = {"rows": thresh,
                                      "volume_range_pct": _range_pct(thresh, "volume_mm3")}

    # --- PS-OCT density thinning dose-response ---
    density = _sweep_table(ctx, ctx["density_sweep"], ref_bin)
    out["density_thinning"] = {"rows": density}
    return out


def analyse_held_out(ctx):
    """Held-out interleaved validation (Eq. angular-error).

    The tract is tracked on EVEN PS-OCT slices; here each streamline tangent is
    compared, in-plane, to the measured optic-axis orientation on the held-out
    ODD slices it passes through. Reports the median acute in-plane angle.
    """
    folder = _resolve_folder(ctx["trk_outputs"], ctx["held_out_run"])
    if folder is None:
        return {"skipped": f"held-out run '{ctx['held_out_run']}' not found"}
    trk = _find(folder, ".trk")
    if trk is None:
        return {"skipped": "no .trk in held-out run folder"}
    slides_dir = ctx.get("held_out_psoct_slides")
    if not slides_dir or not os.path.isdir(slides_dir):
        return {"skipped": "held_out_psoct_slides not set / not a directory"}

    # Heavy deps kept local so the rest of the script runs without fsl/cmc_hybrid.
    try:
        from fsl.data.image import Image
        from tractography.psoct import PSOCTData
    except Exception as e:
        return {"skipped": f"PSOCT deps unavailable: {e}"}

    slides = sorted(glob(os.path.join(slides_dir, "*.nii.gz"))) \
        or sorted(glob(os.path.join(slides_dir, "*.nii")))
    parity = ctx.get("held_out_slice_parity")
    if parity in ("odd", "even") and slides:
        slides = slides[(1 if parity == "odd" else 0)::2]
    if not slides:
        return {"skipped": f"no slides found under {slides_dir}"}

    ps = PSOCTData(slides, Image(ctx["brain_mask_path"]))
    aff_inv = np.linalg.inv(ctx["affine"])
    tg = nib.streamlines.load(trk)
    stride = max(1, int(ctx.get("held_out_vertex_stride", 1) or 1))
    cap = ctx.get("held_out_max_streamlines")

    angles = []
    for s_idx, sl_world in enumerate(tg.streamlines):
        if cap is not None and s_idx >= cap:
            break
        if len(sl_world) < 3:
            continue
        h = np.hstack([sl_world, np.ones((len(sl_world), 1))])
        vox = (aff_inv @ h.T).T[:, :3]
        tan = np.empty_like(vox)
        tan[1:-1] = vox[2:] - vox[:-2]
        tan[0] = vox[1] - vox[0]
        tan[-1] = vox[-1] - vox[-2]
        for i in range(0, len(vox), stride):
            t = tan[i]
            if np.linalg.norm(t) == 0:
                continue
            info = ps.get_orientation(vox[i], return_normal=True)
            if info is None:
                continue
            optic, normal = info
            n = np.asarray(normal, dtype=float)
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
                t_ip = t - np.dot(t, n) * n
                o_ip = np.asarray(optic, dtype=float) - np.dot(optic, n) * n
            else:
                t_ip, o_ip = t, np.asarray(optic, dtype=float)
            tn, on = np.linalg.norm(t_ip), np.linalg.norm(o_ip)
            if tn < 1e-6 or on < 1e-6:
                continue
            cos = abs(float(np.dot(t_ip / tn, o_ip / on)))
            angles.append(float(np.degrees(np.arccos(min(1.0, cos)))))

    if not angles:
        return {"skipped": "no streamline points adjacent to held-out slides"}
    a = np.asarray(angles)
    ctx["_held_out_angles"] = a
    return {
        "n_samples": int(a.size),
        "n_slides": len(slides),
        "median_deg": float(np.median(a)),
        "mean_deg": float(np.mean(a)),
        "p25_deg": float(np.percentile(a, 25)),
        "p75_deg": float(np.percentile(a, 75)),
        "frac_lt20": float(np.mean(a < 20)),
        "frac_lt30": float(np.mean(a < 30)),
        "definition": "acute in-plane angle (deg) between streamline tangent and "
                      "held-out PS-OCT optic axis, projected into the slide plane",
    }


def analyse_reproducibility(ctx):
    out = {}
    for cond in ("hybrid_repeats", "baseline_repeats"):
        dens = []
        for f in ctx["resolve_group"](cond):
            arr, _ = load_density(f)
            if arr is not None:
                dens.append(arr)
        label = "hybrid" if cond.startswith("hybrid") else "baseline"
        if len(dens) < 2:
            out[label] = {"n_runs": len(dens), "note": "need >=2 runs"}
            continue
        pairs = list(combinations(range(len(dens)), 2))
        entry = {"n_runs": len(dens), "n_pairs": len(pairs)}
        for T in ctx["ftc_thresholds"]:
            vals = [ftc_threshold(dens[i], dens[j], T) for i, j in pairs]
            entry[f"ftc_T{T}"] = _summ(vals)
            entry[f"ftc_T{T}_pairs"] = vals
        cont = [fractional_tanimoto_continuous(dens[i], dens[j]) for i, j in pairs]
        entry["ftc_continuous"] = _summ(cont)
        out[label] = entry
    return out


def analyse_reference_overlap(ctx, ref_bin):
    if ref_bin is None:
        return {"skipped": "reference_mask not set"}
    out = {}
    T = ctx["density_threshold"]
    for label, path in (("hybrid", ctx["hybrid_default_path"]),
                        ("baseline", ctx["baseline_default_path"])):
        if not path:
            continue
        arr, _ = load_density(path)
        if arr is None:
            continue
        a_bin = binarise(arr, T)
        if a_bin.shape != ref_bin.shape:
            out[label] = {"error": f"shape {a_bin.shape} != reference {ref_bin.shape}"}
            continue
        out[label] = {"dice": dice(a_bin, ref_bin), "jaccard": jaccard(a_bin, ref_bin)}
    return out


# =============================================================================
# Figures (statistics-based only; spatial overlays are done in FSLeyes)
# =============================================================================
def fig_volume_count(results, path):
    vc = results.get("volume_overlap", {})
    h, b = vc.get("hybrid"), vc.get("baseline")
    if not h or not b:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["Hybrid", "Baseline"]
    for ax, key, title, unit in (
        (axes[0], "volume_mm3", "Tract volume", "mm$^3$"),
        (axes[1], "streamline_count", "Retained streamlines", "count"),
    ):
        means = [h[key]["mean"], b[key]["mean"]]
        errs = [h[key]["std"] or 0, b[key]["std"] or 0]
        means = [m if m is not None else 0 for m in means]
        bars = ax.bar(labels, means, yerr=errs, capsize=6,
                      color=["mediumseagreen", "salmon"], edgecolor="black")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:,.0f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Hybrid vs baseline (mean $\\pm$ sd over repeats)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_reproducibility(results, thresholds, path):
    rp = results.get("reproducibility", {})
    series, labels, colors = [], [], []
    palette = {"hybrid": "mediumseagreen", "baseline": "salmon"}
    for cond in ("hybrid", "baseline"):
        e = rp.get(cond, {})
        for T in thresholds:
            key = f"ftc_T{T}_pairs"
            if key in e and e[key]:
                series.append(e[key]); labels.append(f"{cond}\nT>={T}"); colors.append(palette[cond])
    if not series:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(series) + 2, 5))
    bp = ax.boxplot(series, tick_labels=labels, patch_artist=True, showmeans=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("Fractional Tanimoto Coefficient")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Reproducibility (pairwise FTC across repeats)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_sensitivity_theta(results, path):
    rows = results.get("sensitivity", {}).get("theta", {}).get("rows", [])
    rows = [r for r in rows if "volume_mm3" in r]
    if len(rows) < 2:
        return
    rows = sorted(rows, key=lambda r: r["value"])
    x = [r["value"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, [r["volume_mm3"] for r in rows], "o-", color="steelblue", label="volume")
    ax1.set_xlabel(r"angle threshold $\theta_{max}$ (deg)")
    ax1.set_ylabel("tract volume (mm$^3$)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue"); ax1.grid(alpha=0.3)
    has_d = any("dice_vs_baseline" in r or "dice_vs_reference" in r for r in rows)
    if has_d:
        ax2 = ax1.twinx()
        if all("dice_vs_baseline" in r for r in rows):
            ax2.plot(x, [r["dice_vs_baseline"] for r in rows], "s--", color="darkorange", label="Dice vs baseline")
        if all("dice_vs_reference" in r for r in rows):
            ax2.plot(x, [r["dice_vs_reference"] for r in rows], "^--", color="firebrick", label="Dice vs reference")
        ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1.0)
        ax2.legend(loc="lower right", fontsize=8)
    ax1.set_title(r"Sensitivity to $\theta_{max}$", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_sensitivity_alpha(results, path):
    rows = results.get("sensitivity", {}).get("alpha", {}).get("rows", [])
    rows = [r for r in rows if "volume_mm3" in r]
    if len(rows) < 2:
        return
    rows = sorted(rows, key=lambda r: r["value"])
    x = [r["value"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, [r["volume_mm3"] for r in rows], "o-", color="steelblue", label="volume")
    ax1.set_xlabel(r"blend coefficient $\alpha$  (1 = trajectory only)")
    ax1.set_ylabel("tract volume (mm$^3$)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue"); ax1.grid(alpha=0.3)
    if all("dice_vs_baseline" in r for r in rows):
        ax2 = ax1.twinx()
        ax2.plot(x, [r["dice_vs_baseline"] for r in rows], "s--",
                 color="darkorange", label="Dice vs baseline")
        if all("dice_vs_reference" in r for r in rows):
            ax2.plot(x, [r["dice_vs_reference"] for r in rows], "^--",
                     color="firebrick", label="Dice vs reference")
        ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1.0)
        ax2.legend(loc="lower right", fontsize=8)
    ax1.set_title(r"Sensitivity to blend coefficient $\alpha$", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_sensitivity_density(results, path):
    rows = results.get("sensitivity", {}).get("density_thinning", {}).get("rows", [])
    rows = [r for r in rows if "dice_vs_baseline" in r or "dice_vs_reference" in r]
    if not rows:
        return
    labels = [str(r["folder"] if "value" not in r else r["value"]) for r in rows]
    labels = [str(r.get("value")) for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if all("dice_vs_baseline" in r for r in rows):
        ax.bar(x - w / 2, [r["dice_vs_baseline"] for r in rows], w, label="Dice vs baseline", color="salmon")
    if all("dice_vs_reference" in r for r in rows):
        ax.bar(x + w / 2, [r["dice_vs_reference"] for r in rows], w, label="Dice vs reference", color="firebrick")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.0); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("PS-OCT density dose-response", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_divergence(ctx, path):
    rates = ctx.get("_div_rates")
    if not rates or rates["crossing"].size == 0 or rates["single"].size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot([rates["crossing"], rates["single"]],
                    tick_labels=["crossing\n($\\bar f_2\\geq f_{min}$)", "single-fibre"],
                    patch_artist=True, showmeans=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["mediumpurple", "lightgray"]):
        patch.set_facecolor(c)
    ax.set_ylabel("per-voxel divergence rate (%)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Divergence: crossing vs single-fibre voxels", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_held_out(ctx, results, path):
    a = ctx.get("_held_out_angles")
    if a is None or len(a) == 0:
        return
    med = results.get("held_out", {}).get("median_deg")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(a, bins=np.arange(0, 91, 3), color="#2c7fb8", alpha=0.85)
    if med is not None:
        ax.axvline(med, color="k", ls="--", lw=1.5, label=f"median {med:.1f}$\\degree$")
        ax.legend()
    ax.axvline(45, color="grey", ls=":", lw=1)
    ax.set_xlabel("acute in-plane angle: streamline tangent vs held-out optic axis (deg)")
    ax.set_ylabel("samples")
    ax.set_title("Held-out interleaved validation", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# Methodology-placeholder markdown
# =============================================================================
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "[n/a]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_filled_markdown(results, path):
    L = ["# Filled methodology numbers", "",
         f"_Generated {datetime.now().isoformat(timespec='seconds')}_", ""]

    vo = results.get("volume_overlap", {})
    h, b = vo.get("hybrid", {}), vo.get("baseline", {})
    od = vo.get("overlap_default", {})
    L += ["## Hybrid vs baseline FMA tractography", ""]
    if h:
        L.append(f"- HYBRID VOLUME = {_fmt(h['volume_mm3']['mean'], 0)} mm^3 "
                 f"(mean over {h['volume_mm3']['n']} runs, sd {_fmt(h['volume_mm3']['std'], 0)})")
        L.append(f"- HYBRID STREAMLINE COUNT = {_fmt(h['streamline_count']['mean'], 0)} "
                 f"(sd {_fmt(h['streamline_count']['std'], 0)})")
    if b:
        L.append(f"- BASELINE VOLUME = {_fmt(b['volume_mm3']['mean'], 0)} mm^3 "
                 f"(sd {_fmt(b['volume_mm3']['std'], 0)})")
        L.append(f"- BASELINE STREAMLINE COUNT = {_fmt(b['streamline_count']['mean'], 0)} "
                 f"(sd {_fmt(b['streamline_count']['std'], 0)})")
    if od:
        L.append(f"- DICE (hybrid vs baseline, run01) = {_fmt(od.get('dice'))}")
        L.append(f"- JACCARD = {_fmt(od.get('jaccard'))}")
    if "overlap_all_pairs" in vo:
        ap = vo["overlap_all_pairs"]["dice"]
        L.append(f"  - (mean Dice over all {ap['n']} hybrid x baseline pairs = "
                 f"{_fmt(ap['mean'])} +/- {_fmt(ap['std'])})")
    if "probtrackx2_vs_baseline" in vo:
        L.append(f"- PROBTRACKX2 DICE (vs baseline) = "
                 f"{_fmt(vo['probtrackx2_vs_baseline'].get('dice'))}")
    L.append("")

    dv = results.get("divergence", {})
    L += ["## Population-selection divergence", ""]
    if "divergence_fraction_pct" in dv:
        L.append(f"- DIVERGENCE FRACTION = {_fmt(dv['divergence_fraction_pct']['pooled'], 2)} % "
                 f"(pooled; {dv['divergence_fraction_pct']['definition']})")
    cs = dv.get("crossing_vs_single", {})
    if "crossing_pct" in cs:
        L.append(f"- CROSSING DIVERGENCE = {_fmt(cs['crossing_pct']['mean'], 2)} % "
                 f"(n={cs['crossing_pct']['n']} voxels)")
        L.append(f"- SINGLE-FIBRE DIVERGENCE = {_fmt(cs['single_pct']['mean'], 2)} % "
                 f"(n={cs['single_pct']['n']} voxels)")
        if "p_value" in cs:
            L.append(f"- STATISTIC = Mann-Whitney U={_fmt(cs['mannwhitney_u'], 0)}, "
                     f"p={cs['p_value']:.2e}")
    L.append("")

    se = results.get("sensitivity", {})
    L += ["## Hyperparameter and PS-OCT density sensitivity", ""]

    al = se.get("alpha", {})
    al_rows = [x for x in al.get("rows", []) if "volume_mm3" in x]
    if al_rows:
        L.append(f"- VOLUME RANGE (alpha) = {_fmt(al.get('volume_range_pct'), 1)} %")
        L.append(f"- DICE LO = {_fmt(al.get('dice_vs_baseline_lo'))}, "
                 f"DICE HI = {_fmt(al.get('dice_vs_baseline_hi'))} (vs baseline)")
        if "alpha1_dice_vs_baseline" in al:
            L.append(f"- ALPHA1 DICE = {_fmt(al.get('alpha1_dice_vs_baseline'))}, "
                     f"ALPHA1 VOL DIFF = {_fmt(al.get('alpha1_volume_diff_pct'), 2)} %")
        for r in sorted(al_rows, key=lambda x: x["value"]):
            L.append(f"  - alpha={r['value']}: volume={_fmt(r['volume_mm3'], 0)} mm^3, "
                     f"Dice_vs_baseline={_fmt(r.get('dice_vs_baseline'))}, "
                     f"Dice_vs_ref={_fmt(r.get('dice_vs_reference'))}")
    else:
        L.append("- (alpha sweep: no runs found -- add sweep_alpha_* folders)")

    th = se.get("theta", {})
    if th.get("rows"):
        L.append(f"- THETA VOLUME RANGE = {_fmt(th.get('volume_range_pct'), 1)} %")
        L.append(f"- THETA REF LO = {_fmt(th.get('dice_vs_reference_lo'))}, "
                 f"THETA REF HI = {_fmt(th.get('dice_vs_reference_hi'))}")
        for r in sorted([x for x in th["rows"] if "volume_mm3" in x], key=lambda x: x["value"]):
            L.append(f"  - theta={r['value']}: volume={_fmt(r['volume_mm3'], 0)} mm^3, "
                     f"Dice_vs_baseline={_fmt(r.get('dice_vs_baseline'))}, "
                     f"Dice_vs_ref={_fmt(r.get('dice_vs_reference'))}")
    dt = se.get("density_thinning", {})
    if dt.get("rows"):
        L.append("- Density thinning (THIN DICE / THIN REF):")
        for r in dt["rows"]:
            if "volume_mm3" in r:
                L.append(f"  - {r['value']}: Dice_vs_baseline={_fmt(r.get('dice_vs_baseline'))}, "
                         f"Dice_vs_ref={_fmt(r.get('dice_vs_reference'))}")
    L.append("")

    rp = results.get("reproducibility", {})
    L += ["## Reproducibility (FTC)", ""]
    for cond in ("hybrid", "baseline"):
        e = rp.get(cond, {})
        for T in (1, 5):
            k = f"ftc_T{T}"
            if k in e:
                L.append(f"- FTC {cond.upper()} T{T} = {_fmt(e[k]['mean'])} "
                         f"+/- {_fmt(e[k]['std'])} (n_pairs={e.get('n_pairs')})")
    L.append("")

    ho = results.get("held_out", {})
    L += ["## Held-out interleaved validation", ""]
    if "median_deg" in ho:
        L.append(f"- HELD-OUT ANGLE = {_fmt(ho['median_deg'], 1)} deg "
                 f"(median acute in-plane; n={ho['n_samples']} samples over "
                 f"{ho['n_slides']} held-out slides; <30deg {100*ho['frac_lt30']:.0f}%)")
    else:
        L.append(f"- (skipped: {ho.get('skipped', 'not run')})")
    L.append("")

    ro = results.get("reference_overlap", {})
    L += ["## Anatomical reference overlap", ""]
    if "skipped" in ro:
        L.append(f"- (skipped: {ro['skipped']})")
    else:
        if "hybrid" in ro:
            L.append(f"- HYBRID REF DICE = {_fmt(ro['hybrid'].get('dice'))}")
        if "baseline" in ro:
            L.append(f"- BASELINE REF DICE = {_fmt(ro['baseline'].get('dice'))}")
    L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trk-outputs", default=CONFIG["trk_outputs"])
    ap.add_argument("--reference", default=CONFIG["reference_mask"],
                    help="XTRACT FMA reference warped to native dMRI space (NIfTI)")
    ap.add_argument("--bedpostx-dir", default=CONFIG["bedpostx_dir"])
    ap.add_argument("--held-out-slides", default=CONFIG["held_out_psoct_slides"],
                    help="Directory of held-out (odd) PS-OCT slides for the "
                         "interleaved angular-error validation")
    args = ap.parse_args()

    trk_outputs = args.trk_outputs
    out_dir = os.path.join(trk_outputs, CONFIG["output_subdir"])
    os.makedirs(out_dir, exist_ok=True)

    # Reference grid / voxel volume / mean_f2 from the bedpostx brain mask.
    brain_mask_path = None
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(args.bedpostx_dir, f"nodif_brain_mask{ext}")
        if os.path.exists(p):
            brain_mask_path = p; break
    if brain_mask_path is None:
        sys.exit(f"Could not find nodif_brain_mask in {args.bedpostx_dir}")
    ref_img = nib.load(brain_mask_path)
    voxel_vol = voxel_volume_mm3(ref_img.affine)

    mean_f2 = None
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(args.bedpostx_dir, f"mean_f2samples{ext}")
        if os.path.exists(p):
            mean_f2 = np.asarray(nib.load(p).get_fdata(), dtype=np.float64); break

    # Reference mask (optional).
    ref_bin = None
    if args.reference and os.path.exists(args.reference):
        ref_arr = np.asarray(nib.load(args.reference).get_fdata(), dtype=np.float64)
        ref_bin = ref_arr > 0
        print(f"Reference mask: {args.reference}  ({ref_bin.sum()} voxels)")
    elif args.reference:
        print(f"WARNING: reference '{args.reference}' not found -- skipping reference overlap.")
    else:
        print("No reference mask set -- skipping reference-overlap numbers.")

    def resolve_group(key):
        names = CONFIG[key]
        out = []
        for n in names:
            f = _resolve_folder(trk_outputs, n)
            if f is not None:
                out.append(f)
        missing = [n for n in names if _resolve_folder(trk_outputs, n) is None]
        if missing:
            print(f"  [{key}] {len(out)}/{len(names)} present; missing: {', '.join(missing)}")
        return out

    ctx = dict(CONFIG)
    ctx.update({
        "trk_outputs": trk_outputs,
        "voxel_vol": voxel_vol,
        "mean_f2": mean_f2,
        "resolve_group": resolve_group,
        "hybrid_default_path": _resolve_folder(trk_outputs, CONFIG["hybrid_default"]),
        "baseline_default_path": _resolve_folder(trk_outputs, CONFIG["baseline_default"]),
        "probtrackx2_density": CONFIG["probtrackx2_density"],
        "brain_mask_path": brain_mask_path,
        "affine": ref_img.affine,
        "held_out_psoct_slides": args.held_out_slides,
    })

    print("\n=== volume + overlap ===")
    volume_overlap = analyse_volume_and_overlap(ctx)
    print("=== divergence ===")
    divergence = analyse_divergence(ctx)
    print("=== sensitivity ===")
    sensitivity = analyse_sensitivity(ctx, ref_bin)
    print("=== reproducibility (FTC) ===")
    reproducibility = analyse_reproducibility(ctx)
    print("=== reference overlap ===")
    reference_overlap = analyse_reference_overlap(ctx, ref_bin)
    print("=== held-out angular error ===")
    held_out = analyse_held_out(ctx)
    if "skipped" in held_out:
        print(f"  held-out: skipped ({held_out['skipped']})")
    else:
        print(f"  held-out: median {held_out['median_deg']:.1f} deg "
              f"(n={held_out['n_samples']})")

    results = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": {k: CONFIG[k] for k in
                   ("f_min", "density_threshold", "ftc_thresholds",
                    "hybrid_default", "baseline_default")},
        "voxel_volume_mm3": voxel_vol,
        "volume_overlap": volume_overlap,
        "divergence": divergence,
        "sensitivity": sensitivity,
        "reproducibility": reproducibility,
        "reference_overlap": reference_overlap,
        "held_out": held_out,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_filled_markdown(results, os.path.join(out_dir, "results_filled.md"))

    print("\n=== figures ===")
    fig_volume_count(results, os.path.join(out_dir, "fig_volume_count.png"))
    fig_reproducibility(results, CONFIG["ftc_thresholds"],
                        os.path.join(out_dir, "fig_reproducibility_ftc.png"))
    fig_sensitivity_alpha(results, os.path.join(out_dir, "fig_sensitivity_alpha.png"))
    fig_sensitivity_theta(results, os.path.join(out_dir, "fig_sensitivity_theta.png"))
    fig_sensitivity_density(results, os.path.join(out_dir, "fig_sensitivity_density.png"))
    fig_divergence(ctx, os.path.join(out_dir, "fig_divergence.png"))
    fig_held_out(ctx, results, os.path.join(out_dir, "fig_held_out.png"))

    print(f"\nWrote results.json, results_filled.md and figures to:\n  {out_dir}")


if __name__ == "__main__":
    main()
