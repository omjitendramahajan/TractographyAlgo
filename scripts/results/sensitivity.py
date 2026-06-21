"""
sensitivity.py -- Hyperparameter and PS-OCT density sensitivity.
================================================================

Self-contained. Edit the CONFIG block with the EXACT path to every file, run:

    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    $PY TractographyAlgo/scripts/results/sensitivity.py

Sweeps three parameters and, for each value, computes tract volume, Dice vs the
baseline default run, and (if REFERENCE_MASK is set) Dice vs the reference:
  * alpha   -- blend coefficient (alpha=1 should reproduce the baseline),
  * theta   -- streamline angle threshold,
  * density -- PS-OCT slice density thinning (dose-response).

Each sweep maps {parameter value: visit_map path}.

Writes sensitivity.json, sensitivity.md and fig_sensitivity_{alpha,theta,density}.png
to OUTPUT_DIR.
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
OUTPUT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/results/sensitivity"

DENSITY_THRESHOLD = 1
VOXEL_VOLUME_MM3 = 0.064   # fallback only (used if a density comes from a *.npz)

# Baseline default run -> Dice-vs-baseline column for every sweep row.
BASELINE_DEFAULT_VISIT_MAP = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/baseline_run_06/baseline_run_6_visit_map.nii.gz"

# Anatomical reference mask warped to native dMRI space, or None to skip the
# Dice-vs-reference columns.
REFERENCE_MASK = None

# --- alpha (blend coefficient) sweep: value -> visit_map path. ---
ALPHA_SWEEP = {
    0.25: "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/sweep_alpha_025/sweep_alpha_025_visit_map.nii.gz",
    0.5:  "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz",
    0.75: "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/sweep_alpha_075/sweep_alpha_075_visit_map.nii.gz",
    1.0:  "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/sweep_alpha_100/sweep_alpha_100_visit_map.nii.gz",
}

# --- angle threshold theta sweep (deg): value -> visit_map path. ---
THETA_SWEEP = {
    45: "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/sweep_theta_45/sweep_theta_45_visit_map.nii.gz",
    60: "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz",
    80: "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/sweep_theta_80/sweep_theta_80_visit_map.nii.gz",
}

# --- PS-OCT density thinning dose-response: label -> visit_map path. ---
DENSITY_SWEEP = {
    "full (every slice)": "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/hybrid_run_03/hybrid3_visit_map.nii.gz",
    "every 2nd":          "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/density_every_2nd/density_every_2nd_visit_map.nii.gz",
    "every 4th":          "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/density_every_4th/density_every_4th_visit_map.nii.gz",
}
# =============================================================================


# ---------------------------------------------------------------- helpers ----
def load_density(path):
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


def binarise(arr, thresh):
    return arr >= thresh


def dice(a_bin, b_bin):
    a = a_bin.astype(bool); b = b_bin.astype(bool)
    denom = a.sum() + b.sum()
    return float("nan") if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "[n/a]"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _load_reference():
    if REFERENCE_MASK and os.path.exists(REFERENCE_MASK):
        ref = np.asarray(nib.load(REFERENCE_MASK).get_fdata(), dtype=np.float64) > 0
        print(f"  reference: {int(ref.sum())} voxels")
        return ref
    if REFERENCE_MASK:
        print(f"  WARNING: reference '{REFERENCE_MASK}' not found -- skipping ref columns.")
    return None


def _sweep_table(sweep, bd_bin, ref_bin):
    rows = []
    for val, path in sweep.items():
        arr, aff = load_density(path)
        if arr is None:
            rows.append({"value": val, "path": path, "missing": True}); continue
        vv = voxel_volume_mm3(aff) if aff is not None else VOXEL_VOLUME_MM3
        a_bin = binarise(arr, DENSITY_THRESHOLD)
        row = {"value": val, "path": path, "volume_mm3": float(a_bin.sum() * vv)}
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
    vals = [r[key] for r in rows if key in r and r[key] is not None
            and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return (min(vals), max(vals)) if vals else (None, None)


# --------------------------------------------------------------- analysis ----
def analyse():
    out = {}
    bd, baff = load_density(BASELINE_DEFAULT_VISIT_MAP)
    bd_bin = binarise(bd, DENSITY_THRESHOLD) if bd is not None else None
    ref_bin = _load_reference()

    # alpha
    alpha = _sweep_table(ALPHA_SWEEP, bd_bin, ref_bin)
    dice_lo, dice_hi = _range_of(alpha, "dice_vs_baseline")
    a_entry = {"rows": alpha, "volume_range_pct": _range_pct(alpha, "volume_mm3"),
               "dice_vs_baseline_lo": dice_lo, "dice_vs_baseline_hi": dice_hi}
    base_vol = None
    if bd is not None:
        vv = voxel_volume_mm3(baff) if baff is not None else VOXEL_VOLUME_MM3
        base_vol = float(binarise(bd, DENSITY_THRESHOLD).sum() * vv)
    for r in alpha:
        if r.get("value") == 1.0 and "volume_mm3" in r:
            a_entry["alpha1_dice_vs_baseline"] = r.get("dice_vs_baseline")
            if base_vol:
                a_entry["alpha1_volume_diff_pct"] = 100.0 * abs(r["volume_mm3"] - base_vol) / base_vol
    out["alpha"] = a_entry

    # theta
    theta = _sweep_table(THETA_SWEEP, bd_bin, ref_bin)
    tref_lo, tref_hi = _range_of(theta, "dice_vs_reference")
    tbase_lo, tbase_hi = _range_of(theta, "dice_vs_baseline")
    out["theta"] = {"rows": theta, "volume_range_pct": _range_pct(theta, "volume_mm3"),
                    "dice_vs_reference_lo": tref_lo, "dice_vs_reference_hi": tref_hi,
                    "dice_vs_baseline_lo": tbase_lo, "dice_vs_baseline_hi": tbase_hi}

    # density thinning
    density = _sweep_table(DENSITY_SWEEP, bd_bin, ref_bin)
    out["density_thinning"] = {"rows": density}
    return out


# ---------------------------------------------------------------- figures ----
def figure_theta(result, path):
    rows = sorted([r for r in result.get("theta", {}).get("rows", []) if "volume_mm3" in r],
                  key=lambda r: r["value"])
    if len(rows) < 2:
        return
    x = [r["value"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, [r["volume_mm3"] for r in rows], "o-", color="steelblue")
    ax1.set_xlabel(r"angle threshold $\theta_{max}$ (deg)")
    ax1.set_ylabel("tract volume (mm$^3$)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue"); ax1.grid(alpha=0.3)
    if any("dice_vs_baseline" in r or "dice_vs_reference" in r for r in rows):
        ax2 = ax1.twinx()
        if all("dice_vs_baseline" in r for r in rows):
            ax2.plot(x, [r["dice_vs_baseline"] for r in rows], "s--", color="darkorange", label="Dice vs Diffusion-only")
        if all("dice_vs_reference" in r for r in rows):
            ax2.plot(x, [r["dice_vs_reference"] for r in rows], "^--", color="firebrick", label="Dice vs reference")
        ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1.0); ax2.legend(loc="lower right", fontsize=8)
    ax1.set_title(r"Sensitivity to $\theta_{max}$", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def figure_alpha(result, path):
    rows = sorted([r for r in result.get("alpha", {}).get("rows", []) if "volume_mm3" in r],
                  key=lambda r: r["value"])
    if len(rows) < 2:
        return
    x = [r["value"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, [r["volume_mm3"] for r in rows], "o-", color="steelblue")
    ax1.set_xlabel(r"blend coefficient $\alpha$  (1 = trajectory only)")
    ax1.set_ylabel("tract volume (mm$^3$)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue"); ax1.grid(alpha=0.3)
    if all("dice_vs_baseline" in r for r in rows):
        ax2 = ax1.twinx()
        ax2.plot(x, [r["dice_vs_baseline"] for r in rows], "s--", color="darkorange", label="Dice vs Diffusion-only")
        if all("dice_vs_reference" in r for r in rows):
            ax2.plot(x, [r["dice_vs_reference"] for r in rows], "^--", color="firebrick", label="Dice vs reference")
        ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1.0); ax2.legend(loc="lower right", fontsize=8)
    ax1.set_title(r"Sensitivity to blend coefficient $\alpha$", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt

def figure_density(result, path):
    rows = [r for r in result.get("density_thinning", {}).get("rows", [])
            if "dice_vs_baseline" in r or "dice_vs_reference" in r]
    if not rows:
        return
        
    labels = [str(r.get("value")) for r in rows]
    x = np.arange(len(rows))
    w = 0.38 
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    all_vals = []
    if all("dice_vs_baseline" in r for r in rows):
        vals = [r["dice_vs_baseline"] for r in rows]
        all_vals.extend(vals)
        bars_baseline = ax.bar(x - w / 2, vals, width=w, label="Dice vs Diffusion-only", color="salmon")
        # Add values on top of the bars
        ax.bar_label(bars_baseline, fmt='%.3f', padding=3, fontsize=8)
        
    if all("dice_vs_reference" in r for r in rows):
        vals = [r["dice_vs_reference"] for r in rows]
        all_vals.extend(vals)
        bars_ref = ax.bar(x + w / 2, vals, width=w, label="Dice vs reference", color="firebrick")
        # Add values on top of the bars
        ax.bar_label(bars_ref, fmt='%.3f', padding=3, fontsize=8)
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Dice")
    
    # Increased the top buffer to 0.15 to comfortably fit the text labels and legend
    if all_vals:
        y_min = max(0.0, min(all_vals) - 0.05) 
        y_max = max(all_vals) + 0.05
        ax.set_ylim(bottom=y_min, top=y_max)
        
    ax.legend(loc="upper left")
    
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("PS-OCT density response", fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def markdown(result):
    L = ["## Hyperparameter and PS-OCT density sensitivity", ""]
    al = result.get("alpha", {})
    al_rows = [x for x in al.get("rows", []) if "volume_mm3" in x]
    if al_rows:
        L.append(f"- VOLUME RANGE (alpha) = {fmt(al.get('volume_range_pct'), 1)} %")
        L.append(f"- DICE LO = {fmt(al.get('dice_vs_baseline_lo'))}, "
                 f"DICE HI = {fmt(al.get('dice_vs_baseline_hi'))} (vs baseline)")
        if "alpha1_dice_vs_baseline" in al:
            L.append(f"- ALPHA1 DICE = {fmt(al.get('alpha1_dice_vs_baseline'))}, "
                     f"ALPHA1 VOL DIFF = {fmt(al.get('alpha1_volume_diff_pct'), 2)} %")
        for r in sorted(al_rows, key=lambda x: x["value"]):
            L.append(f"  - alpha={r['value']}: volume={fmt(r['volume_mm3'], 0)} mm^3, "
                     f"Dice_vs_baseline={fmt(r.get('dice_vs_baseline'))}, "
                     f"Dice_vs_ref={fmt(r.get('dice_vs_reference'))}")
    else:
        L.append("- (alpha sweep: no runs found -- check the ALPHA_SWEEP paths)")
    th = result.get("theta", {})
    if th.get("rows"):
        L.append(f"- THETA VOLUME RANGE = {fmt(th.get('volume_range_pct'), 1)} %")
        L.append(f"- THETA REF LO = {fmt(th.get('dice_vs_reference_lo'))}, "
                 f"THETA REF HI = {fmt(th.get('dice_vs_reference_hi'))}")
        for r in sorted([x for x in th["rows"] if "volume_mm3" in x], key=lambda x: x["value"]):
            L.append(f"  - theta={r['value']}: volume={fmt(r['volume_mm3'], 0)} mm^3, "
                     f"Dice_vs_baseline={fmt(r.get('dice_vs_baseline'))}, "
                     f"Dice_vs_ref={fmt(r.get('dice_vs_reference'))}")
    dt = result.get("density_thinning", {})
    if dt.get("rows"):
        L.append("- Density thinning (THIN DICE / THIN REF):")
        for r in dt["rows"]:
            if "volume_mm3" in r:
                L.append(f"  - {r['value']}: Dice_vs_baseline={fmt(r.get('dice_vs_baseline'))}, "
                         f"Dice_vs_ref={fmt(r.get('dice_vs_reference'))}")
    L.append("")
    return L


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== sensitivity ===")
    result = analyse()
    with open(os.path.join(OUTPUT_DIR, "sensitivity.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "sensitivity.md"), "w") as f:
        f.write("\n".join(markdown(result)) + "\n")
    figure_alpha(result, os.path.join(OUTPUT_DIR, "fig_sensitivity_alpha.png"))
    figure_theta(result, os.path.join(OUTPUT_DIR, "fig_sensitivity_theta.png"))
    figure_density(result, os.path.join(OUTPUT_DIR, "fig_sensitivity_density.png"))
    print(f"  wrote sensitivity.json/.md and fig_sensitivity_*.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()