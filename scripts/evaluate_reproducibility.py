"""
Reproducibility evaluation via Fractional Tanimoto Coefficient (FTC).

For each of the three conditions (probtrackx2, BedpostX-only, PSOCT-constrained),
run the tracker N times with different random seeds, compute per-voxel
streamline-density volumes normalised to [0, 1], and compute pairwise FTC across
the N runs. Report mean +- std FTC per condition and a boxplot.

FTC formula (Crum, Camara, Hill 2006):
    FTC(A, B) = sum_i min(A_i, B_i) / sum_i max(A_i, B_i)
A and B are continuous per-voxel densities in [0, 1].

Hypothesis (from planning report Sec 6.2):
    PSOCT-constrained should have the highest FTC because the microscopy
    constraint reduces stochastic variability in population selection.

Outputs (under <results_dir>/reproducibility/):
    figR11_reproducibility.png / .pdf
    reproducibility.json   -- per-condition mean, std, raw pairwise FTCs
    density_*.npz          -- the N density volumes per condition (for reuse by STAPLE)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from itertools import combinations
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.streamlines import load as load_tractogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT   = os.path.dirname(PROJECT_DIR)


def load_config(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "fma_config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg["_resolved"] = {
        "bedpostx_dir":  cfg["paths"]["bedpostx_dir"],
        "masks_dir":     cfg["paths"]["masks_dir"],
        "brain_mask":    cfg["paths"]["brain_mask"],
        "fma_seed":      os.path.join(REPO_ROOT, cfg["paths"]["fma_seed"]),
        "results_dir":   cfg["paths"]["results_dir"],
        "repro_dir":     os.path.join(cfg["paths"]["results_dir"],
                                      "reproducibility"),
    }
    return cfg


# =============================================================================
# Density / FTC
# =============================================================================

def density_volume_from_trk(trk_path, ref_img):
    """Voxel-visit counts, then normalise to [0, 1]."""
    tractogram = load_tractogram(trk_path)
    affine_inv = np.linalg.inv(ref_img.affine)
    shape = ref_img.shape[:3]
    density = np.zeros(shape, dtype=np.float32)
    for sl_world in tractogram.streamlines:
        sl_h = np.hstack([sl_world, np.ones((len(sl_world), 1))])
        sl_vox = (affine_inv @ sl_h.T).T[:, :3]
        vox = np.round(sl_vox).astype(int)
        in_b = ((vox[:, 0] >= 0) & (vox[:, 0] < shape[0]) &
                (vox[:, 1] >= 0) & (vox[:, 1] < shape[1]) &
                (vox[:, 2] >= 0) & (vox[:, 2] < shape[2]))
        vox = vox[in_b]
        if len(vox) == 0:
            continue
        vox = np.unique(vox, axis=0)
        density[vox[:, 0], vox[:, 1], vox[:, 2]] += 1.0
    mx = density.max()
    if mx > 0:
        density /= mx
    return density


def density_volume_from_fdtpaths(nii_path):
    """probtrackx2 fdt_paths.nii.gz is already a streamline count volume."""
    arr = nib.load(nii_path).get_fdata().astype(np.float32)
    mx = arr.max()
    if mx > 0:
        arr /= mx
    return arr


def fractional_tanimoto(A, B):
    """A, B are continuous per-voxel densities in [0, 1]."""
    mn = np.minimum(A, B).sum()
    mx = np.maximum(A, B).sum()
    if mx <= 0:
        return float("nan")
    return float(mn / mx)


# =============================================================================
# Per-condition runners
# =============================================================================

def run_python_tracker_once(condition, seed_value, suffix, max_seeds=None,
                            config_path=None, python_bin=None):
    """Invoke scripts/run_fma_results.py as a subprocess for one seed."""
    cmd = [python_bin or sys.executable,
           os.path.join(SCRIPT_DIR, "run_fma_results.py"),
           "--conditions", condition,
           "--random-seed", str(seed_value),
           "--output-suffix", suffix]
    if config_path:
        cmd += ["--config", config_path]
    if max_seeds is not None:
        cmd += ["--max-seeds", str(max_seeds)]
    print(" ", " ".join(cmd))
    subprocess.check_call(cmd)


def run_probtrackx2_once(seed_value, out_subdir, cfg):
    """Invoke run_probtrackx2.sh with --rseed, redirecting output dir."""
    env = os.environ.copy()
    env["PTX_OUTDIR_OVERRIDE"] = out_subdir
    env["PTX_RSEED"] = str(seed_value)
    cmd = ["bash", os.path.join(SCRIPT_DIR, "run_probtrackx2_one.sh"),
           str(seed_value), out_subdir]
    print(" ", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("-N", type=int, default=10,
                        help="Number of runs per condition (default 10)")
    parser.add_argument("--conditions", nargs="+",
                        default=["probtrackx2", "bedpostx", "psoct"],
                        choices=["probtrackx2", "bedpostx", "psoct"])
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Optional override of seeding.max_seed_voxels for "
                             "the Python trackers (speed knob for repro runs)")
    parser.add_argument("--skip-runs", action="store_true",
                        help="Skip the tractography step; assume the N density "
                             "volumes per condition already exist on disk")
    args = parser.parse_args()

    cfg = load_config(args.config)
    res = cfg["_resolved"]
    os.makedirs(res["repro_dir"], exist_ok=True)
    ref_img = nib.load(res["brain_mask"])

    # ---- Step 1: run trackers (or skip) ----
    seed_values = [cfg["seeding"]["random_seed"] + i for i in range(args.N)]

    if not args.skip_runs:
        for cond in args.conditions:
            print(f"\n=== Reproducibility runs: {cond} (N={args.N}) ===")
            t0 = time.time()
            for i, sv in enumerate(seed_values):
                suffix = f"_repro{i:02d}"
                print(f"[{cond} run {i+1}/{args.N}] seed={sv}")
                if cond == "probtrackx2":
                    out_subdir = os.path.join(res["results_dir"],
                                              f"FMA_probtrackx2{suffix}")
                    run_probtrackx2_once(sv, out_subdir, cfg)
                else:
                    py_cond = "bedpostx" if cond == "bedpostx" else "psoct"
                    run_python_tracker_once(py_cond, sv, suffix,
                                            max_seeds=args.max_seeds,
                                            config_path=args.config)
            print(f"  {cond}: {time.time()-t0:.0f}s for {args.N} runs")

    # ---- Step 2: build density volumes ----
    print("\n=== Building density volumes ===")
    densities = {c: [] for c in args.conditions}
    for cond in args.conditions:
        for i in range(args.N):
            suffix = f"_repro{i:02d}"
            if cond == "probtrackx2":
                nii = os.path.join(res["results_dir"],
                                   f"FMA_probtrackx2{suffix}", "fdt_paths.nii.gz")
                if not os.path.exists(nii):
                    print(f"  MISSING: {nii} (run {i})")
                    continue
                d = density_volume_from_fdtpaths(nii)
            else:
                base = "FMA_bedpostx_only" if cond == "bedpostx" else "FMA_psoct_constrained"
                trk = os.path.join(res["results_dir"],
                                   f"{base}{suffix}", f"{base}.trk")
                if not os.path.exists(trk):
                    print(f"  MISSING: {trk} (run {i})")
                    continue
                d = density_volume_from_trk(trk, ref_img)
            densities[cond].append(d)
        print(f"  {cond}: built {len(densities[cond])}/{args.N} density volumes")

    # Save the density stacks (used by STAPLE later)
    for cond, stack in densities.items():
        if not stack:
            continue
        np.savez_compressed(
            os.path.join(res["repro_dir"], f"density_{cond}.npz"),
            densities=np.stack(stack),
        )

    # ---- Step 3: pairwise FTC ----
    print("\n=== Computing pairwise FTC ===")
    ftc_results = {}
    for cond, stack in densities.items():
        if len(stack) < 2:
            print(f"  {cond}: <2 runs, skipping")
            ftc_results[cond] = []
            continue
        pairs = list(combinations(range(len(stack)), 2))
        vals = [fractional_tanimoto(stack[i], stack[j]) for i, j in pairs]
        ftc_results[cond] = vals
        vals = np.array(vals)
        print(f"  {cond}: n_pairs={len(vals)}, "
              f"mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}")

    # ---- Step 4: figure ----
    cond_order = [c for c in ("probtrackx2", "bedpostx", "psoct") if c in ftc_results
                  and len(ftc_results[c]) > 0]
    pretty = {"probtrackx2": "probtrackx2",
              "bedpostx": "BedpostX-only",
              "psoct": "PSOCT-constrained"}
    colors = {"probtrackx2": "steelblue",
              "bedpostx": "salmon",
              "psoct": "mediumseagreen"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    data = [ftc_results[c] for c in cond_order]
    labels = [pretty[c] for c in cond_order]

    bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                         showmeans=True)
    for patch, c in zip(bp["boxes"], cond_order):
        patch.set_facecolor(colors[c])
    axes[0].set_ylabel("Fractional Tanimoto Coefficient (FTC)")
    axes[0].set_title("(a) Pairwise FTC per condition", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(15)

    means = [np.mean(d) for d in data]
    stds  = [np.std(d)  for d in data]
    bars = axes[1].bar(labels, means, yerr=stds, capsize=8,
                       color=[colors[c] for c in cond_order],
                       edgecolor="black", linewidth=1.0)
    for bar, m in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, m + 0.02,
                     f"{m:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("Mean FTC (+/- 1 sd)")
    axes[1].set_title("(b) Summary", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(15)

    fig.suptitle(f"Figure R11 - reproducibility via FTC (N={args.N} runs per condition)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(res["repro_dir"], "figR11_reproducibility.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")

    # ---- Step 5: save JSON ----
    summary = {
        "timestamp":  datetime.now().isoformat(),
        "N_per_condition": args.N,
        "seed_values": seed_values,
        "results": {
            c: {
                "n_pairs": len(v),
                "mean":   float(np.mean(v)) if v else None,
                "std":    float(np.std(v))  if v else None,
                "median": float(np.median(v)) if v else None,
                "pairs":  v,
            } for c, v in ftc_results.items()
        },
    }
    out_json = os.path.join(res["repro_dir"], "reproducibility.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
