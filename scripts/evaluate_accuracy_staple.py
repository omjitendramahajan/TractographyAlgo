"""
Accuracy evaluation via STAPLE (Warfield et al., 2004).

We do not have a manual ground-truth segmentation of the FMA tract. STAPLE
treats the true tract as a hidden binary variable and uses Expectation-
Maximisation to jointly estimate (a) a per-voxel foreground probability and
(b) per-rater sensitivity p_j and specificity q_j.

Inputs:
    Binary tract masks for each of the three conditions:
        probtrackx2  -- fdt_paths.nii.gz binarised at threshold T
        BedpostX-only -- streamline density binarised at threshold T
        PSOCT-constrained -- streamline density binarised at threshold T

Outputs (under <results_dir>/accuracy/):
    figR12_staple.png / .pdf
    staple_results.json            -- p, q per condition for each threshold
    staple_consensus_T{T}.nii.gz  -- consensus foreground probability volumes

Caveats to flag in the writeup (see Discussion §5.4):
    - With only 3 raters, STAPLE consensus is dominated by 2-vs-1 disagreements.
    - The result is "agreement with the multi-method consensus", NOT absolute
      accuracy against biological ground truth.
"""

import os
import sys
import json
import argparse
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


# =============================================================================
# Config + density loading
# =============================================================================

def load_config(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "fma_config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg["_resolved"] = {
        "brain_mask":  cfg["paths"]["brain_mask"],
        "results_dir": cfg["paths"]["results_dir"],
        "ptx_density": os.path.join(cfg["paths"]["results_dir"],
                                    "FMA_probtrackx2", "fdt_paths.nii.gz"),
        "bx_trk":      os.path.join(cfg["paths"]["results_dir"],
                                    "FMA_bedpostx_only", "FMA_bedpostx_only.trk"),
        "ps_trk":      os.path.join(cfg["paths"]["results_dir"],
                                    "FMA_psoct_constrained",
                                    "FMA_psoct_constrained.trk"),
        "out_dir":     os.path.join(cfg["paths"]["results_dir"],
                                    "accuracy"),
    }
    return cfg


def trk_visit_count(trk_path, ref_img):
    tractogram = load_tractogram(trk_path)
    affine_inv = np.linalg.inv(ref_img.affine)
    shape = ref_img.shape[:3]
    density = np.zeros(shape, dtype=np.int32)
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
        density[vox[:, 0], vox[:, 1], vox[:, 2]] += 1
    return density


# =============================================================================
# STAPLE (pure NumPy, binary)
# =============================================================================

def staple_binary(masks, max_iter=200, tol=1e-6, init_p=0.99999, init_q=0.99999,
                  prior=None, verbose=True):
    """Binary STAPLE for J raters.

    Args:
        masks: (J, *spatial) boolean / 0-1 numpy array of rater decisions
        max_iter, tol: convergence settings
        init_p, init_q: initial sensitivity / specificity per rater
        prior: foreground prior P(T=1). If None, use the global rater mean.

    Returns:
        W: per-voxel posterior P(T=1 | D), same spatial shape as one mask
        p: (J,) sensitivities
        q: (J,) specificities
        n_iter: iterations to convergence
    """
    J = masks.shape[0]
    D = masks.reshape(J, -1).astype(np.float64)  # (J, V)
    V = D.shape[1]

    if prior is None:
        prior = float(D.mean())
    prior = max(min(prior, 1 - 1e-6), 1e-6)

    p = np.full(J, init_p)
    q = np.full(J, init_q)

    prev_W = None
    for it in range(max_iter):
        # E-step: posterior P(T=1 | D, p, q)
        # likelihood for T=1: prod_j p_j^{D_j} (1-p_j)^{1-D_j}
        # likelihood for T=0: prod_j (1-q_j)^{D_j} q_j^{1-D_j}
        # work in log-space for numerical stability
        log_p  = np.log(p)
        log_1p = np.log1p(-p)
        log_q  = np.log(q)
        log_1q = np.log1p(-q)
        log_lik_1 = D.T @ log_p + (1 - D.T) @ log_1p     # (V,)
        log_lik_0 = D.T @ log_1q + (1 - D.T) @ log_q     # (V,)
        log_post_1 = log_lik_1 + np.log(prior)
        log_post_0 = log_lik_0 + np.log(1 - prior)
        # normalise
        mx = np.maximum(log_post_1, log_post_0)
        W = np.exp(log_post_1 - mx) / (
            np.exp(log_post_1 - mx) + np.exp(log_post_0 - mx))     # (V,)

        # M-step
        W_sum = W.sum()
        Wc_sum = (1 - W).sum()
        if W_sum > 0:
            p_new = (D * W).sum(axis=1) / W_sum
        else:
            p_new = p
        if Wc_sum > 0:
            q_new = ((1 - D) * (1 - W)).sum(axis=1) / Wc_sum
        else:
            q_new = q
        # numerical guard
        p_new = np.clip(p_new, 1e-6, 1 - 1e-6)
        q_new = np.clip(q_new, 1e-6, 1 - 1e-6)

        if prev_W is not None:
            diff = float(np.abs(W - prev_W).mean())
            if verbose:
                print(f"    iter {it+1:>3} | p={p_new.round(3)} q={q_new.round(3)} | "
                      f"|dW|={diff:.2e}")
            if diff < tol:
                p, q = p_new, q_new
                break
        else:
            if verbose:
                print(f"    iter {it+1:>3} | p={p_new.round(3)} q={q_new.round(3)}")
        prev_W = W
        p, q = p_new, q_new

    return W.reshape(masks.shape[1:]), p, q, it + 1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[1, 5],
                        help="Visit-count thresholds for binarisation (default: 1 and 5)")
    parser.add_argument("--restrict-to-mask", action="store_true",
                        help="Restrict STAPLE to the brain mask voxels only")
    parser.add_argument("--prior", type=float, default=None,
                        help="Foreground prior P(T=1). Default = rater mean. "
                             "Setting to the expected tract size as a fraction "
                             "of brain volume gives less biased sensitivity "
                             "estimates (we use rater mean by default since the "
                             "true tract size is unknown).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    res = cfg["_resolved"]
    os.makedirs(res["out_dir"], exist_ok=True)

    ref_img = nib.load(res["brain_mask"])
    affine = ref_img.affine
    brain_mask = ref_img.get_fdata() > 0

    # --- Build raw streamline-visit counts per condition ---
    print("Building visit-count volumes ...")
    counts = {}
    if os.path.exists(res["ptx_density"]):
        counts["probtrackx2"] = nib.load(res["ptx_density"]).get_fdata().astype(np.int32)
        print(f"  probtrackx2:     max={counts['probtrackx2'].max()}")
    if os.path.exists(res["bx_trk"]):
        counts["bedpostx"] = trk_visit_count(res["bx_trk"], ref_img)
        print(f"  BedpostX-only:    max={counts['bedpostx'].max()}")
    if os.path.exists(res["ps_trk"]):
        counts["psoct"] = trk_visit_count(res["ps_trk"], ref_img)
        print(f"  PSOCT-constrained: max={counts['psoct'].max()}")

    if len(counts) < 2:
        print("Need at least 2 conditions to run STAPLE. Abort.")
        return

    cond_order = [c for c in ("probtrackx2", "bedpostx", "psoct") if c in counts]
    pretty = {"probtrackx2": "probtrackx2",
              "bedpostx": "BedpostX-only",
              "psoct": "PSOCT-constrained"}

    all_results = {}

    for T in args.thresholds:
        print(f"\n=== STAPLE @ threshold T={T} ===")
        binaries = np.stack([counts[c] >= T for c in cond_order], axis=0)
        if args.restrict_to_mask:
            for j in range(len(cond_order)):
                binaries[j] &= brain_mask

        n_fg = binaries.sum(axis=(1, 2, 3))
        for c, n in zip(cond_order, n_fg):
            print(f"  {pretty[c]:25s}: {n:,} foreground voxels at T>={T}")

        W, p, q, n_iter = staple_binary(binaries, prior=args.prior, verbose=True)
        print(f"  converged in {n_iter} iter")
        for c, pp, qq in zip(cond_order, p, q):
            print(f"  {pretty[c]:25s}: sensitivity p={pp:.4f}  specificity q={qq:.4f}")

        # save consensus
        nii = nib.Nifti1Image(W.astype(np.float32), affine)
        nib.save(nii, os.path.join(res["out_dir"], f"staple_consensus_T{T}.nii.gz"))

        all_results[f"T={T}"] = {
            "threshold": T,
            "conditions": cond_order,
            "sensitivity": {c: float(pp) for c, pp in zip(cond_order, p)},
            "specificity": {c: float(qq) for c, qq in zip(cond_order, q)},
            "n_foreground_per_rater": {c: int(n) for c, n in zip(cond_order, n_fg)},
            "staple_n_iter": int(n_iter),
            "consensus_volume_voxels": int((W > 0.5).sum()),
        }

    # --- Figure R12 ---
    fig, axes = plt.subplots(1, len(args.thresholds) + 1, figsize=(6 + 5 * len(args.thresholds), 6))
    if len(args.thresholds) == 0:
        axes = [axes]

    colors = {"probtrackx2": "steelblue",
              "bedpostx":    "salmon",
              "psoct":       "mediumseagreen"}

    for tx, T in enumerate(args.thresholds):
        rec = all_results[f"T={T}"]
        ax = axes[tx]
        x = np.arange(len(cond_order))
        width = 0.35
        ps = [rec["sensitivity"][c] for c in cond_order]
        qs = [rec["specificity"][c] for c in cond_order]
        ax.bar(x - width/2, ps, width, label="sensitivity (p)",
               color=[colors[c] for c in cond_order],
               edgecolor="black", alpha=0.85)
        ax.bar(x + width/2, qs, width, label="specificity (q)",
               color=[colors[c] for c in cond_order],
               edgecolor="black", hatch="///", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty[c] for c in cond_order], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("STAPLE estimate")
        ax.set_title(f"({chr(97+tx)}) Sensitivity / specificity @ threshold T={T}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for xi, p_, q_ in zip(x, ps, qs):
            ax.text(xi - width/2, p_ + 0.02, f"{p_:.3f}",
                    ha="center", fontsize=8)
            ax.text(xi + width/2, q_ + 0.02, f"{q_:.3f}",
                    ha="center", fontsize=8)

    # consensus volume MIP for the lowest threshold
    T0 = args.thresholds[0]
    consensus_nii = nib.load(os.path.join(res["out_dir"], f"staple_consensus_T{T0}.nii.gz"))
    W_vol = consensus_nii.get_fdata()
    iy = int(np.argmax((W_vol > 0.5).sum(axis=(0, 2))))
    ax = axes[-1]
    anat = ref_img.get_fdata()
    ax.imshow(anat[:, iy, :].T, origin="lower", cmap="gray")
    masked = np.ma.masked_where(W_vol[:, iy, :] <= 0.05, W_vol[:, iy, :])
    im = ax.imshow(masked.T, origin="lower", cmap="hot",
                   vmin=0, vmax=1, alpha=0.8)
    ax.set_title(f"({chr(97+len(args.thresholds))}) STAPLE consensus (T={T0}) "
                 f"\ncoronal slice y={iy}", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(T=1)")

    fig.suptitle("Figure R12 - accuracy via STAPLE (sensitivity/specificity vs. EM consensus)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(res["out_dir"], "figR12_staple.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")

    # --- save JSON ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": args.thresholds,
        "results_by_threshold": all_results,
        "caveats": [
            "STAPLE consensus with 3 raters is dominated by 2-vs-1 disagreements; "
            "the reported sensitivity/specificity is agreement with the multi-method "
            "consensus, not absolute accuracy against biological ground truth.",
            "When the foreground prior is set to the rater mean (default), STAPLE "
            "tends to underestimate absolute sensitivity but preserves rank order "
            "across raters. Compare conditions to each other, not against 1.0.",
        ],
    }
    out_json = os.path.join(res["out_dir"], "staple_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
