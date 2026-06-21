"""
PSOCT <-> BedpostX registration QC by orientation agreement.
==========================================================

Idea
----
If the PSOCT microscopy slides and the BedpostX dMRI volume are correctly
registered (spatially AND in orientation convention), then in a voxel with a
strong, coherent single fibre the two modalities should report the *same*
fibre orientation. This script tests exactly that:

  1. Pick ``n_voxels`` random voxels that
        - have a strong principal fibre  (mean_f1 > ``f_min``), and
        - lie near the middle of the volume (within ``radius`` voxels of
          ``center``, default (173, 173, 50)),
     so the dMRI signal is clean and well inside PSOCT coverage.

  2. For each voxel:
        - BedpostX direction  = the ``dyads1`` principal orientation
          (the mean_f1 dyad), read through ``BedpostxData`` so the same
          world-frame handedness correction the tracker uses is applied.
        - PSOCT direction      = the geometric (axial) circular mean of the
          orientations of *every* PSOCT pixel that falls inside the voxel.
          Each pixel angle is corrected with ``fudge_psoct_orientation`` and
          lifted to a 3D unit vector with the slide-to-volume affine exactly
          as ``PSOCTData.get_orientation`` does, then averaged with the
          directionless (orientation, not direction) tensor mean.

  3. Compare the two with a sign-insensitive angle folded to [0, 90] deg
        ang = arccos(|cos|)  in 3D, and also after projecting the dMRI dyad
        into the PSOCT slide plane (PSOCT can only see in-plane structure).

A correctly registered pair gives a distribution piled up near 0 deg. As a
control, the same dMRI dyads are paired with PSOCT means from *other* random
voxels: that shuffled pairing is uniform on [0, 90] (median ~45 deg), so the
gap between the two distributions is the registration signal.

Run
---
    PYTHONPATH=TractographyAlgo/cmc_hybrid:TractographyAlgo \
    /Users/ommahajan/anaconda3/envs/tractography/bin/python3 \
        TractographyAlgo/tractography/registration_qc.py
"""

import os
import csv
import json
from datetime import datetime

import numpy as np

from fsl.data.image import Image
from tractography.bedpostx import BedpostxData
from tractography.psoct import PSOCTData
from cmc_hybrid.coordinate_mapping import vox_to_pix_per_slide, angle_to_vector
from cmc_hybrid.utils import fudge_psoct_orientation


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CONFIG = {
    'bedpostx_dir': "DATA/data.bedpostX_SSFP_uncompressed",
    'psoct_slides': "DATA/PS-OCT/Coregistered/FLIRT/Orientations/lowres",

    'center':   (173, 173, 50),   # "middle of the dataset" – clean WM, good signal
    'radius':   25,               # voxels: candidates must lie within this of center
    'f_min':    0.6,              # mean_f1 threshold for a strong principal fibre
    'n_voxels': 1000,             # number of voxels to test
    'min_pix':  10,               # need at least this many PSOCT pixels for a mean
    'seed':     0,                # RNG seed for reproducible voxel choice

    'out_dir':  None,             # default: <script>/registration_QC/Output_<ts>
}


# ----------------------------------------------------------------------------
# Orientation helpers (directionless / axial statistics)
# ----------------------------------------------------------------------------
def axial_mean_vector(vectors):
    """Directionless mean of 3D unit orientations via the dyadic tensor.

    ``vectors`` is an (N, 3) array of (near-)unit orientations whose sign is
    meaningless. Returns the principal eigenvector of mean(v v^T) (the axial
    circular mean generalised to 3D) plus a concentration measure in [0, 1]
    (lambda1 - lambda2; 1 = perfectly aligned, 0 = isotropic in a plane).
    """
    v = np.asarray(vectors, dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    T = (v[:, :, None] * v[:, None, :]).mean(axis=0)
    w, V = np.linalg.eigh(T)          # ascending eigenvalues
    principal = V[:, -1]
    concentration = float(w[-1] - w[-2])
    return principal / np.linalg.norm(principal), concentration


def axial_circular_mean_angle(thetas):
    """2D axial circular mean of period-pi angles (cross-check of the above).

    Returns (mean_angle, R) where R in [0, 1] is the resultant length
    (mean cosine of the doubled-angle distribution): R near 1 => tight.
    """
    c = np.cos(2 * thetas).mean()
    s = np.sin(2 * thetas).mean()
    return 0.5 * np.arctan2(s, c), float(np.hypot(c, s))


def acute_angle_deg(a, b):
    """Sign-insensitive angle (deg) between two 3D vectors, folded to [0, 90]."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(cfg):
    rng = np.random.default_rng(cfg['seed'])
    center = np.asarray(cfg['center'])

    print("=" * 64)
    print("PSOCT <-> BedpostX registration QC (orientation agreement)")
    print("=" * 64)

    # --- load data (dyads + mean_f only; no posterior samples needed) -------
    print("\n[1/5] Loading BedpostX (dyads1 + mean_f1) ...")
    bp = BedpostxData(cfg['bedpostx_dir'], num_fibers=1,
                      load_samples=False, load_dyads=True, use_memory_map=True)
    f1 = bp.f_samples[0]
    print(f"      volume shape {bp.shape}, voxel {np.round(bp.voxel_size, 3)} mm")

    print("\n[2/5] Loading PSOCT coronal slides ...")
    ps = PSOCTData(cfg['psoct_slides'], bp.volume_img)

    # --- build candidate pool: strong fibre AND near the centre -------------
    print(f"\n[3/5] Selecting voxels: mean_f1 > {cfg['f_min']}, "
          f"within {cfg['radius']} vox of {tuple(center)} ...")
    R = cfg['radius']
    lo = np.maximum(center - R, 0)
    hi = np.minimum(center + R + 1, np.asarray(bp.shape))
    grid = np.stack(np.meshgrid(
        np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
        np.arange(lo[2], hi[2]), indexing='ij'), axis=-1).reshape(-1, 3)
    d2 = ((grid - center) ** 2).sum(1)
    in_sphere = d2 <= R * R
    i, j, k = grid[in_sphere].T
    strong = bp.mask[i, j, k] & (f1[i, j, k] > cfg['f_min'])
    candidates = grid[in_sphere][strong]
    print(f"      {len(candidates)} candidate voxels in the pool")
    if len(candidates) < cfg['n_voxels']:
        print(f"      WARNING: fewer candidates than requested "
              f"({len(candidates)} < {cfg['n_voxels']}); using all and "
              f"increase 'radius' or lower 'f_min' for more.")

    rng.shuffle(candidates)

    # --- per voxel: dMRI dyad vs PSOCT circular-mean orientation ------------
    print(f"\n[4/5] Comparing orientations (target n = {cfg['n_voxels']}) ...")
    rows = []           # per accepted voxel
    psoct_vecs = []     # for the shuffled-null control
    dyad_vecs = []
    target = cfg['n_voxels']

    for v in candidates:
        if len(rows) >= target:
            break
        vox = v.astype(float)

        # gather every PSOCT pixel inside this voxel, across all slides
        per = vox_to_pix_per_slide(list(v), ps.slides, ps.volume_img)
        if not per:
            continue

        pix_vecs = []
        dom_slide, dom_n = None, -1
        dom_thetas = None
        for sidx, (_pg, _vg, theta) in per.items():
            if len(theta) == 0:
                continue
            theta = np.asarray(theta, dtype=float)
            theta_c = fudge_psoct_orientation(theta)            # PSOCT correction
            vecs = angle_to_vector(theta_c, ps._slide_xforms[sidx])  # -> 3D units
            pix_vecs.append(vecs)
            if len(theta) > dom_n:
                dom_n, dom_slide, dom_thetas = len(theta), sidx, theta_c

        if not pix_vecs:
            continue
        pix_vecs = np.vstack(pix_vecs)
        if len(pix_vecs) < cfg['min_pix']:
            continue

        # PSOCT geometric (axial) circular mean orientation
        psoct_mean, concentration = axial_mean_vector(pix_vecs)
        # 2D circular resultant on the dominant slide (reliability proxy)
        _, R_circ = axial_circular_mean_angle(dom_thetas)

        # BedpostX principal dyad (mean_f1 dyad), same frame as the tracker
        dyad = bp.get_orientation(v, fiber_idx=0)
        if np.linalg.norm(dyad) == 0:
            continue

        ang3d = acute_angle_deg(psoct_mean, dyad)

        # in-plane angle: project the dyad into the PSOCT slide plane, since
        # PSOCT only observes in-plane structure (fairer for tilted fibres)
        n = ps._slide_normals[dom_slide]
        n = n / np.linalg.norm(n)
        dyad_ip = dyad - np.dot(dyad, n) * n
        psoct_ip = psoct_mean - np.dot(psoct_mean, n) * n
        if np.linalg.norm(dyad_ip) > 1e-6 and np.linalg.norm(psoct_ip) > 1e-6:
            ang_ip = acute_angle_deg(psoct_ip, dyad_ip)
            dyad_outofplane = float(np.degrees(np.arccos(
                min(1.0, abs(float(np.dot(dyad, n)))))))  # 90 = fully in-plane
        else:
            ang_ip = np.nan
            dyad_outofplane = np.nan

        rows.append({
            'x': int(v[0]), 'y': int(v[1]), 'z': int(v[2]),
            'f1': float(f1[v[0], v[1], v[2]]),
            'n_pix': int(len(pix_vecs)),
            'concentration': concentration,
            'R_circ': R_circ,
            'angle_3d_deg': ang3d,
            'angle_inplane_deg': ang_ip,
            'dyad_inplane_deg': dyad_outofplane,
            'psoct_x': psoct_mean[0], 'psoct_y': psoct_mean[1], 'psoct_z': psoct_mean[2],
            'dyad_x': dyad[0], 'dyad_y': dyad[1], 'dyad_z': dyad[2],
        })
        psoct_vecs.append(psoct_mean)
        dyad_vecs.append(dyad)

        if len(rows) % 100 == 0:
            print(f"      {len(rows)}/{target} voxels done", flush=True)

    n = len(rows)
    if n == 0:
        print("No voxels with both strong dMRI and PSOCT coverage found.")
        return
    print(f"      accepted {n} voxels")

    ang3d = np.array([r['angle_3d_deg'] for r in rows])
    ang_ip = np.array([r['angle_inplane_deg'] for r in rows])
    ang_ip = ang_ip[np.isfinite(ang_ip)]

    # shuffled-null control: pair each dyad with a *different* voxel's PSOCT mean
    P = np.array(psoct_vecs)
    D = np.array(dyad_vecs)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):      # ensure no self-pairs
        perm = rng.permutation(n)
    null3d = np.array([acute_angle_deg(P[perm[m]], D[m]) for m in range(n)])

    def stats(a):
        return dict(median=float(np.median(a)), mean=float(np.mean(a)),
                    p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                    frac_lt20=float(np.mean(a < 20)), frac_lt30=float(np.mean(a < 30)))

    s3d, sip, snull = stats(ang3d), stats(ang_ip), stats(null3d)

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"voxels compared : {n}")
    print(f"PSOCT pixels/vox: median {int(np.median([r['n_pix'] for r in rows]))}")
    print("\nIn-plane angle  PSOCT(circular mean) vs dyads1  [the headline]")
    print(f"  median {sip['median']:5.1f} deg | mean {sip['mean']:5.1f} | "
          f"IQR {sip['p25']:.1f}-{sip['p75']:.1f} | "
          f"<20deg {100*sip['frac_lt20']:.0f}% | <30deg {100*sip['frac_lt30']:.0f}%")
    print("Full 3D angle   PSOCT vs dyads1")
    print(f"  median {s3d['median']:5.1f} deg | mean {s3d['mean']:5.1f} | "
          f"IQR {s3d['p25']:.1f}-{s3d['p75']:.1f} | "
          f"<20deg {100*s3d['frac_lt20']:.0f}% | <30deg {100*s3d['frac_lt30']:.0f}%")
    print("Shuffled NULL   PSOCT vs *mismatched* dyad  (chance level)")
    print(f"  median {snull['median']:5.1f} deg | mean {snull['mean']:5.1f}")
    print("\nInterpretation: registered data clusters near 0 deg; the shuffled")
    print("null sits near 45 deg. A large gap = correct registration.")

    # --- outputs ------------------------------------------------------------
    print("\n[5/5] Writing figures and tables ...")
    out_dir = cfg['out_dir']
    if out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "registration_QC",
                               "Output_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    _plot(out_dir, ang_ip, ang3d, null3d, sip, s3d, snull, n, cfg)

    # per-voxel CSV
    csv_path = os.path.join(out_dir, "per_voxel_results.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # machine-readable summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        'n_voxels_compared': n,
        'inplane_angle_deg': sip,
        'angle_3d_deg': s3d,
        'shuffled_null_deg': snull,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nOutputs written to:\n  {out_dir}")
    print("  - orientation_agreement.png")
    print("  - per_voxel_results.csv")
    print("  - summary.json")
    print("Done.")


def _plot(out_dir, ang_ip, ang3d, null3d, sip, s3d, snull, n, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(0, 91, 3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(ang_ip, bins=bins, color="#2c7fb8", alpha=0.85,
            label=f"PSOCT vs dyads1 (in-plane)\nmedian {sip['median']:.1f}deg, "
                  f"<20deg {100*sip['frac_lt20']:.0f}%")
    ax.hist(null3d, bins=bins, color="#bdbdbd", alpha=0.6,
            label=f"shuffled null\nmedian {snull['median']:.1f}deg")
    ax.axvline(sip['median'], color="#2c7fb8", ls="--", lw=1.5)
    ax.axvline(45, color="grey", ls=":", lw=1)
    ax.set_xlabel("acute angle between orientations (deg)")
    ax.set_ylabel("voxels")
    ax.set_title(f"In-plane orientation agreement (n={n})")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(ang3d, bins=bins, color="#31a354", alpha=0.85,
            label=f"PSOCT vs dyads1 (full 3D)\nmedian {s3d['median']:.1f}deg")
    ax.hist(null3d, bins=bins, color="#bdbdbd", alpha=0.6,
            label=f"shuffled null\nmedian {snull['median']:.1f}deg")
    ax.axvline(s3d['median'], color="#31a354", ls="--", lw=1.5)
    ax.axvline(45, color="grey", ls=":", lw=1)
    ax.set_xlabel("acute angle between orientations (deg)")
    ax.set_ylabel("voxels")
    ax.set_title("Full-3D orientation agreement")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"PSOCT(circular mean) vs BedpostX dyads1  |  f1>{cfg['f_min']}, "
        f"within {cfg['radius']} vox of {tuple(cfg['center'])}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "orientation_agreement.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(CONFIG)
