"""
held_out.py -- Held-out interleaved validation (angular error).
===============================================================

Self-contained except for the heavy PS-OCT deps (fsl + the project's
``tractography`` package), so THIS is the one script that needs PYTHONPATH set:

    PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
    cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
    PYTHONPATH=TractographyAlgo/cmc_hybrid:TractographyAlgo \
        $PY TractographyAlgo/scripts/results/held_out.py

The tract is tracked on EVEN PS-OCT slices; each streamline tangent is compared,
in-plane, to the measured optic-axis orientation on the held-out ODD slices it
passes through. Reports the median acute in-plane angle.

Writes held_out.json, held_out.md and fig_held_out.png to OUTPUT_DIR.
"""

import os
import re
import glob
import json

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG  -- put the EXACT path to every file.
# =============================================================================
OUTPUT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/results/held_out"

# The tract tracked on EVEN slices (.trk).
HELD_OUT_TRK = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/Outputs_for_report/held_out_even/held_out_even.trk"

# bedpostX brain mask -- supplies the world<->voxel affine AND the mask handed
# to PSOCTData.
BRAIN_MASK = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP_uncompressed/nodif_brain_mask.nii"

# Held-out (ODD) PS-OCT slides.
# Point this at the DIRECTORY that holds the slide files; the script globs them.
HELD_OUT_PSOCT_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/PS-OCT/Coregistered/FLIRT/Orientations/lowres_odd"

# Glob matching the slide files inside that dir.
HELD_OUT_SLIDE_GLOB = "*.nii.gz"

# Fallback explicit list, used ONLY if HELD_OUT_PSOCT_DIR is empty/None.
HELD_OUT_PSOCT_SLIDES = []

# Optional: if the gathered slides hold ALL slides, keep only this parity
# (by index): None | "odd" | "even".
SLICE_PARITY = None

# Speed knobs.
VERTEX_STRIDE = 1          # sample every Nth streamline vertex
MAX_STREAMLINES = None     # cap streamlines processed (None = all)
# =============================================================================


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "[n/a]"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _natkey(path):
    """Natural sort so slide2 < slide10 (lexicographic would give slide10 < slide2)."""
    base = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", base)]


# --------------------------------------------------------------- analysis ----
def analyse():
    """Returns (result_dict, angles_array_or_None)."""
    if not os.path.exists(HELD_OUT_TRK):
        return {"skipped": f"HELD_OUT_TRK not found: {HELD_OUT_TRK}"}, None
    if not os.path.exists(BRAIN_MASK):
        return {"skipped": f"BRAIN_MASK not found: {BRAIN_MASK}"}, None

    if HELD_OUT_PSOCT_DIR and os.path.isdir(HELD_OUT_PSOCT_DIR):
        slides = sorted(
            glob.glob(os.path.join(HELD_OUT_PSOCT_DIR, HELD_OUT_SLIDE_GLOB)),
            key=_natkey,
        )
    else:
        slides = list(HELD_OUT_PSOCT_SLIDES)
    slides = [s for s in slides if os.path.exists(s)]
    if SLICE_PARITY in ("odd", "even") and slides:
        slides = slides[(1 if SLICE_PARITY == "odd" else 0)::2]
    if not slides:
        return {"skipped": "no held-out slides found (check HELD_OUT_PSOCT_DIR / glob)"}, None

    # Heavy deps -- need PYTHONPATH=TractographyAlgo/cmc_hybrid:TractographyAlgo.
    try:
        from fsl.data.image import Image
        from tractography.psoct import PSOCTData
    except Exception as e:
        return {"skipped": f"PSOCT deps unavailable ({e}); set PYTHONPATH"}, None

    affine = nib.load(BRAIN_MASK).affine
    aff_inv = np.linalg.inv(affine)
    ps = PSOCTData(slides, Image(BRAIN_MASK))
    tg = nib.streamlines.load(HELD_OUT_TRK)
    stride = max(1, int(VERTEX_STRIDE or 1))

    angles = []
    for s_idx, sl_world in enumerate(tg.streamlines):
        if MAX_STREAMLINES is not None and s_idx >= MAX_STREAMLINES:
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
        return {"skipped": "no streamline points adjacent to held-out slides"}, None
    a = np.asarray(angles)
    result = {
        "n_samples": int(a.size), "n_slides": len(slides),
        "median_deg": float(np.median(a)), "mean_deg": float(np.mean(a)),
        "p25_deg": float(np.percentile(a, 25)), "p75_deg": float(np.percentile(a, 75)),
        "frac_lt20": float(np.mean(a < 20)), "frac_lt30": float(np.mean(a < 30)),
        "definition": "acute in-plane angle (deg) between streamline tangent and "
                      "held-out PS-OCT optic axis, projected into the slide plane",
    }
    return result, a


import numpy as np
import matplotlib.pyplot as plt

def figure(angles, result, path):
    """Two-panel held-out summary: angle distribution + cumulative curve.

    Uses only the per-sample angles and the summary already in `result`
    (median_deg, p25_deg, p75_deg).
    """
    if angles is None or len(angles) == 0:
        return
    a = np.asarray(angles, dtype=float)
    med = result.get("median_deg", float(np.median(a)))
    p25 = result.get("p25_deg", float(np.percentile(a, 25)))
    p75 = result.get("p75_deg", float(np.percentile(a, 75)))
    col = "#2c7fb8"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- (left) distribution -------------------------------------------------
    ax1.hist(a, bins=np.arange(0, 91, 3), color=col, alpha=0.85,
             edgecolor="white", linewidth=0.4)
    ax1.axvspan(p25, p75, color=col, alpha=0.12, label=f"IQR {p25:.0f}-{p75:.0f}$\\degree$")
    ax1.axvline(med, color="k", ls="--", lw=1.6, label=f"median {med:.1f}$\\degree$")
    ax1.axvline(45, color="grey", ls=":", lw=1.3, label="chance (45$\\degree$)")
    ax1.set_xlim(0, 90)
    ax1.set_xlabel("acute in-plane angle: streamline vs held-out PS-OCT (deg)")
    ax1.set_ylabel("samples")
    ax1.set_title("Angle distribution", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    # The text box containing n, median, <20, and <30 has been removed.

    # --- (right) cumulative --------------------------------------------------
    x = np.sort(a)
    y = np.arange(1, x.size + 1) / x.size
    ax2.plot(x, y, color=col, lw=2.2)
    
    # Shade the IQR and add the median line
    ax2.axvspan(p25, p75, color=col, alpha=0.12, label=f"IQR {p25:.0f}-{p75:.0f}$\\degree$")
    ax2.axvline(med, color="k", ls="--", lw=1.6, label=f"median {med:.1f}$\\degree$")
    
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("acute in-plane angle (deg)")
    ax2.set_ylabel("cumulative fraction of samples")
    ax2.set_title("Cumulative distribution", fontweight="bold")
    
    # Added a legend here since the median and IQR are now plotted
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle("Held-out interleaved validation", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=800, bbox_inches="tight")
    plt.close(fig)

def markdown(result):
    L = ["## Held-out interleaved validation", ""]
    if "median_deg" in result:
        L.append(f"- HELD-OUT ANGLE = {fmt(result['median_deg'], 1)} deg "
                 f"(median acute in-plane; n={result['n_samples']} samples over "
                 f"{result['n_slides']} held-out slides; <30deg {100*result['frac_lt30']:.0f}%)")
    else:
        L.append(f"- (skipped: {result.get('skipped', 'not run')})")
    L.append("")
    return L


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== held-out angular error ===")
    result, angles = analyse()
    if "skipped" in result:
        print(f"  held-out: skipped ({result['skipped']})")
    else:
        print(f"  held-out: median {result['median_deg']:.1f} deg (n={result['n_samples']})")
    with open(os.path.join(OUTPUT_DIR, "held_out.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "held_out.md"), "w") as f:
        f.write("\n".join(markdown(result)) + "\n")
    figure(angles, result, os.path.join(OUTPUT_DIR, "fig_held_out.png"))
    print(f"  wrote held_out.json/.md and fig_held_out.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()