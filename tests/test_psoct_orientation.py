#!/usr/bin/env python
"""
test_psoct_orientation.py — Visualise PSOCT orientation extraction.

Uses the cmc_hybrid test data to demonstrate the full pipeline:
  vox_to_pix → fudge_psoct_orientation → angle_to_vector → make_dyads

Produces three figures:
  1. 2D quiver overlay on a mid-slice of the PSOCT data
  2. 3D quiver of extracted orientation vectors in volume space
  3. Histogram of raw vs corrected theta distributions

Usage:
    python tests/test_psoct_orientation.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from fsl.data.image import Image

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import cmc_hybrid functions directly
from cmc_hybrid.coordinate_mapping import vox_to_pix, slide_to_volume, angle_to_vector
from cmc_hybrid.utils import fudge_psoct_orientation, make_dyads

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TESTDATA_DIR = PROJECT_ROOT / "cmc_hybrid" / "cmc_hybrid" / "tests" / "testdata"
OUTPUT_DIR   = PROJECT_ROOT / "tests" / "psoct_orientation_results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_data():
    """Load the cmc_hybrid test data (2 slices + volume + mask)."""
    volume = Image(str(TESTDATA_DIR / "volume"))
    mask   = Image(str(TESTDATA_DIR / "mask"))
    slides = [
        Image(str(TESTDATA_DIR / "slice1")),
        Image(str(TESTDATA_DIR / "slice2")),
    ]
    print(f"Volume shape : {volume.shape},  pixdim : {volume.pixdim}")
    print(f"Mask shape   : {mask.shape}")
    for i, sl in enumerate(slides):
        print(f"Slide {i+1} shape: {sl.shape},  pixdim : {sl.pixdim}")
    return volume, mask, slides


def extract_orientations(volume, mask, slides):
    """
    Run the PSOCT orientation pipeline on all masked voxels.

    Returns
    -------
    results : list of dicts with keys:
        voxel, raw_theta, corrected_theta, vectors_3d, dyad
    """
    mask_data = mask.data
    voxels = np.argwhere(mask_data > 0)
    xform  = slide_to_volume(slides[0], volume)

    results = []
    n_empty = 0

    print(f"\nProcessing {len(voxels)} masked voxels ...")
    for vox in voxels:
        vox_list = vox.tolist()
        pixgrid, voxgrid, theta_raw = vox_to_pix(vox_list, slides, volume)

        if len(theta_raw) < 3:
            n_empty += 1
            continue

        theta_raw = np.array(theta_raw)
        theta_corrected = fudge_psoct_orientation(theta_raw)
        vecs = angle_to_vector(theta_corrected, xform=xform)
        dyad = make_dyads(vecs)
        norm = np.linalg.norm(dyad)
        if norm > 0:
            dyad = dyad / norm

        results.append({
            "voxel":           vox,
            "raw_theta":       theta_raw,
            "corrected_theta": theta_corrected,
            "vectors_3d":      vecs,
            "dyad":            dyad,
        })

    print(f"  Voxels with PSOCT coverage: {len(results)}")
    print(f"  Voxels without coverage   : {n_empty}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_2d_quiver(results, slides, volume, output_dir):
    """
    Plot a 2D quiver of orientation vectors overlaid on a PSOCT slice image.
    Shows the mid-Y (coronal) slice of the volume with arrows at each voxel.
    """
    # Collect voxel positions and 3D dyads
    positions = np.array([r["voxel"] for r in results])
    dyads     = np.array([r["dyad"]  for r in results])

    # Pick the most common Y coordinate to define the "slice"
    y_coords = positions[:, 1]
    unique_y, counts = np.unique(y_coords, return_counts=True)
    best_y = unique_y[np.argmax(counts)]
    mask_y = y_coords == best_y

    pos_2d = positions[mask_y]
    dya_2d = dyads[mask_y]

    # Load the first slide's image data as background
    slide_data = slides[0].data
    # For the 2D overlay, use the slide's x-z plane
    if slide_data.ndim == 3:
        mid_slice_img = slide_data[:, slide_data.shape[1] // 2, :]
    else:
        mid_slice_img = slide_data

    fig, ax = plt.subplots(figsize=(10, 8))
    # Background: show the slide data (use a relevant slice if 3D)
    ax.imshow(mid_slice_img.T, origin="lower", cmap="gray", alpha=0.6, aspect="auto")

    # Overlay quiver — use X and Z components of dyad for the coronal plane
    ax.quiver(
        pos_2d[:, 0], pos_2d[:, 2],  # voxel X, Z
        dya_2d[:, 0], dya_2d[:, 2],  # dyad components in X, Z
        color="cyan", scale=10, headwidth=3, headlength=4, linewidth=1.2,
        label=f"PSOCT orientations (Y = {best_y})"
    )

    ax.set_xlabel("Voxel X")
    ax.set_ylabel("Voxel Z")
    ax.set_title(f"PSOCT In-Plane Orientations — Coronal Slice (Y = {best_y})")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = output_dir / "quiver_2d.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_3d_quiver(results, output_dir):
    """
    3D scatter + quiver of all orientation vectors in volume space.
    """
    positions = np.array([r["voxel"] for r in results], dtype=float)
    dyads     = np.array([r["dyad"]  for r in results])

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Colour by elevation angle (Z-component of dyad)
    colours = np.abs(dyads[:, 2])

    ax.quiver(
        positions[:, 0], positions[:, 1], positions[:, 2],
        dyads[:, 0],     dyads[:, 1],     dyads[:, 2],
        length=0.8, normalize=True, linewidth=0.8,
        color=plt.cm.coolwarm(colours), arrow_length_ratio=0.15,
    )

    ax.set_xlabel("Voxel I")
    ax.set_ylabel("Voxel J")
    ax.set_zlabel("Voxel K")
    ax.set_title("PSOCT 3D Orientation Vectors (colour = |elevation|)")
    fig.tight_layout()

    out_path = output_dir / "quiver_3d.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"  Saved → {out_path}")
    plt.close(fig)


def plot_angle_histogram(results, output_dir):
    """
    Histograms comparing raw theta vs corrected theta distributions.
    """
    raw_all       = np.concatenate([r["raw_theta"]       for r in results])
    corrected_all = np.concatenate([r["corrected_theta"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw theta
    axes[0].hist(np.degrees(raw_all), bins=60, color="steelblue",
                 edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Theta (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw PSOCT Theta Distribution")
    axes[0].axvline(np.degrees(np.mean(raw_all)), color="red", ls="--",
                    label=f"mean = {np.degrees(np.mean(raw_all)):.1f}°")
    axes[0].legend()

    # Corrected theta
    axes[1].hist(np.degrees(corrected_all), bins=60, color="darkorange",
                 edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Corrected PSOCT Theta Distribution\n(after fudge_psoct_orientation)")
    axes[1].axvline(np.degrees(np.mean(corrected_all)), color="red", ls="--",
                    label=f"mean = {np.degrees(np.mean(corrected_all)):.1f}°")
    axes[1].legend()

    fig.suptitle("Effect of orientation correction on PSOCT angles", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "angle_histogram.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print concise pipeline summary."""
    raw_all       = np.concatenate([r["raw_theta"]       for r in results])
    corrected_all = np.concatenate([r["corrected_theta"] for r in results])
    dyads         = np.array([r["dyad"]                  for r in results])

    print("\n" + "=" * 50)
    print("PSOCT Orientation Extraction — Summary")
    print("=" * 50)
    print(f"  Voxels with PSOCT data : {len(results)}")
    print(f"  Total PSOCT pixels     : {len(raw_all)}")
    print(f"  Pixels per voxel (mean): {len(raw_all)/len(results):.1f}")
    print(f"  Raw theta  — mean: {np.degrees(np.mean(raw_all)):7.2f}°, "
          f"std: {np.degrees(np.std(raw_all)):7.2f}°")
    print(f"  Corrected  — mean: {np.degrees(np.mean(corrected_all)):7.2f}°, "
          f"std: {np.degrees(np.std(corrected_all)):7.2f}°")
    print(f"  Dyad |X̄|={np.mean(np.abs(dyads[:,0])):.3f}, "
          f"|Ȳ|={np.mean(np.abs(dyads[:,1])):.3f}, "
          f"|Z̄|={np.mean(np.abs(dyads[:,2])):.3f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PSOCT Orientation Extraction — Visualisation Script")
    print("=" * 60)

    # 1) Load data
    volume, mask, slides = load_test_data()

    # 2) Extract orientations
    results = extract_orientations(volume, mask, slides)

    if len(results) == 0:
        print("No voxels with PSOCT coverage found — nothing to visualise.")
        sys.exit(1)

    # 3) Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4) Plot
    print("\nGenerating plots ...")
    plot_2d_quiver(results, slides, volume, OUTPUT_DIR)
    plot_3d_quiver(results, OUTPUT_DIR)
    plot_angle_histogram(results, OUTPUT_DIR)

    # 5) Summary
    print_summary(results)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
