"""
plot_single_voxel.py — Sub-voxel PSOCT orientation lookup (tractography-style).

Given ONLY a continuous sub-voxel position (as in tractography), finds the
nearest PSOCT pixel orientation. The enclosing voxel is derived automatically.

Plots:
  - All PSOCT orientation vectors in the enclosing voxel (blue)
  - The nearest pixel highlighted (green, thick)
  - The query point (orange star)
  - The dMRI voxel boundary (red rectangle)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from fsl.data.image import Image
from fsl.transform.affine import concat, transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cmc_hybrid.coordinate_mapping import vox_to_pix_per_slide, cube_vertices
from cmc_hybrid.utils import fudge_psoct_orientation

# ---- Paths ----
SLIDES_DIR = Path("/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/PS-OCT/Coregistered/FLIRT/Orientations/lowres")
VOLUME_PATH = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP/nodif_brain_mask.nii.gz"


def get_nearest_psoct_orientation(query_pos, slides, volume):
    """
    Tractography-style lookup: given a continuous position in voxel space,
    find the nearest PSOCT pixel orientation.

    Args:
        query_pos: (3,) continuous voxel-space position, e.g. [173.5, 173.2, 54.3]
        slides:    list of PSOCT slide Image objects
        volume:    reference volume Image object

    Returns:
        dict with keys:
          'nearest_vox'    — (3,) voxel-space coords of nearest pixel
          'nearest_pix'    — (3,) slide pixel coords of nearest pixel
          'theta_raw'      — raw PSOCT angle at that pixel
          'theta_corrected'— corrected angle after fudge_psoct_orientation
          'distance'       — distance in voxel units from query to nearest pixel
          'enclosing_voxel'— the integer voxel used for the search
        or None if no PSOCT data found.
    """
    # Derive the enclosing voxel (same rounding as in tractography)
    enclosing_vox = [int(round(v)) for v in query_pos]

    # Get all PSOCT pixels in this voxel
    per_slide = vox_to_pix_per_slide(enclosing_vox, slides, volume)
    if not per_slide:
        return None

    # Search across all slides for the nearest pixel to query_pos
    query = np.array(query_pos)
    best_dist = np.inf
    best_pix = best_vox = best_theta = None
    best_slide_idx = None

    for slide_idx, (pixgrid, voxgrid, theta) in per_slide.items():
        dists = np.linalg.norm(voxgrid - query, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < best_dist:
            best_dist = dists[idx]
            best_pix = pixgrid[idx]
            best_vox = voxgrid[idx]
            best_theta = theta[idx]
            best_slide_idx = slide_idx

    theta_corr = fudge_psoct_orientation(np.array([best_theta]))[0]

    return {
        'nearest_vox': best_vox,
        'nearest_pix': best_pix,
        'nearest_slide_idx': best_slide_idx,
        'theta_raw': best_theta,
        'theta_corrected': theta_corr,
        'distance': best_dist,
        'enclosing_voxel': enclosing_vox,
    }


def main():
    volume = Image(VOLUME_PATH)
    slides = [Image(str(p)) for p in sorted(SLIDES_DIR.glob("*.nii.gz"))]
    print(f"Loaded {len(slides)} slides, volume shape {volume.shape}")

    # ---------------------------------------------------------
    # ONLY INPUT: a continuous position (as during tractography)
    # ---------------------------------------------------------
    query_pos = [173.5, 173.2, 54.6]
    # ---------------------------------------------------------

    # Lookup
    result = get_nearest_psoct_orientation(query_pos, slides, volume)
    if result is None:
        print(f"No PSOCT data at position {query_pos}")
        return

    enclosing_vox = result['enclosing_voxel']
    print(f"\nQuery position:     {query_pos}")
    print(f"Enclosing voxel:    {enclosing_vox}")
    print(f"Nearest pixel (vox): ({result['nearest_vox'][0]:.2f}, "
          f"{result['nearest_vox'][1]:.2f}, {result['nearest_vox'][2]:.2f})")
    print(f"Distance:           {result['distance']:.3f} voxels")
    print(f"Raw theta:          {result['theta_raw']:.4f} rad "
          f"({np.degrees(result['theta_raw']):.1f}°)")
    print(f"Corrected theta:    {result['theta_corrected']:.4f} rad "
          f"({np.degrees(result['theta_corrected']):.1f}°)")

    # ---- Plot ----
    per_slide = vox_to_pix_per_slide(enclosing_vox, slides, volume)
    vox2world = volume.getAffine('voxel', 'world')
    cube_verts = np.array(cube_vertices(enclosing_vox, 1))

    n = len(per_slide)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), squeeze=False)

    for col, (slide_idx, (pixgrid, voxgrid, theta)) in enumerate(per_slide.items()):
        ax = axes[0, col]
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # Voxel boundary in slide pixel coords
        world2pix = slides[slide_idx].getAffine('world', 'voxel')
        vox2pix = concat(world2pix, vox2world)
        bounds = transform(cube_verts, vox2pix)
        bx = [bounds[:, 0].min(), bounds[:, 0].max(), bounds[:, 0].max(),
              bounds[:, 0].min(), bounds[:, 0].min()]
        bz = [bounds[:, 2].min(), bounds[:, 2].min(), bounds[:, 2].max(),
              bounds[:, 2].max(), bounds[:, 2].min()]

        # All orientations (blue)
        theta_corr = fudge_psoct_orientation(theta)
        vx, vz = np.cos(theta_corr), -np.sin(theta_corr)
        X, Z = pixgrid[:, 0], pixgrid[:, 2]
        half = 0.4
        segs = [[(xi - half*dx, zi - half*dz), (xi + half*dx, zi + half*dz)]
                for xi, zi, dx, dz in zip(X, Z, vx, vz)]
        ax.add_collection(LineCollection(segs, colors='blue', linewidths=1.0, zorder=4))

        # Highlight nearest pixel (green) if on this slide
        match = np.where(np.all(np.isclose(pixgrid, result['nearest_pix']), axis=1))[0]
        if len(match) > 0:
            ni = match[0]
            nvx = np.cos(result['theta_corrected'])
            nvz = -np.sin(result['theta_corrected'])
            ax.plot([X[ni] - half*nvx, X[ni] + half*nvx],
                    [Z[ni] - half*nvz, Z[ni] + half*nvz],
                    color='lime', linewidth=4, zorder=6)
            ax.plot(X[ni], Z[ni], 'o', color='lime', markersize=6, zorder=7)

            # Query point projected to slide pixel space
            q_pix = transform([query_pos], vox2pix)[0]
            ax.plot(q_pix[0], q_pix[2], '*', color='orange', markersize=15, zorder=8)

        ax.plot(bx, bz, color='red', linewidth=2, zorder=5)

        pad = 2
        ax.set_xlim(min(np.min(X), min(bx)) - pad, max(np.max(X), max(bx)) + pad)
        ax.set_ylim(min(np.min(Z), min(bz)) - pad, max(np.max(Z), max(bz)) + pad)
        ax.set_xlabel("Slide Pixel X")
        ax.set_ylabel("Slide Pixel Z")
        ax.set_title(f"Slide {slide_idx+1}  ({len(theta)} pixels)")

    fig.suptitle(f"Sub-voxel PSOCT Lookup — query {query_pos}", fontsize=14, y=1.02)
    # fig.legend(handles=[
    #     Line2D([0], [0], color='blue', lw=1.5, label='All PSOCT orientations'),
    #     Line2D([0], [0], color='lime', lw=4, label='Nearest pixel orientation'),
    #     Line2D([0], [0], marker='*', color='orange', lw=0, markersize=12, label='Query point'),
    #     Line2D([0], [0], color='red', lw=2, label='dMRI voxel boundary'),
    # ], loc='lower center', ncol=4, framealpha=0.9)

    out_dir = PROJECT_ROOT / "tests" / "psoct_orientation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "plot_subvoxel_lookup.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
