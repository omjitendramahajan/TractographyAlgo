"""
plot_distance_distribution.py — Registration-accuracy figure (in-plane only).

Samples N random sub-voxel query positions inside the brain mask, performs the
same nearest-PSOCT-pixel lookup used during tractography, and plots the
distribution of *in-plane* lookup distances — the component that registration
accuracy actually controls.

The raw 3D nearest-pixel distance conflates two unrelated sources of error:
  1. In-plane snap distance — bounded by the PSOCT in-plane pixel pitch and
     by the FLIRT registration accuracy.
  2. Through-plane snap distance — bounded by the inter-slide spacing of the
     PSOCT stack, which is a sampling property, not a registration property.

Because PSOCT measures only in-plane orientation (the through-plane component
is unobserved and assumed zero in the [cos θ, 0, −sin θ] construction), the
in-plane component is the only term that registration quality is responsible
for. We therefore decompose the 3D displacement at each query into in-plane
and through-plane components using the matched slide's normal vector, and
plot only the in-plane distance.

A well-registered pipeline produces a unimodal distribution whose mode and
median sit below ½ PSOCT pixel pitch.
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from fsl.data.image import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_single_voxel import get_nearest_psoct_orientation

SLIDES_DIR = Path("/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/PS-OCT/Coregistered/FLIRT/Orientations/lowres")
VOLUME_PATH = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP/nodif_brain_mask.nii.gz"

N_SAMPLES = 5000
SEED = 0


def psoct_pixel_pitch_in_voxels(slides, volume):
    """PSOCT in-plane pixel pitch (max of x, z axes) in dMRI voxel units."""
    pix2world = slides[0].getAffine('voxel', 'world')
    world2vox = volume.getAffine('world', 'voxel')
    M = world2vox @ pix2world
    pitch_x = float(np.linalg.norm(M[:3, 0]))
    pitch_z = float(np.linalg.norm(M[:3, 2]))
    return pitch_x, pitch_z


def slide_normals_in_voxels(slides, volume):
    """For each slide, the unit normal of its plane expressed in dMRI voxel space."""
    world2vox = volume.getAffine('world', 'voxel')
    normals = {}
    for i, slide in enumerate(slides):
        pix2world = slide.getAffine('voxel', 'world')
        M = world2vox @ pix2world  # pix → vox
        n = M[:3, 1]
        normals[i] = n / np.linalg.norm(n)
    return normals


def save_plot_data_to_csv(in_plane, out_dir):
    """Saves the raw data, histogram bins/counts, and ECDF data to CSV files."""
    print("\nSaving plot data to CSVs...")
    
    # 1. Save Histogram Data
    counts, bin_edges = np.histogram(in_plane, bins=80)
    hist_csv_path = out_dir / "histogram_data.csv"
    with open(hist_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Bin_Start_vox', 'Bin_End_vox', 'Count'])
        for i in range(len(counts)):
            writer.writerow([bin_edges[i], bin_edges[i+1], counts[i]])
    print(f"  -> Saved histogram data: {hist_csv_path.name}")

    # 2. Save ECDF Data
    sorted_in_plane = np.sort(in_plane)
    yvals = np.arange(1, len(sorted_in_plane) + 1) / len(sorted_in_plane)
    ecdf_csv_path = out_dir / "ecdf_data.csv"
    with open(ecdf_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['In_Plane_Distance_vox', 'Cumulative_Fraction'])
        for x, y in zip(sorted_in_plane, yvals):
            writer.writerow([x, y])
    print(f"  -> Saved ECDF data: {ecdf_csv_path.name}")
    
    # 3. Save Raw Distances
    raw_csv_path = out_dir / "raw_in_plane_distances.csv"
    with open(raw_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['In_Plane_Distance_vox'])
        for val in in_plane:
            writer.writerow([val])
    print(f"  -> Saved raw distance data: {raw_csv_path.name}")


def main():
    volume = Image(VOLUME_PATH)
    slides = [Image(str(p)) for p in sorted(SLIDES_DIR.glob("*.nii.gz"))]
    print(f"Loaded {len(slides)} slides, volume shape {volume.shape}")

    mask = volume.data > 0
    candidate = np.argwhere(mask)
    print(f"{len(candidate)} candidate voxels in brain mask")

    pitch_x, pitch_z = psoct_pixel_pitch_in_voxels(slides, volume)
    pixel_pitch = max(pitch_x, pitch_z)
    print(f"PSOCT pixel pitch (dMRI voxel units): x={pitch_x:.4f}, z={pitch_z:.4f}")

    normals = slide_normals_in_voxels(slides, volume)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(candidate), size=min(N_SAMPLES, len(candidate)), replace=False)
    sampled_vox = candidate[pick]

    in_plane = []
    through_plane = []

    print("\nProcessing samples...")
    for vox in sampled_vox:
        query = vox + rng.uniform(-0.5, 0.5, size=3)
        result = get_nearest_psoct_orientation(query, slides, volume)
        if result is None:
            continue
        
        d = result['nearest_vox'] - query
        n = normals[result['nearest_slide_idx']]
        d_through = abs(float(np.dot(d, n)))
        d_inplane = float(np.sqrt(max(0.0, np.dot(d, d) - d_through ** 2)))
        
        in_plane.append(d_inplane)
        through_plane.append(d_through)

    in_plane = np.array(in_plane)
    through_plane = np.array(through_plane)
    n_hits = len(in_plane)

    print(f"\nVoxels with PSOCT coverage: {n_hits}/{len(sampled_vox)} "
          f"({100*n_hits/len(sampled_vox):.1f}%)")
    print("In-plane lookup distance (dMRI voxel units):")
    print(f"  median:  {np.median(in_plane):.4f}")
    print(f"  mean:    {np.mean(in_plane):.4f}")
    print(f"  95th %:  {np.percentile(in_plane, 95):.4f}")
    print(f"  max:     {np.max(in_plane):.4f}")
    print("Through-plane snap distance (slide-spacing artefact, not registration):")
    print(f"  median:  {np.median(through_plane):.4f}")
    print(f"  95th %:  {np.percentile(through_plane, 95):.4f}")

    frac_subpixel = float(np.mean(in_plane < pixel_pitch))
    frac_halfpixel = float(np.mean(in_plane < pixel_pitch / 2))
    print(f"\nIn-plane within 1 PSOCT pixel:   {frac_subpixel:.1%}")
    print(f"In-plane within ½ PSOCT pixel:   {frac_halfpixel:.1%}")

    # ---- Setup Output Directory ----
    out_dir = PROJECT_ROOT / "tests" / "psoct_orientation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Plot Histogram ----
    fig_hist, ax_hist = plt.subplots(figsize=(9, 5))
    ax_hist.hist(in_plane, bins=80, color='steelblue', edgecolor='white')
    ax_hist.axvline(pixel_pitch, color='red', linestyle='--', linewidth=2,
                    label=f'PSOCT pixel pitch  ({pixel_pitch:.3f} vox)')
    ax_hist.axvline(pixel_pitch / 2, color='orange', linestyle='--', linewidth=2,
                    label=f'½ PSOCT pixel pitch ({pixel_pitch/2:.3f} vox)')
    ax_hist.axvline(np.median(in_plane), color='black', linestyle=':', linewidth=2,
                    label=f'median ({np.median(in_plane):.3f} vox)')
    ax_hist.set_xlabel('In-plane nearest-pixel distance (dMRI voxel units)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title(
        f'Registration accuracy — in-plane snap distance over {n_hits} random queries\n'
        f'{frac_subpixel:.1%} sub-pixel, {frac_halfpixel:.1%} within ½ pixel, '
        f'median {np.median(in_plane):.3f} vox'
    )
    ax_hist.legend()
    hist_out_path = out_dir / "nearest_pixel_distance_histogram.png"
    fig_hist.tight_layout()
    fig_hist.savefig(str(hist_out_path), dpi=150, bbox_inches='tight')
    print(f"Saved histogram plot to {hist_out_path}")

    # ---- 2. Plot ECDF (Publication Ready) ----
    fig_ecdf, ax_ecdf = plt.subplots(figsize=(6, 5))
    sorted_in_plane = np.sort(in_plane)
    yvals = np.arange(1, len(sorted_in_plane) + 1) / len(sorted_in_plane)

    ax_ecdf.plot(sorted_in_plane, yvals, linewidth=2.5, color='#2c7bb6') 
    ax_ecdf.axvline(pixel_pitch, color='#d7191c', linestyle='--', linewidth=1.5, 
                    label=f'1 Pixel Pitch ({pixel_pitch:.2f} vox)')
    ax_ecdf.axvline(pixel_pitch / 2, color='#fdae61', linestyle='--', linewidth=1.5, 
                    label=f'½ Pixel Pitch ({pixel_pitch/2:.2f} vox)')

    ax_ecdf.set_xlabel('In-plane nearest-pixel distance (voxels)', fontsize=12)
    ax_ecdf.set_ylabel('Cumulative fraction of brain mask', fontsize=12)
    ax_ecdf.set_xlim(0, np.percentile(in_plane, 99)) # Cut off extreme outliers for visual clarity
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.grid(True, linestyle=':', alpha=0.7)
    ax_ecdf.legend(frameon=False, loc='lower right', fontsize=11)
    ax_ecdf.spines['top'].set_visible(False)
    ax_ecdf.spines['right'].set_visible(False)

    ecdf_out_path = out_dir / "nearest_pixel_distance_ecdf.pdf"
    fig_ecdf.tight_layout()
    fig_ecdf.savefig(str(ecdf_out_path), format='pdf', dpi=300)
    print(f"Saved ECDF plot to {ecdf_out_path}")

    # ---- 3. Save Plot Data to CSV ----
    save_plot_data_to_csv(in_plane, out_dir)

    plt.show()


if __name__ == "__main__":
    main()