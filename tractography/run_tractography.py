"""
Probabilistic Tractography with XTRACT-Style Masks
===================================================

This tractography script uses PSOCTPriorityTracker (BedpostX-only mode when
psoct_slides is None) with support for:
- Target masks (waypoints - streamlines must pass through)
- Exclusion masks (streamlines entering are discarded)
- Stop masks (termination regions)

Usage:
    Edit the CONFIG dictionary below, then run:
    python run_tractography.py

Requirements:
    - nibabel
    - numpy
    - fslpy
"""

import sys
import os

# Add parent directory to path so we can import tractography package
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import json
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, save as save_tractogram

# Import from tractography package
from tractography.bedpostx import BedpostxData
from tractography.tracker import PSOCTPriorityTracker
from tractography.psoct import PSOCTData

# =============================================================================
# CONFIGURATION - Edit these directly, then run: python run_tractography.py
# =============================================================================
CONFIG = {
    # Data paths
    'bedpostx_dir': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP_uncompressed",
    'seed_mask': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/Registering_Masks/FMA/Moe_rebinarised/FMA_seed_Moe_bin.nii.gz",
    'output': "baseline_3.trk",
    
    # PSOCT coronal slides (list of paths, directory path, or None to disable)
    'psoct_slides': None, #"/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/PS-OCT/Coregistered/FLIRT/Orientations/lowres", 
    
    # XTRACT-style masks (set to None to disable)
    'target_mask': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/Registering_Masks/FMA/Moe_rebinarised/FMA_target_Moe_bin.nii.gz",
    'exclude_mask': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/Registering_Masks/FMA/Moe_rebinarised/FMA_exclude_Moe_bin.nii.gz",
    'stop_mask': None,  # Optional termination mask
    
    # Tracking parameters
    'step_size': 0.2,           # mm per step
    'max_steps': 2000,          # max steps per direction
    'seeds_per_voxel': 10,      # number of seeds per voxel
    'angle_threshold': 60,      # max angle between steps (degrees)
    'min_f_threshold': 0.05,    # minimum fiber fraction
    'psoct_inplane_threshold': 0.3,  # voxel-level gate: defer to BedpostX when
                                     # the f-weighted in-plane dyad magnitude
                                     # falls below this (PSOCT uninformative)
    'alpha': 0.5,               # hybrid blend coefficient: weights the BedpostX
                                # trajectory term vs the PS-OCT microscopy term.
                                # alpha=1 -> trajectory only (== baseline),
                                # alpha=0 -> PS-OCT alignment only.

    # Options
    'use_memory_map': True,     # True = low RAM, False = fast
    'num_fibers': 3,            # Number of fiber populations (1, 2, or 3)
}

def run_tractography(bedpostx_dir, seed_mask_path, output_path,
                     target_mask_path=None, exclude_mask_path=None, stop_mask_path=None,
                     psoct_slides=None,
                     step_size=0.2, max_steps=2000, seeds_per_voxel=1,
                     angle_threshold=60, min_f_threshold=0.05,
                     psoct_inplane_threshold=0.3, alpha=0.5,
                     use_memory_map=False, num_fibers=1):
    """
    Run probabilistic tractography with XTRACT-style masks.
    When psoct_slides is provided, uses PSOCT-constrained BedpostX sampling.
    """
    print("=" * 60)
    print("Probabilistic Tractography with Mask Support")
    print("=" * 60)
    
    # Load BEDPOSTX data
    print("\n[1/4] Loading BEDPOSTX data...")
    if use_memory_map:
        print("  (Memory-mapped mode - low RAM, slower per-voxel access)")
    bedpostx = BedpostxData(bedpostx_dir, num_fibers=num_fibers,
                            load_samples=True, load_dyads=True,
                            use_memory_map=use_memory_map)
    print(f"  Shape: {bedpostx.shape}")
    print(f"  Voxel size: {bedpostx.voxel_size}")
    
    # Load PSOCT data if provided
    psoct_data = None
    if psoct_slides is not None and len(psoct_slides) > 0:
        print("\n[PSOCT] Loading PSOCT coronal slides...")
        psoct_data = PSOCTData(psoct_slides, bedpostx.volume_img)
    
    # Load seed mask
    print("\n[2/4] Loading seed mask...")
    seed_img = nib.load(seed_mask_path)
    seed_data = seed_img.get_fdata() > 0
    seed_voxels = np.argwhere(seed_data)
    print(f"  Found {len(seed_voxels)} seed voxels")
    
    # Load optional masks
    print("\n[3/4] Loading XTRACT-style masks...")
    target_mask = None
    exclude_mask = None
    stop_mask = None
    
    if target_mask_path:
        target_mask = nib.load(target_mask_path).get_fdata() > 0
        print(f"  Target mask: {np.sum(target_mask)} voxels")
    else:
        print("  Target mask: None")
        
    if exclude_mask_path:
        exclude_mask = nib.load(exclude_mask_path).get_fdata() > 0
        print(f"  Exclude mask: {np.sum(exclude_mask)} voxels")
    else:
        print("  Exclude mask: None")
        
    if stop_mask_path:
        stop_mask = nib.load(stop_mask_path).get_fdata() > 0
        print(f"  Stop mask: {np.sum(stop_mask)} voxels")
    else:
        print("  Stop mask: None")
    
    # Initialize tracker
    print("\n[4/4] Running tractography...")
    use_psoct = psoct_data is not None
    print(f"  Mode: {'PSOCT-constrained BedpostX sampling' if use_psoct else 'Standard probabilistic (BedpostX only)'}")
    tracker = PSOCTPriorityTracker(
        bedpostx,
        psoct_data=psoct_data,
        step_size=step_size,
        max_steps=max_steps,
        min_f_threshold=min_f_threshold,
        angle_threshold=angle_threshold,
        psoct_inplane_threshold=psoct_inplane_threshold,
        alpha=alpha,
        target_mask=target_mask,
        exclude_mask=exclude_mask,
        stop_mask=stop_mask,
    )

    streamlines = []
    all_labels = []  # Source labels for PSOCT-constrained mode

    for idx, seed in enumerate(seed_voxels):
        if (idx + 1) % 100 == 0 or idx == 0 or (idx + 1) == len(seed_voxels):
            print(f"  Processing seed {idx + 1}/{len(seed_voxels)}", flush=True)

        for _ in range(seeds_per_voxel):
            # Add jitter for multiple seeds per voxel
            if seeds_per_voxel > 1:
                jittered_seed = seed + np.random.uniform(-0.5, 0.5, 3)
            else:
                jittered_seed = seed.astype(float)

            streamline, labels = tracker.track(jittered_seed)

            if streamline is not None and len(streamline) >= 2:
                streamlines.append(streamline)
                if labels is not None:
                    all_labels.append(labels)

    tracker.print_stats()
    
    if len(streamlines) == 0:
        print("\nNo streamlines generated! Check your seeds and parameters.")
        return
    
    # Create organized output directory structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "TRK_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Use timestamp to ensure unique output folder every time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(outputs_dir, f"Output_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Use just the filename for saving
    output_basename = os.path.basename(output_path)
    final_trk_path = os.path.join(output_folder, output_basename)
    final_json_path = os.path.join(output_folder, output_basename.replace('.trk', '_params.json'))
    
    # Save as TRK
    print(f"\nSaving to {final_trk_path}...")
    save_streamlines_trk(streamlines, bedpostx.affine, final_trk_path)
    
    # Save run parameters for reproducibility
    params = {
        'timestamp': datetime.now().isoformat(),
        'output_file': final_trk_path,
        'bedpostx_dir': bedpostx_dir,
        'seed_mask': seed_mask_path,
        'psoct_slides': psoct_slides,
        'target_mask': target_mask_path,
        'exclude_mask': exclude_mask_path,
        'stop_mask': stop_mask_path,
        'mode': 'psoct_constrained' if use_psoct else 'bedpostx_only',
        'parameters': {
            'step_size': step_size,
            'max_steps': max_steps,
            'seeds_per_voxel': seeds_per_voxel,
            'angle_threshold': angle_threshold,
            'min_f_threshold': min_f_threshold,
            'psoct_inplane_threshold': psoct_inplane_threshold,
            'alpha': alpha
        },
        'results': {
            'total_streamlines': len(streamlines),
            'total_seed_voxels': len(seed_voxels),
        },
        'data_info': {
            'volume_shape': list(bedpostx.shape),
            'voxel_size': list(bedpostx.voxel_size)
        }
    }
    
    # Add mode-specific stats
    if use_psoct:
        params['results']['psoct_constrained_steps'] = tracker.stats['psoct_steps']
        params['results']['bedpostx_only_steps'] = tracker.stats['bedpostx_steps']
    else:
        params['results']['valid_streamlines'] = tracker.stats['valid_streamlines']

    # Full termination/rejection breakdown
    params['results']['terminations'] = tracker.termination_summary()

    hist_path = os.path.join(output_folder,
                             output_basename.replace('.trk', '_terminations.png'))
    tracker.plot_termination_histogram(
        hist_path,
        title=f"Termination reasons — {output_basename}",
    )
    print(f"Termination histogram saved to {hist_path}")

    # Geometry + PSOCT-usage histograms
    geom_path = os.path.join(output_folder,
                             output_basename.replace('.trk', '_geometry.png'))
    if tracker.stats.plot_geometry_summary(
        geom_path,
        title=f"Streamline geometry — {output_basename}",
    ):
        print(f"Geometry summary saved to {geom_path}")

    # TDI-style voxel-visit map
    visit_path = os.path.join(output_folder,
                              output_basename.replace('.trk', '_visit_map.nii.gz'))
    if tracker.stats.save_visit_map(visit_path, bedpostx.affine):
        print(f"Voxel-visit map saved to {visit_path}")

    # Divergence map (only if PS-OCT was used)
    if use_psoct:
        div_path = os.path.join(output_folder,
                                output_basename.replace('.trk', '_divergence_counts.nii.gz'))
        if tracker.stats.save_divergence_map(div_path, bedpostx.affine):
            print(f"Divergence map saved to {div_path}")

    # Raw per-streamline arrays for offline figure-making
    npz_path = os.path.join(output_folder,
                            output_basename.replace('.trk', '_stats.npz'))
    tracker.stats.save_raw_arrays(npz_path)
    print(f"Raw stats arrays saved to {npz_path}")

    with open(final_json_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Parameters saved to {final_json_path}")
    print(f"\nOutput folder: {output_folder}")
    print("Done!")


def save_streamlines_trk(streamlines, affine, output_path):
    """Save streamlines to TRK format."""
    tractogram = Tractogram(
        streamlines=streamlines,
        affine_to_rasmm=affine
    )
    save_tractogram(tractogram, output_path)


if __name__ == "__main__":
    print("Running probabilistic tractography...")
    print(f"  Seed mask: {CONFIG['seed_mask']}")
    print(f"  PSOCT slides: {CONFIG['psoct_slides']}")
    print(f"  Target mask: {CONFIG['target_mask']}")
    print(f"  Exclude mask: {CONFIG['exclude_mask']}")
    print(f"  Output: {CONFIG['output']}")
    
    run_tractography(
        bedpostx_dir=CONFIG['bedpostx_dir'],
        seed_mask_path=CONFIG['seed_mask'],
        output_path=CONFIG['output'],
        target_mask_path=CONFIG['target_mask'],
        exclude_mask_path=CONFIG['exclude_mask'],
        stop_mask_path=CONFIG['stop_mask'],
        psoct_slides=CONFIG['psoct_slides'],
        step_size=CONFIG['step_size'],
        max_steps=CONFIG['max_steps'],
        seeds_per_voxel=CONFIG['seeds_per_voxel'],
        angle_threshold=CONFIG['angle_threshold'],
        min_f_threshold=CONFIG['min_f_threshold'],
        psoct_inplane_threshold=CONFIG['psoct_inplane_threshold'],
        alpha=CONFIG['alpha'],
        use_memory_map=CONFIG['use_memory_map'],
        num_fibers=CONFIG['num_fibers']
    )