#!/usr/bin/env python
"""
PSOCT-Priority Tractography Algorithm
======================================

This tractography algorithm uses PSOCT (microscopy) orientations when available,
falling back to BEDPOSTX (dMRI) orientations only where microscopy data is absent.

The PSOCT data is NOT diluted or blended with diffusion data - it takes full priority.

Usage:
    python custom_tractography.py \\
        --bedpostx /path/to/data.bedpostX \\
        --psoct /path/to/psoct/Slice*_header.nii.gz \\
        --seeds seeds.nii.gz \\
        --output output.trk

Requirements:
    - nibabel
    - numpy
    - scipy
    - fslpy
    - cmc_hybrid (pip install -e ./cmc_hybrid)
"""

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, save as save_tractogram

# Import from tractography package
from tractography import BedpostxData, PSOCTData, PSOCTPriorityTracker

# =============================================================================
# CONFIGURATION - Edit these directly, then run: python custom_tractography.py
# =============================================================================
CONFIG = {
    # Data paths
    'bedpostx_dir': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP_uncompressed",
    'psoct_pattern': None,  # Set to glob pattern for PSOCT files, or None to skip
    'seed_mask': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/test_seed.nii.gz",
    'output': "streamlines_new.trk",
    
    # Tracking parameters
    'step_size': 0.1,          # mm per step
    'max_steps': 2000,        # max steps per direction
    'seeds_per_voxel': 100,      # number of seeds per voxel
    'angle_threshold': 60,     # max angle between steps (degrees)
    
    # Memory options
    'use_memory_map': True,    # True = low RAM (~100MB), False = fast (~15GB)
    'deterministic': False,    # True = use mean orientations, False = probabilistic
    'num_fibers': 1,           # Number of fiber populations (1, 2, or 3)
}


def run_tractography(bedpostx_dir, psoct_pattern, seed_mask_path, output_path,
                     step_size=0.5, max_steps=2000, seeds_per_voxel=1,
                     deterministic=False, use_memory_map=False, num_fibers=1):
    """
    Run PSOCT-priority tractography and save the result.
    """
    print("=" * 60)
    print("PSOCT-Priority Tractography")
    print("=" * 60)
    
    # Load BEDPOSTX data
    print("\n[1/4] Loading BEDPOSTX data...")
    load_samples = not deterministic
    if deterministic:
        print("  (Deterministic mode - skipping posterior samples)")
    if use_memory_map:
        print("  (Memory-mapped mode - low RAM, slower per-voxel access)")
    bedpostx = BedpostxData(bedpostx_dir, num_fibers=num_fibers,
                            load_samples=load_samples, 
                            load_dyads=deterministic, use_memory_map=use_memory_map)
    print(f"  Shape: {bedpostx.shape}")
    print(f"  Voxel size: {bedpostx.voxel_size}")
    
    # Load PSOCT data (optional)
    psoct = None
    if psoct_pattern:
        print("\n[2/4] Loading PSOCT data...")
        psoct_files = sorted(glob.glob(psoct_pattern))
        if len(psoct_files) > 0:
            psoct = PSOCTData(psoct_files, bedpostx.volume_img)
        else:
            print(f"  Warning: No files match pattern '{psoct_pattern}'")
    else:
        print("\n[2/4] PSOCT data: Not provided")
    
    # Load seed mask
    print("\n[3/4] Loading seed mask...")
    seed_img = nib.load(seed_mask_path)
    seed_mask = seed_img.get_fdata() > 0
    seed_voxels = np.argwhere(seed_mask)
    print(f"  Found {len(seed_voxels)} seed voxels")
    
    # Initialize tracker
    print("\n[4/4] Running tractography...")
    tracker = PSOCTPriorityTracker(
        bedpostx,
        psoct,
        step_size=step_size,
        max_steps=max_steps
    )
    
    streamlines = []
    all_labels = []
    
    for idx, seed in enumerate(seed_voxels):
        if (idx + 1) % 10 == 0 or idx == 0 or (idx + 1) == len(seed_voxels):
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
                all_labels.append(labels)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Total streamlines: {len(streamlines)}")
    print(f"  Steps using PSOCT: {tracker.stats['psoct_steps']}")
    print(f"  Steps using BEDPOSTX: {tracker.stats['bedpostx_steps']}")
    
    total_steps = tracker.stats['psoct_steps'] + tracker.stats['bedpostx_steps']
    psoct_pct = 0
    if total_steps > 0:
        psoct_pct = 100 * tracker.stats['psoct_steps'] / total_steps
        print(f"  PSOCT usage: {psoct_pct:.1f}%")
    
    if len(streamlines) == 0:
        print("\nNo streamlines generated! Check your seeds and parameters.")
        return
    
    # Create organized output directory structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "TRK_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Find next available output number
    existing = [d for d in os.listdir(outputs_dir) 
                if d.startswith("Output") and os.path.isdir(os.path.join(outputs_dir, d))]
    if existing:
        nums = [int(d.replace("Output", "")) for d in existing if d.replace("Output", "").isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1
    
    # Create output subfolder
    output_folder = os.path.join(outputs_dir, f"Output{next_num}")
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
        'psoct_pattern': psoct_pattern,
        'seed_mask': seed_mask_path,
        'parameters': {
            'step_size': step_size,
            'max_steps': max_steps,
            'seeds_per_voxel': seeds_per_voxel,
            'deterministic': deterministic
        },
        'results': {
            'total_streamlines': len(streamlines),
            'total_seed_voxels': len(seed_voxels),
            'psoct_steps': tracker.stats['psoct_steps'],
            'bedpostx_steps': tracker.stats['bedpostx_steps'],
            'psoct_usage_pct': psoct_pct
        },
        'data_info': {
            'volume_shape': list(bedpostx.shape),
            'voxel_size': list(bedpostx.voxel_size),
            'num_psoct_slides': len(psoct.slides) if psoct else 0
        }
    }
    
    with open(final_json_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Parameters saved to {final_json_path}")
    print(f"\nOutput folder: {output_folder}")
    print("Done!")


def save_streamlines_trk(streamlines, affine, output_path):
    """Save streamlines to TRK format."""
    # Convert voxel coordinates to world coordinates
    world_streamlines = []
    for sl in streamlines:
        ones = np.ones((len(sl), 1))
        vox_homo = np.hstack([sl, ones])
        world = (affine @ vox_homo.T).T[:, :3]
        world_streamlines.append(world)
    
    # Create and save tractogram
    tractogram = Tractogram(
        streamlines=world_streamlines,
        affine_to_rasmm=np.eye(4)
    )
    save_tractogram(tractogram, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="PSOCT-Priority Tractography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This algorithm uses PSOCT (microscopy) orientations exclusively where available,
falling back to BEDPOSTX (dMRI) only where microscopy data is absent.

The PSOCT data is NOT blended or diluted with diffusion data.

Examples:
    # With PSOCT data
    python custom_tractography.py \\
        --bedpostx /path/to/data.bedpostX \\
        --psoct "/path/to/psoct/Slice*_header.nii.gz" \\
        --seeds seeds.nii.gz \\
        --output tracts.trk

    # Without PSOCT (BEDPOSTX only)
    python custom_tractography.py \\
        --bedpostx /path/to/data.bedpostX \\
        --seeds seeds.nii.gz \\
        --output tracts.trk
        """
    )
    
    parser.add_argument("--bedpostx", "-b", default=DEFAULT_BEDPOSTX,
                        help=f"Path to .bedpostX directory (default: {DEFAULT_BEDPOSTX})")
    parser.add_argument("--psoct", "-p", default=DEFAULT_PSOCT,
                        help=f"Glob pattern for PSOCT slides (default: uses preset path)")
    parser.add_argument("--seeds", "-s", required=True,
                        help="Seed mask (NIfTI file)")
    parser.add_argument("--output", "-o", default="streamlines.trk",
                        help="Output file path (default: streamlines.trk)")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Step size in mm (default: 0.5)")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Maximum steps per direction (default: 2000)")
    parser.add_argument("--seeds-per-voxel", type=int, default=1,
                        help="Seeds per voxel (default: 1)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic tracking (lower memory)")
    parser.add_argument("--memory-map", action="store_true",
                        help="Use memory-mapped loading (reduces RAM from ~15GB to ~100MB, slower)")
    
    args = parser.parse_args()
    
    run_tractography(
        bedpostx_dir=args.bedpostx,
        psoct_pattern=args.psoct,
        seed_mask_path=args.seeds,
        output_path=args.output,
        step_size=args.step_size,
        max_steps=args.max_steps,
        seeds_per_voxel=args.seeds_per_voxel,
        deterministic=args.deterministic,
        use_memory_map=args.memory_map
    )


if __name__ == "__main__":
    import sys
    
    # If no CLI args provided, use embedded CONFIG
    if len(sys.argv) == 1:
        print("Running with embedded configuration...")
        print(f"  Seed mask: {CONFIG['seed_mask']}")
        print(f"  Output: {CONFIG['output']}")
        
        run_tractography(
            bedpostx_dir=CONFIG['bedpostx_dir'],
            psoct_pattern=CONFIG['psoct_pattern'],
            seed_mask_path=CONFIG['seed_mask'],
            output_path=CONFIG['output'],
            step_size=CONFIG['step_size'],
            max_steps=CONFIG['max_steps'],
            seeds_per_voxel=CONFIG['seeds_per_voxel'],
            deterministic=CONFIG['deterministic'],
            use_memory_map=CONFIG['use_memory_map'],
            num_fibers=CONFIG['num_fibers']
        )
    else:
        # Use CLI parser if args provided
        main()
