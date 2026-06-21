"""
Probabilistic Tractography with XTRACT-Style Masks (HPC Version)
===================================================

This script is adapted for running on the HPC by taking input paths from command line arguments.
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
import argparse

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram, save as save_tractogram

# Import from tractography package
from tractography.bedpostx import BedpostxData
from tractography.tracker import PSOCTPriorityTracker
from tractography.psoct import PSOCTData

def run_tractography(bedpostx_dir, seed_mask_path, output_path,
                     target_mask_path=None, exclude_mask_path=None, stop_mask_path=None,
                     psoct_slides=None,
                     step_size=0.2, max_steps=2000, seeds_per_voxel=1,
                     angle_threshold=60, min_f_threshold=0.05,
                     psoct_inplane_threshold=0.3, alpha=0.5,
                     use_memory_map=True, num_fibers=3, output_dir=None):
    """
    Run probabilistic tractography with XTRACT-style masks.
    """
    print("=" * 60)
    print("Probabilistic Tractography with Mask Support (HPC Version)")
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
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "TRK_outputs")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Use timestamp to ensure unique output folder every time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_dir, f"Output_{timestamp}")
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


def str2bool(v):
    """Parse a truthy/falsy string (so PBS can pass e.g. --use_memory_map false)."""
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y'):
        return True
    if str(v).strip().lower() in ('0', 'false', 'f', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean string, got '{v}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probabilistic tractography (HPC)")
    parser.add_argument('--bedpostx_dir', type=str, required=True, help="Path to bedpostX directory")
    parser.add_argument('--seed_mask', type=str, required=True, help="Path to seed mask NIfTI file")
    parser.add_argument('--output', type=str, required=True, help="Output tractogram basename (.trk)")
    parser.add_argument('--target_mask', type=str, default=None, help="Path to target mask NIfTI file")
    parser.add_argument('--exclude_mask', type=str, default=None, help="Path to exclude mask NIfTI file")
    parser.add_argument('--stop_mask', type=str, default=None, help="Path to stop mask NIfTI file")
    parser.add_argument('--psoct_slides', type=str, default=None, help="Path to PSOCT slides directory (optional)")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save the Output_timestamp folder in")
    
    # Optional tracker params
    parser.add_argument('--step_size', type=float, default=0.2, help="Step size in mm")
    parser.add_argument('--max_steps', type=int, default=2000, help="Max steps per streamline direction")
    parser.add_argument('--seeds_per_voxel', type=int, default=10, help="Number of seeds per voxel")
    parser.add_argument('--angle_threshold', type=float, default=60.0, help="Max angle between steps (degrees)")
    parser.add_argument('--min_f_threshold', type=float, default=0.05, help="Minimum fiber fraction")
    parser.add_argument('--psoct_inplane_threshold', type=float, default=0.3,
                        help="Voxel-level gate: defer to BedpostX when the f-weighted "
                             "in-plane dyad magnitude falls below this (PSOCT only)")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Hybrid blend coefficient: trajectory vs PS-OCT term. "
                             "alpha=1 -> trajectory only (==baseline), alpha=0 -> PS-OCT only")
    parser.add_argument('--num_fibers', type=int, default=3, help="Number of fibers to use in BEDPOSTX data")
    parser.add_argument('--use_memory_map', type=str2bool, default=True,
                        help="Memory-mapped loading: true = low RAM / slower per-voxel, "
                             "false = preload arrays (fast, high RAM)")
    
    args = parser.parse_args()

    print("Running probabilistic tractography (HPC Version)...")
    print(f"  BedpostX dir: {args.bedpostx_dir}")
    print(f"  Seed mask: {args.seed_mask}")
    print(f"  Target mask: {args.target_mask}")
    print(f"  Exclude mask: {args.exclude_mask}")
    print(f"  Output basename: {args.output}")
    print(f"  Output directory: {args.output_dir if args.output_dir else 'Default TRK_outputs'}")
    print(f"  Stop mask: {args.stop_mask}")
    print(f"  PSOCT slides: {args.psoct_slides if args.psoct_slides else 'None (BedpostX-only)'}")
    print(f"  step_size: {args.step_size} | max_steps: {args.max_steps} | "
          f"seeds_per_voxel: {args.seeds_per_voxel}")
    print(f"  angle_threshold: {args.angle_threshold} | min_f_threshold: {args.min_f_threshold} | "
          f"num_fibers: {args.num_fibers}")
    print(f"  psoct_inplane_threshold: {args.psoct_inplane_threshold} | alpha (blend): {args.alpha}")
    print(f"  use_memory_map: {args.use_memory_map}")
    
    run_tractography(
        bedpostx_dir=args.bedpostx_dir,
        seed_mask_path=args.seed_mask,
        output_path=args.output,
        target_mask_path=args.target_mask,
        exclude_mask_path=args.exclude_mask,
        stop_mask_path=args.stop_mask,
        psoct_slides=args.psoct_slides,
        step_size=args.step_size,
        max_steps=args.max_steps,
        seeds_per_voxel=args.seeds_per_voxel,
        angle_threshold=args.angle_threshold,
        min_f_threshold=args.min_f_threshold,
        psoct_inplane_threshold=args.psoct_inplane_threshold,
        alpha=args.alpha,
        use_memory_map=args.use_memory_map,
        num_fibers=args.num_fibers,
        output_dir=args.output_dir
    )
