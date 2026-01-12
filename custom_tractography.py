"""
PSOCT-Priority Tractography Algorithm
======================================

This tractography algorithm uses PSOCT (microscopy) orientations when available,
falling back to BEDPOSTX (dMRI) orientations only where microscopy data is absent.

The PSOCT data is NOT diluted or blended with diffusion data - it takes full priority.

Usage:
    python custom_tractography.py \
        --bedpostx /path/to/data.bedpostX \
        --psoct /path/to/psoct/Slice*_header.nii.gz \
        --seeds seeds.nii.gz \
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
import numpy as np
import nibabel as nib
from scipy import ndimage

# Import cmc_hybrid utilities (assumes cmc_hybrid is installed: pip install -e ./cmc_hybrid)
from cmc_hybrid.coordinate_mapping import vox_to_pix, slide_to_volume, angle_to_vector
from cmc_hybrid.utils import fudge_psoct_orientation, make_dyads
from fsl.data.image import Image


class BedpostxData:
    """
    Loads and provides access to BEDPOSTX output files.
    Supports probabilistic sampling from posterior samples.
    """
    
    def __init__(self, bedpostx_dir, num_fibers=2, load_samples=True):
        """
        Load BEDPOSTX data.
        
        Args:
            bedpostx_dir: Path to .bedpostX directory
            num_fibers: Number of fiber populations to load (1, 2, or 3)
            load_samples: If True, load full posterior samples for probabilistic tracking
        """
        self.bedpostx_dir = bedpostx_dir
        self.num_fibers = num_fibers
        
        # Load brain mask (reference for shape and affine)
        mask_path = f"{bedpostx_dir}/nodif_brain_mask.nii.gz"
        self.mask_img = nib.load(mask_path)
        self.mask = self.mask_img.get_fdata() > 0
        self.affine = self.mask_img.affine
        self.shape = self.mask.shape
        
        # For fslpy compatibility
        self.volume_img = Image(mask_path)
        
        # Compute voxel sizes from affine
        self.voxel_size = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
        
        # Load mean fiber orientations and fractions (for deterministic tracking)
        self.dyads = []      # List of 4D arrays (X, Y, Z, 3)
        self.f_samples = []  # List of 3D arrays (mean volume fractions)
        
        for i in range(1, num_fibers + 1):
            dyad_path = f"{bedpostx_dir}/dyads{i}.nii.gz"
            f_path = f"{bedpostx_dir}/mean_f{i}samples.nii.gz"
            
            try:
                dyad_img = nib.load(dyad_path)
                self.dyads.append(dyad_img.get_fdata())
                
                f_img = nib.load(f_path)
                self.f_samples.append(f_img.get_fdata())
                print(f"  Loaded fiber population {i}")
            except FileNotFoundError:
                print(f"  Warning: Could not find fiber population {i}")
                break
        
        self.num_fibers = len(self.dyads)
        
        # Load posterior samples for probabilistic tracking
        self.th_samples = []  # Theta angles: list of 4D arrays (X, Y, Z, N_samples)
        self.ph_samples = []  # Phi angles: list of 4D arrays (X, Y, Z, N_samples)
        self.f_all_samples = []  # Volume fractions: list of 4D arrays (X, Y, Z, N_samples)
        self.n_samples = 0  # Number of samples per voxel
        
        if load_samples:
            self._load_posterior_samples()
    
    def _load_posterior_samples(self):
        """Load merged posterior samples for probabilistic tracking."""
        print("  Loading posterior samples for probabilistic tracking...")
        
        for i in range(1, self.num_fibers + 1):
            th_path = f"{self.bedpostx_dir}/merged_th{i}samples.nii.gz"
            ph_path = f"{self.bedpostx_dir}/merged_ph{i}samples.nii.gz"
            f_path = f"{self.bedpostx_dir}/merged_f{i}samples.nii.gz"
            
            try:
                th_img = nib.load(th_path)
                ph_img = nib.load(ph_path)
                f_img = nib.load(f_path)
                
                self.th_samples.append(th_img.get_fdata())
                self.ph_samples.append(ph_img.get_fdata())
                self.f_all_samples.append(f_img.get_fdata())
                
                # Get number of samples (4th dimension)
                self.n_samples = th_img.shape[3] if len(th_img.shape) > 3 else 1
                print(f"    Fiber {i}: {self.n_samples} samples loaded")
                
            except FileNotFoundError:
                print(f"    Warning: Could not find merged samples for fiber {i}")
                # Fall back to mean values if samples not available
                break
        
        if len(self.th_samples) == 0:
            print("    No posterior samples found - using deterministic mode only")
    
    def sample_orientation_probabilistic(self, voxel):
        """
        Probabilistically sample an orientation from BEDPOSTX posterior.
        
        Samples from all fiber populations weighted by their volume fractions.
        
        Args:
            voxel: (i, j, k) voxel coordinates
            
        Returns:
            Normalized 3D orientation vector, or None if invalid
        """
        i, j, k = [int(round(v)) for v in voxel]
        
        if not self._in_bounds(i, j, k):
            return None
        
        if len(self.th_samples) == 0:
            # No samples available, fall back to deterministic
            return self.get_orientation(voxel, fiber_idx=0)
        
        # Collect all samples from all fiber populations with their weights
        all_orientations = []
        all_weights = []
        
        for fiber_idx in range(len(self.th_samples)):
            # Get all samples for this fiber at this voxel
            th_vals = self.th_samples[fiber_idx][i, j, k, :]  # (N_samples,)
            ph_vals = self.ph_samples[fiber_idx][i, j, k, :]  # (N_samples,)
            f_vals = self.f_all_samples[fiber_idx][i, j, k, :]  # (N_samples,)
            
            for sample_idx in range(self.n_samples):
                # Convert spherical to Cartesian
                theta = th_vals[sample_idx]
                phi = ph_vals[sample_idx]
                f = f_vals[sample_idx]
                
                # Skip if volume fraction too low
                if f < 0.01:
                    continue
                
                # Spherical to Cartesian conversion
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                
                orientation = np.array([x, y, z])
                norm = np.linalg.norm(orientation)
                
                if norm > 0:
                    all_orientations.append(orientation / norm)
                    all_weights.append(f)  # Weight by anisotropy (volume fraction)
        
        if len(all_orientations) == 0:
            return None
        
        # Normalize weights
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
        # Sample one orientation weighted by volume fraction
        chosen_idx = np.random.choice(len(all_orientations), p=all_weights)
        
        return all_orientations[chosen_idx]
    
    def get_orientation(self, voxel, fiber_idx=0):
        """Get the mean orientation vector at a voxel for a specific fiber population."""
        i, j, k = [int(round(v)) for v in voxel]
        
        if not self._in_bounds(i, j, k):
            return np.zeros(3)
        
        if fiber_idx >= self.num_fibers:
            return np.zeros(3)
        
        v = self.dyads[fiber_idx][i, j, k, :]
        norm = np.linalg.norm(v)
        
        if norm > 0:
            return v / norm
        return np.zeros(3)
    
    def get_fiber_fraction(self, voxel, fiber_idx=0):
        """Get the mean fiber volume fraction at a voxel."""
        i, j, k = [int(round(v)) for v in voxel]
        
        if not self._in_bounds(i, j, k):
            return 0.0
        
        if fiber_idx >= self.num_fibers:
            return 0.0
        
        return self.f_samples[fiber_idx][i, j, k]
    
    def is_valid_position(self, pos):
        """Check if position is within the brain mask."""
        i, j, k = [int(round(v)) for v in pos]
        return self._in_bounds(i, j, k) and self.mask[i, j, k]
    
    def _in_bounds(self, i, j, k):
        """Check if voxel indices are within image bounds."""
        return (0 <= i < self.shape[0] and
                0 <= j < self.shape[1] and
                0 <= k < self.shape[2])


class PSOCTData:
    """
    Loads and provides access to PSOCT microscopy orientation data.
    Uses cmc_hybrid's coordinate mapping to find microscopy pixels within dMRI voxels.
    """
    
    def __init__(self, psoct_files, volume_img, slide_direction='coronal'):
        """
        Load PSOCT slides.
        
        Args:
            psoct_files: List of paths to PSOCT NIfTI files (orientation angle images)
            volume_img: fslpy Image object for the reference volume (e.g., BEDPOSTX mask)
            slide_direction: Orientation of the slides ('coronal', 'sagittal', 'axial')
        """
        self.slides = []
        self.volume_img = volume_img
        self.slide_direction = slide_direction
        
        print(f"Loading {len(psoct_files)} PSOCT slides...")
        for path in psoct_files:
            try:
                slide = Image(path)
                self.slides.append(slide)
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
        
        print(f"  Loaded {len(self.slides)} slides successfully")
        
        # Cache for voxel -> PSOCT orientation mapping
        self._cache = {}
    
    def get_orientation(self, voxel):
        """
        Get PSOCT orientation at a dMRI voxel.
        
        Returns the mean orientation vector from all microscopy pixels within the voxel,
        or None if no microscopy data is available at this location.
        
        Args:
            voxel: (i, j, k) voxel coordinates in dMRI space
            
        Returns:
            Normalized 3D orientation vector, or None if no PSOCT data available
        """
        # Round to integer voxel for caching
        vox_key = tuple(int(round(v)) for v in voxel)
        
        if vox_key in self._cache:
            return self._cache[vox_key]
        
        # Get microscopy pixels within this voxel
        try:
            pixgrid, voxgrid, theta_values = vox_to_pix(
                list(vox_key), self.slides, self.volume_img
            )
        except Exception:
            self._cache[vox_key] = None
            return None
        
        if len(theta_values) == 0:
            self._cache[vox_key] = None
            return None
        
        # Convert 2D angles to 3D vectors
        theta_values = np.array(theta_values)
        
        # Apply PSOCT orientation correction (empirically determined)
        theta_corrected = fudge_psoct_orientation(theta_values)
        
        # Get transformation from slide to volume coordinates
        # Use the first slide that contributed data
        xform = slide_to_volume(self.slides[0], self.volume_img)
        
        # Convert angles to 3D vectors
        vectors = angle_to_vector(theta_corrected, xform)
        
        # Compute dyadic mean (handles sign ambiguity of orientations)
        mean_orientation = make_dyads(vectors)
        
        # Normalize
        norm = np.linalg.norm(mean_orientation)
        if norm > 0:
            mean_orientation = mean_orientation / norm
        else:
            mean_orientation = None
        
        self._cache[vox_key] = mean_orientation
        return mean_orientation
    
    def has_data_at(self, voxel):
        """Check if PSOCT data is available at this voxel."""
        return self.get_orientation(voxel) is not None


class PSOCTPriorityTracker:
    """
    Tractography that uses PSOCT orientations when available,
    falling back to BEDPOSTX only where microscopy data is absent.
    
    PSOCT data takes FULL PRIORITY - no blending or dilution with diffusion data.
    """
    
    def __init__(self, bedpostx_data, psoct_data=None, 
                 step_size=0.5, max_steps=2000,
                 min_f_threshold=0.05, angle_threshold=60):
        """
        Initialize the tracker.
        
        Args:
            bedpostx_data: BedpostxData object
            psoct_data: PSOCTData object (optional)
            step_size: Step size in mm
            max_steps: Maximum steps per direction
            min_f_threshold: Minimum fiber fraction to continue (BEDPOSTX only)
            angle_threshold: Maximum angle (degrees) between consecutive steps
        """
        self.bedpostx = bedpostx_data
        self.psoct = psoct_data
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_f_threshold = min_f_threshold
        self.max_angle = np.deg2rad(angle_threshold)
        
        # Convert step size to voxel units
        self.step_vox = step_size / self.bedpostx.voxel_size
        
        # Track statistics
        self.stats = {'psoct_steps': 0, 'bedpostx_steps': 0}
    
    def track(self, seed):
        """
        Track a streamline bidirectionally from a seed point.
        
        Args:
            seed: (i, j, k) seed position in voxel coordinates
            
        Returns:
            Tuple of (streamline, source_labels) where:
            - streamline: Nx3 numpy array of voxel coordinates
            - source_labels: N-length array indicating data source per point
              ('psoct' or 'bedpostx')
        """
        # Track forward
        forward, fwd_labels = self._track_one_direction(seed, forward=True)
        
        # Track backward
        backward, bwd_labels = self._track_one_direction(seed, forward=False)
        
        # Merge results
        if forward is None and backward is None:
            return None, None
        elif forward is None:
            return backward, bwd_labels
        elif backward is None:
            return forward, fwd_labels
        else:
            # Flip backward and concatenate (avoid duplicating seed)
            streamline = np.vstack([np.flipud(backward), forward[1:]])
            labels = np.concatenate([bwd_labels[::-1], fwd_labels[1:]])
            return streamline, labels
    
    def _track_one_direction(self, seed, forward=True):
        """Track in one direction from the seed."""
        pos = np.array(seed, dtype=float)
        
        # Get initial direction
        direction, source = self._get_direction(pos)
        if direction is None:
            return None, None
        
        if not forward:
            direction = -direction
        
        # Initialize streamline with seed
        streamline = [pos.copy()]
        labels = [source]
        
        for step_idx in range(self.max_steps):
            # Get next direction using PSOCT-priority logic
            new_direction, source = self.step(pos, direction)
            
            if new_direction is None:
                break
            
            # Check angle constraint
            angle = np.arccos(np.clip(np.abs(np.dot(direction, new_direction)), -1, 1))
            if angle > self.max_angle:
                break
            
            # Ensure consistent direction (no 180° flips)
            if np.dot(new_direction, direction) < 0:
                new_direction = -new_direction
            
            # Take a step
            new_pos = pos + self.step_vox * new_direction
            
            # Check if still valid
            if not self.bedpostx.is_valid_position(new_pos):
                break
            
            # Check fiber fraction threshold (only for BEDPOSTX regions)
            if source == 'bedpostx':
                max_f = max([self.bedpostx.get_fiber_fraction(new_pos, i) 
                            for i in range(self.bedpostx.num_fibers)], default=0)
                if max_f < self.min_f_threshold:
                    break
            
            # Update for next iteration
            streamline.append(new_pos.copy())
            labels.append(source)
            pos = new_pos
            direction = new_direction
        
        if len(streamline) < 2:
            return None, None
        
        return np.array(streamline), np.array(labels)
    
    def _get_direction(self, pos):
        """
        Get tracking direction at position using PSOCT-priority logic.
        
        Uses probabilistic sampling from BEDPOSTX posterior samples
        weighted by volume fraction (anisotropy).
        
        Returns:
            Tuple of (direction_vector, source_label)
            source_label is 'psoct' or 'bedpostx'
        """
        # PRIORITY 1: Try PSOCT first
        if self.psoct is not None:
            psoct_orientation = self.psoct.get_orientation(pos)
            if psoct_orientation is not None and np.linalg.norm(psoct_orientation) > 0:
                self.stats['psoct_steps'] += 1
                return psoct_orientation, 'psoct'
        
        # PRIORITY 2: Fall back to BEDPOSTX with probabilistic sampling
        bedpostx_orientation = self.bedpostx.sample_orientation_probabilistic(pos)
        if bedpostx_orientation is not None and np.linalg.norm(bedpostx_orientation) > 0:
            self.stats['bedpostx_steps'] += 1
            return bedpostx_orientation, 'bedpostx'
        
        return None, None
    
    def step(self, pos, prev_direction):
        """
        Get the next direction at position.
        Uses PSOCT orientation if available, otherwise BEDPOSTX.
        
        Args:
            pos: Current position (voxel coordinates)
            prev_direction: Previous step direction (for consistency checks)
            
        Returns:
            Tuple of (new_direction, source_label)
        """
        return self._get_direction(pos)


def run_tractography(bedpostx_dir, psoct_pattern, seed_mask_path, output_path,
                     step_size=0.5, max_steps=2000, seeds_per_voxel=1,
                     deterministic=False):
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
    bedpostx = BedpostxData(bedpostx_dir, load_samples=load_samples)
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
        print("\n[2/4] PSOCT data: Not provided or cmc_hybrid unavailable")
    
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
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  Processing seed {idx + 1}/{len(seed_voxels)}")
        
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
    if total_steps > 0:
        psoct_pct = 100 * tracker.stats['psoct_steps'] / total_steps
        print(f"  PSOCT usage: {psoct_pct:.1f}%")
    
    if len(streamlines) == 0:
        print("\nNo streamlines generated! Check your seeds and parameters.")
        return
    
    # Create organized output directory structure
    import json
    import os
    from datetime import datetime
    
    # Get script directory and create TRK_outputs folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "TRK_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Find next available output number
    existing = [d for d in os.listdir(outputs_dir) if d.startswith("Output") and os.path.isdir(os.path.join(outputs_dir, d))]
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
            'psoct_usage_pct': psoct_pct if total_steps > 0 else 0
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
    from nibabel.streamlines import Tractogram, save
    
    # Convert voxel coordinates to world coordinates
    world_streamlines = []
    for sl in streamlines:
        # Apply affine: world = affine @ [voxel, 1]
        ones = np.ones((len(sl), 1))
        vox_homo = np.hstack([sl, ones])
        world = (affine @ vox_homo.T).T[:, :3]
        world_streamlines.append(world)
    
    # Create and save tractogram
    tractogram = Tractogram(
        streamlines=world_streamlines,
        affine_to_rasmm=np.eye(4)  # Already in world coordinates
    )
    save(tractogram, output_path)


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
    
    parser.add_argument("--bedpostx", "-b", required=True,
                        help="Path to .bedpostX directory")
    parser.add_argument("--psoct", "-p", default=None,
                        help="Glob pattern for PSOCT slides (e.g., 'path/Slice*_header.nii.gz')")
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
                        help="Use deterministic tracking (lower memory, uses mean orientations)")
    
    args = parser.parse_args()
    
    run_tractography(
        bedpostx_dir=args.bedpostx,
        psoct_pattern=args.psoct,
        seed_mask_path=args.seeds,
        output_path=args.output,
        step_size=args.step_size,
        max_steps=args.max_steps,
        seeds_per_voxel=args.seeds_per_voxel,
        deterministic=args.deterministic
    )


if __name__ == "__main__":
    main()
