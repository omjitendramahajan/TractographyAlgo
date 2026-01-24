"""
BedpostxData - Load and access BEDPOSTX diffusion MRI data.

This module handles loading BEDPOSTX output files and provides
methods for accessing fiber orientations and probabilistic sampling.

Uses fslpy (fsl.data.image.Image) for NIfTI loading.
"""

import numpy as np
from fsl.data.image import Image


class BedpostxData:
    """
    Loads and provides access to BEDPOSTX output files.
    Supports probabilistic sampling from posterior samples.
    """
    
    def __init__(self, bedpostx_dir, num_fibers=3, load_samples=True, load_dyads=False, 
                 use_memory_map=True):
        """
        Load BEDPOSTX data.
        
        Args:
            bedpostx_dir: Path to .bedpostX directory
            num_fibers: Number of fiber populations to load (1, 2, or 3)
            load_samples: If True, load posterior samples for probabilistic tracking
            load_dyads: If True, load mean fiber orientations (dyads) for deterministic tracking.
                        Set to False if only using probabilistic tracking
            use_memory_map: If True, use lazy loading (fslpy default behavior).
                           Set to False to preload full arrays into RAM.
        """
        self.bedpostx_dir = bedpostx_dir
        self.num_fibers = num_fibers
        self.use_memory_map = use_memory_map
        
        # Load brain mask (reference for shape and affine)
        mask_path = f"{bedpostx_dir}/nodif_brain_mask.nii.gz"
        self.mask_img = Image(mask_path)
        self.mask = np.asarray(self.mask_img[:]) > 0  # Bool mask: True if voxel is in brain
        self.affine = self.mask_img.voxToWorldMat  # 4x4 voxel-to-world transform matrix
        self.shape = self.mask_img.shape[:3]  # (X, Y, Z) - sagittal, coronal, axial
        
        # Compute voxel sizes from affine (in mm)
        self.voxel_size = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
        
        # Load mean fiber orientations and fractions (for deterministic tracking)
        self.dyads = []      # List of 4D arrays (X, Y, Z, 3)
        self.f_samples = []  # List of 3D arrays (mean volume fractions)
        
        if load_dyads:
            print("  Loading mean fiber orientations (dyads)...")
            for i in range(1, num_fibers + 1):
                dyad_path = f"{bedpostx_dir}/dyads{i}.nii.gz"
                f_path = f"{bedpostx_dir}/mean_f{i}samples.nii.gz"
                
                try:
                    dyad_img = Image(dyad_path)
                    self.dyads.append(np.asarray(dyad_img[:]))
                    
                    f_img = Image(f_path)
                    self.f_samples.append(np.asarray(f_img[:]))
                    print(f"    Loaded fiber population {i}")
                except FileNotFoundError:
                    print(f"    Warning: Could not find fiber population {i}")
                    break
            
            self.num_fibers = len(self.dyads)
        else:
            print("  Skipping dyad loading (load_dyads=False)")

        
        # Posterior samples for probabilistic tracking
        # Full load mode: lists of 4D numpy arrays
        # Lazy mode: lists of fslpy Image objects (data loaded on access)
        self.th_samples = []  # Theta angles
        self.ph_samples = []  # Phi angles
        self.f_all_samples = []  # Volume fractions
        self.n_samples = 0  # Number of MCMC samples per voxel
        
        if load_samples:
            self._load_posterior_samples()

        # Store volume image for coordinate transforms
        self.volume_img = self.mask_img
    
    def _load_posterior_samples(self):
        """Load posterior samples for probabilistic tracking.
        
        If use_memory_map=True, stores fslpy Image objects (data loaded lazily).
        Otherwise, preloads full arrays into RAM.
        """
        mode_str = "lazy/memory-mapped" if self.use_memory_map else "full load"
        print(f"  Loading posterior samples ({mode_str})...")
        
        for i in range(1, self.num_fibers + 1):
            th_path = f"{self.bedpostx_dir}/merged_th{i}samples.nii"
            ph_path = f"{self.bedpostx_dir}/merged_ph{i}samples.nii"
            f_path = f"{self.bedpostx_dir}/merged_f{i}samples.nii"
            
            try:
                th_img = Image(th_path)
                ph_img = Image(ph_path)
                f_img = Image(f_path)

                if self.use_memory_map:
                    # Store Image objects - data loaded lazily on access
                    self.th_samples.append(th_img)
                    self.ph_samples.append(ph_img)
                    self.f_all_samples.append(f_img)
                else:
                    # Preload full arrays into RAM
                    self.th_samples.append(np.asarray(th_img[:]))
                    self.ph_samples.append(np.asarray(ph_img[:]))
                    self.f_all_samples.append(np.asarray(f_img[:]))
                
                # Get number of samples (4th dimension)
                self.n_samples = th_img.shape[3] if len(th_img.shape) > 3 else 1
                print(f"    Fiber {i}: {self.n_samples} samples ready")
                
            except FileNotFoundError:
                print(f"    Warning: Could not find merged samples for fiber {i}")
                break
        
        if len(self.th_samples) == 0:
            print("    No posterior samples found - using deterministic mode only")
    
    def sample_orientation_probabilistic(self, voxel):
        """
        Probabilistically sample an orientation from BEDPOSTX posterior.
        
        Samples from all fiber populations weighted by their volume fractions.
        Works with both full-load and lazy/memory-mapped modes.
        
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
            if self.use_memory_map:
                # Lazy mode: access via fslpy Image indexing
                th_vals = np.asarray(self.th_samples[fiber_idx][i, j, k, :])
                ph_vals = np.asarray(self.ph_samples[fiber_idx][i, j, k, :])
                f_vals = np.asarray(self.f_all_samples[fiber_idx][i, j, k, :])
            else:
                # Full load: direct array access
                th_vals = self.th_samples[fiber_idx][i, j, k, :]  # (N_samples,)
                ph_vals = self.ph_samples[fiber_idx][i, j, k, :]  # (N_samples,)
                f_vals = self.f_all_samples[fiber_idx][i, j, k, :]  # (N_samples,)
            
            # Vectorized processing
            # 1. Filter by volume fraction
            mask = f_vals >= 0.01
            if not np.any(mask):
                continue
                
            th_masked = th_vals[mask]
            ph_masked = ph_vals[mask]
            f_masked = f_vals[mask]
            
            # 2. Spherical to Cartesian conversion (vectorized)
            sin_th = np.sin(th_masked)
            x = sin_th * np.cos(ph_masked)
            y = sin_th * np.sin(ph_masked)
            z = np.cos(th_masked)
            
            # Stack into (N, 3) matrix
            vectors = np.stack([x, y, z], axis=1)
            
            # 3. Normalize
            norms = np.linalg.norm(vectors, axis=1)
            valid_norm = norms > 0
            
            if np.any(valid_norm):
                all_orientations.append(vectors[valid_norm] / norms[valid_norm, np.newaxis])
                all_weights.append(f_masked[valid_norm])

        if not all_orientations:
            return None
            
        # Concatenate all populations
        all_orientations = np.vstack(all_orientations)
        all_weights = np.concatenate(all_weights)
        
        # Normalize weights
        if all_weights.sum() == 0:
            return None
        all_weights = all_weights / all_weights.sum()
        
        # Sample one orientation weighted by volume fraction
        chosen_idx = np.random.choice(len(all_orientations), p=all_weights)
        
        return all_orientations[chosen_idx]
    
    def get_orientation(self, voxel, fiber_idx=0):
        """Get the mean orientation vector at a voxel for a specific fiber population."""
        i, j, k = [int(round(v)) for v in voxel]
        
        if not self._in_bounds(i, j, k):
            return np.zeros(3)
        
        if fiber_idx >= len(self.dyads):
            return np.zeros(3)
        
        v = self.dyads[fiber_idx][i, j, k, :]
        norm = np.linalg.norm(v)
        
        if norm > 0:
            return v / norm
        return np.zeros(3)
    
    def get_fiber_fraction(self, voxel, fiber_idx=0):
        """Get the fiber volume fraction at a voxel.
        
        Uses mean f_samples if available, otherwise computes mean from posterior samples.
        """
        i, j, k = [int(round(v)) for v in voxel]
        
        if not self._in_bounds(i, j, k):
            return 0.0
        
        # Use mean f_samples if loaded (from dyads loading)
        if len(self.f_samples) > fiber_idx:
            return float(self.f_samples[fiber_idx][i, j, k])
        
        # Otherwise, compute mean from posterior samples
        if len(self.f_all_samples) > fiber_idx:
            if self.use_memory_map:
                f_vals = np.asarray(self.f_all_samples[fiber_idx][i, j, k, :])
            else:
                f_vals = self.f_all_samples[fiber_idx][i, j, k, :]
            return float(np.mean(f_vals))
        
        return 0.0
    
    def is_valid_position(self, pos):
        """Check if position is within the brain mask."""
        i, j, k = [int(round(v)) for v in pos]
        return self._in_bounds(i, j, k) and self.mask[i, j, k]
    
    def _in_bounds(self, i, j, k):
        """Check if voxel indices are within image bounds."""
        return (0 <= i < self.shape[0] and
                0 <= j < self.shape[1] and
                0 <= k < self.shape[2])
