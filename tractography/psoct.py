"""
PSOCTData - Load and access PSOCT microscopy orientation data (coronal slices).

This module handles loading coronal PSOCT slides and mapping orientations
from microscopy pixels to dMRI voxel space. The underlying cmc_hybrid
functions assume coronal orientation throughout.
"""

import os
from glob import glob
import numpy as np
from fsl.data.image import Image

# Import cmc_hybrid utilities
from cmc_hybrid.coordinate_mapping import (
    vox_to_pix_per_slide, slide_to_volume, angle_to_vector,
)
from cmc_hybrid.utils import fudge_psoct_orientation


class PSOCTData:
    """
    Loads and provides access to PSOCT microscopy orientation data.
    Uses cmc_hybrid's coordinate mapping to find microscopy pixels within dMRI voxels.
    
    Note: Only coronal slices are supported (hardcoded in cmc_hybrid's
    angle_to_vector and slide_vox_intersect).
    """
    
    def __init__(self, psoct_files, volume_img):
        """
        Load PSOCT coronal slides.
        
        Args:
            psoct_files: List of paths to PSOCT NIfTI files, OR a single
                         directory path containing .nii.gz files.
            volume_img: fslpy Image object for the reference volume (e.g., BEDPOSTX mask)
        """
        self.slides = []
        self.volume_img = volume_img
        
        # If a directory path is given, discover slides automatically
        if isinstance(psoct_files, str) and os.path.isdir(psoct_files):
            psoct_files = sorted(glob(os.path.join(psoct_files, '*.nii.gz')))
            if len(psoct_files) == 0:
                # Try .nii as fallback
                psoct_files = sorted(glob(os.path.join(psoct_files, '*.nii')))
            print(f"Found {len(psoct_files)} PSOCT coronal slides in directory")
        
        print(f"Loading {len(psoct_files)} PSOCT coronal slides...")
        for path in psoct_files:
            try:
                slide = Image(path)
                self.slides.append(slide)
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
        
        print(f"  Loaded {len(self.slides)} slides successfully")

        # Cache for enclosing-voxel -> per-slide pixel grids.
        # The expensive lookup (vox_to_pix_per_slide) is keyed by integer
        # enclosing voxel; the nearest-pixel selection itself depends on the
        # continuous sub-voxel position and is cheap, so it isn't cached.
        self._pixgrid_cache = {}

        # Precompute per-slide slide-to-volume affines and slide normals.
        # Both depend only on the slide and the reference volume, so they are
        # constant for the run and can be reused across every PSOCT step.
        self._slide_xforms = [slide_to_volume(s, self.volume_img) for s in self.slides]
        self._slide_normals = []
        for x in self._slide_xforms:
            n = x @ np.array([0.0, 1.0, 0.0])
            n_norm = np.linalg.norm(n)
            self._slide_normals.append(n / n_norm if n_norm > 0 else np.array([0.0, 1.0, 0.0]))

    def get_orientation(self, pos, return_normal=False):
        """
        Get the PSOCT orientation at a continuous sub-voxel position.

        Finds the single nearest PSOCT pixel to ``pos`` across all slides
        that intersect the enclosing voxel (rounded). The pixel angle is
        corrected with ``fudge_psoct_orientation`` and lifted into volume
        space via the slide-to-volume affine (assumes coronal slices, as in
        ``angle_to_vector``: v = [cos(theta), 0, -sin(theta)]).

        Args:
            pos: (3,) continuous voxel-space position.
            return_normal: If True, also return the slide normal (the
                out-of-plane direction of the matched slide) in volume
                space, so callers can project 3D directions into the slide
                plane for in-plane comparison.

        Returns:
            If ``return_normal`` is False (default): the normalised 3D
            orientation vector, or None if no PSOCT data is available.
            If True: a tuple ``(vector, slide_normal)`` or None.
        """
        pos = np.asarray(pos, dtype=float)

        # Enclosing voxel — same rounding as plot_single_voxel.py
        enclosing_vox = tuple(int(round(v)) for v in pos)

        # Per-slide pixel grids in this enclosing voxel (cached)
        if enclosing_vox in self._pixgrid_cache:
            per_slide = self._pixgrid_cache[enclosing_vox]
        else:
            try:
                per_slide = vox_to_pix_per_slide(
                    list(enclosing_vox), self.slides, self.volume_img
                )
            except Exception:
                per_slide = {}
            self._pixgrid_cache[enclosing_vox] = per_slide

        if not per_slide:
            return None

        # Find the single nearest pixel across all intersecting slides,
        # measuring squared distance from the continuous query position
        # (argmin is unaffected by the monotonic sqrt — saves it per pixel).
        best_sq = np.inf
        best_theta = None
        best_slide_idx = None
        for slide_idx, (_pixgrid, voxgrid, theta) in per_slide.items():
            diff = voxgrid - pos
            dists_sq = np.einsum('ij,ij->i', diff, diff)
            idx = int(np.argmin(dists_sq))
            if dists_sq[idx] < best_sq:
                best_sq = float(dists_sq[idx])
                best_theta = float(theta[idx])
                best_slide_idx = slide_idx

        if best_theta is None:
            return None

        # Correct the raw PSOCT angle, then lift the single 2D angle into a
        # 3D vector in volume space using the matched slide's affine.
        theta_corr = fudge_psoct_orientation(np.array([best_theta]))[0]
        xform = self._slide_xforms[best_slide_idx]
        vec = angle_to_vector(np.array([theta_corr]), xform)[0]

        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        vec = vec / norm

        if not return_normal:
            return vec

        # Slide normal in volume space: precomputed at init time.
        return vec, self._slide_normals[best_slide_idx]

    def has_data_at(self, pos):
        """Check if PSOCT data is available at the given position."""
        return self.get_orientation(pos) is not None
