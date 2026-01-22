"""
PSOCTData - Load and access PSOCT microscopy orientation data.

This module handles loading PSOCT slides and mapping orientations
from microscopy pixels to dMRI voxel space.
"""

import numpy as np
from fsl.data.image import Image

# Import cmc_hybrid utilities
from cmc_hybrid.coordinate_mapping import vox_to_pix, slide_to_volume, angle_to_vector
from cmc_hybrid.utils import fudge_psoct_orientation, make_dyads


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
