"""
SimulatedData - Mock data classes for testing tractography.

These classes implement the same interface as BedpostxData and PSOCTData
but use simulated data from GroundTruth3D instead of real NIfTI files.
"""

import numpy as np
from .GroundTruth3D import GroundTruth3D


class SimulatedBedpostxData:
    """
    Simulates BEDPOSTX data from a GroundTruth3D vector field.
    
    Implements the same interface as tractography.BedpostxData so it can
    be used directly with PSOCTPriorityTracker.
    """
    
    def __init__(self, ground_truth, noise_level=0.1, num_fibers=1):
        """
        Initialize from a GroundTruth3D object.
        
        Args:
            ground_truth: GroundTruth3D object with the vector field
            noise_level: Amount of angular noise to add (simulates uncertainty)
            num_fibers: Number of fiber populations (for crossing regions)
        """
        self.gt = ground_truth
        self.noise_level = noise_level
        self.num_fibers = num_fibers
        
        # Copy relevant attributes from ground truth
        self.shape = ground_truth.shape
        self.affine = ground_truth.affine
        self.voxel_size = ground_truth.voxel_size
        self.mask = ground_truth.mask
        
        # Create dyads (mean orientations)
        self.dyads = [ground_truth.vectors.copy()]
        
        # Create f_samples (volume fractions) - all 1.0 for single fiber
        self.f_samples = [np.ones(self.shape)]
        
        # For probabilistic sampling, we add noise to the ground truth
        self.n_samples = 50  # Simulate BEDPOSTX's 50 samples
        
        print(f"  Simulated BEDPOSTX: shape={self.shape}, noise={noise_level}")
    
    def sample_orientation_probabilistic(self, voxel):
        """
        Sample orientation with added noise (simulates uncertainty).
        
        Args:
            voxel: (i, j, k) voxel coordinates
            
        Returns:
            Noisy orientation vector
        """
        base_orientation = self.get_orientation(voxel, fiber_idx=0)
        
        if base_orientation is None or np.linalg.norm(base_orientation) == 0:
            return None
        
        # Add random noise to simulate posterior distribution
        noise = np.random.randn(3) * self.noise_level
        noisy = base_orientation + noise
        
        # Normalize
        norm = np.linalg.norm(noisy)
        if norm > 0:
            return noisy / norm
        return base_orientation
    
    def get_orientation(self, voxel, fiber_idx=0):
        """Get the mean orientation at a voxel."""
        return self.gt.get_orientation(voxel)
    
    def get_fiber_fraction(self, voxel, fiber_idx=0):
        """Get volume fraction (always 1.0 for simulated data)."""
        i, j, k = [int(round(v)) for v in voxel]
        if not self._in_bounds(i, j, k):
            return 0.0
        return 1.0
    
    def is_valid_position(self, pos):
        """Check if position is within the volume."""
        return self.gt.is_valid_position(pos)
    
    def _in_bounds(self, i, j, k):
        """Check if indices are within volume bounds."""
        return self.gt._in_bounds(i, j, k)


class SimulatedPSOCTData:
    """
    Simulates PSOCT data with configurable coverage.
    
    Implements the same interface as tractography.PSOCTData.
    """
    
    def __init__(self, ground_truth, coverage_slices=None, coverage_fraction=0.5):
        """
        Initialize with partial coverage.
        
        Args:
            ground_truth: GroundTruth3D object
            coverage_slices: List of Y-slice indices where PSOCT data exists
                            If None, random slices are selected based on coverage_fraction
            coverage_fraction: Fraction of slices with PSOCT data (0-1)
        """
        self.gt = ground_truth
        self.shape = ground_truth.shape
        
        # Determine coverage
        if coverage_slices is None:
            n_slices = int(self.shape[1] * coverage_fraction)
            self.coverage_slices = set(
                np.random.choice(self.shape[1], n_slices, replace=False)
            )
        else:
            self.coverage_slices = set(coverage_slices)
        
        # Create coverage mask
        self.coverage_mask = np.zeros(self.shape, dtype=bool)
        for y in self.coverage_slices:
            self.coverage_mask[:, y, :] = True
        
        # For compatibility with real PSOCTData
        self.slides = list(self.coverage_slices)
        
        # Cache
        self._cache = {}
        
        coverage_pct = 100 * len(self.coverage_slices) / self.shape[1]
        print(f"  Simulated PSOCT: {len(self.coverage_slices)} slices ({coverage_pct:.1f}% coverage)")
    
    def get_orientation(self, voxel):
        """
        Get PSOCT orientation at a voxel.
        
        Returns None if no PSOCT data exists at this location.
        
        Args:
            voxel: (i, j, k) voxel coordinates
            
        Returns:
            Orientation vector or None
        """
        i, j, k = [int(round(v)) for v in voxel]
        
        # Check cache
        vox_key = (i, j, k)
        if vox_key in self._cache:
            return self._cache[vox_key]
        
        # Check if in coverage
        if not self.gt._in_bounds(i, j, k):
            self._cache[vox_key] = None
            return None
        
        if not self.coverage_mask[i, j, k]:
            self._cache[vox_key] = None
            return None
        
        # Return ground truth orientation (PSOCT is assumed perfect)
        orientation = self.gt.get_orientation(voxel)
        self._cache[vox_key] = orientation
        return orientation
    
    def has_data_at(self, voxel):
        """Check if PSOCT data exists at this voxel."""
        return self.get_orientation(voxel) is not None


def create_simulation_environment(shape=(40, 40, 40), field_type='straight',
                                   psoct_coverage=0.5, bedpostx_noise=0.1):
    """
    Convenience function to create a complete simulation environment.
    
    Args:
        shape: Volume dimensions
        field_type: Type of vector field
        psoct_coverage: Fraction of slices with PSOCT data
        bedpostx_noise: Noise level for BEDPOSTX sampling
        
    Returns:
        Tuple of (ground_truth, simulated_bedpostx, simulated_psoct)
    """
    print("\nCreating simulation environment...")
    
    gt = GroundTruth3D(shape=shape, field_type=field_type)
    print(f"  Ground truth: {field_type} field, shape={shape}")
    
    bedpostx = SimulatedBedpostxData(gt, noise_level=bedpostx_noise)
    psoct = SimulatedPSOCTData(gt, coverage_fraction=psoct_coverage)
    
    return gt, bedpostx, psoct
