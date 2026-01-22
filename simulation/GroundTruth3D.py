"""
GroundTruth3D - 3D vector field generator for tractography simulation.

This module generates synthetic 3D vector fields for testing the tracking
algorithm without needing real BEDPOSTX/PSOCT data.
"""

import numpy as np


class GroundTruth3D:
    """
    Generates a 3D ground truth vector field for simulation.
    
    The field can be configured with different patterns (straight, curved,
    crossing, fan) to test different tracking scenarios.
    """
    
    def __init__(self, shape=(40, 40, 40), field_type='straight', 
                 primary_axis='z', noise_level=0.0):
        """
        Initialize and generate the 3D vector field.
        
        Args:
            shape: (X, Y, Z) dimensions of the volume
            field_type: Type of field to generate:
                - 'straight': Parallel fibers along primary_axis
                - 'curved': Curved bundle
                - 'crossing': Two crossing fiber bundles
                - 'fan': Fanning fibers
                - 'helix': Helical fibers
            primary_axis: Main fiber direction for 'straight' type ('x', 'y', 'z')
            noise_level: Amount of random noise to add (0-1)
        """
        self.shape = shape
        self.field_type = field_type
        self.primary_axis = primary_axis
        self.noise_level = noise_level
        
        # Generate the vector field
        self.vectors = self._generate_field()
        
        # Create brain mask (all voxels inside are valid)
        self.mask = np.ones(shape, dtype=bool)
        
        # Affine transform (identity - voxel = world coords)
        self.affine = np.eye(4)
        self.voxel_size = np.array([1.0, 1.0, 1.0])
    
    def _generate_field(self):
        """Generate the 3D vector field based on field_type."""
        X, Y, Z = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing='ij'
        )
        
        if self.field_type == 'straight':
            vectors = self._straight_field(X, Y, Z)
        elif self.field_type == 'curved':
            vectors = self._curved_field(X, Y, Z)
        elif self.field_type == 'crossing':
            vectors = self._crossing_field(X, Y, Z)
        elif self.field_type == 'fan':
            vectors = self._fan_field(X, Y, Z)
        elif self.field_type == 'helix':
            vectors = self._helix_field(X, Y, Z)
        else:
            raise ValueError(f"Unknown field_type: {self.field_type}")
        
        # Add noise
        if self.noise_level > 0:
            noise = np.random.randn(*vectors.shape) * self.noise_level
            vectors = vectors + noise
        
        # Normalize all vectors
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        return vectors
    
    def _straight_field(self, X, Y, Z):
        """Parallel fibers along primary axis."""
        vectors = np.zeros((*self.shape, 3))
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(self.primary_axis, 2)
        vectors[..., axis_idx] = 1.0
        
        return vectors
    
    def _curved_field(self, X, Y, Z):
        """Curved bundle - fibers curve around center."""
        vectors = np.zeros((*self.shape, 3))
        
        # Center of volume
        cx, cy, cz = [s // 2 for s in self.shape]
        
        # Vector field that curves around Y axis
        dx = X - cx
        dz = Z - cz
        r = np.sqrt(dx**2 + dz**2) + 1e-6
        
        # Tangent to circles around Y axis
        vectors[..., 0] = -dz / r  # X component
        vectors[..., 1] = 0.3      # Slight Y drift
        vectors[..., 2] = dx / r   # Z component
        
        return vectors
    
    def _crossing_field(self, X, Y, Z):
        """Two crossing fiber bundles."""
        vectors = np.zeros((*self.shape, 3))
        
        # Bundle 1: X direction in upper half
        # Bundle 2: Z direction in lower half
        mid_y = self.shape[1] // 2
        
        # Smooth transition zone
        transition_width = self.shape[1] // 8
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    # Weight for each bundle based on Y position
                    dist_to_mid = abs(j - mid_y)
                    
                    if dist_to_mid < transition_width:
                        # In crossing zone - blend both
                        w1 = 0.5 + 0.5 * (mid_y - j) / transition_width
                        w2 = 1 - w1
                        vectors[i, j, k] = w1 * np.array([1, 0, 0]) + w2 * np.array([0, 0, 1])
                    elif j < mid_y:
                        vectors[i, j, k] = [1, 0, 0]  # X direction
                    else:
                        vectors[i, j, k] = [0, 0, 1]  # Z direction
        
        return vectors
    
    def _fan_field(self, X, Y, Z):
        """Fanning fibers from one end."""
        vectors = np.zeros((*self.shape, 3))
        
        # Fan out from X=0 plane
        cx, cy, cz = 0, self.shape[1] // 2, self.shape[2] // 2
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    # Direction from origin
                    dx = i - cx + 1  # Avoid zero
                    dy = (j - cy) * 0.3  # Less spreading in Y
                    dz = (k - cz) * 0.3  # Less spreading in Z
                    
                    vectors[i, j, k] = [dx, dy, dz]
        
        return vectors
    
    def _helix_field(self, X, Y, Z):
        """Helical fibers around Z axis."""
        vectors = np.zeros((*self.shape, 3))
        
        cx, cy = self.shape[0] // 2, self.shape[1] // 2
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    dx = i - cx
                    dy = j - cy
                    r = np.sqrt(dx**2 + dy**2) + 1e-6
                    
                    # Tangent to circles + upward component
                    vectors[i, j, k, 0] = -dy / r * 0.5
                    vectors[i, j, k, 1] = dx / r * 0.5
                    vectors[i, j, k, 2] = 1.0  # Upward along Z
        
        return vectors
    
    def get_orientation(self, voxel):
        """
        Get the ground truth orientation at a voxel.
        
        Args:
            voxel: (i, j, k) voxel coordinates
            
        Returns:
            Normalized 3D orientation vector
        """
        i, j, k = [int(round(v)) for v in voxel]
        
        if not self._in_bounds(i, j, k):
            return None
        
        return self.vectors[i, j, k].copy()
    
    def is_valid_position(self, pos):
        """Check if position is within the volume."""
        i, j, k = [int(round(v)) for v in pos]
        return self._in_bounds(i, j, k) and self.mask[i, j, k]
    
    def _in_bounds(self, i, j, k):
        """Check if indices are within volume bounds."""
        return (0 <= i < self.shape[0] and
                0 <= j < self.shape[1] and
                0 <= k < self.shape[2])
    
    def plot_slice(self, axis='z', slice_idx=None, ax=None):
        """
        Plot a 2D slice of the vector field.
        
        Args:
            axis: Which axis to slice ('x', 'y', 'z')
            slice_idx: Index along that axis (default: middle)
            ax: Matplotlib axis to plot on
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[axis]
        
        if slice_idx is None:
            slice_idx = self.shape[axis_idx] // 2
        
        # Extract slice
        if axis == 'x':
            U = self.vectors[slice_idx, :, :, 1]  # Y component
            V = self.vectors[slice_idx, :, :, 2]  # Z component
            xlabel, ylabel = 'Y', 'Z'
        elif axis == 'y':
            U = self.vectors[:, slice_idx, :, 0]  # X component
            V = self.vectors[:, slice_idx, :, 2]  # Z component
            xlabel, ylabel = 'X', 'Z'
        else:  # z
            U = self.vectors[:, :, slice_idx, 0]  # X component
            V = self.vectors[:, :, slice_idx, 1]  # Y component
            xlabel, ylabel = 'X', 'Y'
        
        # Subsample for clearer visualization
        step = 2
        Y_grid, X_grid = np.mgrid[0:U.shape[0]:step, 0:U.shape[1]:step]
        
        ax.quiver(X_grid, Y_grid, 
                  U[::step, ::step], V[::step, ::step],
                  color='blue', alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{self.field_type} field, {axis}={slice_idx}')
        ax.set_aspect('equal')
        
        return ax
