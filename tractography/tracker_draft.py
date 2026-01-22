"""
ProbabilisticTracker - Probabilistic tractography using BEDPOSTX data only.

This is a simplified tracker that uses only BEDPOSTX diffusion MRI data
for probabilistic tractography. No PSOCT/microscopy data integration.

This represents the initial development stage before adding multimodal support.
"""

import numpy as np


class ProbabilisticTracker:
    """
    Probabilistic tractography using BEDPOSTX posterior samples.
    
    Implements bidirectional streamline tracking with:
    - Probabilistic sampling from BEDPOSTX posterior distributions
    - Angle constraints between consecutive steps
    - Volume fraction thresholding for termination
    """
    
    def __init__(self, bedpostx_data, step_size=0.5, max_steps=2000,
                 min_f_threshold=0.05, angle_threshold=60):
        """
        Initialize the tracker.
        
        Args:
            bedpostx_data: BedpostxData object with loaded posterior samples
            step_size: Step size in mm (default: 0.5)
            max_steps: Maximum steps per direction (default: 2000)
            min_f_threshold: Minimum fiber fraction to continue tracking (default: 0.05)
            angle_threshold: Maximum angle in degrees between consecutive steps (default: 60)
        """
        self.bedpostx = bedpostx_data
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_f_threshold = min_f_threshold
        self.max_angle = np.deg2rad(angle_threshold)
        
        # Convert step size from mm to voxel units
        # step_vox[i] = how many voxels to move per step along axis i
        self.step_vox = step_size / self.bedpostx.voxel_size
        
        # Statistics for tracking
        self.stats = {
            'total_streamlines': 0,
            'total_steps': 0,
            'terminated_angle': 0,
            'terminated_boundary': 0,
            'terminated_low_f': 0,
            'terminated_max_steps': 0
        }
    
    def track(self, seed):
        """
        Track a streamline bidirectionally from a seed point.
        
        The algorithm:
        1. Get initial orientation at seed
        2. Track forward until termination
        3. Track backward until termination
        4. Concatenate into single streamline
        
        Args:
            seed: (i, j, k) seed position in voxel coordinates
            
        Returns:
            numpy.ndarray: Nx3 array of voxel coordinates representing the streamline,
                          or None if tracking failed
        """
        # Track in both directions
        forward = self._track_one_direction(seed, forward=True)
        backward = self._track_one_direction(seed, forward=False)
        
        # Handle edge cases
        if forward is None and backward is None:
            return None
        elif forward is None:
            streamline = backward
        elif backward is None:
            streamline = forward
        else:
            # Concatenate: flip backward, then append forward (skip duplicate seed)
            streamline = np.vstack([np.flipud(backward), forward[1:]])
        
        self.stats['total_streamlines'] += 1
        return streamline
    
    def _track_one_direction(self, seed, forward=True):
        """
        Track in one direction from the seed.
        
        Args:
            seed: Starting position (voxel coordinates)
            forward: If True, track in initial direction; if False, track opposite
            
        Returns:
            numpy.ndarray: Mx3 array of positions, or None if failed
        """
        pos = np.array(seed, dtype=float)
        
        # Get initial direction from probabilistic sampling
        direction = self._get_direction(pos)
        if direction is None:
            return None
        
        # Reverse direction for backward tracking
        if not forward:
            direction = -direction
        
        # Initialize streamline with seed position
        streamline = [pos.copy()]
        
        for step_idx in range(self.max_steps):
            # Sample new direction at current position
            new_direction = self._get_direction(pos)
            
            if new_direction is None:
                break
            
            # Check angle constraint
            cos_angle = np.clip(np.abs(np.dot(direction, new_direction)), -1, 1)
            angle = np.arccos(cos_angle)
            
            if angle > self.max_angle:
                self.stats['terminated_angle'] += 1
                break
            
            # Ensure consistent direction (prevent 180° flips)
            if np.dot(new_direction, direction) < 0:
                new_direction = -new_direction
            
            # Take a step in voxel coordinates
            new_pos = pos + self.step_vox * new_direction
            
            # Check if still within brain mask
            if not self.bedpostx.is_valid_position(new_pos):
                self.stats['terminated_boundary'] += 1
                break
            
            # Check volume fraction threshold
            max_f = self._get_max_fiber_fraction(new_pos)
            if max_f < self.min_f_threshold:
                self.stats['terminated_low_f'] += 1
                break
            
            # Accept the step
            streamline.append(new_pos.copy())
            self.stats['total_steps'] += 1
            pos = new_pos
            direction = new_direction
        else:
            # Reached max_steps without other termination
            self.stats['terminated_max_steps'] += 1
        
        # Need at least 2 points for a valid streamline segment
        if len(streamline) < 2:
            return None
        
        return np.array(streamline)
    
    def _get_direction(self, pos):
        """
        Get tracking direction at position using probabilistic sampling.
        
        Samples from BEDPOSTX posterior distribution weighted by volume fraction.
        
        Args:
            pos: Current position (voxel coordinates)
            
        Returns:
            numpy.ndarray: Normalized 3D direction vector, or None if invalid
        """
        orientation = self.bedpostx.sample_orientation_probabilistic(pos)
        
        if orientation is not None and np.linalg.norm(orientation) > 0:
            return orientation
        
        return None
    
    def _get_max_fiber_fraction(self, pos):
        """
        Get the maximum fiber fraction across all fiber populations at a position.
        
        Args:
            pos: Position in voxel coordinates
            
        Returns:
            float: Maximum volume fraction (0.0 to 1.0)
        """
        max_f = 0.0
        for fiber_idx in range(self.bedpostx.num_fibers):
            f = self.bedpostx.get_fiber_fraction(pos, fiber_idx)
            if f > max_f:
                max_f = f
        return max_f
    
    def reset_stats(self):
        """Reset tracking statistics."""
        self.stats = {
            'total_streamlines': 0,
            'total_steps': 0,
            'terminated_angle': 0,
            'terminated_boundary': 0,
            'terminated_low_f': 0,
            'terminated_max_steps': 0
        }
    
    def print_stats(self):
        """Print tracking statistics."""
        print("\n=== Tracking Statistics ===")
        print(f"Total streamlines: {self.stats['total_streamlines']}")
        print(f"Total steps: {self.stats['total_steps']}")
        print(f"Terminated by angle constraint: {self.stats['terminated_angle']}")
        print(f"Terminated at brain boundary: {self.stats['terminated_boundary']}")
        print(f"Terminated by low fiber fraction: {self.stats['terminated_low_f']}")
        print(f"Terminated at max steps: {self.stats['terminated_max_steps']}")
