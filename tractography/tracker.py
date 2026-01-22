"""
PSOCTPriorityTracker - Tractography with PSOCT priority over dMRI.

This module contains the core tracking algorithm that uses PSOCT
orientations exclusively where available, falling back to BEDPOSTX only
where microscopy data is absent.
"""

import numpy as np


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
