"""
Seed-to-Target Tractography Simulation

This module provides functions for running tractography simulations with both 
a start (seed) point and an end (target) point. It builds on the HybridAgent 
and GroundTruth classes to perform goal-directed fiber tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection
from typing import Tuple, List, Dict, Optional

from .Agent import HybridAgent
from .DataGenerator import GroundTruth


class SeedToTargetSimulation:
    """
    Runs tractography simulations between a seed region and a target region.
    
    Attributes:
        gt_object: The GroundTruth data generator object
        grid_size: Size of the simulation grid
        interp_u_high: Interpolator for high-res U component
        interp_v_high: Interpolator for high-res V component
        grid_bounds: Bounds for the simulation grid
    """
    
    def __init__(self, grid_size: int = 30, microscopy_sample_rate: int = 5, 
                 patch_size: int = 4, gt_generator_func=None):
        """
        Initialize the simulation environment.
        
        Args:
            grid_size: Size of the square simulation grid
            microscopy_sample_rate: Sampling rate for microscopy data (every Nth row)
            patch_size: Size of DWI patches
            gt_generator_func: Optional custom ground truth generator function
        """
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.microscopy_sample_rate = microscopy_sample_rate
        
        # Create ground truth data
        self.gt_object = GroundTruth(
            grid_size, 
            microscopy_sample_rate, 
            patch_size,
            gt_generator_func=gt_generator_func
        )
        self.gt_object.generate_fods(num_bins=360)
        
        # Extract data grids
        self.gt_data = self.gt_object.gt_grid
        self.microscopy_data = self.gt_object.microscopy_grid
        self.dwi_data = self.gt_object.dwi_grid
        
        # Create interpolators
        self._setup_interpolators()
        
    def _setup_interpolators(self):
        """Setup high-resolution interpolators for microscopy data."""
        u_high_res = self.microscopy_data[:, :, 0]
        v_high_res = self.microscopy_data[:, :, 1]
        
        x = np.arange(0, self.grid_size, 1)
        y = np.arange(0, self.grid_size, 1)
        
        self.interp_u_high = RegularGridInterpolator(
            (y, x), u_high_res, 
            method='nearest', 
            bounds_error=False, 
            fill_value=np.nan
        )
        self.interp_v_high = RegularGridInterpolator(
            (y, x), v_high_res, 
            method='nearest', 
            bounds_error=False, 
            fill_value=np.nan
        )
        self.grid_bounds = (0, self.grid_size - 1, 0, self.grid_size - 1)

    def run_seed_to_target_simulation(
        self,
        seed_point: Tuple[float, float],
        target_point: Tuple[float, float],
        num_agents: int = 50,
        pixel_spread: float = 0.8,
        step_size: float = 0.1,
        max_steps: int = 500,
        target_radius: float = 1.0,
        bidirectional: bool = True
    ) -> Dict:
        """
        Run a simulation from a seed region towards a target region.
        
        This seeds multiple agents around the seed point and tracks them.
        Agents that come within the target_radius of the target point are 
        considered successful connections.
        
        Args:
            seed_point: (x, y) coordinates of the seed pixel center
            target_point: (x, y) coordinates of the target pixel center
            num_agents: Number of agents to simulate
            pixel_spread: How much of the seed pixel to cover (0.0 to 1.0)
            step_size: Step size for each agent movement
            max_steps: Maximum number of steps per agent
            target_radius: Radius around target point for success
            bidirectional: If True, also run from target to seed
            
        Returns:
            Dictionary containing:
                - 'forward_paths': List of agent results from seed to target
                - 'reverse_paths': List of agent results from target to seed (if bidirectional)
                - 'successful_forward': Count of agents reaching target
                - 'successful_reverse': Count of agents reaching seed (if bidirectional)
                - 'seed_point': Seed point coordinates
                - 'target_point': Target point coordinates
        """
        results = {
            'forward_paths': [],
            'reverse_paths': [],
            'successful_forward': 0,
            'successful_reverse': 0,
            'seed_point': seed_point,
            'target_point': target_point,
            'target_radius': target_radius
        }
        
        # Run forward simulation (seed -> target)
        print(f"Running forward simulation: seed {seed_point} -> target {target_point}")
        forward_results = self._run_agents_from_point(
            center_point=seed_point,
            target_point=target_point,
            num_agents=num_agents,
            pixel_spread=pixel_spread,
            step_size=step_size,
            max_steps=max_steps,
            target_radius=target_radius
        )
        results['forward_paths'] = forward_results['paths']
        results['successful_forward'] = forward_results['successful_count']
        
        # Run reverse simulation (target -> seed) if bidirectional
        if bidirectional:
            print(f"Running reverse simulation: target {target_point} -> seed {seed_point}")
            reverse_results = self._run_agents_from_point(
                center_point=target_point,
                target_point=seed_point,
                num_agents=num_agents,
                pixel_spread=pixel_spread,
                step_size=step_size,
                max_steps=max_steps,
                target_radius=target_radius
            )
            results['reverse_paths'] = reverse_results['paths']
            results['successful_reverse'] = reverse_results['successful_count']
        
        return results
    
    def _run_agents_from_point(
        self,
        center_point: Tuple[float, float],
        target_point: Tuple[float, float],
        num_agents: int,
        pixel_spread: float,
        step_size: float,
        max_steps: int,
        target_radius: float
    ) -> Dict:
        """Run multiple agents from a center point towards a target."""
        agent_results = []
        successful_count = 0
        
        for i in range(num_agents):
            # Generate random start point with spread around center
            rand_offset_x = (np.random.rand() - 0.5) * pixel_spread
            rand_offset_y = (np.random.rand() - 0.5) * pixel_spread
            
            start_x = center_point[0] + 0.5 + rand_offset_x
            start_y = center_point[1] + 0.5 + rand_offset_y
            current_start_point = [start_x, start_y]
            
            # Create and run the agent
            agent = HybridAgent(
                start_point_xy=current_start_point,
                gt_object=self.gt_object,
                interp_u_high=self.interp_u_high,
                interp_v_high=self.interp_v_high,
                grid_bounds=self.grid_bounds
            )
            
            # Run simulation with early stopping at target
            reached_target = self._run_agent_with_target(
                agent, max_steps, step_size, target_point, target_radius
            )
            
            # Store results
            path = np.array(agent.get_path())
            sources = agent.source_history
            
            result = {
                'path': path,
                'sources': sources,
                'start': current_start_point,
                'reached_target': reached_target
            }
            agent_results.append(result)
            
            if reached_target:
                successful_count += 1
        
        return {'paths': agent_results, 'successful_count': successful_count}
    
    def _run_agent_with_target(
        self,
        agent: HybridAgent,
        max_steps: int,
        step_size: float,
        target_point: Tuple[float, float],
        target_radius: float
    ) -> bool:
        """
        Run an agent simulation with early stopping when target is reached.
        
        Returns:
            True if agent reached within target_radius of target_point
        """
        target = np.array([target_point[0] + 0.5, target_point[1] + 0.5])
        
        for step in range(max_steps):
            if not agent.is_active:
                break
                
            agent.step(step_size)
            
            # Check if we've reached the target
            current_pos = agent.pos
            distance = np.linalg.norm(current_pos - target)
            
            if distance <= target_radius:
                return True
        
        return False
    
    def plot_seed_to_target_results(
        self,
        results: Dict,
        figsize: Tuple[int, int] = (12, 12),
        show_background: bool = True,
        ax=None
    ):
        """
        Visualize the results of a seed-to-target simulation.
        
        Args:
            results: Output from run_seed_to_target_simulation()
            figsize: Figure size if creating new figure
            show_background: Whether to show microscopy/DWI background
            ax: Optional existing axis to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background if requested
        if show_background:
            self.gt_object.plot_microscopy(ax=ax, color='lightgray')
            self.gt_object.plot_dwi_overlay(ax=ax)
        
        # Plot forward paths
        for res in results['forward_paths']:
            path = res['path']
            sources = res['sources']
            reached = res['reached_target']
            
            if len(path) < 2:
                continue
            
            # Create segments
            segments = np.stack((path[:-1], path[1:]), axis=1)
            
            # Color by source (green=high-res, red=FOD)
            colors = ['green' if s == 'high' else 'red' for s in sources[:-1]]
            
            # Adjust alpha for successful vs unsuccessful paths
            alpha = 0.9 if reached else 0.3
            linewidth = 2.0 if reached else 1.0
            
            lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
            ax.add_collection(lc)
        
        # Plot reverse paths if they exist
        for res in results.get('reverse_paths', []):
            path = res['path']
            sources = res['sources']
            reached = res['reached_target']
            
            if len(path) < 2:
                continue
            
            segments = np.stack((path[:-1], path[1:]), axis=1)
            
            # Use different color scheme for reverse (cyan=high-res, magenta=FOD)
            colors = ['cyan' if s == 'high' else 'magenta' for s in sources[:-1]]
            
            alpha = 0.9 if reached else 0.3
            linewidth = 2.0 if reached else 1.0
            
            lc = LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
            ax.add_collection(lc)
        
        # Plot seed and target regions
        seed = results['seed_point']
        target = results['target_point']
        radius = results['target_radius']
        
        # Seed region (blue circle)
        seed_circle = plt.Circle(
            (seed[0] + 0.5, seed[1] + 0.5), 
            radius, 
            fill=False, 
            color='blue', 
            linewidth=3, 
            linestyle='--',
            label='Seed Region'
        )
        ax.add_patch(seed_circle)
        ax.plot(seed[0] + 0.5, seed[1] + 0.5, 'b*', markersize=15, label='Seed Center')
        
        # Target region (orange circle)
        target_circle = plt.Circle(
            (target[0] + 0.5, target[1] + 0.5), 
            radius, 
            fill=False, 
            color='orange', 
            linewidth=3, 
            linestyle='--',
            label='Target Region'
        )
        ax.add_patch(target_circle)
        ax.plot(target[0] + 0.5, target[1] + 0.5, 'o', color='orange', markersize=15, label='Target Center')
        
        # Legend
        ax.plot([], [], color='green', linewidth=2.5, label='Forward: High-Res')
        ax.plot([], [], color='red', linewidth=2.5, label='Forward: FOD')
        if results.get('reverse_paths'):
            ax.plot([], [], color='cyan', linewidth=2.5, label='Reverse: High-Res')
            ax.plot([], [], color='magenta', linewidth=2.5, label='Reverse: FOD')
        
        # Formatting
        ax.set_xlim([-1, self.grid_size])
        ax.set_ylim([-1, self.grid_size])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Title with stats
        fwd_success = results['successful_forward']
        fwd_total = len(results['forward_paths'])
        title = f"Seed-to-Target Simulation\n"
        title += f"Forward: {fwd_success}/{fwd_total} reached target"
        
        if results.get('reverse_paths'):
            rev_success = results['successful_reverse']
            rev_total = len(results['reverse_paths'])
            title += f" | Reverse: {rev_success}/{rev_total} reached seed"
        
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        
        return ax


def run_seed_to_target_demo(
    seed_point: Tuple[float, float] = (15, 9),
    target_point: Tuple[float, float] = (5, 20),
    num_agents: int = 50,
    grid_size: int = 30,
    bidirectional: bool = True
):
    """
    Convenience function to run a complete seed-to-target simulation demo.
    
    Args:
        seed_point: Starting seed location (x, y)
        target_point: Target location (x, y)
        num_agents: Number of tracking agents
        grid_size: Size of the simulation grid
        bidirectional: Whether to also track from target to seed
        
    Returns:
        Tuple of (simulation object, results dictionary)
    """
    # Create simulation
    sim = SeedToTargetSimulation(
        grid_size=grid_size,
        microscopy_sample_rate=5,
        patch_size=4
    )
    
    # Run simulation
    results = sim.run_seed_to_target_simulation(
        seed_point=seed_point,
        target_point=target_point,
        num_agents=num_agents,
        pixel_spread=0.8,
        step_size=0.1,
        max_steps=500,
        target_radius=2.0,
        bidirectional=bidirectional
    )
    
    # Plot
    sim.plot_seed_to_target_results(results)
    plt.tight_layout()
    plt.show()
    
    return sim, results


# --- Example Usage ---
if __name__ == "__main__":
    # Run the demo
    sim, results = run_seed_to_target_demo(
        seed_point=(15, 9),
        target_point=(5, 20),
        num_agents=50,
        bidirectional=True
    )
    
    print(f"\n=== Simulation Results ===")
    print(f"Forward (Seed -> Target): {results['successful_forward']}/{len(results['forward_paths'])} successful")
    print(f"Reverse (Target -> Seed): {results['successful_reverse']}/{len(results['reverse_paths'])} successful")
