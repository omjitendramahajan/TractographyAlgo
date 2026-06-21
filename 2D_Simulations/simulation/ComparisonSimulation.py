"""
Comparison Simulation: Microscopy Priors vs FOD-Only

This module compares tractography performance when:
1. Using Hybrid approach (Microscopy + FOD fallback)
2. Using FOD-only (no microscopy priors)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection
from typing import Tuple, List, Dict

from .Agent import HybridAgent
from .DataGenerator import GroundTruth


class FODOnlyAgent:
    """
    An agent that uses ONLY FOD data (no microscopy priors).
    This serves as a baseline comparison for the HybridAgent.
    """
    def __init__(self, start_point_xy, gt_object, grid_bounds):
        self.pos = np.array(start_point_xy, dtype=float)
        self.gt = gt_object
        self.min_x, self.max_x, self.min_y, self.max_y = grid_bounds
        
        self.path_history = [self.pos.copy()]
        self.is_active = True
        self.source_history = []

    def step(self, step_size):
        if not self.is_active:
            return

        x, y = self.pos
        
        # Check global bounds
        if not (self.min_x <= x < self.max_x and self.min_y <= y < self.max_y):
            self.is_active = False
            return
        
        try:
            # Always use FOD (probabilistic sampling)
            dwi_x_idx = int(x // self.gt.p_patch_size)
            dwi_y_idx = int(y // self.gt.p_patch_size)
            
            max_h, max_w, _ = self.gt.fod_grid.shape
            
            if 0 <= dwi_x_idx < max_w and 0 <= dwi_y_idx < max_h:
                angle = self.gt.sample_fod_direction(dwi_x_idx, dwi_y_idx)
                u = np.cos(angle)
                v = np.sin(angle)
                self.source_history.append('fod')
            else:
                self.is_active = False
                return
                
        except (ValueError, IndexError):
            self.is_active = False
            return

        # Move the agent
        vector = np.array([u, v])
        self.pos += vector * step_size
        self.path_history.append(self.pos.copy())

    def run_simulation(self, num_steps, step_size):
        for _ in range(num_steps):
            if self.is_active:
                self.step(step_size)
            else:
                break
        
    def get_path(self):
        return np.array(self.path_history)


class ComparisonSimulation:
    """
    Runs and compares tractography simulations:
    - Hybrid: Uses microscopy when available, FOD as fallback
    - FOD-Only: Uses only probabilistic FOD sampling
    """
    
    def __init__(self, grid_size: int = 30, microscopy_sample_rate: int = 5, 
                 patch_size: int = 4, gt_generator_func=None):
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.microscopy_sample_rate = microscopy_sample_rate
        
        # Create ground truth data
        self.gt_object = GroundTruth(
            grid_size, microscopy_sample_rate, patch_size,
            gt_generator_func=gt_generator_func
        )
        self.gt_object.generate_fods(num_bins=360)
        
        self.microscopy_data = self.gt_object.microscopy_grid
        self._setup_interpolators()
        
    def _setup_interpolators(self):
        """Setup high-resolution interpolators for microscopy data."""
        u_high_res = self.microscopy_data[:, :, 0]
        v_high_res = self.microscopy_data[:, :, 1]
        
        x = np.arange(0, self.grid_size, 1)
        y = np.arange(0, self.grid_size, 1)
        
        self.interp_u_high = RegularGridInterpolator(
            (y, x), u_high_res, method='nearest', 
            bounds_error=False, fill_value=np.nan
        )
        self.interp_v_high = RegularGridInterpolator(
            (y, x), v_high_res, method='nearest', 
            bounds_error=False, fill_value=np.nan
        )
        self.grid_bounds = (0, self.grid_size - 1, 0, self.grid_size - 1)

    def run_comparison(
        self,
        seed_point: Tuple[float, float],
        target_point: Tuple[float, float],
        num_agents: int = 50,
        pixel_spread: float = 0.8,
        step_size: float = 0.1,
        max_steps: int = 500,
        target_radius: float = 2.0
    ) -> Dict:
        """
        Run both Hybrid and FOD-only simulations for comparison.
        
        Returns:
            Dictionary with 'hybrid' and 'fod_only' results
        """
        print("=" * 60)
        print("COMPARISON: Hybrid (Microscopy + FOD) vs FOD-Only")
        print("=" * 60)
        
        # Run Hybrid simulation
        print(f"\n[1/2] Running HYBRID agents (Microscopy priors + FOD fallback)...")
        hybrid_results = self._run_agents(
            seed_point, target_point, num_agents, pixel_spread,
            step_size, max_steps, target_radius, use_microscopy=True
        )
        
        # Run FOD-only simulation
        print(f"\n[2/2] Running FOD-ONLY agents (no microscopy priors)...")
        fod_results = self._run_agents(
            seed_point, target_point, num_agents, pixel_spread,
            step_size, max_steps, target_radius, use_microscopy=False
        )
        
        results = {
            'hybrid': hybrid_results,
            'fod_only': fod_results,
            'seed_point': seed_point,
            'target_point': target_point,
            'target_radius': target_radius,
            'num_agents': num_agents
        }
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _run_agents(
        self,
        seed_point: Tuple[float, float],
        target_point: Tuple[float, float],
        num_agents: int,
        pixel_spread: float,
        step_size: float,
        max_steps: int,
        target_radius: float,
        use_microscopy: bool
    ) -> Dict:
        """Run agents with or without microscopy priors."""
        agent_results = []
        successful_count = 0
        total_path_length = 0
        high_res_steps = 0
        fod_steps = 0
        
        target = np.array([target_point[0] + 0.5, target_point[1] + 0.5])
        
        for i in range(num_agents):
            # Random start within seed region
            rand_offset_x = (np.random.rand() - 0.5) * pixel_spread
            rand_offset_y = (np.random.rand() - 0.5) * pixel_spread
            start_x = seed_point[0] + 0.5 + rand_offset_x
            start_y = seed_point[1] + 0.5 + rand_offset_y
            current_start = [start_x, start_y]
            
            # Create appropriate agent
            if use_microscopy:
                agent = HybridAgent(
                    start_point_xy=current_start,
                    gt_object=self.gt_object,
                    interp_u_high=self.interp_u_high,
                    interp_v_high=self.interp_v_high,
                    grid_bounds=self.grid_bounds
                )
            else:
                agent = FODOnlyAgent(
                    start_point_xy=current_start,
                    gt_object=self.gt_object,
                    grid_bounds=self.grid_bounds
                )
            
            # Run with target checking
            reached_target = False
            for step in range(max_steps):
                if not agent.is_active:
                    break
                agent.step(step_size)
                
                distance = np.linalg.norm(agent.pos - target)
                if distance <= target_radius:
                    reached_target = True
                    break
            
            # Collect stats
            path = np.array(agent.get_path())
            sources = agent.source_history
            
            path_length = len(path) * step_size
            total_path_length += path_length
            
            for s in sources:
                if s == 'high':
                    high_res_steps += 1
                else:
                    fod_steps += 1
            
            if reached_target:
                successful_count += 1
            
            agent_results.append({
                'path': path,
                'sources': sources,
                'start': current_start,
                'reached_target': reached_target,
                'path_length': path_length
            })
        
        return {
            'paths': agent_results,
            'successful_count': successful_count,
            'success_rate': successful_count / num_agents * 100,
            'avg_path_length': total_path_length / num_agents,
            'high_res_steps': high_res_steps,
            'fod_steps': fod_steps,
            'high_res_ratio': high_res_steps / max(1, high_res_steps + fod_steps) * 100
        }
    
    def _print_summary(self, results: Dict):
        """Print comparison summary."""
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        hybrid = results['hybrid']
        fod = results['fod_only']
        
        print(f"\n{'Metric':<30} {'Hybrid':<15} {'FOD-Only':<15}")
        print("-" * 60)
        print(f"{'Success Rate':<30} {hybrid['success_rate']:.1f}%{'':<10} {fod['success_rate']:.1f}%")
        print(f"{'Successful Agents':<30} {hybrid['successful_count']}/{results['num_agents']:<10} {fod['successful_count']}/{results['num_agents']}")
        print(f"{'Avg Path Length':<30} {hybrid['avg_path_length']:.2f}{'':<10} {fod['avg_path_length']:.2f}")
        print(f"{'High-Res Steps Used':<30} {hybrid['high_res_steps']:<15} {'N/A':<15}")
        print(f"{'FOD Steps Used':<30} {hybrid['fod_steps']:<15} {fod['fod_steps']:<15}")
        print(f"{'High-Res Usage Ratio':<30} {hybrid['high_res_ratio']:.1f}%{'':<10} {'0.0%':<15}")
        print("=" * 60)
    
    def plot_comparison(self, results: Dict, figsize: Tuple[int, int] = (16, 8)):
        """Create side-by-side comparison visualization."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot Hybrid results (left)
        self._plot_paths(axes[0], results['hybrid'], results, 
                        title="HYBRID: Microscopy + FOD", is_hybrid=True)
        
        # Plot FOD-only results (right)
        self._plot_paths(axes[1], results['fod_only'], results, 
                        title="FOD-ONLY: No Microscopy", is_hybrid=False)
        
        plt.tight_layout()
        plt.savefig('/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/figures/comparison_result.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig, axes
    
    def _plot_paths(self, ax, method_results: Dict, all_results: Dict, 
                   title: str, is_hybrid: bool):
        """Plot paths for one method."""
        # Background
        self.gt_object.plot_microscopy(ax=ax, color='lightgray')
        self.gt_object.plot_dwi_overlay(ax=ax)
        
        # Plot paths
        for res in method_results['paths']:
            path = res['path']
            sources = res['sources']
            reached = res['reached_target']
            
            if len(path) < 2:
                continue
            
            segments = np.stack((path[:-1], path[1:]), axis=1)
            
            if is_hybrid:
                colors = ['green' if s == 'high' else 'red' for s in sources[:-1]]
            else:
                colors = ['purple' for _ in sources[:-1]]
            
            alpha = 0.8 if reached else 0.2
            lw = 1.5 if reached else 0.8
            
            lc = LineCollection(segments, colors=colors, linewidths=lw, alpha=alpha)
            ax.add_collection(lc)
        
        # Seed and target markers
        seed = all_results['seed_point']
        target = all_results['target_point']
        radius = all_results['target_radius']
        
        seed_circle = plt.Circle((seed[0] + 0.5, seed[1] + 0.5), radius, 
                                  fill=False, color='blue', linewidth=3, linestyle='--')
        target_circle = plt.Circle((target[0] + 0.5, target[1] + 0.5), radius, 
                                    fill=False, color='orange', linewidth=3, linestyle='--')
        ax.add_patch(seed_circle)
        ax.add_patch(target_circle)
        ax.plot(seed[0] + 0.5, seed[1] + 0.5, 'b*', markersize=12)
        ax.plot(target[0] + 0.5, target[1] + 0.5, 'o', color='orange', markersize=12)
        
        # Legend
        if is_hybrid:
            ax.plot([], [], color='green', lw=2, label='High-Res (Microscopy)')
            ax.plot([], [], color='red', lw=2, label='FOD (Fallback)')
        else:
            ax.plot([], [], color='purple', lw=2, label='FOD Only')
        
        ax.legend(loc='upper right', fontsize=9)
        
        # Formatting
        ax.set_xlim([-1, self.grid_size])
        ax.set_ylim([-1, self.grid_size])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        success_rate = method_results['success_rate']
        ax.set_title(f"{title}\nSuccess: {method_results['successful_count']}/{all_results['num_agents']} ({success_rate:.1f}%)")


def run_comparison_demo(
    seed_point: Tuple[float, float] = (13, 3),
    target_point: Tuple[float, float] = (5, 14),
    num_agents: int = 100
):
    """Run a complete comparison demo."""
    sim = ComparisonSimulation(grid_size=30, microscopy_sample_rate=5, patch_size=4)
    
    results = sim.run_comparison(
        seed_point=seed_point,
        target_point=target_point,
        num_agents=num_agents,
        target_radius=2.0
    )
    
    sim.plot_comparison(results)
    
    return sim, results


if __name__ == "__main__":
    sim, results = run_comparison_demo()
