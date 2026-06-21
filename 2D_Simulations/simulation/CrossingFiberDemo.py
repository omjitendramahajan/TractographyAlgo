"""
Crossing Fiber Demo: Shows Hybrid tracking advantage over FOD-only

This demonstrates that microscopy priors help resolve crossing fiber structures 
where the goal is to track ONE specific bundle through a crossing region.

Key Insight:
- FOD averages all fibers in a voxel -> gives averaged/confused direction at crossings
- Microscopy sees individual rows -> can identify and follow the correct bundle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Tuple

from .ComparisonSimulation import ComparisonSimulation


def two_bundle_crossing_field(X, Y, size):
    """
    Creates a ground truth with TWO distinct fiber bundles crossing in the CENTER.
    
    - Bundle A (rows 0-14): Horizontal fibers going RIGHT
    - Bundle B (rows 15-29): Diagonal fibers going UP-RIGHT
    
    In the CENTER (x from 10-20), the bundles interleave row-by-row.
    This simulates a crossing region where microscopy can resolve individual bundles.
    """
    epsilon = 1e-10
    u = np.zeros_like(X, dtype=float)
    v = np.zeros_like(Y, dtype=float)
    
    center_x = size / 2
    crossing_start = int(size * 0.3)
    crossing_end = int(size * 0.7)
    
    for i in range(size):
        for j in range(size):
            # CROSSING ZONE: fibers interleave by row
            if crossing_start <= j <= crossing_end:
                if i % 2 == 0:
                    # Bundle A: Horizontal
                    u[i, j] = 1.0
                    v[i, j] = 0.0
                else:
                    # Bundle B: Diagonal up
                    u[i, j] = 0.7
                    v[i, j] = 0.7
            else:
                # NON-CROSSING ZONES: separated bundles
                if i < size / 2:
                    # Lower half: Bundle A (horizontal)
                    u[i, j] = 1.0
                    v[i, j] = 0.0
                else:
                    # Upper half: Bundle B (diagonal)
                    u[i, j] = 0.7
                    v[i, j] = 0.7
    
    # Normalize
    magnitude = np.sqrt(u**2 + v**2) + epsilon
    u_normalized = u / magnitude
    v_normalized = v / magnitude
    
    return np.stack((u_normalized, v_normalized), axis=-1)


def curved_crossing_field(X, Y, size):
    """
    Two curved bundles that CROSS in the center:
    - Bundle A: Curves from lower-left to lower-right (concave up)
    - Bundle B: Curves from upper-left to upper-right (concave down)
    
    They intersect in the middle, creating a complex crossing region.
    """
    epsilon = 1e-10
    u = np.zeros_like(X, dtype=float)
    v = np.zeros_like(Y, dtype=float)
    
    center_x, center_y = size / 2, size / 2
    
    for i in range(size):
        for j in range(size):
            # Calculate which bundle this point belongs to
            # Based on row parity in the crossing zone
            
            dx = j - center_x
            dy = i - center_y
            
            # Crossing zone: center third of the grid
            in_crossing_zone = abs(dx) < size/4 and abs(dy) < size/4
            
            if in_crossing_zone:
                # In crossing zone: alternate by row
                if i % 2 == 0:
                    # Bundle A: curving downward
                    curve_factor = dx / (size/2)
                    u[i, j] = 1.0
                    v[i, j] = -0.5 * curve_factor
                else:
                    # Bundle B: curving upward
                    curve_factor = dx / (size/2)
                    u[i, j] = 1.0
                    v[i, j] = 0.5 * curve_factor
            else:
                # Outside crossing: radial flow based on position
                if i < center_y:
                    # Lower region: horizontal with slight downward curve
                    u[i, j] = 1.0
                    v[i, j] = -0.2
                else:
                    # Upper region: horizontal with slight upward curve
                    u[i, j] = 1.0
                    v[i, j] = 0.2
    
    # Normalize
    magnitude = np.sqrt(u**2 + v**2) + epsilon
    u_normalized = u / magnitude
    v_normalized = v / magnitude
    
    return np.stack((u_normalized, v_normalized), axis=-1)


def bundle_tracking_challenge_field(X, Y, size):
    """
    The ULTIMATE challenge for tractography:
    
    Creates a field where you MUST track the correct bundle through a crossing.
    - Two bundles cross at 90 degrees in the center
    - The goal: track Bundle A from left to right (staying horizontal)
    - Challenge: FOD will average to 45 degrees at crossing
    - Microscopy can see which rows are Bundle A
    """
    epsilon = 1e-10
    u = np.zeros_like(X, dtype=float)
    v = np.zeros_like(Y, dtype=float)
    
    center_x, center_y = size / 2, size / 2
    crossing_radius = size / 4
    
    for i in range(size):
        for j in range(size):
            dx = j - center_x
            dy = i - center_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < crossing_radius:
                # CROSSING ZONE: alternating bundles
                if i % 2 == 0:
                    # Bundle A: HORIZONTAL (our target to track)
                    u[i, j] = 1.0
                    v[i, j] = 0.0
                else:
                    # Bundle B: VERTICAL (crossing bundle)
                    u[i, j] = 0.0
                    v[i, j] = 1.0
            else:
                # Outside crossing: identify bundle by which is closer
                # Left/Right sides: Bundle A (horizontal)
                # Top/Bottom sides: Bundle B (vertical)
                if abs(dx) > abs(dy):
                    # Horizontal bundle A region
                    u[i, j] = 1.0
                    v[i, j] = 0.0
                else:
                    # Vertical bundle B region
                    u[i, j] = 0.0
                    v[i, j] = 1.0
    
    # Normalize
    magnitude = np.sqrt(u**2 + v**2) + epsilon
    u_normalized = u / magnitude
    v_normalized = v / magnitude
    
    return np.stack((u_normalized, v_normalized), axis=-1)


def run_crossing_fiber_comparison(
    field_type: str = "90_degree",
    seed_point: Tuple[float, float] = None,
    target_point: Tuple[float, float] = None,
    num_agents: int = 100,
    grid_size: int = 30
):
    """
    Run comparison on crossing fiber structure.
    
    Args:
        field_type: "two_bundle", "curved", or "90_degree"
    """
    # Select field generator and appropriate seed/target
    if field_type == "two_bundle":
        generator = two_bundle_crossing_field
        title = "Two Bundle Crossing"
        default_seed = (3, 8)
        default_target = (27, 8)
    elif field_type == "curved":
        generator = curved_crossing_field
        title = "Curved Bundle Crossing"
        default_seed = (3, 15)
        default_target = (27, 15)
    else:  # 90_degree
        generator = bundle_tracking_challenge_field
        title = "90° Crossing Challenge"
        default_seed = (3, 15)  # Start on even row (Bundle A)
        default_target = (27, 15)  # End on same side
    
    seed_point = seed_point or default_seed
    target_point = target_point or default_target
    
    print(f"\n{'='*70}")
    print(f"CROSSING FIBER COMPARISON: {title}")
    print(f"{'='*70}")
    print(f"\nSeed: {seed_point} -> Target: {target_point}")
    print(f"\nIn the CROSSING ZONE (center of grid):")
    print(f"  - FOD averages perpendicular directions -> 45° or confused")
    print(f"  - Microscopy sees alternating rows -> can pick correct bundle")
    print(f"{'='*70}\n")
    
    # Create simulation with custom field
    sim = ComparisonSimulation(
        grid_size=grid_size,
        microscopy_sample_rate=4,  # Every 4th row sampled
        patch_size=4,
        gt_generator_func=generator
    )
    
    # Run comparison
    results = sim.run_comparison(
        seed_point=seed_point,
        target_point=target_point,
        num_agents=num_agents,
        target_radius=2.0,
        step_size=0.1,
        max_steps=600
    )
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: Ground truth views
    ax1 = fig.add_subplot(2, 3, 1)
    gt = sim.gt_object.gt_grid
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    ax1.quiver(X + 0.5, Y + 0.5, gt[:,:,0], gt[:,:,1], 
               pivot='middle', headwidth=3, headlength=4, scale=40, alpha=0.7,
               color='darkgreen')
    ax1.set_title(f"Ground Truth: {title}\n(Color shows vertical component)")
    ax1.set_xlim([0, grid_size])
    ax1.set_ylim([0, grid_size])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    # Highlight crossing zone
    crossing_center = grid_size / 2
    crossing_radius = grid_size / 4
    circle = plt.Circle((crossing_center, crossing_center), crossing_radius, 
                        fill=False, color='yellow', linewidth=2, linestyle='--', label='Crossing Zone')
    ax1.add_patch(circle)
    ax1.legend(loc='upper right')
    
    # DWI view (what FOD sees)
    ax2 = fig.add_subplot(2, 3, 2)
    dwi = sim.gt_object.dwi_grid
    X_dwi, Y_dwi = sim.gt_object.X_dwi, sim.gt_object.Y_dwi
    ax2.quiver(X_dwi, Y_dwi, dwi[:,:,0], dwi[:,:,1],
               pivot='middle', headwidth=4, headlength=5, scale=15, color='red')
    ax2.set_title(f"DWI/FOD View\n(Averaged directions - notice crossing zone!)")
    ax2.set_xlim([0, grid_size])
    ax2.set_ylim([0, grid_size])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    circle2 = plt.Circle((crossing_center, crossing_center), crossing_radius,
                         fill=False, color='yellow', linewidth=2, linestyle='--')
    ax2.add_patch(circle2)
    
    # Microscopy view
    ax3 = fig.add_subplot(2, 3, 3)
    sim.gt_object.plot_microscopy(ax=ax3, color='blue')
    ax3.set_title(f"Microscopy View\n(Partial but TRUE directions)")
    circle3 = plt.Circle((crossing_center, crossing_center), crossing_radius,
                         fill=False, color='yellow', linewidth=2, linestyle='--')
    ax3.add_patch(circle3)
    
    # Row 2: Tracking results
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    summary_text = f"""
    RESULTS SUMMARY
    ---------------
    
    HYBRID (Microscopy + FOD):
      Success Rate: {results['hybrid']['success_rate']:.1f}%
      Agents Reached Target: {results['hybrid']['successful_count']}/{num_agents}
      High-Res Steps: {results['hybrid']['high_res_steps']} ({results['hybrid']['high_res_ratio']:.1f}%)
      FOD Steps: {results['hybrid']['fod_steps']}
    
    FOD-ONLY:
      Success Rate: {results['fod_only']['success_rate']:.1f}%
      Agents Reached Target: {results['fod_only']['successful_count']}/{num_agents}
      FOD Steps: {results['fod_only']['fod_steps']}
    
    IMPROVEMENT: {results['hybrid']['success_rate'] - results['fod_only']['success_rate']:.1f}%
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hybrid tracking
    ax5 = fig.add_subplot(2, 3, 5)
    sim._plot_paths(ax5, results['hybrid'], results, 
                   title="HYBRID Tracking", is_hybrid=True)
    
    # FOD-only tracking
    ax6 = fig.add_subplot(2, 3, 6)
    sim._plot_paths(ax6, results['fod_only'], results,
                   title="FOD-ONLY Tracking", is_hybrid=False)
    
    plt.suptitle(f"Crossing Fiber Resolution: {title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = f'/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/figures/crossing_fiber_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    return sim, results


if __name__ == "__main__":
    # Run the 90-degree crossing challenge
    # This is the hardest case where FOD averaging really fails
    sim, results = run_crossing_fiber_comparison(
        field_type="90_degree",
        num_agents=100
    )
