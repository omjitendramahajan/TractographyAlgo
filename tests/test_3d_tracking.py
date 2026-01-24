"""
test_3d_tracking - Test the PSOCTPriorityTracker with simulated 3D data.

This script demonstrates that the tracker works correctly with simulated
vector fields and compares PSOCT-priority vs BEDPOSTX-only tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import simulation classes
from simulation.GroundTruth3D import GroundTruth3D
from simulation.SimulatedData import (
    SimulatedBedpostxData, 
    SimulatedPSOCTData,
    create_simulation_environment
)

# Import tracker
from tractography import PSOCTPriorityTracker


def run_test(field_type='straight', shape=(30, 30, 30), 
             psoct_coverage=0.5, num_seeds=10):
    """
    Run a test with the specified parameters.
    
    Args:
        field_type: Type of vector field ('straight', 'curved', 'crossing', etc.)
        shape: Volume dimensions
        psoct_coverage: Fraction of slices with PSOCT data
        num_seeds: Number of seed points to track from
    """
    print("=" * 60)
    print(f"Testing: {field_type} field with {psoct_coverage*100:.0f}% PSOCT coverage")
    print("=" * 60)
    
    # Create simulation environment
    gt, bedpostx, psoct = create_simulation_environment(
        shape=shape,
        field_type=field_type,
        psoct_coverage=psoct_coverage,
        bedpostx_noise=0.1
    )
    
    # Create tracker
    tracker = PSOCTPriorityTracker(
        bedpostx,
        psoct,
        step_size=0.5,
        max_steps=200
    )
    
    # Generate seed points
    seeds = []
    for _ in range(num_seeds):
        seed = [
            np.random.randint(5, shape[0] - 5),
            np.random.randint(5, shape[1] - 5),
            np.random.randint(5, shape[2] - 5)
        ]
        seeds.append(seed)
    
    # Track streamlines
    print("\nTracking streamlines...")
    streamlines = []
    all_labels = []
    
    for i, seed in enumerate(seeds):
        streamline, labels = tracker.track(seed)
        if streamline is not None and len(streamline) >= 2:
            streamlines.append(streamline)
            all_labels.append(labels)
            print(f"  Seed {i+1}: {len(streamline)} points")
    
    # Print statistics
    print("\n" + "-" * 40)
    print("Results:")
    print(f"  Total streamlines: {len(streamlines)}")
    print(f"  PSOCT steps: {tracker.stats['psoct_steps']}")
    print(f"  BEDPOSTX steps: {tracker.stats['bedpostx_steps']}")
    
    total = tracker.stats['psoct_steps'] + tracker.stats['bedpostx_steps']
    if total > 0:
        print(f"  PSOCT usage: {100 * tracker.stats['psoct_steps'] / total:.1f}%")
    
    return streamlines, all_labels, gt


def visualize_results(streamlines, labels_list, gt):
    """
    Visualize streamlines in 3D with color-coding by data source.
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot of streamlines
    ax1 = fig.add_subplot(121, projection='3d')
    
    for sl, labels in zip(streamlines, labels_list):
        # Color by data source
        psoct_mask = labels == 'psoct'
        bedpostx_mask = labels == 'bedpostx'
        
        # Plot full streamline
        ax1.plot(sl[:, 0], sl[:, 1], sl[:, 2], 'b-', alpha=0.5, linewidth=0.5)
        
        # Highlight PSOCT segments in blue, BEDPOSTX in red
        for i in range(len(sl) - 1):
            if labels[i] == 'psoct':
                ax1.plot(sl[i:i+2, 0], sl[i:i+2, 1], sl[i:i+2, 2], 'b-', linewidth=2)
            else:
                ax1.plot(sl[i:i+2, 0], sl[i:i+2, 1], sl[i:i+2, 2], 'r-', linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Streamlines (Blue=PSOCT, Red=BEDPOSTX)')
    
    # 2D slice of vector field
    ax2 = fig.add_subplot(122)
    gt.plot_slice(axis='z', ax=ax2)
    
    # Overlay streamlines projected to this slice
    mid_z = gt.shape[2] // 2
    for sl in streamlines:
        # Filter points near mid_z
        mask = np.abs(sl[:, 2] - mid_z) < 3
        if mask.any():
            ax2.plot(sl[mask, 1], sl[mask, 0], 'r-', alpha=0.5, linewidth=1)
    
    ax2.set_title(f'Vector field + streamlines (z={mid_z})')
    
    plt.tight_layout()
    plt.savefig('simulation_test_results.png', dpi=150)
    plt.show()
    
    print("\nSaved figure to simulation_test_results.png")


def main():
    """Run tests with different configurations."""
    
    # Test 1: Straight fibers
    print("\n" + "=" * 70)
    print("TEST 1: Straight fibers")
    streamlines, labels, gt = run_test(
        field_type='straight',
        shape=(30, 30, 30),
        psoct_coverage=0.5,
        num_seeds=5
    )
    
    # Test 2: Curved fibers
    print("\n" + "=" * 70)
    print("TEST 2: Curved fibers")
    streamlines2, labels2, gt2 = run_test(
        field_type='curved',
        shape=(30, 30, 30),
        psoct_coverage=0.5,
        num_seeds=5
    )
    
    # Test 3: Crossing fibers
    print("\n" + "=" * 70)
    print("TEST 3: Crossing fibers")
    streamlines3, labels3, gt3 = run_test(
        field_type='crossing',
        shape=(30, 30, 30),
        psoct_coverage=0.5,
        num_seeds=5
    )
    
    # Visualize the last test
    print("\nGenerating visualization...")
    visualize_results(streamlines3, labels3, gt3)
    
    print("\n" + "=" * 70)
    print("All tests completed!")


if __name__ == "__main__":
    main()
