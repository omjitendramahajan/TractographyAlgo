"""
Test script for BedpostxData class with memory-mapped loading.
"""

import numpy as np
import sys
import time

# Add tractography package to path
sys.path.insert(0, '/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo')

from tractography import BedpostxData

# =============================================================================
# Configuration
# =============================================================================
BEDPOSTX_DIR = "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP_uncompressed"

print("=" * 60)
print("Testing BedpostxData Class (Memory-Mapped Mode)")
print("=" * 60)

# =============================================================================
# Test 1: Load with memory-mapped mode (default now)
# =============================================================================
print("\n[TEST 1] Loading BedpostxData with memory-mapped mode...")
data = BedpostxData(
    bedpostx_dir=BEDPOSTX_DIR,
    num_fibers=1,
    load_samples=True,
    load_dyads=False,       # Skip dyads to save RAM
    use_memory_map=True    # Memory-mapped mode
)
print(f"\n=== BASIC INFO ===")
print(f"Volume shape: {data.shape}")
print(f"Voxel size (mm): {data.voxel_size}")
print(f"Number of fibers: {data.num_fibers}")
print(f"Number of MCMC samples: {data.n_samples}")
print(f"Memory-mapped mode: {data.use_memory_map}")

# # =============================================================================
# # Test 2: Test sample_orientation_probabilistic
# # =============================================================================
# print("\n[TEST 2] Testing sample_orientation_probabilistic()...")

# # Test at center voxel
center_voxel = (173, 173, 56)
# print(f"\nSampling orientations at voxel {center_voxel}:")

# for i in range(5):
#     orientation = data.sample_orientation_probabilistic(center_voxel)
#     if orientation is not None:
#         print(f"  Sample {i+1}: [{orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}]")
#     else:
#         print(f"  Sample {i+1}: None (no valid orientation)")

# # =============================================================================
# # Test 3: Test at multiple voxels
# # =============================================================================
# print("\n[TEST 3] Testing at multiple voxels...")

# test_voxels = [
#     (173, 173, 56),   # Center
#     (200, 173, 56),   # White matter region
#     (100, 100, 50),   # Another location
# ]

# for voxel in test_voxels:
#     if data.is_valid_position(voxel):
#         orientation = data.sample_orientation_probabilistic(voxel)
#         if orientation is not None:
#             norm = np.linalg.norm(orientation)
#             print(f"  Voxel {voxel}: orientation=[{orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}], norm={norm:.4f}")
#         else:
#             print(f"  Voxel {voxel}: No valid orientation")
#     else:
#         print(f"  Voxel {voxel}: Outside brain mask")

# # =============================================================================
# # Test 4: Performance test + Collect samples for visualization
# # =============================================================================
# print("\n[TEST 4] Performance test (50 samples with timing)...")

# # Warm-up call to handle any lazy loading or initial memory-mapping overhead
# _ = data.sample_orientation_probabilistic(center_voxel)

# # Collect samples for visualization
# orientations = []
# total_start = time.time()
# for i in range(50):
#     sample_start = time.time()
#     orientation = data.sample_orientation_probabilistic(center_voxel)
#     sample_time = time.time() - sample_start
#     if orientation is not None:
#         orientations.append(orientation)
#         if i < 10:  # Only print first 10
#             print(f"  Sample {i+1}: [{orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}] - {sample_time*1000:.0f} ms")

# total_elapsed = time.time() - total_start
# print(f"  ... (50 samples collected)")
# print(f"\n  Total: {total_elapsed:.2f} seconds")
# print(f"  Average per sample: {total_elapsed/50*1000:.1f} ms")

# # =============================================================================
# # Test 5: Check is_valid_position
# # =============================================================================
# print("\n[TEST 5] Testing is_valid_position()...")
# test_positions = [
#     (173, 173, 56),    # Should be valid (center)
#     (0, 0, 0),         # Edge - might be outside mask
#     (-1, 0, 0),        # Invalid (negative)
#     (500, 500, 500),   # Invalid (out of bounds)
# ]

# for pos in test_positions:
#     valid = data.is_valid_position(pos)
#     print(f"  Position {pos}: {'Valid' if valid else 'Invalid'}")

# print("\n" + "=" * 60)
# print("All BedpostxData tests completed!")
# print("=" * 60)

# =============================================================================
# Test 6: Test ProbabilisticTracker
# =============================================================================
print("\n[TEST 6] Testing ProbabilisticTracker...")

from tractography.tracker_draft import ProbabilisticTracker

# Initialize tracker
tracker = ProbabilisticTracker(
    bedpostx_data=data,
    step_size=0.1,        # 0.5 mm steps
    max_steps=20000,       # Maximum 2000 steps per direction
    min_f_threshold=0.05, # Stop if fiber fraction < 5%
    angle_threshold=90    # Stop if angle > 60 degrees
)

print(f"\nTracker initialized:")
print(f"  Step size: {tracker.step_size} mm")
print(f"  Max steps: {tracker.max_steps}")
print(f"  Min fiber fraction: {tracker.min_f_threshold}")
print(f"  Max angle: {np.degrees(tracker.max_angle):.0f} degrees")

# Track a single streamline from center voxel
print(f"\nTracking streamline from seed {center_voxel}...")
start_time = time.time()
streamline = tracker.track(center_voxel)
track_time = time.time() - start_time

if streamline is not None:
    print(f"  Streamline generated!")
    print(f"  Number of points: {len(streamline)}")
    print(f"  Tracking time: {track_time*1000:.1f} ms")
    
    # Calculate streamline length in mm
    diffs = np.diff(streamline, axis=0)
    lengths = np.sqrt(np.sum(diffs**2 * data.voxel_size**2, axis=1))
    total_length = np.sum(lengths)
    print(f"  Streamline length: {total_length:.1f} mm")
    
    # Show first and last points
    print(f"  Start point: ({streamline[0, 0]:.1f}, {streamline[0, 1]:.1f}, {streamline[0, 2]:.1f})")
    print(f"  End point: ({streamline[-1, 0]:.1f}, {streamline[-1, 1]:.1f}, {streamline[-1, 2]:.1f})")
else:
    print("  Failed to generate streamline!")

# Print tracker statistics
tracker.print_stats()

# Track multiple streamlines and collect for visualization
print("\n  Tracking 10 streamlines for visualization...")
tracker.reset_stats()
streamlines = []
for _ in range(10):
    sl = tracker.track(center_voxel)
    if sl is not None:
        streamlines.append(sl)
tracker.print_stats()
print(f"  Collected {len(streamlines)} valid streamlines")

# =============================================================================
# Export streamlines to TRK format for FSLeyes visualization
# =============================================================================
print("\n[EXPORT] Saving streamlines to TRK format for FSLeyes...")

from nibabel.streamlines import Tractogram, save as save_tractogram

# Convert voxel coordinates to world (mm) coordinates
world_streamlines = []
for sl in streamlines:
    # Homogeneous coordinates: append ones for matrix multiplication
    ones = np.ones((len(sl), 1))
    vox_homo = np.hstack([sl, ones])  # (N, 4)
    # Transform: world = affine @ voxel
    world = (data.affine @ vox_homo.T).T[:, :3]  # (N, 3)
    world_streamlines.append(world)

# Create tractogram with proper affine
tractogram = Tractogram(
    streamlines=world_streamlines,
    affine_to_rasmm=np.eye(4)  # Already in world coordinates
)

# Save to TRK file
trk_output = "test_streamlines.trk"
save_tractogram(tractogram, trk_output)
print(f"  Saved to: {trk_output}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)

# =============================================================================
# Visualization: Orientations + Streamlines
# =============================================================================
# print("\n[VISUALIZATION] Creating 3D plots...")

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(16, 6))

# # --- Subplot 1: Orientation uncertainty ---
# ax1 = fig.add_subplot(121, projection='3d')

# # Plot each orientation as a bidirectional line through origin
# for i, orient in enumerate(orientations):
#     x, y, z = orient
#     ax1.plot([-x, x], [-y, y], [-z, z], 
#             color='blue', alpha=0.5, linewidth=1)

# # Plot the mean orientation as a thick red line
# mean_orient = np.mean(orientations, axis=0)
# mean_orient = mean_orient / np.linalg.norm(mean_orient)
# ax1.plot([-mean_orient[0], mean_orient[0]], 
#         [-mean_orient[1], mean_orient[1]], 
#         [-mean_orient[2], mean_orient[2]], 
#         color='red', linewidth=3, label='Mean orientation')

# ax1.scatter([0], [0], [0], color='black', s=50, label='Origin')
# ax1.set_xlim([-1.2, 1.2])
# ax1.set_ylim([-1.2, 1.2])
# ax1.set_zlim([-1.2, 1.2])
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title(f'Orientation Uncertainty\nat Voxel {center_voxel}')
# ax1.legend()

# # --- Subplot 2: Streamlines ---
# ax2 = fig.add_subplot(122, projection='3d')

# # Plot each streamline with different color
# colors = plt.cm.tab10(np.linspace(0, 1, len(streamlines)))
# for i, sl in enumerate(streamlines):
#     ax2.plot(sl[:, 0], sl[:, 1], sl[:, 2], 
#             color=colors[i], linewidth=1.5, alpha=0.8)

# # Mark seed point
# ax2.scatter([center_voxel[0]], [center_voxel[1]], [center_voxel[2]], 
#            color='red', s=100, marker='*', label='Seed')

# # Set axis labels
# ax2.set_xlabel('X (voxels)')
# ax2.set_ylabel('Y (voxels)')
# ax2.set_zlabel('Z (voxels)')
# ax2.set_title(f'Probabilistic Streamlines\n({len(streamlines)} from seed)')
# ax2.legend()

# plt.tight_layout()
# plt.savefig('tractography_visualization.png', dpi=150)
# print("  Saved to: tractography_visualization.png")
# plt.show()