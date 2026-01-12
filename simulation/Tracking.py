import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection

from .Agent import HybridAgent
from .DataGenerator import GroundTruth

# --- 1. Setup Environment and Data (Same as before) ---
grid_size = 30
patch_size = 4
microscopy_sample_rate = 5

sim_default = GroundTruth(grid_size, microscopy_sample_rate, patch_size)
sim_default.generate_fods(num_bins=360)

gt_data = sim_default.gt_grid
microscopy_data = sim_default.microscopy_grid
dwi_data = sim_default.dwi_grid

u_high_res = microscopy_data[:,:,0]
v_high_res = microscopy_data[:,:,1]

u_low_res = dwi_data[:,:,0]
v_low_res = dwi_data[:,:,1]

x = np.arange(0, grid_size, 1)
y = np.arange(0, grid_size, 1)
X, Y = np.meshgrid(x, y)

# --- 2. Create All Interpolators (Same as before) ---

# A. Incomplete High-Res Interpolator
interp_u_high = RegularGridInterpolator((y, x), u_high_res, method='nearest', bounds_error=False, fill_value=np.nan)
interp_v_high = RegularGridInterpolator((y, x), v_high_res, method='nearest', bounds_error=False, fill_value=np.nan)
bounds_high = (0, grid_size - 1, 0, grid_size - 1)

# B. Complete Low-Res Interpolator
coarse_x_coords = np.arange(patch_size / 2 - 0.5, grid_size, patch_size)
coarse_y_coords = np.arange(patch_size / 2 - 0.5, grid_size, patch_size)
interp_u_low = RegularGridInterpolator((coarse_y_coords, coarse_x_coords), u_low_res)
interp_v_low = RegularGridInterpolator((coarse_y_coords, coarse_x_coords), v_low_res)

# --- 3. Initialize and Run Multiple Hybrid Agents ---

# Configuration for multiple agents
NUM_AGENTS = 50          # How many paths to simulate
SEED_PIXEL = [15, 9]     # The (x, y) integer coordinate of the pixel to seed
PIXEL_SPREAD = 0.8       # How much of the pixel to cover (0.0 to 1.0)
STEP_SIZE = 0.1
NUM_STEPS = 500

agent_results = [] # To store (path, source_history) for each agent

print(f"Simulating {NUM_AGENTS} agents starting near pixel {SEED_PIXEL}...")

for i in range(NUM_AGENTS):
    # Generate random start point centered on the pixel (e.g., 10.0 to 11.0)
    # np.random.rand() gives [0, 1). We shift it to be centered or distributed within the pixel.
    rand_offset_x = (np.random.rand() - 0.5) * PIXEL_SPREAD
    rand_offset_y = (np.random.rand() - 0.5) * PIXEL_SPREAD
    
    # Assuming pixel [10, 3] effectively means center is [10.5, 3.5] or similar depending on your coord system.
    # Here I assume integer coordinates are the "grid lines", so a pixel is x to x+1.
    start_x = SEED_PIXEL[0] + 0.5 + rand_offset_x
    start_y = SEED_PIXEL[1] + 0.5 + rand_offset_y
    
    current_start_point = [start_x, start_y]

    # Create the agent
    smart_agent = HybridAgent(
        start_point_xy=current_start_point, 
        gt_object=sim_default,           
        interp_u_high=interp_u_high, 
        interp_v_high=interp_v_high, 
        grid_bounds=bounds_high
    )

    smart_agent.run_simulation(NUM_STEPS, STEP_SIZE)
    
    # Store the result
    path = np.array(smart_agent.get_path())
    sources = smart_agent.source_history
    agent_results.append({'path': path, 'sources': sources, 'start': current_start_point})

# --- 4. Plot the Results ---

fig, ax = plt.subplots(figsize=(10, 10))

# 1. Plot the high-res "ground truth" field background
# ax.quiver(X-0.5, Y-0.5, u_high_res, v_high_res, pivot='middle', 
#           color='gray', alpha=0.3, headwidth=2, headlength=5, scale=30)

sim_default.plot_microscopy(ax=ax, color='gray')

sim_default.plot_dwi_overlay(ax=ax)

# 2. Loop through results and plot each agent
for i, res in enumerate(agent_results):
    path_history = res['path']
    source_history = res['sources']
    
    # Skip if path is too short to form a segment
    if len(path_history) < 2:
        continue

    # Create segments: Shape (N_steps-1, 2, 2)
    segments = np.stack((path_history[:-1], path_history[1:]), axis=1)

    # Map source history to colors
    # Note: source_history length usually matches steps. Segments is len(path)-1.
    # We usually take colors for the starting point of the segment.
    current_colors = ['green' if s == 'high' else 'red' for s in source_history[:-1]]

    # Create collection
    lc = LineCollection(segments, colors=current_colors, linewidths=1.5, alpha=0.8)
    ax.add_collection(lc)
    
    # Optional: Plot the specific start point for this agent
    ax.plot(res['start'][0], res['start'][1], 'bo', markersize=2, alpha=0.6)


# 3. Plot Dummy Lines for Legend (Only need to do this once)
ax.plot([], [], color='green', linewidth=2.5, label='High-Res Microscopy')
ax.plot([], [], color='red', linewidth=2.5, label='Low-Res DWI')
ax.plot([], [], 'bo', markersize=3, label='Start Points')

# 4. Formatting
ax.set_xticks(np.arange(0, grid_size, 1))
ax.set_yticks(np.arange(0, grid_size, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim([-1, grid_size-1])
ax.set_ylim([-1, grid_size-1])
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.title(f'Multi-Agent Simulation (N={NUM_AGENTS})')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()