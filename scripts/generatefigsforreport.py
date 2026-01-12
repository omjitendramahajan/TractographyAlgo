import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

# Import your custom classes
from DataGenerator import GroundTruth
from Agent import HybridAgent

# ==========================================
# Configuration
# ==========================================
GRID_SIZE = 30
PATCH_SIZE = 4       # Size of the "Low Res" DWI voxels
SAMPLE_RATE = 5      # Spacing of the "High Res" Microscopy strips

# Initialize Simulation
sim = GroundTruth(GRID_SIZE, SAMPLE_RATE, PATCH_SIZE)
sim.generate_fods(num_bins=360) # Critical: Generate the probability distributions

# Prepare Interpolators (Used for Figures 2 & 3)
x = np.arange(0, GRID_SIZE, 1)
y = np.arange(0, GRID_SIZE, 1)
u_high = sim.microscopy_grid[:,:,0]
v_high = sim.microscopy_grid[:,:,1]

# Real Interpolators (for Hybrid Mode)
interp_u = RegularGridInterpolator((y, x), u_high, method='nearest', bounds_error=False, fill_value=np.nan)
interp_v = RegularGridInterpolator((y, x), v_high, method='nearest', bounds_error=False, fill_value=np.nan)

# Empty Interpolators (for Standard Mode Simulation)
nan_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
interp_nan = RegularGridInterpolator((y, x), nan_grid, bounds_error=False, fill_value=np.nan)

bounds = (0, GRID_SIZE, 0, GRID_SIZE)

# ==========================================
# Figure 1: The Data Environment
# ==========================================
def plot_figure_1():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Background: Ground Truth (Gray)
    X, Y = np.meshgrid(x, y)
    ax.quiver(X, Y, sim.gt_grid[:,:,0], sim.gt_grid[:,:,1], 
              color='gray', alpha=0.2, scale=30, label='Ground Truth')
    
    # 2. Layer: Sparse Microscopy (Green)
    # Mask to select only valid rows
    mask_micro = ~np.isnan(u_high)
    ax.quiver(X[mask_micro], Y[mask_micro], u_high[mask_micro], v_high[mask_micro], 
              color='green', scale=30, width=0.005, label='Microscopy Strips')
    
    # 3. Layer: Coarse DWI (Red)
    # Calculate centers of the large patches
    dwi_h, dwi_w, _ = sim.dwi_grid.shape
    dwi_x_coords = np.arange(0, dwi_w) * PATCH_SIZE + PATCH_SIZE/2 - 0.5
    dwi_y_coords = np.arange(0, dwi_h) * PATCH_SIZE + PATCH_SIZE/2 - 0.5
    DWI_X, DWI_Y = np.meshgrid(dwi_x_coords, dwi_y_coords)
    
    ax.quiver(DWI_X, DWI_Y, sim.dwi_grid[:,:,0], sim.dwi_grid[:,:,1], 
              color='red', scale=15, width=0.008, label='Coarse DWI')
    
    # Grid formatting
    major_ticks = np.arange(0, GRID_SIZE + 1, PATCH_SIZE)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='major', color='red', linestyle='--', alpha=0.3)
    #ax.set_title("Figure 1: Multiscale Data Environment")
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1, label='Ground Truth'),
        Line2D([0], [0], color='green', lw=2, label='High-Res Microscopy'),
        Line2D([0], [0], color='red', lw=2, label='Low-Res DWI')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

# ==========================================
# Figure 2: Hybrid Tracking Behavior
# ==========================================
def plot_figure_2():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Background
    X, Y = np.meshgrid(x, y)
    ax.quiver(X, Y, sim.gt_grid[:,:,0], sim.gt_grid[:,:,1], color='gray', alpha=0.1)
    
    # Setup Agents
    start_xs = np.linspace(1, 10, 8)
    start_ys = np.full(8, 2.0)
    
    for i in range(len(start_xs)):
        start_pt = (start_xs[i], start_ys[i])
        agent = HybridAgent(start_pt, sim, interp_u, interp_v, bounds)
        agent.run_simulation(num_steps=1000, step_size=0.1)
        path = agent.get_path()
        sources = agent.source_history
        
        if len(path) < 2: continue
        
        # Color segments based on source
        segments = np.stack((path[:-1], path[1:]), axis=1)
        # Match colors to steps. 'high' = Green, 'fod' = Red
        colors = ['green' if s == 'high' else 'red' for s in sources[:len(segments)]]
            
        lc = LineCollection(segments, colors=colors, linewidths=2.5, alpha=0.9)
        ax.add_collection(lc)
        ax.plot(start_pt[0], start_pt[1], 'bo', markersize=3)
        
    # Visual cues for microscopy strips
    for r in range(0, GRID_SIZE, SAMPLE_RATE):
        ax.axhspan(r - 0.5, r + 0.5, color='green', alpha=0.1)
        
    #ax.set_title("Figure 2: Hybrid Agent Switching Modalities")
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, label='Microscopy Guided'),
        Line2D([0], [0], color='red', lw=3, label='DWI/Probabilistic Guided'),
        Line2D([0], [0], color='green', alpha=0.2, lw=10, label='Microscopy Available Region')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

# ==========================================
# Figure 3: Performance Comparison
# ==========================================
def plot_figure_3():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    start_xs = np.linspace(2, 12, 10)
    start_ys = np.full(10, 2.0)

    # --- Panel A: Standard Tracking (No Microscopy) ---
    ax = axes[0]
    #ax.set_title("Standard Probabilistic Tracking (DWI Only)")
    ax.quiver(sim.X_gt, sim.Y_gt, sim.gt_grid[:,:,0], sim.gt_grid[:,:,1], color='gray', alpha=0.1)
    
    for i in range(len(start_xs)):
        start_pt = (start_xs[i], start_ys[i])
        # Pass "interp_nan" to simulate lack of microscopy data
        agent = HybridAgent(start_pt, sim, interp_nan, interp_nan, bounds)
        agent.run_simulation(num_steps=120, step_size=0.4)
        path = agent.get_path()
        if len(path) > 1:
            ax.plot(path[:,0], path[:,1], color='red', alpha=0.6, linewidth=2)
            ax.plot(start_pt[0], start_pt[1], 'bo', markersize=3)

    # --- Panel B: Hybrid Tracking ---
    ax = axes[1]
    #ax.set_title("Hybrid Tracking (Microscopy + DWI)")
    ax.quiver(sim.X_gt, sim.Y_gt, sim.gt_grid[:,:,0], sim.gt_grid[:,:,1], color='gray', alpha=0.1)
    
    for i in range(len(start_xs)):
        start_pt = (start_xs[i], start_ys[i])
        # Pass real interpolators
        agent = HybridAgent(start_pt, sim, interp_u, interp_v, bounds)
        agent.run_simulation(num_steps=120, step_size=0.4)
        path = agent.get_path()
        
        if len(path) < 2: continue
        
        segments = np.stack((path[:-1], path[1:]), axis=1)
        sources = agent.source_history
        colors = ['green' if s == 'high' else 'red' for s in sources[:len(segments)]]
        
        lc = LineCollection(segments, colors=colors, linewidths=2.5, alpha=0.9)
        ax.add_collection(lc)
        ax.plot(start_pt[0], start_pt[1], 'bo', markersize=3)
    
    # Highlight strips
    for r in range(0, GRID_SIZE, SAMPLE_RATE):
        ax.axhspan(r - 0.5, r + 0.5, color='green', alpha=0.1)

    for ax in axes:
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        
    plt.show()

# ==========================================
# Run All
# ==========================================
if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()