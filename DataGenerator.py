import numpy as np
import matplotlib.pyplot as plt

class GroundTruth:
    """
    Generates and visualizes a ground truth vector field and its
    downsampled "microscopy" and "DWI" representations.
    
    The ground truth field can be customized by passing a
    generator function.
    """
    
    def __init__(self, size=40, microscopy_sample_rate=4, dwi_patch_size=2, gt_generator_func=None):
        """
        Initializes and generates all data grids.
        
        Args:
            size (int): The width and height of the master grid.
            microscopy_sample_rate (int): The 'N' value for row sampling.
            dwi_patch_size (int): The 'P' value for patch-based averaging.
            gt_generator_func (callable, optional): A function that
                generates the ground truth grid. If None, a default
                'saddle' field is used. The function must accept
                (X, Y, size) as arguments.
        """
        self.size = size
        self.n_sample_rate = microscopy_sample_rate
        self.p_patch_size = dwi_patch_size
        self.epsilon = 1e-10

        # --- Generate Coordinates ---
        x = np.arange(0, self.size, 1)
        y = np.arange(0, self.size, 1)
        self.X_gt, self.Y_gt = np.meshgrid(x, y)
        
        coarse_x = np.arange(0, self.size, self.p_patch_size) + self.p_patch_size / 2
        coarse_y = np.arange(0, self.size, self.p_patch_size) + self.p_patch_size / 2
        self.X_dwi, self.Y_dwi = np.meshgrid(coarse_x, coarse_y)

        # --- Generate Data Grids ---
        
        # Select the ground truth generator
        if gt_generator_func is None:
            self.generator_to_use = self._default_saddle_field
        else:
            self.generator_to_use = gt_generator_func
            
        # Generate the ground truth grid
        self.gt_grid = self.generator_to_use(self.X_gt, self.Y_gt, self.size)
        
        # Generate other grids based on the ground truth
        self.microscopy_grid = self._create_microscopy_data()
        self.dwi_grid = self._create_dwi_mean_data()

        # Initialize FOD storage as None
        self.fod_grid = None
        self.fod_bins = None
        
        # FOD Placeholders
        self.fod_grid = None
        self.num_bins = None
        self.bin_edges = None
        self.bin_centers = None
        self.bin_width = None

    # --- Data Generation Methods ---
    
    def _default_saddle_field(self, X, Y, size):
        """Generates the default 'saddle' vector field."""
        center_x, center_y = size / 2, size / 2
        
        u_saddle = X - center_x  # Points out along x-axis
        v_saddle = center_y - Y  # Points in along y-axis
        
        magnitude = np.sqrt(u_saddle**2 + v_saddle**2) + self.epsilon
        u_normalized = u_saddle / magnitude
        v_normalized = v_saddle / magnitude
        
        return np.stack((u_normalized, v_normalized), axis=-1)

    def _create_microscopy_data(self):
        """Simulates microscopy by taking every Nth row."""
        microscopy_data = np.full(self.gt_grid.shape, np.nan)
        for i in range(0, self.size, self.n_sample_rate):
            microscopy_data[i, :, :] = self.gt_grid[i, :, :]
        return microscopy_data

    def _create_dwi_mean_data(self):
        """Standard DWI: Single average vector per patch."""
        grid_h, grid_w, _ = self.gt_grid.shape
        new_h = int(np.ceil(grid_h / self.p_patch_size))
        new_w = int(np.ceil(grid_w / self.p_patch_size))
        resultant_grid = np.zeros((new_h, new_w, 2))

        for i in range(new_h):
            for j in range(new_w):
                patch = self._get_patch(i, j)
                res_vec = np.sum(patch, axis=(0, 1))
                mag = np.linalg.norm(res_vec) + self.epsilon
                resultant_grid[i, j] = res_vec / mag
        return resultant_grid
    
    # --- FOD Methods ---
    def generate_fods(self, num_bins=360):
        """
        Calculates the Fiber Orientation Distribution (histogram) for every patch.
        Call this before using sample_fod_direction().
        """
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-np.pi, np.pi, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        
        grid_h, grid_w, _ = self.gt_grid.shape
        new_h = int(np.ceil(grid_h / self.p_patch_size))
        new_w = int(np.ceil(grid_w / self.p_patch_size))
        
        # Grid shape: [DWI_Y, DWI_X, Bins]
        self.fod_grid = np.zeros((new_h, new_w, self.num_bins))

        for i in range(new_h):
            for j in range(new_w):
                patch = self._get_patch(i, j)
                u = patch[:, :, 0].flatten()
                v = patch[:, :, 1].flatten()
                
                # Calculate angles for all vectors in patch
                angles = np.arctan2(v, u)
                
                # Create histogram
                hist, _ = np.histogram(angles, bins=self.bin_edges)
                
                # Normalize to probability mass function (sum = 1)
                total = np.sum(hist)
                if total > 0:
                    self.fod_grid[i, j, :] = hist / total
                else:
                    # Uniform distribution if patch is empty/zero
                    self.fod_grid[i, j, :] = 1.0 / self.num_bins
        
        print(f"FOD Generation Complete. Grid shape: {self.fod_grid.shape}")

    def _get_patch(self, i, j):
        """Helper to slice the GT grid based on DWI indices."""
        y_start = i * self.p_patch_size
        y_end = min(y_start + self.p_patch_size, self.size)
        x_start = j * self.p_patch_size
        x_end = min(x_start + self.p_patch_size, self.size)
        return self.gt_grid[y_start:y_end, x_start:x_end]

    def sample_fod_direction(self, dwi_x, dwi_y):
        """
        Samples a propagation angle from the FOD at the specific voxel.
        Used by tracking algorithms.
        """
        if self.fod_grid is None:
            raise ValueError("FODs not generated! Call object.generate_fods() first.")
            
        # Get Probability Mass Function for this voxel
        probs = self.fod_grid[dwi_y, dwi_x, :]
        
        # 1. Select a bin index based on weights
        bin_idx = np.random.choice(len(self.bin_centers), p=probs)
        
        # 2. Jitter within the bin (Continuous sampling)
        # Without this, streamlines would be 'blocky'
        angle_center = self.bin_centers[bin_idx]
        jitter = np.random.uniform(-self.bin_width/2, self.bin_width/2)
        
        return angle_center + jitter

    # --- Plotting Helper Methods ---
    
    def _setup_base_plot(self, ax, title, size=None):
        if size is None:
            size = self.size
        ax.set_xticks(np.arange(0, size, 1))
        ax.set_yticks(np.arange(0, size, 1))
        ax.set_xlim([0, size])
        ax.set_ylim([0, size])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.set_title(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # ---  Plotting Methods ---

    def _setup_base_plot(self, ax, title):
        """Helper to format the axis standardly."""
        ax.set_xticks(np.arange(0, self.size, 1))
        ax.set_yticks(np.arange(0, self.size, 1))
        ax.set_xlim([0, self.size])
        ax.set_ylim([0, self.size])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.set_title(title)
        # Hide ticks but keep grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot_ground_truth(self, ax=None, figsize=(8, 8), color='g'):
        """
        Plots the full resolution ground truth.
        Returns: ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            
        u_full = self.gt_grid[:, :, 0]
        v_full = self.gt_grid[:, :, 1]
        
        ax.quiver(self.X_gt + 0.5, self.Y_gt + 0.5, u_full, v_full, 
                  pivot='middle', headwidth=3, headlength=5, color=color)
        self._setup_base_plot(ax, 'Complete Ground Truth')
        return ax

    def plot_microscopy(self, ax=None, figsize=(8, 8), color='b'):
        """
        Plots the incomplete microscopy data.
        Returns: ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        u_sampled = self.microscopy_grid[:, :, 0]
        v_sampled = self.microscopy_grid[:, :, 1]
        
        ax.quiver(self.X_gt + 0.5, self.Y_gt + 0.5, u_sampled, v_sampled, 
                  pivot='middle', headwidth=3, headlength=5, color=color)
        self._setup_base_plot(ax, f'Microscope Data (Every {self.n_sample_rate}th Row)')
        return ax

    def plot_dwi_overlay(self, ax=None, figsize=(10, 10), gt_color='gray', dwi_color='r'):
        """
        Plots DWI vectors on top of faded ground truth.
        Returns: ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        u_fine = self.gt_grid[:, :, 0]
        v_fine = self.gt_grid[:, :, 1]
        u_coarse = self.dwi_grid[:, :, 0]
        v_coarse = self.dwi_grid[:, :, 1]
        
        # Background GT
        # ax.quiver(self.X_gt + 0.5, self.Y_gt + 0.5, u_fine, v_fine, 
        #           pivot='middle', color=gt_color, alpha=0.5, headwidth=2, headlength=4) 
        
        # Foreground DWI
        ax.quiver(self.X_dwi, self.Y_dwi, u_coarse, v_coarse, 
                  pivot='middle', color=dwi_color, headwidth=3, headlength=5)
        
        ax.set_title(f'DWI (Red) on GT (Gray) - {self.p_patch_size}x{self.p_patch_size} Patches')
        
        # Custom grid for DWI patches
        major_ticks = np.arange(0, self.size + 1, self.p_patch_size)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        minor_ticks = np.arange(0, self.size + 1, 1)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        
        ax.set_xlim([0, self.size])
        ax.set_ylim([0, self.size])
        
        ax.grid(which='major', color='black', linestyle='-', linewidth=1.0, alpha=0.7)
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        return ax