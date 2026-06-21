import numpy as np

class HybridAgent:
    """
    An agent that uses the best available information.
    1. Tries 'interp_high' (Microscopy) first.
    2. If high-res returns nan, falls back to 'gt_object' (Probabilistic FOD).
    """
    def __init__(self, start_point_xy, 
                 gt_object, 
                 interp_u_high, interp_v_high, 
                 grid_bounds):
        
        self.pos = np.array(start_point_xy, dtype=float)
        
        # Store high-res interpolators
        self.interp_u_high = interp_u_high
        self.interp_v_high = interp_v_high
        
        # Store the GroundTruth object (replaces the low-res interpolators)
        # We need this to access .p_patch_size and .sample_fod_direction()
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
        
        # --- Decision Logic ---
        try:
            point_yx = np.array([y, x])
            
            # 1. Try to get high-res data
            u = self.interp_u_high(point_yx)[0]
            
            # 2. Check if high-res failed (is nan)
            if np.isnan(u):
                # --- PROBABILISTIC LOGIC ---
                
                # A. Calculate which DWI patch we are in
                # (Continuous position -> Grid Index)
                dwi_x_idx = int(x // self.gt.p_patch_size)
                dwi_y_idx = int(y // self.gt.p_patch_size)
                
                # B. Safety check: Ensure indices are within the FOD grid
                max_h, max_w, _ = self.gt.fod_grid.shape
                
                if 0 <= dwi_x_idx < max_w and 0 <= dwi_y_idx < max_h:
                    # C. Sample a direction from the FOD probability distribution
                    angle = self.gt.sample_fod_direction(dwi_x_idx, dwi_y_idx)
                    
                    # D. Convert angle to vector
                    u = np.cos(angle)
                    v = np.sin(angle)
                    self.source_history.append('fod') # Log that we used FOD
                else:
                    # We are technically inside global bounds but outside patch bounds 
                    # (rare, usually due to rounding edges). Stop agent.
                    self.is_active = False
                    return

            else:
                # 3. High-res succeeded
                v = self.interp_v_high(point_yx)[0]
                
                # Optional: Normalize high-res vector if your data isn't pre-normalized
                mag = np.sqrt(u**2 + v**2)
                if mag > 1e-9:
                    u, v = u/mag, v/mag
                    
                self.source_history.append('high')
                
        except (ValueError, IndexError):
            self.is_active = False
            return
        # --- End of Logic ---

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