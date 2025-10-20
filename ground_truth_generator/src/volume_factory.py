# src/volume_factory.py
import numpy as np
import random
from src import patterns, noise

class VolumeFactory:
    """Builds a composite volume based on a configuration dictionary."""
    
    PATTERN_MAP = {
        "straight": patterns.StraightFibers,
        "bending": patterns.BendingFibers,
        "fanning": patterns.FanningFibers,
    }

    def __init__(self, config: dict):
        self.config = config
        self.grid_params = config['grid_params']
        self.master_shape = tuple(self.grid_params['master_shape'])
        self.block_size = self.grid_params['block_size']
        self.sub_shape = (self.block_size,) * 3

    def _create_pattern_instance(self, pattern_config: dict) -> np.ndarray:
        """Creates a single pattern block, applying noise if specified."""
        pattern_type = pattern_config['type']
        params = pattern_config.get('params', {})
        
        # Instantiate the correct pattern class
        PatternClass = self.PATTERN_MAP[pattern_type]
        pattern_obj = PatternClass(self.sub_shape, **params)
        
        # Generate the clean grid
        grid = pattern_obj.generate()
        
        # Apply noise if specified in the config
        if 'noise' in pattern_config:
            noise_config = pattern_config['noise']
            if noise_config['type'] == 'random':
                grid = noise.apply_random_noise(grid, level=noise_config['level'])
        
        return grid

    def create_volume(self) -> np.ndarray:
        """Generates the final master_grid array."""
        # Initialize withrandom background
        background = np.random.normal(0, 1, self.master_shape + (3,))
        master_grid = noise.normalize_vectors(background)

        # Create pool of generated pattern blocks from the config
        pattern_pool = [self._create_pattern_instance(p_conf) for p_conf in self.config['pattern_pool']]
        print(f"Create a pool of {len(pattern_pool)} pattern blocks.")

        # tile the master grid with blocks from the pool
        print(f"Tiling the {self.master_shape} grid...")
        prob = self.grid_params['tiling_probability']
        for x in range(0, self.master_shape[0], self.block_size):
            for y in range(0, self.master_shape[1], self.block_size):
                for z in range(0, self.master_shape[2], self.block_size):
                    if random.random() < prob:
                        block = random.choice(pattern_pool)
                        master_grid[x:x+self.block_size, y:y+self.block_size, z:z+self.block_size] = block
        
        # Apply final, smooth noise to the entire volume
        final_noise_conf = self.config['final_noise']
        if final_noise_conf['type'] == 'simplex':
            print("Applying final smooth Simplex noise")

            simplex_params = final_noise_conf.copy()
            simplex_params.pop('type', None)

            master_grid = noise.apply_simplex_noise(master_grid, **simplex_params)
            
        print(" Volume generated")
        return master_grid