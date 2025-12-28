# src/noise.py
import numpy as np
import noise as pnoise 
from src.utils import normalize_vectors

def apply_random_noise(grid: np.ndarray, level: float) -> np.ndarray:
    """Adds random Gaussian noise to a vector field."""
    noise_vectors = np.random.normal(0, 1, grid.shape)
    noisy_grid = grid + noise_vectors * level
    return normalize_vectors(noisy_grid)

def _compute_simplex_noise(shape, scale, octaves, seed):
    """Simplex noise generation."""
    noise_field = np.zeros(shape + (3,))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x, y, z = (i / shape[0]) * scale, (j / shape[1]) * scale, (k / shape[2]) * scale
                noise_field[i, j, k, 0] = pnoise.snoise4(x, y, z, w=seed, octaves=octaves)
                noise_field[i, j, k, 1] = pnoise.snoise4(x, y, z, w=seed + 10, octaves=octaves)
                noise_field[i, j, k, 2] = pnoise.snoise4(x, y, z, w=seed + 20, octaves=octaves)
    return noise_field

def apply_simplex_noise(grid: np.ndarray, strength: float, scale: float, octaves: int) -> np.ndarray:
    """Applies 3D Simplex noise to a vector field."""
    seed = np.random.randint(0, 100)
    noise_field = _compute_simplex_noise(grid.shape[:3], scale, octaves, seed)
    perturbed_grid = grid + noise_field * strength
    return normalize_vectors(perturbed_grid)