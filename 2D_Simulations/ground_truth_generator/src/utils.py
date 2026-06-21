# src/utils.py
import numpy as np
import os

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalizes a field of vectors."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms + 1e-8, where=(norms != 0))

def save_grid(grid: np.ndarray, directory: str, filename: str):
    """Saves the NumPy grid to a file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    np.save(filepath, grid)
    print(f"Grid saved successfully to {filepath}")