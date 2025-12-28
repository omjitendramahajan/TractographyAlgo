# src/patterns.py
import numpy as np
from abc import ABC, abstractmethod
from src.utils import normalize_vectors

class FiberPattern(ABC):
    """Abstract base class for all fiber pattern generators."""
    def __init__(self, shape: tuple, **kwargs):
        self.shape = shape
    
    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generates the vector field for the pattern."""
        pass

class StraightFibers(FiberPattern):
    def __init__(self, shape: tuple, direction: list):
        super().__init__(shape)
        self.direction = np.array(direction)

    def generate(self) -> np.ndarray:
        grid = np.zeros(self.shape + (3,))
        grid[:] = normalize_vectors(self.direction)
        return grid

class BendingFibers(FiberPattern):
    def generate(self) -> np.ndarray:
        grid = np.zeros(self.shape + (3,))
        nx, _, _ = self.shape
        x = np.arange(nx)
        angle = (np.pi / 2.0) * (x / (nx - 1))
        vx = np.sin(angle)
        vy = np.cos(angle)
        grid[..., 0] = vx[:, np.newaxis, np.newaxis]
        grid[..., 1] = vy[:, np.newaxis, np.newaxis]
        return grid

class FanningFibers(FiberPattern):
    def generate(self) -> np.ndarray:
        grid = np.zeros(self.shape + (3,))
        _, ny, nz = self.shape
        center_y, center_z = ny // 2, nz // 2
        y, z = np.arange(ny), np.arange(nz)
        dy, dz = y - center_y, z - center_z
        angle = np.arctan2(dz[np.newaxis, :], dy[:, np.newaxis])
        grid[..., 0] = 1.0
        grid[..., 1] = 0.4 * np.cos(angle)[np.newaxis, :, :]
        grid[..., 2] = 0.4 * np.sin(angle)[np.newaxis, :, :]
        return normalize_vectors(grid)