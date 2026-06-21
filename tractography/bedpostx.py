"""
BedpostxData - Load and access BEDPOSTX diffusion MRI data.

This module handles loading BEDPOSTX output files and provides
methods for accessing fiber orientations and probabilistic sampling.

Uses fslpy (fsl.data.image.Image) for NIfTI file loading.
"""

import logging
import math
import os

import numpy as np
from fsl.data.image import Image
from fsl.utils.path import PathError

logger = logging.getLogger(__name__)


class BedpostxData:
    """
    Loads and provides access to BEDPOSTX output files.
    Supports probabilistic sampling from posterior samples.
    """

    __slots__ = (
        'bedpostx_dir', 'num_fibers', 'use_memory_map',
        'mask_img', 'mask', 'affine', 'shape', 'voxel_size',
        'f_samples', 'dyads',
        'th_samples', 'ph_samples', 'f_all_samples', 'n_samples',
        'f_all_array', 'volume_img', '_f_scratch',
    )

    def __init__(self, bedpostx_dir, num_fibers=1, load_samples=True, load_dyads=False,
                 use_memory_map=True):
        """
        Load BEDPOSTX data.
        
        Args:
            bedpostx_dir: Path to .bedpostX directory
            num_fibers: Number of fiber populations to load (1, 2, or 3)
            load_samples: If True, load posterior samples for probabilistic tracking
            load_dyads: If True, load mean fiber orientations (dyads) for deterministic tracking.
                        Set to False if only using probabilistic tracking
            use_memory_map: If True, use lazy loading (fslpy default behavior).
                           Set to False to preload full arrays into RAM.
        """
        self.bedpostx_dir = bedpostx_dir
        self.num_fibers = num_fibers
        self.use_memory_map = use_memory_map

        # Load brain mask (reference for shape and affine)
        mask_path = self._resolve_path('nodif_brain_mask')
        self.mask_img = Image(mask_path)

        self.mask = np.asarray(self.mask_img[:]) > 0  # Bool mask: True if voxel is in brain
        self.affine = self.mask_img.voxToWorldMat  # 4x4 voxel-to-world transform matrix
        self.shape = self.mask_img.shape[:3]  # (X, Y, Z) - sagittal, coronal, axial

        # Compute voxel sizes from affine (in mm)
        self.voxel_size = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))

        # Initialise attributes that may not be populated depending on flags
        self.f_samples = []
        self.dyads = []
        self.th_samples = []
        self.ph_samples = []
        self.f_all_samples = []
        self.f_all_array = None
        self.n_samples = 0
        self._f_scratch = None

        if load_dyads:
            self._load_dyads()
        else:
            logger.info("Skipping dyad loading (load_dyads=False)")

        if load_samples:
            self._load_posterior_samples()
        else:
            logger.info("Skipping posterior samples loading (load_samples=False)")

        # Store volume image for coordinate transforms
        self.volume_img = self.mask_img

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename_stem):
        """Resolve a NIfTI file path, checking .nii then .nii.gz.

        Args:
            filename_stem: Filename without extension (e.g. 'nodif_brain_mask')

        Returns:
            Full path to the existing file.

        Raises:
            FileNotFoundError: If neither .nii nor .nii.gz exists.
        """
        for ext in ('.nii', '.nii.gz'):
            path = os.path.join(self.bedpostx_dir, f"{filename_stem}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Cannot find {filename_stem}[.nii|.nii.gz] in {self.bedpostx_dir}"
        )

    @staticmethod
    def _to_voxel_idx(pos):
        """Convert float position to integer voxel indices.

        Args:
            pos: (x, y, z) position (can be float)

        Returns:
            Tuple of (i, j, k) integer indices.
        """
        return int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))

    def _in_bounds(self, i, j, k):
        """Check if voxel indices are within image bounds."""
        return (0 <= i < self.shape[0] and
                0 <= j < self.shape[1] and
                0 <= k < self.shape[2])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_dyads(self):
        """Load mean fiber orientations (dyads) for deterministic tracking."""
        self.dyads = []      # List of 4D arrays (X, Y, Z, 3)
        self.f_samples = []  # List of 3D arrays (mean volume fractions)

        logger.info("Loading mean fiber orientations (dyads)...")
        for i in range(1, self.num_fibers + 1):
            try:
                dyad_path = self._resolve_path(f'dyads{i}')
                f_path = self._resolve_path(f'mean_f{i}samples')

                dyad_img = Image(dyad_path)
                self.dyads.append(np.asarray(dyad_img[:]))

                f_img = Image(f_path)
                self.f_samples.append(np.asarray(f_img[:]))
                logger.info("  Loaded fiber population %d", i)
            except Exception as e:
                logger.warning("  Could not find fiber population %d: %s", i, e)
                break

        self.num_fibers = len(self.dyads)

        # Apply the world-frame X correction to the loaded dyad arrays (see the
        # handedness note in __init__). This covers both get_mean_dyads() and
        # get_orientation(), which read these arrays directly. Flip in place
        # when the array is writable to avoid transiently doubling the (large)
        # dyad memory; copy only if fslpy handed back a read-only buffer.
        for idx, arr in enumerate(self.dyads):
            if not arr.flags.writeable:
                arr = arr.copy()
                self.dyads[idx] = arr
            arr[..., 0] *= -1

    def _load_posterior_samples(self):
        """Load posterior samples for probabilistic tracking.
        
        If use_memory_map=True, stores fslpy Image objects (data loaded lazily).
        Otherwise, preloads full arrays into RAM and pre-stacks volume fractions
        for faster hot-path access.
        """
        # Initialize sample lists
        self.th_samples = []  # Theta angles
        self.ph_samples = []  # Phi angles
        self.f_all_samples = []  # Volume fractions
        self.n_samples = 0  # Number of MCMC samples per voxel

        mode_str = "lazy/memory-mapped" if self.use_memory_map else "full load"
        logger.info("Loading posterior samples (%s)...", mode_str)

        for i in range(1, self.num_fibers + 1):
            try:
                th_path = self._resolve_path(f'merged_th{i}samples')
                ph_path = self._resolve_path(f'merged_ph{i}samples')
                f_path = self._resolve_path(f'merged_f{i}samples')
            except FileNotFoundError:
                raise ValueError(
                    f"Requested num_fibers={self.num_fibers}, but could not find "
                    f"merged samples for fiber {i}. Only {i-1} fiber population(s) "
                    f"exist in '{self.bedpostx_dir}'. Please set num_fibers={i-1} or lower."
                )

            try:
                th_img = Image(th_path)
                ph_img = Image(ph_path)
                f_img = Image(f_path)

                if self.use_memory_map:
                    # Store Image objects - data loaded lazily on access
                    self.th_samples.append(th_img)
                    self.ph_samples.append(ph_img)
                    self.f_all_samples.append(f_img)
                else:
                    # Preload full arrays into RAM
                    self.th_samples.append(np.asarray(th_img[:]))
                    self.ph_samples.append(np.asarray(ph_img[:]))
                    self.f_all_samples.append(np.asarray(f_img[:]))

                # Get number of samples (4th dimension)
                self.n_samples = th_img.shape[3] if len(th_img.shape) > 3 else 1
                logger.info("  Fiber %d: %d samples ready", i, self.n_samples)

            except (FileNotFoundError, PathError):
                raise ValueError(
                    f"Requested num_fibers={self.num_fibers}, but could not find "
                    f"merged samples for fiber {i}. Only {i-1} fiber population(s) "
                    f"exist in '{self.bedpostx_dir}'. Please set num_fibers={i-1} or lower."
                )

        if len(self.th_samples) == 0:
            raise ValueError(
                f"No posterior samples found in '{self.bedpostx_dir}'. "
                "Cannot perform probabilistic tractography."
            )

        # Pre-allocate scratch buffer for the hot path (avoids allocation per call)
        n_fibers = len(self.f_all_samples)
        self._f_scratch = np.empty((n_fibers, self.n_samples))

        # For non-memory-mapped mode, pre-stack f arrays for faster indexing
        if not self.use_memory_map:
            self.f_all_array = np.stack(self.f_all_samples, axis=0)

    # ------------------------------------------------------------------
    # Sampling / orientation access
    # ------------------------------------------------------------------

    def sample_orientation_probabilistic(self, voxel):
        """
        Probabilistically sample an orientation from BEDPOSTX posterior.
        
        Optimized to perform 'Sample First, Calculate Later', reducing 
        trigonometry overhead by only processing the winning sample.
        
        Args:
            voxel: (i, j, k) voxel coordinates
            
        Returns:
            Normalized 3D orientation vector, or None if invalid
        """
        i, j, k = self._to_voxel_idx(voxel)

        if not self._in_bounds(i, j, k):
            return None

        # 1. Gather all weights (f) into pre-allocated scratch buffer
        try:
            if self.f_all_array is not None:
                # Non-memory-mapped: single indexed slice into scratch buffer
                np.copyto(self._f_scratch, self.f_all_array[:, i, j, k, :])
            else:
                # Memory-mapped: fill pre-allocated buffer (avoids np.stack alloc)
                for fi, f in enumerate(self.f_all_samples):
                    self._f_scratch[fi, :] = f[i, j, k, :]
        except IndexError:
            return None

        # 2. Filter low volume fractions by zeroing their probability
        self._f_scratch[self._f_scratch < 0.01] = 0
        total_weight = self._f_scratch.sum()

        if total_weight == 0:
            return None

        # 3. Pick a winner via CDF + uniform draw (faster than np.random.choice)
        flat = self._f_scratch.ravel()
        flat *= (1.0 / total_weight)  # normalise in-place
        cumsum = np.cumsum(flat)
        winner_global_idx = np.searchsorted(cumsum, np.random.random())

        # Clamp to valid range (searchsorted can return len when random() ≈ 1.0)
        if winner_global_idx >= len(flat):
            winner_global_idx = len(flat) - 1

        #    Map flat index back to (fiber_idx, sample_idx)
        n_samples = self._f_scratch.shape[1]
        fiber_idx = winner_global_idx // n_samples
        sample_idx = winner_global_idx % n_samples

        # 4. Retrieve ONLY the winner — minimal IO + trig
        th = float(self.th_samples[fiber_idx][i, j, k, sample_idx])
        ph = float(self.ph_samples[fiber_idx][i, j, k, sample_idx])

        # Spherical to Cartesian (already unit-length by construction).
        # X negated for world-frame consistency (see handedness note in __init__).
        sin_th = math.sin(th)
        return np.array([-sin_th * math.cos(ph),
                         sin_th * math.sin(ph), math.cos(th)])

    def get_mean_dyads(self, voxel):
        """
        Return the mean dyad (3D unit vector) for each fibre population at
        a voxel, read directly from the BEDPOSTX ``dyads{i}.nii.gz`` files
        (i.e. the precomputed mean orientations).

        Requires the BedpostxData object to have been constructed with
        ``load_dyads=True``.

        Args:
            voxel: (i, j, k) voxel coordinates.

        Returns:
            A list of 3D unit vectors — one per fibre population — or None
            if the voxel is out of bounds. Populations whose dyad has zero
            magnitude return ``np.zeros(3)`` in their slot.
        """
        if not self.dyads:
            raise RuntimeError(
                "get_mean_dyads requires dyads to be loaded. "
                "Construct BedpostxData with load_dyads=True."
            )

        i, j, k = self._to_voxel_idx(voxel)

        if not self._in_bounds(i, j, k):
            return None

        mean_dyads = []
        for fib_idx in range(len(self.dyads)):
            v = self.dyads[fib_idx][i, j, k, :]
            norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            mean_dyads.append(v / norm if norm > 0 else np.zeros(3))

        return mean_dyads

    def sample_from_population(self, voxel, pop_idx, sample_idx=None):
        """
        Draw a single 3D orientation from one fibre population's posterior.

        Args:
            voxel: (i, j, k) voxel coordinates.
            pop_idx: Index of the fibre population to sample from.
            sample_idx: Posterior sample index. If None, a uniform random
                sample is drawn.

        Returns:
            Normalised 3D orientation vector, or None if invalid.
        """
        i, j, k = self._to_voxel_idx(voxel)

        if not self._in_bounds(i, j, k):
            return None

        if pop_idx < 0 or pop_idx >= len(self.th_samples):
            return None

        if sample_idx is None:
            sample_idx = np.random.randint(0, self.n_samples)

        th = float(self.th_samples[pop_idx][i, j, k, sample_idx])
        ph = float(self.ph_samples[pop_idx][i, j, k, sample_idx])

        # Spherical to Cartesian (already unit-length by construction).
        # X negated for world-frame consistency (see handedness note in __init__).
        sin_th = math.sin(th)
        return np.array([-sin_th * math.cos(ph),
                         sin_th * math.sin(ph), math.cos(th)])

    def get_orientation(self, voxel, fiber_idx=0):
        """Get the mean orientation vector at a voxel for a specific fiber population."""
        i, j, k = self._to_voxel_idx(voxel)

        if not self._in_bounds(i, j, k):
            return np.zeros(3)

        if fiber_idx >= len(self.dyads):
            return np.zeros(3)

        v = self.dyads[fiber_idx][i, j, k, :]
        norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

        if norm > 0:
            return v / norm
        return np.zeros(3)

    def get_max_fiber_fraction(self, voxel):
        """Return max fiber volume fraction across all populations at a voxel.

        Equivalent to ``max(get_fiber_fraction(voxel, i) for i in range(num_fibers))``
        but avoids re-running ``_to_voxel_idx`` + ``_in_bounds`` per population
        and prefers a single batched index when the pre-stacked array exists.
        """
        i, j, k = self._to_voxel_idx(voxel)

        if not self._in_bounds(i, j, k):
            return 0.0

        if self.f_samples:
            return float(max(self.f_samples[fi][i, j, k]
                             for fi in range(len(self.f_samples))))

        if self.f_all_array is not None:
            return float(self.f_all_array[:, i, j, k, :].mean(axis=1).max())

        if self.f_all_samples:
            return float(max(np.mean(self.f_all_samples[fi][i, j, k, :])
                             for fi in range(len(self.f_all_samples))))

        return 0.0

    def get_fiber_fraction(self, voxel, fiber_idx=0):
        """Get the fiber volume fraction at a voxel.
        
        Uses mean f_samples if available, otherwise computes mean from posterior samples.
        """
        i, j, k = self._to_voxel_idx(voxel)

        if not self._in_bounds(i, j, k):
            return 0.0

        # Use mean f_samples if loaded (from dyads loading)
        if len(self.f_samples) > fiber_idx:
            return float(self.f_samples[fiber_idx][i, j, k])

        # Otherwise, compute mean from posterior samples
        if len(self.f_all_samples) > fiber_idx:
            f_vals = np.asarray(self.f_all_samples[fiber_idx][i, j, k, :])
            return float(np.mean(f_vals))

        return 0.0

    def is_valid_position(self, pos):
        """Check if position is within the brain mask."""
        i, j, k = self._to_voxel_idx(pos)
        return self._in_bounds(i, j, k) and self.mask[i, j, k]

    def position_in_mask(self, pos, mask):
        """
        Check if a position falls within a given mask.
        
        Useful for checking target/exclusion/termination masks.
        
        Args:
            pos: (i, j, k) position in voxel coordinates (can be float)
            mask: 3D boolean numpy array (same shape as volume)
            
        Returns:
            bool: True if position is within mask, False otherwise
        """
        if mask is None:
            return False
        i, j, k = self._to_voxel_idx(pos)
        if not self._in_bounds(i, j, k):
            return False
        return bool(mask[i, j, k])
