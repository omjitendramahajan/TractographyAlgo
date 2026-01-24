# PSOCT-Priority Tractography Algorithm

A tractography algorithm that prioritizes PSOCT (microscopy) orientation data over diffusion MRI (BEDPOSTX), using microscopy exclusively where available.

## Project Structure

```
TractographyAlgo/
├── tractography/              # Core package
│   ├── __init__.py
│   ├── bedpostx.py           # BEDPOSTX data loading (with memory-mapped option)
│   ├── psoct.py              # PSOCT data loading
│   ├── tracker.py            # PSOCTPriorityTracker (hybrid mode)
│   └── tracker_draft.py      # Simplified BEDPOSTX-only tracker
├── scripts/                   # Runnable scripts
│   ├── run_tractography.py   # Main entry point
│   └── create_test_seed.py   # Generate test seed masks
├── tests/                     # Test scripts
│   ├── checkdata.py          # BedpostxData testing
│   └── test_3d_tracking.py   # 3D tracking simulation tests
├── TRK_outputs/               # Tractography outputs (auto-generated)
├── simulation/                # 2D simulation experiments
├── notebooks/                 # Jupyter notebooks
├── figures/                   # Generated figures
├── cmc_hybrid/                # CMC hybrid package (external)
└── fsl_streamlines/           # FSL streamlines package (external)
```

## Installation

```bash
# Create conda environment
conda create -n tractography python=3.11
conda activate tractography

# Install dependencies
pip install numpy nibabel scipy matplotlib fslpy

# Install cmc_hybrid (optional)
cd cmc_hybrid && pip install -e . && cd ..
```

## Quick Start

### Option 1: Edit CONFIG and run (recommended)

Edit the CONFIG dictionary in `scripts/run_tractography.py`:

```python
CONFIG = {
    'bedpostx_dir': "/path/to/data.bedpostX_uncompressed",
    'psoct_pattern': None,        # or glob pattern for PSOCT
    'seed_mask': "/path/to/seed.nii.gz",
    'output': "streamlines.trk",
    'step_size': 0.1,             # mm
    'max_steps': 2000,
    'seeds_per_voxel': 10,
    'angle_threshold': 60,
    'use_memory_map': True,       # True = low RAM (~100MB)
    'num_fibers': 1,
}
```

Then run:
```bash
python scripts/run_tractography.py
```

### Option 2: Command-line arguments

```bash
python scripts/run_tractography.py \
    --bedpostx /path/to/data.bedpostX \
    --seeds seed_mask.nii.gz \
    --output output.trk
```

## Memory Modes

| Mode | RAM Usage | Speed | When to Use |
|------|-----------|-------|-------------|
| `use_memory_map=True` | ~100 MB | Slower | Development, low-RAM machines |
| `use_memory_map=False` | ~5-15 GB | Fast | Production, high-RAM machines |

**Tip:** For full-brain tractography, use `use_memory_map=False` with sufficient RAM.

## Visualize Results

```bash
fsleyes /path/to/brain_mask.nii TRK_outputs/Output1/streamlines.trk
```

## Algorithm

The algorithm uses a **priority-based** approach:

1. **PSOCT Priority**: If microscopy data exists at the current voxel, use it exclusively
2. **BEDPOSTX Fallback**: Only use diffusion data where microscopy is absent
3. **No Blending**: PSOCT is never diluted or averaged with diffusion data

When PSOCT is not provided (`psoct_pattern=None`), the tracker uses BEDPOSTX probabilistic sampling only.

## Output

Each run creates a folder in `TRK_outputs/` containing:
- `.trk` - Streamline file (viewable in FSLeyes)
- `_params.json` - All parameters and results for reproducibility

## Dependencies

- numpy
- nibabel
- scipy
- fslpy
- matplotlib

## Credits

This project uses the following external packages:

- **cmc_hybrid** - Hybrid tractography utilities  
  https://git.fmrib.ox.ac.uk/saad/cmc_hybrid.git

- **fsl_streamlines (ptx2)** - FSL probabilistic tractography  
  https://git.fmrib.ox.ac.uk/fsl/ptx2.git

Both packages are developed at FMRIB, University of Oxford.
