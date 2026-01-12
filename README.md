# PSOCT-Priority Tractography Algorithm

A tractography algorithm that prioritizes PSOCT (microscopy) orientation data over diffusion MRI (BEDPOSTX), using microscopy exclusively where available.

## Project Structure

```
TractographyAlgo/
├── custom_tractography.py    # Main PSOCT-priority tractography script
├── create_test_seed.py       # Generate small test seed masks
├── simulation/               # 2D simulation experiments
│   ├── Agent.py              # HybridAgent class
│   ├── DataGenerator.py      # GroundTruth generator
│   ├── ComparisonSimulation.py
│   ├── CrossingFiberDemo.py
│   ├── SeedSimulation.py
│   └── Tracking.py
├── notebooks/                # Jupyter notebooks
├── figures/                  # Generated figures
├── scripts/                  # Utility scripts
├── TRK_outputs/              # Tractography outputs (auto-generated)
│   └── Output1/
│       ├── streamlines.trk
│       └── streamlines_params.json
├── cmc_hybrid/               # CMC hybrid package (external)
└── fsl_streamlines/          # FSL streamlines package (external)
```

## Installation

```bash
# Create conda environment
conda create -n tractography python=3.11
conda activate tractography

# Install dependencies
pip install numpy nibabel scipy matplotlib

# Install cmc_hybrid
cd cmc_hybrid && pip install -e . && cd ..

# Install fsleyes (optional, for visualization)
pip install fsleyes
```

## Usage

### 1. Create a test seed
```bash
python create_test_seed.py /path/to/data.bedpostX
```

### 2. Run tractography
```bash
python custom_tractography.py \
    --bedpostx /path/to/data.bedpostX \
    --psoct "/path/to/psoct/Slice_*_EnAO_*.nii.gz" \
    --seeds test_seed.nii.gz \
    --output output.trk \
    --deterministic
```

### Command-line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--bedpostx` | Yes | - | Path to .bedpostX directory |
| `--psoct` | No | None | Glob pattern for PSOCT slides |
| `--seeds` | Yes | - | Seed mask (NIfTI) |
| `--output` | No | streamlines.trk | Output filename |
| `--step-size` | No | 0.5 | Step size in mm |
| `--max-steps` | No | 2000 | Max steps per direction |
| `--seeds-per-voxel` | No | 1 | Seeds per voxel |
| `--deterministic` | No | False | Use mean orientations (lower memory) |

### 3. Visualize results
```bash
fsleyes brain_mask.nii.gz TRK_outputs/Output1/output.trk
```

## Algorithm

The algorithm uses a **priority-based** approach:

1. **PSOCT Priority**: If microscopy data exists at the current voxel, use it exclusively
2. **BEDPOSTX Fallback**: Only use diffusion data where microscopy is absent
3. **No Blending**: PSOCT is never diluted or averaged with diffusion data

## Output

Each run creates a folder in `TRK_outputs/` containing:
- `.trk` - Streamline file (viewable in FSLeyes)
- `_params.json` - All parameters and results for reproducibility

## Dependencies

- numpy
- nibabel
- scipy
- fslpy
- cmc_hybrid (included)

## Credits

This project uses the following external packages:

- **cmc_hybrid** - Hybrid tractography utilities  
  https://git.fmrib.ox.ac.uk/saad/cmc_hybrid.git

- **fsl_streamlines (ptx2)** - FSL probabilistic tractography  
  https://git.fmrib.ox.ac.uk/fsl/ptx2.git

Both packages are developed at FMRIB, University of Oxford.
