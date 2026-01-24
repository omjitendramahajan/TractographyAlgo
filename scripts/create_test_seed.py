
"""
Create a small test seed mask for tractography testing.

Usage:
    python create_test_seed.py /path/to/data.bedpostX

This creates a 4x4x4 voxel cube in the center of the brain.
"""

import sys
import nibabel as nib
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_test_seed.py /path/to/data.bedpostX")
        sys.exit(1)
    
    bedpostx_dir = sys.argv[1]
    mask_path = f"{bedpostx_dir}/nodif_brain_mask.nii.gz"
    
    # Load brain mask
    print(f"Loading {mask_path}...")
    ref = nib.load(mask_path)
    brain_mask = ref.get_fdata() > 0
    
    # Find center of mass of brain
    coords = np.argwhere(brain_mask)
    center = coords.mean(axis=0).astype(int)
    cx, cy, cz = center
    
    print(f"Brain center: ({cx}, {cy}, {cz})")
    
    # Create small seed region (4x4x4 cube)
    seed_mask = np.zeros(ref.shape, dtype=np.float32)
    seed_mask[cx-2:cx+2, cy-2:cy+2, cz-2:cz+2] = 1
    
    # Make sure seeds are within brain
    seed_mask = seed_mask * brain_mask
    
    n_seeds = int(seed_mask.sum())
    print(f"Created seed mask with {n_seeds} voxels")
    
    # Save
    output_path = "test_seed.nii.gz"
    nib.save(nib.Nifti1Image(seed_mask, ref.affine, ref.header), output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
