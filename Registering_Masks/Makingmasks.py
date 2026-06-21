import nibabel as nib
import numpy as np

# Load the reference image (e.g., the volume you're seeding into)
ref = nib.load('/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP/nodif_brain_mask.nii.gz')
shape = ref.shape  # (X, Y, Z)

# Create empty mask with same shape
mask = np.zeros(shape, dtype=np.uint8)

# Seed voxel and box size
cx, cy, cz = 165, 130, 28
size = 4
half = size // 2

# Compute bounds (clipped to volume)
x0, x1 = max(0, cx - half), min(shape[0], cx + half)
y0, y1 = max(0, cy - half), min(shape[1], cy + half)
z0, z1 = max(0, cz - half), min(shape[2], cz + half)

mask[x0:x1, y0:y1, z0:z1] = 1

# Save with the same affine + header so it aligns with the reference
mask_img = nib.Nifti1Image(mask, affine=ref.affine, header=ref.header)
mask_img.set_data_dtype(np.uint8)
nib.save(mask_img, 'seed_mask_165_130_28.nii.gz')