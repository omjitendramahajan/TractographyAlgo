
"""
Convert FSL probtrackx2 output (particle_paths.txt) to TRK format.
"""

import numpy as np
import sys
import os
from nibabel.streamlines import Tractogram, save as save_tractogram
from nibabel import load as load_nifti

def convert_fsl_to_trk(txt_path, ref_img_path, output_path):
    print(f"Converting {txt_path} -> {output_path}")
    
    # Load reference image for affine
    img = load_nifti(ref_img_path)
    affine = img.affine
    
    # Read text file
    # probtrackx2 --savepaths format:
    # x y z
    # x y z
    # ...
    # END
    # (or separated by newlines/NaNs depending on version)
    
    streamlines = []
    current_streamline = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Check for terminator
            if parts[0] == 'END' or parts[0] == 'NaN':
                if current_streamline:
                    streamlines.append(np.array(current_streamline))
                    current_streamline = []
                continue
            
            try:
                # FSL output is in voxel coordinates (internal) or world?
                # --savepaths usually outputs voxel coordinates
                x, y, z = map(float, parts[:3])
                current_streamline.append([x, y, z])
            except ValueError:
                # Delimiter like 'nan nan nan'
                if current_streamline:
                    streamlines.append(np.array(current_streamline))
                    current_streamline = []
    
    # Add last one if exists
    if current_streamline:
        streamlines.append(np.array(current_streamline))
        
    print(f"Found {len(streamlines)} streamlines")
    
    if len(streamlines) == 0:
        print("Warning: No streamlines found.")
        return

    # Convert to world coordinates if necessary
    # FSL probtrackx2 output is usually in world coordinates (mm) 
    # if --simple is NOT used, but --seedref might change it.
    # However, usually it matches the seed space.
    # If the file format says "voxel coordinates", we need affine.
    # BUT, most FSL outputs are in world space (mm).
    # Let's assume world space for now, as that's standard for TRK too.
    # Tractogram expects streamlines in RAS+ mm space.
    # FSL is usually LAS/LPS? Nibabel handles this via affine_to_rasmm=np.eye(4) if points are already world.
    
    # IMPORTANT: TRK stores coords in RAS+ mm.
    # FSL world space is usually aligned with the image, but check for stride.
    
    tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    save_tractogram(tractogram, output_path)
    print(f"Saved {output_path}")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'txt_path': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/fsl_output_comparison/saved_paths.txt",
    'ref_img_path': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/DATA/data.bedpostX_SSFP/nodif_brain_mask.nii.gz",
    'output_path': "fsl_streamlines_probtrackx.trk"
}

if __name__ == "__main__":
    convert_fsl_to_trk(
        CONFIG['txt_path'],
        CONFIG['ref_img_path'],
        CONFIG['output_path']
    )
