"""
Convert FSL probtrackx2 --savepaths output to a TRK file.

This is a thin, args-based wrapper around the existing logic in
convert_fsl_to_trk.py so it can be called from run_probtrackx2.sh with
config-driven paths.

probtrackx2 --savepaths writes a text file: one line per (x, y, z) sample,
with streamlines separated by blank lines or NaN markers depending on FSL
version. We parse both.
"""

import argparse
import os
import numpy as np
from nibabel.streamlines import Tractogram, save as save_tractogram
from nibabel import load as load_nifti


def parse_paths_file(path):
    streamlines = []
    current = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                if current:
                    streamlines.append(np.array(current, dtype=np.float32))
                    current = []
                continue
            if parts[0] in ("END", "NaN", "nan"):
                if current:
                    streamlines.append(np.array(current, dtype=np.float32))
                    current = []
                continue
            try:
                x, y, z = map(float, parts[:3])
                current.append([x, y, z])
            except ValueError:
                if current:
                    streamlines.append(np.array(current, dtype=np.float32))
                    current = []
    if current:
        streamlines.append(np.array(current, dtype=np.float32))
    return streamlines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths", required=True,
                   help="probtrackx2 --savepaths output file")
    p.add_argument("--ref",   required=True,
                   help="reference NIfTI for the affine (e.g., brain mask)")
    p.add_argument("--out",   required=True,
                   help="output .trk path")
    p.add_argument("--space", choices=["voxel", "world"], default="voxel",
                   help="coordinate system of the paths file "
                        "(probtrackx2 default is voxel)")
    args = p.parse_args()

    streamlines = parse_paths_file(args.paths)
    print(f"Parsed {len(streamlines)} streamlines from {args.paths}")
    if not streamlines:
        print("No streamlines, abort.")
        return

    img = load_nifti(args.ref)
    affine = img.affine.astype(np.float32)
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    dims = np.array(img.shape[:3], dtype=np.int16)

    # If the paths are in voxel coords, transform to RASMM by applying the affine.
    if args.space == "voxel":
        world_streamlines = []
        for sl in streamlines:
            sl_h = np.hstack([sl, np.ones((len(sl), 1))])
            world_streamlines.append((affine @ sl_h.T).T[:, :3].astype(np.float32))
        streamlines = world_streamlines

    header = {
        "voxel_sizes":    voxel_sizes,
        "dimensions":     dims,
        "voxel_to_rasmm": affine,
        "voxel_order":    "RAS",
    }
    tractogram = Tractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_tractogram(tractogram, args.out, header=header)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
