"""
make_single_fiber_seeds.py

Generate a binary seed mask containing only voxels with one strongly
dominant BedpostX fibre population — useful for isolating the
tractography algorithm from crossing-fibre confounds while debugging.

Criteria (all must hold):
    voxel is in the BedpostX brain mask
    mean_f1 >= --f1-min                          (dominant fibre present)
    max(mean_f2, mean_f3, ...) <= --f-other-max  (no significant secondary)
    voxel is in --restrict-to (optional)         (e.g. an anatomical ROI)

Output: NIfTI binary mask aligned to BedpostX, ready to drop into
run_tractography.py's CONFIG['seed_mask'].

Example:
    python make_single_fiber_seeds.py \
        --bedpostx-dir /path/to/data.bedpostX \
        --output single_fiber_seeds.nii.gz

    # restrict to a known anatomical region (e.g. CC body)
    python make_single_fiber_seeds.py \
        --bedpostx-dir /path/to/data.bedpostX \
        --restrict-to cc_body_mask.nii.gz \
        --output cc_body_single_fiber_seeds.nii.gz
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from tractography.bedpostx import BedpostxData


def main():
    parser = argparse.ArgumentParser(
        description="Build a binary seed mask of single-fibre voxels.")
    parser.add_argument("--bedpostx-dir", required=True,
                        help="Path to the .bedpostX directory.")
    parser.add_argument("--output", required=True,
                        help="Output NIfTI path for the binary seed mask.")
    parser.add_argument("--f1-min", type=float, default=0.4,
                        help="Minimum mean f1 (dominant fibre). Default 0.4. "
                             "Increase for stricter single-fibre selection.")
    parser.add_argument("--f-other-max", type=float, default=0.1,
                        help="Max mean f for any secondary population. "
                             "Default 0.1. Lower means cleaner single-fibre "
                             "voxels but fewer of them.")
    parser.add_argument("--num-fibers", type=int, default=3,
                        help="Number of fibre populations BedpostX was run "
                             "with. Default 3.")
    parser.add_argument("--restrict-to", default=None,
                        help="Optional NIfTI mask. Output seeds are limited "
                             "to voxels also in this mask. Must share the "
                             "BedpostX shape (i.e. be in dMRI voxel space).")
    args = parser.parse_args()

    bp = BedpostxData(args.bedpostx_dir, num_fibers=args.num_fibers,
                      load_samples=False, load_dyads=True,
                      use_memory_map=True)

    in_brain = bp.mask.astype(bool)
    print(f"Brain mask:                          {int(in_brain.sum())} voxels")

    if len(bp.f_samples) == 0:
        raise RuntimeError("BedpostX dyads/mean_f files not loaded — check "
                           "the --bedpostx-dir.")

    f1 = np.asarray(bp.f_samples[0])
    if len(bp.f_samples) > 1:
        f_other = np.stack(
            [np.asarray(bp.f_samples[i]) for i in range(1, len(bp.f_samples))],
            axis=0,
        ).max(axis=0)
    else:
        f_other = np.zeros_like(f1)
        print("  (only one fibre population loaded; f_other = 0 everywhere)")

    primary_strong = f1 >= args.f1_min
    others_weak    = f_other <= args.f_other_max

    seed_mask = in_brain & primary_strong & others_weak
    print(f"AND f1 >= {args.f1_min}:                    "
          f"{int((in_brain & primary_strong).sum())} voxels")
    print(f"AND max(f_other) <= {args.f_other_max}:           "
          f"{int(seed_mask.sum())} voxels")

    if args.restrict_to:
        restrict = nib.load(args.restrict_to).get_fdata() > 0
        if restrict.shape != seed_mask.shape:
            raise ValueError(
                f"Restrict mask shape {restrict.shape} does not match "
                f"BedpostX shape {seed_mask.shape}. Resample the mask into "
                f"dMRI voxel space first."
            )
        seed_mask = seed_mask & restrict
        print(f"AND in --restrict-to:                {int(seed_mask.sum())} voxels")

    n_seeds = int(seed_mask.sum())
    if n_seeds == 0:
        print("\nWARNING: zero voxels remain. Loosen --f1-min or "
              "--f-other-max.")
        return

    out_img = nib.Nifti1Image(seed_mask.astype(np.uint8), bp.affine)
    out_img.header.set_data_dtype(np.uint8)
    nib.save(out_img, args.output)

    print(f"\nWrote {n_seeds} seed voxels to {args.output}")
    print("\nQuality stats over the selected seeds:")
    print(f"  f1            median {np.median(f1[seed_mask]):.3f}, "
          f"range {np.min(f1[seed_mask]):.3f}-{np.max(f1[seed_mask]):.3f}")
    if len(bp.f_samples) > 1:
        print(f"  max(f_other)  median {np.median(f_other[seed_mask]):.3f}, "
              f"range {np.min(f_other[seed_mask]):.3f}-"
              f"{np.max(f_other[seed_mask]):.3f}")


if __name__ == "__main__":
    main()
