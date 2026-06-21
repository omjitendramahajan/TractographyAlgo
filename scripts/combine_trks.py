"""
Combine multiple .trk files into a single .trk file.

Edit the CONFIG section below, then run:
    python combine_trks.py
"""

# ---------------------------------------------------------------------------
# CONFIG — edit these paths
# ---------------------------------------------------------------------------
# List the .trk files you want to combine. Globs (e.g. "TRK_outputs/*/foo.trk")
# are expanded automatically.
INPUT_TRKS = [
    # "TRK_outputs/Output_20260519_CC_withPSOCT/CC_tract.trk",
    # "TRK_outputs/Output_20260519_CC_bedpostxonly/CC_tract.trk",
]

# Where to write the combined .trk file.
OUTPUT_TRK = "TRK_outputs/combined.trk"
# ---------------------------------------------------------------------------


import glob
import sys

import nibabel as nib
import numpy as np
from nibabel.streamlines import Tractogram, save as save_tractogram


def expand_inputs(patterns):
    files = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            files.extend(matches)
        else:
            files.append(p)
    return files


def combine(input_paths, output_path):
    all_streamlines = []
    reference_header = None
    reference_affine = None

    for path in input_paths:
        tfile = nib.streamlines.load(path)
        print(f"  {path}: {len(tfile.streamlines)} streamlines")
        all_streamlines.extend(list(tfile.streamlines))

        if reference_header is None:
            reference_header = tfile.header
            reference_affine = tfile.affine
        else:
            if not np.allclose(tfile.affine, reference_affine, atol=1e-4):
                print(f"  WARNING: affine in {path} differs from first file.",
                      file=sys.stderr)

    tractogram = Tractogram(all_streamlines, affine_to_rasmm=np.eye(4))
    save_tractogram(tractogram, output_path, header=reference_header)
    print(f"Wrote {len(all_streamlines)} streamlines to {output_path}")


def main():
    if not INPUT_TRKS:
        sys.exit("No inputs. Edit INPUT_TRKS in the CONFIG section.")

    files = expand_inputs(INPUT_TRKS)
    if not files:
        sys.exit("No input files matched.")

    print(f"Combining {len(files)} file(s):")
    combine(files, OUTPUT_TRK)


if __name__ == "__main__":
    main()
