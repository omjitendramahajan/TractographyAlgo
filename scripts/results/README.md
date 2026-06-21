# Results post-processing

Split-up, **fully independent** replacement for the old monolithic
`scripts/postprocess_results.py`. Each methodology test is its own standalone
script: nothing is imported from a shared module, and each script has a `CONFIG`
block at the top where you paste the **exact path to every file** that test
needs. Edit one script without touching the others.

## Scripts (each is independent)

| Script | Computes | Key files to set in its CONFIG |
|--------|----------|--------------------------------|
| `volume_overlap.py` | Hybrid vs baseline volumes, streamline counts, Dice/Jaccard, probtrackx2 | per-run `visit_map` + `trk`; the two default visit maps |
| `divergence.py` | Population-selection divergence (fraction + crossing/single split) | per-run `divergence` + `params`; representative divergence/visit maps; `mean_f2samples` |
| `sensitivity.py` | Sweeps: alpha, theta, in-plane threshold, PS-OCT density | a `visit_map` per sweep value; baseline default; optional reference |
| `reproducibility.py` | Pairwise FTC across the hybrid / baseline repeats | the list of hybrid + baseline `visit_map`s |
| `reference_overlap.py` | Dice/Jaccard vs the anatomical (XTRACT FMA) reference | `REFERENCE_MASK`; hybrid + baseline visit maps |
| `held_out.py` | Held-out interleaved PS-OCT angular-error validation | held-out `trk`; brain mask; explicit list of PS-OCT slide files |

Plus `VISUALISATION_FSLEYES.md` — how to make the spatial overlays in FSLeyes.

## Running

Most scripts only need the conda env. **Only `held_out.py`** needs the
`PYTHONPATH` (it imports `fsl` + the project's `tractography` package):

```bash
PY=/Users/ommahajan/anaconda3/envs/tractography/bin/python3
cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"

$PY TractographyAlgo/scripts/results/volume_overlap.py
$PY TractographyAlgo/scripts/results/divergence.py
$PY TractographyAlgo/scripts/results/sensitivity.py
$PY TractographyAlgo/scripts/results/reproducibility.py
$PY TractographyAlgo/scripts/results/reference_overlap.py

PYTHONPATH=TractographyAlgo/cmc_hybrid:TractographyAlgo \
    $PY TractographyAlgo/scripts/results/held_out.py
```

Each script writes, into its configured `OUTPUT_DIR` (default
`TRK_outputs/_postprocessing/`): `<test>.json` (the numbers), `<test>.md` (the
filled methodology lines), and its `fig_*.png`. There is no combined roll-up —
concatenate the `.md` files yourself if you want one document, e.g.
`cat _postprocessing/*.md > all_results.md`.

## ⚠️ The prefilled paths are a TEMPLATE — edit them

Every CONFIG is prefilled with absolute paths that follow `*_run_01`'s naming
(`hybrid_run_NN/FMA_tract_seed_10seeds_PSOCT_*`, `baseline_run_NN/..._BedpostX_*`).
Your actual files vary — the existing baseline runs alone use
`...BedpostX...`, `...Bedpostx...` and `baseline_3_...`. **Open each script and
replace every path with the exact filename on disk.** A missing/misspelled path
is skipped with a note (never fatal), so the script still runs — but that test's
numbers will be `[n/a]` until the path is right. The console prints how many
files each test actually loaded, e.g. `baseline: 2/10 densities loaded`.

## What each test needs (quick reference)

- **Densities** = `*_visit_map.nii.gz` (preferred — carries the affine, so
  volumes are exact) or a `*_stats.npz` (key `visit_map`; then volumes use the
  `VOXEL_VOLUME_MM3` fallback in the CONFIG).
- **Streamline counts** = the `.trk` header (falls back to `*_params.json`
  `results.total_streamlines` if you also give a `params` path).
- **Divergence** = `*_divergence_counts.nii.gz`, or a `*_stats.npz` with a
  `divergence_map` key.
- **Crossing vs single-fibre** = bedpostX `mean_f2samples.nii` (`f2 >= F_MIN`).
- **Held-out** = the held-out `.trk`, the bedpostX `nodif_brain_mask.nii` (for
  the affine + PSOCTData), and the explicit list of held-out PS-OCT slide files.

Spatial overlays are **not** produced here — see
[`VISUALISATION_FSLEYES.md`](VISUALISATION_FSLEYES.md).
