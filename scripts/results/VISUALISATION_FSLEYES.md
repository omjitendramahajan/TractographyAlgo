# Visualising the results in FSLeyes

The Python scripts in this folder compute every **numeric** result and the
**statistics** figures (bar charts, box plots, histograms). They deliberately do
**not** render the spatial / glass-brain overlays — those are done by hand in
FSLeyes. This file is the recipe for each one.

Everything below assumes you launch from the project root:

```bash
cd "/Users/ommahajan/Desktop/Year_4/MEng Individual project"
```

and uses these shorthands (edit to match your machine):

```bash
BX="DATA/data.bedpostX_SSFP_uncompressed"          # bedpostX dir = the grid
TRK="TractographyAlgo/tractography/TRK_outputs"     # per-run output folders
```

All per-run outputs (`*_visit_map.nii.gz`, `*_divergence_counts.nii.gz`,
`*.trk`) are already in **native dMRI space**, i.e. the same grid as the
bedpostX volumes, so they overlay directly with no extra registration.

Each run folder (e.g. `$TRK/hybrid_run_01/`) contains:

| File | What it is | How to view |
|------|------------|-------------|
| `*_visit_map.nii.gz` | per-voxel streamline visit count (the tract density) | volume overlay |
| `*_divergence_counts.nii.gz` | per-voxel population-divergence count (hybrid runs) | volume overlay |
| `*.trk` | the streamlines themselves | tractogram (3D view) |
| `*_geometry.png`, `*_terminations.png` | already-rendered diagnostics | open in any image viewer |

---

## 0. Background image (do this first)

Use a bedpostX volume as the greyscale background so every overlay lands on the
correct grid:

```bash
fsleyes $BX/mean_f1samples.nii          # FA-like; good anatomical context
# or, for just the brain outline:
fsleyes $BX/nodif_brain_mask.nii
```

If you have a structural / FA already registered to dMRI space, load that as the
background instead.

---

## 1. Hybrid vs baseline tract (the headline spatial comparison)

Show the two density maps in contrasting colourmaps and threshold them at the
density cut-off used by the scripts (`DENSITY_THRESHOLD = 1`, set in each
script's CONFIG, e.g. `volume_overlap.py`):

```bash
fsleyes $BX/mean_f1samples.nii \
        $TRK/hybrid_run_01/*_visit_map.nii.gz   -cm red-yellow  -dr 1 50 \
        $TRK/baseline_run_01/*_visit_map.nii.gz -cm blue-lightblue -dr 1 50
```

- `-cm` = colourmap, `-dr LO HI` = display range. Setting `LO = 1` hides voxels
  visited fewer than `DENSITY_THRESHOLD` times, so what you see matches the
  Dice/volume numbers in `volume_overlap.md`.
- In the GUI: **Overlay list** → click each map → **Overlay display panel** to
  tweak colourmap, min/max, and opacity.
- Toggle the eye icons to flick hybrid on/off over baseline and see where the
  hybrid recruits extra voxels (the crossing-fibre regions).
- Bump `HI` (e.g. to the 95th percentile of the map) if the tract looks
  saturated.

**Tip — overlap as outlines:** set the baseline overlay to *Outline* mode
(display panel, for a mask) or threshold it and tick **Show outline only** so
you see the hybrid filled, baseline as a contour — an at-a-glance Dice.

---

## 2. Streamlines / glass brain (3D)

```bash
fsleyes -s 3d $BX/mean_f1samples.nii  $TRK/hybrid_run_01/*.trk
```

- FSLeyes loads `.trk` (and `.tck` / `.vtk`) tractograms directly. Switch the
  view to **3D** (the cube icon, or `-s 3d` above).
- In the streamline display panel: colour **by orientation** (RGB: red = L–R,
  green = A–P, blue = S–I), set line width / tube radius, and lower the
  **subsample** if it is slow.
- Drag the background volume's clipping planes (or hide it) for a clean
  glass-brain look. Load both hybrid and baseline `.trk` in two colours to
  compare bundles side by side.

---

## 3. Population-selection divergence

Where the hybrid PS-OCT scorer overrode the trajectory-only rule:

```bash
fsleyes $BX/mean_f1samples.nii \
        $TRK/hybrid_run_01/*_divergence_counts.nii.gz -cm hot -dr 1 20 \
        $BX/mean_f2samples.nii -cm blue -dr 0.05 0.5 -a 40
```

- The `hot` overlay = divergence hotspots. The `mean_f2samples` overlay (second
  fibre fraction, thresholded at `F_MIN = 0.05`) marks **crossing-fibre**
  voxels — visually confirm the divergence concentrates there, which is the
  point made quantitatively in `divergence.md` (crossing vs single-fibre).
- `-a 40` = 40 % opacity so you can see both layers.
- If a run has no `*_divergence_counts.nii.gz`, the divergence map is inside
  `*_stats.npz` instead (older runs); re-run that case or use a hybrid run that
  saved the NIfTI.

---

## 4. Anatomical reference overlap

Once the XTRACT FMA reference is warped into native dMRI space (the file you set
as `REFERENCE_MASK` in `reference_overlap.py` / `sensitivity.py`), eyeball the
Dice reported in `reference_overlap.md`:

```bash
fsleyes $BX/mean_f1samples.nii \
        REFERENCE_native.nii.gz                 -cm green   -dr 0.5 1  \
        $TRK/hybrid_run_01/*_visit_map.nii.gz   -cm red-yellow -dr 1 50
```

Set the reference to **outline only** so the hybrid tract fills inside the
reference contour — the green border is the "ground truth" envelope.

---

## 5. Sensitivity sweeps (alpha / theta / threshold / density)

Load the sweep runs you configured in `sensitivity.py` with the **same**
colourmap and display range, then toggle them one at a time to see how the tract
grows/shrinks:

```bash
fsleyes $BX/mean_f1samples.nii \
        $TRK/sweep_theta_45/*_visit_map.nii.gz -cm hot -dr 1 50 \
        $TRK/hybrid_run_01/*_visit_map.nii.gz  -cm hot -dr 1 50 \
        $TRK/sweep_theta_80/*_visit_map.nii.gz -cm hot -dr 1 50
```

Use **View → Movie mode**, or just step the opacity / eye toggles, to animate
across the sweep. The volume numbers behind this are in `sensitivity.md` /
`fig_sensitivity_*.png`.

---

## 6. Held-out PS-OCT validation (optional, qualitative)

To eyeball the angular agreement that `held_out.md` quantifies, overlay a
held-out PS-OCT slide, the principal-direction vectors, and the streamlines:

```bash
fsleyes $BX/mean_f1samples.nii \
        /path/to/held_out_slides/<one_slice>.nii.gz -cm hsv \
        $BX/dyads1.nii \
        $TRK/held_out_even/*.trk
```

Set `dyads1` to a **line/vector** display (RGB by direction) and compare its
in-plane orientation to the streamlines where they cross the held-out slide.

---

## Saving what you see

- **Screenshot:** GUI → *File ▸ Save screenshot*, or headless:
  ```bash
  fsleyes render --outfile overlay.png \
      $BX/mean_f1samples.nii \
      $TRK/hybrid_run_01/*_visit_map.nii.gz -cm red-yellow -dr 1 50
  ```
- **Reusable layout:** arrange overlays, then *File ▸ Save perspective* (or
  *Settings ▸ Ortho/3D view ▸ ...*) and reload it next time so every figure uses
  identical colourmaps and ranges.
- Keep the **same `-cm` / `-dr`** across hybrid, baseline and sweep panels so
  the figures are directly comparable in the report.
