"""
animate_trk.py - Render a tractography propagation movie with FURY.

Replays a ``.trk`` file step-by-step: at frame ``t``, every streamline is
drawn from its first point up to its ``t``-th point. Streamlines grow
simultaneously from their seeds outward, so you see the propagation
front advance through the tract. Colours follow the standard tractography
convention (local tangent direction -> RGB).

Output: an MP4 saved next to the ``.trk`` (or to ``CONFIG['output']``).

Usage:
    Edit CONFIG below, then run:
        python scripts/animate_trk.py

Requirements (already in the ``tractography`` conda env):
    - fury, vtk, nibabel, numpy
    - ``ffmpeg`` binary on PATH (used via subprocess; no Python ffmpeg dep)
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import nibabel as nib

import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from fury import actor, window


# =============================================================================
# CONFIGURATION - edit, then run: python scripts/animate_trk.py
# =============================================================================
CONFIG = {
    'trk_path': "/Users/ommahajan/Desktop/Year_4/MEng Individual project/TractographyAlgo/tractography/TRK_outputs/Output_20260530_100606/CC_tract_seed_noPSOCT_8x8x8.trk",
    'output': None,
    'mask_path': None,

    # CHANGE THIS FROM 2 TO 5 OR 6
    # It will skip fewer intermediate tracking steps, reducing total frames 
    # to ~70 instead of 201. Your movie stays smooth, but saves massive disk/RAM.
    'steps_per_frame': 5, 

    'fps': 30,

    # LOWER RESOLUTION SIZING
    # Going down to 720p or standard HD makes individual frame snapshots 
    # dramatically smaller and faster to render.
    'window_size': (1024, 576), 

    'background': (0.05, 0.05, 0.1),
    'line_width': 2.0,         # only used when use_tubes=False
    'rotate_deg_per_frame': 0.4,
    'crf': 18,                 # 18 = visually lossless, 23 = default
    'preset': 'medium',

    # Hard RAM ceiling: random-sample at most this many streamlines to
    # animate. None = animate all. For a "is the algorithm doing what I
    # think" sanity-check movie, 500-1000 streamlines is plenty.
    'max_streamlines': 1000,

    # ---- Quality knobs ----------------------------------------------------
    # Supersampling (SSAA) is the primary anti-aliasing path here. Renders
    # at supersample * window_size, ffmpeg downsamples with Lanczos. This
    # is the cleanest AA available and works in headless/offscreen mode.
    #   1.0 = off (jagged edges)
    #   1.5 = noticeably cleaner, ~2.3x render cost
    #   2.0 = very clean ("retina"), ~4x render cost  (recommended for HD)
    # Per-frame framebuffer at supersample=2: (2 * 1024) * (2 * 576) * 4B
    # = ~9 MB; well within the budget freed up by the persistent actor.
    'supersample': 2.0,

    # MSAA: VTK's hardware multisampling. Useful on platforms where the
    # offscreen renderer supports it; silently ignored on others (e.g.
    # headless macOS). Stack with SSAA when it works. 0 disables, 4 or 8
    # are typical values.
    'multi_samples': 8,

    # Premium "streamtube" mode: each streamline becomes a shaded 3D tube
    # instead of a flat line. Gives depth/occlusion cues. Costs ~2-3x render
    # time because the tube filter re-runs each frame as new points come
    # online. RAM cost is modest (~10x line geometry, but built once).
    'use_tubes': False,
    'tube_radius': 0.15,   # in voxel units; tweak for your data
    'tube_sides': 8,       # 6 = OK, 8 = smooth, 12 = very smooth (heavier)

    # ---- Timing / pacing ---------------------------------------------------
    # Non-linear time mapping. t = max_len * (frame_idx / n_frames) ^ exponent.
    #   1.0 = linear (every frame advances the same amount; default before)
    #   1.5 = gentle front-load
    #   2.0 = moderate front-load  (half the movie shows the first 25% of growth)
    #   3.0 = heavy front-load     (half the movie shows the first 12.5% of growth)
    # Use this when the early "everything sprouts from the seed" phase is the
    # interesting part and the long tail (last few streamlines still growing)
    # is boring.
    'time_exponent': 2.0,

    # Optionally truncate the animation at this percentile of streamline
    # lengths, so outlier-long streamlines don't drag out the ending.
    # None = animate up to the longest streamline. 95 means: stop when 95%
    # of streamlines are done; the longest 5% freeze mid-growth.
    'length_percentile': 95,
}


# =============================================================================
# Persistent VTK actor helpers
# =============================================================================
# Memory-saving strategy: upload all streamline POINTS and COLORS to VTK
# exactly once, then per frame only rebuild the cell array (which point
# indices form which line). Avoids the per-frame actor allocation/destruction
# that drives RAM through the roof on big runs.

def _build_persistent_actor(streamlines, line_width,
                            use_tubes=False, tube_radius=0.15, tube_sides=8):
    """Create one actor whose points and per-point colors are uploaded once.

    Returns (vtkActor, vtkPolyData, point_offsets) - call
    ``_update_truncated_lines(polydata, streamlines, point_offsets, t)``
    each frame to redraw with the first ``t`` points of each streamline.

    When ``use_tubes`` is True the polydata is piped through a vtkTubeFilter
    so each line becomes a shaded 3D tube. The tube filter is part of the
    VTK pipeline, so updating the polydata's line connectivity re-triggers
    the tube mesh automatically.
    """
    # Concatenate every point of every streamline into one flat float32 array.
    all_pts = np.vstack(streamlines).astype(np.float32)

    # Per-point RGB from local tangent direction (standard tractography
    # convention - same colouring as FURY's line_colors).
    rgb_chunks = []
    point_offsets = [0]
    for s in streamlines:
        if len(s) < 2:
            rgb = np.zeros((len(s), 3), dtype=np.float32)
        else:
            diffs = np.diff(s, axis=0)
            # Each point uses the direction of the segment leaving it; the
            # last point reuses the previous direction.
            per_point_dir = np.vstack([diffs, diffs[-1:]])
            norms = np.linalg.norm(per_point_dir, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            rgb = np.abs(per_point_dir / norms).astype(np.float32)
        rgb_chunks.append(rgb)
        point_offsets.append(point_offsets[-1] + len(s))
    rgb_all = (np.vstack(rgb_chunks) * 255).astype(np.uint8)

    # Upload points (deep=True so VTK owns its own buffer; the numpy
    # arrays can be freed when this function returns).
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(all_pts, deep=True))

    color_arr = numpy_to_vtk(rgb_all, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    color_arr.SetName('Colors')

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(color_arr)
    polydata.SetLines(vtk.vtkCellArray())  # filled in per frame

    mapper = vtk.vtkPolyDataMapper()
    if use_tubes:
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(polydata)
        tube_filter.SetRadius(tube_radius)
        tube_filter.SetNumberOfSides(tube_sides)
        tube_filter.SetCapping(True)
        # Pipeline-connected: any polydata.Modified() re-runs the filter,
        # so per-frame truncation just works.
        mapper.SetInputConnection(tube_filter.GetOutputPort())
    else:
        mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOn()

    line_actor = vtk.vtkActor()
    line_actor.SetMapper(mapper)
    if not use_tubes:
        line_actor.GetProperty().SetLineWidth(line_width)
    else:
        # Tubes benefit from a touch of specular for the 3D look.
        prop = line_actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetSpecular(0.25)
        prop.SetSpecularPower(20)

    return line_actor, polydata, point_offsets


def _update_truncated_lines(polydata, streamlines, point_offsets, t):
    """Rebuild the cell array showing only the first ``t`` points of each line.

    Points + colors stay uploaded; only the connectivity changes.
    """
    connectivity_chunks = []
    cell_offsets = [0]
    total = 0
    for i, s in enumerate(streamlines):
        n = min(t, len(s))
        if n < 2:
            continue
        start = point_offsets[i]
        connectivity_chunks.append(np.arange(start, start + n, dtype=np.int64))
        total += n
        cell_offsets.append(total)

    cells = vtk.vtkCellArray()
    if total > 0:
        connectivity = np.concatenate(connectivity_chunks)
        offsets_arr = np.asarray(cell_offsets, dtype=np.int64)
        cells.SetData(
            numpy_to_vtkIdTypeArray(offsets_arr, deep=True),
            numpy_to_vtkIdTypeArray(connectivity, deep=True),
        )
    polydata.SetLines(cells)
    polydata.Modified()


def find_most_recent_trk():
    """Locate the newest .trk under TRK_outputs/ for the default config."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = sorted(
        glob.glob(os.path.join(script_dir, '..', 'tractography',
                               'TRK_outputs', '*', '*.trk')),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def animate(trk_path, output_path=None, mask_path=None,
            steps_per_frame=2, fps=30,
            window_size=(1280, 720), background=(0.05, 0.05, 0.1),
            line_width=1.5, rotate_deg_per_frame=0.4,
            crf=20, preset='medium',
            max_streamlines=None,
            multi_samples=8, supersample=1.0,
            use_tubes=False, tube_radius=0.15, tube_sides=8,
            time_exponent=1.0, length_percentile=None):
    print(f"Loading {trk_path}")
    trk = nib.streamlines.load(trk_path)
    streamlines = [np.asarray(s, dtype=np.float32) for s in trk.streamlines]
    streamlines = [s for s in streamlines if len(s) >= 2]
    if not streamlines:
        print("No streamlines with >=2 points; nothing to animate.")
        return

    # Hard RAM ceiling. Random sample (not every-Nth) so we don't bias the
    # visualisation by seed-order.
    if max_streamlines is not None and len(streamlines) > max_streamlines:
        rng = np.random.default_rng(0)
        keep_idx = rng.choice(len(streamlines), size=max_streamlines, replace=False)
        streamlines = [streamlines[i] for i in keep_idx]
        print(f"  subsampled to {len(streamlines)} streamlines (cap={max_streamlines})")

    max_len = max(len(s) for s in streamlines)
    print(f"  {len(streamlines)} streamlines, longest = {max_len} points")

    # Pacing controls --------------------------------------------------------
    # 1. Percentile truncation: end the animation when most streamlines are
    #    done, so a few long outliers don't drag the movie out.
    if length_percentile is not None and length_percentile < 100:
        lengths = np.array([len(s) for s in streamlines])
        animation_max = int(np.percentile(lengths, length_percentile))
        animation_max = max(2, min(animation_max, max_len))
        print(f"  capping at p{length_percentile} = {animation_max} pts "
              f"(longest streamline is {max_len})")
    else:
        animation_max = max_len

    n_frames = max(1, (animation_max + steps_per_frame - 1) // steps_per_frame)
    if time_exponent != 1.0:
        # At exponent=2 the half-way frame shows only 25% of the growth, so
        # the front of the animation gets disproportionate attention.
        half = ((n_frames / 2) / n_frames) ** time_exponent
        print(f"  time_exponent={time_exponent}: midpoint shows {100*half:.1f}% of growth")
    print(f"  {n_frames} frames @ {fps} fps -> {n_frames / fps:.1f}s movie")

    if output_path is None:
        output_path = os.path.splitext(trk_path)[0] + '_propagation.mp4'

    # ---- Scene ----------------------------------------------------------
    scene = window.Scene()
    scene.background(background)

    if mask_path:
        print(f"  underlay: {mask_path}")
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
        underlay = actor.contour_from_roi(
            mask.astype(np.uint8),
            color=np.array([0.4, 0.4, 0.5]),
            opacity=0.08,
        )
        scene.add(underlay)

    # Camera: bounding box of all streamlines, then back off along -Y so
    # we're looking from the front. Tweak the multipliers if the framing
    # is wrong for your data.
    all_pts = np.vstack(streamlines)
    bbox_min, bbox_max = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    extent = float((bbox_max - bbox_min).max())
    scene.set_camera(
        position=(center[0], center[1] - 2.8 * extent, center[2] + 0.3 * extent),
        focal_point=tuple(center),
        view_up=(0, 0, 1),
    )

    # ---- Build persistent actor (points + colors uploaded once) --------
    line_actor, polydata, point_offsets = _build_persistent_actor(
        streamlines, line_width=line_width,
        use_tubes=use_tubes, tube_radius=tube_radius, tube_sides=tube_sides,
    )
    scene.add(line_actor)
    if use_tubes:
        print(f"  rendering mode: streamtubes (radius={tube_radius}, sides={tube_sides})")
    else:
        print(f"  rendering mode: lines (width={line_width})")

    # ---- Render frames --------------------------------------------------
    # Supersample: render at scale * window_size, then have ffmpeg downscale
    # with Lanczos for SSAA. Stacks with MSAA from multi_samples.
    render_size = (int(round(window_size[0] * supersample)),
                   int(round(window_size[1] * supersample)))
    if supersample != 1.0:
        print(f"  supersampling: rendering at {render_size} -> downscaling to {window_size}")
    if multi_samples:
        print(f"  MSAA: {multi_samples}x")

    tmpdir = tempfile.mkdtemp(prefix='fury_anim_')
    print(f"  rendering frames into {tmpdir}")

    try:
        for frame_idx in range(n_frames):
            # Non-linear time: t = animation_max * (progress)^exponent.
            # exponent=1 -> linear; exponent>1 -> front-loaded (more frames
            # spent on the early growth, the user's preferred pacing).
            progress = ((frame_idx + 1) / n_frames) ** time_exponent
            t = max(2, int(round(progress * animation_max)))

            # Only the cell array changes per frame - points/colors are
            # already uploaded. When use_tubes, the pipeline re-runs the
            # tube filter automatically.
            _update_truncated_lines(polydata, streamlines, point_offsets, t)

            if rotate_deg_per_frame:
                scene.azimuth(rotate_deg_per_frame)

            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.png")
            window.snapshot(scene, fname=frame_path, size=render_size,
                            offscreen=True, multi_samples=multi_samples)

            if (frame_idx + 1) % 20 == 0 or frame_idx + 1 == n_frames:
                print(f"  rendered frame {frame_idx + 1}/{n_frames}")

        # ---- Encode -----------------------------------------------------
        print(f"  encoding mp4 -> {output_path}")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmpdir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',  # broadest player compatibility
            '-crf', str(crf),
            '-preset', preset,
        ]
        # Lanczos downscale for SSAA when supersampling was used. Also
        # ensures dimensions are even (libx264 requires it).
        if supersample != 1.0:
            cmd += ['-vf', f'scale={window_size[0]}:{window_size[1]}:flags=lanczos']
        else:
            # Even-dimension safety for libx264 even without supersampling.
            cmd += ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2']
        cmd.append(output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg stderr (tail):")
            print('\n'.join(result.stderr.splitlines()[-20:]))
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")
        print(f"Saved {output_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    trk_path = CONFIG['trk_path'] or find_most_recent_trk()
    if not trk_path or not os.path.exists(trk_path):
        print("No .trk found. Set CONFIG['trk_path'] to a valid path.")
        sys.exit(1)

    animate(
        trk_path=trk_path,
        output_path=CONFIG['output'],
        mask_path=CONFIG['mask_path'],
        steps_per_frame=CONFIG['steps_per_frame'],
        fps=CONFIG['fps'],
        window_size=CONFIG['window_size'],
        background=CONFIG['background'],
        line_width=CONFIG['line_width'],
        rotate_deg_per_frame=CONFIG['rotate_deg_per_frame'],
        crf=CONFIG['crf'],
        preset=CONFIG['preset'],
        max_streamlines=CONFIG.get('max_streamlines'),
        multi_samples=CONFIG.get('multi_samples', 0),
        supersample=CONFIG.get('supersample', 1.0),
        use_tubes=CONFIG.get('use_tubes', False),
        tube_radius=CONFIG.get('tube_radius', 0.15),
        tube_sides=CONFIG.get('tube_sides', 8),
        time_exponent=CONFIG.get('time_exponent', 1.0),
        length_percentile=CONFIG.get('length_percentile'),
    )
