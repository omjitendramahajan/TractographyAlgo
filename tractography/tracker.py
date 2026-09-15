"""
PSOCTPriorityTracker - Microscopy-constrained probabilistic tractography.

Where usable PS-OCT data is available, fibre populations are scored using
in-plane microscopy alignment and 3D trajectory alignment, weighted by
(1 - alpha) and alpha respectively. The selected population supplies a
BEDPOSTX posterior orientation sample; microscopy does not replace the
sampled 3D direction. At a seed without a previous heading, population
selection uses microscopy alignment alone.

If the microscopy constraint cannot provide a direction, tracking falls
back to diffusion-only selection based on the previous heading, then to
unconditional posterior sampling if needed. With psoct_data=None, the
same diffusion-only path is used. XTRACT-style target/exclude/stop masks
are handled inline.
"""

import math

import numpy as np

from .stats import TrackingStats


class PSOCTPriorityTracker:
    """
    Probabilistic tracking with PS-OCT-constrained fibre-population selection.

    Alpha weights trajectory alignment against in-plane microscopy alignment
    when a previous heading exists. The score blends alignment evidence, not
    orientation vectors. A 3D direction is sampled from the selected BEDPOSTX
    population. Missing or unusable microscopy triggers diffusion-only fallback.

    Supports XTRACT-style masks (inline):
      - exclude_mask: any point entering => entire streamline discarded
      - stop_mask:    streamline terminates at first point in mask (kept)
      - target_mask:  streamline must intersect to be retained
    """

    def __init__(self, bedpostx_data, psoct_data=None,
                 step_size=0.5, max_steps=2000,
                 min_f_threshold=0.05, angle_threshold=60,
                 psoct_inplane_threshold=0.3, alpha=0.5,
                 target_mask=None, exclude_mask=None, stop_mask=None):
        self.bedpostx = bedpostx_data
        self.psoct = psoct_data
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_f_threshold = min_f_threshold
        self.max_angle = np.deg2rad(angle_threshold)
        self.cos_max_angle = math.cos(self.max_angle)
        self.psoct_inplane_threshold = psoct_inplane_threshold
        # Blend coefficient for the hybrid population score (see
        # _psoct_constrained_step): alpha=1 -> trajectory-only (baseline rule),
        # alpha=0 -> PS-OCT microscopy alignment only.
        self.alpha = alpha

        self.target_mask = target_mask
        self.exclude_mask = exclude_mask
        self.stop_mask = stop_mask

        self.step_vox = step_size / self.bedpostx.voxel_size

        self.stats = TrackingStats(
            voxel_size=self.bedpostx.voxel_size,
            volume_shape=self.bedpostx.shape,
            step_size_mm=step_size,
        )

    def reset_stats(self):
        self.stats.reset()

    def termination_summary(self):
        return self.stats.summary()

    def plot_termination_histogram(self, output_path, title=None):
        return self.stats.plot_histogram(output_path, title=title)

    def print_stats(self):
        self.stats.print_summary(
            has_psoct=self.psoct is not None,
            has_stop_mask=self.stop_mask is not None,
            has_exclude_mask=self.exclude_mask is not None,
            has_target_mask=self.target_mask is not None,
        )

    def track(self, seed):
        self.stats['total_streamlines'] += 1

        forward, fwd_labels, fwd_excluded = self._track_one_direction(seed, forward=True)
        backward, bwd_labels, bwd_excluded = self._track_one_direction(seed, forward=False)

        if fwd_excluded or bwd_excluded:
            self.stats['terminated_exclude'] += 1
            return None, None

        if forward is None and backward is None:
            return None, None
        elif forward is None:
            streamline, labels = backward, bwd_labels
        elif backward is None:
            streamline, labels = forward, fwd_labels
        else:
            streamline = np.vstack([np.flipud(backward), forward[1:]])
            labels = np.concatenate([bwd_labels[::-1], fwd_labels[1:]])

        if self.target_mask is not None and not self._streamline_hits(streamline, self.target_mask):
            self.stats['rejected_no_target'] += 1
            return None, None

        self.stats['valid_streamlines'] += 1
        self.stats.add_streamline(streamline, labels)
        return streamline, labels

    def _streamline_hits(self, streamline, mask):
        idx = np.rint(streamline).astype(np.intp)
        sx, sy, sz = self.bedpostx.shape
        ok = (
            (idx[:, 0] >= 0) & (idx[:, 0] < sx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < sy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < sz)
        )
        if not ok.any():
            return False
        i, j, k = idx[ok].T
        return bool(mask[i, j, k].any())

    def _track_one_direction(self, seed, forward=True):
        pos = np.array(seed, dtype=float)

        direction, source = self._get_direction(pos, prev_direction=None)
        if direction is None:
            return None, None, False

        if not forward:
            direction = -direction

        cap = self.max_steps + 1
        streamline = np.empty((cap, 3), dtype=float)
        labels = np.empty(cap, dtype=object)
        streamline[0] = pos
        labels[0] = source
        n = 1

        for step_idx in range(self.max_steps):
            if step_idx == 0:
                new_direction = direction
            else:
                new_direction, source = self._get_direction(pos, prev_direction=direction)

                if new_direction is None:
                    self.stats.record_termination('terminated_no_data', n - 1)
                    break

                cos_angle = abs(float(np.dot(direction, new_direction)))
                if cos_angle < self.cos_max_angle:
                    self.stats.record_termination('terminated_angle', n - 1)
                    break

            if np.dot(new_direction, direction) < 0:
                new_direction = -new_direction

            new_pos = pos + self.step_vox * new_direction

            if not self.bedpostx.is_valid_position(new_pos):
                self.stats.record_termination('terminated_boundary', n - 1)
                break

            if self.exclude_mask is not None and self.bedpostx.position_in_mask(new_pos, self.exclude_mask):
                return None, None, True

            if self.stop_mask is not None and self.bedpostx.position_in_mask(new_pos, self.stop_mask):
                streamline[n] = new_pos
                labels[n] = source
                n += 1
                self.stats.record_termination('terminated_stop', n - 1)
                break

            if source == 'bedpostx':
                max_f = self.bedpostx.get_max_fiber_fraction(new_pos)
                if max_f < self.min_f_threshold:
                    self.stats.record_termination('terminated_low_f', n - 1)
                    break

            streamline[n] = new_pos
            labels[n] = source
            n += 1
            self.stats['total_steps'] += 1
            pos = new_pos
            direction = new_direction
        else:
            self.stats.record_termination('terminated_max_steps', n - 1)

        if n < 2:
            return None, None, False

        return streamline[:n].copy(), labels[:n].copy(), False

    def _get_direction(self, pos, prev_direction=None):
        # 1. Try PS-OCT step first
        if self.psoct is not None:
            constrained_dir, psoct_pop = self._psoct_constrained_step(pos, prev_direction=prev_direction)
            
            if constrained_dir is not None:
                self.stats['psoct_steps'] += 1
                
                # --- DIVERGENCE LOGGING ---
                if prev_direction is not None:
                    _, bpx_pop = self._bedpostx_directional_step(pos, prev_direction)
                    if bpx_pop is not None and psoct_pop != bpx_pop:
                        # Log it directly to the stats object
                        self.stats.record_divergence(pos)
                # --------------------------
                
                return constrained_dir, 'psoct_constrained'

        # 2. Fallback to standard BedpostX directional step
        if prev_direction is not None:
            directional, _ = self._bedpostx_directional_step(pos, prev_direction)
            if directional is not None and np.linalg.norm(directional) > 0:
                self.stats['bedpostx_steps'] += 1
                return directional, 'bedpostx'

        # 3. Fallback to unconditional BedpostX step (initial seed draw)
        bedpostx_orientation = self.bedpostx.sample_orientation_probabilistic(pos)
        if bedpostx_orientation is not None and np.linalg.norm(bedpostx_orientation) > 0:
            self.stats['bedpostx_steps'] += 1
            return bedpostx_orientation, 'bedpostx'

        return None, None

    def _bedpostx_directional_step(self, pos, prev_direction):
        mean_dyads = self.bedpostx.get_mean_dyads(pos)
        if not mean_dyads:
            return None, None

        prev = np.asarray(prev_direction, dtype=float)
        prev_norm = float(np.linalg.norm(prev))
        if prev_norm == 0:
            return None, None
        prev = prev / prev_norm

        best_pop = -1
        best_score = -1.0
        for i, d in enumerate(mean_dyads):
            mag = float(np.linalg.norm(d))
            if mag == 0:
                continue
            if self.bedpostx.get_fiber_fraction(pos, i) < self.min_f_threshold:
                continue
            score = abs(float(np.dot(d / mag, prev)))
            if score > best_score:
                best_score = score
                best_pop = i

        if best_pop < 0:
            return None, None

        return self.bedpostx.sample_from_population(pos, best_pop), best_pop

    def _psoct_constrained_step(self, pos, prev_direction=None):
        psoct_info = self.psoct.get_orientation(pos, return_normal=True)
        if psoct_info is None:
            return None, None

        psoct_orientation, slide_normal = psoct_info
        if np.linalg.norm(psoct_orientation) == 0:
            return None, None

        mean_dyads = self.bedpostx.get_mean_dyads(pos)
        if not mean_dyads:
            return None, None

        EPS = 0.1

        n = np.asarray(slide_normal, dtype=float)
        n_norm = np.linalg.norm(n)

        if n_norm > 0:
            n = n / n_norm

            def project(v):
                return v - np.dot(v, n) * n

            psoct_proj = project(np.asarray(psoct_orientation, dtype=float))
            psoct_mag = np.linalg.norm(psoct_proj)
            if psoct_mag < EPS:
                return None, None
            psoct_cmp = psoct_proj / psoct_mag

            dyads_proj = [project(d) for d in mean_dyads]

            f_weights = [self.bedpostx.get_fiber_fraction(pos, i)
                         for i in range(len(dyads_proj))]
            f_total = sum(f_weights)
            if f_total > 0:
                weighted_inplane = sum(
                    f * float(np.linalg.norm(d))
                    for f, d in zip(f_weights, dyads_proj)
                ) / f_total
                if weighted_inplane < self.psoct_inplane_threshold:
                    self.stats['psoct_skipped_out_of_plane'] += 1
                    return None, None
        else:
            psoct_cmp = np.asarray(psoct_orientation, dtype=float)
            dyads_proj = list(mean_dyads)

        if prev_direction is not None:
            prev = np.asarray(prev_direction, dtype=float)
            prev_norm = float(np.linalg.norm(prev))
            prev_unit = prev / prev_norm if prev_norm > 0 else None
        else:
            prev_unit = None

        # Blend coefficient alpha (self.alpha) weights the BedpostX trajectory
        # term against the PS-OCT microscopy term in the score below:
        #   alpha = 1 -> trajectory only (== BedpostX-only baseline rule)
        #   alpha = 0 -> microscopy (PS-OCT) alignment only
        alignments = []
        for i, d_proj in enumerate(dyads_proj):
            mag = float(np.linalg.norm(d_proj))
            if mag < EPS:
                alignments.append(0.0)
                continue
            if self.bedpostx.get_fiber_fraction(pos, i) < self.min_f_threshold:
                alignments.append(0.0)
                continue
            psoct_align = abs(float(np.dot(psoct_cmp, d_proj / mag)))
            if prev_unit is not None:
                d_full = np.asarray(mean_dyads[i], dtype=float)
                d_full_mag = float(np.linalg.norm(d_full))
                traj_align = abs(float(np.dot(prev_unit, d_full / d_full_mag))) if d_full_mag > 0 else 0.0
                score = (1.0 - self.alpha) * psoct_align + self.alpha * traj_align
            else:
                score = psoct_align
            alignments.append(score)

        if max(alignments) == 0.0:
            return None, None

        best_pop = int(np.argmax(alignments))

        return self.bedpostx.sample_from_population(pos, best_pop), best_pop

    def step(self, pos, prev_direction):
        """Backwards-compatible wrapper used by older test scripts."""
        return self._get_direction(pos, prev_direction=prev_direction)