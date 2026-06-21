"""
TrackingStats - statistics for a tractography run.

Two ingest hooks the tracker calls during a run:

  * ``record_termination(reason, n)`` - increment the counter for a per-direction
    termination reason AND record the step count the direction had reached
    when it terminated (drives the "length-at-termination by reason" report).
  * ``add_streamline(streamline, labels)`` - called once per accepted
    streamline. Updates length, step-count, turn-angle, PSOCT-fraction
    distributions and bumps the per-voxel streamline-visit count map.

Counter mutations that aren't reason-tagged (``stats['psoct_steps'] += 1``,
``stats['total_steps'] += 1``, etc.) still go through the dict-like
passthrough that ``__getitem__`` / ``__setitem__`` expose.

Reports: 
  * ``summary()``        - JSON-serializable nested dict
  * ``print_summary()``  - human-readable stdout
  * ``plot_histogram()`` - termination-reason bar chart (unchanged)
  * ``plot_geometry_summary()`` - 2x2 figure: length, step-count, turn-angle,
    PSOCT-fraction histograms
  * ``save_visit_map()`` - TDI-style per-voxel streamline-count NIfTI
"""

import numpy as np


def _quartile_dict(values):
    """Five-number + mean summary of a numeric sequence."""
    if values is None or len(values) == 0:
        return {'n': 0, 'mean': None, 'min': None, 'q25': None,
                'median': None, 'q75': None, 'max': None}
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {'n': 0, 'mean': None, 'min': None, 'q25': None,
                'median': None, 'q75': None, 'max': None}
    return {
        'n': int(len(arr)),
        'mean': float(arr.mean()),
        'min': float(arr.min()),
        'q25': float(np.percentile(arr, 25)),
        'median': float(np.percentile(arr, 50)),
        'q75': float(np.percentile(arr, 75)),
        'max': float(arr.max()),
    }


class TrackingStats:
    """All statistics for a tractography run."""

    TERMINATION_REASONS = (
        'terminated_angle',
        'terminated_boundary',
        'terminated_low_f',
        'terminated_max_steps',
        'terminated_no_data',
        'terminated_stop',
    )

    REJECTION_REASONS = (
        'terminated_exclude',
        'rejected_no_target',
    )

    _CORE_KEYS = (
        'total_streamlines',
        'valid_streamlines',
        'total_steps',
        'psoct_steps',
        'bedpostx_steps',
        'psoct_skipped_out_of_plane',
    )

    PSOCT_BUCKETS = ('0%', '0-25%', '25-75%', '75-100%', '100%')

    def __init__(self, voxel_size=None, volume_shape=None, step_size_mm=None):
        self._voxel_size = (np.asarray(voxel_size, dtype=float)
                            if voxel_size is not None else None)
        self._volume_shape = tuple(volume_shape) if volume_shape is not None else None
        self._step_size_mm = float(step_size_mm) if step_size_mm is not None else None

        self._counters = {}
        self._lengths_mm = []
        self._step_counts = []
        self._psoct_fractions = []
        self._mean_turn_angles_rad = []
        self._max_turn_angles_rad = []
        self._all_turn_angles_rad = []
        self._term_steps_by_reason = {r: [] for r in self.TERMINATION_REASONS}
        self._visit_map = None
        self._divergence_map = None

        self.reset()

    def reset(self):
        self._counters = {k: 0 for k in self._CORE_KEYS}
        for k in self.TERMINATION_REASONS + self.REJECTION_REASONS:
            self._counters[k] = 0
        self._lengths_mm.clear()
        self._step_counts.clear()
        self._psoct_fractions.clear()
        self._mean_turn_angles_rad.clear()
        self._max_turn_angles_rad.clear()
        self._all_turn_angles_rad.clear()
        for r in self.TERMINATION_REASONS:
            self._term_steps_by_reason[r].clear()
        
        if self._volume_shape is not None:
            self._visit_map = np.zeros(self._volume_shape, dtype=np.int32)
            self._divergence_map = np.zeros(self._volume_shape, dtype=np.int32)
        else:
            self._visit_map = None
            self._divergence_map = None

    def __getitem__(self, key):
        return self._counters[key]

    def __setitem__(self, key, value):
        self._counters[key] = value

    def __contains__(self, key):
        return key in self._counters

    def __iter__(self):
        return iter(self._counters)

    def keys(self):
        return self._counters.keys()

    def items(self):
        return self._counters.items()

    def values(self):
        return self._counters.values()

    def record_termination(self, reason, step_count_at_termination):
        if reason not in self._counters:
            return
        self._counters[reason] += 1
        if reason in self._term_steps_by_reason:
            self._term_steps_by_reason[reason].append(int(step_count_at_termination))

    def record_divergence(self, pos):
        """Record a voxel where PS-OCT overrode the BedpostX baseline choice."""
        if self._divergence_map is not None:
            idx = np.rint(pos).astype(np.intp)
            sx, sy, sz = self._divergence_map.shape
            if (0 <= idx[0] < sx) and (0 <= idx[1] < sy) and (0 <= idx[2] < sz):
                self._divergence_map[idx[0], idx[1], idx[2]] += 1

    def add_streamline(self, streamline, labels):
        streamline = np.asarray(streamline, dtype=float)
        n = len(streamline)
        if n < 2:
            return

        n_steps = n - 1
        self._step_counts.append(n_steps)

        if self._step_size_mm is not None:
            self._lengths_mm.append(n_steps * self._step_size_mm)
        elif self._voxel_size is not None:
            diffs = np.diff(streamline, axis=0) * self._voxel_size
            self._lengths_mm.append(float(np.linalg.norm(diffs, axis=1).sum()))
        else:
            self._lengths_mm.append(float('nan'))

        if labels is not None and len(labels):
            psoct_count = sum(1 for lab in labels if lab == 'psoct_constrained')
            self._psoct_fractions.append(psoct_count / len(labels))
        else:
            self._psoct_fractions.append(float('nan'))

        if n >= 3:
            v1 = streamline[1:-1] - streamline[:-2]
            v2 = streamline[2:] - streamline[1:-1]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            ok = (n1 > 0) & (n2 > 0)
            if ok.any():
                cos = np.clip(np.einsum('ij,ij->i', v1[ok], v2[ok])
                              / (n1[ok] * n2[ok]), -1.0, 1.0)
                angles = np.arccos(cos)
                self._mean_turn_angles_rad.append(float(angles.mean()))
                self._max_turn_angles_rad.append(float(angles.max()))
                self._all_turn_angles_rad.extend(angles.tolist())
            else:
                self._mean_turn_angles_rad.append(float('nan'))
                self._max_turn_angles_rad.append(float('nan'))
        else:
            self._mean_turn_angles_rad.append(float('nan'))
            self._max_turn_angles_rad.append(float('nan'))

        if self._visit_map is not None:
            idx = np.rint(streamline).astype(np.intp)
            sx, sy, sz = self._visit_map.shape
            ok = (
                (idx[:, 0] >= 0) & (idx[:, 0] < sx)
                & (idx[:, 1] >= 0) & (idx[:, 1] < sy)
                & (idx[:, 2] >= 0) & (idx[:, 2] < sz)
            )
            if ok.any():
                idx_unique = np.unique(idx[ok], axis=0)
                i, j, k = idx_unique.T
                self._visit_map[i, j, k] += 1

    def summary(self):
        c = self._counters
        per_direction = {k: c[k] for k in self.TERMINATION_REASONS}
        per_streamline = {k: c[k] for k in self.REJECTION_REASONS}
        n_dir = sum(per_direction.values())
        return {
            'per_direction': per_direction,
            'per_streamline_rejected': per_streamline,
            'totals': {
                'total_streamlines_attempted': c['total_streamlines'],
                'valid_streamlines': c['valid_streamlines'],
                'total_termination_events': n_dir,
            },
            'geometry': self._geometry_summary(),
            'usage': self._usage_summary(),
            'terminations_by_step_count': {
                reason: _quartile_dict(self._term_steps_by_reason[reason])
                for reason in self.TERMINATION_REASONS
            },
            'spatial_coverage': self._coverage_summary(),
        }

    def _geometry_summary(self):
        mean_deg = (np.degrees(self._mean_turn_angles_rad).tolist()
                    if self._mean_turn_angles_rad else [])
        max_deg = (np.degrees(self._max_turn_angles_rad).tolist()
                   if self._max_turn_angles_rad else [])
        all_deg = (np.degrees(self._all_turn_angles_rad).tolist()
                   if self._all_turn_angles_rad else [])
        return {
            'length_mm': _quartile_dict(self._lengths_mm),
            'step_count': _quartile_dict(self._step_counts),
            'turn_angle_deg_mean_per_streamline': _quartile_dict(mean_deg),
            'turn_angle_deg_max_per_streamline': _quartile_dict(max_deg),
            'turn_angle_deg_all_steps': _quartile_dict(all_deg),
        }

    def _usage_summary(self):
        buckets = {b: 0 for b in self.PSOCT_BUCKETS}
        for f in self._psoct_fractions:
            if f <= 0:
                buckets['0%'] += 1
            elif f < 0.25:
                buckets['0-25%'] += 1
            elif f < 0.75:
                buckets['25-75%'] += 1
            elif f < 1.0:
                buckets['75-100%'] += 1
            else:
                buckets['100%'] += 1
        return {
            'psoct_fraction_per_streamline': _quartile_dict(self._psoct_fractions),
            'psoct_fraction_buckets': buckets,
        }

    def _coverage_summary(self):
        if self._visit_map is None:
            return {'unique_voxels_visited': None,
                    'streamline_visits_total': None,
                    'volume_voxels': None,
                    'fraction_covered': None}
        visited = self._visit_map > 0
        vol = int(np.prod(self._volume_shape))
        unique = int(visited.sum())
        return {
            'unique_voxels_visited': unique,
            'streamline_visits_total': int(self._visit_map.sum()),
            'volume_voxels': vol,
            'fraction_covered': (unique / vol) if vol else None,
        }

    def raw_arrays(self):
        n = len(self._step_counts)
        out = {
            'streamline_id':       np.arange(n, dtype=np.int64),
            'length_mm':           np.asarray(self._lengths_mm,           dtype=float),
            'step_count':          np.asarray(self._step_counts,          dtype=np.int64),
            'psoct_fraction':      np.asarray(self._psoct_fractions,      dtype=float),
            'mean_turn_angle_deg': np.degrees(self._mean_turn_angles_rad) if self._mean_turn_angles_rad else np.array([], dtype=float),
            'max_turn_angle_deg':  np.degrees(self._max_turn_angles_rad)  if self._max_turn_angles_rad  else np.array([], dtype=float),
            'all_turn_angles_deg': np.degrees(self._all_turn_angles_rad)  if self._all_turn_angles_rad  else np.array([], dtype=float),
        }
        for reason in self.TERMINATION_REASONS:
            out[f'term_steps_{reason}'] = np.asarray(
                self._term_steps_by_reason[reason], dtype=np.int64)
        out['visit_map'] = self._visit_map  
        out['divergence_map'] = self._divergence_map 
        return out

    def save_raw_arrays(self, output_path):
        arrays = self.raw_arrays()
        if arrays.get('visit_map') is None:
            arrays.pop('visit_map', None)
        if arrays.get('divergence_map') is None:
            arrays.pop('divergence_map', None)
        np.savez_compressed(output_path, **arrays)
        return output_path

    def print_summary(self, *, has_psoct=False, has_stop_mask=False,
                      has_exclude_mask=False, has_target_mask=False):
        c = self._counters
        print("\n=== Tracking Statistics ===")
        print(f"Total streamlines attempted: {c['total_streamlines']}")
        print(f"Valid streamlines: {c['valid_streamlines']}")
        print(f"Total steps: {c['total_steps']}")
        if has_psoct:
            tot = c['psoct_steps'] + c['bedpostx_steps']
            pct = (100 * c['psoct_steps'] / tot) if tot else 0
            print(f"  PSOCT-constrained steps: {c['psoct_steps']} ({pct:.1f}%)")
            print(f"  BedpostX steps:          {c['bedpostx_steps']}")
            print(f"  PSOCT skipped (out-of-plane voxel): "
                  f"{c['psoct_skipped_out_of_plane']}")
        print("\n--- Termination Reasons (per direction; 2x per streamline) ---")
        print(f"  Angle constraint:   {c['terminated_angle']}")
        print(f"  Brain boundary:     {c['terminated_boundary']}")
        print(f"  Low fiber fraction: {c['terminated_low_f']}")
        print(f"  Max steps reached:  {c['terminated_max_steps']}")
        print("\n--- Streamline Rejections ---")
        if has_stop_mask:
            print(f"  Stop mask (termination): {c['terminated_stop']}")
        if has_exclude_mask:
            print(f"  Exclusion mask (discarded): {c['terminated_exclude']}")
        if has_target_mask:
            print(f"  No target reached: {c['rejected_no_target']}")

        geom = self._geometry_summary()
        if geom['length_mm']['n'] > 0:
            print("\n--- Streamline Geometry ---")
            self._print_quartiles("  Length (mm)        ", geom['length_mm'], unit='mm')
            self._print_quartiles("  Step count         ", geom['step_count'])
            self._print_quartiles("  Mean turn angle    ",
                                  geom['turn_angle_deg_mean_per_streamline'], unit='deg')
            self._print_quartiles("  Max  turn angle    ",
                                  geom['turn_angle_deg_max_per_streamline'], unit='deg')

        if has_psoct and self._psoct_fractions:
            usage = self._usage_summary()
            print("\n--- PSOCT Usage per Streamline ---")
            self._print_quartiles("  PSOCT fraction     ",
                                  usage['psoct_fraction_per_streamline'])
            print("  Bucketed:")
            for b in self.PSOCT_BUCKETS:
                print(f"    {b:>8}: {usage['psoct_fraction_buckets'][b]}")

        any_term = any(self._term_steps_by_reason[r] for r in self.TERMINATION_REASONS)
        if any_term:
            print("\n--- Step Count at Termination (by reason) ---")
            for r in self.TERMINATION_REASONS:
                steps = self._term_steps_by_reason[r]
                if steps:
                    arr = np.asarray(steps)
                    print(f"  {r:<22} n={len(arr):<5}  "
                          f"median={int(np.median(arr)):>4}  "
                          f"mean={arr.mean():>6.1f}  "
                          f"max={int(arr.max()):>4}")

        cov = self._coverage_summary()
        if cov['unique_voxels_visited'] is not None:
            frac = cov['fraction_covered'] or 0
            print("\n--- Spatial Coverage ---")
            print(f"  Unique voxels visited:    {cov['unique_voxels_visited']:>10}  "
                  f"({100 * frac:.2f}% of volume)")
            print(f"  Streamline-voxel visits:  {cov['streamline_visits_total']:>10}")

    @staticmethod
    def _print_quartiles(label, q, unit=''):
        if q['n'] == 0:
            return
        u = f" {unit}" if unit else ''
        print(f"{label} n={q['n']:<6} "
              f"mean={q['mean']:>7.2f}{u}  "
              f"min={q['min']:>7.2f}{u}  "
              f"q25={q['q25']:>7.2f}{u}  "
              f"median={q['median']:>7.2f}{u}  "
              f"q75={q['q75']:>7.2f}{u}  "
              f"max={q['max']:>7.2f}{u}")

    def plot_histogram(self, output_path, title=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        summary = self.summary()
        labels, values, colors = [], [], []
        nice = {
            'terminated_angle':       ('angle > threshold',  '#4C78A8'),
            'terminated_boundary':    ('left volume',        '#4C78A8'),
            'terminated_low_f':       ('low fibre fraction', '#4C78A8'),
            'terminated_max_steps':   ('hit max_steps',      '#4C78A8'),
            'terminated_no_data':     ('no fibre data',      '#4C78A8'),
            'terminated_stop':        ('hit stop mask',      '#54A24B'),
            'terminated_exclude':     ('hit exclude (dropped)', '#E45756'),
            'rejected_no_target':     ('missed target (dropped)', '#E45756'),
        }
        for k, v in summary['per_direction'].items():
            lab, col = nice[k]
            labels.append(lab); values.append(v); colors.append(col)
        for k, v in summary['per_streamline_rejected'].items():
            lab, col = nice[k]
            labels.append(lab); values.append(v); colors.append(col)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel('count')
        t = title or 'Termination reasons'
        totals = summary['totals']
        subtitle = (f"{totals['valid_streamlines']} valid / "
                    f"{totals['total_streamlines_attempted']} attempted  |  "
                    f"{totals['total_termination_events']} per-direction events")
        ax.set_title(f"{t}\n{subtitle}", fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        for b, v in zip(bars, values):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v}", ha='center', va='bottom', fontsize=9)
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def plot_geometry_summary(self, output_path, title=None):
        if not self._lengths_mm and not self._step_counts:
            return None

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle(title or 'Streamline geometry & PSOCT usage', fontsize=12)

        ax = axes[0, 0]
        if self._lengths_mm:
            ax.hist(self._lengths_mm, bins=40, color='#4C78A8')
            ax.axvline(np.median(self._lengths_mm), color='k',
                       linestyle='--', linewidth=1, label='median')
            ax.legend()
        ax.set_title('Length')
        ax.set_xlabel('Streamline length (mm)')
        ax.set_ylabel('count')

        ax = axes[0, 1]
        if self._step_counts:
            ax.hist(self._step_counts, bins=40, color='#4C78A8')
        ax.set_title('Step count')
        ax.set_xlabel('Steps per streamline')
        ax.set_ylabel('count')

        ax = axes[1, 0]
        if self._all_turn_angles_rad:
            deg = np.degrees(self._all_turn_angles_rad)
            ax.hist(deg, bins=60, color='#4C78A8')
        ax.set_title('Turn angles (all steps)')
        ax.set_xlabel('Step-to-step turn angle (deg)')
        ax.set_ylabel('count')

        ax = axes[1, 1]
        if self._psoct_fractions:
            ax.hist(self._psoct_fractions, bins=20,
                    range=(0, 1), color='#54A24B')
        ax.set_title('PSOCT usage')
        ax.set_xlabel('PSOCT step fraction per streamline')
        ax.set_ylabel('count')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def save_visit_map(self, output_path, affine):
        if self._visit_map is None:
            return None
        import nibabel as nib
        img = nib.Nifti1Image(self._visit_map, affine)
        nib.save(img, output_path)
        return output_path

    def save_divergence_map(self, output_path, affine):
        """Save per-voxel divergence-count map as a NIfTI."""
        if self._divergence_map is None:
            return None
        import nibabel as nib
        img = nib.Nifti1Image(self._divergence_map, affine)
        nib.save(img, output_path)
        return output_path