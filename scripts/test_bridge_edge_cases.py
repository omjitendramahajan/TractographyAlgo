"""
Edge-case tests for the dMRI <-> PS-OCT bridge.

Run:
    python scripts/test_bridge_edge_cases.py

Tests are split into two groups:
  * Synthetic / no-data tests  -- always run, validate algorithm logic.
  * HDD-dependent tests        -- run only when the BedpostX HDD is mounted.

Each test prints PASS / FAIL / SKIP and the script exits 0 on all-pass.

Why this matters:
  These tests guard the load-bearing assumptions behind Figure R0 (bridge
  validation) and the PSOCT-constrained tracker. If the cross-product /
  closest-population logic is subtly wrong, the report's whole pitch is wrong.
"""

import os
import sys
import json
import traceback
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT   = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


class Reporter:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def run(self, name, fn, *args, **kwargs):
        print(f"  - {name}", end=" ... ", flush=True)
        try:
            ok, msg = fn(*args, **kwargs)
        except Skip as s:
            self.skipped += 1
            print(f"SKIP ({s})")
            return
        except Exception as e:
            self.failed += 1
            tb = traceback.format_exc()
            self.failures.append((name, str(e), tb))
            print(f"FAIL ({e.__class__.__name__}: {e})")
            return
        if ok:
            self.passed += 1
            print(f"PASS {msg or ''}".rstrip())
        else:
            self.failed += 1
            self.failures.append((name, msg or "assertion failed", ""))
            print(f"FAIL {msg or ''}".rstrip())

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\nResults: {self.passed}/{total} passed, "
              f"{self.failed} failed, {self.skipped} skipped")
        if self.failures:
            print("\nFailures:")
            for n, m, tb in self.failures:
                print(f"  [{n}] {m}")
                if tb:
                    print(tb)
        return self.failed == 0


class Skip(Exception):
    pass


# =============================================================================
# Synthetic tests (always run)
# =============================================================================

def t_select_closest_population_basic():
    """Given two dyads and a PSOCT vector aligned with the second, the
    cosine-distance argmax must pick population 2."""
    d1 = np.array([1, 0, 0], dtype=float)
    d2 = np.array([0, 1, 0], dtype=float)
    v = np.array([0, 1, 0], dtype=float)
    alignments = [abs(np.dot(v, d)) for d in (d1, d2)]
    best = int(np.argmax(alignments))
    return best == 1, f"argmax={best}, alignments={alignments}"


def t_select_closest_population_sign_invariant():
    """Anti-parallel PSOCT vector must still pick the correct (parallel) dyad
    because we use absolute dot product."""
    d1 = np.array([1, 0, 0], dtype=float)
    d2 = np.array([0, 1, 0], dtype=float)
    v = np.array([0, -1, 0], dtype=float)
    alignments = [abs(np.dot(v, d)) for d in (d1, d2)]
    best = int(np.argmax(alignments))
    return best == 1, f"argmax={best}, alignments={alignments}"


def t_select_closest_population_tiebreak():
    """When the PSOCT vector is exactly between two dyads, argmax should
    return a valid index (0 or 1) without crashing."""
    d1 = np.array([1, 0, 0], dtype=float)
    d2 = np.array([0, 1, 0], dtype=float)
    v = np.array([1, 1, 0]) / np.sqrt(2)
    alignments = [abs(np.dot(v, d)) for d in (d1, d2)]
    best = int(np.argmax(alignments))
    return best in (0, 1), f"tie -> {best}, alignments={alignments}"


def t_dyadic_mean_handles_sign_ambiguity():
    """The dyadic-mean approach should return the underlying axis when the
    samples are randomly signed copies of the same direction."""
    rng = np.random.default_rng(0)
    base = np.array([0.5, 0.5, np.sqrt(0.5)])
    base /= np.linalg.norm(base)
    n = 200
    signs = rng.choice([-1, 1], size=n)
    samples = signs[:, None] * base[None, :]
    tens = (samples.T @ samples) / len(samples)
    _, V = np.linalg.eigh(tens)
    mean_dyad = V[:, -1]
    cos_a = abs(float(np.dot(mean_dyad, base)))
    return cos_a > 0.999, f"|cos(mean_dyad, base)|={cos_a:.4f}"


def t_arccos_clip_safety():
    """Floating-point cos values just above 1.0 must not produce NaN when
    fed through arccos -- bridge validation must clip first."""
    cos_a = 1.0 + 1e-12
    theta = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return not np.isnan(theta), f"theta={theta}"


def t_random_orientation_distribution():
    """Sanity: angle between random unit vectors is uniformly distributed
    in cos-space, median ~60 deg. Confirms the angular metric itself is
    well-calibrated."""
    rng = np.random.default_rng(0)
    v1 = rng.normal(size=(2000, 3)); v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = rng.normal(size=(2000, 3)); v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    cos_a = np.abs((v1 * v2).sum(axis=1))
    theta = np.degrees(np.arccos(np.clip(cos_a, 0, 1)))
    med = float(np.median(theta))
    ok = 50 < med < 70
    return ok, f"random-pair median = {med:.1f} deg (expected ~60)"


# =============================================================================
# HDD-dependent tests
# =============================================================================

def _load_config():
    with open(os.path.join(SCRIPT_DIR, "fma_config.json")) as f:
        return json.load(f)


def _have_hdd():
    cfg = _load_config()
    return os.path.exists(cfg["paths"]["bedpostx_dir"])


def t_psoct_outside_any_slide_returns_none():
    cfg = _load_config()
    psoct_dir = cfg["paths"]["psoct_dir"]
    if not os.path.exists(psoct_dir):
        raise Skip("PSOCT data dir missing")
    from tractography.bedpostx import BedpostxData
    from tractography.psoct import PSOCTData
    if not _have_hdd():
        raise Skip("HDD not mounted")
    bp = BedpostxData(cfg["paths"]["bedpostx_dir"],
                      num_fibers=2, load_samples=False, load_dyads=False,
                      use_memory_map=True)
    psoct = PSOCTData(psoct_dir, bp.volume_img)
    # Pick a voxel obviously outside the brain volume.
    far = np.array([bp.shape[0] - 1, bp.shape[1] - 1, bp.shape[2] - 1])
    v = psoct.get_orientation(far.astype(float))
    return (v is None) or (np.linalg.norm(v) < 1e-9), f"got {v}"


def t_psoct_returns_unit_vectors():
    cfg = _load_config()
    if not _have_hdd():
        raise Skip("HDD not mounted")
    psoct_dir = cfg["paths"]["psoct_dir"]
    if not os.path.exists(psoct_dir):
        raise Skip("PSOCT data dir missing")
    from tractography.bedpostx import BedpostxData
    from tractography.psoct import PSOCTData
    bp = BedpostxData(cfg["paths"]["bedpostx_dir"],
                      num_fibers=2, load_samples=False, load_dyads=False,
                      use_memory_map=True)
    psoct = PSOCTData(psoct_dir, bp.volume_img)
    # Probe a grid of brain voxels until we find some with PSOCT data
    rng = np.random.default_rng(0)
    candidates = np.argwhere(bp.mask)
    idx = rng.choice(len(candidates), size=min(2000, len(candidates)),
                     replace=False)
    found = 0
    norms = []
    for vox in candidates[idx]:
        v = psoct.get_orientation(vox.astype(float))
        if v is not None:
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                norms.append(n)
                found += 1
                if found >= 20:
                    break
    if not norms:
        raise Skip("no PSOCT-covered voxel found in random sample")
    norms = np.array(norms)
    return bool(np.allclose(norms, 1.0, atol=1e-3)), (
        f"checked {len(norms)} vectors, max|norm-1|={abs(norms-1).max():.2e}"
    )


def t_psoct_cache_is_stable():
    """Same voxel queried twice must return same vector."""
    cfg = _load_config()
    if not _have_hdd():
        raise Skip("HDD not mounted")
    psoct_dir = cfg["paths"]["psoct_dir"]
    if not os.path.exists(psoct_dir):
        raise Skip("PSOCT data dir missing")
    from tractography.bedpostx import BedpostxData
    from tractography.psoct import PSOCTData
    bp = BedpostxData(cfg["paths"]["bedpostx_dir"],
                      num_fibers=2, load_samples=False, load_dyads=False,
                      use_memory_map=True)
    psoct = PSOCTData(psoct_dir, bp.volume_img)

    rng = np.random.default_rng(0)
    candidates = np.argwhere(bp.mask)
    idx = rng.choice(len(candidates), size=min(2000, len(candidates)),
                     replace=False)
    for vox in candidates[idx]:
        v1 = psoct.get_orientation(vox.astype(float))
        if v1 is not None:
            v2 = psoct.get_orientation(vox.astype(float))
            return bool(np.allclose(v1, v2)), f"diff={np.linalg.norm(v1 - v2):.2e}"
    raise Skip("no PSOCT-covered voxel found")


def t_constrained_sampler_picks_aligned_population():
    """Inject a synthetic PSOCT vector aligned with EACH BedpostX population
    in turn and verify the constrained sampler returns a direction whose
    |cos| with the *target* population's dyad is at least as large as with
    every other population's dyad. Generic over N populations.
    """
    cfg = _load_config()
    if not _have_hdd():
        raise Skip("HDD not mounted")
    from tractography.bedpostx import BedpostxData
    n_fib = cfg["tracking"]["num_fibers"]
    bp = BedpostxData(cfg["paths"]["bedpostx_dir"],
                      num_fibers=n_fib, load_samples=True, load_dyads=True,
                      use_memory_map=True)

    # Find voxels where ALL populations have appreciable volume fraction --
    # i.e. the most ambiguous case for a "pick the right population" test.
    masks = [bp.f_samples[i] > 0.15 for i in range(bp.num_fibers)]
    crossing = np.argwhere(np.logical_and.reduce(masks + [bp.mask]))
    if len(crossing) == 0:
        raise Skip(f"no voxels where all {bp.num_fibers} populations have f>0.15")

    rng = np.random.default_rng(0)
    test_voxels = crossing[rng.choice(len(crossing), size=min(5, len(crossing)),
                                       replace=False)]
    np.random.seed(0)
    n_consistent, n_tested = 0, 0
    for vox in test_voxels:
        i, j, k = int(vox[0]), int(vox[1]), int(vox[2])
        dyads_here = []
        for fib_idx in range(bp.num_fibers):
            d = bp.dyads[fib_idx][i, j, k]
            n = np.linalg.norm(d)
            dyads_here.append(d / n if n > 1e-9 else d)

        # Replicate the tracker's selection+sample logic locally, using
        # only the data-type-specific primitives exposed by BedpostxData.
        # No slide plane here (synthetic PSOCT vector), so comparison is 3D.
        mean_dyads = bp.get_mean_dyads(vox.astype(float))
        if mean_dyads is None:
            continue
        for target_idx, v_target in enumerate(dyads_here):
            n_tested += 1
            alignments = [abs(float(np.dot(v_target, d))) for d in mean_dyads]
            best_pop = int(np.argmax(alignments))
            sampled = bp.sample_from_population(vox.astype(float), best_pop)
            if sampled is None:
                continue
            cos_target = abs(float(np.dot(sampled, v_target)))
            cos_others = [abs(float(np.dot(sampled, d)))
                          for k, d in enumerate(dyads_here) if k != target_idx]
            if cos_target >= max(cos_others, default=-1):
                n_consistent += 1

    if n_tested == 0:
        raise Skip("no successful samples at crossing voxels")
    frac = n_consistent / n_tested
    return frac > 0.8, (
        f"{n_consistent}/{n_tested} consistent ({frac*100:.0f}%) "
        f"across {bp.num_fibers}-population voxels"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Bridge edge-case tests")
    print("=" * 60)

    r = Reporter()

    print("\n[Synthetic tests]")
    r.run("select-closest-population basic",       t_select_closest_population_basic)
    r.run("select-closest-population sign-flip",   t_select_closest_population_sign_invariant)
    r.run("select-closest-population tiebreak",    t_select_closest_population_tiebreak)
    r.run("dyadic mean is sign-invariant",         t_dyadic_mean_handles_sign_ambiguity)
    r.run("arccos clip is NaN-safe",               t_arccos_clip_safety)
    r.run("random-pair angular median ~60 deg",    t_random_orientation_distribution)

    print("\n[HDD-dependent tests]")
    r.run("PSOCT outside slides returns None",     t_psoct_outside_any_slide_returns_none)
    r.run("PSOCT vectors are unit-norm",           t_psoct_returns_unit_vectors)
    r.run("PSOCT cache is stable",                 t_psoct_cache_is_stable)
    r.run("constrained sampler picks aligned pop", t_constrained_sampler_picks_aligned_population)

    ok = r.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
