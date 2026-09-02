"""
The mask rule, pinned.

`build_case` used to keep a grid point only if it carried a valid vector in
every one of the 6085 snapshots. Spurious PIV vectors do not appear at random --
they cluster in shear layers and the wake core -- so that rule preferentially
deleted the region the load cells measure, kept the freestream, and then the
reconstruction was scored on the part of the field nothing can see. It removed
9920 of 15741 points to exclude a body of at most 112.

The reference notebook has no mask rule at all: `nan_to_num` over the whole
field. The default here now keeps the field too, and fills the gaps rather than
zeroing them.

These tests use wandering dropout, because a mask rule that only ever sees a
static body looks correct.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import datasets.wake_experiment as we  # noqa: E402
from test_sparse_sensors import _write_fixture  # noqa: E402


def _fixture_with_wandering_holes(root, n_t=200, per_frame=3, seed=0):
    """An RDS-shaped run whose bad vectors move between frames."""
    run, _ = _write_fixture(root, n_t=n_t)
    files = sorted(glob.glob(os.path.join(root, run, "piv_snapshots_highres", "*.npz")))
    rng = np.random.default_rng(seed)
    for f in files:
        d = dict(np.load(f))
        u, v = d["u"].copy(), d["v"].copy()
        ny, nx = u.shape
        for _ in range(per_frame):
            j, i = int(rng.integers(ny)), int(rng.integers(nx))
            u[j, i] = np.nan
            v[j, i] = np.nan
        np.savez(f, u=u, v=v)
    return run


@pytest.fixture(scope="module")
def wandering():
    root = tempfile.mkdtemp()
    try:
        yield root, _fixture_with_wandering_holes(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_keeps_the_whole_field(wandering):
    """The default must not delete points, however they drop out."""
    root, run = wandering
    c = we.build_case(run, root=root, verbose=False)
    assert c.fluid_mask.all(), (
        f"default dropped {(~c.fluid_mask).sum()} of {c.fluid_mask.size} points"
    )
    assert np.isfinite(c.flat()).all(), "kept points still carry NaN"


def test_strict_rule_is_the_disaster_it_looks_like(wandering):
    """Documents what mask_tol=0 costs, so nobody restores it by accident."""
    root, run = wandering
    strict = we.build_case(run, root=root, mask_tol=0.0, verbose=False)
    keep_all = we.build_case(run, root=root, mask_tol=1.0, verbose=False)
    frac = strict.fluid_mask.sum() / keep_all.fluid_mask.sum()
    assert frac < 0.5, (
        f"the fixture no longer reproduces the failure (kept {frac:.0%}); "
        "make the dropout wander more"
    )


def test_invalid_frac_is_recorded_before_masking(wandering):
    """The audit needs the pre-mask dropout; X is overwritten downstream."""
    root, run = wandering
    c = we.build_case(run, root=root, verbose=False)
    assert c.invalid_frac is not None
    assert c.invalid_frac.shape == c.fluid_mask.shape
    assert (c.invalid_frac > 0).any(), "fixture has no dropout to record"
    assert c.invalid_frac.max() <= 1.0 and c.invalid_frac.min() >= 0.0


def test_zero_fill_matches_nan_to_num(wandering):
    """mask_fill='zero' is exactly the notebook's nan_to_num."""
    root, run = wandering
    c = we.build_case(run, root=root, mask_tol=1.0, mask_fill="zero", verbose=False)
    raw = we.load_run(run, root=root)
    assert np.allclose(c.X, np.nan_to_num(raw), equal_nan=False)


def test_interp_beats_zero_on_a_smooth_signal():
    """Why interp is the default: a zero is a measured value to an SVD."""
    t = np.arange(400)
    truth = np.sin(2 * np.pi * t / 50)
    X = np.tile(truth, (1, 1, 1, 1)).reshape(1, 400, 1, 1).copy()
    X[0, 100:104, 0, 0] = np.nan
    fluid = np.ones((1, 1), bool)
    got = we._fill_dropouts(X, fluid, "interp")[0, :, 0, 0]
    zero = we._fill_dropouts(X, fluid, "zero")[0, :, 0, 0]
    e_i = np.abs(got[100:104] - truth[100:104]).max()
    e_z = np.abs(zero[100:104] - truth[100:104]).max()
    assert e_i < e_z, f"interp {e_i:.3f} should beat zero {e_z:.3f}"


def test_always_invalid_points_carry_no_fluctuation(wandering):
    """Keeping the body is harmless: constant fill means zero fluctuation."""
    root, run = wandering
    c = we.build_case(run, root=root, verbose=False)
    body = c.invalid_frac >= 1.0
    if not body.any():
        pytest.skip("fixture has no always-invalid point")
    X = c.X[:, :, body]
    assert np.allclose(X.std(axis=1), 0.0), "body points vary; they should not"
