"""
The lag conventions, pinned.

Three places describe the same offset and they must agree, because the whole
point of --force-lag is to carry a number measured in one of them into the
others:

  * `diagnose_sensors.check_lag` slides with `np.roll(S, lag)` and refits
  * `spectra.section_lag` takes the argmax of a cross-correlation
  * `sparse_sensors.apply_force_lag` is what the jobs actually apply

A sign flip between any two is silent -- every run still completes, and the
"corrected" pairing is twice as wrong as the uncorrected one. This file is the
only thing standing between that and an eight-hour GPU job.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from experiments.april_wake.data_preprocessing import apply_force_lag  # noqa: E402

TRUE_LAG = 25  # the --force-lag that should restore alignment


def _pair(n=2000, lag=TRUE_LAG, seed=0):
    """A field signal and a sensor record misaligned by a known `lag`."""
    rng = np.random.default_rng(seed)
    b = np.cumsum(rng.standard_normal(n))
    b -= b.mean()
    S = np.roll(b, -lag)[None, :]  # S[t] == b[t + lag]
    return b, S


def test_apply_force_lag_restores_alignment():
    b, S = _pair()
    out = apply_force_lag(S.copy(), TRUE_LAG)
    r = np.corrcoef(out[0, TRUE_LAG + 10: -10], b[TRUE_LAG + 10: -10])[0, 1]
    assert r > 0.999, f"corr {r:.4f} -- apply_force_lag has the wrong sign"


def test_wrong_sign_is_worse_than_doing_nothing():
    """A guard on the guard: the test above passes for a no-op too."""
    b, S = _pair()
    ref = np.corrcoef(S[0, 100:-100], b[100:-100])[0, 1]
    bad = apply_force_lag(S.copy(), -TRUE_LAG)
    r = np.corrcoef(bad[0, 100:-100], b[100:-100])[0, 1]
    assert abs(r) < abs(ref), "flipping the sign should make the fit worse"


def test_crosscorr_argmax_is_the_force_lag_to_pass():
    """spectra.py prints its argmax as a --force-lag; that must be literal."""
    b, S = _pair()
    n = len(b)
    Bc = (b - b.mean()) / b.std()
    Sc = (S[0] - S[0].mean()) / S[0].std()
    L = 60
    full = np.correlate(Bc, Sc, mode="full") / n
    mid = len(full) // 2
    lags = np.arange(-L, L + 1)
    got = int(lags[np.argmax(np.abs(full[mid - L: mid + L + 1]))])
    assert got == TRUE_LAG, f"argmax {got:+d}, expected {TRUE_LAG:+d}"


def test_roll_convention_matches_apply_force_lag():
    """diagnose_sensors slides with np.roll; away from the edges they agree."""
    _, S = _pair()
    for lag in (-30, -7, 7, 30):
        a = apply_force_lag(S.copy(), lag)
        b_ = np.roll(S, lag, axis=1)
        k = abs(lag) + 1
        assert np.allclose(a[:, k:-k], b_[:, k:-k]), f"differ at lag {lag:+d}"


def test_zero_is_a_no_op_and_too_large_raises():
    _, S = _pair()
    assert apply_force_lag(S, 0) is S
    with pytest.raises(ValueError, match="longer than the record"):
        apply_force_lag(S, 10_000)


def test_runs_are_shifted_independently():
    """A concatenated record must not leak sensor samples across the seam."""
    rng = np.random.default_rng(1)
    S = rng.standard_normal((2, 400))
    run_id = np.repeat([0, 1], 200)
    out = apply_force_lag(S.copy(), 5, run_id)
    # run 1 starts at column 200; its first 5 columns must be its own first
    # sample, not the tail of run 0
    assert np.allclose(out[:, 200:205], S[:, 200:201]), "leaked across the seam"
    assert np.allclose(out[:, 0:5], S[:, 0:1])
