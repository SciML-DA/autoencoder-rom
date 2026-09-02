"""
The POD algorithms must agree on data that looks like the experiment.

`pod` defaults to a full dense SVD. At the real problem size after the mask fix
-- Q is (31482, 4440) -- that is ~50 s a call, and `diagnose_sensors.check_lag`
refits once per candidate lag, so 161 lags is 2.2 h and job 3936955 was killed
at its walltime doing exactly that. The scan now uses the randomized solver,
which is ~1.3 s.

That is only safe if the answers agree, and whether they agree depends entirely
on the spectrum. On a near-flat spectrum -- 300 comparable singular values --
no rank-r subspace is well determined, the randomized solver returns a
different subspace of equal quality, and a mode-by-mode comparison looks
alarming while the projection error is unchanged. Real PIV data is nothing like
that: the measured energies are 0.213, 0.142, 0.112, ... and the leading modes
are well separated.

So the fixture below uses the measured spectrum, and the assertion is on the
projection error -- the quantity POD is actually for, and the one quoted as the
projection floor.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tools.epod import pod  # noqa: E402

# energy fractions of the leading modes, measured on 4p5d_10ms_yaw_0_0_0
MEASURED = np.array([0.213, 0.142, 0.112, 0.052, 0.047, 0.033, 0.027, 0.024,
                     0.020, 0.018])


@pytest.fixture(scope="module")
def field():
    """A matrix whose spectrum matches the April run's."""
    rng = np.random.default_rng(0)
    m, n, tail = 1200, 600, 90
    sv = np.concatenate([np.sqrt(MEASURED),
                         np.sqrt(0.30) * np.exp(-np.arange(tail) / 20.0) / np.sqrt(tail)])
    U = np.linalg.qr(rng.standard_normal((m, len(sv))))[0]
    V = np.linalg.qr(rng.standard_normal((n, len(sv))))[0]
    return (U * sv) @ V.T


def _projection_error(Psi, Q):
    Qc = Q - Q.mean(1, keepdims=True)
    return 1.0 - (np.linalg.norm(Psi.T @ Qc) ** 2) / (np.linalg.norm(Qc) ** 2)


@pytest.mark.parametrize("r", [8, 16, 64])
@pytest.mark.parametrize("method", ["snapshot", "randomized"])
def test_methods_match_the_dense_svd(field, r, method):
    """Every method must reach the same projection floor as the exact SVD."""
    ref = _projection_error(pod(field, r=r, method="svd", subtract_mean=True)[0], field)
    got = _projection_error(pod(field, r=r, method=method, subtract_mean=True)[0], field)
    assert abs(got - ref) < 1e-4, (
        f"{method} at r={r}: projection floor {got:.8f} vs svd {ref:.8f}"
    )


@pytest.mark.parametrize("method", ["snapshot", "randomized"])
def test_leading_energies_match(field, method):
    """The energy fractions quoted next to every observability number."""
    _, s_ref, _, _ = pod(field, r=16, method="svd", subtract_mean=True)
    _, s_got, _, _ = pod(field, r=16, method=method, subtract_mean=True)
    e_ref = s_ref[:8] ** 2 / (s_ref ** 2).sum()
    e_got = s_got[:8] ** 2 / (s_got ** 2).sum()
    assert np.allclose(e_ref, e_got, atol=1e-4), f"{method}: {e_got} vs {e_ref}"


def test_randomized_is_actually_faster(field):
    """If it ever stops being faster, the reason for using it has gone."""
    import time
    t0 = time.time(); pod(field, r=8, method="svd", subtract_mean=True)
    t_svd = time.time() - t0
    t0 = time.time(); pod(field, r=8, method="randomized", subtract_mean=True)
    t_rand = time.time() - t0
    assert t_rand < t_svd, f"randomized {t_rand:.3f}s vs svd {t_svd:.3f}s"
