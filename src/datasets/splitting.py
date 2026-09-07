# pyright: strict
"""
splitting.py
============

Taking a snapshot set into train/val/test. The diagnostics say whether
the set is good. Every function is pure numpy and takes raw
(Nu, Nt, Nx, Ny) arrays, so it imports without torch or jax.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "split_indices",
    "decorrelation_lag",
    "split_diagnostics",
    "linear_span_ceiling",
    "prepare_split",
    "LEAK_THRESHOLD",
    "SHIFT_THRESHOLD",
]

#: (Nu, Nt, Nx, Ny) snapshot block, NaN at solid-body points
Snapshots = npt.NDArray[np.floating[Any]]
#: (Nt,) index array selecting one contiguous block
Indices = npt.NDArray[np.intp]

# below this, the nearest training snapshot is close enough that the held-out
# block is effectively a copy of the training set
LEAK_THRESHOLD = 0.05
# above this, the mean offset between blocks is big enough to appear as
# model error
SHIFT_THRESHOLD = 0.01

# guards every ratio in this module against a degenerate denominator
_EPS = 1e-30


# consecutive frames are near-identical, so a random split will leak.
# blocks stay contiguous in time and are separated by a gap wide enough that they are
# statistically independent
def split_indices(
    n_t: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    gap: int = 50,
) -> tuple[Indices, Indices, Indices]:
    if gap < 1:
        raise ValueError(
            f"gap={gap} would leave the blocks adjacent, which is the leakage "
            f"this split exists to prevent. Use decorrelation_lag to measure one."
        )

    n_test = int(round(test_frac * n_t))
    n_val = int(round(val_frac * n_t))
    n_train = n_t - n_val - n_test - 2 * gap

    if n_train <= 0:
        raise ValueError(
            f"gap={gap} too large for n_t={n_t} at val_frac={val_frac}, "
            f"test_frac={test_frac}; n_train would be {n_train}."
        )
    # an empty val or test block reaches `split_diagnostics`, whose `.min(1)`
    # then raises on a zero-length axis -- a long way from the cause
    if n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"val_frac={val_frac} and test_frac={test_frac} give n_val={n_val}, "
            f"n_test={n_test} at n_t={n_t}; both blocks must be non-empty."
        )

    # --[train]--a-gap-b--[val]--c-gap-d--[test]--n_t
    a = n_train
    b = a + gap
    c = b + n_val
    d = c + gap

    return np.arange(0, a), np.arange(b, c), np.arange(d, n_t)


# computes the individual gap width for a dataset
def decorrelation_lag(
    X: Snapshots,
    max_lag: int = 200,
    thresh: float = 1.0 / np.e,
    sub: int = 10,
) -> int:
    """Smallest lag at which the mean snapshot correlation drops below ``thresh``.

    Always >= 1. It used to be able to return 0: ``min(max_lag, len(Q) - 1)`` is
    0 for a single-snapshot input, the search loop then never ran, and 0 was
    returned as the gap -- which `split_indices` accepted, producing three
    adjacent blocks and exactly the leakage the gap exists to prevent. A run too
    short to measure a lag on is now an error rather than a silently useless
    split.
    """
    if sub < 1:
        raise ValueError(f"sub must be >= 1, got {sub}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    Q = _flatten_time(X, sub=sub)
    if len(Q) < 2:
        raise ValueError(
            f"need at least 2 snapshots to measure a decorrelation lag, got {len(Q)}"
        )

    norm: npt.NDArray[np.floating[Any]] = np.linalg.norm(Q, axis=1)
    max_lag = min(max_lag, len(Q) - 1)

    for lag in range(1, max_lag + 1):
        dot: npt.NDArray[np.floating[Any]] = np.sum(Q[:-lag] * Q[lag:], axis=1)
        scale = norm[:-lag] * norm[lag:] + _EPS
        r = float(np.mean(dot / scale))
        if r < thresh:
            return lag

    return max_lag


def _flatten_time(X: Snapshots, sub: int = 1) -> npt.NDArray[np.floating[Any]]:
    # (Nu, Nt, Nx, Ny) -> (Nt, Nu * spatial), NaNs zeroed, temporal mean removed
    Q = _flat_raw(X, sub=sub)
    return Q - Q.mean(axis=0, keepdims=True)


def _flat_raw(X: Snapshots, sub: int = 1) -> npt.NDArray[np.floating[Any]]:
    # same layout as _flatten_time but keeps the mean
    if X.ndim != 4:
        raise ValueError(f"expected (Nu, Nt, Nx, Ny), got shape {X.shape}")
    if sub < 1:
        raise ValueError(f"sub must be >= 1, got {sub}")
    return np.nan_to_num(
        X.reshape(X.shape[0], X.shape[1], -1)[:, :, ::sub]
        .transpose(1, 0, 2)
        .reshape(X.shape[1], -1)
    )


# run this on the early-stopping block as well as the test block
def split_diagnostics(
    X_a: Snapshots, X_b: Snapshots, sub: int = 4
) -> dict[str, float]:
    A, B = _flatten_time(X_a, sub=sub), _flatten_time(X_b, sub=sub)
    if len(A) == 0 or len(B) == 0:
        raise ValueError(
            f"both blocks must be non-empty; got {len(A)} and {len(B)} snapshots"
        )

    d2 = (B**2).sum(1)[:, None] - 2 * B @ A.T + (A**2).sum(1)[None]
    b_norm: npt.NDArray[np.floating[Any]] = np.linalg.norm(B, axis=1)
    rel = np.sqrt(np.maximum(d2, 0)).min(1) / (b_norm + _EPS)

    # _flatten_time already removes each block's own mean, so recompute raw means
    m_a: npt.NDArray[np.floating[Any]] = _flat_raw(X_a, sub).mean(axis=0)
    m_b: npt.NDArray[np.floating[Any]] = _flat_raw(X_b, sub).mean(axis=0)
    shift = float(np.mean((m_a - m_b) ** 2)) / (float(np.mean(B**2)) + _EPS)

    return {
        "nn_dist_median": float(np.median(rel)),
        "nn_dist_min": float(rel.min()),
        "mean_shift": shift,
    }


# residual left after projecting the test block onto the entire column space of
# the training block
def linear_span_ceiling(X_train: Snapshots, X_test: Snapshots) -> float:
    """Fraction of test variance outside the training block's column space.

    The denominator is guarded like every other ratio here. It was not, and a
    test block equal to the training mean makes it exactly zero -- returning
    `nan`, which then reached `meta["span_ceiling"]` and the results CSV.
    """
    A, B = _flat_raw(X_train), _flat_raw(X_test)
    mu: npt.NDArray[np.floating[Any]] = A.mean(axis=0)
    A0, B0 = A - mu, B - mu
    P: npt.NDArray[np.floating[Any]] = np.linalg.svd(A0, full_matrices=False)[2]
    residual: np.floating[Any] = np.mean((B0 - (B0 @ P.T) @ P) ** 2)
    total: np.floating[Any] = np.mean(B0**2)
    return float(residual / (total + _EPS))


def prepare_split(
    X: Snapshots,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_lag: int = 200,
    thresh: float = 1.0 / np.e,
    sub: int = 10,
) -> tuple[Snapshots, Snapshots, Snapshots, dict[str, float]]:
    """Split X and return (X_train, X_val, X_test, meta) with the gap measured."""
    gap = decorrelation_lag(X, max_lag=max_lag, thresh=thresh, sub=sub)
    tr, va, te = split_indices(
        X.shape[1], val_frac=val_frac, test_frac=test_frac, gap=gap
    )
    X_train, X_val, X_test = X[:, tr], X[:, va], X[:, te]

    # left as ints: `int` satisfies a `float` annotation, and casting would turn
    # convergence_study's "the full 2400-dim span" into "2400.0-dim"
    meta: dict[str, float] = {
        "gap": gap,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
    }
    for tag, blk in (("test", X_test), ("val", X_val)):
        meta.update(
            {f"{tag}_{k}": v for k, v in split_diagnostics(X_train, blk).items()}
        )
    meta["span_ceiling"] = linear_span_ceiling(X_train, X_test)
    return X_train, X_val, X_test, meta
