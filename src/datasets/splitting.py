"""
splitting.py
============

Taking a snapshot set into train/val/test. The diagnostics say whether
the set is good. Every function is pure numpy and takes raw
(Nu, Nt, Nx, Ny) arrays, so it imports without torch or jax.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "split_indices",
    "decorrelation_lag",
    "split_diagnostics",
    "linear_span_ceiling",
    "prepare_split",
    "LEAK_THRESHOLD",
    "SHIFT_THRESHOLD",
]

# below this, the nearest training snapshot is close enough that the held-out
# block is effectively a copy of the training set
LEAK_THRESHOLD = 0.05
# above this, the mean offset between blocks is big enough to appear as
# model error
SHIFT_THRESHOLD = 0.01


# consecutive frames are near-identical, so a random split will leak.
# blocks stay contiguous in time and are separated by a gap wide enough that they are
# statistically independent
def split_indices(
    n_t: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    gap: int = 50,
):
    n_test = int(round(test_frac * n_t))
    n_val = int(round(val_frac * n_t))
    n_train = n_t - n_val - n_test - 2 * gap

    if n_train <= 0:
        raise ValueError(
            f"gap={gap} too large for n_t={n_t} at val_frac={val_frac}, "
            f"test_frac={test_frac}; n_train would be {n_train}."
        )

    # --[train]--a-gap-b--[val]--c-gap-d--[test]--n_t
    a = n_train
    b = a + gap
    c = b + n_val
    d = c + gap

    return np.arange(0, a), np.arange(b, c), np.arange(d, n_t)


# computes the individual gap width for a dataset
def decorrelation_lag(
    X: np.ndarray,
    max_lag: int = 200,
    thresh: float = 1.0 / np.e,
    sub: int = 10,
) -> int:
    Q = _flatten_time(X, sub=sub)
    norm = np.linalg.norm(Q, axis=1)
    max_lag = min(max_lag, len(Q) - 1)

    for lag in range(1, max_lag + 1):
        r = float(
            np.mean(
                np.sum(Q[:-lag] * Q[lag:], axis=1) / (norm[:-lag] * norm[lag:] + 1e-30)
            )
        )
        if r < thresh:
            return lag

    return max_lag


def _flatten_time(X: np.ndarray, sub: int = 1) -> np.ndarray:
    # (Nu, Nt, Nx, Ny) -> (Nt, Nu * spatial), NaNs zeroed, temporal mean removed
    Q = X.reshape(X.shape[0], X.shape[1], -1)[:, :, ::sub]
    Q = np.nan_to_num(Q.transpose(1, 0, 2).reshape(X.shape[1], -1))
    return Q - Q.mean(axis=0, keepdims=True)


def _flat_raw(X: np.ndarray, sub: int = 1) -> np.ndarray:
    # same layout as _flatten_time but keeps the mean
    return np.nan_to_num(
        X.reshape(X.shape[0], X.shape[1], -1)[:, :, ::sub]
        .transpose(1, 0, 2)
        .reshape(X.shape[1], -1)
    )


# run this on the early-stopping block as well as the test block
def split_diagnostics(X_a: np.ndarray, X_b: np.ndarray, sub: int = 4) -> dict:
    A, B = _flatten_time(X_a, sub=sub), _flatten_time(X_b, sub=sub)
    d2 = (B**2).sum(1)[:, None] - 2 * B @ A.T + (A**2).sum(1)[None]
    rel = np.sqrt(np.maximum(d2, 0)).min(1) / (np.linalg.norm(B, axis=1) + 1e-30)

    # _flatten_time already removes each block's own mean, so recompute raw means
    m_a, m_b = _flat_raw(X_a, sub).mean(0), _flat_raw(X_b, sub).mean(0)
    shift = float(np.mean((m_a - m_b) ** 2)) / (float(np.mean(B**2)) + 1e-30)

    return {
        "nn_dist_median": float(np.median(rel)),
        "nn_dist_min": float(rel.min()),
        "mean_shift": shift,
    }


# residual left after projecting the test block onto the entire column space of
# the training block
def linear_span_ceiling(X_train: np.ndarray, X_test: np.ndarray) -> float:
    A, B = _flat_raw(X_train), _flat_raw(X_test)
    mu = A.mean(0)
    A0, B0 = A - mu, B - mu
    P = np.linalg.svd(A0, full_matrices=False)[2]
    return float(np.mean((B0 - (B0 @ P.T) @ P) ** 2) / np.mean(B0**2))


def prepare_split(
    X: np.ndarray,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_lag: int = 200,
    thresh: float = 1.0 / np.e,
    sub: int = 10,
):
    """Split X and return (X_train, X_val, X_test, meta) with the gap measured."""
    gap = decorrelation_lag(X, max_lag=max_lag, thresh=thresh, sub=sub)
    tr, va, te = split_indices(
        X.shape[1], val_frac=val_frac, test_frac=test_frac, gap=gap
    )
    X_train, X_val, X_test = X[:, tr], X[:, va], X[:, te]

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
