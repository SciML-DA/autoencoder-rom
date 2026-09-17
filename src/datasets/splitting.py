# pyright: strict
"""Splits a snapshot set into train, validation, and test blocks.

Typical usage example:

  X_train, X_val, X_test, meta = prepare_split(X)
  if meta["test_nn_dist_median"] < LEAK_THRESHOLD:
      print("the test block repeats the training set")
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

#: A `(Nu, Nt, Nx, Ny)` snapshot block. NaN marks a solid-body point.
Snapshots = npt.NDArray[np.floating[Any]]
#: A `(Nt,)` index array selecting one contiguous block.
Indices = npt.NDArray[np.intp]

#: Below this nearest-neighbor distance, the closest training snapshot sits near
#: enough that the held-out block repeats the training set.
LEAK_THRESHOLD = 0.05
#: Above this relative mean offset, the difference between two blocks is large
#: enough to show up as model error.
SHIFT_THRESHOLD = 0.01

#: Guards every ratio in this module against a zero denominator.
_EPS = 1e-30


def split_indices(
    n_t: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    gap: int = 50,
    warmup: int = 0,
) -> tuple[Indices, Indices, Indices]:
    """Divides a time axis into contiguous blocks separated by a gap.

    The blocks lie in time order with a gap of `gap` discarded samples between
    them:

      [warmup]--[train]--gap--[val]--gap--[test]

    With `val_frac=0`, the validation block is empty and one gap separates the
    training block from the test block:

      [warmup]--[train]--gap--[test]

    Measure `gap` with `decorrelation_lag`.

    Args:
      n_t: Length of the time axis.
      val_frac: Fraction of `n_t` to hold out for validation. 0 omits the
        validation block.
      test_frac: Fraction of `n_t` to hold out for testing.
      gap: Samples to discard between blocks, so that neighboring blocks are
        statistically independent.
      warmup: Leading samples to discard before the training block. After
        `field_estimation.delay_embed`, set this to `(n_delays - 1) * stride`
        to drop the partly zero-padded windows.

    Returns:
      Train, validation, and test index arrays, in that order. The validation
      array is empty when `val_frac` is 0.

    Raises:
      ValueError: If `gap` is below 1, if `warmup` is negative, if `val_frac` is
        outside `[0, 1)` or `test_frac` is outside `(0, 1)`, if the warmup, gaps,
        and held-out fractions leave no training samples, or if a held-out block
        with a nonzero fraction comes out empty.
    """
    if gap < 1:
        raise ValueError(
            f"gap={gap} leaves the blocks adjacent, which is the leakage this "
            f"split prevents. Measure a gap with decorrelation_lag."
        )
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    n_test = int(round(test_frac * n_t))
    n_val = int(round(val_frac * n_t))
    n_gaps = 2 if val_frac > 0 else 1
    a = n_t - n_val - n_test - n_gaps * gap

    if a <= warmup:
        raise ValueError(
            f"gap={gap} and warmup={warmup} too large for n_t={n_t} at "
            f"val_frac={val_frac}, test_frac={test_frac}; n_train would be {a - warmup}."
        )
    # An empty validation or test block reaches `split_diagnostics`, where
    # `.min(1)` raises on a zero-length axis, far from the cause.
    if (val_frac > 0 and n_val <= 0) or n_test <= 0:
        raise ValueError(
            f"val_frac={val_frac} and test_frac={test_frac} give n_val={n_val}, "
            f"n_test={n_test} at n_t={n_t}; a held-out block with a nonzero "
            f"fraction must be non-empty."
        )

    train = np.arange(warmup, a, dtype=np.intp)
    if n_val == 0:
        return train, np.empty(0, dtype=np.intp), np.arange(a + gap, n_t, dtype=np.intp)

    b = a + gap
    c = b + n_val
    d = c + gap
    return train, np.arange(b, c, dtype=np.intp), np.arange(d, n_t, dtype=np.intp)


def decorrelation_lag(
    X: Snapshots,
    max_lag: int = 200,
    thresh: float = 1.0 / np.e,
    sub: int = 10,
) -> int:
    """Measures how many samples apart two snapshots become independent.

    This is the gap width `split_indices` needs for one dataset. The result is
    always 1 or greater, because a gap of 0 leaves the blocks adjacent and
    reintroduces the leakage the gap prevents.

    Args:
      X: Snapshot block shaped `(Nu, Nt, Nx, Ny)`.
      max_lag: Largest lag to test. The search stops here and returns this value
        if the correlation never falls below `thresh`.
      thresh: Correlation below which two snapshots count as independent. The
        default, `1/e`, is the usual convention for a correlation time.
      sub: Take every `sub`-th spatial point. Correlation is a global measure,
        so subsampling costs little and saves a large matrix product.

    Returns:
      The smallest lag whose mean snapshot correlation falls below `thresh`,
      clamped to `max_lag` and to `Nt - 1`.

    Raises:
      ValueError: If `sub` or `max_lag` is below 1, or if `X` holds fewer than
        two snapshots, which is too few to measure a lag on.
    """
    if sub < 1:
        raise ValueError(f"sub must be >= 1, got {sub}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    Q = _flatten_time(X, sub=sub)
    if len(Q) < 2:
        raise ValueError(f"need at least 2 snapshots to measure a decorrelation lag, got {len(Q)}")

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
    """Flattens a snapshot block and removes its temporal mean.

    Args:
      X: Snapshot block shaped `(Nu, Nt, Nx, Ny)`.
      sub: Take every `sub`-th spatial point.

    Returns:
      A `(Nt, Nu * spatial)` array with NaNs zeroed and each column centered on
      its own mean.

    Raises:
      ValueError: If `X` is not 4-dimensional, or `sub` is below 1.
    """
    Q = _flat_raw(X, sub=sub)
    return Q - Q.mean(axis=0, keepdims=True)


def _flat_raw(X: Snapshots, sub: int = 1) -> npt.NDArray[np.floating[Any]]:
    """Flattens a snapshot block, keeping its temporal mean.

    Args:
      X: Snapshot block shaped `(Nu, Nt, Nx, Ny)`.
      sub: Take every `sub`-th spatial point.

    Returns:
      A `(Nt, Nu * spatial)` array with NaNs zeroed.

    Raises:
      ValueError: If `X` is not 4-dimensional, or `sub` is below 1.
    """
    if X.ndim != 4:
        raise ValueError(f"expected (Nu, Nt, Nx, Ny), got shape {X.shape}")
    if sub < 1:
        raise ValueError(f"sub must be >= 1, got {sub}")
    return np.nan_to_num(X.reshape(X.shape[0], X.shape[1], -1)[:, :, ::sub].transpose(1, 0, 2).reshape(X.shape[1], -1))


def split_diagnostics(X_a: Snapshots, X_b: Snapshots, sub: int = 4) -> dict[str, float]:
    """Measures how far one block sits from another.

    Run this on the early-stopping block as well as the test block. An
    early-stopping block that repeats the training set stops training at the
    wrong epoch, and reports nothing unusual while doing so.

    Args:
      X_a: Reference block, normally the training block.
      X_b: Held-out block to measure against `X_a`.
      sub: Take every `sub`-th spatial point.

    Returns:
      A dictionary with three entries:

      - `nn_dist_median`: Median distance from a snapshot in `X_b` to its
        nearest neighbor in `X_a`, relative to its own norm. Compare against
        `LEAK_THRESHOLD`.
      - `nn_dist_min`: The same distance for the closest snapshot in `X_b`.
      - `mean_shift`: Squared offset between the two block means, relative to
        the variance of `X_b`. Compare against `SHIFT_THRESHOLD`.

    Raises:
      ValueError: If either block is empty or not 4-dimensional.
    """
    A, B = _flatten_time(X_a, sub=sub), _flatten_time(X_b, sub=sub)
    if len(A) == 0 or len(B) == 0:
        raise ValueError(f"both blocks must be non-empty; got {len(A)} and {len(B)} snapshots")

    d2 = (B**2).sum(1)[:, None] - 2 * B @ A.T + (A**2).sum(1)[None]
    b_norm: npt.NDArray[np.floating[Any]] = np.linalg.norm(B, axis=1)
    rel = np.sqrt(np.maximum(d2, 0)).min(1) / (b_norm + _EPS)

    # `_flatten_time` removes each block's own mean, so the offset between the
    # two means needs the raw blocks.
    m_a: npt.NDArray[np.floating[Any]] = _flat_raw(X_a, sub).mean(axis=0)
    m_b: npt.NDArray[np.floating[Any]] = _flat_raw(X_b, sub).mean(axis=0)
    shift = float(np.mean((m_a - m_b) ** 2)) / (float(np.mean(B**2)) + _EPS)

    return {
        "nn_dist_median": float(np.median(rel)),
        "nn_dist_min": float(rel.min()),
        "mean_shift": shift,
    }


def linear_span_ceiling(X_train: Snapshots, X_test: Snapshots) -> float:
    """Measures how much test variance lies outside the training column space.

    This is the residual left after projecting the test block onto the entire
    column space of the training block.

    Args:
      X_train: Training block shaped `(Nu, Nt, Nx, Ny)`.
      X_test: Test block in the same layout.

    Returns:
      The unexplained fraction of test variance, between 0 and 1. A value near 0
      means the training block already spans the test block.

    Raises:
      ValueError: If either block is not 4-dimensional.
    """
    A, B = _flat_raw(X_train), _flat_raw(X_test)
    mu: npt.NDArray[np.floating[Any]] = A.mean(axis=0)
    A0, B0 = A - mu, B - mu
    P: npt.NDArray[np.floating[Any]] = np.linalg.svd(A0, full_matrices=False)[2]
    residual: np.floating[Any] = np.mean((B0 - (B0 @ P.T) @ P) ** 2)
    # `_EPS` guards this denominator as it does every ratio here: a test block
    # equal to the training mean drives `total` to zero.
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
    """Splits a snapshot set and measures the quality of the split.

    Measures the gap with `decorrelation_lag`, cuts the blocks with
    `split_indices`, then runs `split_diagnostics` on both held-out blocks and
    `linear_span_ceiling` on the test block.

    Args:
      X: Snapshot block shaped `(Nu, Nt, Nx, Ny)`.
      val_frac: Fraction of the time axis to hold out for validation.
      test_frac: Fraction of the time axis to hold out for testing.
      max_lag: Largest lag `decorrelation_lag` tests.
      thresh: Correlation below which `decorrelation_lag` calls two snapshots
        independent.
      sub: Take every `sub`-th spatial point when measuring the lag.

    Returns:
      The train, validation, and test blocks, then a metadata dictionary
      holding:

      - `gap`, `n_train`, `n_val`, `n_test`: The measured gap and the block
        sizes.
      - `test_*` and `val_*`: The `split_diagnostics` entries for each held-out
        block, prefixed with the block name.
      - `span_ceiling`: The `linear_span_ceiling` of the test block.

    Raises:
      ValueError: If `val_frac` is not positive, if `X` holds too few snapshots
        to measure a lag, or if the fractions and the measured gap leave a block
        empty.
    """
    if val_frac <= 0:
        raise ValueError(f"prepare_split needs a validation block; val_frac must be > 0, got {val_frac}")
    gap = decorrelation_lag(X, max_lag=max_lag, thresh=thresh, sub=sub)
    tr, va, te = split_indices(X.shape[1], val_frac=val_frac, test_frac=test_frac, gap=gap)
    X_train, X_val, X_test = X[:, tr], X[:, va], X[:, te]

    meta: dict[str, float] = {
        "gap": gap,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
    }
    for tag, blk in (("test", X_test), ("val", X_val)):
        meta.update({f"{tag}_{k}": v for k, v in split_diagnostics(X_train, blk).items()})
    meta["span_ceiling"] = linear_span_ceiling(X_train, X_test)
    return X_train, X_val, X_test, meta
