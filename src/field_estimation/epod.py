# pyright: strict
# `Q`, `B`, `C`, `S`, `Psi`, `G`, and friends are matrices by the linear-algebra
# convention this codebase uses, not module constants.
# pyright: reportConstantRedefinition=false

"""Linear baselines for estimating a flow field from sparse sensors.

Implements snapshot POD, extended POD (Borée 2003), and POD with linear
stochastic estimation (POD-LSE).

Every array holds one snapshot per column: `Q` is `(N_x, N_t)` and `S` is
`(N_s, N_t)`:
    Psi    (N_x, r)   spatial modes, orthonormal columns
    B      (r, N_t)   temporal coefficients, B = Psi.T @ Q
    Sigma  (r,)       singular values, descending

POD-LSE decomposes the field and the sensor record separately, then fits a
ridge-regularized map between their coefficients:

    phi = Psi B,  s = Phi C
    M       = B C.T (C C.T + lam I)^-1
    phi_hat = Psi M Phi.T s_new

`ExtendedPOD` computes the same estimator without truncating the field, and
agrees with `PODLSE(r_field=None)`.

A linear estimator reconstructs into the column space of `Psi M Phi.T`, whose
rank is at most `r_sensor`, and `r_sensor` is at most `N_s`. `delay_embed`
raises that bound to `N_s * n_delays`, and `mode_observability` measures how
much of the field the sensors reach.

Typical usage example:

  from field_estimation.epod import PODLSE, delay_embed, split_train_test

  Sd = delay_embed(S, n_delays=25)
  tr, te = split_train_test(Q.shape[1], 0.25, gap=100, warmup=24)
  model = PODLSE(r_field=100, r_sensor=60, ridge=1e-4).fit(Q[:, tr], Sd[:, tr])
  model.score(Q[:, te], Sd[:, te])
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import numpy as np
import numpy.typing as npt

#: A real-valued array. Each parameter documents its own shape.
FloatArray = npt.NDArray[np.floating[Any]]
#: An index array selecting snapshots along the time axis.
IndexArray = npt.NDArray[np.intp]


class Observability(TypedDict):
    """How much of each field mode a set of sensors resolves linearly.

    Attributes:
      rho2: Shape `(r_f,)`. Fraction of each field mode's energy that lies in
        the linear span of the sensor coefficients.
      R: Shape `(r_f, r_s)`. Correlation between each field mode and each sensor
        mode.
      cum: Shape `(r_f,)`. Energy-weighted cumulative unexplained fraction,
        equal to the best normalized MSE a linear estimator reaches using field
        modes 1 through k.
    """

    rho2: FloatArray
    R: FloatArray
    cum: FloatArray


class _CVEstimator(Protocol):
    """An estimator that `ridge_cv` can fit and score."""

    def fit(self, Q: FloatArray, S: FloatArray) -> _CVEstimator:
        """Fits on one training block.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          This estimator, fitted.
        """
        ...

    def score(self, Q_true: FloatArray, S_new: FloatArray) -> float:
        """Scores a held-out block.

        Args:
          Q_true: True field snapshots, shape `(N_x, N_t)`.
          S_new: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          Normalized mean squared error.
        """
        ...


#: Builds a `_CVEstimator` from a `ridge` keyword argument and the keyword
#: arguments `ridge_cv` forwards.
EstimatorFactory = Callable[..., _CVEstimator]

__all__ = [
    "cosine",
    "energy_ratio",
    "PODLSE",
    "ExtendedPOD",
    "pod",
    "lse_map",
    "extended_pod",
    "mode_observability",
    "delay_embed",
    "sensor_windows",
    "sensor_stats",
    "apply_sensor_stats",
    "projection_floor",
    "nmse",
    "nmse_per_snapshot",
    "fluctuation_variance",
    "split_train_test",
    "blocked_folds",
    "ridge_cv",
]


# ── POD ───────────────────────────────────────────────────────────────────────


def pod(
    Q: FloatArray,
    r: int | None = None,
    subtract_mean: bool = False,
    method: str = "svd",
    n_iter: int = 4,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Computes the snapshot POD of a data matrix.

    Args:
      Q: Data matrix, shape `(N_x, N_t)`, one snapshot per column.
      r: Number of modes to keep. `None` keeps all of them.
      subtract_mean: Remove the temporal mean before decomposing.
      method: Decomposition algorithm. One of:

        - `"svd"`: Dense economy SVD. The most accurate.
        - `"snapshot"`: Eigendecomposition of `Q.T @ Q` (Sirovich 1987). Faster
          when `N_x` is much larger than `N_t`, and less precise in the trailing
          modes.
        - `"randomized"`: Randomized range finder (Halko et al. 2011). Requires
          `r`.
        - `"auto"`: `"randomized"` when `r` is at most a quarter of the smaller
          dimension, otherwise `"svd"`.

      n_iter: Power iterations, for `method="randomized"`.
      seed: Random seed, for `method="randomized"`.

    Returns:
      `(Psi, Sigma, B, mean)`, shaped `(N_x, r)`, `(r,)`, `(r, N_t)`, and
      `(N_x, 1)`. `Sigma` holds the raw singular values of the mean-subtracted
      matrix, whereas `models/data_driven/autoencoders/pod_utils.py` divides its
      singular values by `sqrt(N_t)`.

    Raises:
      ValueError: If `Q` is not two-dimensional, if `r` is below 1, if `method`
        is unknown, or if `method="randomized"` and `r` is `None`.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"Q must be (N_x, N_t), got shape {Q.shape}")
    if r is not None and r < 1:
        raise ValueError(f"r must be >= 1 or None, got {r}")
    mean = Q.mean(axis=1, keepdims=True) if subtract_mean else np.zeros((Q.shape[0], 1))
    Qc = Q - mean

    if method == "auto":
        method = "randomized" if (r and r <= min(Qc.shape) // 4) else "svd"

    if method == "svd":
        Psi, Sigma, Vt = np.linalg.svd(Qc, full_matrices=False)
        if r is not None:
            Psi, Sigma, Vt = Psi[:, :r], Sigma[:r], Vt[:r]
        # Equal to `Psi.T @ Qc`, without the extra matrix product.
        B = Sigma[:, None] * Vt
    elif method == "snapshot":
        Psi, Sigma, B = _pod_snapshot(Qc, r)
    elif method == "randomized":
        if r is None:
            raise ValueError("method='randomized' needs an explicit r")
        Psi, Sigma, B = _pod_randomized(Qc, r, n_iter, seed)
    else:
        raise ValueError(f"method must be svd/snapshot/randomized/auto, got {method!r}")

    return Psi, Sigma, B, mean


def _pod_snapshot(Qc: FloatArray, r: int | None) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Computes POD from the eigendecomposition of the correlation matrix.

    Args:
      Qc: Mean-subtracted data matrix, shape `(N_x, N_t)`.
      r: Modes to keep. `None` keeps all of them.

    Returns:
      `(Psi, Sigma, B)`, shaped `(N_x, r)`, `(r,)`, and `(r, N_t)`.
    """
    lam, A = np.linalg.eigh(Qc.T @ Qc)
    idx = lam.argsort()[::-1]
    lam, A = lam[idx], A[:, idx]
    if r is not None:
        lam, A = lam[:r], A[:, :r]

    Sigma = np.sqrt(np.clip(lam, 0.0, None))
    safe = np.where(Sigma > 0, Sigma, np.inf)
    Psi = (Qc @ A) / safe
    Psi[:, Sigma == 0] = 0.0
    return Psi, Sigma, Sigma[:, None] * A.T


def _pod_randomized(Qc: FloatArray, r: int, n_iter: int, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Computes POD with a randomized range finder followed by a small SVD.

    Follows Halko, Martinsson, and Tropp (2011). Oversamples the sketch by 10
    modes before truncating to `r`.

    Args:
      Qc: Mean-subtracted data matrix, shape `(N_x, N_t)`.
      r: Modes to keep.
      n_iter: Power iterations.
      seed: Random seed for the sketch.

    Returns:
      `(Psi, Sigma, B)`, shaped `(N_x, r)`, `(r,)`, and `(r, N_t)`.
    """
    rng = np.random.default_rng(seed)
    ell = min(r + 10, min(Qc.shape))
    Y = Qc @ rng.standard_normal((Qc.shape[1], ell))
    Y, _ = np.linalg.qr(Y)
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Qc.T @ Y)
        Y, _ = np.linalg.qr(Qc @ Y)

    Ub, Sigma, Vt = np.linalg.svd(Y.T @ Qc, full_matrices=False)
    Psi = Y @ Ub
    return Psi[:, :r], Sigma[:r], (Sigma[:r, None] * Vt[:r])


# ── Delay embedding ───────────────────────────────────────────────────────────


def delay_embed(S: FloatArray, n_delays: int = 1, stride: int = 1, n_ahead: int = 0) -> FloatArray:
    """Stacks lagged copies of a sensor record into a delay embedding.

    Block `i` of the output holds the record delayed by `i * stride` samples,
    zero-padded at the start, so column `t` holds `s_t, s_{t-stride}, ...`. With
    `n_ahead=0` no future sample enters, so the embedding is causal. `n_ahead`
    adds blocks at negative lag, so column `t` also holds
    `s_{t+stride}, ..., s_{t+n_ahead*stride}`.

    Embed the whole record once, before splitting. The first
    `(n_delays - 1) * stride` columns are partly zero padding (pass that count as
    `warmup` to `split_train_test` to exclude them).

    Args:
      S: Sensor record, shape `(N_s, N_t)`.
      n_delays: Number of backward copies, including the unlagged one.
      stride: Sample spacing between consecutive lags.
      n_ahead: Number of forward copies. 0 keeps the embedding causal.

    Returns:
      The embedded record, shape `(N_s * (n_delays + n_ahead), N_t)`, ordered
      from the most advanced block to the most delayed. `S` itself when
      `n_delays` is 1 and `n_ahead` is 0.

    Raises:
      ValueError: If `S` is not two-dimensional, if `n_delays` or `stride` is
        below 1, or if `n_ahead` is negative.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"expected (N_s, N_t), got {S.shape}")

    if n_delays < 1 or stride < 1:
        raise ValueError(f"n_delays and stride must be >= 1, got {n_delays}, {stride}")
    if n_ahead < 0:
        raise ValueError(f"n_ahead must be >= 0, got {n_ahead}")
    if n_delays == 1 and n_ahead == 0:
        return S

    n_s, n_t = S.shape
    lags = [i * stride for i in range(-n_ahead, n_delays)]
    out = np.zeros((len(lags) * n_s, n_t), dtype=S.dtype)

    for i, lag in enumerate(lags):
        blk = out[i * n_s : (i + 1) * n_s]
        if lag == 0:
            blk[:] = S
        elif lag > 0:
            if lag < n_t:
                blk[:, lag:] = S[:, : n_t - lag]
        else:
            k = -lag
            if k < n_t:
                blk[:, : n_t - k] = S[:, k:]
    return out


def sensor_windows(S: FloatArray, n_delays: int = 1, stride: int = 1, n_ahead: int = 0) -> FloatArray:
    """Reshapes a sensor record into windows, oldest sample first.

    Holds the same values as `delay_embed`, so the linear estimators and the
    neural sensor branches see identical observations.

    Args:
      S: Sensor record, shape `(N_s, N_t)`.
      n_delays: Number of backward samples in each window, including the current
        one.
      stride: Sample spacing between consecutive lags.
      n_ahead: Number of forward samples in each window. 0 keeps the windows
        causal.

    Returns:
      Windows, shape `(N_t, n_delays + n_ahead, N_s)`. Window `t` holds its
      samples from oldest to newest, zero-padded at the start of the record.

    Raises:
      ValueError: If `S` is not two-dimensional, if `n_delays` or `stride` is
        below 1, or if `n_ahead` is negative.
    """
    n_s = np.asarray(S).shape[0]
    L = n_delays + n_ahead
    D = delay_embed(S, n_delays, stride, n_ahead)
    W = D.reshape(L, n_s, -1).transpose(2, 0, 1)
    return np.ascontiguousarray(W[:, ::-1, :])


def sensor_stats(S: FloatArray, standardise: bool = True) -> tuple[FloatArray, FloatArray]:
    """Measures the per-channel mean and scale of a sensor record.

    Args:
      S: Sensor record, shape `(N_s, N_t)`.
      standardise: Return a per-channel scale as well as a mean.

    Returns:
      `(mean, scale)`, both shape `(N_s, 1)`. `scale` is empty when
      `standardise` is `False`. A channel with zero variance gets a scale of 1.
    """
    S = np.asarray(S, np.float64)
    mean = S.mean(axis=1, keepdims=True)
    if not standardise:
        return mean, np.empty(0)
    sd = S.std(axis=1, keepdims=True)
    return mean, np.where(sd > 0, sd, 1.0)


def apply_sensor_stats(S: FloatArray, mean: FloatArray, scale: FloatArray) -> FloatArray:
    """Centers and scales a sensor record with statistics from `sensor_stats`.

    Args:
      S: Sensor record, shape `(N_s, N_t)`.
      mean: Per-channel mean, shape `(N_s, 1)`.
      scale: Per-channel scale, shape `(N_s, 1)`, or empty to skip scaling.

    Returns:
      The standardized record, shape `(N_s, N_t)`.

    Raises:
      ValueError: If `S` has a different channel count from `mean`.
    """
    S = np.asarray(S, np.float64)
    if S.shape[0] != mean.shape[0]:
        raise ValueError(f"fitted on {mean.shape[0]} channels, got {S.shape[0]}")
    Sc = S - mean
    return Sc / scale if np.size(scale) else Sc


# ── Extended POD ──────────────────────────────────────────────────────────────


def _check_coefficients(B: FloatArray, other: FloatArray, other_name: str) -> None:
    """Checks that two coefficient matrices can be paired snapshot by snapshot.

    Args:
      B: Field coefficients, expected shape `(r, N_t)`.
      other: The second matrix, expected shape `(k, N_t)`.
      other_name: Name of the second matrix, for the error message.

    Raises:
      ValueError: If either matrix is not two-dimensional, or if the two hold
        different numbers of snapshots.
    """
    if B.ndim != 2 or other.ndim != 2:
        raise ValueError(f"B and {other_name} must be two-dimensional, got {B.shape} and {other.shape}")
    if B.shape[1] != other.shape[1]:
        raise ValueError(f"B has {B.shape[1]} snapshots but {other_name} has {other.shape[1]}")


def extended_pod(B: FloatArray, S: FloatArray) -> FloatArray:
    """Computes extended POD modes (Borée 2003).

    Extracts the part of a synchronized signal `S` that correlates with the field
    POD coefficients `B`. Mode `k` is `S` weighted by `b_k` across the record,
    and describes what the sensor array does while field mode `k` is active:

        psi_ext_k = S b_k / ||b_k||^2

    Args:
      B: Field POD coefficients, shape `(r, N_t)`.
      S: Synchronized signal, shape `(N_s, N_t)`.

    Returns:
      Extended modes, shape `(N_s, r)`.

    Raises:
      ValueError: If either array is not two-dimensional, or if `B` and `S` hold
        different numbers of snapshots.
    """
    B, S = np.asarray(B, np.float64), np.asarray(S, np.float64)
    _check_coefficients(B, S, "S")
    norms = np.einsum("kt,kt->k", B, B)
    norms = np.where(norms > 0, norms, 1.0)
    return (S @ B.T) / norms


def mode_observability(B: FloatArray, C: FloatArray) -> Observability:
    """Measures how much of each field mode the sensors resolve linearly.

    Orthonormalizes `C` first, so the result is exact for a `C` with
    non-orthogonal rows, such as a raw delay-embedded record.

    Args:
      B: Field POD coefficients, shape `(r_f, N_t)`.
      C: Sensor coefficients, shape `(r_s, N_t)`.

    Returns:
      The observability of each field mode. `rho2` bounds what any linear
      estimator achieves mode by mode, independent of ridge and truncation.

    Raises:
      ValueError: If either array is not two-dimensional, or if `B` and `C` hold
        different numbers of snapshots.
    """
    B, C = np.asarray(B, np.float64), np.asarray(C, np.float64)
    _check_coefficients(B, C, "C")

    b_norm = np.linalg.norm(B, axis=1)
    c_norm = np.linalg.norm(C, axis=1)
    safe_b = np.where(b_norm > 0, b_norm, 1.0)
    safe_c = np.where(c_norm > 0, c_norm, 1.0)
    R = (B @ C.T) / np.outer(safe_b, safe_c)

    Qc, _ = np.linalg.qr(C.T)
    proj = B @ Qc
    rho2 = np.einsum("ij,ij->i", proj, proj) / np.where(b_norm > 0, b_norm**2, 1.0)

    energy = b_norm**2
    unexplained = np.cumsum(energy * (1.0 - rho2))
    cum = unexplained / np.sum(energy) if energy.sum() > 0 else np.zeros_like(energy)
    return Observability(rho2=rho2, R=R, cum=cum)


# ── Linear map ────────────────────────────────────────────────────────────────


def lse_map(B: FloatArray, C: FloatArray, ridge: float = 0.0, rcond: float = 1e-12) -> FloatArray:
    """Fits the ridge-regularized least-squares map from `C` to `B`.

    Computes `M = B C.T (C C.T + ridge I)^-1` by solving the normal equations.
    When the Gram matrix is rank deficient, it warns and returns the
    minimum-norm least-squares solution instead. Truncate `r_sensor` or set a
    ridge to avoid the fallback.

    Args:
      B: Target coefficients, shape `(r_f, N_t)`.
      C: Regressor coefficients, shape `(r_s, N_t)`.
      ridge: Tikhonov parameter, in units of sensor coefficient energy, so it
        compares against the eigenvalues of `C C.T`.
      rcond: Reciprocal condition number below which the solve falls back to
        least squares.

    Returns:
      The map `M`, shape `(r_f, r_s)`.

    Raises:
      ValueError: If either array is not two-dimensional, if `B` and `C` hold
        different numbers of snapshots, or if `ridge` is negative.
    """
    B, C = np.asarray(B, np.float64), np.asarray(C, np.float64)
    _check_coefficients(B, C, "C")
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    G = C @ C.T
    if ridge:
        G = G + ridge * np.eye(G.shape[0])

    ev = np.linalg.eigvalsh(G)
    hi = float(ev[-1])
    if hi <= 0:
        return np.zeros((B.shape[0], C.shape[0]))

    if float(ev[0]) / hi <= rcond:
        warnings.warn(
            f"sensor Gram matrix is rank deficient (condition number "
            f"{hi / max(ev[0], np.finfo(float).tiny):.2e}); using the "
            "minimum-norm least-squares solution. Truncate r_sensor or add a "
            "ridge to make this an explicit choice rather than a fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.linalg.lstsq(G, (B @ C.T).T, rcond=None)[0].T
    # `M G = B C.T`, and `G` is symmetric, so `M.T` solves `G M.T = (B C.T).T`.
    return np.linalg.solve(G, (B @ C.T).T).T


# ── Sensor preprocessing ──────────────────────────────────────────────────────


class _SensorMixin:
    """Sensor preprocessing and state checks shared by the linear estimators."""

    if TYPE_CHECKING:
        # The dataclass subclasses declare these fields. A class-level
        # annotation here would become a constructor field of each subclass.
        standardise_sensors: bool
        s_mean: FloatArray
        s_scale: FloatArray
        fitted: bool

    def _prep_sensors(self, S: FloatArray, learn: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Centers and scales a sensor record.

        Args:
          S: Sensor record, shape `(N_s, N_t)`.
          learn: Measure the statistics from `S` rather than reusing the fitted
            ones.

        Returns:
          The standardized record, the per-channel mean, and the per-channel
          scale.

        Raises:
          ValueError: If `learn` is `False` and `S` has a different channel count
            from the fitted statistics.
        """
        if learn:
            mean, scale = sensor_stats(S, self.standardise_sensors)
        else:
            mean, scale = self.s_mean, self.s_scale
        return apply_sensor_stats(S, mean, scale), mean, scale

    def _check_fitted(self) -> None:
        """Raises if the estimator has no fitted state.

        Checks the fitted sensor statistics as well as the `fitted` flag, which
        is a constructor argument and can be set without fitting.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if not self.fitted or self.s_mean.size == 0:
            raise RuntimeError("call fit() before predict()/encode()/score()")

    @staticmethod
    def _check_pair(Q: FloatArray, S: FloatArray) -> None:
        """Checks that a field and a sensor record can be fitted together.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Sensor record, shape `(N_s, N_t)`.

        Raises:
          ValueError: If the two hold different snapshot counts, or if either
            holds non-finite values.
        """
        if Q.shape[1] != S.shape[1]:
            raise ValueError(
                f"field has {Q.shape[1]} snapshots, sensors have {S.shape[1]}; "
                "these must be synchronised sample-for-sample"
            )
        if not np.isfinite(Q).all() or not np.isfinite(S).all():
            raise ValueError(
                "non-finite values in the training data. Drop the disc-mask "
                "points from Q and interpolate or drop sensor dropouts first."
            )


# ── POD-LSE ───────────────────────────────────────────────────────────────────


def _pls_directions(B: FloatArray, C: FloatArray, r: int | None) -> FloatArray:
    """Ranks sensor directions by cross-covariance with the field.

    Takes the right singular vectors of `B @ C.T`. That matrix has rank at most
    `r_field`, so when `r` asks for more directions, the rest come from the
    sensor POD of the residual left after projecting out the ranked ones.

    Args:
      B: Field POD coefficients, shape `(r_field, N_t)`.
      C: Standardized sensor record, shape `(N_s, N_t)`.
      r: Directions to keep. `None` keeps all `N_s`.

    Returns:
      The basis, shape `(N_s, r)`, with orthonormal columns.
    """
    n_s = C.shape[0]
    k = n_s if r is None else min(int(r), n_s)

    G = B @ C.T
    _, _, Vt = np.linalg.svd(G, full_matrices=False)
    V = Vt.T
    k0 = min(V.shape[1], k)
    if k <= k0:
        return np.ascontiguousarray(V[:, :k])

    V = V[:, :k0]
    C_perp = C - V @ (V.T @ C)
    U, _, _ = np.linalg.svd(C_perp, full_matrices=False)
    extra = U[:, : k - k0]
    return np.ascontiguousarray(np.hstack([V, extra]))


@dataclass
class PODLSE(_SensorMixin):
    """Estimates a field from synchronized sparse sensors by POD-LSE.

    Attributes:
      r_field: POD modes kept for the field. `None` keeps all of them. Values
        above `r_sensor` add little, because the map produces at most `r_sensor`
        independent directions.
      r_sensor: Sensor directions kept. `None` keeps all `N_s`.
      sensor_basis: How the sensor directions are ranked before truncation.
        `"pod"` ranks them by sensor variance. `"pls"` ranks them by
        cross-covariance with the field, from the SVD of `B @ C.T`.
      ridge: Tikhonov parameter, relative to the mean sensor coefficient energy.
      standardise_sensors: Divide each channel by its training standard
        deviation before the sensor decomposition.
      pod_method: Algorithm `pod` uses for the field.
      seed: Random seed, forwarded to `pod`.
      Psi: Field modes, shape `(N_x, r_field)`, learned by `fit`.
      Sigma: Field singular values, shape `(r_field,)`, learned by `fit`.
      B: Field coefficients on the training block, learned by `fit`.
      Phi: Sensor directions, shape `(N_s, r_sensor)`, learned by `fit`.
      C: Sensor coefficients on the training block, learned by `fit`.
      M: Map from sensor coefficients to field coefficients, learned by `fit`.
      q_mean: Field mean, shape `(N_x, 1)`, learned by `fit`.
      s_mean: Per-channel sensor mean, learned by `fit`.
      s_scale: Per-channel sensor scale, learned by `fit`.
      fitted: Whether `fit` has completed.
    """

    r_field: int | None = None
    r_sensor: int | None = None
    sensor_basis: str = "pod"
    ridge: float = 0.0
    standardise_sensors: bool = True
    pod_method: str = "svd"
    seed: int = 0

    # ── Learned state ──
    Psi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Sigma: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    B: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    C: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    M: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    # ── API ──

    def fit(self, Q: FloatArray, S: FloatArray) -> PODLSE:
        """Fits the field basis, the sensor basis, and the map between them.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          This estimator, fitted.

        Raises:
          ValueError: If `sensor_basis` is unknown, if `Q` and `S` hold different
            numbers of snapshots, if either holds non-finite values, or if a
            rank or ridge option is out of range.
        """
        if self.sensor_basis not in ("pod", "pls"):
            raise ValueError(f"sensor_basis must be 'pod' or 'pls', got {self.sensor_basis!r}")
        Q = np.asarray(Q, np.float64)
        S = np.asarray(S, np.float64)
        self._check_pair(Q, S)

        self.Psi, self.Sigma, self.B, self.q_mean = pod(
            Q, self.r_field, subtract_mean=True, method=self.pod_method, seed=self.seed
        )

        Sc, self.s_mean, self.s_scale = self._prep_sensors(S, learn=True)
        if self.sensor_basis == "pod":
            self.Phi, _, self.C, _ = pod(Sc, self.r_sensor, subtract_mean=False)
        else:
            self.Phi = _pls_directions(self.B, Sc, self.r_sensor)
            self.C = self.Phi.T @ Sc

        self.M = lse_map(self.B, self.C, self._lam(self.C))
        self.fitted = True
        return self

    def _lam(self, C: FloatArray) -> float:
        """Scales `ridge` by the mean sensor coefficient energy.

        Args:
          C: Sensor coefficients, shape `(r_s, N_t)`.

        Returns:
          The Tikhonov parameter in the units `lse_map` expects.
        """
        energy: FloatArray = np.einsum("kt,kt->k", C, C)
        return self.ridge * float(np.mean(energy))

    def predict(self, S_new: FloatArray) -> FloatArray:
        """Reconstructs fields from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          The reconstructed field, shape `(N_x, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        return self.Psi @ self.encode(S_new) + self.q_mean

    def encode(self, S_new: FloatArray) -> FloatArray:
        """Predicts field POD coefficients from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          Predicted field POD coefficients, shape `(r_field, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check_fitted()
        Sc, _, _ = self._prep_sensors(np.asarray(S_new, np.float64), learn=False)
        return self.M @ (self.Phi.T @ Sc)

    def score(self, Q_true: FloatArray, S_new: FloatArray) -> float:
        """Returns the normalized MSE of the reconstruction.

        Args:
          Q_true: True field snapshots, shape `(N_x, N_t)`.
          S_new: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          Normalized MSE. 0 is a perfect reconstruction; 1 matches predicting the
          temporal mean.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        return nmse(np.asarray(Q_true, np.float64), self.predict(S_new))

    def project(self, Q: FloatArray) -> FloatArray:
        """Projects field data onto the fitted basis.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.

        Returns:
          True field POD coefficients, shape `(r_field, N_t)`, which are what
          `encode` predicts.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check_fitted()
        return self.Psi.T @ (np.asarray(Q, np.float64) - self.q_mean)

    def floor(self, Q_true: FloatArray) -> float:
        """Returns the best NMSE this field basis allows, independent of sensors.

        Equals the truncation error of `Psi` on `Q_true`, so no estimator using
        this basis scores below it.

        Args:
          Q_true: Field snapshots, shape `(N_x, N_t)`.

        Returns:
          The projection floor, as a normalized MSE.

        Raises:
          RuntimeError: If `fit` has not been called.
          ValueError: If `Q_true` has a different row count from `Psi`.
        """
        self._check_fitted()
        return projection_floor(Q_true, self.Psi, self.q_mean)

    @property
    def n_params(self) -> int:
        """Number of free parameters in the coefficient map, or 0 before `fit`."""
        return int(self.M.size) if self.M.size else 0


# ── Extended POD estimator ────────────────────────────────────────────────────


@dataclass
class ExtendedPOD(_SensorMixin):
    """Estimates a field from sparse sensors by extended POD (Borée 2003).

    Decomposes only the sensor record, then builds one extended field mode per
    sensor mode by correlating the field against that mode's coefficient:

        psi_ext_j = Q_c c_j / (||c_j||^2 + lam)       Psi_ext (N_x, r_s)
        Q_hat     = Psi_ext C_new + q_mean

    Computes `PODLSE(r_field=None)` without the field SVD. Each extended mode
    `psi_ext_j` is the flow associated with sensor mode `j`, and the
    reconstruction is a fixed combination of `r_sensor` of them.

    Attributes:
      r_sensor: POD modes kept for the sensor record. `None` keeps all `N_s`.
      ridge: Tikhonov parameter, scaled as in `PODLSE`.
      standardise_sensors: Divide each channel by its training standard
        deviation before the sensor decomposition.
      field_basis: Also compute the field POD, which `encode`, `observability`,
        and `floor` require.
      r_field: Modes kept for the field POD, when `field_basis` is `True`.
      pod_method: Algorithm `pod` uses for the field POD.
      seed: Random seed, forwarded to `pod`.
      Psi_ext: Extended modes, shape `(N_x, r_sensor)`, learned by `fit`.
      Phi: Sensor modes, shape `(N_s, r_sensor)`, learned by `fit`.
      C: Sensor coefficients on the training block, learned by `fit`.
      sigma_s: Sensor singular values, learned by `fit`.
      Psi: Field modes, learned by `fit` when `field_basis` is `True`.
      B: Field coefficients, learned by `fit` when `field_basis` is `True`.
      q_mean: Field mean, shape `(N_x, 1)`, learned by `fit`.
      s_mean: Per-channel sensor mean, learned by `fit`.
      s_scale: Per-channel sensor scale, learned by `fit`.
      fitted: Whether `fit` has completed.
    """

    r_sensor: int | None = None
    ridge: float = 0.0
    standardise_sensors: bool = True
    field_basis: bool = False
    r_field: int | None = None
    pod_method: str = "svd"
    seed: int = 0

    # ── Learned state ──
    Psi_ext: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    C: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    sigma_s: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Psi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    B: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    def fit(self, Q: FloatArray, S: FloatArray) -> ExtendedPOD:
        """Fits the sensor basis and one extended field mode per sensor mode.

        Also fits the field POD when `field_basis` is `True`.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          This estimator, fitted.

        Raises:
          ValueError: If `ridge` is negative, if `Q` and `S` hold different
            numbers of snapshots, if either holds non-finite values, or if a
            rank option is below 1.
        """
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")
        Q = np.asarray(Q, np.float64)
        S = np.asarray(S, np.float64)
        self._check_pair(Q, S)

        self.q_mean = Q.mean(axis=1, keepdims=True)
        Qc = Q - self.q_mean

        Sc, self.s_mean, self.s_scale = self._prep_sensors(S, learn=True)
        self.Phi, self.sigma_s, self.C, _ = pod(Sc, self.r_sensor, subtract_mean=False)

        # The rows of `C` are orthogonal, so `C C.T` is diagonal and the
        # least-squares solve reduces to a division per mode.
        norms: FloatArray = np.einsum("kt,kt->k", self.C, self.C)
        lam = self.ridge * float(np.mean(norms))
        self.Psi_ext = (Qc @ self.C.T) / (norms + lam)

        if self.field_basis:
            self.Psi, _, self.B, _ = pod(
                Qc,
                self.r_field,
                subtract_mean=False,
                method=self.pod_method,
                seed=self.seed,
            )
        self.fitted = True
        return self

    def coefficients(self, S_new: FloatArray) -> FloatArray:
        """Projects sensor data onto the fitted sensor basis.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          Sensor POD coefficients, shape `(r_sensor, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check_fitted()
        Sc, _, _ = self._prep_sensors(np.asarray(S_new, np.float64), learn=False)
        return self.Phi.T @ Sc

    def predict(self, S_new: FloatArray) -> FloatArray:
        """Reconstructs fields from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          The reconstructed field, shape `(N_x, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        return self.Psi_ext @ self.coefficients(S_new) + self.q_mean

    def encode(self, S_new: FloatArray) -> FloatArray:
        """Predicts field POD coefficients from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          Predicted field POD coefficients, shape `(r_field, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called, or if the estimator was
            constructed without `field_basis=True`.
        """
        self._check_fitted()
        if not self.Psi.size:
            raise RuntimeError("encode() needs field_basis=True at construction")
        return self.Psi.T @ (self.predict(S_new) - self.q_mean)

    def score(self, Q_true: FloatArray, S_new: FloatArray) -> float:
        """Returns the normalized MSE of the reconstruction.

        Args:
          Q_true: True field snapshots, shape `(N_x, N_t)`.
          S_new: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          Normalized MSE. 0 is a perfect reconstruction; 1 matches predicting the
          temporal mean.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        return nmse(np.asarray(Q_true, np.float64), self.predict(S_new))

    def observability(self) -> Observability:
        """Measures the observability of the fitted field basis from the sensors.

        Returns:
          The result of `mode_observability` for the fitted field and sensor
          coefficients.

        Raises:
          RuntimeError: If `fit` has not been called, or if the estimator was
            constructed without `field_basis=True`.
        """
        self._check_fitted()
        if not self.B.size:
            raise RuntimeError("observability() needs field_basis=True")
        return mode_observability(self.B, self.C)

    def floor(self, Q_true: FloatArray) -> float:
        """Returns the best NMSE this field basis allows, independent of sensors.

        Args:
          Q_true: Field snapshots, shape `(N_x, N_t)`.

        Returns:
          The truncation error of `Psi` on `Q_true`, as a normalized MSE.

        Raises:
          RuntimeError: If `fit` has not been called, or if the estimator was
            constructed without `field_basis=True`.
          ValueError: If `Q_true` has a different row count from `Psi`.
        """
        self._check_fitted()
        if not self.Psi.size:
            raise RuntimeError("floor() needs field_basis=True at construction")
        return projection_floor(Q_true, self.Psi, self.q_mean)

    @property
    def n_params(self) -> int:
        """Number of free parameters in the extended modes, or 0 before `fit`."""
        return int(self.Psi_ext.size) if self.Psi_ext.size else 0


# ── Metrics ───────────────────────────────────────────────────────────────────


def nmse(Q_true: FloatArray, Q_hat: FloatArray, var: float | None = None) -> float:
    """Returns the mean squared error normalized by the variance of the truth.

     A single snapshot has zero fluctuation variance; pass `var` to normalize by a larger block,
     such as the whole test block.

    Args:
      Q_true: True snapshots, shape `(N_x, N_t)`.
      Q_hat: Predicted snapshots, the same shape as `Q_true`.
      var: Fixed normalizer. Defaults to the fluctuation variance of `Q_true`.

    Returns:
      The normalized MSE, or `inf` if the normalizer is 0.

    Raises:
      ValueError: If the two arrays have different shapes.
    """
    Q_true = np.asarray(Q_true, np.float64)
    Q_hat = np.asarray(Q_hat, np.float64)

    if Q_true.shape != Q_hat.shape:
        raise ValueError(f"shape mismatch: {Q_true.shape} vs {Q_hat.shape}")
    if var is None:
        var = float(np.var(Q_true - Q_true.mean(axis=1, keepdims=True)))

    return float(np.mean((Q_true - Q_hat) ** 2) / var) if var > 0 else float("inf")


def nmse_per_snapshot(Q_true: FloatArray, Q_hat: FloatArray, var: float | None = None) -> FloatArray:
    """Returns the NMSE of each snapshot, normalized as in `nmse`.

    Args:
      Q_true: True snapshots, shape `(N_x, N_t)`.
      Q_hat: Predicted snapshots, the same shape as `Q_true`.
      var: Fixed normalizer. Defaults to the fluctuation variance of `Q_true`.

    Returns:
      Per-snapshot NMSE, shape `(N_t,)`. Every entry is `inf` if the normalizer
      is 0.

    Raises:
      ValueError: If the two arrays have different shapes.
    """
    Q_true = np.asarray(Q_true, np.float64)
    Q_hat = np.asarray(Q_hat, np.float64)

    if Q_true.shape != Q_hat.shape:
        raise ValueError(f"shape mismatch: {Q_true.shape} vs {Q_hat.shape}")
    if var is None:
        var = float(np.var(Q_true - Q_true.mean(axis=1, keepdims=True)))
    if var <= 0:
        return np.full(int(Q_true.shape[1]), np.inf, dtype=np.float64)

    per_snapshot: FloatArray = np.mean((Q_true - Q_hat) ** 2, axis=0) / var
    return per_snapshot


def fluctuation_variance(Q: FloatArray) -> float:
    """Returns the temporal fluctuation variance of a block of snapshots.

    Pass the result as `var` to `nmse` to score several windows against one
    normalizer.

    Args:
      Q: Snapshots, shape `(N_x, N_t)`.

    Returns:
      The variance of `Q` about its temporal mean.
    """
    Q = np.asarray(Q, np.float64)
    return float(np.var(Q - Q.mean(axis=1, keepdims=True)))


def projection_floor(Q_true: FloatArray, Psi: FloatArray, q_mean: FloatArray) -> float:
    """Returns the NMSE of the best reconstruction in the span of `Psi`.

    Args:
      Q_true: True snapshots, shape `(N_x, N_t)`.
      Psi: Spatial modes with orthonormal columns, shape `(N_x, r)`.
      q_mean: Temporal mean subtracted before projecting, shape `(N_x, 1)`.

    Returns:
      The projection floor, as a normalized MSE.

    Raises:
      ValueError: If `Q_true` and `Psi` have different row counts.
    """
    Q_true = np.asarray(Q_true, np.float64)
    if Psi.ndim != 2 or Psi.shape[0] != Q_true.shape[0]:
        raise ValueError(f"Psi {Psi.shape} must have one row per row of Q_true {Q_true.shape}")
    Qc = Q_true - q_mean
    return nmse(Q_true, Psi @ (Psi.T @ Qc) + q_mean)


def cosine(Q_true: FloatArray, Q_hat: FloatArray) -> float:
    """Returns the normalized inner product of the truth and the reconstruction.
    Invariant to scale.

    Args:
      Q_true: True snapshots.
      Q_hat: Predicted snapshots, the same shape as `Q_true`.

    Returns:
      The cosine, between -1 and 1, or `nan` if either array is all zeros.

    Raises:
      ValueError: If the two arrays have different shapes.
    """
    a = np.asarray(Q_true, np.float64).ravel()
    b = np.asarray(Q_hat, np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {Q_true.shape} vs {Q_hat.shape}")

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def energy_ratio(Q_true: FloatArray, Q_hat: FloatArray) -> float:
    """Returns the ratio of the reconstruction's norm to the truth's.

    Args:
      Q_true: True snapshots.
      Q_hat: Predicted snapshots.

    Returns:
      `||Q_hat|| / ||Q_true||`, or `nan` if `Q_true` is all zeros.
    """
    a = np.linalg.norm(np.asarray(Q_true, np.float64))
    b = np.linalg.norm(np.asarray(Q_hat, np.float64))
    return float(b / a) if a > 0 else float("nan")


# ── Splitting ─────────────────────────────────────────────────────────────────


def split_train_test(
    n_t: int, test_fraction: float = 0.25, gap: int = 0, warmup: int = 0
) -> tuple[IndexArray, IndexArray]:
    """Splits a record into a contiguous training block and test block.

    The test block is the end of the record. Set `gap` to at least the delay
    window, so no test window reaches back into a training snapshot.

    Args:
      n_t: Number of snapshots in the record.
      test_fraction: Fraction of the record held out for testing.
      gap: Snapshots dropped between the training and test blocks.
      warmup: Leading snapshots to drop. After `delay_embed`, set this to
        `(n_delays - 1) * stride` to drop the partly zero-padded windows.

    Returns:
      The training indices and the test indices.

    Raises:
      ValueError: If `test_fraction` is outside `[0, 1)`, if `gap` or `warmup`
        is negative, if `warmup` consumes the record, or if no training
        snapshots remain.
    """
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if gap < 0 or warmup < 0:
        raise ValueError(f"gap and warmup must be >= 0, got {gap}, {warmup}")
    if warmup >= n_t:
        raise ValueError(f"warmup={warmup} consumes the whole record (n_t={n_t})")
    n_te = int(round(n_t * test_fraction))
    n_tr = n_t - n_te - gap
    if n_tr <= warmup:
        raise ValueError(f"nothing left to train on: n_t={n_t}, gap={gap}, warmup={warmup}")
    return np.arange(warmup, n_tr), np.arange(n_t - n_te, n_t)


def blocked_folds(idx: IndexArray, k: int = 5, gap: int = 0) -> Iterator[tuple[IndexArray, IndexArray]]:
    """Yields contiguous cross-validation folds with guard bands.

    Each validation block is contiguous, and `gap` snapshots are dropped from
    the training indices on either side of it.

    Args:
      idx: Indices to split, usually the training block.
      k: Number of folds.
      gap: Snapshots dropped on each side of every validation block.

    Yields:
      The training indices and the validation indices for each fold.

    Raises:
      ValueError: If `k` is below 2, if `idx` holds fewer than `k` indices, or
        if `gap` is negative.
    """
    idx = np.asarray(idx)
    n = len(idx)
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if n < k:
        raise ValueError(f"cannot split {n} indices into {k} folds")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        val = idx[lo:hi]
        keep = np.ones(n, dtype=bool)
        keep[max(0, lo - gap) : min(n, hi + gap)] = False
        yield idx[keep], val


def ridge_cv(
    Q: FloatArray,
    S: FloatArray,
    idx: IndexArray,
    ridges: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    k: int = 5,
    gap: int = 0,
    estimator: EstimatorFactory | None = None,
    **kwargs: Any,
) -> tuple[float, dict[float, float]]:
    """Selects a ridge by blocked cross-validation within the training block.

    Caps `gap` at a quarter of a fold, and warns when it does. Each fold trains
    on `(k - 1) / k` of the block, so the selected ridge suits a slightly
    smaller problem than the final fit.

    Args:
      Q: Field snapshots, shape `(N_x, N_t)`.
      S: Synchronized sensor record, shape `(N_s, N_t)`.
      idx: Training indices to cross-validate within.
      ridges: Candidate ridge values.
      k: Number of folds.
      gap: Snapshots dropped on each side of every validation block.
      estimator: Builds the estimator for each candidate. Defaults to
        `ExtendedPOD`.
      **kwargs: Forwarded to the estimator constructor.

    Returns:
      The ridge with the lowest mean validation NMSE, and the mean validation
      NMSE of every candidate.

    Raises:
      ValueError: If `ridges` is empty, if `k` is below 2 or exceeds the number
        of indices, or if a fit fails.
    """
    if not ridges:
        raise ValueError("ridges must hold at least one candidate")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    estimator = estimator or ExtendedPOD
    idx = np.asarray(idx)
    max_gap = max(0, len(idx) // (4 * k))

    if gap > max_gap:
        warnings.warn(
            f"ridge_cv: gap={gap} would consume most of each fold "
            f"({len(idx)} training samples, k={k}); capping it at {max_gap}. "
            "The folds are then separated by less than one correlation time, so "
            "the selected ridge is optimistic -- use a longer training block if "
            "this matters.",
            RuntimeWarning,
            stacklevel=2,
        )
        gap = max_gap

    folds = list(blocked_folds(idx, k=k, gap=gap))
    scores: dict[float, float] = {}

    for lam in ridges:
        vals: list[float] = []
        for tr, va in folds:
            m = estimator(ridge=lam, **kwargs).fit(Q[:, tr], S[:, tr])
            vals.append(m.score(Q[:, va], S[:, va]))
        scores[lam] = float(np.mean(vals))

    best = min(scores, key=lambda lam: scores[lam])
    return best, scores
