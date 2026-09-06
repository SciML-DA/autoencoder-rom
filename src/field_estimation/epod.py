"""Linear baselines for reconstructing a flow field from sparse sensors.

Snapshot POD, extended POD (Borée 2003), and POD with linear stochastic
estimation (POD-LSE). These are the baseline and the reference implementation
for the two-branch autoencoder in `branched_ae.py`.

All arrays are column-per-snapshot: `Q` is (N_x, N_t), `S` is (N_s, N_t). Array
conventions match `models/data_driven/autoencoders/pod_utils.py`:

    Psi    (N_x, r)   spatial modes, orthonormal columns
    B      (r, N_t)   temporal coefficients, B = Psi.T @ Q
    Sigma  (r,)       singular values, descending

POD-LSE decomposes the field and the sensor record separately, then fits a
ridge-regularised map between their coefficients:

    phi = Psi B,  s = Phi C
    M       = B C.T (C C.T + lam I)^-1
    phi_hat = Psi M Phi.T s_new

`ExtendedPOD` computes the same estimator without truncating the field. The rows
of `C` are orthogonal, so `C C.T` is diagonal and the solve reduces to a
per-mode division, giving Borée's formula. `ExtendedPOD` and
`PODLSE(r_field=None)` agree; `tests/test_sparse_sensors.py` asserts it.

Rank limit
----------
A linear estimator reconstructs into the column space of `Psi M Phi.T`, whose
rank is at most `r_sensor <= N_s`, so `r_field` above `r_sensor` gains nothing.
Widen it with `delay_embed`, which raises the bound to `N_s * n_delays`, or with
the nonlinear map in `branched_ae.py`. Call `mode_observability` to
measure how much of the field the sensors actually reach.

Dense NumPy throughout. On a 31258 x 6085 field a truncated fit takes seconds;
an exact SVD takes minutes and about 4 GB.

Typical usage:

    from field_estimation.epod import PODLSE, delay_embed, split_train_test

    Sd = delay_embed(S, n_delays=25)          # embed the full record once
    tr, te = split_train_test(Q.shape[1], 0.25, gap=100, warmup=24)
    model = PODLSE(r_field=100, r_sensor=60, ridge=1e-4).fit(Q[:, tr], Sd[:, tr])
    model.score(Q[:, te], Sd[:, te])
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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


# ── plain POD ─────────────────────────────────────────────────────────────────


def pod(
    Q: np.ndarray,
    r: Optional[int] = None,
    subtract_mean: bool = False,
    method: str = "svd",
    n_iter: int = 4,
    seed: int = 0,
):
    """Computes the snapshot POD of a data matrix.

    Args:
        Q: Data matrix, shape (N_x, N_t), one snapshot per column.
        r: Number of modes to keep. `None` keeps all of them.
        subtract_mean: Whether to remove the temporal mean before decomposing.
        method: Decomposition algorithm. One of:
            `"svd"`: Dense economy SVD. The default, and the most accurate:
                forming the correlation matrix instead squares the condition
                number, which costs about half the significant digits in the
                trailing modes.
            `"snapshot"`: Eigendecomposition of `Q.T @ Q` (Sirovich 1987).
                Cheaper when `N_x >> N_t`. The trailing modes carry the
                precision loss, so truncate well inside the spectrum.
            `"randomized"`: Randomized range finder (Halko et al. 2011).
                Requires an explicit `r`. Use this on datasets the size of the
                April experiment, where `r` is about 100 out of 6085.
            `"auto"`: Uses `"randomized"` when `r` is set and small relative to
                the matrix, `"svd"` otherwise.
        n_iter: Power iterations, for `method="randomized"`.
        seed: Random seed, for `method="randomized"`.

    Returns:
        A tuple `(Psi, Sigma, B, mean)`. `Sigma` holds the raw singular values
        of the mean-subtracted matrix. `models/data_driven/autoencoders/pod_utils.py` divides its own
        singular values by `sqrt(N_t)`, so the two modules' `Sigma` values are
        not interchangeable.

    Raises:
        ValueError: If `Q` is not two-dimensional, if `method` is unknown, or
            if `method="randomized"` and `r` is `None`.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"Q must be (N_x, N_t), got shape {Q.shape}")
    mean = Q.mean(axis=1, keepdims=True) if subtract_mean else np.zeros((Q.shape[0], 1))
    Qc = Q - mean

    if method == "auto":
        method = "randomized" if (r and r <= min(Qc.shape) // 4) else "svd"

    if method == "svd":
        Psi, Sigma, Vt = np.linalg.svd(Qc, full_matrices=False)
        if r is not None:
            Psi, Sigma, Vt = Psi[:, :r], Sigma[:r], Vt[:r]
        B = Sigma[:, None] * Vt  # == Psi.T @ Qc, but without the extra matmul
    elif method == "snapshot":
        Psi, Sigma, B = _pod_snapshot(Qc, r)
    elif method == "randomized":
        if r is None:
            raise ValueError("method='randomized' needs an explicit r")
        Psi, Sigma, B = _pod_randomized(Qc, r, n_iter, seed)
    else:
        raise ValueError(f"method must be svd/snapshot/randomized/auto, got {method!r}")

    return Psi, Sigma, B, mean


def _pod_snapshot(Qc: np.ndarray, r: Optional[int]):
    """Computes POD from the eigendecomposition of the temporal correlation matrix."""
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


def _pod_randomized(Qc: np.ndarray, r: int, n_iter: int, seed: int):
    """Computes a randomized range finder followed by a small SVD.

    Follows Halko, Martinsson, and Tropp (2011). The sketch is oversampled by
    10 modes and then truncated to `r`, because the trailing computed modes
    absorb most of the approximation error. Power iterations sharpen the
    spectral gap, which matters for PIV data, whose POD spectrum decays slowly.
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


# ── delay embedding ───────────────────────────────────────────────────────────


def delay_embed(S: np.ndarray, n_delays: int = 1, stride: int = 1,
                n_ahead: int = 0) -> np.ndarray:
    """Stacks lagged copies of a sensor record into a delay embedding.

    Block `i` of the output holds the record delayed by `i * stride` samples,
    zero-padded at the start. Column `t` holds `s_t, s_{t-stride}, ...`, so with
    the default `n_ahead=0` the embedding is causal and no future sample enters.

    `n_ahead` adds blocks at *negative* lag, so column `t` also sees
    `s_{t+stride}, ..., s_{t+n_ahead*stride}`. That makes the estimator a
    two-sided filter rather than a one-sided one. Two reasons to want it here:

    * A load cell and a PIV window several diameters downstream are not in a
      cause-and-effect order that a causal filter can express -- the same
      structure passes the disc first and the window later, so the field at `t`
      is correlated with the force both before and after `t`. The lag scan found
      its optimum at -25 samples, outside the causal window entirely.
    * For reconstruction from a recorded run, as opposed to real-time control,
      there is no reason to forbid future samples. The optimal linear estimator
      of a signal from a correlated record is two-sided (Wiener), and forcing it
      one-sided throws away half the available correlation for nothing.

    Use `n_ahead=0` when the estimator has to run causally in real time.

    Embedding raises the reachable rank from `N_s` to `N_s * n_delays` and lets
    the estimator resolve the sensor lag rather than requiring it as an input.

    Embed the whole record once, before splitting. Embedding a block in
    isolation zero-pads its first `(n_delays - 1) * stride` columns. Those
    columns are zero-padded in the full record too; pass
    `warmup=(n_delays - 1) * stride` to `split_train_test` to exclude them.

    Args:
        S: Sensor record, shape (N_s, N_t).
        n_delays: Number of *backward* lagged copies, including the unlagged one.
        stride: Sample spacing between consecutive lags.
        n_ahead: Number of *forward* copies. 0 keeps the embedding causal.

    Returns:
        The embedded record, shape (N_s * (n_delays + n_ahead), N_t), ordered
        from the most advanced block to the most delayed. Returns `S` unchanged
        when `n_delays` is 1 and `n_ahead` is 0.

    Raises:
        ValueError: If `S` is not two-dimensional, if `n_delays` or `stride` is
            less than 1, or if `n_ahead` is negative.
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


def sensor_windows(S: np.ndarray, n_delays: int = 1, stride: int = 1,
                   n_ahead: int = 0) -> np.ndarray:
    """Reshapes a sensor record into causal windows, oldest sample first.

    The network-facing view of `delay_embed`, and a reshape of it, so the linear
    estimators and the neural branches consume identical observations.

    Args:
        S: Sensor record, shape (N_s, N_t).
        n_delays: Window length in samples.
        stride: Sample spacing between consecutive lags.

    Returns:
        Windows, shape (N_t, n_delays, N_s). Window `t` holds
        `s_{t-(L-1)h}, ..., s_{t-h}, s_t`, zero-padded at the start of the
        record.
    """
    n_s = np.asarray(S).shape[0]
    L = n_delays + n_ahead
    D = delay_embed(S, n_delays, stride, n_ahead)   # (L*N_s, N_t)
    W = D.reshape(L, n_s, -1).transpose(2, 0, 1)    # (N_t, L, N_s)
    return np.ascontiguousarray(W[:, ::-1, :])      # oldest first


def sensor_stats(S: np.ndarray, standardise: bool = True):
    """
    Returns the per-channel (mean, scale) of a sensor record.

    Six-component load cells mix newtons and newton-metres, so without the
    per-channel scale the sensor POD ranks channels by unit magnitude.

    Returns:
        (mean, scale), both (N_s, 1). scale is empty when standardise is False.
    """
    S = np.asarray(S, np.float64)
    mean = S.mean(axis=1, keepdims=True)
    if not standardise:
        return mean, np.empty(0)
    sd = S.std(axis=1, keepdims=True)
    return mean, np.where(sd > 0, sd, 1.0)


def apply_sensor_stats(S: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Centres and scales a sensor record with statistics from sensor_stats.

    Raises:
        ValueError: If S has a different channel count from mean.
    """
    S = np.asarray(S, np.float64)
    if S.shape[0] != mean.shape[0]:
        raise ValueError(f"fitted on {mean.shape[0]} channels, got {S.shape[0]}")
    Sc = S - mean
    return Sc / scale if np.size(scale) else Sc


# ── extended POD ──────────────────────────────────────────────────────────────


def extended_pod(B: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Computes extended POD modes (Borée 2003).

    Extracts the part of a synchronised signal `S` that correlates with the
    field POD coefficients `B`. Mode `k` is `S` weighted by `b_k` across the
    record:

        psi_ext_k = S b_k / ||b_k||^2

    Mode `k` describes what the sensor array does while field mode `k` is
    active, which is the field-to-sensor direction. `ExtendedPOD` applies the
    sensor-to-field direction used at inference time.

    Args:
        B: Field POD coefficients, shape (r, N_t).
        S: Synchronised signal, shape (N_s, N_t).

    Returns:
        Extended modes, shape (N_s, r).

    Raises:
        ValueError: If `B` and `S` hold different numbers of snapshots.
    """
    B, S = np.asarray(B, np.float64), np.asarray(S, np.float64)
    if B.shape[1] != S.shape[1]:
        raise ValueError(f"B has {B.shape[1]} snapshots but S has {S.shape[1]}")
    norms = np.einsum("kt,kt->k", B, B)
    norms = np.where(norms > 0, norms, 1.0)
    return (S @ B.T) / norms  # (N_s, r)


def mode_observability(B: np.ndarray, C: np.ndarray) -> dict:
    """Measures how much of each field mode the sensors resolve linearly.

    `rho2` bounds what any linear estimator achieves mode by mode, independent
    of ridge and truncation. Modes where it reaches zero are unobservable from
    these sensors.

    Orthonormalises `C` first, so a non-orthogonal `C` such as a raw
    delay-embedded record also gives an exact result.

    Args:
        B: Field POD coefficients, shape (r_f, N_t).
        C: Sensor POD coefficients, shape (r_s, N_t).

    Returns:
        A dict with three entries:
            `rho2`: Shape (r_f,). Fraction of each field mode's energy that
                lies in the linear span of the sensor coefficients.
            `R`: Shape (r_f, r_s). Correlation coefficient between `b_k` and
                `c_j`, which identifies which sensor mode carries which field
                mode.
            `cum`: Shape (r_f,). Energy-weighted cumulative unexplained
                fraction, equal to the best normalised MSE a linear method
                reaches using field modes 1 through k.

    Raises:
        ValueError: If `B` and `C` hold different numbers of snapshots.
    """
    B, C = np.asarray(B, np.float64), np.asarray(C, np.float64)
    if B.shape[1] != C.shape[1]:
        raise ValueError(f"B has {B.shape[1]} snapshots but C has {C.shape[1]}")

    b_norm = np.linalg.norm(B, axis=1)
    c_norm = np.linalg.norm(C, axis=1)
    safe_b = np.where(b_norm > 0, b_norm, 1.0)
    safe_c = np.where(c_norm > 0, c_norm, 1.0)
    R = (B @ C.T) / np.outer(safe_b, safe_c)

    # orthonormal basis for the row space of C, so this is exact even if the
    # caller passed a non-orthogonal C (e.g. a raw delay-embedded record)
    Qc, _ = np.linalg.qr(C.T)  # (N_t, k)
    proj = B @ Qc  # (r_f, k)
    rho2 = np.einsum("ij,ij->i", proj, proj) / np.where(b_norm > 0, b_norm**2, 1.0)

    energy = b_norm**2
    unexplained = np.cumsum(energy * (1.0 - rho2))
    cum = unexplained / np.sum(energy) if energy.sum() > 0 else np.zeros_like(energy)
    return {"rho2": rho2, "R": R, "cum": cum}


# ── the linear map ────────────────────────────────────────────────────────────


def lse_map(B: np.ndarray, C: np.ndarray, ridge: float = 0.0, rcond: float = 1e-12) -> np.ndarray:
    """Fits the ridge-regularised least-squares map from `C` to `B`.

    Computes `M = B C.T (C C.T + ridge I)^-1` through the normal equations
    rather than by forming an inverse, which is the same arithmetic with better
    conditioning.

    Rank-deficient Gram matrices are common here rather than exceptional:
    requesting more sensor modes than the record has independent directions
    makes `C C.T` singular, and `r_sensor=None` on a delay-embedded record
    nearly always does. `np.linalg.solve` does not raise in that case; it
    returns a map with large components in the null space, which fits the
    training data and fails on test data. This function therefore checks the
    condition number and falls back to the minimum-norm least-squares solution,
    warning when it does. Truncate `r_sensor` or set a ridge to make that choice
    explicit.

    Args:
        B: Target coefficients, shape (r_f, N_t).
        C: Regressor coefficients, shape (r_s, N_t).
        ridge: Tikhonov parameter, in units of sensor coefficient energy, so it
            compares against the eigenvalues of `C C.T`. `PODLSE` scales it by
            `trace(C C.T) / r_s` to make it dimensionless.
        rcond: Reciprocal condition number below which the solve falls back to
            least squares.

    Returns:
        The map `M`, shape (r_f, r_s).

    Raises:
        ValueError: If `B` and `C` hold different numbers of snapshots.
    """
    B, C = np.asarray(B, np.float64), np.asarray(C, np.float64)
    if B.shape[1] != C.shape[1]:
        raise ValueError(f"B has {B.shape[1]} snapshots but C has {C.shape[1]}")
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
    # M G = B C^T  ->  solve G^T M^T = (B C^T)^T; G is symmetric so G^T = G
    return np.linalg.solve(G, (B @ C.T).T).T


# ── sensor preprocessing, shared by both estimators ───────────────────────────


class _SensorMixin:
    """Shared sensor preprocessing and state checks for the linear estimators."""

    def _prep_sensors(self, S, learn: bool):
        if learn:
            mean, scale = sensor_stats(S, self.standardise_sensors)
        else:
            mean, scale = self.s_mean, self.s_scale
        return apply_sensor_stats(S, mean, scale), mean, scale

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("call fit() before predict()/encode()/score()")

    @staticmethod
    def _check_pair(Q, S):
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


def _pls_directions(B: np.ndarray, C: np.ndarray, r: Optional[int]) -> np.ndarray:
    """Sensor directions ranked by cross-covariance with the field.

    The right singular vectors of `B @ C^T` are the directions in sensor space
    whose projection covaries most strongly with the field coefficients -- the
    reduced-rank-regression / PLS ordering, as opposed to the sensor PCA
    ordering, which ranks by sensor variance alone.

    Both use only training data: `B` is the field's own coefficients on the
    training snapshots, so choosing the basis this way is supervised fitting,
    not leakage, provided `fit` is called on the training split.

    Args:
        B: Field POD coefficients, shape (r_field, N_t).
        C: Standardised sensor record, shape (N_s, N_t).
        r: Directions to keep. `None` keeps all of them.

    Returns:
        The basis, shape (N_s, r), orthonormal columns.
    """
    n_s = C.shape[0]
    k = n_s if r is None else min(int(r), n_s)

    G = B @ C.T                                   # (r_field, N_s)
    _, _, Vt = np.linalg.svd(G, full_matrices=False)
    V = Vt.T                                      # (N_s, k0), k0 <= r_field
    k0 = min(V.shape[1], k)
    if k <= k0:
        return np.ascontiguousarray(V[:, :k])

    # G has rank at most r_field, so relevance alone cannot rank more than
    # r_field directions. Asking for more and silently returning fewer would
    # make r_sensor stop doing anything above r_field -- a flat sweep curve
    # that looks like a converged result and is a truncated basis.
    #
    # Complete the basis with the leading sensor-POD directions of what is left
    # after projecting the relevant subspace out. The first k0 directions are
    # still ranked by what they say about the field; the remainder fall back to
    # variance, which is the best available ordering once relevance is spent.
    V = V[:, :k0]
    C_perp = C - V @ (V.T @ C)
    U, _, _ = np.linalg.svd(C_perp, full_matrices=False)
    extra = U[:, : k - k0]
    return np.ascontiguousarray(np.hstack([V, extra]))


@dataclass
class PODLSE(_SensorMixin):
    """Reconstructs a field from synchronised sparse sensors by POD-LSE.

    Attributes:
        r_field: POD modes kept for the field. `None` keeps all of them. Values
            above `r_sensor` change little, because the map produces at most
            `r_sensor` independent directions. See the module docstring.
        r_sensor: Sensor directions kept. `None` keeps all `N_s`. Truncating
            here guards against ill-conditioned load cells and is the main
            control once the record is delay-embedded.
        sensor_basis: How those directions are *ranked* before truncation.

            `"pod"` ranks by sensor variance, which is what the sensors do most
            of, not what they say about the flow. On a rig with unfiltered
            structural resonances those are the same thing only by accident:
            the spectra here show peaks at 15-77 Hz with peak-to-median ratios
            up to 12000, so the leading sensor POD modes are the rig shaking,
            and truncation keeps the vibration and discards the flow. It is
            also why test error keeps improving out to `r_sensor=200` of 300 --
            the useful directions are buried under the resonance and you only
            reach them by keeping almost everything, ill-conditioning and all.

            `"pls"` ranks by cross-covariance with the field instead, from the
            SVD of `B @ C^T`. A direction earns its place by predicting the
            flow, not by being loud. Same estimator downstream; it changes only
            which subspace survives truncation, so a small `r_sensor` becomes
            usable and the fit stops spending rank on vibration.
        ridge: Tikhonov parameter, relative to the mean sensor coefficient
            energy, so one value means the same thing across datasets.
        standardise_sensors: Whether to divide each channel by its training
            standard deviation before the sensor POD.
        pod_method: Algorithm passed to `pod`. Use `"randomized"` with an
            explicit `r_field` on datasets the size of the April experiment.
        seed: Random seed, forwarded to `pod`.
    """

    r_field: Optional[int] = None
    r_sensor: Optional[int] = None
    sensor_basis: str = "pod"
    ridge: float = 0.0
    standardise_sensors: bool = True
    pod_method: str = "svd"
    seed: int = 0

    # -- learned ---------------------------------------------------------------
    Psi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Sigma: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    B: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    C: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    M: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    # -- api -------------------------------------------------------------------

    def fit(self, Q: np.ndarray, S: np.ndarray) -> PODLSE:
        """Fits the field basis, the sensor basis, and the map between them.

        Args:
            Q: Field snapshots, shape (N_x, N_t).
            S: Synchronised sensor record, shape (N_s, N_t).

        Returns:
            This estimator, fitted.

        Raises:
            ValueError: If `Q` and `S` hold different numbers of snapshots, or
                if either contains non-finite values.
        """
        Q = np.asarray(Q, np.float64)
        S = np.asarray(S, np.float64)
        self._check_pair(Q, S)

        self.Psi, self.Sigma, self.B, self.q_mean = pod(
            Q, self.r_field, subtract_mean=True, method=self.pod_method, seed=self.seed
        )

        Sc, self.s_mean, self.s_scale = self._prep_sensors(S, learn=True)
        if self.sensor_basis == "pod":
            self.Phi, _, self.C, _ = pod(Sc, self.r_sensor, subtract_mean=False)
        elif self.sensor_basis == "pls":
            self.Phi = _pls_directions(self.B, Sc, self.r_sensor)
            self.C = self.Phi.T @ Sc
        else:
            raise ValueError(
                f"sensor_basis must be 'pod' or 'pls', got {self.sensor_basis!r}"
            )

        self.M = lse_map(self.B, self.C, self._lam(self.C))
        self.fitted = True
        return self

    def _lam(self, C: np.ndarray) -> float:
        return self.ridge * float(np.mean(np.einsum("kt,kt->k", C, C)))

    def predict(self, S_new: np.ndarray) -> np.ndarray:
        """Reconstructs fields from sensor data.

        Args:
            S_new: Sensor record, shape (N_s, N_t).

        Returns:
            The reconstructed field, shape (N_x, N_t).
        """
        return self.Psi @ self.encode(S_new) + self.q_mean

    def encode(self, S_new: np.ndarray) -> np.ndarray:
        """Predicts field POD coefficients from sensor data.

        The two-branch autoencoder's sensor map `G` replaces this step, so
        compare a learned sensor branch against this output.

        Args:
            S_new: Sensor record, shape (N_s, N_t).

        Returns:
            Predicted field POD coefficients, shape (r_field, N_t).
        """
        self._check_fitted()
        Sc, _, _ = self._prep_sensors(np.asarray(S_new, np.float64), learn=False)
        return self.M @ (self.Phi.T @ Sc)

    def score(self, Q_true: np.ndarray, S_new: np.ndarray) -> float:
        """Returns the normalised MSE of the reconstruction.

        Args:
            Q_true: True field snapshots, shape (N_x, N_t).
            S_new: Synchronised sensor record, shape (N_s, N_t).

        Returns:
            Normalised MSE. 0 is a perfect reconstruction; 1 matches predicting
            the temporal mean.
        """
        return nmse(np.asarray(Q_true, np.float64), self.predict(S_new))

    def project(self, Q: np.ndarray) -> np.ndarray:
        """Projects new field data onto the fitted basis.

        Args:
            Q: Field snapshots, shape (N_x, N_t).

        Returns:
            True field POD coefficients, shape (r_field, N_t). These are the
            target that `encode` predicts.
        """
        self._check_fitted()
        return self.Psi.T @ (np.asarray(Q, np.float64) - self.q_mean)

    def floor(self, Q_true: np.ndarray) -> float:
        """Returns the best NMSE this field basis allows, independent of sensors.

        Equals the truncation error of `Psi` on the given data. No estimator
        using this basis scores below it. A score near the floor means the basis
        limits the result; a score well above it means the sensors or the fit do.

        Args:
            Q_true: Field snapshots, shape (N_x, N_t).

        Returns:
            The projection floor, as a normalised MSE.
        """
        self._check_fitted()
        return projection_floor(Q_true, self.Psi, self.q_mean)

    @property
    def n_params(self) -> int:
        """Number of free parameters in the map, for comparison against a network."""
        return int(self.M.size) if self.M.size else 0


# ── extended POD ──────────────────────────────────────────────────────────────


@dataclass
class ExtendedPOD(_SensorMixin):
    """Reconstructs a field from sparse sensors by extended POD (Borée 2003).

    Decomposes only the sensor record, then builds one extended field mode per
    sensor mode by correlating the field against that sensor coefficient:

        psi_ext_j = Q_c c_j / (||c_j||^2 + lam)       Psi_ext (N_x, r_s)
        Q_hat     = Psi_ext C_new + q_mean

    This computes `PODLSE(r_field=None)` without the field SVD, replacing an
    (N_x, N_t) decomposition with a single `Q_c C.T` product. On the April
    experiment that takes seconds rather than minutes, which makes this the
    cheaper estimator to sweep hyperparameters with.

    The extended modes also describe directly what a linear estimator produces:
    the reconstruction is a fixed combination of `r_sensor` spatial fields, and
    `psi_ext_j` is the flow associated with sensor mode `j`. For this problem
    they carry more information than the field POD modes, because they span the
    only part of the flow the sensors resolve.

    Attributes:
        r_sensor: POD modes kept for the sensor record. `None` keeps all `N_s`.
        ridge: Tikhonov parameter, scaled as in `PODLSE` so both accept the
            same value.
        standardise_sensors: Whether to divide each channel by its training
            standard deviation before the sensor POD.
        field_basis: Whether to also compute the field POD. Costs an SVD, and
            enables `encode`, `observability`, and `floor`.
        r_field: Modes kept for the field POD, when `field_basis` is `True`.
        pod_method: Algorithm passed to `pod` for the field POD.
        seed: Random seed, forwarded to `pod`.
    """

    r_sensor: Optional[int] = None
    ridge: float = 0.0
    standardise_sensors: bool = True
    field_basis: bool = False
    r_field: Optional[int] = None
    pod_method: str = "svd"
    seed: int = 0

    # -- learned ---------------------------------------------------------------
    Psi_ext: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    C: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    sigma_s: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Psi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    B: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    def fit(self, Q: np.ndarray, S: np.ndarray) -> ExtendedPOD:
        Q = np.asarray(Q, np.float64)
        S = np.asarray(S, np.float64)
        self._check_pair(Q, S)

        self.q_mean = Q.mean(axis=1, keepdims=True)
        Qc = Q - self.q_mean

        Sc, self.s_mean, self.s_scale = self._prep_sensors(S, learn=True)
        self.Phi, self.sigma_s, self.C, _ = pod(Sc, self.r_sensor, subtract_mean=False)

        # The rows of C are orthogonal, so C C.T is diag(||c_j||^2) and the
        # least-squares solve reduces to a division. lam uses the same scaling
        # as PODLSE so that both classes accept the same `ridge` value.
        norms = np.einsum("kt,kt->k", self.C, self.C)
        lam = self.ridge * float(np.mean(norms))
        self.Psi_ext = (Qc @ self.C.T) / (norms + lam)  # (N_x, r_s)

        if self.field_basis:
            self.Psi, _, self.B, _ = pod(
                Qc, self.r_field, subtract_mean=False, method=self.pod_method,
                seed=self.seed,
            )
        self.fitted = True
        return self

    def coefficients(self, S_new: np.ndarray) -> np.ndarray:
        """Projects new sensor data onto the fitted sensor basis.

        Args:
            S_new: Sensor record, shape (N_s, N_t).

        Returns:
            Sensor POD coefficients, shape (r_sensor, N_t).
        """
        self._check_fitted()
        Sc, _, _ = self._prep_sensors(np.asarray(S_new, np.float64), learn=False)
        return self.Phi.T @ Sc

    def predict(self, S_new: np.ndarray) -> np.ndarray:
        return self.Psi_ext @ self.coefficients(S_new) + self.q_mean

    def encode(self, S_new: np.ndarray) -> np.ndarray:
        """Predicts field POD coefficients from sensor data.

        Args:
            S_new: Sensor record, shape (N_s, N_t).

        Returns:
            Predicted field POD coefficients, shape (r_field, N_t).

        Raises:
            RuntimeError: If the estimator was constructed without
                `field_basis=True`.
        """
        if not self.Psi.size:
            raise RuntimeError("encode() needs field_basis=True at construction")
        return self.Psi.T @ (self.predict(S_new) - self.q_mean)

    def score(self, Q_true: np.ndarray, S_new: np.ndarray) -> float:
        return nmse(np.asarray(Q_true, np.float64), self.predict(S_new))

    def observability(self) -> dict:
        """Returns `mode_observability` for the fitted field and sensor bases.

        Raises:
            RuntimeError: If the estimator was constructed without
                `field_basis=True`.
        """
        if not self.B.size:
            raise RuntimeError("observability() needs field_basis=True")
        return mode_observability(self.B, self.C)

    def floor(self, Q_true: np.ndarray) -> float:
        if not self.Psi.size:
            raise RuntimeError("floor() needs field_basis=True at construction")
        return projection_floor(Q_true, self.Psi, self.q_mean)

    @property
    def n_params(self) -> int:
        return int(self.Psi_ext.size) if self.Psi_ext.size else 0


# ── metrics ───────────────────────────────────────────────────────────────────


def nmse(Q_true: np.ndarray, Q_hat: np.ndarray, var: Optional[float] = None) -> float:
    """Returns the mean squared error normalised by the variance of the truth.

    Normalises by fluctuation energy, so predicting the temporal mean scores
    exactly 1.0. The mean is taken from the block being scored, not from the
    training block.

    Set `var` explicitly in two cases. A single column has zero fluctuation
    variance and scores `inf`. A horizon sweep normalised per horizon confounds
    the model with the flow's movement over each window; pass the whole test
    block's variance instead.

    Args:
        Q_true: True snapshots, shape (N_x, N_t).
        Q_hat: Predicted snapshots, same shape as `Q_true`.
        var: Fixed normaliser. Defaults to the fluctuation variance of
            `Q_true`.

    Returns:
        The normalised MSE, or `inf` if the normaliser is zero.

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


def nmse_per_snapshot(Q_true: np.ndarray, Q_hat: np.ndarray,
                      var: Optional[float] = None) -> np.ndarray:
    """Returns the per-snapshot NMSE, using the same normalisation as `nmse`.

    Args:
        Q_true: True snapshots, shape (N_x, N_t).
        Q_hat: Predicted snapshots, same shape as `Q_true`.
        var: Fixed normaliser. Defaults to the fluctuation variance of
            `Q_true`.

    Returns:
        Per-snapshot NMSE, shape (N_t,).
    """
    Q_true = np.asarray(Q_true, np.float64)
    Q_hat = np.asarray(Q_hat, np.float64)
    if var is None:
        var = float(np.var(Q_true - Q_true.mean(axis=1, keepdims=True)))
    return np.mean((Q_true - Q_hat) ** 2, axis=0) / var


def fluctuation_variance(Q: np.ndarray) -> float:
    """Returns the temporal fluctuation variance of a block of snapshots.

    Compute this once on the test block and pass it as `var` to every `nmse`
    call in a horizon sweep, so the resulting curve compares models rather than
    window lengths.

    Args:
        Q: Snapshots, shape (N_x, N_t).

    Returns:
        The fluctuation variance.
    """
    Q = np.asarray(Q, np.float64)
    return float(np.var(Q - Q.mean(axis=1, keepdims=True)))


def projection_floor(Q_true: np.ndarray, Psi: np.ndarray, q_mean: np.ndarray) -> float:
    """Returns the NMSE of the best reconstruction available in the span of `Psi`.

    Separates a basis-limited result, which sits at the floor, from a
    sensor-limited or fit-limited one, which sits above it.

    Args:
        Q_true: True snapshots, shape (N_x, N_t).
        Psi: Spatial modes with orthonormal columns, shape (N_x, r).
        q_mean: Temporal mean subtracted before projection, shape (N_x, 1).

    Returns:
        The projection floor, as a normalised MSE.
    """
    Q_true = np.asarray(Q_true, np.float64)
    Qc = Q_true - q_mean
    return nmse(Q_true, Psi @ (Psi.T @ Qc) + q_mean)


# ── splitting ─────────────────────────────────────────────────────────────────


def cosine(Q_true: np.ndarray, Q_hat: np.ndarray) -> float:
    """Returns the normalised inner product between truth and reconstruction.

    Provided because the reference notebook reports this metric rather than
    NMSE. Being scale-invariant, it cannot detect the under-energetic
    reconstruction that ridge regularisation produces, so report `energy_ratio`
    alongside it.

    The two relate as `nmse = 1 - 2 beta cos + beta^2` for
    `beta = ||Q_hat|| / ||Q_true||`, so a cosine bounds NMSE below at
    `1 - cos^2`.

    Args:
        Q_true: True snapshots.
        Q_hat: Predicted snapshots, same shape as `Q_true`.

    Returns:
        The cosine, in [-1, 1], or `nan` if either array is all zeros.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    a = np.asarray(Q_true, np.float64).ravel()
    b = np.asarray(Q_hat, np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {Q_true.shape} vs {Q_hat.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def energy_ratio(Q_true: np.ndarray, Q_hat: np.ndarray) -> float:
    """Returns `||Q_hat|| / ||Q_true||`, the energy ratio of a reconstruction.

    Together with `cosine` this determines the NMSE, separating a reconstruction
    with the wrong shape from one with the wrong gain.

    Args:
        Q_true: True snapshots.
        Q_hat: Predicted snapshots.

    Returns:
        The ratio of norms, or `nan` if `Q_true` is all zeros.
    """
    a = np.linalg.norm(np.asarray(Q_true, np.float64))
    b = np.linalg.norm(np.asarray(Q_hat, np.float64))
    return float(b / a) if a > 0 else float("nan")


def split_train_test(
    n_t: int, test_fraction: float = 0.25, gap: int = 0, warmup: int = 0
):
    """Splits a record into contiguous train and test blocks.

    The split is contiguous and `gap` removes samples at the seam. Set `gap` to
    at least the delay window so no test input reaches back into a training
    snapshot.

    Args:
        n_t: Number of snapshots in the record.
        test_fraction: Fraction of the record held out for testing.
        gap: Samples dropped between the train and test blocks.
        warmup: Leading columns to drop. After `delay_embed`, set this to
            `(n_delays - 1) * stride`: those columns are zero-padded rather than
            holding real history, and training on them fits the estimator to
            partially blank observations.

    Returns:
        A tuple `(train_idx, test_idx)` of index arrays.

    Raises:
        ValueError: If `warmup` consumes the record, or if no training samples
            remain.
    """
    if warmup >= n_t:
        raise ValueError(f"warmup={warmup} consumes the whole record (n_t={n_t})")
    n_te = int(round(n_t * test_fraction))
    n_tr = n_t - n_te - gap
    if n_tr <= warmup:
        raise ValueError(
            f"nothing left to train on: n_t={n_t}, gap={gap}, warmup={warmup}"
        )
    return np.arange(warmup, n_tr), np.arange(n_t - n_te, n_t)


def blocked_folds(idx: np.ndarray, k: int = 5, gap: int = 0):
    """Yields contiguous cross-validation folds with guard bands.

    Unlike `sklearn.model_selection.KFold`, each validation block stays
    contiguous and `gap` samples are removed on either side of it.

    Args:
        idx: Indices to split, usually the training block.
        k: Number of folds.
        gap: Samples dropped on each side of every validation block.

    Yields:
        Tuples `(train_idx, val_idx)` of index arrays.
    """
    idx = np.asarray(idx)
    n = len(idx)
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        val = idx[lo:hi]
        keep = np.ones(n, dtype=bool)
        keep[max(0, lo - gap) : min(n, hi + gap)] = False
        yield idx[keep], val


def ridge_cv(
    Q: np.ndarray,
    S: np.ndarray,
    idx: np.ndarray,
    ridges=(0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    k: int = 5,
    gap: int = 0,
    estimator=None,
    **kwargs,
):
    """Selects a ridge by blocked cross-validation within the training block.

    Two properties of K-fold on a short record affect the result. `gap` is
    charged per fold, so a gap sized for the whole record can consume most of
    one; this function caps it at a quarter of a fold and warns. Each fold also
    trains on `(k - 1) / k` of the data, so the selected ridge suits a smaller
    problem than the final fit — negligible at 4500 training snapshots, large at
    400. Treat a ridge selected at the top of the grid as unreliable.

    Args:
        Q: Field snapshots, shape (N_x, N_t).
        S: Synchronised sensor record, shape (N_s, N_t).
        idx: Training indices to cross-validate within.
        ridges: Candidate ridge values.
        k: Number of folds.
        gap: Samples dropped on each side of every validation block.
        estimator: Estimator class to fit. Defaults to `ExtendedPOD`, which
            skips the field SVD and so runs much faster than `PODLSE` while
            selecting the same value: the two differ only in whether the field
            basis is truncated, and the ridge acts on the sensor side.
        **kwargs: Forwarded to the estimator constructor.

    Returns:
        A tuple `(best_ridge, scores)`, where `scores` maps each candidate ridge
        to its mean validation NMSE.
    """
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
    scores = {}
    for lam in ridges:
        vals = []
        for tr, va in folds:
            m = estimator(ridge=lam, **kwargs).fit(Q[:, tr], S[:, tr])
            vals.append(m.score(Q[:, va], S[:, va]))
        scores[lam] = float(np.mean(vals))
    best = min(scores, key=scores.get)
    return best, scores
