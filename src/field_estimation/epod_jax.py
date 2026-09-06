"""JAX implementations of the linear sparse-sensor baselines.

Ports snapshot POD, extended POD (Borée 2003), and POD-LSE from `epod.py`,
running the decompositions on device. The API mirrors `epod.py`, so the two are
interchangeable in `experiments/april_wake/scripts/sparse_sensor_study.py` and
`experiments/april_wake/scripts/sparse_sensor_sweep.py`.

`epod.py` remains the reference implementation; where the two disagree, trust
the NumPy version. Use this module when the linear stage costs enough to matter.
The full field matrix is (31000, 6085) in float64, so an economy SVD of it is
about 1e12 flops, and each sweep stage calls the randomized path at least once.
The same decomposition takes seconds on a GPU, and `ridge_cv_jax` evaluates a
ridge sweep as one batched solve rather than as eight sequential fits.

Precision
---------
Every function defaults to float64, which JAX enables only through
`jax.config.update("jax_enable_x64", True)`. This module applies that setting at
import. Without it, JAX truncates to float32 and the exactness checks that make
these methods verifiable no longer pass: `Psi.T @ Psi = I` to 1e-14, and exact
recovery under a linear sensor model. In float32 the singular values bottom out
around 5e-4, well above the 1e-10 an exact reconstruction reaches. Pass
`dtype="float32"` to trade that accuracy for speed, and expect the trailing
modes to be unreliable.

Typical usage:

    from field_estimation.epod_jax import PODLSEJax

    model = PODLSEJax(r_field=64, r_sensor=None, ridge=1e-4).fit(Q_tr, S_tr)
    model.score(Q_te, S_te)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .epod import (  # noqa: F401,E402
    apply_sensor_stats, delay_embed, nmse, projection_floor,
    sensor_stats, split_train_test,
)

__all__ = [
    "pod_jax",
    "extended_pod_jax",
    "lse_map_jax",
    "PODLSEJax",
    "ExtendedPODJax",
    "ridge_cv_jax",
    "delay_embed",
    "nmse",
    "projection_floor",
    "split_train_test",
]


def _dt(dtype):
    return jnp.float64 if dtype == "float64" else jnp.float32


# ── decompositions ────────────────────────────────────────────────────────────


def pod_jax(Q, r: Optional[int] = None, subtract_mean: bool = True,
            method: str = "auto", n_iter: int = 4, seed: int = 0, dtype="float64"):
    """Computes the snapshot POD of a data matrix on device.

    Args:
        Q: Data matrix, shape (N_x, N_t), one snapshot per column.
        r: Number of modes to keep. `None` keeps all of them.
        subtract_mean: Whether to remove the temporal mean before decomposing.
        method: Decomposition algorithm. One of:
            `"svd"`: Full economy SVD. Exact, and the one to validate against.
            `"randomized"`: Halko et al. (2011) sketch. Costs O(N_x N_t r)
                rather than O(N_x N_t^2), which reduces the dominant term by
                about 90 times at r=64 out of 6085 snapshots.
            `"auto"`: Uses `"randomized"` when `r` is small relative to the
                record, `"svd"` otherwise.
        n_iter: Power iterations, for `method="randomized"`.
        seed: Random seed, for `method="randomized"`.
        dtype: `"float64"` or `"float32"`. See the module docstring.

    Returns:
        A tuple `(Psi, Sigma, B, mean)` of NumPy arrays. The results convert
        back from device arrays because the rest of this repository works in
        NumPy, and a device array passed downstream fails later with an
        unrelated-looking dtype error.

    Raises:
        ValueError: If `Q` is not two-dimensional, or if `method="randomized"`
            and `r` is `None`.
    """
    dt = _dt(dtype)
    Qj = jnp.asarray(Q, dt)
    if Qj.ndim != 2:
        raise ValueError(f"Q must be (N_x, N_t), got {Qj.shape}")
    mean = Qj.mean(axis=1, keepdims=True) if subtract_mean else jnp.zeros((Qj.shape[0], 1), dt)
    Qc = Qj - mean

    if method == "auto":
        method = "randomized" if (r is not None and r < 0.25 * min(Qc.shape)) else "svd"

    if method == "randomized":
        if r is None:
            raise ValueError("randomized POD needs an explicit r")
        Psi, Sigma, Vt = _rsvd(Qc, int(r), n_iter, seed, dt)
    else:
        Psi, Sigma, Vt = jnp.linalg.svd(Qc, full_matrices=False)
        if r is not None:
            Psi, Sigma, Vt = Psi[:, :r], Sigma[:r], Vt[:r]

    B = Sigma[:, None] * Vt
    return (np.asarray(Psi), np.asarray(Sigma), np.asarray(B), np.asarray(mean))


@partial(jax.jit, static_argnames=("r", "n_iter", "dtype"))
def _rsvd(Qc, r, n_iter, seed, dtype):
    """Computes a randomized SVD with `n_iter` power iterations.

    Follows Halko et al. (2011). The QR factorisation after each multiply is
    required: without it the sketch collapses onto the leading singular vector
    within two or three iterations, because the spectrum decays and round-off
    destroys the smaller directions.

    This implements the same algorithm as `epod._pod_randomized`, so the two
    agree to the tolerance of the sketch rather than to machine precision.
    """
    key = jax.random.PRNGKey(seed)
    n_t = Qc.shape[1]
    Om = jax.random.normal(key, (n_t, min(r + 10, n_t)), dtype)
    Y = Qc @ Om
    Y, _ = jnp.linalg.qr(Y)
    for _ in range(n_iter):
        Z, _ = jnp.linalg.qr(Qc.T @ Y)
        Y, _ = jnp.linalg.qr(Qc @ Z)
    Ub, Sig, Vt = jnp.linalg.svd(Y.T @ Qc, full_matrices=False)
    return (Y @ Ub)[:, :r], Sig[:r], Vt[:r]


def extended_pod_jax(B, S, dtype="float64"):
    """Computes extended POD modes (Borée 2003).

    Extracts the part of `S` that correlates with `B`, as
    `psi_ext_k = S b_k / ||b_k||^2`, which describes what the sensor array does
    while field mode `k` is active. This is the field-to-sensor direction; the
    LSE map applies the inverse.

    Args:
        B: Field POD coefficients, shape (r, N_t).
        S: Synchronised signal, shape (N_s, N_t).
        dtype: `"float64"` or `"float32"`.

    Returns:
        Extended modes, shape (N_s, r).

    Raises:
        ValueError: If `B` and `S` hold different numbers of snapshots.
    """
    dt = _dt(dtype)
    Bj, Sj = jnp.asarray(B, dt), jnp.asarray(S, dt)
    if Bj.shape[1] != Sj.shape[1]:
        raise ValueError(f"B has {Bj.shape[1]} snapshots but S has {Sj.shape[1]}")
    norms = jnp.einsum("kt,kt->k", Bj, Bj)
    return np.asarray((Sj @ Bj.T) / jnp.where(norms > 0, norms, 1.0))


@partial(jax.jit, static_argnames=("dtype",))
def _lse_map(B, C, ridge, dtype):
    G = C @ C.T
    G = G + ridge * jnp.eye(G.shape[0], dtype=G.dtype)
    # Solve the normal equations rather than forming the inverse: the same
    # arithmetic with better conditioning.
    return jnp.linalg.solve(G, (B @ C.T).T).T


def lse_map_jax(B, C, ridge: float = 0.0, dtype="float64"):
    """Fits the ridge-regularised least-squares map from `C` to `B`.

    Computes `M = B C.T (C C.T + ridge I)^-1`.

    Args:
        B: Target coefficients, shape (r_f, N_t).
        C: Regressor coefficients, shape (r_s, N_t).
        ridge: Tikhonov parameter, in units of sensor coefficient energy.
        dtype: `"float64"` or `"float32"`.

    Returns:
        The map `M`, shape (r_f, r_s).
    """
    dt = _dt(dtype)
    return np.asarray(_lse_map(jnp.asarray(B, dt), jnp.asarray(C, dt),
                               float(ridge), dtype))


# ── ridge sweep ───────────────────────────────────────────────────────────────


def ridge_cv_jax(B, C, B_val, C_val, ridges, dtype="float64"):
    """Evaluates validation NMSE for every ridge value in one batched solve.

    The eight-value ridge sweep in `experiments/april_wake/scripts/sparse_sensor_study.py` takes
    between 19 and 192 seconds per delay length in NumPy, because each value
    refits from scratch. This function forms the Gram matrix once and maps only
    the solve over `ridges`, so the whole sweep costs little more than one fit.

    Args:
        B: Training target coefficients, shape (r_f, N_t).
        C: Training regressor coefficients, shape (r_s, N_t).
        B_val: Validation target coefficients.
        C_val: Validation regressor coefficients.
        ridges: Candidate ridge values.
        dtype: `"float64"` or `"float32"`.

    Returns:
        A tuple `(best_ridge, errors)`, where `errors` holds one validation NMSE
        per candidate, in the order given.
    """
    dt = _dt(dtype)
    Bj, Cj = jnp.asarray(B, dt), jnp.asarray(C, dt)
    Bv, Cv = jnp.asarray(B_val, dt), jnp.asarray(C_val, dt)
    G, RHS = Cj @ Cj.T, (Bj @ Cj.T).T
    eye = jnp.eye(G.shape[0], dtype=dt)
    var = jnp.var(Bv)

    def one(lam):
        M = jnp.linalg.solve(G + lam * eye, RHS).T
        return jnp.mean((Bv - M @ Cv) ** 2) / jnp.where(var > 0, var, 1.0)

    errs = np.asarray(jax.vmap(one)(jnp.asarray(ridges, dt)))
    return float(np.asarray(ridges)[int(np.argmin(errs))]), errs


# ── estimators ────────────────────────────────────────────────────────────────


@dataclass
class _Base:
    r_sensor: Optional[int] = None
    ridge: float = 0.0
    standardise_sensors: bool = True
    dtype: str = "float64"
    s_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    def _prep(self, S, learn: bool):
        """Centres and optionally standardises a sensor record per channel."""
        if learn:
            self.s_mean, self.s_scale = sensor_stats(S, self.standardise_sensors)
        return apply_sensor_stats(S, self.s_mean, self.s_scale)

    def _check(self):
        if not self.fitted:
            raise RuntimeError("call fit() before predict()/encode()/score()")

    def score(self, Q_true, S_new) -> float:
        return nmse(np.asarray(Q_true, np.float64), self.predict(S_new))


@dataclass
class PODLSEJax(_Base):
    """Reconstructs a field from sparse sensors by POD-LSE, computed on device.

    Mirrors `epod.PODLSE`. See that class for the meaning of each attribute.

    Attributes:
        r_field: POD modes kept for the field. `None` keeps all of them.
        r_sensor: POD modes kept for the sensor record.
        ridge: Tikhonov parameter, relative to the mean sensor coefficient
            energy.
        standardise_sensors: Whether to scale each channel by its training
            standard deviation.
        pod_method: Algorithm passed to `pod_jax`.
        dtype: `"float64"` or `"float32"`.
        seed: Random seed, forwarded to `pod_jax`.
    """

    r_field: Optional[int] = None
    pod_method: str = "auto"
    seed: int = 0
    Psi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Sigma: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    M: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def fit(self, Q, S) -> PODLSEJax:
        Q, S = np.asarray(Q, np.float64), np.asarray(S, np.float64)
        if Q.shape[1] != S.shape[1]:
            raise ValueError(f"field has {Q.shape[1]} snapshots, sensors have "
                             f"{S.shape[1]}; these must be synchronised sample-for-sample")
        if not np.isfinite(Q).all() or not np.isfinite(S).all():
            raise ValueError("non-finite values in the training data")

        self.Psi, self.Sigma, B, self.q_mean = pod_jax(
            Q, self.r_field, True, self.pod_method, seed=self.seed, dtype=self.dtype)
        Sc = self._prep(S, learn=True)
        self.Phi, _, C, _ = pod_jax(Sc, self.r_sensor, subtract_mean=False,
                                    method="svd", dtype=self.dtype)
        # Scale the ridge by the mean sensor-coefficient energy, so one value
        # means the same thing across datasets.
        lam = self.ridge * float(np.mean(np.einsum("kt,kt->k", C, C)))
        self.M = lse_map_jax(B, C, lam, self.dtype)
        self.fitted = True
        return self

    def encode(self, S_new) -> np.ndarray:
        self._check()
        return self.M @ (self.Phi.T @ self._prep(np.asarray(S_new, np.float64), False))

    def predict(self, S_new) -> np.ndarray:
        self._check()
        return self.Psi @ self.encode(S_new) + self.q_mean

    def project(self, Q) -> np.ndarray:
        """Projects new field data onto the fitted basis.

        Args:
            Q: Field snapshots, shape (N_x, N_t).

        Returns:
            True field POD coefficients, the target that `encode` estimates.
        """
        self._check()
        return self.Psi.T @ (np.asarray(Q, np.float64) - self.q_mean)

    def floor(self, Q) -> float:
        self._check()
        return projection_floor(np.asarray(Q, np.float64), self.Psi, self.q_mean)

    @property
    def n_params(self) -> int:
        return int(self.M.size)


@dataclass
class ExtendedPODJax(_Base):
    """Reconstructs a field through extended POD modes, computed on device.

    Equivalent to POD-LSE with the field basis left untruncated, which is the
    same projection expressed the other way round. Agreement between the two is
    therefore a correctness check, and
    `tests/test_sparse_sensors.py` asserts it.

    This class stays separate from `PODLSEJax` because the extended modes are
    useful in their own right: each one is the flow pattern associated with a
    single sensor mode.

    Attributes:
        r_sensor: POD modes kept for the sensor record.
        ridge: Tikhonov parameter, scaled as in `PODLSEJax`.
        standardise_sensors: Whether to scale each channel by its training
            standard deviation.
        dtype: `"float64"` or `"float32"`.
    """

    Psi_ext: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def fit(self, Q, S) -> ExtendedPODJax:
        Q, S = np.asarray(Q, np.float64), np.asarray(S, np.float64)
        if Q.shape[1] != S.shape[1]:
            raise ValueError(f"field has {Q.shape[1]} snapshots, sensors have {S.shape[1]}")
        self.q_mean = Q.mean(axis=1, keepdims=True)
        Sc = self._prep(S, learn=True)
        self.Phi, _, C, _ = pod_jax(Sc, self.r_sensor, subtract_mean=False,
                                    method="svd", dtype=self.dtype)
        dt = _dt(self.dtype)
        Cj = jnp.asarray(C, dt)
        G = Cj @ Cj.T
        lam = self.ridge * float(jnp.mean(jnp.einsum("kt,kt->k", Cj, Cj)))
        Qc = jnp.asarray(Q - self.q_mean, dt)
        # Psi_ext = Qc C^T (C C^T + lam I)^-1, shape (N_x, r_s). Solved as
        # G X = (Qc C^T)^T then transposed back: `solve` wants the r_s axis
        # first, and Qc C^T has it last.
        self.Psi_ext = np.asarray(
            jnp.linalg.solve(G + lam * jnp.eye(G.shape[0], dtype=dt),
                             (Qc @ Cj.T).T).T)
        self.fitted = True
        return self

    def encode(self, S_new) -> np.ndarray:
        self._check()
        return self.Phi.T @ self._prep(np.asarray(S_new, np.float64), False)

    def predict(self, S_new) -> np.ndarray:
        self._check()
        return self.Psi_ext @ self.encode(S_new) + self.q_mean

    @property
    def n_params(self) -> int:
        return int(self.Psi_ext.size)
