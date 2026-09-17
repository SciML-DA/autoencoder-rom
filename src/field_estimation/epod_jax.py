# pyright: strict
# `Q`, `B`, `C`, `S`, `G`, and `Psi` are matrices by the linear-algebra
# convention this codebase uses, not module constants.
# pyright: reportConstantRedefinition=false
#
# JAX ships `py.typed` but annotates its array API incompletely: `jnp.zeros`, jax.jit`,
# and related resolve to partially unknowntypes. Every binding this module owns
# is annotated explicitly.
# pyright: reportUnknownMemberType=false

"""JAX implementations of the linear sparse-sensor estimators.

Implements snapshot POD, extended POD (Borée 2003), and POD-LSE from `epod.py`
with the decompositions on device. The functions and classes take the same
arguments as their `epod.py` counterparts, and `epod.py` is the reference
implementation.

Importing this module enables 64-bit precision in JAX, and every function
computes in float64 by default. With `dtype="float32"`, the trailing modes are
unreliable: the singular values bottom out near 5e-4.

Every function takes and returns NumPy arrays.

Typical usage example:

  from field_estimation.epod_jax import PODLSEJax

  model = PODLSEJax(r_field=64, r_sensor=None, ridge=1e-4).fit(Q_tr, S_tr)
  model.score(Q_te, S_te)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.typing import DTypeLike

from .epod import (
    apply_sensor_stats,
    delay_embed,
    nmse,
    projection_floor,
    sensor_stats,
)

jax.config.update("jax_enable_x64", True)

#: A real-valued NumPy array. Each parameter documents its own shape.
FloatArray = npt.NDArray[np.floating[Any]]
#: An array on device.
DeviceArray = jax.Array
#: Precision to compute in.
Dtype = Literal["float64", "float32"]

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
]


def _dt(dtype: Dtype) -> DTypeLike:
    """Maps a precision name onto the JAX scalar type it selects.

    Args:
      dtype: `"float64"` or `"float32"`.

    Returns:
      The corresponding `jnp` scalar type.
    """
    return jnp.float64 if dtype == "float64" else jnp.float32


def _svd(A: DeviceArray) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    """Computes an economy SVD.

    Args:
      A: Matrix on device.

    Returns:
      `(U, S, Vt)` on device.
    """
    U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
    return U, S, Vt


def _check_coefficients(B: DeviceArray, other: DeviceArray, other_name: str) -> None:
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


# ── Decompositions ────────────────────────────────────────────────────────────


def pod_jax(
    Q: FloatArray,
    r: int | None = None,
    subtract_mean: bool = True,
    method: str = "auto",
    n_iter: int = 4,
    seed: int = 0,
    dtype: Dtype = "float64",
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Computes the snapshot POD of a data matrix on device.

    Args:
      Q: Data matrix, shape `(N_x, N_t)`, one snapshot per column.
      r: Number of modes to keep. `None` keeps all of them.
      subtract_mean: Remove the temporal mean before decomposing.
      method: Decomposition algorithm. One of:

        - `"svd"`: Dense economy SVD.
        - `"snapshot"`: Accepted for compatibility with `epod.pod`, and computed
          with the dense economy SVD.
        - `"randomized"`: Randomized range finder (Halko et al. 2011). Requires
          `r`.
        - `"auto"`: `"randomized"` when `r` is below a quarter of the smaller
          dimension, otherwise `"svd"`.

      n_iter: Power iterations, for `method="randomized"`.
      seed: Random seed, for `method="randomized"`.
      dtype: Precision to compute in.

    Returns:
      `(Psi, Sigma, B, mean)`, shaped `(N_x, r)`, `(r,)`, `(r, N_t)`, and
      `(N_x, 1)`.

    Raises:
      ValueError: If `Q` is not two-dimensional, if `r` is below 1, if `method`
        is unknown, or if `method="randomized"` and `r` is `None`.
    """
    if method not in ("svd", "snapshot", "randomized", "auto"):
        raise ValueError(f"method must be svd/snapshot/randomized/auto, got {method!r}")
    if r is not None and r < 1:
        raise ValueError(f"r must be >= 1 or None, got {r}")

    dt = _dt(dtype)
    Qj: DeviceArray = jnp.asarray(Q, dt)
    if Qj.ndim != 2:
        raise ValueError(f"Q must be (N_x, N_t), got {Qj.shape}")

    mean: DeviceArray = Qj.mean(axis=1, keepdims=True) if subtract_mean else jnp.zeros((Qj.shape[0], 1), dt)
    Qc = Qj - mean

    if method == "auto":
        method = "randomized" if (r is not None and r < 0.25 * min(Qc.shape)) else "svd"

    if method == "randomized":
        if r is None:
            raise ValueError("randomized POD needs an explicit r")
        Psi, Sigma, Vt = cast(
            "tuple[DeviceArray, DeviceArray, DeviceArray]",
            _rsvd(Qc, int(r), n_iter, seed, dt),
        )
    else:
        Psi, Sigma, Vt = _svd(Qc)
        if r is not None:
            Psi, Sigma, Vt = Psi[:, :r], Sigma[:r], Vt[:r]

    B: DeviceArray = Sigma[:, None] * Vt
    return (np.asarray(Psi), np.asarray(Sigma), np.asarray(B), np.asarray(mean))


@partial(jax.jit, static_argnames=("r", "n_iter", "dtype"))
def _rsvd(
    Qc: DeviceArray, r: int, n_iter: int, seed: int, dtype: DTypeLike
) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    """Computes a randomized SVD with power iterations.

    Follows Halko et al. (2011), re-orthonormalizing after each multiplication.
    Implements the same algorithm as `epod._pod_randomized`, so the two agree to
    the tolerance of the sketch rather than to machine precision.

    Args:
      Qc: Mean-subtracted data matrix, shape `(N_x, N_t)`.
      r: Modes to keep.
      n_iter: Power iterations.
      seed: Random seed for the sketch.
      dtype: JAX scalar type to compute in.

    Returns:
      `(Psi, Sigma, Vt)` on device, shaped `(N_x, r)`, `(r,)`, and `(r, N_t)`.
    """
    key = jax.random.PRNGKey(seed)
    n_t = Qc.shape[1]
    Om: DeviceArray = jax.random.normal(key, (n_t, min(r + 10, n_t)), dtype)
    Y: DeviceArray = Qc @ Om
    Y, _ = jnp.linalg.qr(Y)

    for _ in range(n_iter):
        Z, _ = jnp.linalg.qr(Qc.T @ Y)
        Y, _ = jnp.linalg.qr(Qc @ Z)

    Ub, Sig, Vt = _svd(Y.T @ Qc)
    return (Y @ Ub)[:, :r], Sig[:r], Vt[:r]


def extended_pod_jax(B: FloatArray, S: FloatArray, dtype: Dtype = "float64") -> FloatArray:
    """Computes extended POD modes (Borée 2003).

    Extracts the part of a synchronized signal `S` that correlates with the field
    POD coefficients `B`. Mode `k` describes what the sensor array does while
    field mode `k` is active:

        psi_ext_k = S b_k / ||b_k||^2

    Args:
      B: Field POD coefficients, shape `(r, N_t)`.
      S: Synchronized signal, shape `(N_s, N_t)`.
      dtype: Precision to compute in.

    Returns:
      Extended modes, shape `(N_s, r)`.

    Raises:
      ValueError: If either array is not two-dimensional, or if `B` and `S` hold
        different numbers of snapshots.
    """
    dt = _dt(dtype)
    Bj: DeviceArray = jnp.asarray(B, dt)
    Sj: DeviceArray = jnp.asarray(S, dt)
    _check_coefficients(Bj, Sj, "S")

    norms: DeviceArray = jnp.einsum("kt,kt->k", Bj, Bj)
    return np.asarray((Sj @ Bj.T) / jnp.where(norms > 0, norms, 1.0))


@partial(jax.jit, static_argnames=("dtype",))
def _lse_map(B: DeviceArray, C: DeviceArray, ridge: float, dtype: Dtype) -> DeviceArray:
    """Solves the ridge-regularized normal equations on device.

    Args:
      B: Target coefficients, shape `(r_f, N_t)`.
      C: Regressor coefficients, shape `(r_s, N_t)`.
      ridge: Tikhonov parameter, in units of sensor coefficient energy.
      dtype: Precision name. Static, so each precision compiles separately.

    Returns:
      The map `M` on device, shape `(r_f, r_s)`.
    """
    del dtype
    G: DeviceArray = C @ C.T
    G = G + ridge * jnp.eye(G.shape[0], dtype=G.dtype)
    solved = cast("DeviceArray", jnp.linalg.solve(G, (B @ C.T).T))
    return solved.T


def lse_map_jax(B: FloatArray, C: FloatArray, ridge: float = 0.0, dtype: Dtype = "float64") -> FloatArray:
    """Fits the ridge-regularized least-squares map from `C` to `B`.

    Computes `M = B C.T (C C.T + ridge I)^-1` by solving the normal equations.

    Args:
      B: Target coefficients, shape `(r_f, N_t)`.
      C: Regressor coefficients, shape `(r_s, N_t)`.
      ridge: Tikhonov parameter, in units of sensor coefficient energy.
      dtype: Precision to compute in.

    Returns:
      The map `M`, shape `(r_f, r_s)`.

    Raises:
      ValueError: If either array is not two-dimensional, if `B` and `C` hold
        different numbers of snapshots, or if `ridge` is negative.
    """
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    dt = _dt(dtype)
    Bj: DeviceArray = jnp.asarray(B, dt)
    Cj: DeviceArray = jnp.asarray(C, dt)
    _check_coefficients(Bj, Cj, "C")
    M = cast("DeviceArray", _lse_map(Bj, Cj, float(ridge), dtype))
    return np.asarray(M)


# ── Ridge sweep ───────────────────────────────────────────────────────────────


def ridge_cv_jax(
    B: FloatArray,
    C: FloatArray,
    B_val: FloatArray,
    C_val: FloatArray,
    ridges: Sequence[float],
    dtype: Dtype = "float64",
) -> tuple[float, FloatArray]:
    """Scores every candidate ridge on a validation block in one batched solve.

    Forms the Gram matrix once and maps only the solve over `ridges`.

    Args:
      B: Training target coefficients, shape `(r_f, N_t)`.
      C: Training regressor coefficients, shape `(r_s, N_t)`.
      B_val: Validation target coefficients, shape `(r_f, N_v)`.
      C_val: Validation regressor coefficients, shape `(r_s, N_v)`.
      ridges: Candidate ridge values.
      dtype: Precision to compute in.

    Returns:
      The ridge with the lowest validation NMSE, and the validation NMSE of every
      candidate, in the order given.

    Raises:
      ValueError: If `ridges` is empty.
    """
    if len(ridges) == 0:
        raise ValueError("ridges must hold at least one candidate")
    dt = _dt(dtype)
    Bj: DeviceArray = jnp.asarray(B, dt)
    Cj: DeviceArray = jnp.asarray(C, dt)
    Bv: DeviceArray = jnp.asarray(B_val, dt)
    Cv: DeviceArray = jnp.asarray(C_val, dt)
    G: DeviceArray = Cj @ Cj.T
    RHS: DeviceArray = (Bj @ Cj.T).T
    eye: DeviceArray = jnp.eye(G.shape[0], dtype=dt)
    var: DeviceArray = jnp.var(Bv)

    def one(lam: DeviceArray) -> DeviceArray:
        """Scores one ridge value against the validation block.

        Args:
          lam: The ridge value.

        Returns:
          Validation NMSE, as a scalar on device.
        """
        M = cast("DeviceArray", jnp.linalg.solve(G + lam * eye, RHS)).T
        err: DeviceArray = jnp.mean((Bv - M @ Cv) ** 2) / jnp.where(var > 0, var, 1.0)
        return err

    errs = np.asarray(jax.vmap(one)(jnp.asarray(ridges, dt)))
    return float(np.asarray(ridges)[int(np.argmin(errs))]), errs


# ── Estimators ────────────────────────────────────────────────────────────────


@dataclass
class _Base:
    """Sensor preprocessing and state checks shared by both JAX estimators.

    Attributes:
      r_sensor: POD modes kept for the sensor record. `None` keeps all `N_s`.
      ridge: Tikhonov parameter, relative to the mean sensor coefficient energy.
      standardise_sensors: Divide each channel by its training standard
        deviation before the sensor decomposition.
      dtype: Precision to compute in.
      s_mean: Per-channel sensor mean, learned by `fit`.
      s_scale: Per-channel sensor scale, learned by `fit`.
      fitted: Whether `fit` has completed.
    """

    if TYPE_CHECKING:
        # The subclasses define `predict`. Declaring it here as an annotation
        # would make it a constructor field of each dataclass subclass.
        def predict(self, S_new: FloatArray) -> FloatArray:
            """Reconstructs fields from sensor data.

            Args:
              S_new: Sensor record, shape `(N_s, N_t)`.

            Returns:
              The reconstructed field, shape `(N_x, N_t)`.
            """
            ...

    r_sensor: int | None = None
    ridge: float = 0.0
    standardise_sensors: bool = True
    dtype: Dtype = "float64"
    s_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    def _prep(self, S: FloatArray, learn: bool) -> FloatArray:
        """Centers and scales a sensor record.

        Args:
          S: Sensor record, shape `(N_s, N_t)`.
          learn: Measure the statistics from `S` and store them, rather than
            reusing the fitted ones.

        Returns:
          The standardized record, shape `(N_s, N_t)`.

        Raises:
          ValueError: If `learn` is `False` and `S` has a different channel count
            from the fitted statistics.
        """
        if learn:
            self.s_mean, self.s_scale = sensor_stats(S, self.standardise_sensors)
        return apply_sensor_stats(S, self.s_mean, self.s_scale)

    def _check(self) -> None:
        """Raises if the estimator has no fitted state.

        Checks the fitted sensor statistics as well as the `fitted` flag, which
        is a constructor argument and can be set without fitting.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if not self.fitted or self.s_mean.size == 0:
            raise RuntimeError("call fit() before predict()/encode()/score()")

    def _check_fit_inputs(self, Q: FloatArray, S: FloatArray) -> None:
        """Checks that a field and a sensor record can be fitted together.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Sensor record, shape `(N_s, N_t)`.

        Raises:
          ValueError: If `ridge` is negative, if the two records hold different
            snapshot counts, or if either holds non-finite values.
        """
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}")

        if Q.shape[1] != S.shape[1]:
            raise ValueError(
                f"field has {Q.shape[1]} snapshots, sensors have "
                f"{S.shape[1]}; these must be synchronised sample-for-sample"
            )
        if not np.isfinite(Q).all() or not np.isfinite(S).all():
            raise ValueError("non-finite values in the training data")

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


@dataclass
class PODLSEJax(_Base):
    """Estimates a field from sparse sensors by POD-LSE, computed on device.

    Mirrors `epod.PODLSE`, ranking sensor directions by POD.

    Attributes:
      r_field: POD modes kept for the field. `None` keeps all of them.
      pod_method: Algorithm `pod_jax` uses for the field.
      seed: Random seed, forwarded to `pod_jax`.
      Psi: Field modes, shape `(N_x, r_field)`, learned by `fit`.
      Sigma: Field singular values, shape `(r_field,)`, learned by `fit`.
      Phi: Sensor modes, shape `(N_s, r_sensor)`, learned by `fit`.
      M: Map from sensor coefficients to field coefficients, learned by `fit`.
      q_mean: Field mean, shape `(N_x, 1)`, learned by `fit`.
    """

    r_field: int | None = None
    pod_method: str = "auto"
    seed: int = 0
    Psi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Sigma: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    M: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)

    def fit(self, Q: FloatArray, S: FloatArray) -> PODLSEJax:
        """Fits the field basis, the sensor basis, and the map between them.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          This estimator, fitted.

        Raises:
          ValueError: If `ridge` is negative, if `Q` and `S` hold different
            numbers of snapshots, if either holds non-finite values, or if
            `pod_method` or a rank option is invalid.
        """
        Q, S = np.asarray(Q, np.float64), np.asarray(S, np.float64)
        self._check_fit_inputs(Q, S)

        self.Psi, self.Sigma, B, self.q_mean = pod_jax(
            Q, self.r_field, True, self.pod_method, seed=self.seed, dtype=self.dtype
        )
        Sc = self._prep(S, learn=True)
        self.Phi, _, C, _ = pod_jax(Sc, self.r_sensor, subtract_mean=False, method="svd", dtype=self.dtype)
        # `ridge` is relative to the mean sensor coefficient energy.
        energy: FloatArray = np.einsum("kt,kt->k", C, C)
        lam = self.ridge * float(np.mean(energy))
        self.M = lse_map_jax(B, C, lam, self.dtype)
        self.fitted = True
        return self

    def encode(self, S_new: FloatArray) -> FloatArray:
        """Predicts field POD coefficients from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          Predicted field POD coefficients, shape `(r_field, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check()
        return self.M @ (self.Phi.T @ self._prep(np.asarray(S_new, np.float64), False))

    def predict(self, S_new: FloatArray) -> FloatArray:
        """Reconstructs fields from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          The reconstructed field, shape `(N_x, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check()
        return self.Psi @ self.encode(S_new) + self.q_mean

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
        self._check()
        return self.Psi.T @ (np.asarray(Q, np.float64) - self.q_mean)

    def floor(self, Q: FloatArray) -> float:
        """Returns the best NMSE this field basis allows, independent of sensors.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.

        Returns:
          The truncation error of `Psi` on `Q`, as a normalized MSE.

        Raises:
          RuntimeError: If `fit` has not been called.
          ValueError: If `Q` has a different row count from `Psi`.
        """
        self._check()
        return projection_floor(np.asarray(Q, np.float64), self.Psi, self.q_mean)

    @property
    def n_params(self) -> int:
        """Number of free parameters in the coefficient map."""
        return int(self.M.size)


@dataclass
class ExtendedPODJax(_Base):
    """Estimates a field through extended POD modes, computed on device.

    Mirrors `epod.ExtendedPOD`, and is equivalent to POD-LSE with the field
    basis left untruncated. Each extended mode is the flow associated with one
    sensor mode.

    Attributes:
      Psi_ext: Extended modes, shape `(N_x, r_sensor)`, learned by `fit`.
      Phi: Sensor modes, shape `(N_s, r_sensor)`, learned by `fit`.
      q_mean: Field mean, shape `(N_x, 1)`, learned by `fit`.
    """

    Psi_ext: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)

    def fit(self, Q: FloatArray, S: FloatArray) -> ExtendedPODJax:
        """Fits the sensor basis and one extended field mode per sensor mode.

        Args:
          Q: Field snapshots, shape `(N_x, N_t)`.
          S: Synchronized sensor record, shape `(N_s, N_t)`.

        Returns:
          This estimator, fitted.

        Raises:
          ValueError: If `ridge` is negative, if `Q` and `S` hold different
            numbers of snapshots, if either holds non-finite values, or if
            `r_sensor` is below 1.
        """
        Q, S = np.asarray(Q, np.float64), np.asarray(S, np.float64)
        self._check_fit_inputs(Q, S)
        self.q_mean = Q.mean(axis=1, keepdims=True)
        Sc = self._prep(S, learn=True)
        self.Phi, _, C, _ = pod_jax(Sc, self.r_sensor, subtract_mean=False, method="svd", dtype=self.dtype)
        dt = _dt(self.dtype)
        Cj = jnp.asarray(C, dt)
        G = Cj @ Cj.T
        lam = self.ridge * float(jnp.mean(jnp.einsum("kt,kt->k", Cj, Cj)))
        Qc = jnp.asarray(Q - self.q_mean, dt)
        # `Psi_ext = Qc C.T (C C.T + lam I)^-1`. `solve` takes the `r_s` axis
        # first, so this solves for the transpose.
        solved = cast(
            "DeviceArray",
            jnp.linalg.solve(G + lam * jnp.eye(G.shape[0], dtype=dt), (Qc @ Cj.T).T),
        )
        self.Psi_ext = np.asarray(solved.T)
        self.fitted = True
        return self

    def encode(self, S_new: FloatArray) -> FloatArray:
        """Projects sensor data onto the fitted sensor basis.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          Sensor POD coefficients, shape `(r_sensor, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check()
        return self.Phi.T @ self._prep(np.asarray(S_new, np.float64), False)

    def predict(self, S_new: FloatArray) -> FloatArray:
        """Reconstructs fields from sensor data.

        Args:
          S_new: Sensor record, shape `(N_s, N_t)`.

        Returns:
          The reconstructed field, shape `(N_x, N_t)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._check()
        return self.Psi_ext @ self.encode(S_new) + self.q_mean

    @property
    def n_params(self) -> int:
        """Number of free parameters in the extended modes."""
        return int(self.Psi_ext.size)
