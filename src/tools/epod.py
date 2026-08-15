"""
epod.py
=======

Reference implementation of the linear baseline in Novoa's notes: POD combined
with linear stochastic estimation (POD-LSE), and the closely-related extended
POD (EPOD, Borée 2003) used on the "snapshot-to-snapshot nowcasting" slides.

This exists to be a *verification oracle*, not a research contribution. It is
written the short, obvious way so that when your own implementation disagrees
with it, the reference is the thing you trust. Everything is dense NumPy and
runs in well under a second on the synthetic cases.

Method (notes section 2.1)
--------------------------
Stage 1, decomposition. Snapshot POD of the field snapshots and, separately,
of the sensor record::

    phi = Psi B          Psi (N_x, r_f)   B (r_f, N_t)
    s   = Phi C          Phi (N_s, r_s)   C (r_s, N_t)

Stage 2, linear map. Least squares from sensor coefficients to field
coefficients, ridge-regularised::

    M = argmin ||B - M C||_F^2 + lam ||M||_F^2  =  B C^T (C C^T + lam I)^-1

Prediction for new sensor data::

    phi_hat_new = Psi M Phi^T s_new

Conventions match ``tools/pod_spod.py``:

    Psi   (N_x, r)     spatial modes, orthonormal columns
    B     (r, N_t)     temporal coefficients, B = Psi^T Q
    Sigma (r,)         singular values, descending

All inputs are *column-per-snapshot*: ``Q`` is (N_x, N_t) and ``S`` is
(N_s, N_t). Note that this is the transpose of the (N_t, N_s) layout that
``wake_synthetic`` returns, because POD here is column-oriented throughout --
pass ``S.T``.

Usage
-----
::

    from tools.epod import PODLSE

    model = PODLSE(r_field=20, r_sensor=12, ridge=1e-6).fit(Q_tr, S_tr)
    Q_hat = model.predict(S_te)
    model.score(Q_te, S_te)          # normalised MSE, 0 is perfect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["PODLSE", "pod", "lse_map", "extended_pod", "nmse", "split_train_test"]


# ── plain POD ─────────────────────────────────────────────────────────────────


def pod(Q: np.ndarray, r: Optional[int] = None, subtract_mean: bool = False):
    """Snapshot POD via SVD. Q is (N_x, N_t), zero-mean unless told otherwise.

    Returns (Psi, Sigma, B, mean).

    SVD rather than an eigendecomposition of Q^T Q: the correlation-matrix
    route squares the condition number, which costs you roughly half your
    significant digits in the trailing modes. That does not matter for the
    leading few modes, but the whole point here is to be the thing you check
    against, so it is worth the extra flops.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError(f"Q must be (N_x, N_t), got shape {Q.shape}")
    mean = Q.mean(axis=1, keepdims=True) if subtract_mean else np.zeros((Q.shape[0], 1))
    Qc = Q - mean

    Psi, Sigma, Vt = np.linalg.svd(Qc, full_matrices=False)
    if r is not None:
        Psi, Sigma, Vt = Psi[:, :r], Sigma[:r], Vt[:r]
    B = Sigma[:, None] * Vt  # == Psi.T @ Qc, but without the extra matmul
    return Psi, Sigma, B, mean


def extended_pod(B: np.ndarray, S: np.ndarray):
    """Extended POD modes (Borée 2003): the part of ``S`` correlated with ``B``.

    Given field POD coefficients B (r, N_t) and any other synchronised signal
    S (N_s, N_t), the extended mode for field mode k is the S-field weighted by
    b_k over the record::

        psi_ext_k = (1 / ||b_k||^2) * S b_k

    So ``psi_ext_k`` answers "what does the sensor array do when field mode k
    is active?". It is the forward direction on the nowcasting slide; the LSE
    map below is the inverse direction.
    """
    B, S = np.asarray(B, np.float64), np.asarray(S, np.float64)
    if B.shape[1] != S.shape[1]:
        raise ValueError(f"B has {B.shape[1]} snapshots but S has {S.shape[1]}")
    norms = np.einsum("kt,kt->k", B, B)
    norms = np.where(norms > 0, norms, 1.0)
    return (S @ B.T) / norms  # (N_s, r)


# ── the linear map ────────────────────────────────────────────────────────────


def lse_map(B: np.ndarray, C: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """M = B C^T (C C^T + ridge I)^-1, the ridge-regularised LSE map.

    ``ridge`` is in units of the *sensor coefficient energy*, i.e. it is
    compared against the eigenvalues of C C^T. Scale it relative to
    ``trace(C C^T) / r_s`` if you want it dimensionless -- ``PODLSE`` does.

    ``solve`` on the normal equations rather than forming the inverse: same
    arithmetic, better conditioned, and it raises on a singular system instead
    of returning quiet garbage.
    """
    B, C = np.asarray(B, np.float64), np.asarray(C, np.float64)
    if B.shape[1] != C.shape[1]:
        raise ValueError(f"B has {B.shape[1]} snapshots but C has {C.shape[1]}")
    G = C @ C.T
    if ridge:
        G = G + ridge * np.eye(G.shape[0])
    # M G = B C^T  ->  solve G^T M^T = (B C^T)^T; G is symmetric so G^T = G
    return np.linalg.solve(G, (B @ C.T).T).T


# ── the estimator ─────────────────────────────────────────────────────────────


@dataclass
class PODLSE:
    """POD-LSE reconstruction of a field from synchronised sparse sensors.

    Parameters
    ----------
    r_field : int or None
        POD modes kept for the field. None keeps everything.
    r_sensor : int or None
        POD modes kept for the sensor record. None keeps all N_s. Truncating
        here is the main defence against ill-conditioned load cells.
    ridge : float
        Tikhonov parameter, *relative* to the mean sensor coefficient energy,
        so the same value means the same thing across datasets.
    standardise_sensors : bool
        Divide each channel by its training std before the sensor POD. Six-
        component load cells mix newtons and newton-metres; without this the
        sensor POD is dominated by whichever channel has the larger units.
    """

    r_field: Optional[int] = None
    r_sensor: Optional[int] = None
    ridge: float = 0.0
    standardise_sensors: bool = True

    # -- learned ---------------------------------------------------------------
    Psi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Sigma: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    Phi: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    M: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    q_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fitted: bool = False

    # -- api -------------------------------------------------------------------

    def fit(self, Q: np.ndarray, S: np.ndarray) -> "PODLSE":
        """Q (N_x, N_t) field snapshots, S (N_s, N_t) synchronised sensors."""
        Q = np.asarray(Q, np.float64)
        S = np.asarray(S, np.float64)
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

        self.Psi, self.Sigma, B, self.q_mean = pod(
            Q, self.r_field, subtract_mean=True
        )

        Sc, self.s_mean, self.s_scale = self._prep_sensors(S, learn=True)
        self.Phi, _, C, _ = pod(Sc, self.r_sensor, subtract_mean=False)

        lam = self.ridge * float(np.mean(np.einsum("kt,kt->k", C, C)))
        self.M = lse_map(B, C, lam)
        self.fitted = True
        return self

    def predict(self, S_new: np.ndarray) -> np.ndarray:
        """Reconstruct fields from sensors. S_new (N_s, N_t) -> (N_x, N_t)."""
        self._check_fitted()
        Sc, _, _ = self._prep_sensors(np.asarray(S_new, np.float64), learn=False)
        return self.Psi @ (self.M @ (self.Phi.T @ Sc)) + self.q_mean

    def encode(self, S_new: np.ndarray) -> np.ndarray:
        """Predicted field POD coefficients B_hat (r_field, N_t).

        This is the quantity the two-branch autoencoder's G(s) replaces, so it
        is the fair thing to compare a learned sensor branch against.
        """
        self._check_fitted()
        Sc, _, _ = self._prep_sensors(np.asarray(S_new, np.float64), learn=False)
        return self.M @ (self.Phi.T @ Sc)

    def score(self, Q_true: np.ndarray, S_new: np.ndarray) -> float:
        """Normalised MSE of the reconstruction. 0 is perfect, 1 is the mean."""
        return nmse(np.asarray(Q_true, np.float64), self.predict(S_new))

    # -- internals -------------------------------------------------------------

    def _prep_sensors(self, S, learn: bool):
        if learn:
            mean = S.mean(axis=1, keepdims=True)
            scale = S.std(axis=1, keepdims=True) if self.standardise_sensors else None
            if scale is not None:
                scale = np.where(scale > 0, scale, 1.0)
        else:
            mean = self.s_mean
            scale = self.s_scale if self.s_scale.size else None
            if S.shape[0] != mean.shape[0]:
                raise ValueError(
                    f"fitted on {mean.shape[0]} channels, got {S.shape[0]}"
                )
        Sc = S - mean
        if scale is not None:
            Sc = Sc / scale
        return Sc, mean, (scale if scale is not None else np.empty(0))

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("call fit() before predict()/encode()/score()")


# ── metrics and splitting ─────────────────────────────────────────────────────


def nmse(Q_true: np.ndarray, Q_hat: np.ndarray) -> float:
    """Mean squared error normalised by the variance of the truth.

    Normalising by the *fluctuation* energy, not by ||Q||^2, so that predicting
    the temporal mean scores exactly 1.0 whatever the mean happens to be. A
    method that cannot beat 1.0 has learned nothing.
    """
    Q_true = np.asarray(Q_true, np.float64)
    Q_hat = np.asarray(Q_hat, np.float64)
    if Q_true.shape != Q_hat.shape:
        raise ValueError(f"shape mismatch: {Q_true.shape} vs {Q_hat.shape}")
    var = np.var(Q_true - Q_true.mean(axis=1, keepdims=True))
    return float(np.mean((Q_true - Q_hat) ** 2) / var) if var > 0 else float("inf")


def split_train_test(n_t: int, test_fraction: float = 0.25, gap: int = 0):
    """Contiguous train/test split with an optional guard band.

    Random splits leak: PIV at 1 kHz oversamples a wake by more than an order
    of magnitude, so a randomly held-out snapshot has near-copies of itself in
    the training set and every method looks excellent. The split is contiguous
    for that reason, and ``gap`` drops samples at the seam so the two halves
    are separated by more than one correlation time.
    """
    n_te = int(round(n_t * test_fraction))
    n_tr = n_t - n_te - gap
    if n_tr <= 0:
        raise ValueError(f"nothing left to train on: n_t={n_t}, gap={gap}")
    return np.arange(n_tr), np.arange(n_t - n_te, n_t)
