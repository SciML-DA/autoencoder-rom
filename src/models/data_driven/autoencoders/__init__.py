"""Dimensionality-reduction building blocks for data-driven ROMs.

Only 2D snapshot data is supported for now, but the API is designed to be
extensible to 3D and multi-field data.

All projectors -- linear or nonlinear -- share the same sklearn-style API:

    p.fit(X)         -- learn the representation from data  X (N_x, N_t)
    p.encode(X)      -- X (N_x, N_t) --> Z (N_latent, N_t)
    p.decode(Z)      -- Z (N_latent, N_t) --> X_hat (N_x, N_t)
    p.reconstruct(X) -- full round-trip
    p.score(X)       -- mean squared reconstruction error
    p.N_latent       -- size of the latent (bottleneck) space

Class hierarchy
---------------

    Projector (ABC)      shared interface + N_latent + reconstruct/score  [here]
    |-- POD(Projector)   Proper Orthogonal Decomposition (linear)      [pod.py]
    |     N_latent == N_modes retained
    |     Sigma, Psi, Phi   decomposition results
    |-- SPOD(POD)        Spectral POD (Sieber et al. JFM 2016)         [pod.py]
    |     inherits all POD helpers; only _decompose is overridden
    |-- AE(Projector)    Fully-connected autoencoder (MLP, PyTorch)     [ae.py]
    |-- CAE(Projector)   Convolutional autoencoder (PyTorch)            [ae.py]
    |-- AEJax / CAEJax   The same two, hand-written in JAX          [ae_jax.py]

These are pure dimensionality-reduction tools -- they have no temporal
forecaster. Combine with an ESN or LSTM in ``models/data_driven/`` to build a
complete ROM.

Import boundary
---------------
``Projector``, ``POD`` and ``SPOD`` are pure numpy. ``AE``/``CAE`` need torch
and ``AEJax``/``CAEJax`` need jax, so those four are resolved lazily by the
module-level ``__getattr__`` below: importing this package, or naming only the
linear projectors, never imports either framework. That keeps a POD-only
workflow cheap and keeps ``ae.py`` liftable into a torch-free package as an
optional extra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = ["Projector", "POD", "SPOD", "AE", "CAE", "AEJax", "CAEJax",
           "spod_towne", "print_spod_towne_summary"]


class Projector(ABC):
    """
    Abstract base for all dimensionality-reduction building blocks,
    both linear (POD, SPOD) and nonlinear (AE, CAE).

    Every projector exposes:

        N_latent  -- size of the latent / bottleneck space. In POD/SPOD, this is the number of modes retained.
        fit       -- learn the representation from data
        encode    -- map state space X --> latent Z
        decode    -- map latent Z --> reconstructed state X_hat
        reconstruct, score, copy -- provided as concrete methods
    """

    N_latent: int = 20
    fitted: bool = False
    _Q_mean: Optional[np.ndarray] = None

    # ── shared geometry ──────────────────────────────────────────────────
    # Both the grid helpers here (`_to_physical_grid`, `preprocess_snapshot`)
    # and sensor placement (`SensorPlacementMixin`) need these, so they are
    # part of the Projector contract rather than POD-specific. `fit` on raw
    # grid input fills grid_shape/fluid_mask_flat in; `domain` is caller-set.
    grid_shape: Optional[tuple] = None   # (Nu, Nx, Ny)
    domain: Optional[list] = None        # [x0, x1, y0, y1]
    fluid_mask_flat: Optional[np.ndarray] = None

    @property
    def Q_mean(self) -> np.ndarray:
        if self._Q_mean is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Q_mean

    @Q_mean.setter
    def Q_mean(self, value: np.ndarray) -> None:
        self._Q_mean = value

    @abstractmethod
    def fit(self, X: np.ndarray) -> Projector:
        """Learn the projection from data X (N_x, N_t). Returns self."""

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Map X (N_x, N_t) to latent representation Z (N_latent, N_t)."""

    @abstractmethod
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Map latent Z (N_latent, N_t) back to state space Q_hat (N_x, N_t)."""

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full round-trip: encode then decode."""
        return self.decode(self.encode(X))

    def score(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error ||X - reconstruct(X)||^2 / N."""
        return float(np.mean((X - self.reconstruct(X)) ** 2))

    # -------
    # what a ROM needs from a projector: a spatial basis to place sensors on,
    # and a way to read the decoded field at those sensors
    # -------

    def decode_at(self, Z: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode ``Z`` and read out only the rows ``idx`` (the sensors).

        The generic route decodes the whole field and then indexes it, because
        a nonlinear decoder cannot produce a subset of its outputs. `POD`
        overrides this with the cheap linear route (``Psi[idx] @ Z``).
        """
        X_hat = self.decode(Z)
        return X_hat if idx is None else X_hat[idx]

    def spatial_basis(self, z0: Optional[np.ndarray] = None) -> np.ndarray:
        r"""``(N_x, r)`` -- r spatial fields whose span locally approximates the
        reconstruction manifold, in the same flat masked layout as `decode`
        output. QR column-pivoting on this is what places sensors.

        For a linear projector this is exact and constant: `POD` overrides it to
        return :math:`\Psi`. For a nonlinear one the manifold has no global
        basis, so the local one at ``z0`` is used -- the decoder Jacobian
        :math:`\partial \hat{x} / \partial z`, whose columns are the directions
        state space moves in when each latent coordinate is perturbed. That is
        the nonlinear analogue of a POD mode.

        Computed by central differences on the *public* ``decode``, not by
        autodiff: it then works identically for the torch and JAX projectors and
        for any future one, and keeps this base class free of either framework.
        Cost is ``2 * N_latent`` single-column decodes, once.

        Parameters
        ----------
        z0 : (N_latent,) array, optional
            Latent state to linearise about. Defaults to the origin. Callers
            that have the training latents should pass their mean -- the basis
            is only representative near where the trajectory actually lives.
        """
        n = self.N_latent
        z0 = np.zeros(n) if z0 is None else np.asarray(z0, dtype=float).ravel()
        if z0.size != n:
            raise ValueError(f"z0 has {z0.size} entries, expected N_latent={n}")

        # step per coordinate: scaled to the latent's own magnitude so the
        # difference is neither swamped by round-off nor large enough to leave
        # the locally-linear regime
        eps = 1e-4 * np.maximum(np.abs(z0), 1.0)

        cols = []
        for k in range(n):
            zp, zm = z0.copy(), z0.copy()
            zp[k] += eps[k]
            zm[k] -= eps[k]
            dp = self.decode(zp[:, np.newaxis])
            dm = self.decode(zm[:, np.newaxis])
            cols.append((dp - dm).ravel() / (2.0 * eps[k]))
        return np.column_stack(cols)  # (N_x, N_latent)

    # -------
    # utilities for grid handling
    # -------

    def _to_physical_grid(self, X_hat: np.ndarray) -> np.ndarray:
        """Map flat (N_fluid*Nu, N_t) back to (Nu, N_t, Nx, Ny) - exact inverse of _to_flat."""
        Nu, Nx, Ny = self.grid_shape
        if X_hat.ndim == 1:
            X_hat = X_hat[:, np.newaxis]
        Nt = X_hat.shape[1]
        N_fluid = int(self.fluid_mask_flat.sum())

        out = np.full((Nu, Nt, Nx, Ny), np.nan)

        # Inverse the flatten: (N_fluid*Nu, Nt) → (N_fluid, Nu, Nt) → (Nu, Nt, N_fluid)
        X_unflatten = X_hat.reshape(N_fluid, Nu, Nt).transpose(
            1, 2, 0
        )  # (Nu, Nt, N_fluid)

        for u in range(Nu):
            # X_unflatten[u] is (Nt, N_fluid) — all time steps for field u
            grid_flat = np.full((Nt, Nx * Ny), np.nan)
            grid_flat[:, self.fluid_mask_flat] = X_unflatten[
                u
            ]  # Place fluid values back

            # Reshape (Nt, Nx*Ny) → (Nt, Nx, Ny) and assign
            out[u] = grid_flat.reshape(Nt, Nx, Ny)

        return out[:, 0] if Nt == 1 else out

    def _to_flat(self, X: np.ndarray) -> np.ndarray:
        """Map raw grid input (Nu, Nt, Nx, Ny) to flat (N_fluid * n_fields, N_t."""
        X_masked = X.reshape(X.shape[0], X.shape[1], -1)[
            :, :, self.fluid_mask_flat
        ]  # (Nu, Nt, N_fluid)

        return X_masked.transpose(2, 0, 1).reshape(
            -1, X.shape[1]
        )  # (N_fluid * n_fields, N_t)

    # -----
    # preprocessing for raw grid input.
    # Note: could implement different ones including normalization/standardization.
    # -----

    def preprocess_snapshot(self, X: np.ndarray, subtract_mean=True):
        """
        Build the zero-mean data matrix Q from raw snapshot fields,
        automatically detecting and removing NaN-masked solid-body points.

        Parameters
        ----------
        X : ndarray
            Raw snapshot data, either as a single field (N_t, Nx, Ny) or a list of fields (Nu, N_t, Nx, Ny).

        subtract_mean : bool
            If True (default), subtract the temporal mean row-wise.

        Returns
        -------
        Q          : ndarray (N_fluid * n_fields, N_t)   zero-mean data matrix for decomposition
        """

        if not self.fitted:
            # if the input is raw grid data, we need to detect the fluid points and flatten the data
            assert (
                X.ndim == 4
            ), f"Expected raw grid input with 4 dimensions, got {X.ndim}."
            Nu, Nt, Nx, Ny = X.shape
            self.grid_shape = (Nu, Nx, Ny)

            ref = X[0]
            fluid_mask = ~np.isnan(ref[0])
            self.fluid_mask_flat = fluid_mask.ravel()

            X_masked_flat = self._to_flat(X)  # (N_fluid * n_fields, N_t)

            if subtract_mean:
                self.Q_mean = X_masked_flat.mean(axis=1, keepdims=True)
            else:
                self.Q_mean = np.zeros_like(X_masked_flat[:, :1])

            Q = X_masked_flat - self.Q_mean  # shape (N_fluid * n_fields, N_t)
            # store the total kinetic energy for later use in relative error metrics
            self._TKE = 0.5 * float(np.sum(np.mean(Q**2, axis=1)))
            return Q

        elif X.shape[0] != self.Q_mean.shape[0]:

            # if the decomosition is already fitted, can expect 1 snapshot only
            assert X.ndim in (
                3,
                4,
            ), f"Expected flat input with 2, 3 or 4 dimensions, got {X.ndim}."
            if X.ndim == 3:
                X = X[:, np.newaxis]  # (n_fields, 1, Nx, Ny)

            # check grid
            Nu, _, Nx, Ny = X.shape
            grid_shape = (Nu, Nx, Ny)
            assert (
                grid_shape == self.grid_shape
            ), f"Expected grid shape {self.grid_shape}, got {grid_shape}."

            X_masked_flat = self._to_flat(X)  # (N_fluid * n_fields, N_t)
            return X_masked_flat - self.Q_mean  # shape (N_fluid * n_fields, N_t)
        else:
            # already flat input, just check dimensions and remove mean
            assert X.ndim == 2, f"Expected flat input with 2 dimensions, got {X.ndim}."

            return X - self.Q_mean  # shape (N_fluid * n_fields, N_t)

    def _field_scale(self, Q: np.ndarray) -> np.ndarray:
        """
        Per-field standardization scale for the autoencoders.

        Returns a column vector (N_x, 1) holding the std of each field (U, V,
        W, ...) broadcast over its rows, so encode/decode standardize each
        field by its own std
        """
        Nu = self.grid_shape[0]
        scale = np.ones((Q.shape[0], 1), dtype=Q.dtype)
        for u in range(Nu):
            s = float(Q[u::Nu].std())
            scale[u::Nu, 0] = s if s > 0 else 1.0
        return scale


# ────────────────────────────────────────────────────────────────────────────
# Nonlinear Autoencoders -- UROP project
# ────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Lazy re-exports. POD/SPOD are cheap and torch-free, so they could be imported
# eagerly -- but resolving every projector through one path keeps the boundary
# obvious and stops an eager `from .ae import AE` creeping back in later.
_LAZY = {
    "POD": ".pod",
    "SPOD": ".pod",
    "AE": ".ae",
    "CAE": ".ae",
    "AEJax": ".ae_jax",
    "CAEJax": ".ae_jax",
    "spod_towne": ".pod_utils",
    "print_spod_towne_summary": ".pod_utils",
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access -- imports torch/jax only on first use."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
