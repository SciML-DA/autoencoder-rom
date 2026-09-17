"""Linear projectors: POD and SPOD.

Split out of the former ``tools/autoencoders.py`` so that the projector
package mirrors romda's ``models/data_driven/autoencoders/`` layout.

Deliberately torch-free -- importing this module must never pull in torch,
so that a POD-only workflow (and romda, which has no torch dependency) can
use it unchanged. See ``__init__.py`` for the lazy-import boundary.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import Projector
from .pod_utils import snapshot_pod, snapshot_pod_randomized, spod_sieber

__all__ = ["POD", "SPOD"]


class POD(Projector):
    """
    Snapshot POD.

    Inherits the shared interface from ``Projector`` (fit/encode/decode/
    reconstruct/score/copy/N_latent) and adds linear-specific attributes
    and geometry helpers.

    Two solvers are available via ``method``:

    ``'exact'``
        Full eigendecomposition of the temporal correlation matrix C = Q^T Q / N_t.
        Returns all N_t modes.  Exact but O(N_t^3).

    ``'randomized'``  (default)
        Randomized SVD (Halko, Martinsson & Tropp 2011).  Returns only the
        leading ``n_modes`` modes.  Fast and memory-efficient for large data.

    After ``fit(X)`` the following attributes are available:

        Sigma  (N_latent,)      singular values, descending
        Psi    (N_x, N_latent)  spatial modes (orthonormal columns)
        Phi    (N_latent, N_t)  temporal coefficients from training data
        Q_mean (N_x, 1)         temporal mean

        Note:
            X = Psi @ Phi + Q_mean  (N_x, N_t)  is the training data reconstruction.
            Q = X - Q_mean is the zero-mean data used for the decomposition.

    Parameters
    ----------
    n_modes      : int         Modes / latent-space size.  Default: 20.
    method       : str         'exact' | 'randomized'.  Default: 'randomized'.
    n_iter       : int         Power-iteration steps for 'randomized'.  Default: 4.
    random_state : int | None  Seed for reproducibility.
    grid_shape   : tuple       Optional (Nu, Nx, Ny) for restore_shape().
    domain       : list        Optional [x0, x1, y0, y1] for domain_mesh.
    **kwargs     : Pre-set any instance attribute (e.g. Sigma=s, Psi=p, …).

    Examples
    --------
    ::

        pod = POD(n_modes=20).fit(Q)
        Z   = pod.encode(Q)              # (20, N_t)
        Q_r = pod.reconstruct(Q)         # (N_x, N_t)

        # if directly from data:
        pod = POD(X=X, n_modes=20)

    Loading pre-computed results::

        pod = POD(Sigma=s, Psi=p, Phi=ph, Q_mean=m, grid_shape=gs)
    """

    # ── class-level defaults ─────────────────────────────────────────────────
    _Sigma: Optional[np.ndarray] = None
    _Psi: Optional[np.ndarray] = None
    _Phi: Optional[np.ndarray] = None
    grid_shape: Optional[tuple] = None  # (Nu, Nx, Ny) for restore_shape
    domain: Optional[list] = None  # [x0, x1, y0, y1]
    field_labels: list = ["$u_x$", "$u_y$"]
    _TKE: Optional[float] = None
    indices_to_original_grid: Optional[np.ndarray] = None
    method: str = "randomized"
    n_iter: int = 4
    random_state: Optional[int] = None

    @property
    def Sigma(self) -> np.ndarray:
        if self._Sigma is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Sigma

    @Sigma.setter
    def Sigma(self, v: np.ndarray) -> None:
        self._Sigma = v

    @property
    def Psi(self) -> np.ndarray:
        if self._Psi is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Psi

    @Psi.setter
    def Psi(self, v: np.ndarray) -> None:
        self._Psi = v

    @property
    def Phi(self) -> np.ndarray:
        if self._Phi is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Phi

    @Phi.setter
    def Phi(self, v: np.ndarray) -> None:
        self._Phi = v

    def __init__(
        self,
        n_modes: int = 20,
        method: str = "randomized",
        n_iter: int = 4,
        random_state: Optional[int] = None,
        grid_shape: Optional[tuple] = None,
        domain: Optional[list] = None,
        **kwargs,
    ):
        self.N_latent = n_modes
        self.grid_shape = grid_shape
        self.domain = domain
        self.method = method
        self.n_iter = n_iter
        self.random_state = random_state
        for key, val in kwargs.items():
            if hasattr(type(self), key) or key in (
                "Sigma",
                "Psi",
                "Phi",
                "Q_mean",
                "_TKE",
                "indices_to_original_grid",
            ):
                setattr(self, key, val)
        # infer latent size from pre-loaded Phi if provided
        if self._Phi is not None and self.N_latent == 20:
            self.N_latent = self._Phi.shape[0]
        # backward compat: auto-fit if raw data provided as kwarg 'X'
        if "X" in kwargs and kwargs["X"] is not None:
            self.fit(kwargs["X"])

    # ── N_modes backward-compat alias ─────────────────────────────────────────

    @property
    def N_modes(self) -> int:
        """Backward-compatible alias for N_latent."""
        return self.N_latent

    @N_modes.setter
    def N_modes(self, value: int) -> None:
        self.N_latent = value

    # ── decomposition hook (overridden by SPOD) ────────────────────────────────

    def _decompose(self, Q: np.ndarray) -> tuple:
        """
        Run the SVD/eigendecomposition on zero-mean Q.

        Returns
        -------
        Sigma  : (N_latent,)
        Psi    : (N_x, N_latent)
        Phi    : (N_latent, N_t)
        """
        if self.method == "exact":
            Sigma, Psi, Phi, C = snapshot_pod(Q)
            self._C = C
            return Sigma[: self.N_latent], Psi[:, : self.N_latent], Phi[: self.N_latent]
        elif self.method == "randomized":
            return snapshot_pod_randomized(
                Q,
                n_modes=self.N_latent,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'exact' or 'randomized'.")

    # ── Projector interface ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> POD:
        """
        Fit the POD to data X.

        Accepts a flat matrix X (N_x, N_t) or a raw grid array
        (Nu, N_t, Nx, Ny) / (N_t, Nx, Ny). For raw grid input the NaN
        cylinder mask is detected and stored automatically.
        """

        Q = self.preprocess_snapshot(X)

        result = self._decompose(Q)
        self.Sigma = result[0]
        self.Psi = result[1]
        self.Phi = result[2]
        assert self._Sigma is not None and self._Psi is not None and self._Phi is not None, (
            "Decomposition must return Sigma, Psi, Phi."
        )
        self.N_latent = self.Sigma.shape[0]
        self.fitted = True
        return self

    @property
    def n_params(self) -> int:
        return int(self.Psi.size)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the spatial modes.

            Z = Psi^T (X - Q_mean)    shape (N_latent, N_t)
        """
        Q = self.preprocess_snapshot(X)
        return self.Psi.T @ Q

    def decode(self, Z: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct from latent coefficients.
            - Z : latent coefficients (N_latent, N_t)
            - idx : optional indices to select a subset of the spatial modes (Psi) and mean (Q_mean).
        Returns the reconstructed state in the original space:
            Q_hat = Psi Z + Q_mean    shape (len(idx), N_t) or (N_fluid * n_fields, N_t) if idx is provided.
        """

        if idx is not None:
            return self.Psi[idx, :] @ Z + self.Q_mean[idx, :]
        else:
            return self.Psi @ Z + self.Q_mean

    def decode_at(self, Z: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Sensor readout by the cheap linear route -- ``Psi[idx] @ Z``, never
        forming the full field. Overrides `Projector.decode_at`, which has to
        decode everything and then index."""
        return self.decode(Z, idx=idx)

    def spatial_basis(self, z0: Optional[np.ndarray] = None) -> np.ndarray:
        r"""The POD spatial modes :math:`\Psi`, ``(N_x, N_latent)``.

        A linear decoder's Jacobian is its own matrix, so the local basis the
        base class approximates by finite differences is here exact and
        independent of ``z0`` (accepted and ignored, for interface parity).
        Returning `Psi` unchanged is what keeps sensor placement identical to
        what `POD_ESN` produced before placement was generalised.
        """
        return self.Psi

    def reconstruct(
        self,
        X: Optional[np.ndarray] = None,
        n_modes: Optional[int] = None,
        Phi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Full round-trip: encode → decode → to_grid (when mask is available).

        Parameters
        ----------
        X       : raw input (grid or flat).  If None, uses stored Phi.
        n_modes : retain only the first n_modes modes.  Default: all.
        Phi     : pre-computed latent coefficients (N_latent, N_t); skips encode.
        """
        if Phi is not None:
            Z = Phi
        elif X is not None:
            Z = self.encode(X)
        else:
            Z = self.Phi

        nm = n_modes if n_modes is not None else self.N_latent
        if nm < self.N_latent:
            X_hat = self.Psi[:, :nm] @ Z[:nm] + self.Q_mean
        else:
            X_hat = self.decode(Z)
        if getattr(self, "to_grid", None) is not None and self.grid_shape is not None:
            return self._to_physical_grid(X_hat)
        return X_hat

    # ── utilities ─────────────────────────────────────────────────────────────

    def energy_fraction(self):
        """
        Relative and cumulative energy per mode.
        - rel = lam / sum(lam) where lam = Sigma^2 are the eigenvalues of the correlation matrix C.
        - cum = np.cumsum(rel) is the cumulative fraction of energy captured by the first j modes.

        Returns
        -------
        rel : ndarray (N_latent,)  fraction of total energy per mode
        cum : ndarray (N_latent,)  cumulative fraction
        """
        lam = self.Sigma**2  # eigenvalues of C = Q^T Q / N_t

        return lam / lam.sum(), np.cumsum(lam) / lam.sum()

    def truncate(self, n_modes: int) -> POD:
        """Truncate to the first n_modes modes in-place."""
        if n_modes >= self.N_latent:
            print(f"Requested n_modes={n_modes} >= N_latent={self.N_latent}. No truncation applied.")
            return self
        self.Psi = self.Psi[:, :n_modes]
        self.Phi = self.Phi[:n_modes, :]
        self.Sigma = self.Sigma[:n_modes]
        self.N_latent = n_modes
        return self

    @property
    def domain_mesh(self):
        """
        Meshgrid for the spatial domain.

        Returns
        -------
        X1, X2 : ndarray  Coordinate arrays shaped (Nx, Ny).
        """
        if self.domain is None or self.grid_shape is None:
            raise ValueError("Both domain and grid_shape must be set.")
        x1 = np.linspace(*self.domain[:2], num=self.grid_shape[-2])
        x2 = np.linspace(*self.domain[2:], num=self.grid_shape[-1])
        return np.meshgrid(x1, x2, indexing="ij")

    def original_data_to_domain_of_interest(self, original_data: np.ndarray):
        """Crop original-grid data to the fitted domain of interest."""
        if self.indices_to_original_grid is None:
            return original_data
        original_data = original_data.copy()
        try:
            if original_data.ndim == 2:
                return original_data[self.indices_to_original_grid]
            return original_data[:, self.indices_to_original_grid[0], self.indices_to_original_grid[1]]
        except Exception:
            raise ValueError("Pass original_data in shape [(Nu) x Nx x Ny x (Nt)].")

    # ── metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_MSE(ROM_data: np.ndarray, original_data: np.ndarray, time_evolution: bool = False):
        """Mean Squared Error between ROM reconstruction and original data."""
        ROM_data, original_data = POD.flatten(ROM_data, original_data)
        original_data[np.isnan(original_data)] = 0.0
        if time_evolution:
            return np.mean((original_data - ROM_data) ** 2, axis=0)
        return float(np.mean((original_data - ROM_data) ** 2))

    @staticmethod
    def compute_RMS(ROM_data: np.ndarray, original_data: np.ndarray):
        """Root Mean Square error (field)."""
        original_data[np.isnan(original_data)] = 0.0
        return np.sqrt((original_data - ROM_data) ** 2)

    @staticmethod
    def flatten(*args):
        """Flatten multi-dimensional arrays to 2-D (space × time)."""
        return [a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a.copy() for a in args]


# ─────────────────────────────────────────────────────────────────────────────
# SPOD (Sieber)  — inherits from POD, only overrides _decompose
# ─────────────────────────────────────────────────────────────────────────────


class SPOD(POD):
    """
    Spectral POD — Sieber, Paschereit & Oberleithner (JFM 2016).

    Inherits the full POD interface.  The only difference is that the snapshot
    correlation matrix C is replaced by a low-pass filtered version

        C_tilde = G^T C G

    where G is a banded symmetric Toeplitz filter matrix.  Setting ``Nf=0``
    exactly recovers snapshot POD.

    Parameters
    ----------
    Nf          : int   Filter half-width (0 = POD limit, N_t/2 = DFT limit).
    filter_kind : str   'gaussian' | 'box' | 'hann'.  Default: 'gaussian'.
    n_modes     : int   Modes to retain.  Default: all (N_t).
    grid_shape  : tuple Optional (Nu, Nx, Ny) for
    domain      : list  Optional [x0, x1, y0, y1] for domain_mesh.

    Attributes
    ----------
    C_tilde : ndarray (N_t, N_t)  Filtered correlation matrix (stored after fit).
    """

    def __init__(
        self,
        Nf: int = 0,
        filter_kind: str = "gaussian",
        n_modes: int = 20,
        grid_shape: Optional[tuple] = None,
        domain: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(n_modes=n_modes, grid_shape=grid_shape, domain=domain, **kwargs)
        self.Nf = Nf
        self.filter_kind = filter_kind
        self._n_modes_requested = n_modes  # None = keep all after fit

    def _decompose(self, Q: np.ndarray) -> tuple:
        Sigma, Psi, Phi, C_tilde = spod_sieber(Q, self.Nf, self.filter_kind)
        self.C_tilde = C_tilde
        if self._n_modes_requested is not None:
            nm = min(self._n_modes_requested, Sigma.shape[0])
            return Sigma[:nm], Psi[:, :nm], Phi[:nm]
        self.N_latent = Sigma.shape[0]
        return Sigma, Psi, Phi
