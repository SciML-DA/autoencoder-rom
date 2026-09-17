"""Nonlinear projectors: dense (``AE``) and convolutional (``CAE``) autoencoders.

PyTorch lives here and ONLY here among the projectors. ``Projector``, ``POD``
and ``SPOD`` are torch-free, and ``autoencoders/__init__.py`` resolves this
module lazily, so ``from models.data_driven.autoencoders import POD`` never
imports torch. Keep it that way: it is what lets this file be lifted into a
torch-free package as an optional extra.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import Projector

__all__ = ["AE"]


def _alloc_snapshot(nets) -> list[dict]:
    """Buffers to hold the best-so-far weights, allocated once."""
    return [{k: torch.empty_like(v) for k, v in n.state_dict().items()} for n in nets]


@torch.no_grad()
def _save_snapshot(nets, buffers: list[dict]) -> None:
    """Copy current weights into the preallocated buffers.

    Early in training the val loss improves on almost every epoch, so a
    `deepcopy(state_dict())` here reallocated every parameter each time (236 MB
    per improvement at AE@256). Same bytes are copied, but into buffers that
    already exist, so the allocator stays out of the loop.
    """
    for n, buf in zip(nets, buffers):
        for k, v in n.state_dict().items():
            buf[k].copy_(v)


def _fused_adam_ok(params) -> dict:
    """`fused=True` where torch supports it, plain Adam otherwise.

    Fused Adam is CUDA-only and needs floating-point params; asking for it on
    CPU raises rather than falling back, so this is a capability check, not a
    preference. `foreach=True` is the next best thing and is what torch would
    pick anyway on CUDA -- naming it keeps behaviour identical across versions.
    """
    if not params:
        return {}
    p0 = params[0]
    if p0.is_cuda and p0.dtype in (torch.float32, torch.float64, torch.float16):
        return {"fused": True}
    return {"foreach": True} if p0.is_cuda else {}


class AE(Projector):
    """
    Fully-connected Autoencoder (MLP).

    Nonlinear generalisation of POD. Encoder MLP maps the flattened,
    zero-mean state to a bottleneck of size ``n_latent``, a decoder MLP maps
    back. Hidden layers use ``activation_function`` (tanh by default); the
    bottleneck and output layers are linear so the latent code and the
    reconstruction are unbounded. Trained end-to-end on MSE with Adam.

    Preprocessing (shared ``Projector`` path): ``preprocess_snapshot`` removes
    the NaN solid mask and subtracts the temporal mean ``Q_mean``; the AE then
    divides by a per-field scale ``_scale`` (one std per field, shape (N_x, 1))
    so each field is O(1) and none is underweighted in the MSE. encode/decode
    invert both.

    Two solvers are shared with every projector: ``fit`` / ``encode`` /
    ``decode`` / ``reconstruct`` / ``score`` from the base, so an AE is a
    drop-in replacement for POD in the ROM pipeline.

    After ``fit(X)`` the following are available:

        encoder, decoder : nn.Sequential   trained networks
        loss_history     : list[float]     mean training loss per epoch
        _scale           : (N_x, 1)        per-field input normalisation
        Q_mean           : (N_x, 1)        temporal mean (from the base)
        fitted           : bool

        Note:
            X_hat = decode(encode(X)) ~= X   (N_x, N_t) in the original space.
            encode returns Z (n_latent, N_t); decode maps Z back to (N_x, N_t).

    Parameters
    ----------
    n_latent            : int    Bottleneck size.  Default: 10.
    layer_dims          : tuple  Encoder hidden widths; decoder mirrors them.
                                 Must stay above the largest latent so the
                                 bottleneck is the latent, not a hidden layer.
                                 Default: (512, 128).
    activation_function : str    'tanh' | 'relu' | 'elu' | 'identity'.
    learning_rate       : float  Adam step size.  Default: 1e-3.
    n_epochs            : int    Max epochs.  Default: 500.
    batch_size          : int    Minibatch size.  Default: 32.
    val_fraction        : float  Held-out fraction for early stopping.  Default: 0.2.
    weight_decay        : float  L2 penalty (Adam).  Default: 0.0.
    patience            : int    Early-stopping patience in epochs.  Default: 50.
    lr_factor           : float  ReduceLROnPlateau decay factor.  Default: 0.5.
    lr_patience         : int    Epochs on a val plateau before decaying the LR.
                                 Keep well below ``patience``.  Default: 10.
    min_lr              : float  Lower bound on the LR.  Default: 1e-6.
    seed                : int    Torch RNG seed.  Default: 0.
    device              : str    'cpu' | 'cuda'.  Default: 'cpu'.
    **kwargs            : Override any of the above at construction.

    Examples
    --------
    ::

        ae  = AE(n_latent=10, layer_dims=(128, 32)).fit(X)   # X (Nu, N_t, Nx, Ny)
        Z   = ae.encode(X)               # (10, N_t)
        Xr  = ae.reconstruct(X)          # (N_x, N_t)
        mse = ae.score(X)
    """

    layer_dims: tuple = (512, 128)
    activation_function: str = "tanh"
    learning_rate: float = 1e-3
    n_epochs: int = 500
    batch_size: int = 32
    val_fraction: float = 0.2  # early stopping
    weight_decay: float = 0.0  # l2 regularization
    patience: int = 50
    threshold: float = 1e-4  # early stopping relative err

    # ReduceLROnPlateau on the val loss
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    seed: int = 0
    device: str = "cpu"

    _scale: Optional[np.ndarray] = None  # per-field input normalization (N_x, 1)

    _ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "identity": nn.Identity}

    def __init__(self, n_latent: int = 10, **kwargs):
        self.N_latent = n_latent
        for key, val in kwargs.items():
            if hasattr(type(self), key):
                setattr(self, key, val)
        torch.manual_seed(self.seed)

    def _make_mlp(self, dims: list, last_linear: bool) -> nn.Sequential:
        act = self._ACT[self.activation_function]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not (is_last and last_linear):
                layers.append(act())
        return nn.Sequential(*layers)

    def _build_networks(self, n_x: int) -> None:
        enc_dims = [n_x, *self.layer_dims, self.N_latent]
        dec_dims = [self.N_latent, *reversed(self.layer_dims), n_x]
        self.encoder = self._make_mlp(enc_dims, last_linear=True).to(self.device)
        self.decoder = self._make_mlp(dec_dims, last_linear=True).to(self.device)

    # ── data helpers ──────────────────────────────────────────────────────────

    def _to_torch(self, Q: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((Q / self._scale).T, dtype=torch.float32, device=self.device)

    def _from_torch(self, T: torch.Tensor) -> np.ndarray:
        return T.detach().cpu().numpy().T * self._scale

    def fit(self, X: np.ndarray) -> AE:
        Q = self.preprocess_snapshot(X)  # (N_x, N_t), zero-mean
        n_x, n_t = Q.shape
        self._scale = self._field_scale(Q)  # per-field std (N_x, 1)
        self._build_networks(n_x)

        data = self._to_torch(Q)  # (N_t, N_x)
        n_val = int(round(self.val_fraction * n_t))
        X_tr, X_val = data[: n_t - n_val], data[n_t - n_val :]

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        # The default Adam path loops over parameter tensors in Python and
        # launches several tiny kernels per tensor; profiling put 47% of CUDA
        # time in Adam.step. fused=True does the whole update in one kernel.
        # It reassociates the elementwise arithmetic, so results shift by
        # ~float32 eps -- about a thousand times smaller than the seed-to-seed
        # spread these results already carry.
        opt = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            **_fused_adam_ok(params),
        )
        loss_fn = nn.MSELoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
        )

        best_val, wait = float("inf"), 0
        best_state = _alloc_snapshot((self.encoder, self.decoder))
        have_best = False
        self.loss_history = []
        self.val_loss_history = []
        self.n_epochs_run = 0
        for _ in range(self.n_epochs):
            self.encoder.train()
            self.decoder.train()
            # randperm stays on CPU so the RNG stream -- and therefore the exact
            # batch ordering -- is unchanged; only the transfer moves. Indexing a
            # device tensor with a CPU index tensor copies it per step; sending
            # the permutation once per epoch does it once instead.
            # NOT non_blocking: from pageable memory that copy is genuinely
            # async and the very next line indexes with it, which under real
            # training load silently gathered stale rows.
            order = torch.randperm(X_tr.shape[0]).to(X_tr.device)
            run = 0.0
            for s in range(0, X_tr.shape[0], self.batch_size):
                batch = X_tr[order[s : s + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.decoder(self.encoder(batch)), batch)
                loss.backward()
                opt.step()
                run += loss.item() * batch.shape[0]
            self.loss_history.append(run / X_tr.shape[0])
            self.n_epochs_run += 1

            # early stopping on held-out reconstruction
            if n_val > 0:
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    v = loss_fn(self.decoder(self.encoder(X_val)), X_val).item()
                self.val_loss_history.append(v)
                sched.step(v)
                if v < best_val * (1.0 - self.threshold):
                    best_val, wait = v, 0
                    _save_snapshot((self.encoder, self.decoder), best_state)
                    have_best = True
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if have_best:
            self.encoder.load_state_dict(best_state[0])
            self.decoder.load_state_dict(best_state[1])
        self.fitted = True
        return self

    @property
    def n_params(self) -> int:
        return sum(q.numel() for net in (self.encoder, self.decoder) for q in net.parameters())

    @property
    def scale(self) -> np.ndarray:
        """The per-field scale that `fit` divides inputs by, shape `(N_x, 1)`.

        Raises:
          AttributeError: If the autoencoder is not fitted.
        """
        scale: np.ndarray | None = getattr(self, "_scale", None)
        if scale is None:
            raise AttributeError("Not fitted — call fit() first.")
        return scale

    def encode(self, X: np.ndarray) -> np.ndarray:
        Q = self.preprocess_snapshot(X)
        self.encoder.eval()
        with torch.no_grad():
            Z = self.encoder(self._to_torch(Q))
        return Z.detach().cpu().numpy().T  # (N_latent, N_t)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        Zt = torch.as_tensor(np.asarray(Z).T, dtype=torch.float32, device=self.device)
        self.decoder.eval()
        with torch.no_grad():
            Q_hat = self.decoder(Zt)
        return self._from_torch(Q_hat) + self.Q_mean  # (N_x, N_t)
