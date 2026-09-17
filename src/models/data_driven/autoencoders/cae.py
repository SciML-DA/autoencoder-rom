from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import Projector

__all__ = ["CAE"]


class CAE(Projector):
    """
    Convolutional Autoencoder.

    PyTorch Conv2d encoder / ConvTranspose2d decoder on the 2-D spatial grid,
    ending in a dense bottleneck of size ``n_latent``.
    Stride-2 3x3 convs halving the grid each stage (tanh), flatten + Linear to
    the latent, then the transposed mirror with output_padding to recover the size,
    final conv linear.

    Follows Racca et al. (2021) and Ozalp et al. (2024) — single-CAE variant

    Grid requirement: each stride-2 stage halves the grid, and the decoder
    inverts it with ``output_padding``. Only works when every stage keeps
    the ``output_padding`` in ``[0, stride)``.
    Odd dims raise a ``ValueError`` telling you to pad the grid.

    After ``fit(X)`` the following are available:

        enc_conv, dec_conv : nn.Sequential   conv / transposed-conv stacks
        enc_fc, dec_fc     : nn.Linear       bottleneck in / out
        _red_shape         : (C, W, H)       grid size at the bottleneck
        loss_history       : list[float]     mean training loss per epoch
        _scale             : (N_x, 1)        per-field input normalisation
        Q_mean             : (N_x, 1)        temporal mean (from the base)

        NB:
            encode returns Z (n_latent, N_t); decode maps Z back to flat
            (N_x, N_t) in the original space, consistent with POD/AE.

    Parameters
    ----------
    n_latent            : int    Bottleneck size.  Default: 10.
    channels            : tuple  Conv channel widths per encoder stage; the
                                 decoder mirrors them.  Default: (16, 32, 64).
    kernel_size         : int    Conv kernel size.  Default: 3.
    stride              : int    Downsampling stride per stage.  Default: 2.
    pad                 : int    Conv padding.  Default: 1.
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

        cae = CAE(n_latent=8, channels=(16, 32, 64)).fit(X)  # X (Nu, N_t, Nx, Ny)
        Z   = cae.encode(X)              # (8, N_t)
        Xr  = cae.reconstruct(X)         # (N_x, N_t)
    """

    channels: tuple = (16, 32, 64)
    kernel_size: int = 3
    stride: int = 2
    pad: int = 1
    activation_function: str = "tanh"
    learning_rate: float = 1e-3
    threshold: float = 1e-4
    n_epochs: int = 500
    batch_size: int = 32
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    patience: int = 50
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

    # ── network construction ──────────────────────────────────────────────────

    def _build_networks(self, c_in: int, nx: int, ny: int) -> None:
        act = self._ACT[self.activation_function]
        k, s, p = self.kernel_size, self.stride, self.pad

        enc = []
        c, w, h = c_in, nx, ny
        sizes = [(w, h)]
        for c_out in self.channels:
            enc += [nn.Conv2d(c, c_out, k, s, p), act()]
            c = c_out
            w = (w + 2 * p - k) // s + 1
            h = (h + 2 * p - k) // s + 1
            sizes.append((w, h))

        self._red_shape = (c, w, h)  # (C, W, H) at bottleneck
        flat = c * w * h
        self.enc_conv = nn.Sequential(*enc).to(self.device)
        self.enc_fc = nn.Linear(flat, self.N_latent).to(self.device)
        self.dec_fc = nn.Linear(self.N_latent, flat).to(self.device)

        # decoder: mirror the encoder back up to c_in
        dec_out = list(self.channels[-2::-1]) + [c_in]
        targets = sizes[-2::-1]  # sizes to recover, top-down
        dec = []
        for i, c_out in enumerate(dec_out):
            tw, th = targets[i]
            op_w = tw - ((w - 1) * s - 2 * p + k)
            op_h = th - ((h - 1) * s - 2 * p + k)
            if not (0 <= op_w < s and 0 <= op_h < s):
                raise ValueError(
                    f"grid {nx}x{ny} not invertible with k={k},s={s},p={p}; "
                    f"got output_padding ({op_w},{op_h}). Pad the grid to even dims."
                )
            dec.append(nn.ConvTranspose2d(c, c_out, k, s, p, output_padding=(op_w, op_h)))
            if i < len(dec_out) - 1:  # final conv stays linear
                dec.append(act())
            c, w, h = c_out, tw, th
        self.dec_conv = nn.Sequential(*dec).to(self.device)

    def _networks(self) -> list:
        return [self.enc_conv, self.enc_fc, self.dec_fc, self.dec_conv]

    def _encode_grid(self, G: torch.Tensor) -> torch.Tensor:
        return self.enc_fc(self.enc_conv(G).flatten(1))

    def _decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        return self.dec_conv(self.dec_fc(Z).view(-1, *self._red_shape))

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

    def networks(self) -> list[nn.Module]:
        """Lists every network the autoencoder trains.

        Returns:
          The convolutional encoder, the dense encoder stage, the dense decoder
          stage, and the transposed-convolution decoder, in that order.
        """
        return self._networks()

    def decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes onto the spatial grid, differentiably.

        Args:
          Z: Latent codes, shape `(B, N_latent)`.

        Returns:
          Fields in scaled units, shape `(B, Nu, Nx, Ny)`.
        """
        return self._decode_grid(Z)

    # ── flat (N_x, N_t) <-> grid (N_t, Nu, Nx, Ny) with the solid mask ─────────

    def _flat_to_grid(self, Q: np.ndarray) -> np.ndarray:
        Nu, Nx, Ny = self.grid_shape
        n_t = Q.shape[1]
        n_fluid = int(self.fluid_mask_flat.sum())
        full = np.zeros((Nu, n_t, Nx * Ny), dtype=Q.dtype)
        A = Q.reshape(n_fluid, Nu, n_t).transpose(1, 2, 0)  # (Nu, Nt, N_fluid)
        full[:, :, self.fluid_mask_flat] = A
        return full.reshape(Nu, n_t, Nx, Ny).transpose(1, 0, 2, 3)  # (Nt,Nu,Nx,Ny)

    def _grid_to_flat(self, G: np.ndarray) -> np.ndarray:
        Nu, Nx, Ny = self.grid_shape
        Gf = G.reshape(G.shape[0], Nu, Nx * Ny)[:, :, self.fluid_mask_flat]
        return Gf.transpose(2, 1, 0).reshape(-1, G.shape[0])  # (N_fluid*Nu, Nt)

    # ── Projector interface ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> CAE:
        Q = self.preprocess_snapshot(X)  # (N_x, N_t), zero-mean
        assert self.grid_shape is not None, "CAE needs raw grid input to fit."
        Nu, Nx, Ny = self.grid_shape
        self._scale = self._field_scale(Q)  # per-field std (N_x, 1)
        self._build_networks(Nu, Nx, Ny)

        G = self._flat_to_grid(Q / self._scale)  # (N_t, Nu, Nx, Ny)
        data = torch.as_tensor(G, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(
            self.fluid_mask_flat.reshape(Nx, Ny),
            dtype=torch.float32,
            device=self.device,
        )[None, None]  # (1,1,Nx,Ny)

        n_t = data.shape[0]
        n_val = int(round(self.val_fraction * n_t))
        X_tr, X_val = data[: n_t - n_val], data[n_t - n_val :]

        params = [q for net in self._networks() for q in net.parameters()]
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
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
        )

        def masked_mse(recon, target):
            # divide by Nu too: the sum runs over the field channels, so without
            # it this is Nu x a true mean and the CAE curves sit a constant
            # factor above the AE ones for no physical reason
            return ((recon - target) ** 2 * mask).sum() / (mask.sum() * target.shape[0] * Nu)

        best_val, wait = float("inf"), 0
        best_state = _alloc_snapshot(tuple(self._networks()))
        have_best = False
        self.loss_history = []
        self.val_loss_history = []
        self.n_epochs_run = 0
        for _ in range(self.n_epochs):
            for net in self._networks():
                net.train()
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
                recon = self._decode_grid(self._encode_grid(batch))
                loss = masked_mse(recon, batch)
                loss.backward()
                opt.step()
                run += loss.item() * batch.shape[0]
            self.loss_history.append(run / X_tr.shape[0])
            self.n_epochs_run += 1

            if n_val > 0:
                for net in self._networks():
                    net.eval()
                with torch.no_grad():
                    v = masked_mse(self._decode_grid(self._encode_grid(X_val)), X_val).item()
                self.val_loss_history.append(v)
                sched.step(v)
                if v < best_val * (1.0 - self.threshold):
                    best_val, wait = v, 0
                    _save_snapshot(tuple(self._networks()), best_state)
                    have_best = True
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if have_best:
            for net, st in zip(self._networks(), best_state):
                net.load_state_dict(st)
        self.fitted = True
        return self

    @property
    def n_params(self) -> int:
        return sum(q.numel() for net in self._networks() for q in net.parameters())

    def encode(self, X: np.ndarray) -> np.ndarray:
        Q = self.preprocess_snapshot(X)
        G = torch.as_tensor(self._flat_to_grid(Q / self._scale), dtype=torch.float32, device=self.device)
        for net in self._networks():
            net.eval()
        with torch.no_grad():
            Z = self._encode_grid(G)
        return Z.detach().cpu().numpy().T  # (N_latent, N_t)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        Zt = torch.as_tensor(np.asarray(Z).T, dtype=torch.float32, device=self.device)
        for net in self._networks():
            net.eval()
        with torch.no_grad():
            G = self._decode_grid(Zt).detach().cpu().numpy()  # (N_t,Nu,Nx,Ny)
        return self._grid_to_flat(G) * self._scale + self.Q_mean  # (N_x, N_t)
