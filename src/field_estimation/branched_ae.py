"""Two-branch autoencoder for reconstructing a flow field from sparse sensors.

Adds a third operator `G` to a trained encoder/decoder pair, mapping sensor
measurements into the latent space:

    y = E(phi),  phi_hat = D(y),  y_hat = G(s),  phi_hat = D(G(s))

POD-LSE in `epod.py` is the same architecture with `E`, `D`, and `G` all
linear.

`E` and `D` arrive as a `LatentSpace` — `LinearLatent` for a POD basis,
`TorchLatent` for a fitted `AE` or `CAE` — and this module trains only `G`.
Keeping them separable supports the four-way comparison that attributes any
improvement to a cause:

    E / D   G        configuration
    POD     linear   POD-LSE
    POD     MLP      nonlinear sensor map, linear manifold
    AE      linear   linear sensor map, nonlinear manifold
    AE      MLP      full two-branch autoencoder

`G` reads a causal window of `n_delays` samples, which raises the reachable rank
from `N_s` to `N_s * n_delays` and lets the network resolve the sensor lag.

Loss
----
Both terms are normalised by the variance of their own target, so both are
order 1 and `lambda_field` is dimensionless:

    L = ||G(s) - E(phi)||^2 / var(E(phi))
      + lambda_field * ||D(G(s)) - phi||^2 / var(phi)

`lambda_field=0` trains on the latent term alone. For an orthonormal POD basis
that is equivalent to the field term and avoids a decoder pass; the two differ
for an AE decoder.

Train in stages: fit `E` and `D` on reconstruction, then `G` with `D` frozen,
then optionally fine-tune end to end.

Typical usage:

    from field_estimation.branched_ae import LinearLatent, BranchedAE
    from field_estimation.epod import pod, split_train_test

    tr, te = split_train_test(Q.shape[1], 0.25, gap=100, warmup=24)
    Psi, _, _, q_mean = pod(Q[:, tr], r=64, subtract_mean=True,
                            method="randomized")
    model = BranchedAE(LinearLatent(Psi, q_mean), branch="gru",
                       n_delays=25).fit(Q, S, tr)
    model.score(Q, S, te)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn

from .epod import apply_sensor_stats, nmse, sensor_stats, sensor_windows

__all__ = [
    "LatentSpace",
    "LinearLatent",
    "TorchLatent",
    "SensorBranch",
    "BranchedAE",
    "LatentForecaster",
    "sensor_windows",
    "default_device",
]


def default_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    # mps is available on the laptop, but the conv-transpose kernels the CAE
    # needs have been unreliable on it; cpu is the safe local default
    return "cpu"


# ── sensor windows ────────────────────────────────────────────────────────────




# ── latent spaces: the E/D pair the branch plugs into ─────────────────────────


class LatentSpace:
    """Base class for an encoder/decoder pair used by `BranchedAE`.

    Subclasses implement the methods below in both NumPy and differentiable
    torch. The torch side works in the decoder's internal units, called scaled
    units here. `decode` converts to physical units for scoring; `field_target`
    converts the other way, so the field loss is evaluated in the units the
    decoder emits.

    Attributes:
        n_latent: Dimension of the latent space.
        device: Torch device the decoder runs on.
    """

    n_latent: int
    device: str = "cpu"

    def encode(self, Q: np.ndarray) -> np.ndarray:
        """Encodes physical fields, shape (N_x, N_t), to latents (n_latent, N_t)."""
        raise NotImplementedError

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decodes latents, shape (n_latent, N_t), to physical fields (N_x, N_t)."""
        raise NotImplementedError

    def decode_torch(self, Zt: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latents differentiably, returning scaled units.

        Args:
            Zt: Latent codes, shape (B, n_latent).

        Returns:
            Fields in scaled units, shape (B, N_x).
        """
        raise NotImplementedError

    def field_target(self, Q: np.ndarray) -> np.ndarray:
        """Converts physical fields to the units `decode_torch` returns.

        Args:
            Q: Physical fields, shape (N_x, N_t).

        Returns:
            Fields in scaled units, shape (N_t, N_x).
        """
        raise NotImplementedError

    def decoder_parameters(self) -> list:
        """Returns the decoder parameters to unfreeze for end-to-end fine-tuning.

        Returns:
            A list of tensors. Empty for a fixed basis such as POD.
        """
        return []


class LinearLatent(LatentSpace):
    """Wraps a POD basis as an encoder/decoder pair, with `E = Psi.T`, `D = Psi`.

    `BranchedAE(branch="linear")` over this latent space reproduces POD-LSE up
    to the optimiser. Fit the basis on the training block only.

    Args:
        Psi: Spatial modes with orthonormal columns, shape (N_x, r).
        q_mean: Temporal mean, shape (N_x, 1) or (N_x,).
        device: Torch device. Defaults to `default_device()`.
    """

    def __init__(self, Psi: np.ndarray, q_mean: np.ndarray, device: Optional[str] = None):
        self.Psi = np.asarray(Psi, np.float64)
        self.q_mean = np.asarray(q_mean, np.float64).reshape(-1, 1)
        self.n_latent = self.Psi.shape[1]
        self.device = default_device(device)
        self._Psi_t = torch.as_tensor(self.Psi.T, dtype=torch.float32, device=self.device)

    def encode(self, Q):
        return self.Psi.T @ (np.asarray(Q, np.float64) - self.q_mean)

    def decode(self, Z):
        return self.Psi @ np.asarray(Z, np.float64) + self.q_mean

    def decode_torch(self, Zt):
        return Zt @ self._Psi_t  # (B, N_x) fluctuation

    def field_target(self, Q):
        return (np.asarray(Q, np.float64) - self.q_mean).T

    def to(self, device: str) -> LinearLatent:
        self.device = device
        self._Psi_t = self._Psi_t.to(device)
        return self


class TorchLatent(LatentSpace):
    """Wraps a fitted `AE` or `CAE` from `models.data_driven.autoencoders` as an E/D pair.

    Wraps a projector; does not train one. The decoder is set to evaluation mode
    with its parameters frozen. `BranchedAE(finetune_decoder=True)` unfreezes
    them.

    A `CAE` decoder emits a grid, so the map to the flat `(N_fluid * Nu, N_t)`
    layout is built once as a gather index and applied in torch. The projector's
    NumPy `_grid_to_flat` is not differentiable, which the field loss
    requires.

    Args:
        projector: A fitted `AE` or `CAE`.
        device: Torch device. Defaults to the projector's device.

    Raises:
        ValueError: If the projector is not fitted.
    """

    def __init__(self, projector, device: Optional[str] = None):
        if not getattr(projector, "fitted", False):
            raise ValueError("wrap a *fitted* AE/CAE; call projector.fit(X) first")
        self.p = projector
        self.n_latent = int(projector.N_latent)
        self.device = default_device(device or getattr(projector, "device", None))
        self.p.device = self.device
        self._is_conv = hasattr(projector, "dec_conv")
        self._nets = projector._networks() if self._is_conv else [projector.encoder, projector.decoder]
        for net in self._nets:
            net.to(self.device).eval()
        self._scale = np.asarray(projector._scale, np.float64)
        self._gather = self._build_gather() if self._is_conv else None

    def _build_gather(self) -> torch.Tensor:
        """Builds gather indices from the decoder's grid layout to the flat layout.

        The flat layout is (N_fluid, Nu) in row-major order, so row `f * Nu + u`.
        The decoder emits (B, Nu, Nx, Ny), which flattens to `u * (Nx * Ny) + p`.
        """
        Nu, Nx, Ny = self.p.grid_shape
        pos = np.flatnonzero(self.p.fluid_mask_flat)  # (N_fluid,)
        idx = (np.arange(Nu)[None, :] * (Nx * Ny) + pos[:, None]).ravel()
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    # -- numpy side ------------------------------------------------------------

    def encode(self, Q):
        return self.p.encode(Q)

    def decode(self, Z):
        return self.p.decode(Z)

    def field_target(self, Q):
        Qc = np.asarray(Q, np.float64) - self.p.Q_mean
        return (Qc / self._scale).T  # (N_t, N_x), scaled units

    # -- torch side ------------------------------------------------------------

    def decode_torch(self, Zt):
        if not self._is_conv:
            return self.p.decoder(Zt)
        G = self.p._decode_grid(Zt)  # (B, Nu, Nx, Ny)
        return G.reshape(G.shape[0], -1)[:, self._gather]

    def decoder_parameters(self):
        nets = [self.p.dec_fc, self.p.dec_conv] if self._is_conv else [self.p.decoder]
        return [q for net in nets for q in net.parameters()]


# ── the sensor branch G ───────────────────────────────────────────────────────


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


class SensorBranch(nn.Module):
    """Maps a window of sensor history to a latent code.

    Takes (B, n_delays, n_channels), oldest sample first, and returns
    (B, n_latent). No activation follows the output layer, so latent codes stay
    unbounded and match an AE bottleneck and POD coefficients.

    Args:
        n_channels: Number of sensor channels.
        n_delays: Window length in samples.
        n_latent: Dimension of the latent space.
        kind: Architecture. One of:
            `"linear"`: A single affine map on the flattened window. Matches
                LSE up to the sensor POD truncation. Use it as the control,
                rather than comparing against a closed-form solve that
                preprocesses its input differently.
            `"mlp"`: Flattens the window and applies `hidden`.
            `"cnn"`: One-dimensional convolutions along time, channels as
                features. Weight sharing across lags keeps the parameter count
                nearly flat as the window grows.
            `"gru"`: Recurrent over the window, reading out the final hidden
                state. The only head that steps online one sample at a time.
        hidden: Hidden layer widths, for `kind="mlp"`.
        activation: Activation name. One of `tanh`, `relu`, `elu`, `gelu`.
        dropout: Dropout probability applied between hidden layers.
        cnn_channels: Channel widths, for `kind="cnn"`.
        kernel_size: Convolution kernel length, for `kind="cnn"`.
    """

    def __init__(
        self,
        n_channels: int,
        n_delays: int,
        n_latent: int,
        kind: str = "mlp",
        hidden: Sequence[int] = (128, 128),
        activation: str = "tanh",
        dropout: float = 0.0,
        cnn_channels: Sequence[int] = (32, 64),
        kernel_size: int = 5,
        gru_hidden: int = 64,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.kind = kind
        self.n_channels, self.n_delays, self.n_latent = n_channels, n_delays, n_latent
        act = _ACT[activation]
        n_flat = n_channels * n_delays

        if kind == "linear":
            self.head = nn.Linear(n_flat, n_latent)
        elif kind == "mlp":
            dims, layers = [n_flat, *hidden], []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]), act()]
                if dropout:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dims[-1], n_latent))
            self.head = nn.Sequential(*layers)
        elif kind == "cnn":
            conv, c = [], n_channels
            length = n_delays
            for c_out in cnn_channels:
                # Pad to keep the window length constant, so the receptive
                # field does not consume a short window. The flatten in the head
                # performs the reduction instead, in one predictable step.
                conv += [nn.Conv1d(c, c_out, kernel_size, padding=kernel_size // 2), act()]
                c = c_out
            self.conv = nn.Sequential(*conv)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * length, n_latent))
        elif kind == "gru":
            self.gru = nn.GRU(n_channels, gru_hidden, gru_layers, batch_first=True)
            self.head = nn.Linear(gru_hidden, n_latent)
        else:
            raise ValueError(f"kind must be linear/mlp/cnn/gru, got {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in ("linear", "mlp"):
            return self.head(x.flatten(1))
        if self.kind == "cnn":
            return self.head(self.conv(x.transpose(1, 2)))  # (B, C, L)
        out, _ = self.gru(x)
        return self.head(out[:, -1])  # the last window step is time t

    @property
    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters())


# ── the estimator ─────────────────────────────────────────────────────────────


@dataclass
class BranchedAE:
    """Trains the sensor branch `G` against a frozen `E`/`D` pair.

    Exposes `fit`, `encode`, `predict`, and `score`, matching
    `field_estimation.epod.PODLSE` so both classes fit into one comparison loop. The
    windows are causal, so each method takes the whole record plus an index
    array rather than a pre-sliced block:

        model.fit(Q, S, train_idx)     # Q and S full; targets taken at train_idx
        model.score(Q, S, test_idx)

    Attributes:
        latent: A fitted `LinearLatent` or `TorchLatent`. Fit it on the training
            block only.
        branch: Sensor branch architecture, one of `linear`, `mlp`, `cnn`, or
            `gru`. See `SensorBranch`.
        n_delays: Causal window length in samples. 25 samples is 0.1 s at
            250 Hz.
        delay_stride: Sample spacing between consecutive lags.
        hidden: Hidden widths for an MLP branch.
        activation: Activation name for the branch.
        dropout: Dropout probability inside the branch.
        cnn_channels: Channel widths for a CNN branch.
        kernel_size: Convolution kernel length for a CNN branch.
        gru_hidden: Hidden width for a GRU branch.
        gru_layers: Number of stacked GRU layers.
        lambda_field: Weight on the field-space loss term. The default of 0
            trains on the latent term alone, which is equivalent for an
            orthonormal POD basis and cheaper otherwise.
        latent_weight: Scaling of the latent targets. `"energy"` leaves them
            unchanged, weighting each mode by its energy and tracking field MSE.
            `"unit"` whitens them, raising field MSE but producing a better
            input for a latent forecaster.
        learning_rate: Adam step size.
        weight_decay: L2 penalty applied by Adam.
        sensor_noise: Standard deviation of Gaussian noise added to the sensor
            window at every training step, in units of the standardised channel
            (the branch's input is already divided by each channel's training
            standard deviation, so 0.1 is 10% of a channel's own variability).
            Fresh noise every minibatch and every epoch; the validation split
            and every prediction are left clean.

            This is the regulariser the diagnosis asked for. The branched models
            reach a training NMSE of ~0.50 against a test NMSE of ~0.94 -- they
            have capacity to spare and not enough data to constrain it, which is
            overfitting rather than the under-fitting the epoch sweep found for
            the linear branch. For a linear model, input noise of variance s^2
            is exactly equivalent to ridge with lambda = s^2 * n, so this puts
            the nonlinear branches on the same footing as the ridge that the
            closed-form estimator already gets, and does it in the one place
            that generalises to a network.
        track_sets: Optional `{name: (Q, S, idx)}`. After every `track_every`
            epochs, the model is scored on each set in *field NMSE* and the
            result appended to `track_history[name]` as `(epoch, nmse)`.

            This is the overfitting diagnostic. `loss_history` and
            `val_loss_history` are latent-space training objectives on a holdout
            carved out of the training block; they answer "is the optimiser
            still descending", not "is this model generalising". Passing the
            held-out *test* block here scores the thing actually reported, in
            the same units, on the same axes -- so a training curve that keeps
            falling while the test curve flattens or turns up is visible rather
            than inferred from two final numbers.

            Scoring costs a full decode per set per evaluation, which is why
            `track_every` and `track_max_cols` exist: raise the first and lower
            the second on a long run.
        track_every: Epochs between tracking evaluations.
        track_max_cols: Snapshots sampled (evenly) from each tracked index set.
            The curve is a diagnostic, not a reported score, so a few hundred
            columns is ample and the full test block is wasteful per epoch.
        n_epochs: Maximum training epochs.
        batch_size: Minibatch size.
        val_fraction: Fraction of the training block held out for early
            stopping, taken contiguously from the end of the block.
        patience: Epochs without improvement before stopping.
        threshold: Relative improvement required to reset `patience`.
        lr_factor: Decay factor for `ReduceLROnPlateau`.
        lr_patience: Plateau epochs before decaying the learning rate. Keep it
            well below `patience`.
        min_lr: Lower bound on the learning rate.
        grad_clip: Maximum global gradient norm.
        finetune_decoder: Whether to unfreeze the decoder and train it jointly
            with `G`. Applies only to `TorchLatent`; a POD basis has no
            parameters to fine-tune.
        seed: Random seed for torch and NumPy.
        device: Torch device. Defaults to `default_device()`.
        verbose: Whether to print per-epoch losses.
    """

    latent: LatentSpace
    branch: str = "mlp"
    n_delays: int = 25
    delay_stride: int = 1

    # branch architecture
    hidden: Sequence[int] = (128, 128)
    activation: str = "tanh"
    dropout: float = 0.0
    cnn_channels: Sequence[int] = (32, 64)
    kernel_size: int = 5
    gru_hidden: int = 64
    gru_layers: int = 1

    # loss
    lambda_field: float = 0.0
    latent_weight: str = "energy"

    # optimisation
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    sensor_noise: float = 0.0
    n_epochs: int = 500
    batch_size: int = 128
    val_fraction: float = 0.2
    patience: int = 50
    threshold: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    grad_clip: float = 5.0
    finetune_decoder: bool = False
    seed: int = 0
    device: Optional[str] = None
    verbose: bool = False

    # -- learned ---------------------------------------------------------------
    G: Optional[nn.Module] = field(default=None, repr=False)
    s_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    z_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    track_sets: Optional[dict] = None
    track_every: int = 1
    track_max_cols: int = 400
    loss_history: list = field(default_factory=list, repr=False)
    val_loss_history: list = field(default_factory=list, repr=False)
    track_history: dict = field(default_factory=dict, repr=False)
    fitted: bool = False

    # -- api -------------------------------------------------------------------

    @property
    def warmup(self) -> int:
        """Number of leading columns whose window is partly zero padding.

        Exclude these indices from both `train_idx` and any evaluation indices.
        """
        return (self.n_delays - 1) * self.delay_stride

    def fit(self, Q: np.ndarray, S: np.ndarray, train_idx: np.ndarray) -> BranchedAE:
        """Trains the sensor branch on the given indices.

        Args:
            Q: Full field record, shape (N_x, N_t).
            S: Full synchronised sensor record, shape (N_s, N_t).
            train_idx: Indices to train on. Must start at or after `warmup`.

        Returns:
            This estimator, fitted.

        Raises:
            ValueError: If `Q` and `S` disagree on snapshot count, if either
                holds non-finite values, or if `train_idx` reaches into the
                warm-up region.
        """
        dev = self.device = default_device(self.device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if hasattr(self.latent, "to"):
            self.latent.to(dev)

        train_idx = np.asarray(train_idx)
        self._check(Q, S, train_idx)

        # sensor standardisation, learned on the training columns only
        S = np.asarray(S, np.float64)
        self.s_mean, self.s_scale = sensor_stats(S[:, train_idx])
        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale),
                           self.n_delays, self.delay_stride)

        # latent targets, from the frozen encoder
        Z = self.latent.encode(Q[:, train_idx])  # (r, n_tr)
        zs = Z.std(axis=1, keepdims=True) if self.latent_weight == "unit" else np.ones((Z.shape[0], 1))
        self.z_scale = np.where(zs > 0, zs, 1.0)

        # contiguous validation tail -- see the class docstring
        n_val = int(round(self.val_fraction * len(train_idx)))
        cut = len(train_idx) - n_val
        idx_tr, idx_va = train_idx[:cut], train_idx[cut:]

        Xtr = torch.as_tensor(W[idx_tr], dtype=torch.float32, device=dev)
        Ztr = torch.as_tensor((Z[:, :cut] / self.z_scale).T, dtype=torch.float32, device=dev)
        Xva = torch.as_tensor(W[idx_va], dtype=torch.float32, device=dev) if n_val else None
        Zva = torch.as_tensor((Z[:, cut:] / self.z_scale).T, dtype=torch.float32, device=dev) if n_val else None

        # Normalise each loss term by the variance of its own target, so both
        # are order 1 and lambda_field is dimensionless.
        self._z_var = float(Ztr.var().item()) or 1.0
        Ftr = Fva = None
        if self.lambda_field:
            F = self.latent.field_target(Q[:, train_idx])  # (n_tr, N_x), scaled units
            Ftr = torch.as_tensor(F[:cut], dtype=torch.float32, device=dev)
            Fva = torch.as_tensor(F[cut:], dtype=torch.float32, device=dev) if n_val else None
            self._f_var = float(Ftr.var().item()) or 1.0

        self.G = SensorBranch(
            n_channels=S.shape[0],
            n_delays=self.n_delays,
            n_latent=self.latent.n_latent,
            kind=self.branch,
            hidden=self.hidden,
            activation=self.activation,
            dropout=self.dropout,
            cnn_channels=self.cnn_channels,
            kernel_size=self.kernel_size,
            gru_hidden=self.gru_hidden,
            gru_layers=self.gru_layers,
        ).to(dev)

        params = list(self.G.parameters())
        if self.finetune_decoder:
            dec = self.latent.decoder_parameters()
            if not dec:
                raise ValueError(
                    "finetune_decoder=True but the latent space has no trainable "
                    "decoder. A POD basis is fixed by construction -- use a "
                    "TorchLatent, or leave this off."
                )
            for q in dec:
                q.requires_grad_(True)
            params += dec

        opt = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_lr
        )

        best, wait, best_state = float("inf"), 0, None
        self.loss_history, self.val_loss_history = [], []
        self.track_history = {}
        n = Xtr.shape[0]
        for ep in range(self.n_epochs):
            self.G.train()
            order = torch.randperm(n, device=dev)
            run = 0.0
            for s in range(0, n, self.batch_size):
                sl = order[s : s + self.batch_size]
                xb = Xtr[sl]
                if self.sensor_noise > 0:
                    # fresh every step: the network must not be able to average
                    # a fixed perturbation away over epochs
                    xb = xb + self.sensor_noise * torch.randn_like(xb)
                opt.zero_grad(set_to_none=True)
                loss = self._loss(xb, Ztr[sl], None if Ftr is None else Ftr[sl])
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                opt.step()
                run += loss.item() * len(sl)
            self.loss_history.append(run / n)

            if self.track_sets and (ep % max(self.track_every, 1) == 0
                                    or ep == self.n_epochs - 1):
                self.G.eval()
                self.fitted = True   # predict() checks this; restored below
                with torch.no_grad():
                    for name, (Qt, St, idx) in self.track_sets.items():
                        idx = np.asarray(idx)
                        if len(idx) > self.track_max_cols:
                            idx = idx[np.linspace(0, len(idx) - 1,
                                                  self.track_max_cols).astype(int)]
                        e = nmse(np.asarray(Qt, np.float64)[:, idx],
                                 self.predict(St, idx))
                        self.track_history.setdefault(name, []).append((ep, e))

            if n_val:
                self.G.eval()
                with torch.no_grad():
                    v = float(self._loss(Xva, Zva, Fva).item())
                self.val_loss_history.append(v)
                sched.step(v)
                if v < best * (1.0 - self.threshold):
                    best, wait = v, 0
                    best_state = {k: t.detach().clone() for k, t in self.G.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if self.verbose and ep % 25 == 0:
                tail = f"  val {self.val_loss_history[-1]:.4e}" if n_val else ""
                print(f"    epoch {ep:4d}  train {self.loss_history[-1]:.4e}{tail}")

        if best_state is not None:
            self.G.load_state_dict(best_state)
        self.G.eval()
        self.fitted = True
        return self

    def _loss(self, X, Z, F):
        Zh = self.G(X)
        loss = ((Zh - Z) ** 2).mean() / self._z_var
        if F is not None:
            Zphys = Zh * torch.as_tensor(
                self.z_scale.T, dtype=torch.float32, device=Zh.device
            )
            loss = loss + self.lambda_field * ((self.latent.decode_torch(Zphys) - F) ** 2).mean() / self._f_var
        return loss

    def encode(self, S: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicts latent codes from sensor data.

        The counterpart of `PODLSE.encode`.

        Args:
            S: Full sensor record, shape (N_s, N_t). Pass the whole record: the
                window at `idx[j]` reaches back into earlier columns, so slicing
                `S` beforehand zero-pads those windows.
            idx: Snapshots to evaluate. `None` evaluates every column.

        Returns:
            Predicted latent codes, shape (n_latent, len(idx)).
        """
        self._check_fitted()
        S = np.asarray(S, np.float64)
        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale),
                           self.n_delays, self.delay_stride)
        if idx is not None:
            W = W[np.asarray(idx)]
        out = []
        with torch.no_grad():
            for s in range(0, W.shape[0], 4096):
                x = torch.as_tensor(W[s : s + 4096], dtype=torch.float32, device=self.device)
                out.append(self.G(x).cpu().numpy())
        return np.concatenate(out, axis=0).T * self.z_scale

    def predict(self, S: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
        return self.latent.decode(self.encode(S, idx))

    def score(self, Q: np.ndarray, S: np.ndarray, idx: Optional[np.ndarray] = None) -> float:
        idx = np.arange(Q.shape[1]) if idx is None else np.asarray(idx)
        return nmse(np.asarray(Q, np.float64)[:, idx], self.predict(S, idx))

    def latent_score(self, Q: np.ndarray, S: np.ndarray, idx: np.ndarray) -> float:
        """Returns the NMSE in latent space, which isolates `G` from `D`.

        A high latent score with a low field score means the modes `G` misses
        carry little energy. The reverse points at the decoder, not the
        sensors.

        Args:
            Q: Full field record, shape (N_x, N_t).
            S: Full sensor record, shape (N_s, N_t).
            idx: Snapshots to evaluate.

        Returns:
            Normalised MSE between true and predicted latent codes.
        """
        idx = np.asarray(idx)
        Z = self.latent.encode(np.asarray(Q, np.float64)[:, idx])
        return nmse(Z, self.encode(S, idx))

    @property
    def n_params(self) -> int:
        return self.G.n_params if self.G is not None else 0

    # -- internals -------------------------------------------------------------

    def _check(self, Q, S, train_idx):
        if Q.shape[1] != S.shape[1]:
            raise ValueError(
                f"field has {Q.shape[1]} snapshots, sensors have {S.shape[1]}; "
                "pass the whole synchronised record to fit(), not a slice"
            )
        if train_idx.min() < self.warmup:
            raise ValueError(
                f"train_idx starts at {train_idx.min()} but the {self.n_delays}-step "
                f"window needs {self.warmup} samples of history. Pass "
                f"warmup={self.warmup} to split_train_test."
            )
        if not np.isfinite(S[:, train_idx]).all():
            raise ValueError("non-finite sensor values; interpolate dropouts first")

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("call fit() before encode()/predict()/score()")


# ── latent forecaster (task 3) ────────────────────────────────────────────────


@dataclass
class LatentForecaster:
    """Rolls a latent state forward in time with a GRU.

    Computes `y_{t+1} = F(y_{t-L+1..t})`, the third operator in the full
    pipeline:

        s_{t-L..t} --G--> y_t --F--> y_{t+1..t+h} --D--> phi_{t+1..t+h}

    Trains on latent trajectories only, so it is independent of the sensors and
    the decoder, and `echostatenetwork.EchoStateNetwork` can replace it without
    changes elsewhere.

    Set `n_unroll` above 1. Unrolling feeds predictions back during training,
    trading a little accuracy at horizon 1 for stability over long rollouts.

    Attributes:
        n_latent: Dimension of the latent space.
        n_delays: Input window length in samples.
        hidden: GRU hidden width.
        layers: Number of stacked GRU layers.
        n_unroll: Steps to unroll during training.
        learning_rate: Adam step size.
        n_epochs: Maximum training epochs.
        batch_size: Minibatch size.
        val_fraction: Fraction of windows held out for early stopping.
        patience: Epochs without improvement before stopping.
        grad_clip: Maximum global gradient norm.
        seed: Random seed.
        device: Torch device. Defaults to `default_device()`.
        verbose: Whether to print per-epoch losses.
    """

    n_latent: int
    n_delays: int = 25
    hidden: int = 128
    layers: int = 1
    n_unroll: int = 5
    learning_rate: float = 1e-3
    n_epochs: int = 300
    batch_size: int = 128
    val_fraction: float = 0.2
    patience: int = 40
    grad_clip: float = 5.0
    seed: int = 0
    device: Optional[str] = None
    verbose: bool = False

    net: Optional[nn.Module] = field(default=None, repr=False)
    z_mean: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    z_scale: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    loss_history: list = field(default_factory=list, repr=False)
    val_loss_history: list = field(default_factory=list, repr=False)
    fitted: bool = False

    def fit(self, Z: np.ndarray, train_idx: Optional[np.ndarray] = None) -> LatentForecaster:
        """Trains the forecaster on a latent trajectory.

        Args:
            Z: Latent trajectory, shape (n_latent, N_t).
            train_idx: Contiguous indices to train on. `None` uses the whole
                trajectory.

        Returns:
            This forecaster, fitted.

        Raises:
            ValueError: If `train_idx` is not contiguous, or if the block yields
                fewer than ten training windows.
        """
        dev = self.device = default_device(self.device)
        torch.manual_seed(self.seed)
        Z = np.asarray(Z, np.float64)
        idx = np.arange(Z.shape[1]) if train_idx is None else np.asarray(train_idx)
        if np.any(np.diff(idx) != 1):
            raise ValueError("LatentForecaster needs a contiguous training block")

        Zt = Z[:, idx]
        self.z_mean = Zt.mean(axis=1, keepdims=True)
        sd = Zt.std(axis=1, keepdims=True)
        self.z_scale = np.where(sd > 0, sd, 1.0)
        Zn = ((Zt - self.z_mean) / self.z_scale).T  # (n, r)

        L, H = self.n_delays, self.n_unroll
        starts = np.arange(0, len(Zn) - L - H + 1)
        if len(starts) < 10:
            raise ValueError(f"only {len(starts)} training windows; shorten n_delays/n_unroll")
        n_val = int(round(self.val_fraction * len(starts)))
        cut = len(starts) - n_val

        X = np.stack([Zn[s : s + L] for s in starts])          # (n, L, r)
        Y = np.stack([Zn[s + L : s + L + H] for s in starts])  # (n, H, r)
        Xt = torch.as_tensor(X[:cut], dtype=torch.float32, device=dev)
        Yt = torch.as_tensor(Y[:cut], dtype=torch.float32, device=dev)
        Xv = torch.as_tensor(X[cut:], dtype=torch.float32, device=dev) if n_val else None
        Yv = torch.as_tensor(Y[cut:], dtype=torch.float32, device=dev) if n_val else None

        self.net = _GRUStep(self.n_latent, self.hidden, self.layers).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

        best, wait, best_state = float("inf"), 0, None
        self.loss_history, self.val_loss_history = [], []
        for ep in range(self.n_epochs):
            self.net.train()
            order = torch.randperm(Xt.shape[0], device=dev)
            run = 0.0
            for s in range(0, Xt.shape[0], self.batch_size):
                sl = order[s : s + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = ((self.net.unroll(Xt[sl], H) - Yt[sl]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                opt.step()
                run += loss.item() * len(sl)
            self.loss_history.append(run / Xt.shape[0])
            if n_val:
                self.net.eval()
                with torch.no_grad():
                    v = float(((self.net.unroll(Xv, H) - Yv) ** 2).mean().item())
                self.val_loss_history.append(v)
                sched.step(v)
                if v < best * (1 - 1e-4):
                    best, wait = v, 0
                    best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if self.verbose and ep % 25 == 0:
                print(f"    epoch {ep:4d}  train {self.loss_history[-1]:.4e}")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.fitted = True
        return self

    def rollout(self, Z_window: np.ndarray, horizon: int) -> np.ndarray:
        """Forecasts a latent trajectory in closed loop.

        Each prediction feeds back as the next input.

        Args:
            Z_window: Latent history, shape (n_latent, n_delays).
            horizon: Number of steps to forecast.

        Returns:
            The forecast, shape (n_latent, horizon).

        Raises:
            RuntimeError: If the forecaster is not fitted.
        """
        if not self.fitted:
            raise RuntimeError("call fit() first")
        Zn = ((np.asarray(Z_window, np.float64) - self.z_mean) / self.z_scale).T
        x = torch.as_tensor(Zn[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Y = self.net.unroll(x, horizon)[0].cpu().numpy()  # (H, r)
        return Y.T * self.z_scale + self.z_mean


class _GRUStep(nn.Module):
    """One-step latent map with an explicit closed-loop unroll."""

    def __init__(self, n_latent: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(n_latent, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_latent)

    def unroll(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """x (B, L, r) -> (B, horizon, r), feeding predictions back."""
        h_out, h = self.gru(x)
        # residual form: the network predicts the *increment*. At 250 Hz
        # consecutive latent states are nearly identical, so predicting the
        # state directly means learning the identity plus a small correction --
        # the correction is what carries the dynamics and it gets swamped.
        z = x[:, -1] + self.out(h_out[:, -1])
        preds = [z]
        for _ in range(horizon - 1):
            h_out, h = self.gru(z.unsqueeze(1), h)
            z = z + self.out(h_out[:, -1])
            preds.append(z)
        return torch.stack(preds, dim=1)
