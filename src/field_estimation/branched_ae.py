"""Two-branch autoencoder for reconstructing a flow field from sparse sensors.

Adds a sensor branch `G` to a trained encoder and decoder pair, mapping sensor
measurements into the latent space:

    y = E(phi),  phi_hat = D(y),  y_hat = G(s),  phi_hat = D(G(s))

`E` and `D` arrive as a `LatentSpace`, either `LinearLatent` for a POD basis or
`TorchLatent` for a fitted `AE` or `CAE`, and this module trains only `G`. The
two choices combine into four configurations:

    E / D   G        configuration
    POD     linear   POD-LSE
    POD     MLP      nonlinear sensor map, linear manifold
    AE      linear   linear sensor map, nonlinear manifold
    AE      MLP      full two-branch autoencoder

`G` reads a causal window of `n_delays` samples. The loss normalizes each term
by the variance of its own target, so `lambda_field` is dimensionless:

    L = ||G(s) - E(phi)||^2 / var(E(phi))
      + lambda_field * ||D(G(s)) - phi||^2 / var(phi)

Fit `E` and `D` on reconstruction first, then fit `G` against them.

Typical usage example:

  from field_estimation.branched_ae import BranchedAE, LinearLatent
  from datasets import split_indices
  from field_estimation.epod import pod

  tr, _, te = split_indices(Q.shape[1], val_frac=0, test_frac=0.25, gap=100, warmup=24)
  Psi, _, _, q_mean = pod(Q[:, tr], r=64, subtract_mean=True, method="randomized")
  model = BranchedAE(LinearLatent(Psi, q_mean), branch="gru", n_delays=25).fit(Q, S, tr)
  model.score(Q, S, te)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeGuard

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .epod import FloatArray, IndexArray, apply_sensor_stats, nmse, sensor_stats, sensor_windows

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

#: A tracked evaluation set: the full field record, the full sensor record, and
#: the snapshot indices to score.
TrackSet = tuple[FloatArray, FloatArray, IndexArray]


def default_device(device: str | None) -> str:
    """Chooses the torch device to run on.

    Args:
      device: A device name, returned unchanged when set.

    Returns:
      `device` when set, otherwise `"cuda"` when CUDA is available, otherwise
      `"cpu"`.
    """
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    # MPS kernels for the transposed convolutions the CAE needs are unreliable,
    # so the local default is the CPU.
    return "cpu"


# ── Latent spaces ─────────────────────────────────────────────────────────────


class LatentSpace(ABC):
    """An encoder and decoder pair that `BranchedAE` trains a sensor branch against.

    Each subclass works in two unit systems. The NumPy methods `encode` and
    `decode` take and return physical units. The torch method `decode_torch`
    returns the decoder's internal scaled units, and `field_target` converts
    physical fields into those units so the field loss compares like with like.

    Attributes:
      n_latent: Dimension of the latent space.
      device: Torch device the decoder runs on.
    """

    n_latent: int
    device: str = "cpu"

    @abstractmethod
    def encode(self, Q: FloatArray) -> FloatArray:
        """Encodes physical fields into latent codes.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Latent codes, shape `(n_latent, N_t)`.
        """

    @abstractmethod
    def decode(self, Z: FloatArray) -> FloatArray:
        """Decodes latent codes into physical fields.

        Args:
          Z: Latent codes, shape `(n_latent, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """

    @abstractmethod
    def decode_torch(self, Zt: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latent codes differentiably.

        Args:
          Zt: Latent codes, shape `(B, n_latent)`.

        Returns:
          Fields in scaled units, shape `(B, N_x)`.
        """

    @abstractmethod
    def field_target(self, Q: FloatArray) -> FloatArray:
        """Converts physical fields into the units `decode_torch` returns.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Fields in scaled units, shape `(N_t, N_x)`.
        """

    def decoder_parameters(self) -> list[nn.Parameter]:
        """Lists the decoder parameters that end-to-end fine-tuning unfreezes.

        Returns:
          The decoder's parameters, or an empty list for a fixed basis such as
          POD.
        """
        return []

    def to(self, device: str) -> LatentSpace:
        """Moves the decoder's tensors to a device.

        The base implementation holds no tensors of its own and returns `self`
        unchanged.

        Args:
          device: Torch device name.

        Returns:
          This latent space.
        """
        del device
        return self


class LinearLatent(LatentSpace):
    """Wraps a POD basis as an encoder and decoder pair.

    The encoder is `Psi.T` and the decoder is `Psi`. `BranchedAE` with
    `branch="linear"` over the latent space reproduces POD-LSE up to the
    optimizer. The basis should be fit on the training block exclusively.

    Args:
      Psi: Spatial modes with orthonormal columns, shape `(N_x, r)`.
      q_mean: Temporal mean, shape `(N_x, 1)` or `(N_x,)`.
      device: Torch device. `None` selects one with `default_device`.

    Raises:
      ValueError: If `Psi` is not two-dimensional, or if `q_mean` has a
        different length from the columns of `Psi`.
    """

    def __init__(self, Psi: FloatArray, q_mean: FloatArray, device: str | None = None):
        self.Psi = np.asarray(Psi, np.float64)
        if self.Psi.ndim != 2:
            raise ValueError(f"Psi must be (N_x, r), got shape {self.Psi.shape}")

        self.q_mean = np.asarray(q_mean, np.float64).reshape(-1, 1)
        if self.q_mean.shape[0] != self.Psi.shape[0]:
            raise ValueError(f"q_mean has {self.q_mean.shape[0]} entries but Psi has {self.Psi.shape[0]} rows")

        self.n_latent = self.Psi.shape[1]
        self.device = default_device(device)
        self._Psi_t = torch.as_tensor(self.Psi.T, dtype=torch.float32, device=self.device)

    def encode(self, Q: FloatArray) -> FloatArray:
        """Projects physical fields onto the basis.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          POD coefficients, shape `(r, N_t)`.
        """
        return self.Psi.T @ (np.asarray(Q, np.float64) - self.q_mean)

    def decode(self, Z: FloatArray) -> FloatArray:
        """Reconstructs physical fields from POD coefficients.

        Args:
          Z: POD coefficients, shape `(r, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        return self.Psi @ np.asarray(Z, np.float64) + self.q_mean

    def decode_torch(self, Zt: torch.Tensor) -> torch.Tensor:
        """Reconstructs field fluctuations from a batch of POD coefficients.

        Args:
          Zt: POD coefficients, shape `(B, r)`.

        Returns:
          Fluctuations about the temporal mean, shape `(B, N_x)`.
        """
        return Zt @ self._Psi_t

    def field_target(self, Q: FloatArray) -> FloatArray:
        """Subtracts the temporal mean from physical fields.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Fluctuations, shape `(N_t, N_x)`.
        """
        return (np.asarray(Q, np.float64) - self.q_mean).T

    def to(self, device: str) -> LinearLatent:
        """Moves the basis tensor to a device.

        Args:
          device: Torch device name.

        Returns:
          This latent space.
        """
        self.device = device
        self._Psi_t = self._Psi_t.to(device)
        return self


class _Autoencoder(Protocol):
    """The members of a fitted `AE` or `CAE` that `TorchLatent` reads."""

    fitted: bool
    N_latent: int
    device: str

    @property
    def scale(self) -> FloatArray:
        """The per-field scale the projector divides inputs by, shape `(N_x, 1)`."""
        ...

    @property
    def Q_mean(self) -> FloatArray:
        """The temporal mean the projector subtracts, shape `(N_x, 1)`."""
        ...

    def encode(self, X: FloatArray) -> FloatArray:
        """Encodes physical fields into latent codes.

        Args:
          X: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Latent codes, shape `(N_latent, N_t)`.
        """
        ...

    def decode(self, Z: FloatArray) -> FloatArray:
        """Decodes latent codes into physical fields.

        Args:
          Z: Latent codes, shape `(N_latent, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        ...


class _DenseAutoencoder(_Autoencoder, Protocol):
    """The members of a fitted dense `AE` that `TorchLatent` reads."""

    @property
    def encoder(self) -> nn.Module:
        """The encoder network."""
        ...

    @property
    def decoder(self) -> nn.Module:
        """The decoder network."""
        ...


class _ConvAutoencoder(_Autoencoder, Protocol):
    """The members of a fitted `CAE` that `TorchLatent` reads."""

    grid_shape: tuple[int, int, int] | None
    fluid_mask_flat: npt.NDArray[np.bool_] | None

    @property
    def dec_fc(self) -> nn.Module:
        """The linear map from the latent space to the bottleneck grid."""
        ...

    @property
    def dec_conv(self) -> nn.Module:
        """The transposed-convolution decoder stages."""
        ...

    def networks(self) -> list[nn.Module]:
        """Lists every network the projector trains.

        Returns:
          The encoder and decoder modules.
        """
        ...

    def decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes onto the spatial grid.

        Args:
          Z: Latent codes, shape `(B, N_latent)`.

        Returns:
          Fields in scaled units, shape `(B, Nu, Nx, Ny)`.
        """
        ...


def _is_conv(projector: _Autoencoder) -> TypeGuard[_ConvAutoencoder]:
    """Reports whether a projector is convolutional.

    Args:
      projector: A fitted `AE` or `CAE`.

    Returns:
      `True` for a `CAE`.
    """
    return hasattr(projector, "dec_conv")


def _is_dense(projector: _Autoencoder) -> TypeGuard[_DenseAutoencoder]:
    """Reports whether a projector is a dense autoencoder.

    Args:
      projector: A fitted `AE` or `CAE`.

    Returns:
      `True` for an `AE`.
    """
    return hasattr(projector, "decoder")


class TorchLatent(LatentSpace):
    """Wraps a fitted `AE` or `CAE` as an encoder and decoder pair.

    Puts the projector's networks in evaluation mode. It doesn't train them;
    `BranchedAE` with `finetune_decoder=True` does. A `CAE` decoder emits a
    grid, so this class builds a gather index once to map the grid onto the flat
    `(Nu * N_fluid, N_t)` layout in torch.

    Args:
      projector: A fitted `AE` or `CAE`.
      device: Torch device. `None` uses the projector's device, or selects one
        with `default_device`.

    Raises:
      ValueError: If the projector is not fitted, or if a `CAE` has no grid
        geometry.
      TypeError: If the projector is neither an `AE` nor a `CAE`.
    """

    def __init__(self, projector: _Autoencoder, device: str | None = None):
        if not getattr(projector, "fitted", False):
            raise ValueError("wrap a *fitted* AE/CAE; call projector.fit(X) first")

        self.p = projector
        self.n_latent = int(projector.N_latent)
        self.device = default_device(device or getattr(projector, "device", None))
        self.p.device = self.device
        self._is_conv = _is_conv(projector)

        if _is_conv(projector):
            self._nets = projector.networks()
        elif _is_dense(projector):
            self._nets = [projector.encoder, projector.decoder]
        else:
            raise TypeError(f"projector must be an AE or a CAE, got {type(projector).__name__}")

        for net in self._nets:
            net.to(self.device).eval()

        self._scale = np.asarray(projector.scale, np.float64)
        self._gather = self._build_gather() if self._is_conv else torch.empty(0, dtype=torch.long)

    def _build_gather(self) -> torch.Tensor:
        """Builds the index that maps the decoder's grid onto the flat layout.

        The flat layout groups rows by component, so row `u * N_fluid + f`
        holds component `u` at the `f`-th fluid point. The decoder emits
        `(B, Nu, Nx, Ny)`, which flattens to entry `u * (Nx * Ny) + p`.

        Returns:
          Gather indices into the flattened grid, shape `(Nu * N_fluid,)`.

        Raises:
          ValueError: If the projector has no grid shape or fluid mask.
        """
        p = self.p
        if not _is_conv(p) or p.grid_shape is None or p.fluid_mask_flat is None:
            raise ValueError("a CAE needs grid_shape and fluid_mask_flat; fit it on gridded data")

        Nu, Nx, Ny = p.grid_shape
        pos = np.flatnonzero(p.fluid_mask_flat)
        idx = (np.arange(Nu)[:, None] * (Nx * Ny) + pos[None, :]).ravel()

        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    # ── NumPy side ────────────────────────────────────────────────────────────

    def encode(self, Q: FloatArray) -> FloatArray:
        """Encodes physical fields with the wrapped projector.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Latent codes, shape `(n_latent, N_t)`.
        """
        return self.p.encode(Q)

    def decode(self, Z: FloatArray) -> FloatArray:
        """Decodes latent codes with the wrapped projector.

        Args:
          Z: Latent codes, shape `(n_latent, N_t)`.

        Returns:
          Physical fields, shape `(N_x, N_t)`.
        """
        return self.p.decode(Z)

    def field_target(self, Q: FloatArray) -> FloatArray:
        """Removes the projector's mean and scale from physical fields.

        Args:
          Q: Physical fields, shape `(N_x, N_t)`.

        Returns:
          Fields in scaled units, shape `(N_t, N_x)`.
        """
        Qc = np.asarray(Q, np.float64) - self.p.Q_mean
        return (Qc / self._scale).T

    # ── Torch side ────────────────────────────────────────────────────────────

    def decode_torch(self, Zt: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latent codes differentiably.

        Args:
          Zt: Latent codes, shape `(B, n_latent)`.

        Returns:
          Fields in scaled units, shape `(B, N_x)`.

        Raises:
          TypeError: If the wrapped projector is neither an `AE` nor a `CAE`.
        """
        p = self.p
        if _is_conv(p):
            G = p.decode_grid(Zt)
            return G.reshape(G.shape[0], -1)[:, self._gather]
        if _is_dense(p):
            decoded: torch.Tensor = p.decoder(Zt)
            return decoded
        raise TypeError(f"projector must be an AE or a CAE, got {type(p).__name__}")

    def decoder_parameters(self) -> list[nn.Parameter]:
        """Lists the decoder parameters that end-to-end fine-tuning unfreezes.

        Returns:
          The parameters of the dense decoder, or of the CAE's fully connected
          and transposed-convolution decoder stages.
        """
        p = self.p
        nets: list[nn.Module] = [p.dec_fc, p.dec_conv] if _is_conv(p) else [p.decoder] if _is_dense(p) else []
        return [q for net in nets for q in net.parameters()]


# ── Sensor branch ─────────────────────────────────────────────────────────────


_ACT: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


class SensorBranch(nn.Module):
    """Maps a window of sensor history to a latent code.

    Takes `(B, n_delays, n_channels)`, oldest sample first, and returns
    `(B, n_latent)`. The output layer has no activation, so the latent codes are
    unbounded, like an AE bottleneck or POD coefficients.

    Args:
      n_channels: Number of sensor channels.
      n_delays: Window length in samples.
      n_latent: Dimension of the latent space.
      kind: Architecture. One of:

        - `"linear"`: A single affine map on the flattened window.
        - `"mlp"`: Flattens the window and applies the `hidden` layers.
        - `"cnn"`: One-dimensional convolutions along time, with channels as
          features.
        - `"gru"`: A recurrent network over the window that reads out the final
          hidden state.

      hidden: Hidden layer widths, for `kind="mlp"`.
      activation: Activation name. One of `"tanh"`, `"relu"`, `"elu"`, or
        `"gelu"`.
      dropout: Dropout probability between hidden layers, for `kind="mlp"`.
      cnn_channels: Channel widths, for `kind="cnn"`.
      kernel_size: Convolution kernel length, for `kind="cnn"`.
      gru_hidden: Hidden state width, for `kind="gru"`.
      gru_layers: Number of stacked recurrent layers, for `kind="gru"`.

    Raises:
      ValueError: If `kind` or `activation` is not one of the listed names.
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

        if activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}, got {activation!r}")
        act = _ACT[activation]
        n_flat = n_channels * n_delays

        if kind == "linear":
            self.head: nn.Module = nn.Linear(n_flat, n_latent)
        elif kind == "mlp":
            dims = [n_flat, *hidden]
            layers: list[nn.Module] = []

            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]), act()]

                if dropout:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(dims[-1], n_latent))
            self.head = nn.Sequential(*layers)

        elif kind == "cnn":
            conv: list[nn.Module] = []
            c = n_channels
            length = n_delays

            for c_out in cnn_channels:
                # Padding keeps the window length constant through every
                # convolution, so the flatten in the head sees `n_delays` steps.
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
        """Maps a batch of sensor windows to latent codes.

        Args:
          x: Sensor windows, shape `(B, n_delays, n_channels)`, oldest sample
            first.

        Returns:
          Latent codes, shape `(B, n_latent)`.
        """
        if self.kind in ("linear", "mlp"):
            flat: torch.Tensor = self.head(x.flatten(1))
            return flat
        if self.kind == "cnn":
            convolved: torch.Tensor = self.head(self.conv(x.transpose(1, 2)))
            return convolved
        out, _ = self.gru(x)
        # The last step of the window is time `t`.
        last: torch.Tensor = self.head(out[:, -1])
        return last

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in the branch."""
        return sum(q.numel() for q in self.parameters())


# ── Estimator ─────────────────────────────────────────────────────────────────


@dataclass
class BranchedAE:
    """Trains a sensor branch `G` against a frozen encoder and decoder pair.

    Exposes `fit`, `encode`, `predict`, and `score`, matching
    `field_estimation.epod.PODLSE`. The sensor windows are causal, so each method
    takes the whole record and an index array rather than a pre-sliced block:

      model.fit(Q, S, train_idx)
      model.score(Q, S, test_idx)

    Attributes:
      latent: A fitted `LinearLatent` or `TorchLatent`. Fit it on the training
        block only.
      branch: Sensor branch architecture. One of `"linear"`, `"mlp"`, `"cnn"`,
        or `"gru"`. See `SensorBranch`.
      n_delays: Causal window length in samples.
      delay_stride: Sample spacing between consecutive lags.
      hidden: Hidden widths for an MLP branch.
      activation: Activation name for the branch.
      dropout: Dropout probability inside an MLP branch.
      cnn_channels: Channel widths for a CNN branch.
      kernel_size: Convolution kernel length for a CNN branch.
      gru_hidden: Hidden state width for a GRU branch.
      gru_layers: Number of stacked recurrent layers for a GRU branch.
      lambda_field: Weight on the field-space loss term. 0 trains on the latent
        term alone.
      latent_weight: Scaling of the latent targets. `"energy"` leaves them
        unchanged, so each mode counts in proportion to its energy. `"unit"`
        divides each mode by its standard deviation.
      learning_rate: Adam step size.
      weight_decay: L2 penalty applied by Adam.
      sensor_noise: Standard deviation of the Gaussian noise added to each
        training minibatch of sensor windows, in units of the standardized
        channel. Validation and prediction use unchanged batches.
      n_epochs: Maximum number of training epochs.
      batch_size: Minibatch size.
      val_fraction: Fraction of the training block held out for early stopping,
        taken contiguously from the end of the block.
      patience: Epochs without improvement before training stops.
      threshold: Relative improvement that resets `patience`.
      lr_factor: Decay factor for `ReduceLROnPlateau`.
      lr_patience: Plateau epochs before the learning rate decays.
      min_lr: Lower bound on the learning rate.
      grad_clip: Maximum global gradient norm. 0 disables clipping.
      finetune_decoder: Train the decoder jointly with `G`. Applies only to a
        `TorchLatent`.
      seed: Random seed for torch and NumPy.
      device: Torch device. `None` selects one with `default_device`.
      verbose: Print training and validation losses every 25 epochs.
      G: The trained sensor branch, or `None` before `fit`.
      s_mean: Per-channel sensor mean, learned by `fit`.
      s_scale: Per-channel sensor scale, learned by `fit`.
      z_scale: Per-mode latent scale, learned by `fit`.
      track_sets: Evaluation sets to score during training, keyed by name. Each
        is scored in field NMSE every `track_every` epochs, and the results are
        appended to `track_history`.
      track_every: Epochs between tracking evaluations.
      track_max_cols: Maximum snapshots scored from each tracked set, sampled
        evenly.
      loss_history: Mean training loss per epoch.
      val_loss_history: Validation loss per epoch.
      track_history: `(epoch, nmse)` pairs per tracked set.
      fitted: Whether `fit` has completed.
    """

    latent: LatentSpace
    branch: str = "mlp"
    n_delays: int = 25
    delay_stride: int = 1

    # ── Branch architecture ──
    hidden: Sequence[int] = (128, 128)
    activation: str = "tanh"
    dropout: float = 0.0
    cnn_channels: Sequence[int] = (32, 64)
    kernel_size: int = 5
    gru_hidden: int = 64
    gru_layers: int = 1

    # ── Loss ──
    lambda_field: float = 0.0
    latent_weight: str = "energy"

    # ── Optimization ──
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
    device: str | None = None
    verbose: bool = False

    # ── Learned state ──
    G: SensorBranch | None = field(default=None, repr=False)
    s_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    s_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    z_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    track_sets: dict[str, TrackSet] | None = None
    track_every: int = 1
    track_max_cols: int = 400
    loss_history: list[float] = field(default_factory=list[float], repr=False)
    val_loss_history: list[float] = field(default_factory=list[float], repr=False)
    track_history: dict[str, list[tuple[int, float]]] = field(
        default_factory=dict[str, list[tuple[int, float]]], repr=False
    )
    fitted: bool = False
    _z_var: float = field(default=1.0, init=False, repr=False)
    _f_var: float = field(default=1.0, init=False, repr=False)

    # ── API ──

    @property
    def warmup(self) -> int:
        """Number of leading snapshots whose window is partly zero padding.

        Exclude these indices from `train_idx` and from every evaluation set.
        """
        return (self.n_delays - 1) * self.delay_stride

    def fit(self, Q: FloatArray, S: FloatArray, train_idx: IndexArray) -> BranchedAE:
        """Trains the sensor branch on the given snapshots.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full synchronized sensor record, shape `(N_s, N_t)`.
          train_idx: Snapshot indices to train on. Every index must be at or
            after `warmup`.

        Returns:
          This estimator, fitted.

        Raises:
          ValueError: If an option is out of range, if `Q` and `S` hold
            different snapshot counts, if `train_idx` is empty or reaches into
            the warm-up region, if the sensor record holds non-finite values, if
            `val_fraction` leaves no training snapshots, or if
            `finetune_decoder` is set for a latent space with no trainable
            decoder.
        """
        if self.latent_weight not in ("energy", "unit"):
            raise ValueError(f"latent_weight must be 'energy' or 'unit', got {self.latent_weight!r}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        dev = self.device = default_device(self.device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.latent.to(dev)

        train_idx = np.asarray(train_idx)
        self._check(Q, S, train_idx)

        # The sensor statistics come from the training columns only.
        S = np.asarray(S, np.float64)
        self.s_mean, self.s_scale = sensor_stats(S[:, train_idx])
        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale), self.n_delays, self.delay_stride)

        Z = self.latent.encode(Q[:, train_idx])
        zs = Z.std(axis=1, keepdims=True) if self.latent_weight == "unit" else np.ones((Z.shape[0], 1))
        self.z_scale = np.where(zs > 0, zs, 1.0)

        n_val = int(round(self.val_fraction * len(train_idx)))
        cut = len(train_idx) - n_val
        if cut < 1:
            raise ValueError(f"val_fraction={self.val_fraction} leaves no training snapshots out of {len(train_idx)}")
        idx_tr, idx_va = train_idx[:cut], train_idx[cut:]

        Xtr = torch.as_tensor(W[idx_tr], dtype=torch.float32, device=dev)
        Ztr = torch.as_tensor((Z[:, :cut] / self.z_scale).T, dtype=torch.float32, device=dev)
        Xva = torch.as_tensor(W[idx_va], dtype=torch.float32, device=dev) if n_val else None
        Zva = torch.as_tensor((Z[:, cut:] / self.z_scale).T, dtype=torch.float32, device=dev) if n_val else None

        self._z_var = float(Ztr.var().item()) or 1.0
        Ftr = Fva = None
        if self.lambda_field:
            F = self.latent.field_target(Q[:, train_idx])
            Ftr = torch.as_tensor(F[:cut], dtype=torch.float32, device=dev)
            Fva = torch.as_tensor(F[cut:], dtype=torch.float32, device=dev) if n_val else None
            self._f_var = float(Ftr.var().item()) or 1.0

        G = SensorBranch(
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
        self.G = G

        params = list(G.parameters())
        if self.finetune_decoder:
            dec = self.latent.decoder_parameters()
            if not dec:
                raise ValueError(
                    "finetune_decoder=True but the latent space has no trainable "
                    "decoder. A POD basis is fixed by construction -- use a "
                    "TorchLatent, or disable the option."
                )
            for q in dec:
                q.requires_grad_(True)
            params += dec

        opt = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_lr
        )

        best: float = float("inf")
        wait = 0
        best_state: dict[str, torch.Tensor] | None = None
        self.loss_history, self.val_loss_history = [], []
        self.track_history = {}
        n = Xtr.shape[0]

        for ep in range(self.n_epochs):
            G.train()
            order = torch.randperm(n, device=dev)
            run = 0.0

            for s in range(0, n, self.batch_size):
                sl = order[s : s + self.batch_size]
                xb = Xtr[sl]

                if self.sensor_noise > 0:
                    # Fresh noise every step, so the branch doesn't learn to
                    # cancel a fixed perturbation.
                    xb = xb + self.sensor_noise * torch.randn_like(xb)

                opt.zero_grad(set_to_none=True)
                loss = self._loss(G, xb, Ztr[sl], None if Ftr is None else Ftr[sl])
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

                opt.step()
                run += loss.item() * len(sl)

            self.loss_history.append(run / n)

            if self.track_sets and (ep % max(self.track_every, 1) == 0 or ep == self.n_epochs - 1):
                G.eval()
                # `predict` requires a fitted estimator. The flag is set for
                # good at the end of `fit`.
                self.fitted = True
                with torch.no_grad():
                    for name, (Qt, St, idx) in self.track_sets.items():
                        idx = np.asarray(idx)
                        if len(idx) > self.track_max_cols:
                            idx = idx[np.linspace(0, len(idx) - 1, self.track_max_cols).astype(int)]

                        e = nmse(np.asarray(Qt, np.float64)[:, idx], self.predict(St, idx))
                        self.track_history.setdefault(name, []).append((ep, e))

            if Xva is not None and Zva is not None:
                G.eval()
                with torch.no_grad():
                    v = float(self._loss(G, Xva, Zva, Fva).item())
                self.val_loss_history.append(v)
                sched.step(v)

                if v < best * (1.0 - self.threshold):
                    best, wait = v, 0
                    best_state = {k: t.detach().clone() for k, t in G.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

            if self.verbose and ep % 25 == 0:
                tail = f"  val {self.val_loss_history[-1]:.4e}" if n_val else ""
                print(f"    epoch {ep:4d}  train {self.loss_history[-1]:.4e}{tail}")

        if best_state is not None:
            G.load_state_dict(best_state)

        G.eval()
        self.fitted = True
        return self

    def _loss(self, G: SensorBranch, X: torch.Tensor, Z: torch.Tensor, F: torch.Tensor | None) -> torch.Tensor:
        """Computes the normalized training loss for one batch.

        Args:
          G: The sensor branch being trained.
          X: Sensor windows, shape `(B, n_delays, N_s)`.
          Z: Scaled latent targets, shape `(B, n_latent)`.
          F: Field targets in scaled units, shape `(B, N_x)`, or `None` to skip
            the field term.

        Returns:
          The scalar loss.
        """
        Zh: torch.Tensor = G(X)
        loss = ((Zh - Z) ** 2).mean() / self._z_var
        if F is not None:
            Zphys = Zh * torch.as_tensor(self.z_scale.T, dtype=torch.float32, device=Zh.device)
            loss = loss + self.lambda_field * ((self.latent.decode_torch(Zphys) - F) ** 2).mean() / self._f_var
        return loss

    def encode(self, S: FloatArray, idx: IndexArray | None = None) -> FloatArray:
        """Predicts latent codes from sensor data.

        Args:
          S: Full sensor record, shape `(N_s, N_t)`. Pass the whole record,
            because the window at each index reaches back into earlier
            snapshots.
          idx: Snapshots to evaluate. `None` evaluates every snapshot.

        Returns:
          Predicted latent codes, shape `(n_latent, len(idx))`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        G = self._branch()
        S = np.asarray(S, np.float64)
        W = sensor_windows(apply_sensor_stats(S, self.s_mean, self.s_scale), self.n_delays, self.delay_stride)
        if idx is not None:
            W = W[np.asarray(idx)]
        out: list[FloatArray] = []
        with torch.no_grad():
            for s in range(0, W.shape[0], 4096):
                x = torch.as_tensor(W[s : s + 4096], dtype=torch.float32, device=self.device)
                codes: torch.Tensor = G(x)
                out.append(codes.cpu().numpy())
        return np.concatenate(out, axis=0).T * self.z_scale

    def predict(self, S: FloatArray, idx: IndexArray | None = None) -> FloatArray:
        """Reconstructs fields from sensor data.

        Args:
          S: Full sensor record, shape `(N_s, N_t)`.
          idx: Snapshots to evaluate. `None` evaluates every snapshot.

        Returns:
          The reconstructed field, shape `(N_x, len(idx))`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        return self.latent.decode(self.encode(S, idx))

    def score(self, Q: FloatArray, S: FloatArray, idx: IndexArray | None = None) -> float:
        """Returns the normalized MSE of the reconstruction.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full sensor record, shape `(N_s, N_t)`.
          idx: Snapshots to score. `None` scores every snapshot.

        Returns:
          Normalized NMSE

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        idx = np.arange(Q.shape[1], dtype=np.intp) if idx is None else np.asarray(idx)
        return nmse(np.asarray(Q, np.float64)[:, idx], self.predict(S, idx))

    def latent_score(self, Q: FloatArray, S: FloatArray, idx: IndexArray) -> float:
        """Returns the normalized MSE in latent space.

        Scores the sensor branch against the encoder's codes, independent of the
        decoder.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full sensor record, shape `(N_s, N_t)`.
          idx: Snapshots to score.

        Returns:
          Normalized MSE between the true and predicted latent codes.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        idx = np.asarray(idx)
        Z = self.latent.encode(np.asarray(Q, np.float64)[:, idx])
        return nmse(Z, self.encode(S, idx))

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in the sensor branch, or 0 before `fit`."""
        return self.G.n_params if self.G is not None else 0

    # ── Internals ──

    def _check(self, Q: FloatArray, S: FloatArray, train_idx: IndexArray) -> None:
        """Checks that a record and its training indices can be fitted.

        Args:
          Q: Full field record, shape `(N_x, N_t)`.
          S: Full sensor record, shape `(N_s, N_t)`.
          train_idx: Snapshot indices to train on.

        Raises:
          ValueError: If `Q` and `S` hold different snapshot counts, if
            `train_idx` is empty or reaches into the warm-up region, or if the
            sensor record holds non-finite values at those indices.
        """
        if Q.shape[1] != S.shape[1]:
            raise ValueError(
                f"field has {Q.shape[1]} snapshots, sensors have {S.shape[1]}; "
                "pass the whole synchronised record to fit(), not a slice"
            )
        if train_idx.size == 0:
            raise ValueError("train_idx is empty")
        if train_idx.min() < self.warmup:
            raise ValueError(
                f"train_idx starts at {train_idx.min()} but the {self.n_delays}-step "
                f"window needs {self.warmup} samples of history. Pass "
                f"warmup={self.warmup} to datasets.split_indices."
            )
        if not np.isfinite(S[:, train_idx]).all():
            raise ValueError("non-finite sensor values; interpolate dropouts first")

    def _check_fitted(self) -> None:
        """Raises if the estimator has no trained sensor branch.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        self._branch()

    def _branch(self) -> SensorBranch:
        """Returns the trained sensor branch.

        Checks the branch itself rather than only the `fitted` flag, which is a
        constructor argument and can be set without training.

        Returns:
          The sensor branch.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if not self.fitted or self.G is None:
            raise RuntimeError("call fit() before encode()/predict()/score()")
        return self.G


# ── Latent forecaster ─────────────────────────────────────────────────────────


@dataclass
class LatentForecaster:
    """Rolls a latent state forward in time with a GRU.

    Computes `y_{t+1} = F(y_{t-L+1..t})`, the third operator in the full
    pipeline:

      s_{t-L..t} --G--> y_t --F--> y_{t+1..t+h} --D--> phi_{t+1..t+h}

    Trains on latent trajectories only, independent of the sensors and the
    decoder. Training unrolls `n_unroll` steps, feeding each prediction back as
    the next input.

    Attributes:
      n_latent: Dimension of the latent space.
      n_delays: Input window length in samples.
      hidden: GRU hidden state width.
      layers: Number of stacked GRU layers.
      n_unroll: Steps to unroll during training.
      learning_rate: Adam step size.
      n_epochs: Maximum number of training epochs.
      batch_size: Minibatch size.
      val_fraction: Fraction of windows held out for early stopping.
      patience: Epochs without improvement before training stops.
      grad_clip: Maximum global gradient norm.
      seed: Random seed for torch.
      device: Torch device. `None` selects one with `default_device`.
      verbose: Print the training loss every 25 epochs.
      net: The trained network, or `None` before `fit`.
      z_mean: Per-mode latent mean, learned by `fit`.
      z_scale: Per-mode latent scale, learned by `fit`.
      loss_history: Mean training loss per epoch.
      val_loss_history: Validation loss per epoch.
      fitted: Whether `fit` has completed.
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
    device: str | None = None
    verbose: bool = False

    net: _GRUStep | None = field(default=None, repr=False)
    z_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    z_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    loss_history: list[float] = field(default_factory=list[float], repr=False)
    val_loss_history: list[float] = field(default_factory=list[float], repr=False)
    fitted: bool = False

    def fit(self, Z: FloatArray, train_idx: IndexArray | None = None) -> LatentForecaster:
        """Trains the forecaster on a latent trajectory.

        Args:
          Z: Latent trajectory, shape `(n_latent, N_t)`.
          train_idx: Contiguous snapshot indices to train on. `None` uses the
            whole trajectory.

        Returns:
          This forecaster, fitted.

        Raises:
          ValueError: If an option is out of range, if `Z` is not shaped
            `(n_latent, N_t)`, if `train_idx` is not contiguous, or if the block
            yields fewer than ten training windows.
        """
        if self.n_delays < 1 or self.n_unroll < 1:
            raise ValueError(f"n_delays and n_unroll must be >= 1, got {self.n_delays}, {self.n_unroll}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        dev = self.device = default_device(self.device)
        torch.manual_seed(self.seed)

        Z = np.asarray(Z, np.float64)
        if Z.ndim != 2 or Z.shape[0] != self.n_latent:
            raise ValueError(f"Z must be (n_latent={self.n_latent}, N_t), got shape {Z.shape}")

        idx = np.arange(Z.shape[1], dtype=np.intp) if train_idx is None else np.asarray(train_idx)
        if np.any(np.diff(idx) != 1):
            raise ValueError("LatentForecaster needs a contiguous training block")

        Zt = Z[:, idx]
        self.z_mean = Zt.mean(axis=1, keepdims=True)

        sd = Zt.std(axis=1, keepdims=True)
        self.z_scale = np.where(sd > 0, sd, 1.0)

        Zn = ((Zt - self.z_mean) / self.z_scale).T

        L, H = self.n_delays, self.n_unroll
        starts = np.arange(0, len(Zn) - L - H + 1)
        if len(starts) < 10:
            raise ValueError(f"only {len(starts)} training windows; shorten n_delays/n_unroll")
        n_val = int(round(self.val_fraction * len(starts)))
        cut = len(starts) - n_val

        X = np.stack([Zn[s : s + L] for s in starts])
        Y = np.stack([Zn[s + L : s + L + H] for s in starts])
        Xt = torch.as_tensor(X[:cut], dtype=torch.float32, device=dev)
        Yt = torch.as_tensor(Y[:cut], dtype=torch.float32, device=dev)
        Xv = torch.as_tensor(X[cut:], dtype=torch.float32, device=dev) if n_val else None
        Yv = torch.as_tensor(Y[cut:], dtype=torch.float32, device=dev) if n_val else None

        self.net = _GRUStep(self.n_latent, self.hidden, self.layers).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

        best: float = float("inf")
        wait = 0
        best_state: dict[str, torch.Tensor] | None = None
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

            if Xv is not None and Yv is not None:
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

    def rollout(self, Z_window: FloatArray, horizon: int) -> FloatArray:
        """Forecasts a latent trajectory in closed loop.

        Each prediction feeds back as the next input.

        Args:
          Z_window: Latent history, shape `(n_latent, n_delays)`.
          horizon: Number of steps to forecast.

        Returns:
          The forecast, shape `(n_latent, horizon)`.

        Raises:
          RuntimeError: If `fit` has not been called.
          ValueError: If `horizon` is below 1, or if `Z_window` has a different
            latent dimension from the forecaster.
        """
        net = self._network()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        window = np.asarray(Z_window, np.float64)
        if window.ndim != 2 or window.shape[0] != self.n_latent:
            raise ValueError(f"Z_window must be (n_latent={self.n_latent}, L), got shape {window.shape}")

        Zn = ((window - self.z_mean) / self.z_scale).T
        x = torch.as_tensor(Zn[None], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            Y = net.unroll(x, horizon)[0].cpu().numpy()
        return Y.T * self.z_scale + self.z_mean

    def _network(self) -> _GRUStep:
        """Returns the trained network.

        Checks the network itself rather than only the `fitted` flag, which is a
        constructor argument and can be set without training.

        Returns:
          The network.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if not self.fitted or self.net is None:
            raise RuntimeError("call fit() first")
        return self.net


class _GRUStep(nn.Module):
    """A one-step latent map with a closed-loop unroll.

    Args:
      n_latent: Dimension of the latent space.
      hidden: GRU hidden state width.
      layers: Number of stacked GRU layers.
    """

    def __init__(self, n_latent: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(n_latent, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_latent)

    def unroll(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """Forecasts `horizon` steps, feeding each prediction back as input.

        Each step predicts the increment from the previous latent state.

        Args:
          x: Latent history, shape `(B, L, n_latent)`.
          horizon: Number of steps to forecast.

        Returns:
          The forecast, shape `(B, horizon, n_latent)`.
        """
        h_out, h = self.gru(x)
        z: torch.Tensor = x[:, -1] + self.out(h_out[:, -1])
        preds = [z]

        for _ in range(horizon - 1):
            h_out, h = self.gru(z.unsqueeze(1), h)
            z = z + self.out(h_out[:, -1])
            preds.append(z)
        return torch.stack(preds, dim=1)
